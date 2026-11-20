"""Streaming readers for prepared datasets and raw source folders."""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


SOURCE_EXTENSIONS = {
    ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".java",
    ".go", ".rs", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".php", ".rb",
    ".swift", ".kt", ".kts", ".scala", ".sh", ".bash", ".sql", ".html", ".css",
    ".md", ".yaml", ".yml", ".toml", ".xml", ".lua", ".pl", ".r",
    ".ipynb", ".ex", ".exs", ".hs", ".clj", ".vim", ".dockerfile",
    # COBOL
    ".cbl", ".cob", ".cpy", ".cobol",
    # CUDA
    ".cu", ".cuh",
    # HCL / Terraform
    ".tf", ".tfvars", ".hcl",
}
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "vendor", "__pycache__", ".venv", "venv", "build", "dist"}

SPDX_REGEX = re.compile(r"SPDX-License-Identifier:\s*([\w.-]+)", re.IGNORECASE)
COPYRIGHT_REGEX = re.compile(r"Copyright\s+(?:\(c\)\s+)?\d{4}", re.IGNORECASE)

LANGUAGE_EXTENSION_MAP = {
    "pyi": "python",
    "cbl": "cobol",
    "cob": "cobol",
    "cpy": "cobol",
    "cobol": "cobol",
    "cu": "cuda",
    "cuh": "cuda",
    "tf": "hcl",
    "tfvars": "hcl",
    "hcl": "hcl",
    "sql": "sql",
}


def normalize_language_name(ext_or_lang: str) -> str:
    cleaned = ext_or_lang.lower().lstrip(".")
    return LANGUAGE_EXTENSION_MAP.get(cleaned, cleaned)


def scan_file_header_license(text: str, fallback_license: str = "") -> tuple[str, str]:
    """Scan first 25 lines of text for SPDX-License-Identifier and copyright."""
    first_25_lines = "\n".join(text.splitlines()[:25])
    m = SPDX_REGEX.search(first_25_lines)
    if m:
        return m.group(1).strip(), "file_header_spdx"
    return fallback_license, "repository_level"


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    license_id: str = ""
    description: str = ""


@dataclass
class SourceRow:
    row: dict[str, Any]
    source_path: str
    source_row: int
    error: str = ""


def expand_sources(specs: list[SourceSpec]) -> list[SourceSpec]:
    """Validate source paths while keeping folders as one provenance source."""
    expanded: list[SourceSpec] = []
    for spec in specs:
        path = spec.path.expanduser().resolve()
        if path.is_dir():
            expanded.append(SourceSpec(path, spec.license_id, spec.description or f"Imported from {path}"))
        elif path.is_file():
            expanded.append(SourceSpec(path, spec.license_id, spec.description))
        else:
            raise FileNotFoundError(f"Input path does not exist: {spec.path}")
    if not expanded:
        raise ValueError("No supported files were found in the selected inputs.")
    return expanded


def source_format(path: Path) -> str:
    if path.is_dir():
        return "code_folder"
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix == ".parquet":
        return "parquet"
    if suffix in SOURCE_EXTENSIONS:
        return "code"
    raise ValueError(f"Unsupported input format: {path.name}")


def iter_source(spec: SourceSpec) -> Iterator[SourceRow]:
    """Yield records without materializing the full source in memory."""
    path = spec.path
    if path.is_dir():
        row_index = 0
        for current, dirs, files in os.walk(path):
            dirs[:] = sorted(name for name in dirs if name not in SKIP_DIRS and not name.startswith("."))
            for name in sorted(files):
                item_path = Path(current) / name
                if item_path.is_symlink() or item_path.suffix.lower() not in SOURCE_EXTENSIONS:
                    continue
                try:
                    text = item_path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    row_index += 1
                    yield SourceRow({"filename": item_path.relative_to(path).as_posix(), "language": item_path.suffix.lower().lstrip(".")}, str(item_path), row_index, f"Could not read source file ({type(exc).__name__}).")
                    continue
                row_index += 1
                language = normalize_language_name(item_path.suffix)
                lic_id, lic_conf = scan_file_header_license(text, spec.license_id)
                row_dict = {
                    "content": text,
                    "code": text,
                    "filename": item_path.relative_to(path).as_posix(),
                    "language": language,
                    "license": lic_id,
                    "license_confidence": lic_conf,
                }
                yield SourceRow(row_dict, str(item_path), row_index)
        return
    kind = source_format(path)
    if kind == "csv":
        with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"CSV contains duplicate column names: {path}")
            for index, row in enumerate(reader, 1):
                record = dict(row)
                if None in record:
                    record["__extra_columns__"] = record.pop(None)
                yield SourceRow(record, str(path), index)
        return
    if kind == "jsonl":
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for index, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        yield SourceRow({"content": item}, str(path), index)
                    else:
                        yield SourceRow(item, str(path), index)
                except json.JSONDecodeError as exc:
                    yield SourceRow({"content": line.rstrip("\r\n")}, str(path), index, f"Invalid JSON: {exc.msg}")
        return
    if kind == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet support requires PyArrow. Install with: pip install '.[parquet]'") from exc
        parquet = pq.ParquetFile(path)
        offset = 0
        for batch in parquet.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                offset += 1
                yield SourceRow(dict(row), str(path), offset)
        return
    if kind == "code":
        text = path.read_text(encoding="utf-8", errors="replace")
        language = normalize_language_name(path.suffix)
        lic_id, lic_conf = scan_file_header_license(text, spec.license_id)
        row_dict = {
            "content": text,
            "code": text,
            "filename": path.name,
            "language": language,
            "license": lic_id,
            "license_confidence": lic_conf,
        }
        yield SourceRow(row_dict, str(path), 1)


"""Disk-backed dataset scan, review state, and export operations."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sqlite3
import struct
import sys
import time
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable

from .sources import SourceSpec, expand_sources, iter_source, source_format
from .sandbox import MAX_SOURCE_CHARS, PYTHON_AST_CHECKER, SandboxInputLimit, SandboxOutputLimit, SandboxTimeout, SandboxUnavailable, run_sandboxed_validator

Progress = Callable[[int, int | None, str], None]
PROMPT_KEYS = {"instruction", "prompt", "question", "task", "user", "input", "query"}
ANSWER_KEYS = {"completion", "output", "response", "answer", "assistant", "code"}
TEXT_KEYS = PROMPT_KEYS | ANSWER_KEYS | {"content", "text", "source", "target", "resource"}
EMPTY = {"", "none", "null", "nan"}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _tokens(value: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", value.casefold(), flags=re.UNICODE)


def _shingles(value: str) -> set[str]:
    tokens = _tokens(value)
    if len(tokens) < 6:
        return set(tokens)
    return {" ".join(tokens[i : i + 5]) for i in range(len(tokens) - 4)}


def _mix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def _signature(shingles: set[str], permutations: int = 32) -> tuple[int, ...]:
    hashes = [int.from_bytes(hashlib.blake2b(item.encode("utf-8"), digest_size=8).digest(), "big") for item in shingles]
    return tuple(min(_mix64(item ^ (seed * 0x9E3779B97F4A7C15)) for item in hashes) for seed in range(permutations))


def _similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _content_fields(row: dict[str, Any], preferred: list[str] | None = None) -> list[str]:
    if preferred:
        chosen = [name for name in preferred if name in row]
        if chosen:
            return chosen
    recognized = [name for name in row if name.casefold() in TEXT_KEYS]
    return recognized or [name for name, value in row.items() if isinstance(value, str)]


def _content(row: dict[str, Any], preferred: list[str] | None = None) -> str:
    fields = _content_fields(row, preferred)
    return "\n".join(str(row.get(name, "")) for name in fields).strip()


def _exact_fingerprint(row: dict[str, Any], preferred: list[str] | None = None) -> str:
    """Fingerprint selected fields without collapsing source or prompt whitespace."""
    fields = _content_fields(row, preferred)
    canonical = {name: str(row.get(name, "")).replace("\r\n", "\n").replace("\r", "\n") for name in fields}
    return hashlib.sha256(_json(canonical).encode("utf-8")).hexdigest() if canonical else ""


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-") or "dataset"


def unique_output_path(directory: Path | str, filename: str) -> Path:
    """Choose a non-overwriting output path inside the requested directory."""
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    requested = Path(filename).name
    stem = Path(requested).stem
    suffix = Path(requested).suffix
    candidate = root / requested
    version = 1
    while candidate.exists():
        version += 1
        candidate = root / f"{stem}-v{version}{suffix}"
    return candidate


def portable_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Remove machine-specific paths before a report is included in an export."""
    result = dict(summary)
    result["sources"] = [{**source, "path": Path(source["path"]).name} for source in summary.get("sources", [])]
    result["source_paths_redacted"] = True
    return result


def _find_row_issues(row: dict[str, Any], parse_error: str, source_path: str, check_syntax: bool = True) -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
    if parse_error:
        issues.append(("malformed", parse_error))
    names = {key.casefold(): key for key in row}
    prompt = next((names[key] for key in PROMPT_KEYS if key in names), None)
    answer = next((names[key] for key in ANSWER_KEYS if key in names), None)
    if not any(str(value).strip().casefold() not in EMPTY for value in row.values()):
        issues.append(("empty_row", "Every field is empty."))
    else:
        if prompt and str(row.get(prompt, "")).strip().casefold() in EMPTY:
            issues.append(("incomplete", f"Required training input '{prompt}' is empty."))
        if answer and str(row.get(answer, "")).strip().casefold() in EMPTY:
            issues.append(("incomplete", f"Training output '{answer}' is empty."))
    language_key = next((names[key] for key in ("language", "lang") if key in names), None)
    lang = str(row.get(language_key, "") if language_key else "").casefold()
    code_fields = [key for key in row if key.casefold() in {"code", "content", "completion", "output", "answer"}]
    if not code_fields and "resource" in names:
        code_fields = [names["resource"]]
    should_parse_python = lang in {"python", "py", "pyi"} or Path(source_path).suffix.casefold() in {".py", ".pyi"}
    source_suffix = Path(source_path).suffix.casefold()
    js_languages = {"javascript", "js", "mjs", "cjs"}
    should_parse_jsx = lang == "jsx" or source_suffix == ".jsx"
    c_languages = {"c", "h", "cpp", "c++", "cc", "cxx", "hpp", "hh"}
    should_parse_js = lang in js_languages or source_suffix in {".js", ".jsx", ".mjs", ".cjs"}
    should_parse_c = lang in c_languages or source_suffix in {".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh"}
    should_parse_cobol = lang in {"cobol", "cob", "cbl", "cpy"} or source_suffix in {".cbl", ".cob", ".cpy", ".cobol"}
    should_parse_cuda = lang in {"cuda", "cu", "cuh"} or source_suffix in {".cu", ".cuh"}
    should_parse_hcl = lang in {"hcl", "tf", "tfvars", "terraform"} or source_suffix in {".tf", ".tfvars", ".hcl"}
    should_parse_sql = lang in {"sql"} or source_suffix in {".sql"}

    if check_syntax and (should_parse_python or should_parse_js or should_parse_jsx or should_parse_c or should_parse_cobol or should_parse_cuda or should_parse_hcl or should_parse_sql):
        for key in code_fields or list(row):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                if len(value) > MAX_SOURCE_CHARS:
                    issues.append(("syntax_unchecked", f"Syntax in '{key}' was not checked: source exceeds the {MAX_SOURCE_CHARS:,}-character limit."))
                    continue
                if should_parse_python:
                    try:
                        checked = run_sandboxed_validator(
                            sys.executable,
                            ["-B", "-I", "-S", "-c", PYTHON_AST_CHECKER, "{input}"],
                            value,
                            ".py",
                        )
                        if checked.returncode:
                            issues.append(("syntax_unchecked", f"Python syntax in '{key}' was not checked: sandboxed parser exited with code {checked.returncode}."))
                            continue
                        try:
                            parser_result = json.loads(checked.stdout)
                        except (json.JSONDecodeError, TypeError):
                            issues.append(("syntax_unchecked", f"Python syntax in '{key}' was not checked: sandboxed parser returned an invalid result."))
                            continue
                        if not isinstance(parser_result, dict) or parser_result.get("parser_error"):
                            reason = parser_result.get("parser_error", "unexpected parser response") if isinstance(parser_result, dict) else "unexpected parser response"
                            issues.append(("syntax_unchecked", f"Python syntax in '{key}' was not checked: {reason}"))
                            continue
                        syntax_error = parser_result.get("syntax_error")
                        if syntax_error:
                            issues.append(("syntax", f"Python syntax in '{key}': {str(syntax_error)[:500]}"))
                        for warning in parser_result.get("warnings", []):
                            issues.append(("syntax_warning", f"Python parser warning in '{key}': {str(warning)[:400]}"))
                    except (SandboxUnavailable, SandboxTimeout, SandboxOutputLimit, SandboxInputLimit) as exc:
                        try:
                            ast.parse(value)
                        except SyntaxError as syn_err:
                            issues.append(("syntax", f"Python syntax in '{key}': {syn_err}"))
                        except Exception:
                            issues.append(("syntax_unchecked", f"Python syntax in '{key}' was not checked: {exc}"))
                    except OSError as exc:
                        try:
                            ast.parse(value)
                        except SyntaxError as syn_err:
                            issues.append(("syntax", f"Python syntax in '{key}': {syn_err}"))
                        except Exception:
                            issues.append(("syntax_unchecked", f"Python syntax in '{key}' was not checked: {type(exc).__name__}."))
                elif should_parse_js:
                    node = shutil.which("node")
                    if not node:
                        issues.append(("syntax_unchecked", f"JavaScript syntax in '{key}' was not checked: Node.js is unavailable."))
                    else:
                        try:
                            module_mode = lang == "mjs" or source_suffix == ".mjs"
                            suffix = ".mjs" if module_mode else ".js"
                            checked = run_sandboxed_validator(node, ["--check", "{input}"], value, suffix)
                            if checked.returncode and not module_mode:
                                # A valid ES module can fail Node's default script-mode parse.
                                checked = run_sandboxed_validator(node, ["--input-type=module", "--check", "{input}"], value, ".mjs")
                            if checked.returncode:
                                detail = (checked.stderr or checked.stdout).strip().splitlines()
                                issues.append(("syntax", f"JavaScript syntax in '{key}': {(detail[-1] if detail else 'invalid syntax')[:500]}"))
                        except (SandboxUnavailable, SandboxTimeout, SandboxOutputLimit, SandboxInputLimit) as exc:
                            issues.append(("syntax_unchecked", f"JavaScript syntax in '{key}' was not checked: {exc}"))
                        except OSError as exc:
                            issues.append(("syntax_unchecked", f"JavaScript syntax in '{key}' was not checked: {type(exc).__name__}."))
                elif should_parse_jsx:
                    issues.append(("syntax_unchecked", f"JSX syntax in '{key}' was not checked: Node.js does not parse JSX."))
                elif should_parse_cobol:
                    cobc = shutil.which("cobc")
                    if not cobc:
                        issues.append(("syntax_unchecked", f"COBOL syntax in '{key}' was not checked: GnuCOBOL compiler (cobc) is unavailable."))
                    else:
                        try:
                            checked = run_sandboxed_validator(cobc, ["-fsyntax-only", "{input}"], value, ".cob")
                            if checked.returncode:
                                detail = (checked.stderr or checked.stdout).strip().splitlines()
                                issues.append(("syntax", f"COBOL syntax in '{key}': {(detail[-1] if detail else 'invalid syntax')[:500]}"))
                        except (SandboxUnavailable, SandboxTimeout, SandboxOutputLimit, SandboxInputLimit) as exc:
                            issues.append(("syntax_unchecked", f"COBOL syntax in '{key}' was not checked: {exc}"))
                        except OSError as exc:
                            issues.append(("syntax_unchecked", f"COBOL syntax in '{key}' was not checked: {type(exc).__name__}."))
                elif should_parse_cuda:
                    nvcc = shutil.which("nvcc")
                    if nvcc:
                        try:
                            checked = run_sandboxed_validator(nvcc, ["-fsyntax-only", "-x", "cu", "{input}"], value, ".cu")
                            if checked.returncode:
                                detail = (checked.stderr or checked.stdout).strip().splitlines()
                                issues.append(("syntax", f"CUDA syntax in '{key}': {(detail[-1] if detail else 'invalid syntax')[:500]}"))
                        except Exception as exc:
                            issues.append(("syntax_unchecked", f"CUDA syntax in '{key}' was not checked: {exc}"))
                    else:
                        compiler = shutil.which("gcc")
                        if compiler:
                            try:
                                checked = run_sandboxed_validator(compiler, ["-fsyntax-only", "-x", "c++", "{input}"], value, ".cu")
                                if checked.returncode:
                                    detail = (checked.stderr or checked.stdout).strip().splitlines()
                                    issues.append(("syntax", f"CUDA syntax in '{key}': {(detail[-1] if detail else 'invalid syntax')[:500]}"))
                            except Exception as exc:
                                issues.append(("syntax_unchecked", f"CUDA syntax in '{key}' was not checked: {exc}"))
                        else:
                            issues.append(("syntax_unchecked", f"CUDA syntax in '{key}' was not checked: nvcc compiler is unavailable."))
                elif should_parse_hcl:
                    terraform = shutil.which("terraform")
                    if not terraform:
                        issues.append(("syntax_unchecked", f"HCL syntax in '{key}' was not checked: terraform CLI is unavailable."))
                    else:
                        try:
                            checked = run_sandboxed_validator(terraform, ["fmt", "-check", "{input}"], value, ".tf")
                            if checked.returncode:
                                issues.append(("syntax", f"HCL syntax in '{key}': formatting/syntax validation failed."))
                        except Exception as exc:
                            issues.append(("syntax_unchecked", f"HCL syntax in '{key}' was not checked: {exc}"))
                elif should_parse_sql:
                    try:
                        sql_con = sqlite3.connect(":memory:")
                        q = value.strip().rstrip(";")
                        sql_con.execute(f"EXPLAIN {q}")
                    except sqlite3.OperationalError as sql_err:
                        duck_ok = False
                        try:
                            import duckdb
                            duckdb.connect().sql(f"EXPLAIN {value.strip().rstrip(';')}")
                            duck_ok = True
                        except Exception:
                            pass
                        if not duck_ok:
                            issues.append(("syntax", f"SQL syntax in '{key}': {str(sql_err)[:500]}"))
                    except Exception as exc:
                        issues.append(("syntax_unchecked", f"SQL syntax in '{key}' was not checked: {exc}"))
                elif should_parse_c:
                    compiler = shutil.which("gcc")
                    if not compiler:
                        issues.append(("syntax_unchecked", f"C/C++ syntax in '{key}' was not checked: GCC is unavailable."))
                    else:
                        language_arg = "c++" if lang in {"cpp", "c++", "cc", "cxx", "hpp", "hh"} or source_suffix in {".cpp", ".cc", ".cxx", ".hpp", ".hh"} else "c"
                        try:
                            suffix = ".cpp" if language_arg == "c++" else ".c"
                            checked = run_sandboxed_validator(compiler, ["-fsyntax-only", "-x", language_arg, "{input}"], value, suffix)
                            if checked.returncode:
                                detail = (checked.stderr or checked.stdout).strip().splitlines()
                                diagnostic = " ".join(detail)
                                if "No such file or directory" in diagnostic or "file not found" in diagnostic:
                                    issues.append(("syntax_unchecked", f"{language_arg.upper()} syntax in '{key}' needs unavailable headers."))
                                else:
                                    issues.append(("syntax", f"{language_arg.upper()} syntax in '{key}': {(detail[-1] if detail else 'invalid syntax')[:500]}"))
                        except (SandboxUnavailable, SandboxTimeout, SandboxOutputLimit, SandboxInputLimit) as exc:
                            issues.append(("syntax_unchecked", f"{language_arg.upper()} syntax in '{key}' was not checked: {exc}"))
                        except OSError as exc:
                            issues.append(("syntax_unchecked", f"{language_arg.upper()} syntax in '{key}' was not checked: {type(exc).__name__}."))
    return issues


def _normalize_frozen_schema(data: dict[str, Any]) -> dict[str, Any]:
    row = dict(data)
    inst = row.get("instruction") or row.get("prompt") or ""
    comp = row.get("completion") or row.get("solution") or row.get("code") or row.get("answer") or ""
    fmt = row.get("format") or "code_task"
    lang = row.get("language") or row.get("lang") or "python"

    gen_prov = row.get("generation_provenance")
    if not gen_prov:
        if fmt in ("code_task", "repo_dependency"):
            gen_prov = "reconstructed_task"
        elif fmt == "bug_patch":
            gen_prov = "synthetic_bug" if row.get("bug_origin") == "introduced" else "model_inferred_defect"
        elif fmt == "commit_repair":
            gen_prov = "synthetic_commit"
        elif fmt == "compiler_repair":
            gen_prov = "synthetic_syntax_mutation"
        elif fmt == "doc_to_impl":
            gen_prov = "derived_documentation"
        elif fmt == "function_tests":
            gen_prov = "generated_tests"
        else:
            gen_prov = "reconstructed_task"

    syn_origin = row.get("synthetic_origin")
    if not syn_origin:
        if gen_prov in ("reconstructed_task", "derived_documentation"):
            syn_origin = "source_derived"
        elif gen_prov in ("synthetic_commit", "generated_tests"):
            syn_origin = "model_reconstructed"
        elif gen_prov in ("synthetic_bug", "synthetic_syntax_mutation"):
            syn_origin = "artificially_introduced"
        else:
            syn_origin = "model_inferred"

    art_prov = row.get("artifact_provenance")
    if not art_prov or not isinstance(art_prov, dict):
        art_prov = {
            "instruction": "model_reconstructed",
            "completion": "repository_derived" if fmt not in ("compiler_repair", "function_tests") else "model_generated",
            "source_code": "repository_derived",
            "tests": "model_generated" if fmt == "function_tests" else "n/a",
            "patch": "model_generated" if fmt in ("bug_patch", "commit_repair") else "n/a",
        }

    ver_json = row.get("verification_json")
    if not ver_json:
        ver_json = json.dumps({
            "syntax_valid": True,
            "method": "ast.parse",
            "executed": False,
            "solution_correctness": {"level": "static_ast", "passed": True},
            "instruction_sufficiency": {"level": "unreviewed", "coverage": None, "missing_behaviors": []}
        }, ensure_ascii=False, sort_keys=True)

    row["instruction"] = str(inst)
    row["completion"] = str(comp)
    row["format"] = fmt
    row["language"] = lang
    row["generation_provenance"] = gen_prov
    row["synthetic_origin"] = syn_origin
    row["artifact_provenance"] = art_prov
    row["synthetic"] = bool(row.get("synthetic", False))
    row["license_confidence"] = row.get("license_confidence") or "repository_level"
    row["source_content_sha256"] = str(row.get("source_content_sha256") or hashlib.sha256(str(comp).encode("utf-8")).hexdigest()[:16])
    row["verification_json"] = ver_json if isinstance(ver_json, str) else json.dumps(ver_json, ensure_ascii=False, sort_keys=True)
    return row


class RunSession:
    """SQLite-backed analysis and review session. Source files are always read-only."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sources (
              id INTEGER PRIMARY KEY, path TEXT NOT NULL, license_id TEXT NOT NULL,
              description TEXT NOT NULL, format TEXT NOT NULL, row_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS records (
              id INTEGER PRIMARY KEY, source_id INTEGER NOT NULL, record_path TEXT NOT NULL, source_row INTEGER NOT NULL,
              data TEXT NOT NULL, content TEXT NOT NULL, exact_key TEXT NOT NULL, signature BLOB,
              include_row INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS findings (
              id INTEGER PRIMARY KEY, row_id INTEGER, code TEXT NOT NULL, detail TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS buckets (
              band INTEGER NOT NULL, value TEXT NOT NULL, row_id INTEGER NOT NULL,
              PRIMARY KEY (band, value, row_id)
            );
            CREATE INDEX IF NOT EXISTS findings_row_idx ON findings(row_id);
            CREATE INDEX IF NOT EXISTS records_exact_idx ON records(exact_key);
            CREATE INDEX IF NOT EXISTS buckets_lookup_idx ON buckets(band, value);
            """
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "RunSession":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def analyze(
        self,
        sources: list[SourceSpec],
        field_map: dict[str, str] | None = None,
        preferred_fields: list[str] | None = None,
        threshold: float = 0.9,
        detect_near: bool = True,
        progress: Progress | None = None,
        check_syntax: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(check_syntax, bool):
            raise ValueError("check_syntax must be a boolean.")
        self.db.executescript("DELETE FROM meta; DELETE FROM sources; DELETE FROM records; DELETE FROM findings; DELETE FROM buckets;")
        self.db.commit()
        expanded = expand_sources(sources)
        field_map = field_map or {}
        source_meta: list[dict[str, Any]] = []
        field_types: dict[str, set[str]] = {}
        total = 0
        issue_counts: Counter[str] = Counter()
        start = time.perf_counter()
        cursor = self.db.cursor()
        for source_id, spec in enumerate(expanded, 1):
            fmt = source_format(Path(spec.path))
            cursor.execute(
                "INSERT INTO sources(id,path,license_id,description,format) VALUES(?,?,?,?,?)",
                (source_id, str(spec.path), spec.license_id.strip(), spec.description.strip(), fmt),
            )
            source_count = 0
            for item in iter_source(spec):
                row = dict(item.row)
                mapped: dict[str, Any] = {}
                for key, value in row.items():
                    source_name = Path(item.source_path).name if Path(spec.path).is_dir() else Path(spec.path).name
                    target = field_map.get(f"{source_name}.{key}", field_map.get(key, key))
                    if target in mapped and mapped[target] != value:
                        raise ValueError(f"Field mapping collision in {item.source_path} row {item.source_row}: multiple values map to '{target}'.")
                    mapped[target] = value
                    kind = "null" if value is None else "bool" if isinstance(value, bool) else "number" if isinstance(value, (int, float)) else "string" if isinstance(value, str) else "object"
                    field_types.setdefault(target, set()).add(kind)
                content = _content(mapped, preferred_fields)
                exact_key = _exact_fingerprint(mapped, preferred_fields)
                cursor.execute(
                    "INSERT INTO records(source_id,record_path,source_row,data,content,exact_key) VALUES(?,?,?,?,?,?)",
                    (source_id, item.source_path, item.source_row, _json(mapped), content, exact_key),
                )
                row_id = cursor.lastrowid
                issues = _find_row_issues(mapped, item.error, item.source_path, check_syntax)
                if exact_key:
                    first_exact = cursor.execute("SELECT id FROM records WHERE exact_key=? AND id<? LIMIT 1", (exact_key, row_id)).fetchone()
                    if first_exact:
                        issues.append(("duplicate_exact", f"Exact duplicate of record {first_exact[0]} (line endings normalized; content whitespace preserved)."))
                shards = _shingles(content) if detect_near else set()
                if detect_near and shards and len(shards) > 1:
                    sig = _signature(shards)
                    candidates: set[int] = set()
                    cursor.execute("UPDATE records SET signature=? WHERE id=?", (struct.pack("<32Q", *sig), row_id))
                    band_size = 4
                    for band in range(8):
                        band_value = _json(sig[band * band_size : (band + 1) * band_size])
                        hits = cursor.execute("SELECT row_id FROM buckets WHERE band=? AND value=? LIMIT 100", (band, band_value)).fetchall()
                        if len(candidates) < 500:
                            candidates.update(int(hit[0]) for hit in hits)
                        cursor.execute("INSERT OR IGNORE INTO buckets(band,value,row_id) VALUES(?,?,?)", (band, band_value, row_id))
                    if not any(code == "duplicate_exact" for code, _ in issues):
                        for candidate in sorted(candidates)[:500]:
                            old = cursor.execute("SELECT content,signature FROM records WHERE id=?", (candidate,)).fetchone()
                            if old and old[1]:
                                old_sig = struct.unpack("<32Q", old[1])
                                estimate = sum(left == right for left, right in zip(sig, old_sig)) / 32
                                if estimate + 0.2 < threshold:
                                    continue
                            if old and _similarity(shards, _shingles(old[0])) >= threshold:
                                issues.append(("duplicate_near", f"Near duplicate of record {candidate} (Jaccard ≥ {threshold:.2f})."))
                                break
                for code, detail in issues:
                    cursor.execute("INSERT INTO findings(row_id,code,detail) VALUES(?,?,?)", (row_id, code, detail))
                    issue_counts[code] += 1
                source_count += 1
                total += 1
                if total % 500 == 0:
                    self.db.commit()
                    if progress:
                        progress(total, None, f"Scanned {total:,} records…")
            cursor.execute("UPDATE sources SET row_count=? WHERE id=?", (source_count, source_id))
            source_meta.append({"id": source_id, "path": str(spec.path), "license_id": spec.license_id.strip(), "description": spec.description.strip(), "format": fmt, "row_count": source_count})
            self.db.commit()
        licenses = {source["license_id"] for source in source_meta if source["license_id"]}
        license_warnings = []
        if any(not source["license_id"] for source in source_meta):
            license_warnings.append("One or more input sources have no license recorded. Confirm redistribution rights before selling or sharing the output.")
        if len(licenses) > 1:
            license_warnings.append("Inputs declare different license identifiers. Review their combined redistribution terms before selling or sharing the output.")
        fields = sorted(field_types)
        # Store compatibility decisions for stable Parquet batches.
        arrow_types = {}
        for name, kinds in field_types.items():
            stable = kinds - {"null"}
            arrow_types[name] = next(iter(stable)) if len(stable) == 1 else "mixed"
        summary = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "records": total,
            "sources": source_meta,
            "fields": fields,
            "field_types": arrow_types,
            "issues": dict(issue_counts),
            "license_warnings": license_warnings,
            "near_duplicate_threshold": threshold,
            "near_duplicates_enabled": detect_near,
            "syntax_checks_enabled": check_syntax,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "field_map": field_map,
            "preferred_fields": preferred_fields or [],
            "local_processing": True,
        }
        self.set_meta("summary", summary)
        if progress:
            progress(total, total, "Analysis complete")
        return summary

    def set_meta(self, key: str, value: Any) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, _json(value)))
        self.db.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def summary(self) -> dict[str, Any]:
        return self.get_meta("summary", {})

    def set_include(self, row_id: int, include: bool) -> None:
        self.db.execute("UPDATE records SET include_row=? WHERE id=?", (int(include), row_id))
        self.db.commit()

    def records(self, limit: int = 500, offset: int = 0) -> list[dict[str, Any]]:
        rows = self.db.execute(
            """SELECT r.id,r.source_id,r.source_row,r.data,r.include_row,r.record_path AS path,s.path AS input_path,
                      GROUP_CONCAT(f.code, ', ') AS issue_codes,
                      GROUP_CONCAT(f.detail, ' | ') AS issue_details
               FROM records r JOIN sources s ON s.id=r.source_id LEFT JOIN findings f ON f.row_id=r.id
               GROUP BY r.id ORDER BY r.id LIMIT ? OFFSET ?""", (limit, offset)
        ).fetchall()
        return [{**dict(row), "data": json.loads(row["data"])} for row in rows]

    def iter_findings(self) -> Iterable[dict[str, Any]]:
        cursor = self.db.execute(
            """SELECT f.id,f.row_id,f.code,f.detail,r.record_path AS path,s.path AS input_path,r.source_row,r.data
               FROM findings f LEFT JOIN records r ON r.id=f.row_id LEFT JOIN sources s ON s.id=r.source_id
               ORDER BY f.id"""
        )
        while batch := cursor.fetchmany(1000):
            for row in batch:
                yield {**dict(row), "data": json.loads(row["data"]) if row["data"] else None}

    def findings(self, limit: int = 1000) -> list[dict[str, Any]]:
        return list(islice(self.iter_findings(), limit))

    def detect_semantic_duplicates(
        self,
        threshold_review: float = 0.85,
        threshold_duplicate: float = 0.92,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        progress: ProgressCallback = None,
    ) -> dict[str, Any]:
        """Level 4 Multi-Channel Semantic Deduplication with Algorithmic Preservation Rule."""
        rows = self.db.execute("SELECT id, data FROM records ORDER BY id").fetchall()
        if not rows:
            return {"records_analyzed": 0, "duplicates_found": 0, "review_flags": 0}

        records_data = []
        for row in rows:
            data = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
            inst = str(data.get("instruction") or data.get("prompt") or "").strip()
            comp = str(data.get("completion") or data.get("solution") or data.get("code") or data.get("answer") or "").strip()
            records_data.append((row["id"], inst, comp))

        use_st = False
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(embedding_model_name)
            use_st = True
        except Exception:
            use_st = False

        if use_st:
            inst_texts = [r[1] for r in records_data]
            comp_texts = [r[2] for r in records_data]
            inst_vecs = model.encode(inst_texts, normalize_embeddings=True)
            comp_vecs = model.encode(comp_texts, normalize_embeddings=True)

            def _cos(v1, v2):
                import numpy as np
                return float(np.dot(v1, v2))
        else:
            def _text_vec(text: str) -> dict[str, float]:
                toks = _tokens(text)
                if not toks:
                    return {}
                from collections import Counter
                counts = Counter(toks)
                norm = sum(c * c for c in counts.values()) ** 0.5
                return {k: v / norm for k, v in counts.items()}

            inst_vecs = [_text_vec(r[1]) for r in records_data]
            comp_vecs = [_text_vec(r[2]) for r in records_data]

            def _cos(v1, v2):
                if not v1 or not v2:
                    return 0.0
                common = set(v1) & set(v2)
                return sum(v1[k] * v2[k] for k in common)

        dups = 0
        reviews = 0
        cursor = self.db.cursor()
        n = len(records_data)
        for i in range(n):
            id_i, _, _ = records_data[i]
            for j in range(i + 1, n):
                id_j, _, _ = records_data[j]
                sim_inst = _cos(inst_vecs[i], inst_vecs[j])
                sim_comp = _cos(comp_vecs[i], comp_vecs[j])

                # The Algorithmic Preservation Rule:
                # If prompt/instruction is identical/near-identical, but implementation/completion
                # is distinct (sim_comp < threshold_review), do NOT flag as duplicate!
                # Retains distinct algorithms (e.g. recursive vs iterative).
                if sim_inst >= threshold_duplicate and sim_comp < threshold_review:
                    continue

                if sim_inst >= threshold_duplicate and sim_comp >= threshold_duplicate:
                    cursor.execute(
                        "INSERT INTO findings(row_id, code, detail) VALUES(?,?,?)",
                        (id_j, "duplicate_semantic_candidate", f"Semantic duplicate candidate of row {id_i} (sim_inst={sim_inst:.2f}, sim_comp={sim_comp:.2f})")
                    )
                    dups += 1
                elif sim_inst >= threshold_review or sim_comp >= threshold_review:
                    cursor.execute(
                        "INSERT INTO findings(row_id, code, detail) VALUES(?,?,?)",
                        (id_j, "semantic_review_flag", f"Shares high semantic similarity with row {id_i} (sim_inst={sim_inst:.2f}, sim_comp={sim_comp:.2f})")
                    )
                    reviews += 1
        self.db.commit()
        result = {
            "records_analyzed": n,
            "duplicates_found": dups,
            "review_flags": reviews,
            "embedding_model": embedding_model_name if use_st else "token_tfidf_cosine",
            "thresholds": {"review": threshold_review, "duplicate": threshold_duplicate},
            "metric": "cosine",
        }
        self.set_meta("semantic_dedup", result)
        return result

    def export(
        self,
        output_dir: Path | str,
        dataset_name: str,
        description: str,
        output_format: str = "jsonl",
        exclude_exact: bool = True,
        exclude_near: bool = False,
        exclude_incomplete: bool = False,
        exclude_malformed: bool = True,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = output_dir / "raw"
        curated_dir = output_dir / "curated"
        approved_dir = output_dir / "approved"
        raw_dir.mkdir(parents=True, exist_ok=True)
        curated_dir.mkdir(parents=True, exist_ok=True)
        approved_dir.mkdir(parents=True, exist_ok=True)

        try:
            target_sqlite = (curated_dir / "review.sqlite").resolve()
            if self.db_path.resolve() != target_sqlite:
                shutil.copy2(self.db_path, target_sqlite)
        except Exception:
            pass

        output_format = output_format.lower()
        if output_format not in {"jsonl", "csv", "parquet"}:
            raise ValueError("Output format must be jsonl, csv, or parquet.")
        summary = self.summary()
        stem = safe_name(dataset_name)
        suffix = "jsonl" if output_format == "jsonl" else output_format
        final_names = [f"{stem}.{suffix}", f"{stem}.quality-report.json", f"{stem}.provenance.json", f"{stem}.findings.jsonl"]
        version = 1
        while any((output_dir / name).exists() for name in final_names):
            version += 1
            final_names = [f"{stem}-v{version}.{suffix}", f"{stem}-v{version}.quality-report.json", f"{stem}-v{version}.provenance.json", f"{stem}-v{version}.findings.jsonl"]
        stem = Path(final_names[0]).stem
        data_path = output_dir / final_names[0]
        source_paths = {Path(source["path"]).resolve() for source in summary.get("sources", [])}
        if any((output_dir / name).resolve() in source_paths for name in final_names):
            raise ValueError("The output package would overwrite an input source. Choose a different output folder or dataset name.")

        # Copy raw input files to raw_dir if accessible
        for source in summary.get("sources", []):
            try:
                sp = Path(source["path"])
                if sp.is_file():
                    target_raw = raw_dir / sp.name
                    if not target_raw.exists() and sp.resolve() != target_raw.resolve():
                        shutil.copy2(sp, target_raw)
            except Exception:
                pass

        fields = summary.get("fields", [])
        where = ["r.include_row=1"]
        excluded_codes: list[str] = []
        if exclude_exact:
            excluded_codes.append("duplicate_exact")
        if exclude_near:
            excluded_codes.append("duplicate_near")
        if exclude_incomplete:
            excluded_codes.extend(["incomplete", "empty_row"])
        if exclude_malformed:
            excluded_codes.append("malformed")
        if excluded_codes:
            marks = ",".join("?" for _ in excluded_codes)
            where.append(f"NOT EXISTS (SELECT 1 FROM findings f WHERE f.row_id=r.id AND f.code IN ({marks}))")
        sql = "SELECT r.data FROM records r WHERE " + " AND ".join(where) + " ORDER BY r.id"
        rows = self.db.execute(sql, excluded_codes)
        written = 0
        if output_format == "jsonl":
            with data_path.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    raw_item = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                    normalized = _normalize_frozen_schema(raw_item)
                    handle.write(_json(normalized) + "\n")
                    written += 1
        elif output_format == "csv":
            with data_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for item in rows:
                    raw_item = json.loads(item[0]) if isinstance(item[0], str) else item[0]
                    data = _normalize_frozen_schema(raw_item)
                    writer.writerow({key: self._csv_value(data.get(key)) for key in fields})
                    written += 1
        else:
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("Parquet export requires PyArrow. Install with: pip install '.[parquet]'") from exc
            type_map = summary.get("field_types", {})
            arrow_schema = pa.schema([pa.field(field, self._arrow_type(pa, type_map.get(field, "mixed")), nullable=True) for field in fields])
            writer = pq.ParquetWriter(data_path, arrow_schema, compression="zstd")
            batch: list[dict[str, Any]] = []
            try:
                for item in rows:
                    raw_item = json.loads(item[0]) if isinstance(item[0], str) else item[0]
                    data = _normalize_frozen_schema(raw_item)
                    batch.append({key: self._arrow_value(data.get(key), type_map.get(key, "mixed")) for key in fields})
                    if len(batch) >= 2048:
                        writer.write_table(pa.Table.from_pylist(batch, schema=arrow_schema))
                        written += len(batch)
                        batch.clear()
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=arrow_schema))
                    written += len(batch)
            finally:
                writer.close()
        report = portable_summary(summary)
        report.update({"exported_records": written, "output_file": data_path.name, "output_format": output_format,
                       "dataset_name": dataset_name, "dataset_description": description,
                       "cleanup_rules": {"exclude_exact_duplicates": exclude_exact, "exclude_near_duplicates": exclude_near,
                                         "exclude_incomplete": exclude_incomplete, "exclude_malformed": exclude_malformed},
                       "quality_counts": summary.get("issues", {})})
        report_path = output_dir / f"{stem}.quality-report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        source_path = output_dir / f"{stem}.provenance.json"
        provenance = {"dataset_name": dataset_name, "description": description, "sources": portable_summary(summary).get("sources", []),
                      "field_map": summary.get("field_map", {}), "local_processing": True,
                      "license_warnings": summary.get("license_warnings", []), "source_paths_redacted": True}
        source_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
        findings_path = output_dir / f"{stem}.findings.jsonl"
        with findings_path.open("w", encoding="utf-8") as handle:
            for finding in self.iter_findings():
                finding.pop("data", None)
                finding.pop("path", None)
                finding.pop("input_path", None)
                handle.write(_json(finding) + "\n")

        # Copy approved artifacts to approved_dir
        for p in [data_path, report_path, source_path, findings_path]:
            try:
                shutil.copy2(p, approved_dir / p.name)
            except Exception:
                pass

        return {"data_file": str(data_path), "report_file": str(report_path), "provenance_file": str(source_path),
                "findings_file": str(findings_path), "exported_records": written,
                "raw_dir": str(raw_dir), "curated_dir": str(curated_dir), "approved_dir": str(approved_dir)}

    @staticmethod
    def _csv_value(value: Any) -> str:
        if value is None:
            return ""
        return value if isinstance(value, str) else _json(value)

    @staticmethod
    def _arrow_type(pa: Any, kind: str) -> Any:
        return {"bool": pa.bool_(), "number": pa.float64(), "string": pa.string()} .get(kind, pa.string())

    @staticmethod
    def _arrow_value(value: Any, kind: str) -> Any:
        if value is None:
            return None
        if kind == "bool" and isinstance(value, bool):
            return value
        if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if kind == "string" and isinstance(value, str):
            return value
        return _json(value)

    def propose_rewrite(self, row_id: int, model: str, prompt: str = "Improve clarity and correctness while preserving meaning. Return only the revised training example.") -> str:
        """Ask a local Ollama endpoint to suggest a revision; never send data off-device."""
        import urllib.error
        import urllib.request

        row = self.db.execute("SELECT data FROM records WHERE id=?", (row_id,)).fetchone()
        if not row:
            raise KeyError(f"Unknown record: {row_id}")
        request_body = _json({"model": model, "prompt": f"{prompt}\n\nTraining example:\n{row[0]}", "stream": False}).encode("utf-8")
        request = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=request_body, headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=180) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("Could not reach Ollama at 127.0.0.1:11434. Install and start a local model; no remote endpoint is used.") from exc
        return str(result.get("response", "")).strip()


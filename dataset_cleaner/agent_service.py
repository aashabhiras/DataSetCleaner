"""Shared local operations for MCP hosts and the loopback HTTP API."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .engine import RunSession
from .sources import SourceSpec

RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
MAX_PAGE = 100
MAX_CANDIDATES = 100


def default_agent_home() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "DataSetCleaner" / "agent-runs"


class AgentService:
    """Persistent, local-only review sessions shared by separate agent hosts."""

    def __init__(self, home: Path | str | None = None):
        self.home = Path(home or os.environ.get("DATASETCLEANER_AGENT_HOME") or default_agent_home()).expanduser().resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="datasetcleaner-agent")
        self._metadata_lock = threading.Lock()

    def _write_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        target = self.home / f"{run_id}.run.json"
        temporary = self.home / f"{run_id}.run.tmp"
        with self._metadata_lock:
            temporary.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
            temporary.replace(target)

    def _db_path(self, run_id: str) -> Path:
        if not RUN_ID_RE.fullmatch(run_id):
            raise ValueError("Invalid run_id.")
        metadata_path = self.home / f"{run_id}.run.json"
        if not metadata_path.exists():
            raise KeyError(f"Unknown review run: {run_id}")
        metadata = self._read_metadata(run_id)
        if metadata.get("status") != "completed":
            raise RuntimeError(f"Review run is {metadata.get('status', 'unavailable')}; wait for a completed analysis before reviewing or exporting.")
        path = (self.home / f"{run_id}.sqlite").resolve()
        if path.parent != self.home:
            raise ValueError("Invalid run_id.")
        if not path.exists():
            raise KeyError(f"Unknown review run: {run_id}")
        return path

    def start_analysis(
        self,
        sources: list[dict[str, str]],
        field_map: dict[str, str] | None = None,
        preferred_fields: list[str] | None = None,
        threshold: float = 0.9,
        detect_near: bool = False,
        check_syntax: bool = True,
    ) -> dict[str, Any]:
        if not isinstance(sources, list) or not sources or len(sources) > 100:
            raise ValueError("Provide between 1 and 100 local sources.")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0.5 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.50 and 1.00.")
        if not isinstance(detect_near, bool):
            raise ValueError("detect_near must be a boolean.")
        if not isinstance(check_syntax, bool):
            raise ValueError("check_syntax must be a boolean.")
        specs: list[SourceSpec] = []
        for item in sources:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"].strip():
                raise ValueError("Each source needs a local path.")
            path = Path(item["path"]).expanduser().resolve()
            if not path.exists():
                raise ValueError(f"Source does not exist: {path.name}")
            license_id = item.get("license_id", "")
            description = item.get("description", "")
            if not isinstance(license_id, str) or not isinstance(description, str):
                raise ValueError("Source license_id and description must be strings.")
            specs.append(SourceSpec(path, license_id, description))
        if field_map is not None and (not isinstance(field_map, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in field_map.items())):
            raise ValueError("field_map must contain string keys and values.")
        if preferred_fields is not None and (not isinstance(preferred_fields, list) or len(preferred_fields) > 200 or any(not isinstance(field, str) for field in preferred_fields)):
            raise ValueError("preferred_fields must be a list of up to 200 strings.")

        run_id = uuid.uuid4().hex
        metadata = {"run_id": run_id, "created_at": time.time(), "status": "running", "scanned_records": 0, "syntax_checks_enabled": check_syntax}
        self._write_metadata(run_id, metadata)
        self._executor.submit(self._analyze_run, run_id, specs, field_map or {}, preferred_fields, threshold, detect_near, check_syntax)
        return {"run_id": run_id, "status": "running", "review_required": True, "local_processing": True, "syntax_checks_enabled": check_syntax}

    def _analyze_run(
        self,
        run_id: str,
        specs: list[SourceSpec],
        field_map: dict[str, str],
        preferred_fields: list[str] | None,
        threshold: float,
        detect_near: bool,
        check_syntax: bool,
    ) -> None:
        db_path = self.home / f"{run_id}.sqlite"
        try:
            with RunSession(db_path) as session:
                summary = session.analyze(
                    specs,
                    field_map=field_map,
                    preferred_fields=preferred_fields,
                    threshold=threshold,
                    detect_near=detect_near,
                    check_syntax=check_syntax,
                    progress=lambda count, _total, _message: self._write_metadata(
                        run_id, {**self._read_metadata(run_id), "scanned_records": count}
                    ) if count % 50000 == 0 else None,
                )
            self._write_metadata(run_id, {**self._read_metadata(run_id), "status": "completed", "scanned_records": summary["records"], "summary": summary})
        except Exception as exc:
            self._write_metadata(run_id, {**self._read_metadata(run_id), "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    def _read_metadata(self, run_id: str) -> dict[str, Any]:
        metadata_path = self.home / f"{self._validate_run_id(run_id)}.run.json"
        with self._metadata_lock:
            return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _summary(self, run_id: str) -> dict[str, Any]:
        db_path = self._db_path(run_id)
        with RunSession(db_path) as session:
            summary = session.summary()
        return summary

    def get_run(self, run_id: str) -> dict[str, Any]:
        metadata_path = self.home / f"{self._validate_run_id(run_id)}.run.json"
        if not metadata_path.exists():
            raise KeyError(f"Unknown review run: {run_id}")
        metadata = self._read_metadata(run_id)
        if metadata.get("status") == "completed":
            metadata["summary"] = self._summary(run_id)
        return {**metadata, "review_required": True}

    @staticmethod
    def _validate_run_id(run_id: str) -> str:
        if not RUN_ID_RE.fullmatch(run_id):
            raise ValueError("Invalid run_id.")
        return run_id

    def list_findings(self, run_id: str, offset: int = 0, limit: int = 50, code: str = "") -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= MAX_PAGE:
            raise ValueError(f"Use offset >= 0 and limit between 1 and {MAX_PAGE}.")
        db_path = self._db_path(run_id)
        with RunSession(db_path) as session:
            if code:
                total = session.db.execute("SELECT COUNT(*) FROM findings WHERE code=?", (code,)).fetchone()[0]
                rows = session.db.execute(
                    "SELECT f.id,f.row_id,f.code,f.detail,r.source_row,s.path AS source_path "
                    "FROM findings f LEFT JOIN records r ON r.id=f.row_id LEFT JOIN sources s ON s.id=r.source_id "
                    "WHERE f.code=? ORDER BY f.id LIMIT ? OFFSET ?", (code, limit, offset)
                ).fetchall()
            else:
                total = session.db.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
                rows = session.db.execute(
                    "SELECT f.id,f.row_id,f.code,f.detail,r.source_row,s.path AS source_path "
                    "FROM findings f LEFT JOIN records r ON r.id=f.row_id LEFT JOIN sources s ON s.id=r.source_id "
                    "ORDER BY f.id LIMIT ? OFFSET ?", (limit, offset)
                ).fetchall()
            findings = [{**dict(row), "source_path": Path(row["source_path"]).name if row["source_path"] else None} for row in rows]
        return {"run_id": run_id, "total": total, "offset": offset, "limit": limit, "findings": findings}

    def get_records(self, run_id: str, offset: int = 0, limit: int = 25) -> dict[str, Any]:
        if offset < 0 or not 1 <= limit <= MAX_PAGE:
            raise ValueError(f"Use offset >= 0 and limit between 1 and {MAX_PAGE}.")
        with RunSession(self._db_path(run_id)) as session:
            records = session.records(limit=limit, offset=offset)
            total = session.db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        for record in records:
            record["path"] = Path(record["path"]).name if record.get("path") else None
            record["input_path"] = Path(record["input_path"]).name if record.get("input_path") else None
        return {"run_id": run_id, "total": total, "offset": offset, "limit": limit, "records": records}

    def set_record_included(self, run_id: str, record_id: int, include: bool) -> dict[str, Any]:
        if record_id < 1 or not isinstance(include, bool):
            raise ValueError("record_id must be positive and include must be boolean.")
        with RunSession(self._db_path(run_id)) as session:
            found = session.db.execute("SELECT 1 FROM records WHERE id=?", (record_id,)).fetchone()
            if not found:
                raise KeyError(f"Unknown record: {record_id}")
            session.set_include(record_id, include)
        return {"run_id": run_id, "record_id": record_id, "include": include, "source_modified": False}

    def stage_generated_examples(
        self,
        run_id: str,
        examples: list[dict[str, Any]],
        provider: str,
        model: str,
    ) -> dict[str, Any]:
        self._db_path(run_id)
        if not isinstance(examples, list) or not 1 <= len(examples) <= MAX_CANDIDATES:
            raise ValueError(f"Stage between 1 and {MAX_CANDIDATES} examples per call.")
        if not isinstance(provider, str) or not isinstance(model, str) or not provider.strip() or not model.strip():
            raise ValueError("Record the provider and model used to generate candidates.")
        for index, example in enumerate(examples):
            if not isinstance(example, dict) or not example or any(not isinstance(k, str) for k in example):
                raise ValueError(f"Example {index + 1} must be a non-empty JSON object with string keys.")
            if not any(isinstance(value, str) and value.strip() for value in example.values()):
                raise ValueError(f"Example {index + 1} must contain at least one non-empty text field.")
        candidates = self.home / "candidates"
        candidates.mkdir(exist_ok=True)
        candidate_id = uuid.uuid4().hex
        data_path = candidates / f"{run_id}-{candidate_id}.jsonl"
        manifest_path = candidates / f"{run_id}-{candidate_id}.manifest.json"
        with data_path.open("x", encoding="utf-8", newline="\n") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
        manifest_path.write_text(
            json.dumps({
                "candidate_id": candidate_id,
                "run_id": run_id,
                "provider": provider.strip(),
                "model": model.strip(),
                "records": len(examples),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "proposed",
                "approved": False,
                "source_files_modified": False,
            }, indent=2),
            encoding="utf-8",
        )
        return {"candidate_id": candidate_id, "data_file": str(data_path), "manifest_file": str(manifest_path), "records": len(examples), "approved": False}

    def list_candidates(self, run_id: str) -> dict[str, Any]:
        self._db_path(run_id)
        candidates = self.home / "candidates"
        items: list[dict[str, Any]] = []
        for manifest in sorted(candidates.glob(f"{run_id}-*.manifest.json")):
            try:
                item = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            item["data_file"] = str(manifest.with_name(manifest.name.removesuffix(".manifest.json") + ".jsonl"))
            item["manifest_file"] = str(manifest)
            items.append(item)
        return {"run_id": run_id, "candidates": items}

    def preview_export(
        self,
        run_id: str,
        exclude_near: bool = False,
        exclude_incomplete: bool = False,
        exclude_malformed: bool = True,
    ) -> dict[str, Any]:
        with RunSession(self._db_path(run_id)) as session:
            total = session.db.execute("SELECT COUNT(*) FROM records WHERE include_row=1").fetchone()[0]
            exclude = ["duplicate_exact"]
            if exclude_near:
                exclude.append("duplicate_near")
            if exclude_incomplete:
                exclude.extend(["incomplete", "empty_row"])
            if exclude_malformed:
                exclude.append("malformed")
            marks = ",".join("?" for _ in exclude)
            excluded = session.db.execute(
                f"SELECT COUNT(*) FROM records r WHERE r.include_row=1 AND EXISTS "
                f"(SELECT 1 FROM findings f WHERE f.row_id=r.id AND f.code IN ({marks}))", exclude
            ).fetchone()[0]
            summary = session.summary()
        return {
            "run_id": run_id,
            "analyzed_records": summary.get("records", 0),
            "included_before_cleanup": total,
            "excluded_by_cleanup_rules": excluded,
            "estimated_export_records": total - excluded,
            "rules": {"exclude_exact_duplicates": True, "exclude_near_duplicates": exclude_near,
                      "exclude_incomplete": exclude_incomplete, "exclude_malformed": exclude_malformed},
            "license_warnings": summary.get("license_warnings", []),
            "exported": False,
        }

    def export_reviewed(
        self,
        run_id: str,
        output_dir: str,
        dataset_name: str,
        description: str = "",
        output_format: str = "jsonl",
        exclude_near: bool = False,
        exclude_incomplete: bool = False,
        exclude_malformed: bool = True,
        user_approved: bool = False,
    ) -> dict[str, Any]:
        if user_approved is not True:
            raise PermissionError("Export requires explicit user approval after reviewing findings and rights information.")
        if not isinstance(output_dir, str) or not isinstance(dataset_name, str) or not isinstance(description, str):
            raise ValueError("output_dir, dataset_name, and description must be strings.")
        if not isinstance(output_format, str):
            raise ValueError("output_format must be a string.")
        if not output_dir.strip():
            raise ValueError("output_dir is required.")
        if not all(isinstance(flag, bool) for flag in (exclude_near, exclude_incomplete, exclude_malformed)):
            raise ValueError("Export cleanup rules must be booleans.")
        destination = Path(output_dir).expanduser().resolve()
        if not dataset_name.strip():
            raise ValueError("dataset_name is required.")
        with RunSession(self._db_path(run_id)) as session:
            return session.export(
                destination,
                dataset_name,
                description,
                output_format,
                exclude_exact=True,
                exclude_near=exclude_near,
                exclude_incomplete=exclude_incomplete,
                exclude_malformed=exclude_malformed,
            )


def run_metadata_files(home: Path | str | None = None) -> list[dict[str, Any]]:
    root = Path(home or os.environ.get("DATASETCLEANER_AGENT_HOME") or default_agent_home()).expanduser()
    if not root.exists():
        return []
    runs: list[dict[str, Any]] = []
    for path in root.glob("*.run.json"):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    runs.sort(key=lambda item: item.get("created_at", 0), reverse=True)
    return runs

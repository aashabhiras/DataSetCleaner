"""Periodic progress reporter and cron monitor for dataset generation and cleaning runs.

Monitors local or remote checkpoints (checkpoint.json, manifest.json, rlvr_status.json),
computes processing velocity, verification rates, failure breakdowns, and ETAs.
Can run as a one-shot cron job or as a continuous loop daemon.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def find_checkpoint_files(target_dir: Path) -> dict[str, Path]:
    """Find available checkpoint, manifest, or status files in a directory tree."""
    results: dict[str, Path] = {}
    if not target_dir.is_dir():
        return results

    # Search current and subdirectories (up to depth 3)
    for p in target_dir.glob("*"):
        name_lower = p.name.lower()
        if "rlvr_status" in name_lower and p.is_file():
            results["rlvr_status"] = p
        elif "checkpoint" in name_lower and p.suffix == ".json" and p.is_file():
            results["checkpoint"] = p
        elif "manifest" in name_lower and p.suffix == ".json" and p.is_file():
            results["manifest"] = p
        elif "analysis-report" in name_lower and p.suffix == ".json" and p.is_file():
            results["analysis_report"] = p

    # If not found directly, check one level down
    if len(results) < 2:
        for sub in target_dir.glob("*/*"):
            name_lower = sub.name.lower()
            if "checkpoint" in name_lower and sub.suffix == ".json" and "checkpoint" not in results:
                results["checkpoint"] = sub
            elif "manifest" in name_lower and sub.suffix == ".json" and "manifest" not in results:
                results["manifest"] = sub
            elif "rlvr_status" in name_lower and "rlvr_status" not in results:
                results["rlvr_status"] = sub

    return results


def parse_progress_state(target_dir: Path) -> dict[str, Any]:
    """Extract progress metrics from discovered checkpoint files."""
    files = find_checkpoint_files(target_dir)
    state: dict[str, Any] = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "directory": str(target_dir),
        "source_files": [f.name for f in files.values()],
        "total_records": 0,
        "processed_records": 0,
        "verified_records": 0,
        "rejected_records": 0,
        "failed_records": 0,
        "pass_rate_pct": 0.0,
        "rejection_breakdown": {
            "tier1_syntax": 0,
            "tier2_import_compile": 0,
            "tier3_assertion": 0,
            "malformed": 0,
        },
        "output_bytes": 0,
        "active_status": "idle",
        "eta_seconds": None,
        "records_per_minute": 0.0,
    }

    # 1. Inspect rlvr_status.json if present
    if "rlvr_status" in files:
        try:
            with open(files["rlvr_status"], "r", encoding="utf-8") as f:
                data = json.load(f)
            state["active_status"] = data.get("status", "running")
            state["processed_records"] = data.get("processed", 0)
            state["total_records"] = data.get("total", state["total_records"])
            state["verified_records"] = data.get("verified", data.get("success", 0))
            state["rejected_records"] = data.get("rejected", 0)
            state["failed_records"] = data.get("failed", 0)
            if "rejection_breakdown" in data:
                state["rejection_breakdown"].update(data["rejection_breakdown"])
            if "records_per_minute" in data:
                state["records_per_minute"] = data["records_per_minute"]
        except Exception:
            pass

    # 2. Inspect checkpoint.json if present
    if "checkpoint" in files and state["processed_records"] == 0:
        try:
            with open(files["checkpoint"], "r", encoding="utf-8") as f:
                data = json.load(f)
            state["processed_records"] = data.get("processed_records", data.get("completed_rows", data.get("offset", 0)))
            state["total_records"] = data.get("total_records", state["total_records"])
            state["verified_records"] = data.get("verified_count", data.get("success_count", state["processed_records"]))
            state["active_status"] = data.get("state", "in_progress")
        except Exception:
            pass

    # 3. Inspect manifest.json for output file sizes
    if "manifest" in files:
        try:
            with open(files["manifest"], "r", encoding="utf-8") as f:
                data = json.load(f)
            total_size = sum(p.get("size_bytes", p.get("size", 0)) for p in data.get("parts", []))
            state["output_bytes"] = total_size
            if state["total_records"] == 0:
                state["total_records"] = data.get("total_records", sum(p.get("record_count", 0) for p in data.get("parts", [])))
        except Exception:
            pass

    # 4. Measure disk usage of output parts
    if state["output_bytes"] == 0:
        total_disk = 0
        for ext in ("*.jsonl", "*.jsonl.gz", "*.parquet"):
            for f in target_dir.glob(ext):
                try:
                    total_disk += f.stat().st_size
                except OSError:
                    pass
        state["output_bytes"] = total_disk

    # Derive pass rate
    if state["processed_records"] > 0:
        state["pass_rate_pct"] = round((state["verified_records"] / state["processed_records"]) * 100.0, 2)

    return state


def format_bytes(num_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def format_dashboard(state: dict[str, Any]) -> str:
    """Format progress metrics into an ASCII dashboard."""
    total = state["total_records"]
    processed = state["processed_records"]
    verified = state["verified_records"]
    rejected = state["rejected_records"]
    pct = round((processed / total * 100.0), 1) if total > 0 else 0.0

    # Build ASCII progress bar
    bar_width = 30
    filled = int(bar_width * (pct / 100.0))
    bar = "=" * filled + ">" + " " * max(0, bar_width - filled - 1) if filled < bar_width else "=" * bar_width

    lines = [
        "================================================================================",
        f"  DataSetCleaner & Free-RLVR Execution Monitor  --  {state['timestamp'][:19]}Z",
        "================================================================================",
        f"  Status        : {state['active_status'].upper()}",
        f"  Target Dir    : {state['directory']}",
        f"  Found Files   : {', '.join(state['source_files']) or 'Scanning directory...'}",
        "--------------------------------------------------------------------------------",
        f"  Progress      : [{bar}] {pct}% ({processed:,} / {total:,} records)",
        f"  Verified (r=1): {verified:,} ({state['pass_rate_pct']}%)",
        f"  Rejected      : {rejected:,}",
        f"  Failed        : {state['failed_records']:,}",
        "--------------------------------------------------------------------------------",
        "  Rejection Breakdown:",
        f"    * Tier 1 (Syntax Error)         : {state['rejection_breakdown']['tier1_syntax']:,}",
        f"    * Tier 2 (Import/Compile Error) : {state['rejection_breakdown']['tier2_import_compile']:,}",
        f"    * Tier 3 (Assertion Failure)    : {state['rejection_breakdown']['tier3_assertion']:,}",
        f"    * Malformed / Incomplete        : {state['rejection_breakdown']['malformed']:,}",
        "--------------------------------------------------------------------------------",
        f"  Throughput    : {state['records_per_minute']:.1f} records/min",
        f"  Artifact Size : {format_bytes(state['output_bytes'])}",
        "================================================================================",
    ]
    return "\n".join(lines)


def log_progress(state: dict[str, Any], log_file: Path | None) -> None:
    """Append a concise single-line entry to a progress log file."""
    if not log_file:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    summary_line = (
        f"{state['timestamp']} | STATUS={state['active_status']} | "
        f"PROCESSED={state['processed_records']}/{state['total_records']} | "
        f"VERIFIED={state['verified_records']} ({state['pass_rate_pct']}%) | "
        f"SIZE={format_bytes(state['output_bytes'])}\n"
    )
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(summary_line)


def run_reporter(
    target_dir: str | Path,
    *,
    interval: int = 0,
    output_format: str = "text",
    log_file: str | Path | None = None,
    max_iterations: int = 0,
) -> int:
    """Run progress reporting once or periodically."""
    dir_path = Path(target_dir).resolve()
    log_path = Path(log_file).resolve() if log_file else None
    iterations = 0

    while True:
        state = parse_progress_state(dir_path)

        if output_format == "json":
            print(json.dumps(state, indent=2))
        else:
            print(format_dashboard(state))

        log_progress(state, log_path)

        iterations += 1
        if max_iterations > 0 and iterations >= max_iterations:
            break

        if interval <= 0:
            break

        time.sleep(interval)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="datasetcleaner-reporter",
        description="Monitor and report progress of dataset curation or Free-RLVR jobs (one-shot or cron daemon).",
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="Directory to inspect for checkpoint.json, manifest.json, or rlvr_status.json (default: current directory).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Poll interval in seconds. Default is 0 (one-shot check for cron). If > 0, runs continuously as a daemon.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format: 'text' (ASCII dashboard) or 'json' (machine-readable).",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional path to append progress log lines.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        help="Stop after N iterations (useful for testing or finite schedules).",
    )

    args = parser.parse_args()
    return run_reporter(
        args.dir,
        interval=args.interval,
        output_format=args.format,
        log_file=args.log_file,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    sys.exit(main())

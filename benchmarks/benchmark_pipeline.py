"""Generate deterministic local data and measure analysis time/RSS."""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_cleaner.engine import RunSession
from dataset_cleaner.sources import SourceSpec


def run(rows: int, base: Path, bytes_per_record: int, detect_near: bool) -> dict[str, object]:
    temporary = tempfile.TemporaryDirectory(prefix=f"run-{rows}-", dir=base)
    work = Path(temporary.name)
    source = work / "synthetic.jsonl"
    rng = random.Random(42)
    vocabulary = ["token" + "".join(rng.choice(string.ascii_lowercase) for _ in range(8)) for _ in range(8192)]
    payload = " ".join(rng.choices(vocabulary, k=max(8, bytes_per_record // 14)))
    with source.open("w", encoding="utf-8") as handle:
        for index in range(rows):
            sample = {
                "instruction": f"Complete this code task {index:08x} with a correct implementation",
                "completion": " ".join(rng.choices(vocabulary, k=32)),
                "language": "text",
                "context": payload,
            }
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
            if (index + 1) % 100_000 == 0:
                print(f"Generated {index + 1:,}/{rows:,} records", file=sys.stderr, flush=True)
    source_size = source.stat().st_size
    try:
        import psutil
        process = psutil.Process()
        peak_rss = [process.memory_info().rss]
        stop = threading.Event()

        def sample_memory() -> None:
            while not stop.wait(0.05):
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

        monitor = threading.Thread(target=sample_memory, daemon=True)
        monitor.start()
    except ImportError:
        process = None
        peak_rss = [0]
        stop = threading.Event()
        monitor = None
    session = RunSession(work / "review.sqlite")
    started = time.perf_counter()
    try:
        result = session.analyze(
            [SourceSpec(source, "benchmark-fixture")],
            detect_near=detect_near,
            progress=lambda count, _total, message: print(message, file=sys.stderr, flush=True)
            if count % 100_000 == 0 or _total is not None else None,
        )
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        if monitor:
            monitor.join()
        if process:
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
        session.close()
    result_summary = {
        "records": rows,
        "near_duplicates_enabled": detect_near,
        "input_bytes": source_size,
        "elapsed_seconds": round(elapsed, 3),
        "records_per_second": round(rows / elapsed, 1) if elapsed else None,
        "process_peak_rss_mb": round(peak_rss[0] / 1024**2, 1) if peak_rss[0] else None,
        "quality_findings": result["issues"],
    }
    temporary.cleanup()
    return result_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=[1000, 10000], help="Record counts to benchmark, in increasing order.")
    parser.add_argument("--bytes-per-record", type=int, default=1024, help="Approximate additional metadata payload per record.")
    parser.add_argument("--near-duplicates", action="store_true", help="Include the slower near-duplicate scan.")
    parser.add_argument("--work-dir", type=Path, default=Path(".benchmark-output"))
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    results = [run(count, args.work_dir, args.bytes_per_record, args.near_duplicates) for count in args.rows]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

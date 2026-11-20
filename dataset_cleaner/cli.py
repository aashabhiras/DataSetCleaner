"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import __version__
from .engine import RunSession, portable_summary, safe_name, unique_output_path
from .sources import SourceSpec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datasetcleaner",
        description="Merge, audit, review, and export coding datasets locally.",
        epilog="Dataset files remain on this device. Export requires --approve; inputs are never modified.",
    )
    parser.add_argument("--version", action="version", version=f"DataSetCleaner {__version__}")
    parser.add_argument("--input", action="append", required=True, help="CSV, JSONL, Parquet, or a folder containing code files. Repeat to merge inputs.")
    parser.add_argument("--license", action="append", default=[], help="License identifier for corresponding --input (repeat in the same order).")
    parser.add_argument("--source-description", action="append", default=[], help="Description for corresponding --input (repeat in the same order).")
    parser.add_argument("--field-map", type=Path, help="JSON map from source column or 'filename.column' to output column.")
    parser.add_argument("--fields", help="Comma-separated fields used for duplicate checks (default: inferred training fields).")
    parser.add_argument("--threshold", type=float, default=0.90, help="Near-duplicate Jaccard threshold (0.50-1.00, default 0.90).")
    parser.add_argument("--near-duplicates", action="store_true", help="Run the more expensive MinHash/LSH near-duplicate scan (default: exact duplicates only).")
    parser.add_argument("--semantic-duplicates", action="store_true", help="Run Level 4 Multi-Channel Semantic Deduplication with Algorithmic Preservation.")
    parser.add_argument("--semantic-review-threshold", type=float, default=0.85, help="Cosine similarity threshold for semantic review flag (default: 0.85).")
    parser.add_argument("--semantic-duplicate-threshold", type=float, default=0.92, help="Cosine similarity threshold for semantic duplicate candidate (default: 0.92).")
    parser.add_argument("--name", default="coding-dataset", help="Name used for output files.")
    parser.add_argument("--description", default="", help="Dataset description included in the export metadata.")
    parser.add_argument("--output", type=Path, default=Path("datasetcleaner-output"), help="Export/report directory.")
    parser.add_argument("--format", choices=("jsonl", "csv", "parquet"), default="jsonl", help="Dataset export format.")
    parser.add_argument("--approve", action="store_true", help="Approve the reviewed cleanup settings and export the dataset.")
    parser.add_argument("--exclude-near-duplicates", action="store_true", help="Remove near-duplicate records from the export (default: retain and report).")
    parser.add_argument("--exclude-incomplete", action="store_true", help="Remove incomplete/empty records from the export (default: retain and report).")
    parser.add_argument("--keep-malformed", action="store_true", help="Keep malformed JSON records in the export (default: exclude).")
    parser.add_argument("--rewrite-row", type=int, help="Propose a rewrite for this analyzed row using a local Ollama model. Saves a separate proposal and never changes exports.")
    parser.add_argument("--model", default="qwen2.5-coder:1.5b", help="Locally installed Ollama model name for --rewrite-row.")
    return parser


def _benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datasetcleaner benchmark",
        description="Run dataset-only quality benchmarking without fine-tuning.",
    )
    parser.add_argument("--language", required=True, help="Target language (python, cobol, sql, cuda, hcl, etc.)")
    parser.add_argument("--input", type=Path, required=True, help="Input dataset file (.jsonl, .csv, .parquet)")
    parser.add_argument("--output", type=Path, default=Path("benchmark_output"), help="Output directory for benchmark reports and logs")
    return parser


def _package_release_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="datasetcleaner package-release",
        description="Package immutable 3-tier enterprise dataset release distribution.",
    )
    parser.add_argument("--language", required=True, help="Target language (python, cobol, sql, cuda, hcl, etc.)")
    parser.add_argument("--input", type=Path, required=True, help="Raw input directory or dataset source directory")
    parser.add_argument("--output", type=Path, default=Path("releases"), help="Target releases directory")
    parser.add_argument("--db", type=Path, default=None, help="Path to curated review.sqlite")
    parser.add_argument("--approved", type=Path, default=None, help="Path to approved records directory")
    parser.add_argument("--benchmark", type=Path, default=None, help="Path to benchmark artifacts directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "benchmark":
        bench_args = _benchmark_parser().parse_args(argv[1:])
        from .benchmark import DatasetBenchmarkRunner
        runner = DatasetBenchmarkRunner(output_dir=bench_args.output)
        res = runner.benchmark_dataset(bench_args.input, bench_args.language)
        print(json.dumps(res, indent=2))
        return 0

    if argv and argv[0] == "package-release":
        pkg_args = _package_release_parser().parse_args(argv[1:])
        from .benchmark import ReleasePackager
        packager = ReleasePackager(output_dir=pkg_args.output)
        pkg_dir = packager.package_release(
            language=pkg_args.language,
            raw_dir=pkg_args.input,
            db_path=pkg_args.db,
            approved_dir=pkg_args.approved,
            benchmark_dir=pkg_args.benchmark,
        )
        print(f"Release package created: {pkg_dir}")
        return 0

    args = _parser().parse_args(argv)
    if not 0.5 <= args.threshold <= 1.0:
        print("error: --threshold must be between 0.50 and 1.00", file=sys.stderr)
        return 2
    if args.exclude_near_duplicates and not args.near_duplicates:
        print("error: --exclude-near-duplicates requires --near-duplicates", file=sys.stderr)
        return 2
    if args.license and len(args.license) != len(args.input):
        print("error: provide either no --license values or one --license per --input, in the same order", file=sys.stderr)
        return 2
    if args.source_description and len(args.source_description) != len(args.input):
        print("error: provide either no --source-description values or one per --input, in the same order", file=sys.stderr)
        return 2
    try:
        mapping = json.loads(args.field_map.read_text(encoding="utf-8")) if args.field_map else {}
        if not isinstance(mapping, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()):
            raise ValueError("--field-map must contain a JSON object of string keys and string values.")
        specs = [SourceSpec(Path(path), args.license[i] if args.license else "", args.source_description[i] if args.source_description else "")
                 for i, path in enumerate(args.input)]
        args.output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".datasetcleaner-work-", dir=args.output) as temp_dir:
            session = RunSession(Path(temp_dir) / "review.sqlite")
            try:
                summary = session.analyze(specs, field_map=mapping, preferred_fields=[x.strip() for x in args.fields.split(",") if x.strip()] if args.fields else None,
                                          threshold=args.threshold,
                                          detect_near=args.near_duplicates,
                                          progress=lambda count, _total, message: print(message, file=sys.stderr) if count % 500 == 0 else None)
                print(json.dumps({"records": summary["records"], "issues": summary["issues"], "license_warnings": summary["license_warnings"],
                                  "elapsed_seconds": summary["elapsed_seconds"]}, indent=2))
                if args.semantic_duplicates:
                    sem_res = session.detect_semantic_duplicates(
                        threshold_review=args.semantic_review_threshold,
                        threshold_duplicate=args.semantic_duplicate_threshold,
                        progress=lambda count, _total, message: print(message, file=sys.stderr) if count % 500 == 0 else None,
                    )
                    print(json.dumps({"semantic_deduplication": sem_res}, indent=2))
                if args.rewrite_row is not None:
                    suggestion = session.propose_rewrite(args.rewrite_row, args.model)
                    proposal_path = unique_output_path(args.output, f"rewrite-proposal-{args.rewrite_row}.json")
                    proposal_path.write_text(json.dumps({"record_id": args.rewrite_row, "model": args.model, "proposal": suggestion,
                                                        "approved": False, "local_only": True}, indent=2), encoding="utf-8")
                    print(f"Rewrite proposal saved (not applied): {proposal_path}")
                if not args.approve:
                    report = portable_summary(summary)
                    report["review_required"] = True
                    base_name = safe_name(args.name)
                    report_path = unique_output_path(args.output, f"{base_name}.analysis-report.json")
                    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
                    findings_path = unique_output_path(args.output, f"{base_name}.analysis-findings.jsonl")
                    with findings_path.open("w", encoding="utf-8") as handle:
                        for finding in session.iter_findings():
                            finding.pop("data", None)
                            finding.pop("path", None)
                            finding.pop("input_path", None)
                            handle.write(json.dumps(finding, ensure_ascii=False) + "\n")
                    print(f"Analysis only. Review findings, then rerun with --approve to export. Report: {report_path}")
                    return 0
                exported = session.export(args.output, args.name, args.description, args.format,
                                          exclude_exact=True, exclude_near=args.exclude_near_duplicates,
                                          exclude_incomplete=args.exclude_incomplete, exclude_malformed=not args.keep_malformed)
                print(json.dumps(exported, indent=2))
                return 0
            finally:
                session.close()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

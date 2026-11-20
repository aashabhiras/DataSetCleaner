from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_cleaner.cli import main


class CliTests(unittest.TestCase):
    def test_analysis_then_explicit_approved_export(self) -> None:
        work_root = Path(__file__).resolve().parent.parent / ".test-tmp"
        work_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work_root) as directory:
            root = Path(directory)
            source = root / "tiny.jsonl"
            original = '{"prompt":"Write a simple Python function","completion":"def f(): return 1","language":"python"}\n'
            source.write_text(original, encoding="utf-8")
            output = root / "output"
            common = ["--input", str(source), "--license", "MIT", "--output", str(output), "--name", "tiny"]
            self.assertEqual(main(common), 0)
            self.assertTrue((output / "tiny.analysis-report.json").exists())
            self.assertFalse((output / "tiny.jsonl").exists())
            self.assertEqual(main(common + ["--approve"]), 0)
            report_path = next(output.glob("tiny*.quality-report.json"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["exported_records"], 1)
            self.assertEqual(source.read_text(encoding="utf-8"), original)

    def test_cli_semantic_duplicates_flag(self) -> None:
        work_root = Path(__file__).resolve().parent.parent / ".test-tmp"
        work_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work_root) as directory:
            root = Path(directory)
            source = root / "sample.jsonl"
            source.write_text('{"instruction":"task1","completion":"print(1)"}\n', encoding="utf-8")
            output = root / "output"
            args = ["--input", str(source), "--output", str(output), "--semantic-duplicates"]
            self.assertEqual(main(args), 0)

    def test_cli_benchmark_subcommand(self) -> None:
        work_root = Path(__file__).resolve().parent.parent / ".test-tmp"
        work_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work_root) as directory:
            root = Path(directory)
            source = root / "sample.jsonl"
            source.write_text('{"instruction":"add","completion":"def add(a, b): return a + b"}\n', encoding="utf-8")
            bm_out = root / "benchmark_out"
            args = ["benchmark", "--language", "python", "--input", str(source), "--output", str(bm_out)]
            self.assertEqual(main(args), 0)
            self.assertTrue((bm_out / "benchmark_results.json").is_file())

    def test_cli_package_release_subcommand(self) -> None:
        work_root = Path(__file__).resolve().parent.parent / ".test-tmp"
        work_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work_root) as directory:
            root = Path(directory)
            raw_dir = root / "raw_src"
            raw_dir.mkdir()
            (raw_dir / "test.py").write_text("x = 1", encoding="utf-8")
            releases_dir = root / "releases"
            args = ["package-release", "--language", "python", "--input", str(raw_dir), "--output", str(releases_dir)]
            self.assertEqual(main(args), 0)
            self.assertTrue((releases_dir / "python" / "manifest.sha256").is_file())


if __name__ == "__main__":
    unittest.main()

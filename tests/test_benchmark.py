from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from dataset_cleaner.benchmark import DatasetBenchmarkRunner, ReleasePackager


class BenchmarkTests(unittest.TestCase):
    def test_dataset_benchmark_runner_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_file = root / "dataset.jsonl"
            records = [
                {
                    "instruction": "Write a Python function to add two numbers.",
                    "completion": "def add(a, b):\n    return a + b\n\nassert add(1, 2) == 3\n",
                    "language": "python",
                },
                {
                    "instruction": "Write a function that has a syntax error.",
                    "completion": "def broken(\n",
                    "language": "python",
                },
            ]
            with dataset_file.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            out_dir = root / "benchmark_out"
            runner = DatasetBenchmarkRunner(output_dir=out_dir)
            res = runner.benchmark_dataset(dataset_file, "python")

            self.assertIn("metrics", res)
            metrics = res["metrics"]
            self.assertEqual(metrics["total_examples"], 2)
            self.assertEqual(metrics["syntactically_valid"], 1)
            self.assertEqual(metrics["parse_rate_pct"], 50.0)

            # Check that files were generated
            self.assertTrue((out_dir / "benchmark_results.json").is_file())
            self.assertTrue((out_dir / "benchmark_config.json").is_file())
            self.assertTrue((out_dir / "DATASET_QUALITY_REPORT.md").is_file())

            report_text = (out_dir / "DATASET_QUALITY_REPORT.md").read_text(encoding="utf-8")
            self.assertIn("# Enterprise Dataset Quality Audit Report", report_text)
            self.assertIn("Parse Rate", report_text)

    def test_release_packager(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            (raw_dir / "main.py").write_text("print('hello')", encoding="utf-8")

            db_path = root / "review.sqlite"
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE meta (key TEXT, value TEXT)")
            conn.commit()
            conn.close()

            approved_dir = root / "approved"
            approved_dir.mkdir()
            (approved_dir / "dataset.jsonl").write_text('{"prompt": "p", "completion": "c"}\n', encoding="utf-8")

            benchmark_dir = root / "benchmark_artifacts"
            benchmark_dir.mkdir()
            (benchmark_dir / "benchmark_results.json").write_text('{"score": 100}\n', encoding="utf-8")

            releases_out = root / "releases"
            packager = ReleasePackager(output_dir=releases_out)
            pkg_path = packager.package_release(
                language="python",
                raw_dir=raw_dir,
                db_path=db_path,
                approved_dir=approved_dir,
                benchmark_dir=benchmark_dir,
            )

            self.assertTrue(pkg_path.is_dir())
            self.assertEqual(pkg_path.name, "python")
            self.assertTrue((pkg_path / "raw" / "main.py").is_file())
            self.assertTrue((pkg_path / "curated" / "review.sqlite").is_file())
            self.assertTrue((pkg_path / "approved" / "dataset.jsonl").is_file())
            self.assertTrue((pkg_path / "benchmark" / "benchmark_results.json").is_file())

            manifest_path = pkg_path / "manifest.sha256"
            self.assertTrue(manifest_path.is_file())
            manifest_content = manifest_path.read_text(encoding="utf-8").strip()
            self.assertGreater(len(manifest_content.splitlines()), 0)


if __name__ == "__main__":
    unittest.main()

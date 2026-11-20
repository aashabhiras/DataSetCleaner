from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_cleaner.reporter import (
    find_checkpoint_files,
    format_bytes,
    format_dashboard,
    parse_progress_state,
    run_reporter,
)


class ReporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_format_bytes(self) -> None:
        self.assertEqual(format_bytes(500), "500.0 B")
        self.assertEqual(format_bytes(1024), "1.0 KB")
        self.assertEqual(format_bytes(1024 * 1024 * 5), "5.0 MB")
        self.assertEqual(format_bytes(1024 * 1024 * 1024 * 2), "2.0 GB")

    def test_find_checkpoint_files_empty(self) -> None:
        files = find_checkpoint_files(self.test_dir)
        self.assertEqual(files, {})

    def test_find_and_parse_rlvr_status(self) -> None:
        status_path = self.test_dir / "rlvr_status.json"
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump({
                "status": "processing",
                "processed": 150,
                "total": 500,
                "verified": 140,
                "rejected": 8,
                "failed": 2,
                "records_per_minute": 25.5,
                "rejection_breakdown": {
                    "tier1_syntax": 3,
                    "tier2_import_compile": 3,
                    "tier3_assertion": 2,
                    "malformed": 0
                }
            }, f)

        files = find_checkpoint_files(self.test_dir)
        self.assertIn("rlvr_status", files)

        state = parse_progress_state(self.test_dir)
        self.assertEqual(state["active_status"], "processing")
        self.assertEqual(state["processed_records"], 150)
        self.assertEqual(state["total_records"], 500)
        self.assertEqual(state["verified_records"], 140)
        self.assertEqual(state["pass_rate_pct"], 93.33)
        self.assertEqual(state["rejection_breakdown"]["tier1_syntax"], 3)

        dashboard = format_dashboard(state)
        self.assertIn("150 / 500 records", dashboard)
        self.assertIn("Tier 1 (Syntax Error)", dashboard)
        self.assertIn("93.33%", dashboard)

    def test_run_reporter_one_shot(self) -> None:
        log_file = self.test_dir / "reporter.log"
        exit_code = run_reporter(self.test_dir, interval=0, log_file=log_file, max_iterations=1)
        self.assertEqual(exit_code, 0)
        self.assertTrue(log_file.exists())
        content = log_file.read_text(encoding="utf-8")
        self.assertIn("STATUS=", content)


if __name__ == "__main__":
    unittest.main()

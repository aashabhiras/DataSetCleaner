from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dataset_cleaner.agent_service import AgentService
from dataset_cleaner.api import _ApiServer


class AgentServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        test_root = Path(__file__).resolve().parent.parent / ".test-tmp"
        test_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.temp.name)
        self.service = AgentService(self.root / "state")
        self.source = self.root / "sample.jsonl"
        self.original = (
            json.dumps({"prompt": "Write a Python function that adds two values", "completion": "def add(a, b): return a + b", "language": "python"}) + "\n"
            + json.dumps({"prompt": "Write a Python function that multiplies two values", "completion": "def multiply(a, b): return a * b", "language": "python"}) + "\n"
        )
        self.source.write_text(self.original, encoding="utf-8")

    def tearDown(self) -> None:
        self.service._executor.shutdown(wait=True)
        self.temp.cleanup()

    def complete_run(self) -> str:
        result = self.service.start_analysis([{"path": str(self.source), "license_id": "MIT", "description": "test"}])
        run_id = result["run_id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = self.service.get_run(run_id)
            if status["status"] != "running":
                self.assertEqual(status["status"], "completed", status)
                return run_id
            time.sleep(0.02)
        self.fail("Analysis did not complete within 10 seconds.")

    def test_analysis_review_candidates_and_approved_export(self) -> None:
        run_id = self.complete_run()
        self.assertEqual(self.source.read_text(encoding="utf-8"), self.original)
        records = self.service.get_records(run_id)
        self.assertEqual(records["total"], 2)
        self.service.set_record_included(run_id, records["records"][0]["id"], False)
        candidate = self.service.stage_generated_examples(
            run_id, [{"prompt": "Explain a Python list", "completion": "A mutable ordered collection."}], "Gemini", "test-model"
        )
        self.assertFalse(candidate["approved"])
        self.assertEqual(len(self.service.list_candidates(run_id)["candidates"]), 1)
        preview = self.service.preview_export(run_id)
        self.assertEqual(preview["included_before_cleanup"], 1)
        with self.assertRaises(PermissionError):
            self.service.export_reviewed(run_id, str(self.root / "export"), "sample")
        output = self.service.export_reviewed(run_id, str(self.root / "export"), "sample", user_approved=True)
        self.assertEqual(output["exported_records"], 1)
        self.assertEqual(self.source.read_text(encoding="utf-8"), self.original)


class AgentApiTests(unittest.TestCase):
    def setUp(self) -> None:
        test_root = Path(__file__).resolve().parent.parent / ".test-tmp"
        test_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.temp.name)
        self.service = AgentService(self.root / "state")
        self.server = _ApiServer(("127.0.0.1", 0), self.service)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.service._executor.shutdown(wait=True)
        self.temp.cleanup()

    def request_json(self, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(self.base + path, data=data, headers={"Content-Type": "application/json", "Connection": "close"})
        try:
            response = urlopen(request, timeout=3)
        except HTTPError as exc:
            with exc:
                return exc.code, json.loads(exc.read())
        with response:
            return response.status, json.loads(response.read())

    def test_loopback_dashboard_and_strict_boolean_validation(self) -> None:
        status, health = self.request_json("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["local_only"])
        page = urlopen(self.base + "/", timeout=3).read().decode("utf-8")
        self.assertIn("Analyze local sources", page)

        source = self.root / "sample.jsonl"
        source.write_text(json.dumps({"prompt": "Explain a for loop", "completion": "It repeats a block."}) + "\n", encoding="utf-8")
        status, error = self.request_json("/api/runs", {"sources": [{"path": str(source)}], "detect_near": "false"})
        self.assertEqual(status, 400)
        self.assertIn("JSON boolean", error["error"])
        status, error = self.request_json("/api/runs", {"sources": [{"path": str(source)}], "check_syntax": "false"})
        self.assertEqual(status, 400)
        self.assertIn("check_syntax", error["error"])

    def test_get_validation_errors_return_json_not_connection_failure(self) -> None:
        status, error = self.request_json("/api/runs/not-a-run/records?limit=abc")
        self.assertEqual(status, 400)
        self.assertIn("invalid literal", error["error"])


if __name__ == "__main__":
    unittest.main()

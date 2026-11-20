from __future__ import annotations

import json
import tempfile
import unittest
import importlib.util
from unittest.mock import patch
from pathlib import Path
from unittest.mock import Mock, patch

from dataset_cleaner.engine import RunSession
from dataset_cleaner.sandbox import SandboxedResult
from dataset_cleaner.sources import SourceSpec
from dataset_cleaner.token_count import count_jsonl_tokens


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        work_root = Path(__file__).resolve().parent.parent / ".test-tmp"
        work_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=work_root)
        self.root = Path(self.temp.name)
        self.db_path = self.root / "review.sqlite"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_merge_review_findings_and_export_preserve_source(self) -> None:
        prepared = self.root / "prepared.jsonl"
        csv_source = self.root / "supplement.csv"
        task = "Implement a Python function that adds two integer values while preserving the original order and returning the result without mutating inputs and using clear parameter names in the documented function"
        first = {"instruction": task, "completion": "def add(a, b): return a + b", "language": "python"}
        exact = {"instruction": task, "completion": "def add(a, b): return a + b", "language": "python"}
        near = {"instruction": task + " correctly", "completion": "def add(a, b): return a + b", "language": "python"}
        invalid = {"instruction": "Write Python code", "completion": "def broken(:", "language": "python"}
        lines = [first, exact, near, invalid, {"instruction": "", "completion": "", "language": "python"}]
        original = "\n".join(json.dumps(item) for item in lines) + '\n{"broken":\n'
        prepared.write_text(original, encoding="utf-8")
        csv_source.write_text("task,answer\nExplain a loop,for item in items: pass\n", encoding="utf-8")
        with RunSession(self.db_path) as session:
            summary = session.analyze(
                [SourceSpec(prepared, "MIT", "Prepared examples"), SourceSpec(csv_source, "Apache-2.0", "Supplement")],
                field_map={"supplement.csv.task": "instruction", "supplement.csv.answer": "completion"},
                threshold=0.75,
            )
            self.assertEqual(summary["records"], 7)
            self.assertEqual(summary["issues"].get("duplicate_exact"), 1)
            self.assertEqual(summary["issues"].get("duplicate_near"), 1)
            self.assertGreaterEqual(
                summary["issues"].get("syntax", 0) + summary["issues"].get("syntax_unchecked", 0), 1
            )
            self.assertEqual(summary["issues"].get("malformed"), 1)
            self.assertGreaterEqual(summary["issues"].get("incomplete", 0), 1)
            self.assertEqual(len(summary["license_warnings"]), 1)
            exported = session.export(self.root / "export", "sample", "review fixture", "jsonl", exclude_exact=True)
            self.assertEqual(exported["exported_records"], 5)
            output = [json.loads(line) for line in Path(exported["data_file"]).read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(row.get("instruction") == "Explain a loop" for row in output))
            self.assertTrue(Path(exported["report_file"]).exists())
            self.assertTrue(Path(exported["provenance_file"]).exists())
            self.assertTrue(Path(exported["findings_file"]).exists())
            self.assertNotIn(str(prepared), Path(exported["report_file"]).read_text(encoding="utf-8"))
            self.assertNotIn(str(prepared), Path(exported["provenance_file"]).read_text(encoding="utf-8"))
            self.assertNotIn(str(prepared), Path(exported["findings_file"]).read_text(encoding="utf-8"))
        self.assertEqual(prepared.read_text(encoding="utf-8"), original)

    def test_export_does_not_overwrite_existing_package(self) -> None:
        source = self.root / "data.jsonl"
        source.write_text('{"prompt":"How to return a value","completion":"return value"}\n', encoding="utf-8")
        output = self.root / "out"
        output.mkdir()
        (output / "dataset.jsonl").write_text("keep this", encoding="utf-8")
        with RunSession(self.db_path) as session:
            session.analyze([SourceSpec(source, "MIT")])
            result = session.export(output, "dataset", "", "jsonl")
        self.assertTrue(result["data_file"].endswith("dataset-v2.jsonl"))
        self.assertEqual((output / "dataset.jsonl").read_text(encoding="utf-8"), "keep this")

    def test_code_folder_is_streamed_and_skips_repository_artifacts(self) -> None:
        folder = self.root / "code"
        (folder / "src").mkdir(parents=True)
        (folder / ".git").mkdir()
        (folder / "src" / "module.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
        (folder / ".git" / "object.py").write_text("def broken(:\n", encoding="utf-8")
        with RunSession(self.db_path) as session:
            summary = session.analyze([SourceSpec(folder, "MIT", "Owned code")])
            self.assertEqual(summary["records"], 1)
            self.assertEqual(len(summary["sources"]), 1)
            self.assertEqual(summary["sources"][0]["format"], "code_folder")
            self.assertTrue(set(summary["issues"]).issubset({"syntax_unchecked"}))
            self.assertLessEqual(summary["issues"].get("syntax_unchecked", 0), 2)
            self.assertEqual(session.records()[0]["data"]["filename"], "src/module.py")

    def test_exact_duplicate_detection_preserves_code_indentation(self) -> None:
        source = self.root / "indentation.jsonl"
        rows = [
            {"instruction": "Show the Python block", "code": "if ready:\n    return 1", "language": "python"},
            {"instruction": "Show the Python block", "code": "if ready:\n        return 1", "language": "python"},
        ]
        source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        with RunSession(self.db_path) as session:
            summary = session.analyze([SourceSpec(source, "MIT")])
            self.assertEqual(summary["issues"].get("duplicate_exact", 0), 0)

    def test_query_lang_resource_schema_and_local_syntax_checks(self) -> None:
        source = self.root / "examples.jsonl"
        rows = [
            {"query": "Write JavaScript", "answer": "const value = ; // 🌍", "resource": "source text", "lang": "javascript"},
            {"query": "Use an ES module", "answer": "import fs from 'node:fs';", "resource": "module source", "lang": "javascript"},
            {"query": "Use JSX", "answer": "const view = <div />;", "resource": "JSX source", "lang": "jsx"},
            {"query": "Write C", "answer": "int main( { /* café 🌍 */", "resource": "source text", "lang": "c"},
            {"query": "Write C++", "answer": "int main( {", "resource": "source text", "lang": "cpp"},
            {"query": "Write Python", "answer": "def broken(:", "resource": "source text", "lang": "python"},
            {"query": "Inspect an escape warning", "answer": r"x = '\q'", "resource": "source text", "lang": "python"},
        ]
        source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        def sandboxed_result(executable: str, args: list[str], code: str, suffix: str) -> SandboxedResult:
            if "-c" in args:
                parser_json = {"syntax_error": None, "parser_error": None, "warnings": []}
                if code == "def broken(:":
                    parser_json["syntax_error"] = "invalid syntax"
                if code == r"x = '\q'":
                    parser_json["warnings"] = ["invalid escape sequence '\\q'"]
                return SandboxedResult(0, json.dumps(parser_json), "")
            if "--input-type=module" in args:
                return SandboxedResult(0 if code == "import fs from 'node:fs';" else 1, "", "SyntaxError: invalid syntax")
            if code == "import fs from 'node:fs';":
                return SandboxedResult(1, "", "Cannot use import statement outside a module")
            return SandboxedResult(1, "", "SyntaxError: invalid syntax\nexpected declaration")

        with patch("dataset_cleaner.engine.shutil.which", side_effect=lambda name: f"C:/tools/{name}.exe"), patch(
            "dataset_cleaner.engine.run_sandboxed_validator", side_effect=sandboxed_result
        ), RunSession(self.db_path) as session:
            summary = session.analyze([SourceSpec(source, "MIT")])
            self.assertEqual(summary["records"], 7)
            self.assertGreaterEqual(summary["issues"].get("syntax", 0), 4)
            self.assertEqual(summary["issues"].get("syntax_warning", 0), 1)
            self.assertEqual(summary["issues"].get("syntax_unchecked", 0), 1)
            self.assertTrue(all("query" in row["data"] for row in session.records()))

    def test_syntax_checks_can_be_skipped_and_are_reported(self) -> None:
        source = self.root / "skip-syntax.jsonl"
        source.write_text(json.dumps({"query": "Write Python", "answer": "def broken(:", "lang": "python"}) + "\n", encoding="utf-8")
        with RunSession(self.db_path) as session:
            summary = session.analyze([SourceSpec(source, "MIT")], check_syntax=False)
        self.assertFalse(summary["syntax_checks_enabled"])
        self.assertNotIn("syntax", summary["issues"])

    def test_rewrite_proposal_uses_only_loopback_and_is_separate(self) -> None:
        source = self.root / "prompt.jsonl"
        source.write_text('{"prompt":"Write a function","completion":"def f(): return 1"}\n', encoding="utf-8")

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return b'{"response":"def f():\\n    return 1"}'

        opener = Mock()
        opener.open.return_value = Response()
        with RunSession(self.db_path) as session:
            session.analyze([SourceSpec(source, "MIT")])
            with patch("urllib.request.build_opener", return_value=opener) as build_opener:
                proposal = session.propose_rewrite(1, "local-coder")
            self.assertIn("return 1", proposal)
            proxy_handler = build_opener.call_args.args[0]
            self.assertEqual(proxy_handler.proxies, {})
            request = opener.open.call_args.args[0]
            self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/generate")

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional PyArrow dependency is not installed")
    def test_parquet_read_and_write_round_trip(self) -> None:
        import pyarrow.parquet as pq

        source = self.root / "typed.jsonl"
        rows = [
            {"prompt": "Explain a value", "completion": "A value is data.", "score": 0.75, "active": True, "metadata": {"source": "local"}},
            {"prompt": "Explain a function", "completion": "A function returns a result.", "score": 1, "active": False, "metadata": {"source": "local"}},
        ]
        source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        with RunSession(self.db_path) as session:
            session.analyze([SourceSpec(source, "MIT")])
            exported = session.export(self.root / "parquet-out", "typed", "typed records", "parquet")
        table = pq.read_table(exported["data_file"])
        data = table.to_pylist()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["prompt"], rows[0]["prompt"])
        self.assertEqual(data[0]["active"], True)
        self.assertEqual(data[0]["metadata"], '{"source":"local"}')
        with RunSession(self.root / "parquet-review.sqlite") as parquet_session:
            summary = parquet_session.analyze([SourceSpec(Path(exported["data_file"]), "MIT")])
            self.assertEqual(summary["records"], 2)

    def test_token_counter_reports_record_and_field_counts_without_values(self) -> None:
        class WordEncoding:
            def encode_ordinary(self, text: str) -> list[str]:
                return text.split()

        source = self.root / "tokens.jsonl"
        original = '{"query":"write code","answer":"return value","lang":"python"}\nnot json\n'
        source.write_text(original, encoding="utf-8")
        with patch("tiktoken.get_encoding", return_value=WordEncoding()):
            result = count_jsonl_tokens(source, fields=["query", "answer"])
        self.assertEqual(result["records"], 2)
        self.assertEqual(result["malformed_records"], 1)
        self.assertEqual(result["field_value_tokens_total"], 4)
        self.assertEqual(source.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()

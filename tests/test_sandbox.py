from __future__ import annotations

import unittest
import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import patch

from dataset_cleaner.engine import _find_row_issues
from dataset_cleaner.sandbox import (
    MAX_SOURCE_CHARS,
    SandboxInputLimit,
    SandboxUnavailable,
    SandboxedResult,
    run_sandboxed_validator,
)


class SandboxTests(unittest.TestCase):
    def test_validator_reports_successful_parser_result(self) -> None:
        temp_root = Path(__file__).resolve().parent.parent / ".test-tmp"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            executable = Path(directory) / "parser.exe"
            executable.write_bytes(b"test")
            with patch("dataset_cleaner.sandbox._run_appcontainer", return_value=SandboxedResult(0, "", "")) as run:
                result = run_sandboxed_validator(str(executable), ["--check", "{input}"], "const x = 1", ".js")
            self.assertEqual(result.returncode, 0)
            run.assert_called_once_with(str(executable.resolve()), ["--check", "{input}"], "const x = 1", ".js", 5)

    def test_oversized_source_is_rejected_before_process_setup(self) -> None:
        with patch("dataset_cleaner.sandbox._win_api") as win_api:
            with self.assertRaises(SandboxInputLimit):
                # The executable path is checked before the source bound.
                run_sandboxed_validator(__file__, ["{input}"], "x" * (MAX_SOURCE_CHARS + 1), ".js")
        win_api.assert_not_called()

    def test_external_syntax_check_fails_closed_without_sandbox(self) -> None:
        row = {"lang": "javascript", "answer": "throw new Error('must not run')"}
        with patch("dataset_cleaner.engine.shutil.which", return_value="C:/trusted/node.exe"), patch(
            "dataset_cleaner.engine.run_sandboxed_validator",
            side_effect=SandboxUnavailable("AppContainer unavailable"),
        ) as run:
            issues = _find_row_issues(row, "", "examples.jsonl")
        self.assertEqual([kind for kind, _ in issues], ["syntax_unchecked"])
        self.assertIn("AppContainer unavailable", issues[0][1])
        run.assert_called_once()

    def test_external_syntax_finding_uses_sandboxed_parser_result(self) -> None:
        row = {"lang": "c", "code": "int main( {"}
        with patch("dataset_cleaner.engine.shutil.which", return_value="C:/trusted/gcc.exe"), patch(
            "dataset_cleaner.engine.run_sandboxed_validator",
            return_value=SandboxedResult(1, "", "input.c:1: error: expected declaration"),
        ) as run:
            issues = _find_row_issues(row, "", "examples.jsonl")
        self.assertEqual([kind for kind, _ in issues], ["syntax"])
        self.assertIn("expected declaration", issues[0][1])
        run.assert_called_once()
        self.assertIn("{input}", run.call_args.args[1])

    def test_python_parser_is_confined_and_receives_text_as_data(self) -> None:
        source = "__import__('pathlib').Path('sentinel').write_text('changed')"
        parsed = {"syntax_error": None, "parser_error": None, "warnings": []}
        with patch("dataset_cleaner.engine.run_sandboxed_validator", return_value=SandboxedResult(0, json.dumps(parsed), "")) as run:
            issues = _find_row_issues({"lang": "python", "code": source}, "", "examples.jsonl")
        self.assertEqual(issues, [])
        executable, args, received_source, suffix = run.call_args.args
        self.assertEqual(executable, sys.executable)
        self.assertEqual(received_source, source)
        self.assertEqual(suffix, ".py")
        self.assertIn("-B", args)
        self.assertIn("-I", args)
        self.assertIn("-S", args)
        self.assertIn("ast.parse", args[args.index("-c") + 1])
        self.assertNotIn(source, args[args.index("-c") + 1])

    def test_python_parser_fails_closed_when_sandbox_is_unavailable(self) -> None:
        with patch("dataset_cleaner.engine.run_sandboxed_validator", side_effect=SandboxUnavailable("AppContainer unavailable")):
            issues = _find_row_issues({"lang": "python", "code": "print('must not run')"}, "", "examples.jsonl")
        self.assertEqual([kind for kind, _ in issues], ["syntax_unchecked"])
        self.assertIn("AppContainer unavailable", issues[0][1])


if __name__ == "__main__":
    unittest.main()

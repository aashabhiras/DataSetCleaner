from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from dataset_cleaner.engine import _content_fields


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "kaggle_batch_inference.ipynb"


class NotebookDataSetCleanerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        cls.cells = ["".join(cell.get("source", [])) for cell in cls.notebook["cells"]]

        verify_cell = ast.parse(cls.cells[9])
        pair_function = next(
            node for node in verify_cell.body
            if isinstance(node, ast.FunctionDef) and node.name == "datasetcleaner_pair"
        )
        namespace = {
            "json": json,
            "LANGUAGE": "python",
            "DATASET_SHA": "dataset-sha",
            "DATASET_REPO": "bigcode/the-stack-dedup",
            "SHARD_FILE": "data/python/part-000.parquet",
            "MODEL_REPO": "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit",
            "MODEL_SHA": "model-sha",
        }
        module = ast.Module(body=[pair_function], type_ignores=[])
        exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
        cls.to_row = staticmethod(namespace["datasetcleaner_pair"])

        reader_cell = ast.parse(cls.cells[7])
        normalize_function = next(
            node for node in reader_cell.body
            if isinstance(node, ast.FunctionDef) and node.name == "normalize_source_record"
        )
        reader_namespace = {"SOURCE_CONTENT_FIELD": "auto"}
        exec(compile(ast.Module(body=[normalize_function], type_ignores=[]), str(NOTEBOOK_PATH), "exec"), reader_namespace)
        cls.normalize_source = staticmethod(reader_namespace["normalize_source_record"])
        cls.reader_namespace = reader_namespace

    def test_notebook_json_and_python_cells_are_valid(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        for index, cell in enumerate(self.notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = self.cells[index]
            if source.lstrip().startswith("%"):
                continue
            ast.parse(source)

    def test_model_is_prequantized_awq_checkpoint(self) -> None:
        self.assertIn('MODEL_REPO = "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"', self.cells[1])
        self.assertIn('quantization="awq"', self.cells[5])
        self.assertIn("from vllm import LLM", self.cells[5])

    def test_dual_gpu_model_placement_is_required_and_verified(self) -> None:
        self.assertIn("torch.cuda.device_count() < 2", self.cells[5])
        self.assertIn("tensor_parallel_size=min(torch.cuda.device_count(), 2)", self.cells[5])
        self.assertIn('device_map="balanced"', self.cells[5])

    def test_program_runner_is_isolated_and_only_receives_generated_code(self) -> None:
        self.assertIn('"isolate"', self.cells[2])
        self.assertIn('"build-essential"', self.cells[2])
        self.assertIn('"nodejs"', self.cells[2])
        self.assertIn('"default-jdk-headless"', self.cells[2])
        self.assertIn('"gnucobol"', self.cells[2])
        self.assertIn('"--cg", "--quota=4096,128", "--init"', self.cells[3])
        self.assertIn('SANDBOX_LANGUAGES = ("python", "c", "cpp", "javascript", "java", "cobol")', self.cells[6])
        self.assertIn('"--cg"', self.cells[6])
        self.assertIn('"--cg-mem=', self.cells[6])
        self.assertIn('"--processes=', self.cells[6])
        self.assertIn('"--fsize=256"', self.cells[6])
        self.assertIn("run_sandboxed_program(\"python\", harness)", self.cells[6])
        self.assertIn("generated_test_execution", self.cells[7])
        self.assertIn('"raw_source_executed": False', self.cells[7])

    def test_sandbox_execution_fails_closed_and_is_provenanced(self) -> None:
        self.assertIn("RUN_SANDBOXED_TESTS = True", self.cells[1])
        self.assertIn('OUTPUT_FORMAT == "function_tests" and not SANDBOX_READY', self.cells[3])
        self.assertIn("refusing to execute code", self.cells[6])
        self.assertIn('"sandbox": {"engine": "Isolate"', self.cells[7])
        self.assertIn('"network_isolated": True', self.cells[6])

    def test_drive_input_is_configured_and_maps_content_without_mutating_source(self) -> None:
        self.assertIn('INPUT_SOURCE = "google_drive"', self.cells[1])
        self.assertIn('DRIVE_TOKEN_SECRET_NAME = "GOOGLE_DRIVE_TOKEN_JSON"', self.cells[1])
        self.assertIn('googleapiclient.http import MediaIoBaseDownload', self.cells[4])
        self.assertIn('googleapiclient.http import MediaFileUpload', self.cells[9])
        self.assertIn('HF_STORAGE_DESTINATION = "kaggle_disk"', self.cells[1])
        self.assertIn('raw_dir = WORK / "raw_datasets"', self.cells[4])
        source = {"code": "def add(a, b):\n    return a + b", "license": "MIT"}
        normalized = self.normalize_source(source)
        self.assertEqual(normalized["content"], source["code"])
        self.assertEqual(normalized["_source_content_field"], "code")
        self.assertNotIn("content", source)
        self.reader_namespace["SOURCE_CONTENT_FIELD"] = "python_code"
        mapped = self.normalize_source({"content": "wrong field", "python_code": "def x():\n    pass"})
        self.assertEqual(mapped["content"], "def x():\n    pass")

    def test_all_generated_formats_map_to_flat_reviewable_rows(self) -> None:
        examples = {
            "code_task": {"instruction": "Implement add", "solution": "def add(a, b):\n    return a + b", "explanation": "Adds."},
            "doc_to_impl": {"documentation": "A documented API.", "instruction": "Implement API", "solution": "def api():\n    return 1"},
            "compiler_repair": {"task": "Repair Python", "error_message": "SyntaxError", "broken_code": "def f(:", "fixed_code": "def f():\n    pass", "explanation": "Added parameters."},
            "bug_patch": {"buggy_code": "def f():\n    return None", "fixed_code": "def f():\n    return 1", "reason": "Wrong value."},
            "commit_repair": {"commit_message": "fix result", "before_code": "def f():\n    return 0", "after_code": "def f():\n    return 1", "explanation": "Correct value."},
            "function_tests": {"expected_behavior": "adds values", "function_code": "def add(a,b):\n    return a+b", "tests": "def test_add():\n    assert add(1, 2) == 3"},
            "repo_dependency": {"instruction": "Use math", "available_dependencies": ["math"], "solution": "import math\ndef f():\n    return math.floor(1.2)", "dependencies_used": ["math"]},
        }
        for fmt, data in examples.items():
            with self.subTest(format=fmt):
                row = self.to_row({
                    "data": data,
                    "format": fmt,
                    "index": 12,
                    "source": {
                        "lang": "python",
                        "dataset": "bigcode/the-stack-dedup",
                        "shard": "data/python/part-000.parquet",
                        "path": "src/module.py",
                        "repo": "sample/repo",
                        "licenses": ["MIT"],
                        "content_sha256": "abc123",
                    },
                })
                self.assertTrue(row["instruction"])
                self.assertTrue(row["completion"])
                self.assertIn("instruction", _content_fields(row))
                self.assertIn("completion", _content_fields(row))
                self.assertEqual(row["language"], "python")
                self.assertEqual(row["lang"], "python")
                self.assertEqual(row["source_licenses_json"], '["MIT"]')
                self.assertEqual(row["source_content_sha256"], "abc123")
                self.assertTrue(row["generation_provenance"])
                self.assertTrue(row["synthetic_origin"])
                self.assertIsInstance(row["artifact_provenance"], dict)
                self.assertTrue(row["license_confidence"])
                self.assertEqual(row["datasetcleaner_status"], "unreviewed_candidate")
                self.assertFalse(row["approved"])


if __name__ == "__main__":
    unittest.main()

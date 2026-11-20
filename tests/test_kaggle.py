from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from dataset_cleaner.kaggle import (
    DEFAULT_MODEL_HANDLE,
    KaggleCommandResult,
    KaggleError,
    _kaggle_kernel_source,
    _run_kaggle,
    download_dataset,
    download_kernel_outputs,
    get_kaggle_auth_status,
    kaggle_cli_environment,
    kaggle_config_dir,
    prepare_kaggle_config_dir,
    push_batch_notebook,
    run_curation,
    save_kaggle_credentials,
)


class KaggleTests(unittest.TestCase):
    def test_cli_uses_dedicated_config_directory_for_login_and_commands(self) -> None:
        self.assertTrue(str(kaggle_config_dir()).endswith("DataSetCleaner/Kaggle") or
                        str(kaggle_config_dir()).endswith("DataSetCleaner\\Kaggle"))
        root = Path(__file__).resolve().parent.parent / ".test-tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            config_dir = Path(directory) / "Kaggle"
            self.assertEqual(kaggle_cli_environment(config_dir)["KAGGLE_CONFIG_DIR"], str(config_dir))
            with patch("dataset_cleaner.kaggle.kaggle_cli_executable", return_value="kaggle"), patch(
                "dataset_cleaner.kaggle.prepare_kaggle_config_dir", return_value=config_dir
            ), patch(
                "dataset_cleaner.kaggle.subprocess.run",
                return_value=Mock(returncode=0, stdout="Kaggle CLI 2.2.5", stderr=""),
            ) as run:
                _run_kaggle(["--version"])
            self.assertEqual(run.call_args.kwargs["env"]["KAGGLE_CONFIG_DIR"], str(config_dir))

    def test_kaggle_config_uses_temp_fallback_if_preferred_path_is_not_a_directory(self) -> None:
        root = Path(__file__).resolve().parent.parent / ".test-tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            blocked_path = Path(directory) / "blocked"
            blocked_path.write_text("occupied", encoding="utf-8")
            fallback = Path(directory) / "temp" / "Kaggle"
            self.assertEqual(prepare_kaggle_config_dir(blocked_path, fallback), fallback)
            self.assertTrue(fallback.is_dir())

    def test_download_uses_official_cli_without_touching_remote_source(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent / ".test-tmp") as directory:
            destination = Path(directory) / "download"
            def fake_run(args, timeout=900):
                self.assertEqual(args[:4], ["datasets", "download", "-d", "owner/sample-data"])
                self.assertIn("--unzip", args)
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "examples.jsonl").write_text('{"prompt":"p"}\n', encoding="utf-8")
                return KaggleCommandResult("", "")
            with patch("dataset_cleaner.kaggle._run_kaggle", side_effect=fake_run):
                result = download_dataset("owner/sample-data", destination)
            self.assertEqual(result["dataset"], "owner/sample-data")
            self.assertEqual(len(result["files"]), 1)

    def test_private_kernel_and_downloaded_jsonl_candidate(self) -> None:
        root = Path(__file__).resolve().parent.parent / ".test-tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            output_root = Path(directory) / "results"
            seen_commands = []
            def fake_run(args, timeout=900):
                seen_commands.append(list(args))
                if args[:2] == ["kernels", "push"]:
                    job_dir = Path(args[args.index("-p") + 1])
                    metadata = json.loads((job_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
                    self.assertEqual(metadata["is_private"], "true")
                    self.assertEqual(metadata["enable_internet"], "false")
                    self.assertEqual(metadata["dataset_sources"], ["owner/input-data"])
                    self.assertEqual(metadata["model_sources"], [DEFAULT_MODEL_HANDLE])
                    source = (job_dir / "curate.py").read_text(encoding="utf-8")
                    ast.parse(source)
                    self.assertNotIn("private prompt contents", source)
                if args[:2] == ["kernels", "output"]:
                    download_dir = Path(args[args.index("-p") + 1])
                    (download_dir / "datasetcleaner-curated.jsonl").write_text(
                        '{"instruction":"add two numbers","completion":"return a + b"}\n', encoding="utf-8"
                    )
                    (download_dir / "datasetcleaner-manifest.json").write_text(
                        json.dumps({"approved": False, "responses_generated": 1}), encoding="utf-8"
                    )
                return KaggleCommandResult("ok", "")

            with patch("dataset_cleaner.kaggle._run_kaggle", side_effect=fake_run), patch(
                "dataset_cleaner.kaggle._wait_for_kernel", return_value="KernelWorkerStatus.COMPLETE"
            ):
                result = run_curation(
                    dataset_ref="owner/input-data",
                    kaggle_username="test-user",
                    prompt_field="instruction",
                    instruction="Answer the coding prompt.",
                    output_dir=output_root,
                )
            candidate = Path(result["local_candidate_file"])
            self.assertTrue(candidate.is_file())
            self.assertEqual(result["candidate_records"], 1)
            self.assertFalse(result["approved"])
            self.assertTrue(any(command[:2] == ["kernels", "push"] for command in seen_commands))

    def test_local_upload_requires_opt_in_and_keeps_source_bytes(self) -> None:
        root = Path(__file__).resolve().parent.parent / ".test-tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            source = Path(directory) / "local.jsonl"
            original = b'{"instruction":"private prompt contents"}\n'
            source.write_bytes(original)
            with patch("dataset_cleaner.kaggle._run_kaggle") as cli:
                with self.assertRaisesRegex(ValueError, "upload_local=True"):
                    run_curation(local_input=source, kaggle_username="user", output_dir=Path(directory) / "out")
            cli.assert_not_called()
            with patch("dataset_cleaner.kaggle._private_dataset_upload", return_value="user/private-copy") as upload, patch(
                "dataset_cleaner.kaggle._run_kaggle", side_effect=lambda *_args, **_kwargs: KaggleCommandResult("", "")
            ), patch("dataset_cleaner.kaggle._wait_for_kernel", side_effect=KaggleError("stop before output")):
                with self.assertRaisesRegex(KaggleError, "stop before output"):
                    run_curation(
                        local_input=source,
                        upload_local=True,
                        kaggle_username="user",
                        output_dir=Path(directory) / "remote",
                    )
            upload.assert_called_once()
            self.assertEqual(source.read_bytes(), original)

    def test_generated_kernel_script_is_valid_and_uses_row_data_as_input(self) -> None:
        source = _kaggle_kernel_source(
            "owner/data", DEFAULT_MODEL_HANDLE, "instruction", "completion", "answer",
            "Respond to prompts", 50, 512, None,
        )
        ast.parse(source)
        self.assertIn("model.generate", source)
        self.assertIn("datasetcleaner-curated.jsonl", source)
        self.assertIn("source_files_modified", source)
        self.assertNotIn("builtins.eval", source)
        self.assertNotIn("exec(", source)

    def test_save_kaggle_credentials_json_and_token(self) -> None:
        root = Path(__file__).resolve().parent.parent / ".test-tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            target = Path(directory) / "config"
            # Test username and key
            save_kaggle_credentials(username="user1", key="key123", target_dir=target)
            self.assertTrue((target / "kaggle.json").is_file())
            saved = json.loads((target / "kaggle.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, {"username": "user1", "key": "key123"})

            # Test status detection
            status = get_kaggle_auth_status(config_dir=target, probe_cli=False)
            self.assertTrue(status["has_credentials"])
            self.assertEqual(status["username"], "user1")
            self.assertEqual(status["credential_type"], "kaggle.json")

            # Test token save
            target_token = Path(directory) / "config_token"
            save_kaggle_credentials(token="token_abc_123", target_dir=target_token)
            self.assertTrue((target_token / "access_token").is_file())
            self.assertEqual((target_token / "access_token").read_text(encoding="utf-8"), "token_abc_123")

            status_token = get_kaggle_auth_status(config_dir=target_token, probe_cli=False)
            self.assertTrue(status_token["has_credentials"])
            self.assertEqual(status_token["credential_type"], "access_token")

            # Test JSON content string
            target_raw_json = Path(directory) / "config_raw"
            save_kaggle_credentials(json_content='{"username":"user2","key":"key456"}', target_dir=target_raw_json)
            self.assertTrue((target_raw_json / "kaggle.json").is_file())
            saved2 = json.loads((target_raw_json / "kaggle.json").read_text(encoding="utf-8"))
            self.assertEqual(saved2["username"], "user2")

    def test_push_batch_notebook_generates_metadata_and_pushes(self) -> None:
        root = Path(__file__).resolve().parent.parent / ".test-tmp"
        root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            out_dir = Path(directory) / "output"
            dummy_nb = Path(directory) / "test.ipynb"
            dummy_nb.write_text('{"cells":[],"nbformat":4}', encoding="utf-8")

            with patch("dataset_cleaner.kaggle._run_kaggle", return_value=KaggleCommandResult("Kernel pushed successfully", "")) as run:
                res = push_batch_notebook(
                    kaggle_username="my_user",
                    kernel_slug="test-batch-run",
                    notebook_path=dummy_nb,
                    output_dir=out_dir,
                    wait=False,
                )

            self.assertEqual(res["kernel_ref"], "my_user/test-batch-run")
            self.assertEqual(res["url"], "https://www.kaggle.com/code/my_user/test-batch-run")
            self.assertEqual(res["status"], "pushed")

            meta_file = Path(res["job_dir"]) / "kernel-metadata.json"
            self.assertTrue(meta_file.is_file())
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            self.assertEqual(meta["machine_shape"], "NvidiaTeslaT4x2")
            self.assertEqual(meta["kernel_type"], "notebook")
            self.assertEqual(meta["is_private"], "true")
            run.assert_called_once()

    def test_push_batch_notebook_language_presets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dummy_nb = root / "input.ipynb"
            nb_content = {
                "cells": [
                    {"cell_type": "markdown", "source": ["# Title\n"]},
                    {"cell_type": "code", "source": [
                        'LANGUAGE = "python"\n',
                        'OUTPUT_FORMAT = "bug_patch"\n',
                        'CLEANED_FOLDER_NAME = None\n',
                    ]},
                ]
            }
            dummy_nb.write_text(json.dumps(nb_content), encoding="utf-8")
            out_dir = root / "out"

            with patch("dataset_cleaner.kaggle._run_kaggle", return_value=KaggleCommandResult("OK", "")):
                res = push_batch_notebook(
                    kaggle_username="cobol_user",
                    kernel_slug="cobol-run",
                    notebook_path=dummy_nb,
                    language="cobol",
                    output_dir=out_dir,
                    wait=False,
                )

            dest_nb = Path(res["job_dir"]) / "kaggle_batch_inference.ipynb"
            self.assertTrue(dest_nb.is_file())
            written_nb = json.loads(dest_nb.read_text(encoding="utf-8"))
            cell1_src = "".join(written_nb["cells"][1]["source"])
            self.assertIn('LANGUAGE = "cobol"', cell1_src)
            self.assertIn('CLEANED_FOLDER_NAME = "DataSetCleaner-COBOL-Cleaned"', cell1_src)
            self.assertIn('OUTPUT_FORMAT = "code_task"', cell1_src)

    def test_download_kernel_outputs_sha256_verification(self) -> None:
        import hashlib
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dest = root / "downloads"
            dest.mkdir()

            def fake_download(*args, **kwargs):
                # simulate downloaded files
                (dest / "data.jsonl").write_bytes(b'{"a": 1}\n')
                h = hashlib.sha256(b'{"a": 1}\n').hexdigest()
                (dest / "manifest.sha256").write_text(f"{h}  data.jsonl\n", encoding="utf-8")
                return KaggleCommandResult("OK", "")

            with patch("dataset_cleaner.kaggle._run_kaggle", side_effect=fake_download):
                files = download_kernel_outputs("test/run", dest)
                self.assertIn("data.jsonl", files)

            # Now simulate corrupted file
            (dest / "data.jsonl").write_bytes(b'{"a": 999}\n')
            with patch("dataset_cleaner.kaggle._run_kaggle", return_value=KaggleCommandResult("OK", "")):
                with self.assertRaises(KaggleError) as ctx:
                    download_kernel_outputs("test/run", dest)
                self.assertIn("Checksum mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

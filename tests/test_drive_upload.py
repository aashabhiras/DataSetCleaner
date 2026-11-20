from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset_cleaner.drive_upload import _build_drive_service, upload_file, validate_source
import json


class FakeRequest:
    def __init__(self, response: dict[str, str]):
        self.response = response
        self.calls = 0

    def next_chunk(self):
        self.calls += 1
        return None, self.response


class FakeFiles:
    def create(self, **kwargs):
        self.kwargs = kwargs
        self.request = FakeRequest({"id": "drive-file-id", "name": kwargs["body"]["name"], "size": "23", "webViewLink": "https://drive.google.com/file/d/drive-file-id"})
        return self.request


class FakeDrive:
    def __init__(self):
        self.resource = FakeFiles()

    def files(self):
        return self.resource


class DriveUploadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parent.parent / ".test-tmp"
        self.root.mkdir(exist_ok=True)

    def test_upload_streams_to_selected_folder_and_preserves_local_bytes(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            source = Path(temporary) / "examples.jsonl"
            original = b'{"instruction":"add","completion":"return 2"}\n'
            source.write_bytes(original)
            drive = FakeDrive()
            media_args = {}

            def media_factory(path, **kwargs):
                media_args.update(path=path, **kwargs)
                return "resumable-media"

            result = upload_file(source, "drive-folder-123", drive_service=drive, media_upload_factory=media_factory)
            self.assertEqual(result["id"], "drive-file-id")
            self.assertTrue(result["local_source_unchanged"])
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(drive.resource.kwargs["body"]["parents"], ["drive-folder-123"])
            self.assertTrue(drive.resource.kwargs["media_body"] == "resumable-media")
            self.assertTrue(media_args["resumable"])

    def test_input_format_and_destination_are_validated_before_upload(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            invalid = Path(temporary) / "dataset.zip"
            invalid.write_bytes(b"data")
            with self.assertRaisesRegex(ValueError, "Supported input formats"):
                validate_source(invalid)
            source = Path(temporary) / "data.csv"
            source.write_text("content\nprint(1)\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "folder ID"):
                upload_file(source, "bad", drive_service=FakeDrive(), media_upload_factory=lambda *_a, **_k: None)

    def test_build_drive_service_handles_service_account_structure(self) -> None:
        sa_json = json.dumps({
            "type": "service_account",
            "project_id": "test-proj",
            "client_email": "service-account@example.com",
            "private_key": "invalid-key-for-test",
            "token_uri": "https://oauth2.googleapis.com/token",
        })
        with self.assertRaises(RuntimeError) as ctx:
            _build_drive_service(sa_json)
        msg = str(ctx.exception)
        if "Google Drive upload support is optional" in msg:
            self.assertIn("datasetcleaner[google-drive]", msg)
        else:
            self.assertIn("Could not authorize Google Drive", msg)
            self.assertNotIn("OAuth credentials are expired", msg)


if __name__ == "__main__":
    unittest.main()

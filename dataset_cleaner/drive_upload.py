"""Stream a local dataset file to a user-selected Google Drive folder."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

SUPPORTED_INPUTS = (".csv", ".jsonl", ".ndjson", ".parquet", ".jsonl.gz", ".ndjson.gz")
DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
CHUNK_SIZE = 8 * 1024 * 1024


def validate_source(path: str | Path) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("Input must be an existing local dataset file.")
    if not source.name.casefold().endswith(SUPPORTED_INPUTS):
        raise ValueError("Supported input formats are CSV, JSONL/NDJSON, Parquet, and gzip-compressed JSONL/NDJSON.")
    return source


def _build_drive_service(token_json: str) -> Any:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError("Google Drive upload support is optional. Install `datasetcleaner[google-drive]`.") from exc
    try:
        info = json.loads(token_json)
        if isinstance(info, dict) and info.get("type") == "service_account":
            credentials = service_account.Credentials.from_service_account_info(info, scopes=[DRIVE_SCOPE])
        else:
            credentials = Credentials.from_authorized_user_info(info, scopes=[DRIVE_SCOPE])
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            if not credentials.valid:
                raise ValueError("OAuth credentials are expired or have no refresh token.")
        return build("drive", "v3", credentials=credentials, cache_discovery=False)
    except Exception as exc:
        raise RuntimeError(f"Could not authorize Google Drive ({type(exc).__name__}); check the OAuth token and Drive API access.") from None


def upload_file(
    path: str | Path,
    folder_id: str,
    *,
    name: str | None = None,
    token_json: str | None = None,
    drive_service: Any = None,
    media_upload_factory: Any = None,
) -> dict[str, Any]:
    """Upload a file resumably; the local source is opened read-only and never rewritten."""
    source = validate_source(path)
    if not isinstance(folder_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{10,}", folder_id.strip()):
        raise ValueError("Provide a valid Google Drive destination folder ID.")
    if drive_service is None:
        token_json = token_json or os.environ.get("GOOGLE_DRIVE_TOKEN_JSON")
        if not token_json:
            raise ValueError("Set GOOGLE_DRIVE_TOKEN_JSON to your authorized-user OAuth token JSON.")
        drive_service = _build_drive_service(token_json)
    if media_upload_factory is None:
        try:
            from googleapiclient.http import MediaFileUpload
        except ImportError as exc:
            raise RuntimeError("Google Drive upload support is optional. Install `datasetcleaner[google-drive]`.") from exc
        media_upload_factory = MediaFileUpload

    display_name = name.strip() if isinstance(name, str) and name.strip() else source.name
    media = media_upload_factory(str(source), mimetype="application/octet-stream", resumable=True, chunksize=CHUNK_SIZE)
    try:
        request = drive_service.files().create(
            body={"name": display_name, "parents": [folder_id.strip()]},
            media_body=media,
            fields="id,name,size,webViewLink",
            supportsAllDrives=True,
        )
        response = None
        while response is None:
            _, response = request.next_chunk()
    except Exception as exc:
        raise RuntimeError(f"Google Drive upload failed ({type(exc).__name__}); the local source remains unchanged.") from None
    return {
        "id": response["id"],
        "name": response.get("name", display_name),
        "size": int(response.get("size", source.stat().st_size)),
        "web_view_link": response.get("webViewLink"),
        "local_source_unchanged": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datasetcleaner-drive-upload", description="Stream one local dataset file into Google Drive using resumable upload.")
    parser.add_argument("input", type=Path, help="Local CSV, JSONL/NDJSON, or Parquet file")
    parser.add_argument("--folder-id", required=True, help="Destination Google Drive folder ID")
    parser.add_argument("--name", help="Optional Drive filename (defaults to the input filename)")
    parser.add_argument("--confirm-upload", action="store_true", help="Confirm copying this file to Google Drive")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.confirm_upload:
        parser.error("upload requires --confirm-upload because it transfers the local dataset to Google Drive")
    try:
        result = upload_file(args.input, args.folder_id, name=args.name)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

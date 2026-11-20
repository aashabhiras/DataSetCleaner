"""Opt-in Kaggle dataset and hosted-model workflows.

This module delegates authentication to Kaggle's official CLI. Local inputs are
copied to a private Kaggle Dataset only when the caller explicitly requests it.
Generated JSONL is downloaded as an unapproved candidate and never merged into
an existing source file or approved export.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

DATASET_REF_RE = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")
MODEL_HANDLE_RE = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[0-9]+$")
DEFAULT_MODEL_HANDLE = "qwen-lm/qwen2-5-coder/Transformers/14b-instruct-gptq-int4/1"
DEFAULT_ANSWER_INSTRUCTION = "Write a correct, helpful answer for this coding task. Return only the answer text."
DEFAULT_REWRITE_INSTRUCTION = "Improve the existing answer for correctness, clarity, and completeness. Return only the revised answer text."
SUPPORTED_SUFFIXES = {".csv", ".jsonl", ".ndjson", ".parquet"}
MAX_GENERATION_TOKENS = 4096
MAX_PROMPT_CHARS = 32_000
MAX_INPUT_TOKENS = 8192


class KaggleError(RuntimeError):
    """Raised for a missing Kaggle CLI, failed command, or invalid output."""


@dataclass(frozen=True)
class KaggleCommandResult:
    stdout: str
    stderr: str


def kaggle_config_dir() -> Path:
    """Return DataSetCleaner-owned Kaggle credentials/config directory."""
    configured = os.environ.get("KAGGLE_CONFIG_DIR")
    if configured:
        return Path(configured)
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "DataSetCleaner" / "Kaggle"


def prepare_kaggle_config_dir(preferred: Path | None = None, fallback: Path | None = None) -> Path:
    """Create the preferred config folder, falling back to the writable temp area."""
    preferred = Path(preferred) if preferred is not None else kaggle_config_dir()
    fallback = Path(fallback) if fallback is not None else Path(tempfile.gettempdir()) / "DataSetCleaner" / "Kaggle"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        return preferred
    except OSError as preferred_error:
        if fallback == preferred:
            raise KaggleError(f"Cannot create Kaggle config directory {preferred}: {preferred_error}") from preferred_error
        try:
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
        except OSError as fallback_error:
            raise KaggleError(
                f"Cannot create Kaggle config directory {preferred} ({preferred_error}); "
                f"temp fallback {fallback} also failed ({fallback_error})."
            ) from fallback_error


def kaggle_cli_environment(config_dir: Path | None = None) -> dict[str, str]:
    """Use the same private config directory for login and subsequent CLI calls."""
    environment = os.environ.copy()
    environment["KAGGLE_CONFIG_DIR"] = str(config_dir or kaggle_config_dir())
    return environment


def save_kaggle_credentials(
    username: str | None = None,
    key: str | None = None,
    token: str | None = None,
    json_content: str | None = None,
    target_dir: Path | None = None,
) -> Path:
    """Save Kaggle credentials (kaggle.json or access_token) into the app's config directory."""
    config_dir = prepare_kaggle_config_dir(preferred=target_dir)
    if json_content and json_content.strip():
        data = json.loads(json_content.strip())
        if not isinstance(data, dict) or "username" not in data or "key" not in data:
            raise ValueError("Kaggle JSON must contain 'username' and 'key' fields.")
        username = str(data["username"]).strip()
        key = str(data["key"]).strip()

    if username and key:
        username = username.strip()
        key = key.strip()
        if not username or not key:
            raise ValueError("Both Kaggle username and API key are required.")
        target_file = config_dir / "kaggle.json"
        target_file.write_text(json.dumps({"username": username, "key": key}, indent=2), encoding="utf-8")
        try:
            target_file.chmod(0o600)
        except OSError:
            pass
        return config_dir

    if token and token.strip():
        target_file = config_dir / "access_token"
        target_file.write_text(token.strip(), encoding="utf-8")
        try:
            target_file.chmod(0o600)
        except OSError:
            pass
        return config_dir

    raise ValueError("Provide either username and key, an API token, or valid kaggle.json content.")


def get_kaggle_auth_status(config_dir: Path | None = None, probe_cli: bool = False) -> dict[str, Any]:
    """Inspect local Kaggle credential status, optionally verifying with a quick CLI check."""
    target_dir = config_dir or kaggle_config_dir()
    fallback_dir = Path(tempfile.gettempdir()) / "DataSetCleaner" / "Kaggle"

    found_dir = None
    credential_file = None
    detected_user = None

    dot_kaggle = Path.home() / ".kaggle"
    downloads_dir = Path.home() / "Downloads"
    workspace_dir = Path.cwd()

    for check_dir in (target_dir, dot_kaggle, fallback_dir, workspace_dir, downloads_dir):
        if not check_dir.is_dir():
            continue
        oauth_file = check_dir / "credentials.json"
        json_file = check_dir / "kaggle.json"
        token_file = check_dir / "access_token"
        if oauth_file.is_file():
            found_dir = check_dir
            credential_file = "credentials.json"
            try:
                data = json.loads(oauth_file.read_text(encoding="utf-8"))
                detected_user = data.get("username")
                if check_dir != target_dir and not (target_dir / "credentials.json").is_file():
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(oauth_file, target_dir / "credentials.json")
                    found_dir = target_dir
            except Exception:
                pass
            break
        elif json_file.is_file():
            found_dir = check_dir
            credential_file = "kaggle.json"
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                detected_user = data.get("username")
                if check_dir != target_dir and not (target_dir / "kaggle.json").is_file():
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(json_file, target_dir / "kaggle.json")
                    found_dir = target_dir
            except Exception:
                pass
            break
        elif token_file.is_file():
            found_dir = check_dir
            credential_file = "access_token"
            if check_dir != target_dir and not (target_dir / "access_token").is_file():
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(token_file, target_dir / "access_token")
                found_dir = target_dir
            break

    env_user = os.environ.get("KAGGLE_USERNAME")
    env_token = os.environ.get("KAGGLE_API_TOKEN")

    has_credentials = bool(credential_file or env_token or (env_user and os.environ.get("KAGGLE_KEY")))
    active_user = detected_user or env_user

    result: dict[str, Any] = {
        "has_credentials": has_credentials,
        "username": active_user,
        "config_dir": str(found_dir or target_dir),
        "credential_type": credential_file or ("env" if env_token or env_user else None),
        "authenticated": False,
        "error": None,
    }

    if probe_cli:
        executable = kaggle_cli_executable()
        if not executable:
            result["error"] = "Kaggle CLI is not installed."
            return result
        try:
            cli_res = _run_kaggle(["datasets", "list", "--page-size", "1"], timeout=12)
            result["authenticated"] = True
            result["error"] = None
        except Exception as exc:
            result["authenticated"] = False
            msg = str(exc)
            if "Authentication required" in msg or "401" in msg or "Unauthorized" in msg:
                result["error"] = "Authentication required. Please connect or provide credentials."
            else:
                result["error"] = msg[-300:]

    return result


def kaggle_cli_executable() -> str | None:
    executable = shutil.which("kaggle")
    if executable:
        return executable
    sibling = Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle")
    return str(sibling) if sibling.is_file() else None


def _validate_dataset_ref(value: str) -> str:
    if not DATASET_REF_RE.fullmatch(value):
        raise ValueError("Kaggle dataset must use the owner/dataset-slug format.")
    return value


def _validate_model_handle(value: str) -> str:
    if not MODEL_HANDLE_RE.fullmatch(value):
        raise ValueError("Kaggle model must use owner/model/framework/variation/version format.")
    return value


def _run_kaggle(arguments: Sequence[str], timeout: int = 900) -> KaggleCommandResult:
    executable = kaggle_cli_executable()
    if not executable:
        raise KaggleError("Kaggle CLI is unavailable. Install DataSetCleaner with `pip install '.[kaggle]'` or install Kaggle CLI.")
    try:
        config_dir = prepare_kaggle_config_dir()
        result = subprocess.run(
            [executable, *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
            env=kaggle_cli_environment(config_dir),
        )
    except KaggleError:
        raise
    except subprocess.TimeoutExpired as exc:
        raise KaggleError(f"Kaggle CLI timed out after {timeout} seconds.") from exc
    except OSError as exc:
        raise KaggleError(f"Could not start Kaggle CLI ({type(exc).__name__}).") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise KaggleError(f"Kaggle CLI exited with code {result.returncode}: {detail[-1200:]}")
    return KaggleCommandResult(result.stdout, result.stderr)


def search_datasets(query: str, limit: int = 20) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip() or len(query) > 200:
        raise ValueError("Search query must contain 1 to 200 characters.")
    if not 1 <= limit <= 100:
        raise ValueError("Search limit must be between 1 and 100.")
    result = _run_kaggle(["datasets", "list", "-s", query.strip(), "--page-size", str(limit), "--format", "json"])
    try:
        entries = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise KaggleError("Kaggle CLI returned an unrecognized dataset search response.") from None
    return {"query": query.strip(), "datasets": entries}


def download_dataset(dataset_ref: str, output_dir: str | Path) -> dict[str, Any]:
    dataset_ref = _validate_dataset_ref(dataset_ref)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _run_kaggle(["datasets", "download", "-d", dataset_ref, "-p", str(destination), "--unzip"], timeout=3600)
    files = sorted(str(path) for path in destination.rglob("*") if path.is_file())
    if not files:
        raise KaggleError("Kaggle download completed but returned no files.")
    return {"dataset": dataset_ref, "output_dir": str(destination), "files": files}


def _private_dataset_upload(source: Path, owner: str, license_id: str) -> str:
    if not source.is_file() or source.suffix.casefold() not in SUPPORTED_SUFFIXES:
        raise ValueError("Local Kaggle upload accepts one CSV, JSONL/NDJSON, or Parquet file.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", owner):
        raise ValueError("Kaggle username must contain only letters, numbers, underscores, or hyphens.")
    if not isinstance(license_id, str) or not license_id.strip():
        raise ValueError("Provide a license identifier or 'unknown' for Kaggle upload metadata.")

    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "-", source.stem).strip("-").lower()[:28] or "dataset"
    dataset_slug = f"dsc-{safe_stem}-{uuid.uuid4().hex[:8]}"
    title = f"DataSetCleaner {safe_stem}"[:50]
    with tempfile.TemporaryDirectory(prefix="datasetcleaner-kaggle-upload-") as temporary:
        upload_dir = Path(temporary)
        shutil.copy2(source, upload_dir / source.name)
        metadata = {
            "title": title,
            "id": f"{owner}/{dataset_slug}",
            "licenses": [{"name": license_id.strip()}],
            "description": "Private input copy for a DataSetCleaner Kaggle curation run. Source rights remain the uploader's responsibility.",
        }
        (upload_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        # Kaggle datasets are private by default. Never pass --public here.
        _run_kaggle(["datasets", "create", "-p", str(upload_dir), "--dir-mode", "skip"], timeout=7200)
    return f"{owner}/{dataset_slug}"


def _kaggle_kernel_source(dataset_ref: str, model_handle: str, prompt_field: str | None,
                          response_field: str, mode: str, instruction: str,
                          max_records: int, max_new_tokens: int, input_file: str | None) -> str:
    constants = {
        "DATASET_REF": dataset_ref,
        "MODEL_HANDLE": model_handle,
        "PROMPT_FIELD": prompt_field,
        "RESPONSE_FIELD": response_field,
        "MODE": mode,
        "INSTRUCTION": instruction,
        "MAX_RECORDS": max_records,
        "MAX_NEW_TOKENS": max_new_tokens,
        "INPUT_FILE": input_file,
    }
    header = "\n".join(f"{key} = {value!r}" for key, value in constants.items())
    return header + r'''

import csv
import json
import os
import sys
import time
from pathlib import Path

import kagglehub
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_KEYS = ("instruction", "prompt", "question", "task", "user", "input", "query")
INPUT_ROOT = Path("/kaggle/input") / DATASET_REF.split("/", 1)[1]
OUTPUT_ROOT = Path("/kaggle/working")
OUTPUT_FILE = OUTPUT_ROOT / "datasetcleaner-curated.jsonl"
REJECT_FILE = OUTPUT_ROOT / "datasetcleaner-rejected.jsonl"

def iter_jsonl(path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("record is not a JSON object")
                yield item, line_number, None
            except (json.JSONDecodeError, ValueError) as error:
                yield None, line_number, str(error)[:300]

def iter_csv(path):
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 2):
            yield dict(row), row_number, None

def iter_parquet(path):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    row_number = 0
    for batch in parquet.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            row_number += 1
            yield row, row_number, None

def find_inputs():
    if not INPUT_ROOT.is_dir():
        raise RuntimeError("Attached Kaggle dataset path was not found: " + str(INPUT_ROOT))
    paths = sorted(path for path in INPUT_ROOT.rglob("*") if path.is_file() and path.suffix.casefold() in (".jsonl", ".ndjson", ".csv", ".parquet"))
    if INPUT_FILE:
        paths = [path for path in paths if path.name == INPUT_FILE]
    if not paths:
        raise RuntimeError("No supported JSONL, CSV, or Parquet input file was attached.")
    return paths

def rows_for(path):
    suffix = path.suffix.casefold()
    if suffix in (".jsonl", ".ndjson"):
        return iter_jsonl(path)
    if suffix == ".csv":
        return iter_csv(path)
    if suffix == ".parquet":
        return iter_parquet(path)
    raise RuntimeError("Unsupported input file: " + path.name)

def main():
    if not isinstance(RESPONSE_FIELD, str) or not RESPONSE_FIELD.strip():
        raise RuntimeError("response_field is required")
    model_path = kagglehub.model_download(MODEL_HANDLE)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", trust_remote_code=False)
    model.eval()
    generated = skipped = rejected = seen = 0
    input_paths = find_inputs()
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as output, REJECT_FILE.open("w", encoding="utf-8", newline="\n") as rejects:
        for path in input_paths:
            for row, row_number, error in rows_for(path):
                if MAX_RECORDS and seen >= MAX_RECORDS:
                    break
                seen += 1
                if error:
                    rejects.write(json.dumps({"source_file": path.name, "source_row": row_number, "reason": error}, ensure_ascii=False) + "\n")
                    rejected += 1
                    continue
                prompt_key = PROMPT_FIELD if PROMPT_FIELD in row else next((key for key in PROMPT_KEYS if key in row), None)
                prompt = row.get(prompt_key) if prompt_key else None
                if not isinstance(prompt, str) or not prompt.strip():
                    rejects.write(json.dumps({"source_file": path.name, "source_row": row_number, "reason": "prompt field is missing or empty"}, ensure_ascii=False) + "\n")
                    rejected += 1
                    continue
                if len(prompt) > MAX_PROMPT_CHARS:
                    rejects.write(json.dumps({"source_file": path.name, "source_row": row_number, "reason": "prompt exceeds the 32,000-character limit"}, ensure_ascii=False) + "\n")
                    rejected += 1
                    continue
                old_response = row.get(RESPONSE_FIELD)
                response_present = old_response is not None and (not isinstance(old_response, str) or bool(old_response.strip()))
                if MODE == "answer" and response_present:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    skipped += 1
                    continue
                if MODE == "rewrite" and response_present and not isinstance(old_response, str):
                    rejects.write(json.dumps({"source_file": path.name, "source_row": row_number, "reason": "existing response is not text and was not overwritten"}, ensure_ascii=False) + "\n")
                    rejected += 1
                    continue
                user_text = prompt.strip()
                if MODE == "rewrite" and isinstance(old_response, str) and old_response.strip():
                    user_text += "\n\nExisting answer to improve:\n" + old_response.strip()
                user_text = INSTRUCTION + "\n\n" + user_text
                messages = [{"role": "user", "content": user_text}]
                if getattr(tokenizer, "chat_template", None):
                    model_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    model_input = user_text + "\n\nResponse:"
                encoded = tokenizer(model_input, return_tensors="pt")
                if encoded["input_ids"].shape[1] > MAX_INPUT_TOKENS:
                    rejects.write(json.dumps({"source_file": path.name, "source_row": row_number, "reason": "prompt exceeds the 8,192-token context limit"}, ensure_ascii=False) + "\n")
                    rejected += 1
                    continue
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                with torch.inference_mode():
                    tokens = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                response = tokenizer.decode(tokens[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                if not response:
                    rejects.write(json.dumps({"source_file": path.name, "source_row": row_number, "reason": "model returned an empty response"}, ensure_ascii=False) + "\n")
                    rejected += 1
                    continue
                candidate = dict(row)
                candidate[RESPONSE_FIELD] = response
                output.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                generated += 1
                if generated % 25 == 0:
                    print("generated", generated, flush=True)
    manifest = {
        "provider": "Kaggle Notebooks",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": DATASET_REF,
        "input_files": [path.name for path in input_paths],
        "model": MODEL_HANDLE,
        "model_quantization": "GPTQ Int4" if "gptq-int4" in MODEL_HANDLE.casefold() else "Review model card for quantization",
        "mode": MODE,
        "prompt_field": PROMPT_FIELD or "auto-detected",
        "response_field": RESPONSE_FIELD,
        "records_seen": seen,
        "responses_generated": generated,
        "records_skipped_with_existing_response": skipped,
        "records_rejected": rejected,
        "approved": False,
        "source_files_modified": False,
        "rights_review_required": True,
    }
    (OUTPUT_ROOT / "datasetcleaner-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)

if __name__ == "__main__":
    main()
'''


def _wait_for_kernel(kernel_ref: str, timeout: int, poll_interval: float = 15.0) -> str:
    deadline = time.monotonic() + timeout
    last_status = "unknown"
    while time.monotonic() < deadline:
        status_result = _run_kaggle(["kernels", "status", kernel_ref], timeout=120)
        text = (status_result.stdout + "\n" + status_result.stderr).strip()
        normalized = text.casefold()
        if "complete" in normalized or "succeeded" in normalized:
            return text
        if any(term in normalized for term in ("error", "failed", "cancelled")):
            raise KaggleError(f"Kaggle kernel did not complete successfully: {text[-1500:]}")
        last_status = text
        time.sleep(min(poll_interval, max(0, deadline - time.monotonic())))
    raise KaggleError(f"Timed out waiting for Kaggle kernel. Last status: {last_status[-500:]}")


def run_curation(
    *,
    dataset_ref: str | None = None,
    local_input: str | Path | None = None,
    upload_local: bool = False,
    kaggle_username: str,
    upload_license: str = "unknown",
    model_handle: str = DEFAULT_MODEL_HANDLE,
    prompt_field: str | None = None,
    response_field: str = "completion",
    mode: str = "answer",
    instruction: str = "Write a correct, helpful answer for this coding task. Return only the answer text.",
    input_file: str | None = None,
    max_records: int = 1000,
    max_new_tokens: int = 1024,
    accelerator: str = "NvidiaTeslaT4",
    output_dir: str | Path = "datasetcleaner-kaggle-output",
    wait_timeout: int = 7200,
) -> dict[str, Any]:
    """Run private Kaggle Notebook inference and download an unapproved JSONL candidate."""
    if (dataset_ref is None) == (local_input is None):
        raise ValueError("Provide exactly one of dataset_ref or local_input.")
    if local_input is not None and not upload_local:
        raise ValueError("Uploading a local source requires upload_local=True; the file will be copied to a private Kaggle Dataset.")
    if dataset_ref is not None:
        dataset_ref = _validate_dataset_ref(dataset_ref)
    model_handle = _validate_model_handle(model_handle)
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", kaggle_username):
        raise ValueError("Kaggle username must contain only letters, numbers, underscores, or hyphens.")
    if mode not in {"answer", "rewrite"}:
        raise ValueError("mode must be 'answer' or 'rewrite'.")
    if not isinstance(response_field, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,99}", response_field):
        raise ValueError("response_field must be a JSON field name.")
    if prompt_field is not None and (not isinstance(prompt_field, str) or not prompt_field.strip() or len(prompt_field) > 100):
        raise ValueError("prompt_field must be omitted or contain at most 100 characters.")
    if prompt_field and prompt_field == response_field:
        raise ValueError("prompt_field and response_field must be different so the generated response cannot replace the prompt.")
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 4000:
        raise ValueError("instruction must contain 1 to 4000 characters.")
    if mode == "rewrite" and instruction == DEFAULT_ANSWER_INSTRUCTION:
        instruction = DEFAULT_REWRITE_INSTRUCTION
    if not isinstance(max_records, int) or isinstance(max_records, bool) or not 0 <= max_records <= 100_000:
        raise ValueError("max_records must be between 0 (all rows) and 100,000.")
    if not 32 <= max_new_tokens <= MAX_GENERATION_TOKENS:
        raise ValueError(f"max_new_tokens must be between 32 and {MAX_GENERATION_TOKENS}.")
    if not isinstance(wait_timeout, int) or not 60 <= wait_timeout <= 24 * 60 * 60:
        raise ValueError("wait_timeout must be between 60 seconds and 24 hours.")
    if input_file is not None and (Path(input_file).name != input_file or len(input_file) > 255):
        raise ValueError("input_file must be a filename, not a path.")
    if not isinstance(accelerator, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", accelerator):
        raise ValueError("accelerator must be a Kaggle accelerator identifier.")

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if local_input is not None:
        dataset_ref = _private_dataset_upload(Path(local_input).expanduser().resolve(), kaggle_username, upload_license)
    kernel_slug = "datasetcleaner-curate-" + uuid.uuid4().hex[:10]
    job_dir = root / "jobs" / kernel_slug
    job_dir.mkdir(parents=True, exist_ok=False)
    kernel_ref = f"{kaggle_username}/{kernel_slug}"
    script = _kaggle_kernel_source(dataset_ref, model_handle, prompt_field, response_field, mode,
                                   instruction.strip(), max_records, max_new_tokens, input_file)
    (job_dir / "curate.py").write_text(script, encoding="utf-8")
    metadata = {
        "id": kernel_ref,
        "title": f"DataSetCleaner Curation {kernel_slug[-10:]}",
        "code_file": "curate.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "false",
        "machine_shape": accelerator,
        "dataset_sources": [dataset_ref],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [model_handle],
    }
    (job_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _run_kaggle(["kernels", "push", "-p", str(job_dir), "--accelerator", accelerator, "--timeout", str(wait_timeout)], timeout=900)
    remote_status = _wait_for_kernel(kernel_ref, wait_timeout)
    download_dir = root / "downloads" / kernel_slug
    download_dir.mkdir(parents=True, exist_ok=False)
    _run_kaggle(["kernels", "output", kernel_ref, "-p", str(download_dir), "--file-pattern", r"datasetcleaner-(curated|rejected|manifest)\.(jsonl|json)$"], timeout=3600)
    candidate_path = download_dir / "datasetcleaner-curated.jsonl"
    manifest_path = download_dir / "datasetcleaner-manifest.json"
    rejected_path = download_dir / "datasetcleaner-rejected.jsonl"
    if not candidate_path.is_file() or not manifest_path.is_file():
        raise KaggleError("Kaggle run completed but its required JSONL candidate or manifest was not returned.")
    count = 0
    with candidate_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise KaggleError(f"Kaggle returned malformed candidate JSONL at line {line_number}: {exc.msg}.") from None
            if not isinstance(row, dict):
                raise KaggleError(f"Kaggle returned a non-object candidate at line {line_number}.")
            count += 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({"kernel": kernel_ref, "remote_status": remote_status, "local_candidate_file": str(candidate_path),
                     "local_manifest_file": str(manifest_path), "candidate_records": count,
                     "rejected_file": str(rejected_path) if rejected_path.exists() else None,
                     "approved": False, "source_files_modified": False})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def push_batch_notebook(
    kaggle_username: str,
    kernel_slug: str = "datasetcleaner-batch-inference",
    title: str = "DataSetCleaner Batch Inference Dual T4",
    notebook_path: Path | str | None = None,
    dataset_sources: list[str] | None = None,
    output_dir: str | Path = "datasetcleaner-kaggle-output",
    input_source: str = "huggingface",
    language: str | None = None,
    output_format: str | None = None,
    dry_run: bool | None = None,
    start_record: int | None = None,
    stop_after_record: int | None = None,
    stop_after_seconds: int | None = None,
    wait: bool = False,
    wait_timeout: int = 12 * 3600,
) -> dict[str, Any]:
    """Automate pushing and launching kaggle_batch_inference.ipynb on Kaggle with dual-T4 GPUs."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,50}", kaggle_username):
        raise ValueError("Kaggle username must contain only letters, numbers, underscores, or hyphens.")

    nb_path = Path(notebook_path) if notebook_path else Path(__file__).resolve().parent.parent / "kaggle_batch_inference.ipynb"
    if not nb_path.is_file():
        raise KaggleError(f"Notebook file not found: {nb_path}")

    root = Path(output_dir).expanduser().resolve()
    job_dir = root / "jobs" / kernel_slug
    job_dir.mkdir(parents=True, exist_ok=True)

    dest_nb = job_dir / "kaggle_batch_inference.ipynb"
    token_files = [Path("token.json")] if Path("token.json").is_file() else (list(Path.cwd().glob("fabled-variety-*.json")) + list(Path.cwd().glob("*serviceaccount*.json")))
    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))

    if len(nb_data.get("cells", [])) > 1:
        c1_src = "".join(nb_data["cells"][1]["source"])
        if input_source:
            c1_src = re.sub(r'^(INPUT_SOURCE\s*=\s*)["\'][^"\']+["\']', f'\\g<1>"{input_source}"', c1_src, flags=re.MULTILINE)
        if language:
            lang_clean = language.lower().strip()
            c1_src = re.sub(r'^(LANGUAGE\s*=\s*)["\'][^"\']+["\']', f'\\g<1>"{lang_clean}"', c1_src, flags=re.MULTILINE)
            # Language preset mappings
            if lang_clean == "cobol":
                folder_name = "DataSetCleaner-COBOL-Cleaned"
                if not output_format:
                    output_format = "code_task"
            elif lang_clean == "sql":
                folder_name = "DataSetCleaner-SQL-Cleaned"
                if not output_format:
                    output_format = "code_task"
            elif lang_clean == "cuda":
                folder_name = "DataSetCleaner-CUDA-Cleaned"
                if not output_format:
                    output_format = "code_task"
            else:
                folder_name = f"DataSetCleaner-{lang_clean.capitalize()}-Cleaned"
            c1_src = re.sub(r'^(CLEANED_FOLDER_NAME\s*=\s*).*', f'\\g<1>"{folder_name}"', c1_src, flags=re.MULTILINE)
        if output_format:
            c1_src = re.sub(r'^(OUTPUT_FORMAT\s*=\s*)["\'][^"\']+["\']', f'\\g<1>"{output_format}"', c1_src, flags=re.MULTILINE)
        if dry_run is not None:
            c1_src = re.sub(r'^(DRY_RUN\s*=\s*)(?:True|False)', f'\\g<1>{dry_run}', c1_src, flags=re.MULTILINE)
        # Ensure MODEL_REPO uses the official AWQ checkpoint for vLLM continuous batching
        c1_src = re.sub(r'^(MODEL_REPO\s*=\s*)["\'][^"\']+["\']', r'\g<1>"Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"', c1_src, flags=re.MULTILINE)
        c1_src = re.sub(r'^(MICRO_BATCH\s*=\s*)\d+', r'\g<1>32', c1_src, flags=re.MULTILINE)
        # Ensure DRIVE_OUTPUT_FOLDER_ID is set if Kaggle folder is known
        c1_src = re.sub(r'^(DRIVE_OUTPUT_FOLDER_ID\s*=\s*)["\'][^"\']*["\']', r'\g<1>"1GY6W-IEvaluhBybKLzqnbh5qzidHl7rq"', c1_src, flags=re.MULTILINE)
        if start_record is not None:
            if re.search(r'^(START_RECORD\s*=\s*)\d+', c1_src, flags=re.MULTILINE):
                c1_src = re.sub(r'^(START_RECORD\s*=\s*)\d+', f'\\g<1>{start_record}', c1_src, flags=re.MULTILINE)
            else:
                c1_src += f"\nSTART_RECORD = {start_record}\n"
        if stop_after_record is not None:
            if re.search(r'^(STOP_AFTER_RECORD\s*=\s*)\d+', c1_src, flags=re.MULTILINE):
                c1_src = re.sub(r'^(STOP_AFTER_RECORD\s*=\s*)\d+', f'\\g<1>{stop_after_record}', c1_src, flags=re.MULTILINE)
            else:
                c1_src += f"\nSTOP_AFTER_RECORD = {stop_after_record}\n"
        if stop_after_seconds is not None:
            if re.search(r'^(STOP_AFTER_SECONDS\s*=\s*).*', c1_src, flags=re.MULTILINE):
                c1_src = re.sub(r'^(STOP_AFTER_SECONDS\s*=\s*).*', f'\\g<1>{stop_after_seconds}', c1_src, flags=re.MULTILINE)
            else:
                c1_src += f"\nSTOP_AFTER_SECONDS = {stop_after_seconds}\n"
        nb_data["cells"][1]["source"] = [line + "\n" for line in c1_src.split("\n")]
        if nb_data["cells"][1]["source"] and nb_data["cells"][1]["source"][-1] == "\n":
            nb_data["cells"][1]["source"][-1] = ""

    if token_files and token_files[0].is_file() and len(nb_data.get("cells", [])) > 4:
        token_content = token_files[0].read_text(encoding="utf-8").strip()
        c4_src = "".join(nb_data["cells"][4]["source"])
        if not re.search(r'^\s*EMBEDDED_DRIVE_TOKEN_JSON\s*=', c4_src, flags=re.MULTILINE):
            inject = f"EMBEDDED_DRIVE_TOKEN_JSON = {json.dumps(token_content)}\n\n"
            c4_src = inject + c4_src
            nb_data["cells"][4]["source"] = [line + "\n" for line in c4_src.split("\n")]
            if nb_data["cells"][4]["source"] and nb_data["cells"][4]["source"][-1] == "\n":
                nb_data["cells"][4]["source"][-1] = ""

    hf_candidates = [
        Path("hf_token.txt"),
        Path("hf_token"),
        Path.home() / ".cache" / "huggingface" / "token",
        Path.home() / ".huggingface" / "token",
    ]
    hf_val = os.environ.get("HF_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")
    if not hf_val:
        for p in hf_candidates:
            if p.is_file():
                try:
                    txt = p.read_text(encoding="utf-8").strip()
                    if txt.startswith("hf_"):
                        hf_val = txt
                        break
                except Exception:
                    pass
    if hf_val and len(nb_data.get("cells", [])) > 4:
        c4_src = "".join(nb_data["cells"][4]["source"])
        if not re.search(r'^\s*EMBEDDED_HF_TOKEN\s*=', c4_src, flags=re.MULTILINE):
            inject = f"EMBEDDED_HF_TOKEN = {json.dumps(hf_val.strip())}\n\n"
            c4_src = inject + c4_src
            nb_data["cells"][4]["source"] = [line + "\n" for line in c4_src.split("\n")]
            if nb_data["cells"][4]["source"] and nb_data["cells"][4]["source"][-1] == "\n":
                nb_data["cells"][4]["source"][-1] = ""
    dest_nb.write_text(json.dumps(nb_data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    kernel_ref = f"{kaggle_username}/{kernel_slug}"
    metadata = {
        "id": kernel_ref,
        "title": title,
        "code_file": "kaggle_batch_inference.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4x2",
        "dataset_sources": dataset_sources or [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (job_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    push_res = _run_kaggle(["kernels", "push", "-p", str(job_dir)], timeout=300)

    url = f"https://www.kaggle.com/code/{kernel_ref}"
    result: dict[str, Any] = {
        "kernel_ref": kernel_ref,
        "url": url,
        "status": "pushed",
        "output": push_res.stdout.strip(),
        "job_dir": str(job_dir),
    }

    if wait:
        status_text = _wait_for_kernel(kernel_ref, wait_timeout)
        download_dir = root / "downloads" / kernel_slug
        download_dir.mkdir(parents=True, exist_ok=True)
        downloaded_files = download_kernel_outputs(kernel_ref, download_dir)
        result["status"] = "complete"
        result["download_dir"] = str(download_dir)
        result["remote_status"] = status_text
        result["downloaded_files"] = downloaded_files

    return result


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_kernel_status(kernel_ref: str) -> str:
    """Return status string for a running or completed Kaggle kernel."""
    result = _run_kaggle(["kernels", "status", kernel_ref], timeout=60)
    return (result.stdout + "\n" + result.stderr).strip()


def download_kernel_outputs(kernel_ref: str, output_dir: str | Path) -> list[str]:
    """Download output artifacts from a completed Kaggle kernel and verify integrity against manifest.sha256."""
    dest = Path(output_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    _run_kaggle(["kernels", "output", kernel_ref, "-p", str(dest)], timeout=3600)

    manifest_sha_path = dest / "manifest.sha256"
    manifest_json_path = dest / "manifest.json"

    if manifest_sha_path.is_file():
        content = manifest_sha_path.read_text(encoding="utf-8").strip()
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        for line in lines:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                expected_hash = parts[0].strip().lower()
                rel_fname = parts[1].lstrip("*").strip()
                target_file = dest / rel_fname
                if not target_file.is_file():
                    matches = list(dest.rglob(rel_fname))
                    if matches:
                        target_file = matches[0]
                    else:
                        raise KaggleError(f"File listed in manifest.sha256 not found: {rel_fname}")
                actual_hash = _compute_sha256(target_file)
                if actual_hash != expected_hash:
                    raise KaggleError(
                        f"Checksum mismatch for {rel_fname}: expected {expected_hash}, got {actual_hash}"
                    )
            elif len(parts) == 1 and manifest_json_path.is_file():
                expected_hash = parts[0].strip().lower()
                actual_hash = _compute_sha256(manifest_json_path)
                if actual_hash != expected_hash:
                    raise KaggleError(
                        f"Checksum mismatch for manifest.json: expected {expected_hash}, got {actual_hash}"
                    )

    if manifest_json_path.is_file():
        try:
            m_data = json.loads(manifest_json_path.read_text(encoding="utf-8"))
            for f_entry in m_data.get("files", []):
                rel_p = f_entry.get("path")
                exp_sha = f_entry.get("sha256")
                if rel_p and exp_sha:
                    t_file = dest / rel_p
                    if not t_file.is_file():
                        matches = list(dest.rglob(Path(rel_p).name))
                        if matches:
                            t_file = matches[0]
                        else:
                            raise KaggleError(f"File listed in manifest.json not found: {rel_p}")
                    act_sha = _compute_sha256(t_file)
                    if act_sha != exp_sha.lower():
                        raise KaggleError(
                            f"Checksum mismatch for {rel_p}: expected {exp_sha}, got {act_sha}"
                        )
        except json.JSONDecodeError:
            pass

    return [p.name for p in dest.iterdir() if p.is_file()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datasetcleaner-kaggle", description="Download Kaggle datasets or curate examples with a private Kaggle GPU notebook.")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="Search Kaggle datasets")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    download = commands.add_parser("download", help="Download a Kaggle dataset locally")
    download.add_argument("dataset_ref")
    download.add_argument("--output", type=Path, required=True)
    push_nb = commands.add_parser("push-notebook", help="Push and run kaggle_batch_inference.ipynb directly on Kaggle with dual-T4 GPUs")
    push_nb.add_argument("--username", required=True, help="Your Kaggle username")
    push_nb.add_argument("--slug", default="datasetcleaner-batch-inference", help="Kernel slug")
    push_nb.add_argument("--title", default="DataSetCleaner Batch Inference Dual T4", help="Kernel title")
    push_nb.add_argument("--language", choices=("python", "cobol", "sql", "cuda", "hcl"), default=None, help="Target language preset")
    push_nb.add_argument("--output-format", choices=("code_task", "bug_patch", "compiler_repair", "doc_to_impl", "commit_repair", "function_tests", "repo_dependency"), default=None, help="Output format")
    push_nb.add_argument("--output", type=Path, default=Path("datasetcleaner-kaggle-output"))
    push_nb.add_argument("--wait", action="store_true", help="Wait for completion and download results")
    curate = commands.add_parser("curate", help="Generate/rewrite response fields using an attached Kaggle model")
    source = curate.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-ref", help="Existing Kaggle dataset owner/slug")
    source.add_argument("--local-input", type=Path, help="Local CSV, JSONL/NDJSON, or Parquet file (uploaded privately)")
    curate.add_argument("--upload-local", action="store_true", help="Explicitly authorize copying --local-input to a private Kaggle Dataset; Kaggle retains that private dataset until you delete it.")
    curate.add_argument("--username", required=True, help="Your Kaggle username or organization slug")
    curate.add_argument("--upload-license", default="unknown", help="License recorded on private upload; defaults to unknown")
    curate.add_argument("--model-handle", default=DEFAULT_MODEL_HANDLE, help=f"Kaggle model owner/model/framework/variation/version (default: {DEFAULT_MODEL_HANDLE})")
    curate.add_argument("--prompt-field", help="Prompt field; omitted means infer from common coding-task field names")
    curate.add_argument("--response-field", default="completion")
    curate.add_argument("--mode", choices=("answer", "rewrite"), default="answer")
    curate.add_argument("--instruction", default="Write a correct, helpful answer for this coding task. Return only the answer text.")
    curate.add_argument("--input-file", help="Optional filename when the attached dataset has multiple supported files")
    curate.add_argument("--max-records", type=int, default=1000, help="Generation cap; use 0 for all rows")
    curate.add_argument("--max-new-tokens", type=int, default=1024)
    curate.add_argument("--accelerator", default="NvidiaTeslaT4")
    curate.add_argument("--output", type=Path, default=Path("datasetcleaner-kaggle-output"))
    curate.add_argument("--wait-timeout", type=int, default=7200)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "search":
            result = search_datasets(args.query, args.limit)
        elif args.command == "download":
            result = download_dataset(args.dataset_ref, args.output)
        elif args.command == "push-notebook":
            result = push_batch_notebook(
                kaggle_username=args.username,
                kernel_slug=args.slug,
                title=args.title,
                output_dir=args.output,
                language=args.language,
                output_format=args.output_format,
                wait=args.wait,
            )
        else:
            if args.local_input is not None and not args.upload_local:
                raise ValueError("Pass --upload-local to explicitly upload the local input as a private Kaggle Dataset.")
            result = run_curation(
                dataset_ref=args.dataset_ref,
                local_input=args.local_input,
                upload_local=args.upload_local,
                kaggle_username=args.username,
                upload_license=args.upload_license,
                model_handle=args.model_handle,
                prompt_field=args.prompt_field,
                response_field=args.response_field,
                mode=args.mode,
                instruction=args.instruction,
                input_file=args.input_file,
                max_records=args.max_records,
                max_new_tokens=args.max_new_tokens,
                accelerator=args.accelerator,
                output_dir=args.output,
                wait_timeout=args.wait_timeout,
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (KaggleError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

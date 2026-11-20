# AI Agent Setup & Execution Guide (`ai.setup.md`)

> **Instruction manual for autonomous AI coding agents (Antigravity, Claude Code, Cursor, Copilot, OpenCode, Aider, Roo Code).**
> This document specifies exact bootstrap steps, architectural contracts, operational boundaries, and verification procedures.

---

## 1. System Overview & Agent Invariants

DataSetCleaner & Free-RLVR is a zero-cost, high-throughput dataset synthesis, deduplication, and execution-verification engine.

### Strict Agent Invariants
1. **Zero Secret Leakage:** Never read `cred.json` or private tokens into persistent logs, commits, or git history. Always reference `cred.example.json` for schema definitions.
2. **Read-Only Raw Inputs:** Source dataset files (CSV, JSONL, Parquet) are strictly read-only. Never mutate or overwrite input files in place.
3. **Fail-Closed Sandbox Validation:** Code syntax and unit test verification must always execute in isolated subshell sandboxes (`sys.executable -I`, Windows AppContainer, or Linux namespaces/cgroups). Never fall back to unconfined execution.
4. **Deterministic Checkpointing:** All batch operations must write incremental checkpoints (`checkpoint.json`, `manifest.json`, and SHA-256 digests) to support zero-loss resumption.

---

## 2. Environment & Dependency Setup

### Step 1: Environment Isolation
```bash
# Verify Python version (Requires Python 3.11+)
python --version

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
# Core package (editable mode)
pip install -e .

# Full capabilities (recommended for agents running all workflows)
pip install -e ".[parquet,tokens,agents,kaggle,google-drive]"
```

### Step 3: Run Baseline Verification Suite
Agents must verify the environment before making changes:
```bash
python -m unittest discover -s tests -v
```
*Expected result:* 62 tests passing (`Ran 62 tests in ~2.5s -> OK`).

---

## 3. Configuration Contract (`cred.json`)

If the agent needs to invoke remote APIs or cloud storage:
1. Check if `cred.json` exists in the project root.
2. If absent, copy from `cred.example.json`:
   ```bash
   cp cred.example.json cred.json
   ```
3. `cred.json` is strictly ignored by `.gitignore`.

### Schema Reference:
```json
{
  "groq": {
    "api_keys": [
      { "id": "key_1", "key": "YOUR_GROQ_KEY", "label": "Account 1" }
    ],
    "primary_model": "qwen/qwen3.8-27b",
    "fallback_model": "openai/gpt-oss-120b"
  },
  "cloudflare": {
    "accounts": [
      {
        "id": "cf_acc_1",
        "account_id": "YOUR_CF_ACCOUNT_ID",
        "api_token": "YOUR_CF_API_TOKEN"
      }
    ]
  },
  "google_drive": {
    "client_id": "YOUR_CLIENT_ID",
    "client_secret": "YOUR_CLIENT_SECRET",
    "refresh_token": "YOUR_REFRESH_TOKEN",
    "streaming_folder_id": "YOUR_GDRIVE_FOLDER_ID"
  }
}
```

---

## 4. Operational Modes & How to Run

### Mode 1: Local Desktop GUI & CLI
- **Desktop Application:**
  ```powershell
  datasetcleaner-gui
  ```
- **CLI Analysis & Approval:**
  ```powershell
  # Step 1: Analyze dataset (read-only pass)
  datasetcleaner --input ./examples.jsonl --license MIT --output ./prepared

  # Step 2: Approved export
  datasetcleaner --input ./examples.jsonl --license MIT --output ./release --approve
  ```

### Mode 2: Loopback MCP Server & Local HTTP Daemon
Agents can integrate directly via Model Context Protocol (MCP) or HTTP:
- **MCP Server (stdio):**
  ```powershell
  python -m dataset_cleaner.mcp_server
  ```
- **HTTP Loopback Daemon (Port 8765):**
  ```powershell
  datasetcleaner-api --host 127.0.0.1 --port 8765
  ```
  Open `http://127.0.0.1:8765/` to view the interactive dataset inspection dashboard.

### Mode 3: Headless Cloud Accelerator (Kaggle GPU/TPU)
- **Batch Inference Notebook:** `kaggle_batch_inference.ipynb`
- **Execution via Kaggle CLI:**
  ```bash
  # Check kernel status
  kaggle kernels status <username>/rlvr-dataset-builder

  # Push updated headless batch job
  kaggle kernels push -p ./
  ```
- **Secrets Protocol:**
  Secrets are injected via Kaggle Secrets (`GOOGLE_DRIVE_TOKEN_JSON`, `HF_TOKEN`). Do not hardcode credentials in notebook cells.

### Mode 4: Multi-Key API Rotation (Groq & Cloudflare)
- Use async round-robin pools across configured keys.
- On HTTP `429 Too Many Requests`:
  1. Catch `RateLimitError`.
  2. Rotate active key index immediately.
  3. Apply exponential jittered backoff (1s -> 2s -> 4s) to exhausted keys.
  4. Continue pipeline without dropping in-flight batch requests.

### Mode 5: Periodic Cron Reporter & Progress Daemon
Agents or cron jobs can monitor batch execution progress without polling the UI:
```bash
# One-shot progress snapshot (cron/CLI)
python reporter.py --dir ./output

# Continuous progress monitoring daemon (every 60s)
python reporter.py --dir ./output --interval 60

# Machine-readable JSON metrics for agent workflows
python reporter.py --dir ./output --format json
```

---

## 5. Verification & Quality Gates

When synthesizing or cleaning code datasets, enforce the 3-Tier Execution Verification Gate:

```
[Candidate Code Example]
           │
           ▼
[Tier 1: AST Syntax Gate] ──(Fails)──► Reject (syntax_unchecked / malformed)
           │ (Passes)
           ▼
[Tier 2: Compilation & Import Gate] ──(Fails)──► 1-Step In-Line Reflection Repair
           │ (Passes)
           ▼
[Tier 3: Isolated Test Execution] ──(Fails)──► Reject ($r = 0.0$)
           │ (Passes $r = 1.0$)
           ▼
[Google Drive Streaming Checkpoint]
```

### Self-Diagnostic Check
Agents can verify the pipeline end-to-end using the built-in benchmark runner:
```powershell
python -m dataset_cleaner.benchmark --dataset tests/fixtures/tiny.jsonl --language python
```

---

## 6. Git Hygiene & Contribution Rules
- **No Private State in Git:** Always run `git status` before committing. Ensure `cred.json`, `token.json`, and `scratch/` are untracked.
- **Run Tests Before Commit:** Never commit code if `python -m unittest discover -s tests` fails.
- **Commit Format:** Use Conventional Commits (`feat: ...`, `fix: ...`, `docs: ...`, `test: ...`).

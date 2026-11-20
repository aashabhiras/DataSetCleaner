# DataSetCleaner & Free-RLVR: Hybrid Setup & Architecture Guide

DataSetCleaner & Free-RLVR is a zero-cost, high-throughput AI data synthesis and execution-verification engine. It supports **hybrid compute orchestration**: seamlessly alternating between **free cloud accelerator runtimes (Kaggle / Google Colab GPU & TPU)** and **high-speed multi-key cloud inference APIs (Groq & Cloudflare Workers AI)**.

---

## 1. Hybrid Compute Architecture

The engine is engineered to maximize free compute without hitting cost or hardware barriers:

```
                        ┌───────────────────────────────────────────────┐
                        │            INPUT SPECIFICATIONS / SEEDS       │
                        └───────────────────────┬───────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
  ┌─────────────────────────────┐                               ┌─────────────────────────────┐
  │   MODE A: CLOUD ACCELERATOR │                               │  MODE B: MULTI-KEY API POOL │
  │   RUNTIMES (GPU / TPU)      │                               │  (GROQ LPU & CLOUDFLARE AI) │
  │                             │                               │                             │
  │ • Kaggle Dual Tesla T4      │                               │ • Groq LPU Key Pool         │
  │   (2x 16GB = 32GB VRAM)     │                               │   (1,000–14,400 RPD / key)  │
  │ • Kaggle TPU v3-8 (JAX/XLA) │                               │ • Cloudflare Workers AI     │
  │ • Google Colab T4 Free      │                               │   (10k Neurons/day / acc)   │
  │ • Local vLLM Tensor Parallel│                               │ • Async Dynamic Round-Robin │
  │   (Qwen 2.5 Coder 14B AWQ)  │                               │ • 429 Cooldown Auto-Backoff │
  └──────────────┬──────────────┘                               └──────────────┬──────────────┘
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                │
                                                ▼
                        ┌───────────────────────────────────────────────┐
                        │          SANDBOX EXECUTION VERIFIER           │
                        │ • Isolated Subprocess (`sys.executable -I`)   │
                        │ • Compiler Toolchains (`cobc` / `iverilog`)   │
                        │ • 3-Tier Rejection: T1 Syntax, T2 Import,     │
                        │   T3 Assertion Failures                       │
                        │ • 1-Step In-Line Traceback Reflection         │
                        └───────────────────────┬───────────────────────┘
                                                │ (Only Verified $r = 1.0$ Trajectories)
                                                ▼
                        ┌───────────────────────────────────────────────┐
                        │       AUTOMATED GOOGLE DRIVE PERSISTENCE      │
                        │ • Live Telemetry (`rlvr_status.json`)         │
                        │ • Cursor Tracking (`rlvr_cursor.json`)        │
                        │ • Streamed Gzip Checkpoints (`.jsonl.gz`)     │
                        └───────────────────────────────────────────────┘
```

---

## 2. Execution Modes

### Mode A: Free Cloud Accelerator Runtimes (Kaggle / Colab GPU & TPU)
* **Zero API Cost, Offline Execution:** Run local open-weights quantized models directly on free cloud GPUs or TPUs.
* **Dual T4 GPUs (32GB VRAM):** Run `vLLM` with `tensor_parallel_size=2` hosting `Qwen/Qwen2.5-Coder-14B-Instruct-AWQ` in `float16` for high-throughput batch generation.
* **TPU v3-8 Acceleration:** Supported via JAX/Flax pipelines in `tpu_workspace/` for high-speed batch embeddings and text classification.
* **Session Length:** Runs up to 12 hours headless per Kaggle session.

### Mode B: High-Throughput Multi-Key API Calling
* **Massive Daily Quota:** Pool multiple free API keys across distinct developer accounts to multiply daily limits:
  * **Groq LPU:** ~1,000 calls/day per key on 70B/27B models, up to 14,400 calls/day on 8B models.
  * **Cloudflare Workers AI:** 10,000 Neurons/day per account for native 32B coding (`@cf/qwen/qwen2.5-coder-32b-instruct`).
* **Zero GPU Quota Used:** When running in API mode, the script executes on **Kaggle or Colab CPU instances** (which have no weekly GPU limits and can run 12-hour sessions continuously).
* **Zero Data Retention (ZDR):** Fully compatible with ZDR-enabled endpoints to ensure prompts and outputs are discarded from memory immediately after generation.

---

## 3. Quickstart & Installation

### Local Setup
```bash
# 1. Clone the repository
git clone https://github.com/your-username/DataSetCleaner.git
cd DataSetCleaner

# 2. Set up virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 3. Install in editable mode
pip install -e .
```

---

## 4. Configuration (`cred.json`)

1. Copy the example configuration template:
   ```bash
   cp cred.example.json cred.json
   ```

2. Configure your desired providers in `cred.json` (all keys are strictly ignored by `.gitignore`):

```json
{
  "groq": {
    "api_keys": [
      { "id": "key_1", "key": "gsk_...", "label": "Account 1" },
      { "id": "key_2", "key": "gsk_...", "label": "Account 2" }
    ],
    "primary_model": "qwen/qwen3.8-27b",
    "fallback_model": "openai/gpt-oss-120b"
  },
  "cloudflare": {
    "accounts": [
      {
        "id": "cf_acc_1",
        "label": "Account 1",
        "account_id": "your_account_id",
        "api_token": "cfut_..."
      }
    ]
  },
  "google_drive": {
    "client_id": "your_oauth_client_id",
    "client_secret": "your_oauth_client_secret",
    "refresh_token": "your_refresh_token",
    "streaming_folder_id": "your_target_gdrive_folder_id"
  }
}
```

---

## 5. Running on Kaggle (Headless Cloud Execution)

### Running via Kaggle CLI:
1. Ensure your `kernel-metadata.json` is configured:
   ```json
   {
     "id": "your-username/rlvr-dataset-builder",
     "title": "RLVR Dataset Builder",
     "code_file": "kaggle_batch_inference.ipynb",
     "language": "python",
     "kernel_type": "notebook",
     "is_private": true,
     "enable_gpu": false,
     "enable_internet": true
   }
   ```
2. Push and execute headlessly:
   ```bash
   kaggle kernels push -p ./
   ```
3. Check status:
   ```bash
   kaggle kernels status your-username/rlvr-dataset-builder
   ```

---

## 6. Verification and Proof-of-Work

Every verified sample includes cryptographic metadata and sandbox execution confirmation:
```json
{
  "instruction": "Write a context manager to redirect stdout...",
  "solution": "...",
  "data": {
    "tests": "def test_redirect(): ..."
  },
  "meta": {
    "verification": {
      "checked": ["solution", "tests"],
      "executed": true,
      "method": "sandbox_subprocess",
      "pass": true,
      "verified_at": "2026-11-20T18:00:00Z"
    }
  }
}
```

Live progress and checkpoints automatically stream to your Google Drive folder (`the-stack-python-shard0-rlvr-verified.jsonl.gz` + `rlvr_status.json`), providing fully transparent proof-of-work.

---

## 7. Periodic Progress Monitoring & Cron Reporter (`reporter.py`)

A built-in execution monitor tracks active generation and cleaning jobs across local or remote checkpoints (`checkpoint.json`, `manifest.json`, `rlvr_status.json`).

### Running Locally as a One-Shot Check (Crontab / Task Scheduler)
Run once, inspect current progress metrics, and exit:
```bash
python reporter.py --dir ./output
```

### Running as a Continuous Monitor Daemon
Run as a background monitor that refreshes the terminal dashboard every 60 seconds:
```bash
python reporter.py --dir ./output --interval 60
```

### JSON Output for Automated CI/CD Pipelines
```bash
python reporter.py --dir ./output --format json
```

### Cron Schedule Examples
- **Linux Crontab (every 10 minutes):**
  ```cron
  */10 * * * * cd /path/to/DataSetCleaner && python reporter.py --dir ./output --log-file ./progress.log
  ```
- **Windows Task Scheduler / Background Task:**
  ```powershell
  python reporter.py --dir ./output --interval 300 --log-file ./progress.log
  ```

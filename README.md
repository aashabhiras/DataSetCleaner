# DataSetCleaner & Free-RLVR

> **High-Throughput Hybrid Data Synthesis & Verification Engine for AI Training Datasets**
>
> Seamlessly orchestrates **Free Cloud Accelerator Runtimes (Kaggle & Google Colab GPU/TPU)** and **Multi-Key High-Speed API Pools (Groq LPUs & Cloudflare Workers AI)** with **Google Drive Remote Checkpointing** and **Isolated Subprocess AST Verification**.

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](https://www.python.org/)
[![Hardware: GPU / TPU / API](https://img.shields.io/badge/Compute-Kaggle%20%7C%20Colab%20%7C%20Groq%20%7C%20Cloudflare-orange.svg)](SETUP.md)
[![GitHub Stars](https://img.shields.io/github/stars/aashabhiras/DataSetCleaner?style=social)](https://github.com/aashabhiras/DataSetCleaner)

---

## ⚡ Architecture Highlights

* **Hybrid Compute Execution:**
  * **Mode A (Cloud GPU / TPU):** Batch inference on Kaggle Dual Tesla T4s (vLLM tensor parallel) & TPU v3-8 (JAX/Pallas) for zero-cost, high-VRAM execution.
  * **Mode B (Multi-Key API Pool):** Asynchronous rate-limited rotation across Groq LPUs (Llama 3.3 70B, Qwen 2.5 32B/Coder) and Cloudflare Workers AI edge models running on continuous free cloud CPU instances.
* **Continuous Cloud Checkpointing:** Resumable streaming sync directly to Google Drive (`manifest.json`, `checkpoint.json`, SHA-256 deduplicated gzip records) with zero-loss recovery across session restarts.
* **Sandbox Verification:** 3-tier compiler/runtime execution gates (AppContainer on Windows, Linux namespaces / cgroups / `sys.executable -I` in cloud kernels) rejecting syntactically broken code or failing unit tests before dataset inclusion.
* **Local-First & Non-Destructive:** Local Windows desktop GUI, CLI, and loopback MCP server for inspecting, deduplicating (MinHash/LSH), and auditing dataset quality without mutating source files.
* **100% Privacy & Zero-PII:** Absolute file paths redacted from manifests, all secrets isolated in `.gitignore` via [`cred.example.json`](SETUP.md), and ZDR (Zero Data Retention) compatibility.

> ⭐ **If you find this project useful, please consider giving it a star on GitHub—it helps support free open-source AI tooling!**  
> 📖 **Looking to set up and run? Check out [SETUP.md](SETUP.md) for developer setup instructions and [ai.setup.md](ai.setup.md) for AI coding agent configuration.**

---

DataSetCleaner is a local-first Windows desktop app, command-line tool, and cloud batch engine for synthesizing, combining, and reviewing coding datasets for AI training. It reads CSV, JSONL/NDJSON, Parquet, and common code files in local folders. Input files are read-only. Local analysis and optional Ollama rewrites stay on the device. The opt-in Kaggle workflow sends input rows to a private Kaggle dataset/kernel for hosted model inference. Kaggle's model runtime and entitlement rules can differ by model version; the default Qwen GPTQ model may require a compatible preinstalled GPTQ backend or a user-selected alternative model handle.

## What it does

- Combines input records into a union of fields and supports user-defined column mappings.
- Finds exact duplicates (normalizing line endings while preserving content whitespace) and candidate near duplicates using MinHash/LSH and Jaccard verification.
- Flags malformed JSONL, empty or incomplete prompt/answer rows, and syntax findings for Python, JavaScript (Node.js), and C/C++ (GCC) tagged code. Missing validators and validator timeouts are recorded as unchecked; input is never executed.
- Agent API and dashboard scans can skip syntax checking for a faster structural/deduplication pass; syntax checks remain enabled by default and the run summary records whether they were enabled.

### Syntax validator isolation

Python, JavaScript, and C/C++ syntax-only validators are launched in a Windows AppContainer with no network capability, explicit standard-stream handles, a bounded Job Object (four processes, 256 MiB per process, five seconds of CPU time per process), a five-second wall-clock timeout, and capped diagnostics. The Python child uses a fixed `ast.parse` script with isolated mode, site initialization disabled, and bytecode writes disabled; dataset text is passed as parser input and is never imported or executed. Source submitted to any syntax parser is limited to one million characters. The only staged source copy is placed in the AppContainer's private temporary folder and deleted after the check; original datasets are read-only. If the AppContainer cannot be created or a limit is reached, the record is marked `syntax_unchecked`; validation never falls back to an unrestricted process.

The AppContainer launch path requires a normal Windows desktop process able to create the per-user AppContainer profile. Environments that already confine the application inside another AppContainer may prevent profile creation; in that case external syntax checks are intentionally reported as unchecked.
- Recognizes common coding-dataset columns including `query`, `answer`, `resource`, and `lang`.
- Tracks input paths, user-supplied license identifiers, descriptions, and source row counts. Missing or differing license declarations are flagged for human review; the tool does not decide whether data can be redistributed.
- Exports JSONL, CSV, or Parquet with a quality report, provenance manifest, and record-level findings file.
- Can ask an Ollama service bound to `127.0.0.1` for a separate rewrite proposal. Proposals are never applied automatically.

The analysis index and temporary review database are created in a temporary subfolder of the chosen local output folder. They are removed when the desktop app closes or after a CLI run. Existing output packages are preserved; a version suffix is used when a name is already taken.
Exported reports retain source basenames and license details but redact machine-specific absolute paths.

## Install

Python 3.11 or newer is required. Tkinter is included with standard Windows Python distributions.

```powershell
python -m pip install .
```

Parquet read/write support is optional so the rest of the app works without extra packages:

```powershell
python -m pip install ".[parquet]"
```

Token counts use an optional tokenizer package:

```powershell
python -m pip install ".[tokens]"
datasetcleaner-tokens .\dataset.jsonl --encoding cl100k_base
```

## Start the desktop app

```powershell
datasetcleaner-gui
```

Add one or more prepared datasets or a folder of code, enter the license and provenance details for each source, analyze, inspect findings, then approve an export. Double-click a row in the review table to include or exclude it manually. The GUI pages through records in groups of 500; export rules and manual inclusion state apply to every analyzed record.

Field mapping is a JSON object from a source column name to an output name. To disambiguate identical column names across files, prefix a key with the input filename, for example:

```json
{"supplier_a.csv.prompt": "instruction", "supplier_a.csv.answer": "completion"}
```

## CLI

The first run analyzes and writes a report without creating a cleaned dataset:

```powershell
datasetcleaner --input .\examples.jsonl --license MIT --output .\prepared
```

Export requires explicit approval. Repeated `--input`, `--license`, and `--source-description` values correspond by position:

```powershell
datasetcleaner `
  --input .\supplier-a.csv --license MIT `
  --input .\source-code-folder --license Apache-2.0 `
  --field-map .\field-map.json `
  --name python-instruction-data `
  --description "Reviewed Python instruction examples" `
  --format parquet `
  --output .\release `
  --approve
```

By default, the CLI scans for exact duplicates only; pass `--near-duplicates` to run the more expensive near-duplicate scan. Approved exports exclude exact duplicates and malformed JSONL rows, while reporting but retaining near duplicates and incomplete rows. Pass `--exclude-near-duplicates` or `--exclude-incomplete` to drop those categories too. To retain malformed rows, pass `--keep-malformed`. These settings do not modify the inputs.

For an explicit list of options:

```powershell
datasetcleaner --help
```

## Local rewrite proposals

Install and start Ollama separately, download a coding model on the same device, and then use **Suggest local rewrite…** in the GUI. The CLI can create a proposal for an analyzed record:

```powershell
datasetcleaner --input .\examples.jsonl --output .\review `
  --rewrite-row 12 --model qwen2.5-coder:1.5b
```

The request uses only `http://127.0.0.1:11434`; rewrite suggestions are saved separately with `approved: false`. They do not alter the dataset or export automatically. Rewrites are optional and the 4 GB GPU in the development machine limits model size and speed.

## Kaggle datasets and hosted-model curation

Kaggle does not provide a general chat-inference endpoint for Kaggle Models. DataSetCleaner uses a **private Kaggle Notebook run** with the requested dataset and model attached. Kaggle supports private GPU kernels and attached model/dataset inputs in its [Notebook metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md); the official [Kaggle Models guide](https://www.kaggle.com/docs/models) describes downloading or attaching models. The default model is Qwen2.5-Coder-14B-Instruct-GPTQ-Int4, a practical 14B 4-bit code model; it is a default, not a benchmark-backed claim that it is always best. Kaggle GPU availability, session limits, attached-model compatibility, and account quotas can vary.

Install Kaggle CLI support and configure Kaggle credentials using Kaggle's own instructions. DataSetCleaner does not store or print your API key:

```powershell
python -m pip install ".[kaggle,google-drive]"
datasetcleaner-kaggle search "coding instruction dataset"
datasetcleaner-kaggle download owner/dataset-slug --output .\kaggle-input
```

Run private model-assisted generation from a Kaggle dataset:

```powershell
datasetcleaner-kaggle curate --dataset-ref owner/dataset-slug `
  --username YOUR_KAGGLE_USERNAME --prompt-field instruction `
  --response-field completion --mode answer --output .\kaggle-runs
```

For a local input file, pass `--local-input .\dataset.jsonl --upload-local --upload-license unknown`. This explicitly copies the file to a **private Kaggle Dataset**; Kaggle retains that private copy until you delete it there. The original local file is read-only. The run creates a private GPU kernel with notebook internet disabled, downloads a separate candidate JSONL plus rejected-row and provenance reports, and marks the result `approved: false`. The generated response is written into the selected response field. `answer` fills missing responses and keeps existing ones; use `--mode rewrite` to propose replacements for existing responses. The private input Dataset and Kernel remain in your Kaggle account until you remove them there. No public publishing occurs, and no candidate is merged into an approved export automatically.

The uploaded data is processed on Kaggle, so do not send sensitive data unless you accept that transfer and Kaggle's terms. Review the dataset/model licenses and the generated records before redistribution. For a Kaggle dataset already attached to a run, rights and provenance still need human review. The returned JSONL can be imported into DataSetCleaner for local deduplication, syntax checking, review, and an explicitly approved export.

The same workflow is available through the MCP tools `search_kaggle_datasets`, `download_kaggle_dataset`, and `run_kaggle_model_curation`. The model tool requires `confirm_remote_processing=true`; a local-file upload also requires `confirm_local_upload=true`.

The desktop app's **Kaggle / Drive…** panel provides OAuth connection launch, Kaggle dataset search/download, private model curation, local-to-Drive staging, and a button to open the Kaggle notebook. Connect Kaggle preflights the official CLI and uses `%LOCALAPPDATA%\DataSetCleaner\Kaggle` on Windows (or the platform config directory on Linux/macOS). If that path is blocked, it falls back to a DataSetCleaner folder in the OS temporary directory, which may be cleared and require signing in again. Search and curation use the same selected config directory. The app does not read or display Kaggle or Google OAuth tokens; set up Google Drive access through `GOOGLE_DRIVE_TOKEN_JSON` as described above.

### Google Drive with the Kaggle notebook

The notebook defaults to a Google Drive input file and uploads its generated candidate, manifests, and compressed audit files into a new run folder in Drive. In its config cell, set `DRIVE_INPUT_FILE_ID`; set `DRIVE_OUTPUT_FOLDER_ID` if outputs should go to a different folder. The input must be a raw CSV, JSONL/NDJSON, or Parquet dataset with a source-code column (`content` by default; set `SOURCE_CONTENT_FIELD` to map another column). The notebook downloads that one file to Kaggle working storage for processing, then uploads the run artifacts. It does not edit or delete the Drive source.

The Kaggle notebook setup also installs Isolate and common runtimes (Python, C/C++, JavaScript, and Java) inside the temporary Kaggle VM. Its sandbox runner uses Linux namespaces, cgroups, no network, and CPU/memory/process/file limits. It fails closed if Kaggle does not permit the required sandbox setup. Automatic test execution is limited to model-generated Python tests in `function_tests` mode; raw input code remains unexecuted. The notebook also exposes `run_sandboxed_program(language, source, stdin="")` for standalone examples in the listed languages. These packages change the Kaggle runtime only, not this computer.

Create a Google OAuth authorized-user token with the Drive API enabled and store its full JSON in Kaggle Secrets as `GOOGLE_DRIVE_TOKEN_JSON`. The notebook uses the broad Drive OAuth scope to read the selected file and create output files; use a dedicated Google account/folder and revoke the token when finished. Kaggle internet must be enabled for Drive API access. The OAuth token is never written into the notebook or outputs.

To stage a local dataset directly from this computer into a Drive folder, install the optional uploader dependency and set `GOOGLE_DRIVE_TOKEN_JSON` in the process environment to the authorized-user token JSON. The uploader streams the selected file in resumable chunks and leaves the local source unchanged:

```powershell
python -m pip install ".[google-drive]"
$env:GOOGLE_DRIVE_TOKEN_JSON = Get-Content -Raw .\google-drive-token.json
datasetcleaner-drive-upload E:\Downloads\CodeFeedback-Filtered-Instruction.jsonl `
  --folder-id YOUR_DRIVE_FOLDER_ID --confirm-upload
```

The Drive file ID is printed after upload; put it in `DRIVE_INPUT_FILE_ID` in the notebook. Treat the OAuth token JSON as a password and do not commit it.

## Use with VS Code, Antigravity, OpenCode, or another agent

DataSetCleaner exposes the same review operations through an optional local MCP server and a loopback-only HTTP API. The MCP integration is suitable for agent hosts that support local stdio MCP. The API is available to other local agents and scripts that can make HTTP requests. Both use the same persistent review store, and both leave source files untouched.

Install the agent extra from this repository (Python 3.11+):

```powershell
python -m pip install ".[agents]"
```

For the Kaggle MCP tools, install `python -m pip install ".[agents,kaggle]"` and configure Kaggle credentials with Kaggle's CLI.

### MCP setup

Copy the matching example from `integrations/` into the host's MCP configuration and restart/reload its MCP servers:

- **OpenCode:** merge `integrations/opencode.jsonc` into `opencode.json`/`opencode.jsonc` at the project or user level. The example uses OpenCode's `mcp.servers` local command format.
- **VS Code Copilot:** merge `integrations/vscode-copilot.mcp.json` into the workspace-root `.mcp.json` or Copilot user config at `~/.copilot/mcp-config.json`. VS Code's current Copilot harness uses the top-level `mcpServers` format; `.vscode/mcp.json` belongs to the deprecated Local harness.
- **Antigravity IDE, CLI, or app:** merge `integrations/antigravity.mcp_config.json` into the MCP config exposed by MCP Servers > Manage (or workspace `.agents/mcp_config.json`).

The examples run `python -m dataset_cleaner.mcp_server`, so install DataSetCleaner into the same Python environment used by the agent host. In OpenCode, run `/connect`, configure Google's provider with your AI Studio API key, then run `/models` and select Gemini. In Antigravity, choose its Gemini model in the host. DataSetCleaner does not store provider credentials or make model calls itself. OpenCode documents Gemini as a built-in provider, but availability and free quotas depend on the provider's current terms. Gemini API free-tier data-use terms differ from paid-tier terms; check the current [Gemini API pricing and data-use disclosure](https://ai.google.dev/gemini-api/docs/pricing) before sending proprietary records. Agents only see records they explicitly request through `get_dataset_records`; use the Ollama rewrite workflow or a local agent model for private data.

The MCP tools can start analyses, inspect findings and record pages, exclude records for review, stage generated examples separately with provider/model provenance, search and download Kaggle datasets, run private Kaggle model curation after explicit remote-processing confirmation, preview cleanup rules, and export after explicit user approval. Generated examples remain separate unapproved candidates; they are not silently mixed into an export. Agent output must still be reviewed for correctness and rights.

Example agent request: “Use DataSetCleaner on `E:\data\prepared.jsonl` and `E:\data\supplement.csv`. Map the supplement's `task` and `answer` fields to the prepared schema, analyze locally, inspect the findings and representative records, exclude only clearly unusable rows, and stage up to 50 additional reviewed-quality examples as separate candidates. Show me the export preview and rights gaps; do not export until I approve.”

### Open the front end inside VS Code or Antigravity

Start the dashboard and local API from a terminal:

```powershell
datasetcleaner-api
```

Then open `http://127.0.0.1:8765/` in VS Code's **Simple Browser** (Command Palette → “Simple Browser: Show”) or in Antigravity's browser. The page and API stay on loopback. The dashboard uses absolute local input paths, so the API process must run on the machine that can read those files. Its generated candidate and review state are stored under the OS application data directory; set `DATASETCLEANER_AGENT_HOME` to choose another local location.

### Generic local HTTP API

The request/response schema is in `integrations/openapi.yaml`. Start the API with `datasetcleaner-api --host 127.0.0.1 --port 8765`; it refuses non-loopback bind addresses. Use `POST /api/runs` to analyze, poll `GET /api/runs/{run_id}`, then review findings and records before posting decisions. Set `check_syntax: false` for a faster structural/deduplication pass; its run summary records that syntax parsing was skipped. `GET /api/runs/{run_id}/records` returns source content to the caller, so using a cloud model provider through an agent can send that requested content to the provider. `POST /api/runs/{run_id}/export` requires `user_approved: true`; the API writes a new output package, never overwrites the source dataset.

The HTTP API has no authentication because it is loopback-only and intended for this machine. Do not expose it through a reverse proxy, port forward, or public interface.

## Current quality limits

- Near-deduplication is heuristic. Review the findings and adjust the threshold for your domain; it can miss duplicates or flag similar but distinct examples.
- Syntax checks are parser/compiler checks, not semantic correctness tests. Python uses the standard-library parser; JavaScript and C/C++ checks require Node.js and GCC on `PATH`. JSX is reported unchecked because Node.js does not parse it, and other languages are not parsed.
- Numeric and boolean Parquet columns preserve their types when consistent. Nested and mixed-type values are serialized as JSON strings to produce stable output schemas.
- This first release has no marketplace integration, network processing, automatic license detection, or claim that a generated dataset is saleable. You must verify rights and marketplace requirements yourself.
- Near-duplicate detection uses this app's disk-backed MinHash/LSH review path. The upstream `text-dedup` package is a configurable Hugging Face Dataset pipeline with a much larger dependency stack, so it was not embedded as a Windows runtime dependency in this first release. No upstream code is vendored.
- The 10 GB run target is an initial benchmark goal, not a performance guarantee. Actual capacity depends on record size, duplicate density, and available free disk space.

## Development-machine benchmark

Synthetic JSONL fixture runs on the i5-9300H machine (8 logical cores, 40 GB RAM, GTX 1650 4 GB; Python 3.14.7):

| Mode | Records | Input size | Time | Throughput | Peak process RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact dedup + validation | 10,000 | 10.8 MB | 6.0 s | 1,674 records/s | 28.5 MB |
| Exact dedup + validation | 100,000 | 107.9 MB | 76.8 s | 1,302 records/s | 28.8 MB |
| With near-dedup scan | 5,000 | 4.1 MB | 10.8 s | 462 records/s | 28.6 MB |

These synthetic runs show practical local work at hundreds of thousands of examples. They do not validate 10 GB runs; a multi-hour runtime and additional working disk are plausible at that scale. Re-run or adapt `benchmarks/benchmark_pipeline.py` with representative data before setting a production limit.

Run the focused verification suite with `python -m unittest discover -s tests -v`.

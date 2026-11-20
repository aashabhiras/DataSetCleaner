"""Modern Tkinter desktop interface for reviewed local dataset curation and cloud workflows."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .engine import RunSession, unique_output_path
from .sources import SourceSpec


class DataSetCleanerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DataSetCleaner — Local Coding Dataset Curation & Cloud Pipeline")
        self.root.geometry("1280x850")
        self.root.minsize(1050, 700)

        self.sources: list[SourceSpec] = []
        self.session: RunSession | None = None
        self.temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self.summary: dict[str, Any] = {}
        self.page_offset = 0
        self.page_size = 500
        self.busy = False
        self.rewrite_busy = False
        self.api_server: Any = None
        self.api_thread: threading.Thread | None = None

        self._configure_styles()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self._build()
        self.root.after(300, self._check_kaggle_connection_silent)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            if "clam" in style.theme_names():
                style.theme_use("clam")
        except Exception:
            pass

        font_family = "Segoe UI" if os.name == "nt" else "Helvetica"
        mono_family = "Consolas" if os.name == "nt" else "Courier"

        style.configure(".", font=(font_family, 10))
        style.configure("Header.TLabel", font=(font_family, 18, "bold"))
        style.configure("Subheader.TLabel", font=(font_family, 10), foreground="#555555")
        style.configure("CardTitle.TLabel", font=(font_family, 11, "bold"))
        style.configure("Mono.TLabel", font=(mono_family, 9))
        style.configure("Badge.TLabel", font=(font_family, 9, "bold"), padding=(6, 2))
        style.configure("StatValue.TLabel", font=(font_family, 14, "bold"))
        style.configure("StatLabel.TLabel", font=(font_family, 9), foreground="#666666")
        style.configure("TNotebook.Tab", padding=(14, 7), font=(font_family, 10, "bold"))

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=(16, 12, 16, 14))
        outer.pack(fill="both", expand=True)

        # Top Header & Status Bar
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 10))

        title_box = ttk.Frame(header)
        title_box.pack(side="left", fill="x", expand=True)
        ttk.Label(title_box, text="DataSetCleaner", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            title_box,
            text="Local-First Coding Dataset Curation • On-Device Merging, Quality Auditing, De-duplication & Cloud Workflows",
            style="Subheader.TLabel",
        ).pack(anchor="w")

        # Global System Status Pills
        status_bar = ttk.Frame(header)
        status_bar.pack(side="right", anchor="e")

        self.kaggle_status_pill = ttk.Label(
            status_bar,
            text="Kaggle: Checking…",
            relief="groove",
            style="Badge.TLabel",
        )
        self.kaggle_status_pill.pack(side="left", padx=4)

        self.api_status_pill = ttk.Label(
            status_bar,
            text="Agent API: Stopped",
            relief="groove",
            style="Badge.TLabel",
        )
        self.api_status_pill.pack(side="left", padx=4)

        ttk.Button(status_bar, text="Check Kaggle Auth", command=self._check_kaggle_connection_manual).pack(
            side="left", padx=4
        )

        # Main Tabbed Interface
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)

        self.curation_tab = ttk.Frame(self.tabs, padding=10)
        self.token_tab = ttk.Frame(self.tabs, padding=14)
        self.agent_tab = ttk.Frame(self.tabs, padding=14)
        self.kaggle_tab = ttk.Frame(self.tabs, padding=14)
        self.drive_tab = ttk.Frame(self.tabs, padding=14)

        self.tabs.add(self.curation_tab, text="  Dataset Curation & Review  ")
        self.tabs.add(self.token_tab, text="  Token Counter  ")
        self.tabs.add(self.agent_tab, text="  Agent & MCP Server  ")
        self.tabs.add(self.kaggle_tab, text="  Kaggle & Cloud Models  ")
        self.tabs.add(self.drive_tab, text="  Google Drive & Hugging Face  ")

        self.benchmark_tab = ttk.Frame(self.tabs, padding=14)
        self.tabs.add(self.benchmark_tab, text="  Dataset Benchmark & Release  ")

        self._build_curation_tab()
        self._build_token_tab()
        self._build_agent_tab()
        self._build_kaggle_tab()
        self._build_drive_tab()
        self._build_benchmark_tab()

    # -------------------------------------------------------------------------
    # TAB 1: CURATION & REVIEW
    # -------------------------------------------------------------------------
    def _build_curation_tab(self) -> None:
        panes = ttk.Panedwindow(self.curation_tab, orient="horizontal")
        panes.pack(fill="both", expand=True)

        left = ttk.Frame(panes, padding=(0, 0, 10, 0))
        right = ttk.Frame(panes)
        panes.add(left, weight=1)
        panes.add(right, weight=3)

        # 1. Inputs Card
        in_card = ttk.LabelFrame(left, text="1. Dataset Inputs", padding=8)
        in_card.pack(fill="x", pady=(0, 8))

        self.source_list = tk.Listbox(in_card, height=5, exportselection=False, font=("Consolas", 9))
        self.source_list.pack(fill="x", pady=(2, 6))

        src_btn_row = ttk.Frame(in_card)
        src_btn_row.pack(fill="x")
        ttk.Button(src_btn_row, text="Add Files…", command=self._add_files).pack(side="left", padx=(0, 4))
        ttk.Button(src_btn_row, text="Add Code Folder…", command=self._add_folder).pack(side="left", padx=(0, 4))
        ttk.Button(src_btn_row, text="Remove", command=self._remove_source).pack(side="left")

        # 2. Rights & Metadata Card
        meta_card = ttk.LabelFrame(left, text="2. Source Rights & Provenance", padding=8)
        meta_card.pack(fill="x", pady=(0, 8))

        ttk.Label(meta_card, text="License Identifier (e.g. MIT, Apache-2.0)").pack(anchor="w")
        self.license_var = tk.StringVar()
        ttk.Entry(meta_card, textvariable=self.license_var).pack(fill="x", pady=(2, 4))

        ttk.Label(meta_card, text="Source Description").pack(anchor="w")
        self.source_description_var = tk.StringVar()
        ttk.Entry(meta_card, textvariable=self.source_description_var).pack(fill="x", pady=(2, 6))

        ttk.Button(meta_card, text="Save Selected Details", command=self._save_source_metadata).pack(anchor="e")
        self.source_list.bind("<<ListboxSelect>>", self._source_selected)

        # 3. Merge & Deduplication Settings Card
        cfg_card = ttk.LabelFrame(left, text="3. Deduplication & Audit", padding=8)
        cfg_card.pack(fill="x", pady=(0, 8))

        ttk.Label(cfg_card, text="Field Map JSON (source → output)").pack(anchor="w")
        self.field_map_var = tk.StringVar(value="{}")
        ttk.Entry(cfg_card, textvariable=self.field_map_var).pack(fill="x", pady=(2, 4))

        ttk.Label(cfg_card, text="Deduplication Fields (comma separated)").pack(anchor="w")
        self.fields_var = tk.StringVar()
        ttk.Entry(cfg_card, textvariable=self.fields_var).pack(fill="x", pady=(2, 4))

        threshold_row = ttk.Frame(cfg_card)
        threshold_row.pack(fill="x", pady=(2, 4))
        ttk.Label(threshold_row, text="Near-dup Threshold:").pack(side="left")
        self.threshold_var = tk.StringVar(value="0.90")
        ttk.Entry(threshold_row, textvariable=self.threshold_var, width=6).pack(side="left", padx=6)

        self.detect_near_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cfg_card, text="Scan near-duplicates (MinHash/LSH)", variable=self.detect_near_var).pack(anchor="w", pady=2)

        self.detect_semantic_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(cfg_card, text="Level 4 Semantic Dedup (preserve algorithms)", variable=self.detect_semantic_var).pack(anchor="w", pady=2)

        self.field_map_var.trace_add("write", self._settings_changed)
        self.fields_var.trace_add("write", self._settings_changed)
        self.threshold_var.trace_add("write", self._settings_changed)
        self.detect_near_var.trace_add("write", self._settings_changed)
        self.detect_semantic_var.trace_add("write", self._settings_changed)

        ttk.Button(cfg_card, text="Analyze & Audit Inputs", command=self._start_analysis).pack(fill="x", pady=(6, 2))

        # 4. Export Package Card
        exp_card = ttk.LabelFrame(left, text="4. Export Cleaned Dataset", padding=8)
        exp_card.pack(fill="x")

        ttk.Label(exp_card, text="Dataset Name").pack(anchor="w")
        self.name_var = tk.StringVar(value="coding-dataset")
        ttk.Entry(exp_card, textvariable=self.name_var).pack(fill="x", pady=(1, 4))

        ttk.Label(exp_card, text="Output Folder").pack(anchor="w")
        f_row = ttk.Frame(exp_card)
        f_row.pack(fill="x", pady=(1, 4))
        self.output_var = tk.StringVar(value=str(Path.cwd() / "datasetcleaner-output"))
        ttk.Entry(f_row, textvariable=self.output_var).pack(side="left", fill="x", expand=True)
        ttk.Button(f_row, text="Browse…", command=self._choose_output).pack(side="left", padx=(4, 0))

        fmt_row = ttk.Frame(exp_card)
        fmt_row.pack(fill="x", pady=(2, 4))
        ttk.Label(fmt_row, text="Format:").pack(side="left")
        self.format_var = tk.StringVar(value="jsonl")
        ttk.Combobox(
            fmt_row,
            textvariable=self.format_var,
            values=("jsonl", "csv", "parquet"),
            state="readonly",
            width=10,
        ).pack(side="left", padx=6)

        self.exclude_exact_var = tk.BooleanVar(value=True)
        self.exclude_near_var = tk.BooleanVar(value=False)
        self.exclude_incomplete_var = tk.BooleanVar(value=False)
        self.exclude_malformed_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(exp_card, text="Exclude exact duplicates", variable=self.exclude_exact_var).pack(anchor="w")
        ttk.Checkbutton(exp_card, text="Exclude near duplicates", variable=self.exclude_near_var).pack(anchor="w")
        ttk.Checkbutton(exp_card, text="Exclude incomplete/empty rows", variable=self.exclude_incomplete_var).pack(anchor="w")
        ttk.Checkbutton(exp_card, text="Exclude malformed JSON", variable=self.exclude_malformed_var).pack(anchor="w")

        self.dataset_description_var = tk.StringVar()
        self.export_button = ttk.Button(exp_card, text="Review & Export Package…", command=self._export)
        self.export_button.pack(fill="x", pady=(8, 2))

        # Right Pane: Status, Statistics, Interactive Review Table
        top_bar = ttk.Frame(right)
        top_bar.pack(fill="x", pady=(0, 6))

        self.status_var = tk.StringVar(value="Add CSV, JSONL, Parquet, or a source code folder to begin.")
        ttk.Label(top_bar, textvariable=self.status_var, font=("Segoe UI", 10, "italic")).pack(side="left", fill="x", expand=True)

        self.rewrite_button = ttk.Button(top_bar, text="Suggest local rewrite…", command=self._rewrite_selected)
        self.rewrite_button.pack(side="right")

        # Statistics Cards
        self.stats_frame = ttk.Frame(right, relief="groove", padding=8)
        self.stats_frame.pack(fill="x", pady=(0, 8))

        self.stat_records = self._make_stat_box(self.stats_frame, "Records", "0", 0)
        self.stat_valid = self._make_stat_box(self.stats_frame, "Valid", "0", 1)
        self.stat_dups = self._make_stat_box(self.stats_frame, "Duplicates", "0", 2)
        self.stat_syntax = self._make_stat_box(self.stats_frame, "Syntax Findings", "0", 3)
        self.stat_license = self._make_stat_box(self.stats_frame, "License Warnings", "0", 4)
        for i in range(5):
            self.stats_frame.columnconfigure(i, weight=1)

        # Summary findings textbox
        self.summary_text = tk.Text(right, height=4, wrap="word", state="disabled", font=("Consolas", 9), relief="solid", bd=1)
        self.summary_text.pack(fill="x", pady=(0, 6))

        # Review Table Frame
        table_frame = ttk.Frame(right)
        table_frame.pack(fill="both", expand=True)

        columns = ("include", "id", "issues", "provenance", "source", "preview")
        self.table = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        for key, title, width in [
            ("include", "Include", 75),
            ("id", "ID", 60),
            ("issues", "Findings", 160),
            ("provenance", "Provenance", 130),
            ("source", "Source / Row", 180),
            ("preview", "Record Content Preview", 460),
        ]:
            self.table.heading(key, text=title)
            self.table.column(key, width=width, stretch=(key in {"source", "preview"}), anchor="w")

        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.table.yview)
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        self.table.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.table.bind("<Double-1>", self._toggle_include)
        self.table.bind("<space>", self._toggle_include)

        # Pagination & Table Controls
        page_controls = ttk.Frame(right)
        page_controls.pack(fill="x", pady=(6, 0))

        self.prev_button = ttk.Button(page_controls, text="← Previous 500", command=lambda: self._change_page(-500), state="disabled")
        self.prev_button.pack(side="left")

        self.page_var = tk.StringVar(value="No review rows")
        ttk.Label(page_controls, textvariable=self.page_var, font=("Segoe UI", 9, "bold")).pack(side="left", padx=12)

        self.next_button = ttk.Button(page_controls, text="Next 500 →", command=lambda: self._change_page(500), state="disabled")
        self.next_button.pack(side="left")

    def _make_stat_box(self, parent: ttk.Frame, label: str, val: str, col: int) -> ttk.Label:
        f = ttk.Frame(parent, padding=4)
        f.grid(row=0, column=col, sticky="nsew")
        vl = ttk.Label(f, text=val, style="StatValue.TLabel", anchor="center")
        vl.pack(anchor="center")
        ttk.Label(f, text=label, style="StatLabel.TLabel", anchor="center").pack(anchor="center")
        return vl

    # -------------------------------------------------------------------------
    # TAB 2: TOKEN COUNTER
    # -------------------------------------------------------------------------
    def _build_token_tab(self) -> None:
        ttk.Label(self.token_tab, text="JSONL Token Volume Counter", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            self.token_tab,
            text="Estimate training corpus token volume locally using OpenAI/Qwen tiktoken encodings.",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(0, 14))

        input_frame = ttk.LabelFrame(self.token_tab, text="Token Counting Settings", padding=12)
        input_frame.pack(fill="x", pady=(0, 14))

        f_row = ttk.Frame(input_frame)
        f_row.pack(fill="x", pady=(0, 8))
        ttk.Label(f_row, text="Target JSONL File:").pack(side="left", padx=(0, 8))
        self.token_file_var = tk.StringVar()
        ttk.Entry(f_row, textvariable=self.token_file_var).pack(side="left", fill="x", expand=True)
        ttk.Button(f_row, text="Browse…", command=lambda: self._choose_dataset_file(self.token_file_var)).pack(side="left", padx=(6, 0))

        opt_row = ttk.Frame(input_frame)
        opt_row.pack(fill="x", pady=(0, 8))

        ttk.Label(opt_row, text="Tokenizer Encoding:").pack(side="left", padx=(0, 6))
        self.token_enc_var = tk.StringVar(value="cl100k_base")
        ttk.Combobox(
            opt_row,
            textvariable=self.token_enc_var,
            values=("cl100k_base", "p50k_base", "r50k_base", "gpt2"),
            state="readonly",
            width=14,
        ).pack(side="left", padx=(0, 16))

        ttk.Label(opt_row, text="Fields to Count (comma-separated, optional):").pack(side="left", padx=(0, 6))
        self.token_fields_var = tk.StringVar()
        ttk.Entry(opt_row, textvariable=self.token_fields_var, width=28).pack(side="left", fill="x", expand=True)

        ttk.Button(input_frame, text="Count Tokens Locally", command=self._run_token_count).pack(anchor="e", pady=(4, 0))

        # Token Results Frame
        res_frame = ttk.LabelFrame(self.token_tab, text="Token Breakdown Results", padding=12)
        res_frame.pack(fill="both", expand=True)

        self.token_status_var = tk.StringVar(value="Select a JSONL dataset file and click Count Tokens Locally.")
        ttk.Label(res_frame, textvariable=self.token_status_var, font=("Segoe UI", 10, "italic")).pack(anchor="w", pady=(0, 8))

        cols = ("field", "tokens")
        self.token_table = ttk.Treeview(res_frame, columns=cols, show="headings", height=8)
        self.token_table.heading("field", text="Dataset Field Name")
        self.token_table.heading("tokens", text="Total Token Count")
        self.token_table.column("field", width=300, anchor="w")
        self.token_table.column("tokens", width=200, anchor="e")
        self.token_table.pack(fill="both", expand=True, pady=(0, 8))

        self.token_summary_var = tk.StringVar(value="")
        ttk.Label(res_frame, textvariable=self.token_summary_var, font=("Consolas", 10, "bold")).pack(anchor="w")

    def _run_token_count(self) -> None:
        path_str = self.token_file_var.get().strip()
        if not path_str or not Path(path_str).is_file():
            messagebox.showerror("Invalid File", "Please select an existing JSONL dataset file.", parent=self.root)
            return

        encoding = self.token_enc_var.get().strip() or "cl100k_base"
        raw_fields = self.token_fields_var.get().strip()
        fields = [f.strip() for f in raw_fields.split(",") if f.strip()] if raw_fields else None

        self.token_status_var.set("Counting tokens locally with tiktoken…")
        self.token_table.delete(*self.token_table.get_children())
        self.token_summary_var.set("")

        def worker() -> None:
            try:
                from .token_count import count_jsonl_tokens

                res = count_jsonl_tokens(path_str, encoding_name=encoding, fields=fields)
                self.root.after(0, self._render_token_results, res)
            except Exception as exc:
                self.root.after(0, self.token_status_var.set, f"Token counting failed: {exc}")
                self.root.after(0, messagebox.showerror, "Token Counting Error", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _render_token_results(self, res: dict[str, Any]) -> None:
        self.token_status_var.set(
            f"Completed: {res['file']} | {res['records']:,} records ({res['malformed_records']} malformed) | Tokenizer: {res['tokenizer']}"
        )
        for key, count in res.get("field_value_tokens", {}).items():
            self.token_table.insert("", "end", values=(key, f"{count:,}"))

        summary = (
            f"Serialized JSONL Tokens (all nonblank lines): {res.get('serialized_jsonl_tokens', 0):,}  |  "
            f"Field Values Token Sum: {res.get('field_value_tokens_total', 0):,}"
        )
        self.token_summary_var.set(summary)

    # -------------------------------------------------------------------------
    # TAB 3: AGENT & MCP SERVER
    # -------------------------------------------------------------------------
    def _build_agent_tab(self) -> None:
        ttk.Label(self.agent_tab, text="Local Agent API & MCP Server", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            self.agent_tab,
            text="Expose review-safe DataSetCleaner tools to Google Antigravity, VS Code Copilot, and OpenCode.",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(0, 14))

        # Service Control Card
        svc_card = ttk.LabelFrame(self.agent_tab, text="Loopback REST API & Web Dashboard Service", padding=12)
        svc_card.pack(fill="x", pady=(0, 14))

        row1 = ttk.Frame(svc_card)
        row1.pack(fill="x", pady=(0, 8))

        ttk.Label(row1, text="Host:").pack(side="left")
        ttk.Label(row1, text="127.0.0.1 (Loopback Only)", font=("Segoe UI", 9, "bold")).pack(side="left", padx=(4, 16))

        ttk.Label(row1, text="Port:").pack(side="left")
        self.api_port_var = tk.StringVar(value="8765")
        ttk.Entry(row1, textvariable=self.api_port_var, width=8).pack(side="left", padx=(4, 16))

        self.btn_start_api = ttk.Button(row1, text="Start API Service", command=self._start_agent_api)
        self.btn_start_api.pack(side="left", padx=4)

        self.btn_stop_api = ttk.Button(row1, text="Stop API Service", command=self._stop_agent_api, state="disabled")
        self.btn_stop_api.pack(side="left", padx=4)

        ttk.Button(row1, text="Open Web Dashboard in Browser ↗", command=self._open_web_dashboard).pack(side="right")

        self.agent_service_status_var = tk.StringVar(
            value="Service stopped. Start the service to enable the local web dashboard at http://127.0.0.1:8765/."
        )
        ttk.Label(svc_card, textvariable=self.agent_service_status_var, font=("Segoe UI", 9, "italic")).pack(anchor="w")

        # Host Configuration Hub Card
        cfg_card = ttk.LabelFrame(self.agent_tab, text="MCP Host Configuration Presets", padding=12)
        cfg_card.pack(fill="both", expand=True)

        sel_row = ttk.Frame(cfg_card)
        sel_row.pack(fill="x", pady=(0, 8))

        ttk.Label(sel_row, text="Select Agent Host:").pack(side="left", padx=(0, 8))
        self.host_cfg_var = tk.StringVar(value="Google Antigravity")
        host_combo = ttk.Combobox(
            sel_row,
            textvariable=self.host_cfg_var,
            values=("Google Antigravity", "VS Code Copilot", "OpenCode"),
            state="readonly",
            width=22,
        )
        host_combo.pack(side="left", padx=(0, 12))
        host_combo.bind("<<ComboboxSelected>>", self._update_host_cfg_text)

        ttk.Button(sel_row, text="Copy Configuration JSON", command=self._copy_host_cfg).pack(side="left")

        self.host_cfg_text = tk.Text(cfg_card, wrap="none", font=("Consolas", 9), relief="solid", bd=1)
        self.host_cfg_text.pack(fill="both", expand=True, pady=(4, 0))
        self._update_host_cfg_text()

    def _start_agent_api(self) -> None:
        try:
            port = int(self.api_port_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Port", "Port must be an integer.", parent=self.root)
            return

        try:
            from .agent_service import AgentService
            from .api import _ApiServer

            service = AgentService()
            self.api_server = _ApiServer(("127.0.0.1", port), service)
            self.api_thread = threading.Thread(target=self.api_server.serve_forever, daemon=True)
            self.api_thread.start()

            self.btn_start_api.configure(state="disabled")
            self.btn_stop_api.configure(state="normal")
            self.api_status_pill.configure(text=f"Agent API: Running ({port})")
            self.agent_service_status_var.set(f"Running on http://127.0.0.1:{port}/ (Loopback only). Ready for agent calls.")
        except Exception as exc:
            messagebox.showerror("Could not start API service", str(exc), parent=self.root)

    def _stop_agent_api(self) -> None:
        if self.api_server:
            try:
                self.api_server.shutdown()
                self.api_server.server_close()
            except Exception:
                pass
            self.api_server = None
        self.btn_start_api.configure(state="normal")
        self.btn_stop_api.configure(state="disabled")
        self.api_status_pill.configure(text="Agent API: Stopped")
        self.agent_service_status_var.set("Service stopped.")

    def _open_web_dashboard(self) -> None:
        port = self.api_port_var.get().strip() or "8765"
        if not self.api_server:
            # Auto-start if stopped
            self._start_agent_api()
        webbrowser.open_new_tab(f"http://127.0.0.1:{port}/")

    def _update_host_cfg_text(self, *_: Any) -> None:
        host = self.host_cfg_var.get()
        p = Path(__file__).resolve().parent.parent / "integrations"
        cfg_file = {
            "Google Antigravity": p / "antigravity.mcp_config.json",
            "VS Code Copilot": p / "vscode-copilot.mcp.json",
            "OpenCode": p / "opencode.jsonc",
        }.get(host, p / "antigravity.mcp_config.json")

        content = cfg_file.read_text(encoding="utf-8") if cfg_file.is_file() else f"Config file not found: {cfg_file}"
        self.host_cfg_text.delete("1.0", "end")
        self.host_cfg_text.insert("1.0", content)

    def _copy_host_cfg(self) -> None:
        text = self.host_cfg_text.get("1.0", "end").strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        messagebox.showinfo("Copied", f"Copied {self.host_cfg_var.get()} configuration to clipboard.", parent=self.root)

    # -------------------------------------------------------------------------
    # TAB 4: KAGGLE & CLOUD MODELS
    # -------------------------------------------------------------------------
    def _build_kaggle_tab(self) -> None:
        ttk.Label(self.kaggle_tab, text="Kaggle Connection & Hosted GPU Pipelines", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            self.kaggle_tab,
            text="Authenticate Kaggle CLI, download datasets, and run private batch inference across dual T4 GPUs.",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        # Top Auth Card
        auth_card = ttk.LabelFrame(self.kaggle_tab, text="Kaggle Authentication & Configuration", padding=12)
        auth_card.pack(fill="x", pady=(0, 10))

        st_row = ttk.Frame(auth_card)
        st_row.pack(fill="x", pady=(0, 8))

        self.kaggle_auth_label = ttk.Label(st_row, text="Status: Checking…", font=("Segoe UI", 10, "bold"))
        self.kaggle_auth_label.pack(side="left")

        ttk.Button(st_row, text="Verify / Test Connection", command=self._check_kaggle_connection_manual).pack(side="left", padx=12)
        ttk.Button(st_row, text="Connect via Web OAuth (Browser)", command=lambda: self._kaggle_login_from_dialog(self.kaggle_auth_label)).pack(
            side="left", padx=4
        )
        ttk.Button(st_row, text="Open Kaggle Notebook ↗", command=self._open_kaggle_notebook).pack(side="right")

        # Direct Credential Inputs
        cred_row = ttk.Frame(auth_card)
        cred_row.pack(fill="x", pady=(4, 4))

        ttk.Label(cred_row, text="Username:").pack(side="left")
        self.kg_user_var = tk.StringVar()
        ttk.Entry(cred_row, textvariable=self.kg_user_var, width=16).pack(side="left", padx=(4, 12))

        ttk.Label(cred_row, text="API Key:").pack(side="left")
        self.kg_key_var = tk.StringVar()
        ttk.Entry(cred_row, textvariable=self.kg_key_var, show="*", width=24).pack(side="left", padx=(4, 12))

        ttk.Button(cred_row, text="Save Direct Credentials", command=self._save_direct_kaggle_creds).pack(side="left", padx=4)
        ttk.Button(cred_row, text="Import kaggle.json…", command=self._import_kaggle_json).pack(side="left", padx=4)

        self.kg_config_path_var = tk.StringVar()
        ttk.Label(auth_card, textvariable=self.kg_config_path_var, font=("Consolas", 8), foreground="#666").pack(anchor="w", pady=(4, 0))

        # Split: Left (Dataset Search/Download) | Right (Hosted GPU Batch Curation)
        body = ttk.Panedwindow(self.kaggle_tab, orient="horizontal")
        body.pack(fill="both", expand=True)

        left_kg = ttk.Frame(body, padding=(0, 0, 8, 0))
        right_kg = ttk.Frame(body, padding=(8, 0, 0, 0))
        body.add(left_kg, weight=1)
        body.add(right_kg, weight=1)

        # Dataset Search & Download
        ds_card = ttk.LabelFrame(left_kg, text="Search & Download Kaggle Datasets", padding=10)
        ds_card.pack(fill="both", expand=True)

        s_row = ttk.Frame(ds_card)
        s_row.pack(fill="x", pady=(0, 6))
        self.kg_query_var = tk.StringVar(value="coding instruction dataset")
        ttk.Entry(s_row, textvariable=self.kg_query_var).pack(side="left", fill="x", expand=True)
        ttk.Button(s_row, text="Search", command=self._kaggle_search_action).pack(side="left", padx=(4, 0))

        self.kg_results_list = tk.Listbox(ds_card, height=10, exportselection=False, font=("Consolas", 9))
        self.kg_results_list.pack(fill="both", expand=True, pady=(2, 6))
        self.kg_results_list.result_refs = []  # type: ignore[attr-defined]

        ttk.Label(ds_card, text="Download destination:").pack(anchor="w")
        dest_row = ttk.Frame(ds_card)
        dest_row.pack(fill="x", pady=(2, 6))
        self.kg_download_dir_var = tk.StringVar(value=str(Path.cwd() / "datasetcleaner-output" / "kaggle-input"))
        ttk.Entry(dest_row, textvariable=self.kg_download_dir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(dest_row, text="Browse…", command=lambda: self._choose_folder(self.kg_download_dir_var)).pack(side="left", padx=(4, 0))

        self.kg_search_status_var = tk.StringVar(value="Enter a search term to find datasets on Kaggle.")
        ttk.Label(ds_card, textvariable=self.kg_search_status_var, font=("Segoe UI", 9, "italic")).pack(anchor="w", pady=(0, 4))
        ttk.Button(ds_card, text="Download Selected Dataset", command=self._kaggle_download_action).pack(anchor="e")

        # Hosted Batch Curation
        cur_card = ttk.LabelFrame(right_kg, text="Run Private Dual-T4 GPU Curation (Kaggle Kernel)", padding=10)
        cur_card.pack(fill="both", expand=True)

        f_grid = ttk.Frame(cur_card)
        f_grid.pack(fill="x", pady=(0, 6))

        ttk.Label(f_grid, text="Kaggle Username:").grid(row=0, column=0, sticky="w", pady=2)
        self.cur_user_var = tk.StringVar()
        ttk.Entry(f_grid, textvariable=self.cur_user_var).grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(f_grid, text="Dataset Ref (owner/slug):").grid(row=1, column=0, sticky="w", pady=2)
        self.cur_ref_var = tk.StringVar()
        ttk.Entry(f_grid, textvariable=self.cur_ref_var).grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(f_grid, text="Prompt Field:").grid(row=2, column=0, sticky="w", pady=2)
        self.cur_prompt_var = tk.StringVar(value="instruction")
        ttk.Entry(f_grid, textvariable=self.cur_prompt_var).grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Label(f_grid, text="Response Field:").grid(row=3, column=0, sticky="w", pady=2)
        self.cur_response_var = tk.StringVar(value="completion")
        ttk.Entry(f_grid, textvariable=self.cur_response_var).grid(row=3, column=1, sticky="ew", pady=2)

        ttk.Label(f_grid, text="Model Handle:").grid(row=4, column=0, sticky="w", pady=2)
        self.cur_model_var = tk.StringVar(value="qwen-lm/qwen2-5-coder/Transformers/14b-instruct-gptq-int4/1")
        ttk.Entry(f_grid, textvariable=self.cur_model_var).grid(row=4, column=1, sticky="ew", pady=2)

        ttk.Label(f_grid, text="Curation Mode:").grid(row=5, column=0, sticky="w", pady=2)
        self.cur_mode_var = tk.StringVar(value="answer")
        ttk.Combobox(f_grid, textvariable=self.cur_mode_var, values=("answer", "rewrite"), state="readonly", width=12).grid(
            row=5, column=1, sticky="w", pady=2
        )
        f_grid.columnconfigure(1, weight=1)

        self.cur_remote_confirm = tk.BooleanVar(value=False)
        self.cur_upload_confirm = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            cur_card,
            text="I authorize sending selected dataset to Kaggle for private GPU processing.",
            variable=self.cur_remote_confirm,
        ).pack(anchor="w", pady=(4, 2))
        ttk.Checkbutton(
            cur_card,
            text="I authorize uploading a private Kaggle dataset copy if local input is selected.",
            variable=self.cur_upload_confirm,
        ).pack(anchor="w", pady=(0, 6))

        self.cur_status_var = tk.StringVar(value="Configured for Kaggle GPU T4 x2 execution.")
        ttk.Label(cur_card, textvariable=self.cur_status_var, font=("Segoe UI", 9, "italic")).pack(anchor="w", pady=(0, 4))
        ttk.Button(cur_card, text="Start Private Curation Run", command=self._kaggle_curate_action).pack(anchor="e")

        # Automated Dual-T4 Notebook Runner Card
        nb_card = ttk.LabelFrame(right_kg, text="Automated Dual-T4 Batch Notebook (kaggle_batch_inference.ipynb)", padding=10)
        nb_card.pack(fill="x", pady=(10, 0))

        ttk.Label(
            nb_card,
            text="Automatically push kaggle_batch_inference.ipynb to your Kaggle account and execute across dual T4 GPUs using Kaggle CLI.",
            wraplength=480,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(0, 6))

        preset_row = ttk.Frame(nb_card)
        preset_row.pack(fill="x", pady=(2, 6))
        ttk.Label(preset_row, text="Language Preset:").pack(side="left")
        self.nb_lang_preset_var = tk.StringVar(value="Python")
        ttk.Combobox(
            preset_row,
            textvariable=self.nb_lang_preset_var,
            values=("Python", "COBOL", "SQL", "CUDA", "HCL"),
            state="readonly",
            width=12,
        ).pack(side="left", padx=(6, 12))

        nb_btn_row = ttk.Frame(nb_card)
        nb_btn_row.pack(fill="x")
        ttk.Button(nb_btn_row, text="Push & Run Notebook on Kaggle 🚀", command=self._push_batch_notebook_action).pack(side="left", padx=(0, 6))
        ttk.Button(nb_btn_row, text="Check Status 🔄", command=self._check_kernel_status_action).pack(side="left", padx=(0, 6))
        ttk.Button(nb_btn_row, text="Download Outputs 📥", command=self._download_kernel_outputs_action).pack(side="left")

        self.nb_runner_status_var = tk.StringVar(value="Requires Kaggle authentication. Pushes private notebook configured for T4 x2.")
        ttk.Label(nb_card, textvariable=self.nb_runner_status_var, font=("Segoe UI", 9, "italic")).pack(anchor="w", pady=(6, 0))

    def _check_kaggle_connection_silent(self) -> None:
        def worker() -> None:
            try:
                from .kaggle import get_kaggle_auth_status, kaggle_config_dir

                status = get_kaggle_auth_status(probe_cli=False)
                self.root.after(0, self._render_kaggle_status, status)
            except Exception:
                pass

        threading.Thread(target=worker, daemon=True).start()

    def _check_kaggle_connection_manual(self) -> None:
        self.kaggle_auth_label.configure(text="Status: Probing Kaggle CLI…")

        def worker() -> None:
            try:
                from .kaggle import get_kaggle_auth_status

                status = get_kaggle_auth_status(probe_cli=True)
                self.root.after(0, self._render_kaggle_status, status)
            except Exception as exc:
                self.root.after(0, self.kaggle_auth_label.configure, {"text": f"Status: Error - {exc}"})

        threading.Thread(target=worker, daemon=True).start()

    def _render_kaggle_status(self, status: dict[str, Any]) -> None:
        cfg_dir = status.get("config_dir", "")
        self.kg_config_path_var.set(f"Credentials Directory: {cfg_dir}")

        if status.get("authenticated"):
            user = status.get("username") or "Authenticated"
            txt = f"Status: 🟢 Connected ({user})"
            pill_txt = f"Kaggle: 🟢 {user}"
            if status.get("username"):
                self.cur_user_var.set(status["username"])
        elif status.get("has_credentials"):
            user = status.get("username") or "Saved Credentials"
            txt = f"Status: 🟡 Credentials Found ({user}) — Ready to verify"
            pill_txt = f"Kaggle: 🟡 {user}"
            if status.get("username"):
                self.cur_user_var.set(status["username"])
        else:
            txt = "Status: ⚪ Not Connected / No Credentials"
            pill_txt = "Kaggle: ⚪ Not Logged In"

        self.kaggle_auth_label.configure(text=txt)
        self.kaggle_status_pill.configure(text=pill_txt)

    def _save_direct_kaggle_creds(self) -> None:
        user = self.kg_user_var.get().strip()
        key = self.kg_key_var.get().strip()
        if not user or not key:
            messagebox.showerror("Missing Information", "Enter both Kaggle Username and API Key.", parent=self.root)
            return
        try:
            from .kaggle import save_kaggle_credentials

            cfg = save_kaggle_credentials(username=user, key=key)
            messagebox.showinfo("Saved", f"Kaggle credentials saved to:\n{cfg / 'kaggle.json'}", parent=self.root)
            self._check_kaggle_connection_manual()
        except Exception as exc:
            messagebox.showerror("Error Saving Credentials", str(exc), parent=self.root)

    def _import_kaggle_json(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select kaggle.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=self.root,
        )
        if not file_path:
            return
        try:
            content = Path(file_path).read_text(encoding="utf-8")
            from .kaggle import save_kaggle_credentials

            cfg = save_kaggle_credentials(json_content=content)
            messagebox.showinfo("Imported", f"Kaggle credentials imported successfully into:\n{cfg}", parent=self.root)
            self._check_kaggle_connection_manual()
        except Exception as exc:
            messagebox.showerror("Import Failed", str(exc), parent=self.root)

    def _kaggle_search_action(self) -> None:
        query = self.kg_query_var.get().strip()
        if not query:
            return
        self.kg_search_status_var.set("Searching Kaggle datasets…")
        self.kg_results_list.delete(0, "end")

        def worker() -> None:
            try:
                from .kaggle import search_datasets

                res = search_datasets(query)
                self.root.after(0, self._render_kaggle_search_results, res.get("datasets", []))
            except Exception as exc:
                self.root.after(0, self.kg_search_status_var.set, f"Search failed: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _render_kaggle_search_results(self, datasets: list[dict[str, Any]]) -> None:
        self.kg_results_list.delete(0, "end")
        refs = []
        for d in datasets:
            ref = d.get("ref", "")
            title = d.get("title", ref)
            size = d.get("totalBytes", 0)
            size_mb = f"{size / (1024*1024):.1f} MB" if size else ""
            refs.append(ref)
            self.kg_results_list.insert("end", f"{ref} — {title} ({size_mb})")
        self.kg_results_list.result_refs = refs  # type: ignore[attr-defined]
        self.kg_search_status_var.set(f"Found {len(datasets)} dataset(s). Select one to download.")

    def _kaggle_download_action(self) -> None:
        sel = self.kg_results_list.curselection()
        if not sel or not hasattr(self.kg_results_list, "result_refs"):
            messagebox.showwarning("Select Dataset", "Please select a dataset from the list.", parent=self.root)
            return
        ref = self.kg_results_list.result_refs[sel[0]]  # type: ignore[attr-defined]
        dest = self.kg_download_dir_var.get().strip()
        if not dest:
            messagebox.showerror("Missing Destination", "Specify a destination download directory.", parent=self.root)
            return

        self.kg_search_status_var.set(f"Downloading {ref}…")

        def worker() -> None:
            try:
                from .kaggle import download_dataset

                res = download_dataset(ref, dest)
                self.root.after(0, self.kg_search_status_var.set, f"Downloaded {ref}: {len(res.get('files', []))} files ready.")
                self.root.after(0, messagebox.showinfo, "Download Complete", f"Downloaded {ref} to:\n{dest}")
            except Exception as exc:
                self.root.after(0, self.kg_search_status_var.set, f"Download failed: {exc}")
                self.root.after(0, messagebox.showerror, "Download Failed", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _kaggle_curate_action(self) -> None:
        if not self.cur_remote_confirm.get():
            messagebox.showerror("Authorization Required", "You must authorize remote Kaggle GPU processing.", parent=self.root)
            return
        user = self.cur_user_var.get().strip()
        dataset_ref = self.cur_ref_var.get().strip()
        if not user or not dataset_ref:
            messagebox.showerror("Missing Fields", "Enter both Kaggle Username and Dataset Ref (owner/slug).", parent=self.root)
            return

        self.cur_status_var.set("Launching private Kaggle GPU curation kernel…")

        def worker() -> None:
            try:
                from .kaggle import run_curation

                res = run_curation(
                    dataset_ref=dataset_ref,
                    kaggle_username=user,
                    prompt_field=self.cur_prompt_var.get().strip(),
                    response_field=self.cur_response_var.get().strip(),
                    model_handle=self.cur_model_var.get().strip(),
                    mode=self.cur_mode_var.get().strip(),
                    output_dir=Path(self.output_var.get()) / "kaggle-curated",
                )
                self.root.after(0, self.cur_status_var.set, f"Curation complete: {res.get('candidate_records', 0)} candidate rows.")
                self.root.after(
                    0,
                    messagebox.showinfo,
                    "Curation Finished",
                    f"Generated unapproved candidates saved to:\n{res.get('local_candidate_file')}",
                )
            except Exception as exc:
                self.root.after(0, self.cur_status_var.set, f"Curation failed: {exc}")
                self.root.after(0, messagebox.showerror, "Kaggle Curation Error", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _push_batch_notebook_action(self) -> None:
        user = self.cur_user_var.get().strip() or self.kg_user_var.get().strip()
        if not user:
            messagebox.showerror("Username Required", "Enter your Kaggle Username first.", parent=self.root)
            return

        msg = (
            f"Push and launch kaggle_batch_inference.ipynb on Kaggle?\n\n"
            f"• Account: {user}\n"
            f"• Kernel: {user}/datasetcleaner-batch-inference\n"
            f"• Machine Shape: NvidiaTeslaT4x2 (Dual T4 GPUs)\n"
            f"• Privacy: Private Kernel\n\n"
            f"The notebook will be pushed and started on Kaggle automatically."
        )
        if not messagebox.askyesno("Confirm Kaggle Push", msg, parent=self.root):
            return

        self.nb_runner_status_var.set("Pushing kaggle_batch_inference.ipynb to Kaggle…")

        def worker() -> None:
            try:
                from .kaggle import push_batch_notebook

                lang = self.nb_lang_preset_var.get().lower() if hasattr(self, "nb_lang_preset_var") else None
                res = push_batch_notebook(
                    kaggle_username=user,
                    language=lang,
                    output_dir=Path(self.output_var.get()) / "kaggle-runs",
                )
                self.root.after(
                    0,
                    self.nb_runner_status_var.set,
                    f"Pushed: {res['kernel_ref']} | URL: {res['url']}",
                )
                self.root.after(
                    0,
                    messagebox.showinfo,
                    "Notebook Launched on Kaggle",
                    f"Notebook successfully pushed and started on Kaggle:\n\n{res['url']}\n\nExecution has begun on dual T4 GPUs.",
                )
            except Exception as exc:
                self.root.after(0, self.nb_runner_status_var.set, f"Push failed: {exc}")
                self.root.after(0, messagebox.showerror, "Kaggle Notebook Push Failed", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _check_kernel_status_action(self) -> None:
        user = self.cur_user_var.get().strip() or self.kg_user_var.get().strip()
        if not user:
            messagebox.showerror("Username Required", "Enter your Kaggle Username first.", parent=self.root)
            return
        kernel_ref = f"{user}/datasetcleaner-batch-inference"
        self.nb_runner_status_var.set(f"Checking status of {kernel_ref}…")

        def worker() -> None:
            try:
                from .kaggle import get_kernel_status

                status = get_kernel_status(kernel_ref)
                self.root.after(0, self.nb_runner_status_var.set, f"{kernel_ref}: {status.strip()}")
                self.root.after(0, messagebox.showinfo, "Kernel Status", f"Status for {kernel_ref}:\n\n{status}")
            except Exception as exc:
                self.root.after(0, self.nb_runner_status_var.set, f"Status check failed: {exc}")
                self.root.after(0, messagebox.showerror, "Status Check Failed", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _download_kernel_outputs_action(self) -> None:
        user = self.cur_user_var.get().strip() or self.kg_user_var.get().strip()
        if not user:
            messagebox.showerror("Username Required", "Enter your Kaggle Username first.", parent=self.root)
            return
        kernel_ref = f"{user}/datasetcleaner-batch-inference"
        dest = Path(self.output_var.get()) / "kaggle-downloads" / "datasetcleaner-batch-inference"
        self.nb_runner_status_var.set(f"Downloading outputs for {kernel_ref}…")

        def worker() -> None:
            try:
                from .kaggle import download_kernel_outputs

                files = download_kernel_outputs(kernel_ref, dest)
                self.root.after(0, self.nb_runner_status_var.set, f"Downloaded {len(files)} files to {dest}")
                self.root.after(
                    0,
                    messagebox.showinfo,
                    "Download Complete",
                    f"Downloaded {len(files)} output files for {kernel_ref} into:\n{dest}\n\nFiles: {', '.join(files)}",
                )
            except Exception as exc:
                self.root.after(0, self.nb_runner_status_var.set, f"Download failed: {exc}")
                self.root.after(0, messagebox.showerror, "Download Failed", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    # -------------------------------------------------------------------------
    # TAB 5: GOOGLE DRIVE & HUGGING FACE
    # -------------------------------------------------------------------------
    def _build_drive_tab(self) -> None:
        ttk.Label(self.drive_tab, text="Google Drive & Hugging Face Cloud Storage", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            self.drive_tab,
            text="Stage local datasets in Google Drive, stream Hugging Face datasets into Drive, and connect with Kaggle.",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(0, 14))

        # Drive Staging Card
        drv_card = ttk.LabelFrame(self.drive_tab, text="Stage Local Dataset into Google Drive", padding=12)
        drv_card.pack(fill="x", pady=(0, 14))

        p_row = ttk.Frame(drv_card)
        p_row.pack(fill="x", pady=(0, 6))
        ttk.Label(p_row, text="Local File:").pack(side="left")
        self.drive_local_path_var = tk.StringVar(value=self._selected_local_file())
        ttk.Entry(p_row, textvariable=self.drive_local_path_var).pack(side="left", fill="x", expand=True, padx=(8, 4))
        ttk.Button(p_row, text="Browse…", command=lambda: self._choose_dataset_file(self.drive_local_path_var)).pack(side="left")

        f_row = ttk.Frame(drv_card)
        f_row.pack(fill="x", pady=(0, 6))
        ttk.Label(f_row, text="Destination Folder ID:").pack(side="left")
        self.drive_folder_id_var = tk.StringVar()
        ttk.Entry(f_row, textvariable=self.drive_folder_id_var).pack(side="left", fill="x", expand=True, padx=(8, 0))

        self.drive_confirm_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            drv_card,
            text="I authorize copying this dataset file to Google Drive.",
            variable=self.drive_confirm_var,
        ).pack(anchor="w", pady=(2, 6))

        self.drive_status_var = tk.StringVar(
            value="Requires GOOGLE_DRIVE_TOKEN_JSON in the process environment or credentials file."
        )
        ttk.Label(drv_card, textvariable=self.drive_status_var, font=("Segoe UI", 9, "italic")).pack(anchor="w", pady=(0, 6))

        ttk.Button(drv_card, text="Upload File to Google Drive in Chunks", command=self._drive_upload_action).pack(anchor="e")

        # Hugging Face Direct Storage Downloader Card
        hf_card = ttk.LabelFrame(self.drive_tab, text="Hugging Face → Google Drive Direct Downloader", padding=12)
        hf_card.pack(fill="both", expand=True)

        ttk.Label(
            hf_card,
            text="Download full Hugging Face datasets directly to Google Drive without filling your local computer storage.",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            hf_card,
            text="DataSetCleaner includes a dedicated client tool for direct Hugging Face to Google Drive transfers.\n"
            "Once saved in Google Drive, large multi-shard datasets can be streamed directly into Kaggle dual-T4 GPUs.",
            wraplength=850,
        ).pack(anchor="w", pady=(4, 12))

        btn_row = ttk.Frame(hf_card)
        btn_row.pack(anchor="w")
        ttk.Button(btn_row, text="Open HF → Drive Downloader Tool ↗", command=self._open_hf_drive_tool).pack(side="left", padx=(0, 10))
        ttk.Button(btn_row, text="Open Kaggle Batch Notebook ↗", command=self._open_kaggle_notebook).pack(side="left")

    def _drive_upload_action(self) -> None:
        source = self.drive_local_path_var.get().strip()
        folder_id = self.drive_folder_id_var.get().strip()
        confirmed = self.drive_confirm_var.get()
        if not confirmed:
            messagebox.showerror("Confirmation Required", "Authorize Google Drive copy before uploading.", parent=self.root)
            return
        if not source or not folder_id:
            messagebox.showerror("Missing Information", "Select a local dataset file and enter folder ID.", parent=self.root)
            return
        self.drive_status_var.set("Uploading to Google Drive in resumable chunks…")

        def worker() -> None:
            try:
                from .drive_upload import upload_file

                res = upload_file(source, folder_id)
                self.root.after(0, self.drive_status_var.set, f"Upload complete: {res['name']} (ID: {res['id']})")
                self.root.after(
                    0,
                    messagebox.showinfo,
                    "Upload Finished",
                    f"File successfully uploaded to Google Drive:\n{res['name']}\nID: {res['id']}",
                )
            except Exception as exc:
                self.root.after(0, self.drive_status_var.set, f"Upload failed: {exc}")
                self.root.after(0, messagebox.showerror, "Drive Upload Error", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _open_hf_drive_tool(self) -> None:
        html_file = Path(__file__).resolve().parent.parent / "hf_drive_hugging_face_to_google_drive_downloader.html"
        if not html_file.is_file():
            messagebox.showerror("Tool Not Found", f"Could not find {html_file}", parent=self.root)
            return
        webbrowser.open_new_tab(html_file.as_uri())

    # -------------------------------------------------------------------------
    # DATASET CURATION & EVENT HANDLERS (Preserved Contracts)
    # -------------------------------------------------------------------------
    def _add_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="Select dataset files",
            filetypes=[("Supported datasets", "*.csv *.jsonl *.ndjson *.parquet"), ("All files", "*.*")],
            parent=self.root,
        )
        if files:
            self._add_sources([SourceSpec(Path(f), self.license_var.get().strip(), self.source_description_var.get().strip()) for f in files])

    def _add_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select folder containing code files", parent=self.root)
        if folder:
            self._add_sources([SourceSpec(Path(folder), self.license_var.get().strip(), self.source_description_var.get().strip())])

    def _add_sources(self, specs: list[SourceSpec]) -> None:
        for spec in specs:
            self.sources.append(spec)
            desc = f" [{spec.license_id}]" if spec.license_id else ""
            self.source_list.insert("end", f"{spec.path.name}{desc} — {spec.path}")
        self._invalidate_session()
        self.status_var.set(f"Loaded {len(self.sources)} source(s). Click Analyze & Audit Inputs to review.")
        if len(self.sources) == 1 and hasattr(self, "drive_local_path_var"):
            self.drive_local_path_var.set(str(self.sources[0].path))

    def _remove_source(self) -> None:
        sel = self.source_list.curselection()
        if sel:
            idx = sel[0]
            self.source_list.delete(idx)
            del self.sources[idx]
            self._invalidate_session()
            self.status_var.set("Source removed.")

    def _source_selected(self, _: Any) -> None:
        sel = self.source_list.curselection()
        if sel:
            spec = self.sources[sel[0]]
            self.license_var.set(spec.license_id)
            self.source_description_var.set(spec.description)
            if hasattr(self, "drive_local_path_var") and spec.path.is_file():
                self.drive_local_path_var.set(str(spec.path))

    def _save_source_metadata(self) -> None:
        sel = self.source_list.curselection()
        if sel:
            idx = sel[0]
            current = self.sources[idx]
            self.sources[idx] = SourceSpec(current.path, self.license_var.get().strip(), self.source_description_var.get().strip())
            desc = f" [{self.sources[idx].license_id}]" if self.sources[idx].license_id else ""
            self.source_list.delete(idx)
            self.source_list.insert(idx, f"{self.sources[idx].path.name}{desc} — {self.sources[idx].path}")
            self.source_list.selection_set(idx)
            self._invalidate_session()
            self.status_var.set("Saved source metadata.")

    def _settings_changed(self, *_: Any) -> None:
        if self.session is not None:
            self._invalidate_session()

    def _invalidate_session(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None
        self.summary = {}
        self.page_offset = 0
        self.table.delete(*self.table.get_children())
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.configure(state="disabled")
        self.page_var.set("No review rows")
        self.prev_button.configure(state="disabled")
        self.next_button.configure(state="disabled")
        if hasattr(self, "stat_records"):
            for lbl in (self.stat_records, self.stat_valid, self.stat_dups, self.stat_syntax, self.stat_license):
                lbl.configure(text="0")

    def _start_analysis(self) -> None:
        if not self.sources:
            messagebox.showwarning("No sources", "Add at least one input file or code folder.", parent=self.root)
            return
        if self.busy:
            return
        self.busy = True
        self.status_var.set("Analyzing dataset inputs locally…")
        threading.Thread(target=self._analyze_worker, daemon=True).start()

    def _analyze_worker(self) -> None:
        try:
            field_map_text = self.field_map_var.get().strip()
            field_map = json.loads(field_map_text) if field_map_text else {}
            fields = [f.strip() for f in self.fields_var.get().split(",") if f.strip()]
            threshold = float(self.threshold_var.get().strip() or "0.90")
            detect_near = self.detect_near_var.get()

            appdata = os.environ.get("LOCALAPPDATA") or tempfile.gettempdir()
            store_root = Path(appdata) / "DataSetCleaner" / "work"
            store_root.mkdir(parents=True, exist_ok=True)
            temp = tempfile.TemporaryDirectory(dir=store_root)

            session = RunSession(Path(temp.name))
            summary = session.analyze(
                self.sources,
                field_map=field_map,
                preferred_fields=fields or None,
                threshold=threshold,
                detect_near=detect_near,
            )
            if self.detect_semantic_var.get():
                session.detect_semantic_duplicates()
                summary = session.summary()
            self.root.after(0, self._analysis_finished, temp, session, summary)
        except Exception as exc:
            self.root.after(0, self._analysis_failed, str(exc))

    def _analysis_finished(self, temp: tempfile.TemporaryDirectory[str], session: RunSession, summary: dict[str, Any]) -> None:
        self.temp_dir = temp
        self.session = session
        self.summary = summary
        self.busy = False
        self._render_summary()
        self.page_offset = 0
        self._render_rows()
        self.status_var.set(f"Analysis complete: {summary['records']:,} rows in {summary['elapsed_seconds']:.2f} s. Review findings below.")

    def _analysis_failed(self, message: str) -> None:
        self.busy = False
        self.status_var.set("Analysis failed.")
        messagebox.showerror("Analysis failed", message, parent=self.root)

    def _render_summary(self) -> None:
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        text = json.dumps(self.summary, indent=2)
        self.summary_text.insert("1.0", text)
        self.summary_text.configure(state="disabled")

        if hasattr(self, "stat_records"):
            recs = self.summary.get("records", 0)
            issues = self.summary.get("issues", {})
            dups = issues.get("duplicate_exact", 0) + issues.get("duplicate_near", 0)
            syntax = issues.get("syntax", 0) + issues.get("syntax_unchecked", 0)
            lic = len(self.summary.get("license_warnings", []))
            valid = recs - dups - issues.get("malformed_json", 0)

            self.stat_records.configure(text=f"{recs:,}")
            self.stat_valid.configure(text=f"{valid:,}")
            self.stat_dups.configure(text=f"{dups:,}")
            self.stat_syntax.configure(text=f"{syntax:,}")
            self.stat_license.configure(text=f"{lic}")

    def _render_rows(self) -> None:
        self.table.delete(*self.table.get_children())
        if self.session is None:
            return
        rows = self.session.list_records(limit=self.page_size, offset=self.page_offset)
        for row in rows:
            inc = "✓ Yes" if row.get("included") or row.get("include_row") else "✗ No"
            issues_summary = ", ".join(row["issues"]) if row.get("issues") else "clean"
            data = row.get("data") or {}
            prov = data.get("generation_provenance") or data.get("synthetic_origin") or "original"
            source_preview = f"{row.get('source_file') or ''} #{row.get('source_row') or ''}"
            preview = json.dumps(data, ensure_ascii=False)
            if len(preview) > 180:
                preview = preview[:177] + "…"
            self.table.insert("", "end", iid=str(row["id"]), values=(inc, row["id"], issues_summary, prov, source_preview, preview))

        total = self.summary.get("records", 0)
        from_idx = self.page_offset + 1 if total else 0
        to_idx = min(self.page_offset + len(rows), total)
        self.page_var.set(f"Rows {from_idx:,} – {to_idx:,} of {total:,}")
        self.prev_button.configure(state="normal" if self.page_offset > 0 else "disabled")
        self.next_button.configure(state="normal" if to_idx < total else "disabled")

    def _toggle_include(self, _: Any = None) -> None:
        sel = self.table.selection()
        if not sel or self.session is None:
            return
        row_id = int(sel[0])
        new_state = self.session.toggle_included(row_id)
        current = list(self.table.item(str(row_id), "values"))
        current[0] = "✓ Yes" if new_state else "✗ No"
        self.table.item(str(row_id), values=current)

    def _change_page(self, delta: int) -> None:
        if self.session is None:
            return
        self.page_offset = max(0, self.page_offset + delta)
        self._render_rows()

    def _choose_output(self) -> None:
        folder = filedialog.askdirectory(title="Choose output export folder", parent=self.root)
        if folder:
            self.output_var.set(folder)

    def _choose_dataset_file(self, variable: tk.StringVar) -> None:
        selected = filedialog.askopenfilename(
            title="Choose dataset file",
            filetypes=[("Dataset files", "*.csv *.jsonl *.ndjson *.parquet"), ("All files", "*.*")],
            parent=self.root,
        )
        if selected:
            variable.set(selected)

    def _choose_folder(self, variable: tk.StringVar) -> None:
        selected = filedialog.askdirectory(title="Choose folder", parent=self.root)
        if selected:
            variable.set(selected)

    def _selected_local_file(self) -> str:
        if hasattr(self, "source_list"):
            selection = self.source_list.curselection()
            if selection and self.sources[selection[0]].path.is_file():
                return str(self.sources[selection[0]].path)
        if len(self.sources) == 1 and self.sources[0].path.is_file():
            return str(self.sources[0].path)
        return ""

    def _export(self) -> None:
        if self.session is None:
            messagebox.showinfo("Analyze required", "Run analysis before exporting.", parent=self.root)
            return
        out_dir = Path(self.output_var.get().strip() or "datasetcleaner-output")
        fmt = self.format_var.get()
        name = self.name_var.get().strip() or "dataset"
        data_path = unique_output_path(out_dir, name, fmt)

        msg = (
            f"Review export parameters:\n\n"
            f"• Output format: {fmt}\n"
            f"• Output file: {data_path}\n"
            f"• Exclude exact duplicates: {self.exclude_exact_var.get()}\n"
            f"• Exclude near duplicates: {self.exclude_near_var.get()}\n"
            f"• Exclude incomplete rows: {self.exclude_incomplete_var.get()}\n"
            f"• Exclude malformed JSON: {self.exclude_malformed_var.get()}\n\n"
            f"Do you approve this reviewed cleanup and export?"
        )
        if not messagebox.askyesno("Approve Export", msg, parent=self.root):
            return

        try:
            result = self.session.export(
                out_dir,
                fmt=fmt,
                name=name,
                description=self.dataset_description_var.get().strip(),
                exclude_exact=self.exclude_exact_var.get(),
                exclude_near=self.exclude_near_var.get(),
                exclude_incomplete=self.exclude_incomplete_var.get(),
                exclude_malformed=self.exclude_malformed_var.get(),
            )
            self.status_var.set(f"Export complete: {result['exported_records']:,} rows written.")
            messagebox.showinfo("Export complete", json.dumps(result, indent=2), parent=self.root)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc), parent=self.root)

    def _rewrite_selected(self) -> None:
        sel = self.table.selection()
        if not sel or self.session is None:
            messagebox.showinfo("Select a row", "Select an analyzed row from the table first.", parent=self.root)
            return
        if self.rewrite_busy:
            return
        row_id = int(sel[0])
        self.rewrite_busy = True
        self.status_var.set(f"Requesting local model rewrite proposal for row #{row_id}…")
        threading.Thread(target=self._rewrite_worker, args=(row_id,), daemon=True).start()

    def _rewrite_worker(self, row_id: int) -> None:
        try:
            assert self.session is not None
            out_dir = Path(self.output_var.get().strip() or "datasetcleaner-output")
            proposal = self.session.propose_rewrite(row_id, output_dir=out_dir)
            self.root.after(0, self._show_rewrite, proposal)
        except Exception as exc:
            self.root.after(0, self._rewrite_failed, str(exc))

    def _show_rewrite(self, proposal: dict[str, Any]) -> None:
        self.rewrite_busy = False
        self.status_var.set("Rewrite proposal generated.")
        messagebox.showinfo(
            "Rewrite Proposal",
            f"Proposal saved to separate file (does not alter dataset export):\n\n{json.dumps(proposal, indent=2)}",
            parent=self.root,
        )

    def _rewrite_failed(self, message: str) -> None:
        self.rewrite_busy = False
        self.status_var.set("Rewrite proposal failed.")
        messagebox.showerror("Rewrite proposal failed", message, parent=self.root)

    def _open_kaggle_notebook(self) -> None:
        notebook = Path(__file__).resolve().parent.parent / "kaggle_batch_inference.ipynb"
        if not notebook.is_file():
            messagebox.showerror("Notebook Not Found", f"Could not find {notebook}", parent=self.root)
            return
        try:
            code = shutil.which("code")
            if code:
                subprocess.Popen([code, str(notebook)], shell=False)
            elif os.name == "nt":
                os.startfile(str(notebook))  # type: ignore[attr-defined]
            else:
                messagebox.showinfo("Open notebook", f"Open notebook at:\n{notebook}", parent=self.root)
        except OSError as exc:
            messagebox.showerror("Could not open notebook", str(exc), parent=self.root)

    def _kaggle_login_from_dialog(self, status: ttk.Label | tk.StringVar) -> None:
        from .kaggle import kaggle_cli_executable

        executable = kaggle_cli_executable()
        if not executable:
            messagebox.showerror(
                "Kaggle CLI Unavailable",
                "Install with `python -m pip install '.[kaggle]'` to enable Kaggle features.",
                parent=self.root,
            )
            return

        def worker() -> None:
            try:
                from .kaggle import kaggle_cli_environment, prepare_kaggle_config_dir

                cfg = prepare_kaggle_config_dir()
                env = kaggle_cli_environment(cfg)
                flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
                proc = subprocess.Popen([executable, "auth", "login"], shell=False, creationflags=flags, env=env)
                self.root.after(
                    0,
                    messagebox.showinfo,
                    "Kaggle Sign-in Console Launched",
                    f"A console window opened for Kaggle sign-in. Complete authentication in browser/console.\nCredentials use:\n{cfg}",
                )
            except Exception as exc:
                self.root.after(0, messagebox.showerror, "Kaggle Sign-in Error", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    # -------------------------------------------------------------------------
    # TAB 6: DATASET BENCHMARK & ENTERPRISE RELEASE
    # -------------------------------------------------------------------------
    def _build_benchmark_tab(self) -> None:
        ttk.Label(self.benchmark_tab, text="Dataset Benchmark & Enterprise Release Packager", style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            self.benchmark_tab,
            text="Compute rigorous Parse, Compile, Oracle Pass, and Instruction Coverage rates without fine-tuning, and package verified distributions.",
            style="Subheader.TLabel",
        ).pack(anchor="w", pady=(0, 10))

        # Split: Left (Dataset Benchmark Runner) | Right (Release Packager)
        panes = ttk.Panedwindow(self.benchmark_tab, orient="horizontal")
        panes.pack(fill="both", expand=True)

        left = ttk.Frame(panes, padding=(0, 0, 8, 0))
        right = ttk.Frame(panes, padding=(8, 0, 0, 0))
        panes.add(left, weight=1)
        panes.add(right, weight=1)

        # 1. Dataset Benchmark Runner
        bench_card = ttk.LabelFrame(left, text="Dataset-Only Quality Benchmark Gate", padding=10)
        bench_card.pack(fill="both", expand=True)

        ttk.Label(bench_card, text="Target Dataset File (.jsonl, .csv, .parquet):").pack(anchor="w")
        ds_row = ttk.Frame(bench_card)
        ds_row.pack(fill="x", pady=(2, 6))
        self.bm_dataset_path_var = tk.StringVar()
        ttk.Entry(ds_row, textvariable=self.bm_dataset_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(ds_row, text="Browse…", command=self._choose_benchmark_dataset).pack(side="left", padx=(4, 0))

        cfg_row = ttk.Frame(bench_card)
        cfg_row.pack(fill="x", pady=(2, 6))
        ttk.Label(cfg_row, text="Target Language:").pack(side="left")
        self.bm_language_var = tk.StringVar(value="Python")
        ttk.Combobox(
            cfg_row,
            textvariable=self.bm_language_var,
            values=("Python", "COBOL", "SQL", "CUDA", "HCL"),
            state="readonly",
            width=12,
        ).pack(side="left", padx=(6, 12))

        ttk.Label(bench_card, text="Output Directory:").pack(anchor="w")
        out_row = ttk.Frame(bench_card)
        out_row.pack(fill="x", pady=(2, 6))
        self.bm_output_dir_var = tk.StringVar(value=str(Path.cwd() / "benchmark_output"))
        ttk.Entry(out_row, textvariable=self.bm_output_dir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(out_row, text="Browse…", command=lambda: self._choose_folder(self.bm_output_dir_var)).pack(side="left", padx=(4, 0))

        ttk.Button(bench_card, text="Run Dataset Benchmark 📊", command=self._run_benchmark_action).pack(anchor="w", pady=(6, 8))

        # KPI Stats Grid
        self.bm_stats_frame = ttk.Frame(bench_card, relief="groove", padding=8)
        self.bm_stats_frame.pack(fill="x", pady=(0, 8))
        self.bm_stat_parse = self._make_stat_box(self.bm_stats_frame, "Parse Rate", "0%", 0)
        self.bm_stat_compile = self._make_stat_box(self.bm_stats_frame, "Compile Rate", "0%", 1)
        self.bm_stat_oracle = self._make_stat_box(self.bm_stats_frame, "Oracle Pass", "0%", 2)
        self.bm_stat_coverage = self._make_stat_box(self.bm_stats_frame, "Coverage", "0%", 3)
        self.bm_stat_review = self._make_stat_box(self.bm_stats_frame, "Review Rate", "0%", 4)
        for i in range(5):
            self.bm_stats_frame.columnconfigure(i, weight=1)

        self.bm_status_var = tk.StringVar(value="Ready to benchmark dataset.")
        ttk.Label(bench_card, textvariable=self.bm_status_var, font=("Segoe UI", 9, "italic")).pack(anchor="w", pady=(4, 0))

        # 2. Release Packager Card
        pkg_card = ttk.LabelFrame(right, text="Enterprise 3-Tier Release Packager", padding=10)
        pkg_card.pack(fill="both", expand=True)

        ttk.Label(pkg_card, text="Raw Sources Directory:").pack(anchor="w")
        raw_row = ttk.Frame(pkg_card)
        raw_row.pack(fill="x", pady=(2, 6))
        self.pkg_raw_dir_var = tk.StringVar()
        ttk.Entry(raw_row, textvariable=self.pkg_raw_dir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(raw_row, text="Browse…", command=lambda: self._choose_folder(self.pkg_raw_dir_var)).pack(side="left", padx=(4, 0))

        ttk.Label(pkg_card, text="Curated SQLite DB (review.sqlite, optional):").pack(anchor="w")
        db_row = ttk.Frame(pkg_card)
        db_row.pack(fill="x", pady=(2, 6))
        self.pkg_db_path_var = tk.StringVar()
        ttk.Entry(db_row, textvariable=self.pkg_db_path_var).pack(side="left", fill="x", expand=True)
        ttk.Button(db_row, text="Browse…", command=self._choose_pkg_db).pack(side="left", padx=(4, 0))

        ttk.Label(pkg_card, text="Approved Directory (optional):").pack(anchor="w")
        app_row = ttk.Frame(pkg_card)
        app_row.pack(fill="x", pady=(2, 6))
        self.pkg_approved_dir_var = tk.StringVar()
        ttk.Entry(app_row, textvariable=self.pkg_approved_dir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(app_row, text="Browse…", command=lambda: self._choose_folder(self.pkg_approved_dir_var)).pack(side="left", padx=(4, 0))

        ttk.Label(pkg_card, text="Benchmark Artifacts Directory (optional):").pack(anchor="w")
        bm_art_row = ttk.Frame(pkg_card)
        bm_art_row.pack(fill="x", pady=(2, 6))
        self.pkg_benchmark_dir_var = tk.StringVar()
        ttk.Entry(bm_art_row, textvariable=self.pkg_benchmark_dir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(bm_art_row, text="Browse…", command=lambda: self._choose_folder(self.pkg_benchmark_dir_var)).pack(side="left", padx=(4, 0))

        ttk.Label(pkg_card, text="Releases Destination Directory:").pack(anchor="w")
        rel_row = ttk.Frame(pkg_card)
        rel_row.pack(fill="x", pady=(2, 6))
        self.pkg_output_dir_var = tk.StringVar(value=str(Path.cwd() / "releases"))
        ttk.Entry(rel_row, textvariable=self.pkg_output_dir_var).pack(side="left", fill="x", expand=True)
        ttk.Button(rel_row, text="Browse…", command=lambda: self._choose_folder(self.pkg_output_dir_var)).pack(side="left", padx=(4, 0))

        ttk.Button(pkg_card, text="Package Enterprise Release (manifest.sha256) 📦", command=self._package_release_action).pack(anchor="w", pady=(6, 8))

        self.pkg_status_var = tk.StringVar(value="Assembles releases/<language>/ hierarchy with root manifest.sha256.")
        ttk.Label(pkg_card, textvariable=self.pkg_status_var, font=("Segoe UI", 9, "italic")).pack(anchor="w", pady=(4, 0))

    def _choose_benchmark_dataset(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Dataset File to Benchmark",
            filetypes=[("Dataset files", "*.jsonl;*.csv;*.parquet;*.ndjson"), ("All files", "*.*")],
            parent=self.root,
        )
        if path:
            self.bm_dataset_path_var.set(path)

    def _choose_pkg_db(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Curated Database (review.sqlite)",
            filetypes=[("SQLite Database", "*.sqlite;*.db"), ("All files", "*.*")],
            parent=self.root,
        )
        if path:
            self.pkg_db_path_var.set(path)

    def _run_benchmark_action(self) -> None:
        ds_path = self.bm_dataset_path_var.get().strip()
        if not ds_path or not Path(ds_path).is_file():
            messagebox.showerror("Dataset Required", "Select an existing dataset file to benchmark.", parent=self.root)
            return
        lang = self.bm_language_var.get().strip().lower()
        out_dir = Path(self.bm_output_dir_var.get().strip() or "benchmark_output")
        self.bm_status_var.set(f"Running quality benchmark for {lang}…")

        def worker() -> None:
            try:
                from .benchmark import DatasetBenchmarkRunner
                runner = DatasetBenchmarkRunner(output_dir=out_dir)
                res = runner.benchmark_dataset(ds_path, lang)
                metrics = res.get("metrics", {})
                self.root.after(0, lambda: self.bm_stat_parse.configure(text=f"{metrics.get('parse_rate_pct', 0)}%"))
                self.root.after(0, lambda: self.bm_stat_compile.configure(text=f"{metrics.get('compile_rate_pct', 0)}%"))
                self.root.after(0, lambda: self.bm_stat_oracle.configure(text=f"{metrics.get('oracle_pass_rate_pct', 0)}%"))
                self.root.after(0, lambda: self.bm_stat_coverage.configure(text=f"{metrics.get('instruction_sufficiency_coverage_pct', 0)}%"))
                self.root.after(0, lambda: self.bm_stat_review.configure(text=f"{metrics.get('human_review_rate_pct', 0)}%"))
                self.root.after(0, self.bm_status_var.set, f"Benchmark complete! Audit report: {res.get('report_path')}")
                self.root.after(0, messagebox.showinfo, "Benchmark Complete", f"Benchmark completed successfully!\n\nParse Rate: {metrics.get('parse_rate_pct')}%\nCompile Rate: {metrics.get('compile_rate_pct')}%\nOracle Pass: {metrics.get('oracle_pass_rate_pct')}%\nReport: {res.get('report_path')}")
            except Exception as exc:
                self.root.after(0, self.bm_status_var.set, f"Benchmark failed: {exc}")
                self.root.after(0, messagebox.showerror, "Benchmark Error", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _package_release_action(self) -> None:
        raw_p = self.pkg_raw_dir_var.get().strip()
        if not raw_p or not Path(raw_p).exists():
            messagebox.showerror("Raw Sources Required", "Specify a valid raw sources folder or file.", parent=self.root)
            return
        lang = self.bm_language_var.get().strip().lower()
        out_root = Path(self.pkg_output_dir_var.get().strip() or "releases")
        db_p = Path(self.pkg_db_path_var.get().strip()) if self.pkg_db_path_var.get().strip() else None
        app_p = Path(self.pkg_approved_dir_var.get().strip()) if self.pkg_approved_dir_var.get().strip() else None
        bm_p = Path(self.pkg_benchmark_dir_var.get().strip()) if self.pkg_benchmark_dir_var.get().strip() else None

        self.pkg_status_var.set(f"Packaging {lang} release…")

        def worker() -> None:
            try:
                from .benchmark import ReleasePackager
                packager = ReleasePackager(output_dir=out_root)
                pkg_dir = packager.package_release(
                    language=lang,
                    raw_dir=raw_p,
                    db_path=db_p,
                    approved_dir=app_p,
                    benchmark_dir=bm_p,
                )
                self.root.after(0, self.pkg_status_var.set, f"Release packaged: {pkg_dir}")
                self.root.after(0, messagebox.showinfo, "Release Packaged", f"Enterprise dataset release assembled at:\n\n{pkg_dir}\n\nmanifest.sha256 generated.")
            except Exception as exc:
                self.root.after(0, self.pkg_status_var.set, f"Packaging failed: {exc}")
                self.root.after(0, messagebox.showerror, "Packaging Error", str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _open_kaggle_drive_dialog(self) -> None:
        """Switch directly to the Kaggle tab."""
        self.tabs.select(self.kaggle_tab)

    def _close(self) -> None:
        self._stop_agent_api()
        if self.session is not None:
            self.session.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="DataSetCleaner GUI")
    parser.add_argument("--tab", choices=["curation", "token", "agent", "kaggle", "drive", "benchmark"], default="curation")
    args, _ = parser.parse_known_args()
    root = tk.Tk()
    app = DataSetCleanerApp(root)
    if args.tab == "kaggle":
        app.tabs.select(app.kaggle_tab)
    elif args.tab == "token":
        app.tabs.select(app.token_tab)
    elif args.tab == "agent":
        app.tabs.select(app.agent_tab)
    elif args.tab == "drive":
        app.tabs.select(app.drive_tab)
    elif args.tab == "benchmark":
        app.tabs.select(app.benchmark_tab)
    root.mainloop()


if __name__ == "__main__":
    main()

"""MCP server that exposes review-safe DataSetCleaner operations to agents."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .agent_service import AgentService
from .kaggle import DEFAULT_MODEL_HANDLE, download_dataset, run_curation, search_datasets

INSTRUCTIONS = """DataSetCleaner analyzes and reviews local coding datasets. Source files are read-only. Use analyze_dataset, then wait for get_analysis_status to return completed before listing findings or records. Changes through set_dataset_record_included affect only the local review database. Generated examples are staged separately with provenance and approved=false. Kaggle model curation transfers rows to Kaggle for private remote GPU processing; call run_kaggle_model_curation only when the user explicitly requests that transfer and set confirm_remote_processing=true. For local input, also require user authorization to upload a private Kaggle copy and set confirm_local_upload=true. Always review quality findings and source rights. Do not call export_reviewed_dataset until the user explicitly authorizes export; exports never establish redistribution rights. Gemini/free cloud providers may receive data that the agent explicitly reads through these tools; use a local model for private data."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def build_server(service: AgentService | None = None) -> Any:
    try:
        from mcp.server import MCPServer
    except ImportError as exc:
        raise RuntimeError("The MCP integration requires the optional dependency. Install with: pip install '.[agents]'") from exc

    agent_service = service or AgentService()
    server = MCPServer("DataSetCleaner", instructions=INSTRUCTIONS)

    @server.tool()
    def analyze_dataset(
        sources: list[dict[str, str]],
        field_map: dict[str, str] | None = None,
        preferred_fields: list[str] | None = None,
        threshold: float = 0.9,
        detect_near: bool = False,
        check_syntax: bool = True,
    ) -> str:
        """Start a local dataset analysis. Each source has path and optional user-supplied license_id/description. Returns a run_id immediately; poll get_analysis_status."""
        return _json(agent_service.start_analysis(sources, field_map, preferred_fields, threshold, detect_near, check_syntax))

    @server.tool()
    def get_analysis_status(run_id: str) -> str:
        """Get status, scan progress, summary, and license warnings for an analysis run."""
        return _json(agent_service.get_run(run_id))

    @server.tool()
    def list_dataset_findings(run_id: str, offset: int = 0, limit: int = 50, code: str = "") -> str:
        """List a bounded page of quality findings. Optional code filters by finding type."""
        return _json(agent_service.list_findings(run_id, offset, limit, code))

    @server.tool()
    def get_dataset_records(run_id: str, offset: int = 0, limit: int = 25) -> str:
        """Read a bounded page of source records for inspection. Record contents become visible to this model/provider."""
        return _json(agent_service.get_records(run_id, offset, limit))

    @server.tool()
    def set_dataset_record_included(run_id: str, record_id: int, include: bool) -> str:
        """Set a reversible include/exclude choice in the local review database; never edits source files."""
        return _json(agent_service.set_record_included(run_id, record_id, include))

    @server.tool()
    def stage_generated_examples(run_id: str, examples: list[dict[str, Any]], provider: str, model: str) -> str:
        """Save up to 100 AI-generated examples per call as a separate local candidate JSONL with a provenance manifest and approved=false."""
        return _json(agent_service.stage_generated_examples(run_id, examples, provider, model))

    @server.tool()
    def list_generated_candidates(run_id: str) -> str:
        """List separate, unapproved JSONL candidates generated for this review run."""
        return _json(agent_service.list_candidates(run_id))

    @server.tool()
    def search_kaggle_datasets(query: str, limit: int = 20) -> str:
        """Search Kaggle datasets through the configured Kaggle CLI; requires Kaggle credentials."""
        return _json(search_datasets(query, limit))

    @server.tool()
    def download_kaggle_dataset(dataset_ref: str, output_dir: str) -> str:
        """Download a Kaggle dataset to a local folder. This writes downloaded files there and does not modify the Kaggle source."""
        return _json(download_dataset(dataset_ref, output_dir))

    @server.tool()
    def run_kaggle_model_curation(
        kaggle_username: str,
        confirm_remote_processing: bool,
        dataset_ref: str | None = None,
        local_input: str | None = None,
        confirm_local_upload: bool = False,
        model_handle: str = DEFAULT_MODEL_HANDLE,
        prompt_field: str | None = None,
        response_field: str = "completion",
        mode: str = "answer",
        instruction: str = "Write a correct, helpful answer for this coding task. Return only the answer text.",
        max_records: int = 1000,
        output_dir: str = "datasetcleaner-kaggle-output",
    ) -> str:
        """Run model inference on Kaggle GPUs and return a separate, unapproved JSONL candidate. Requires confirm_remote_processing=true because row data is processed on Kaggle; local_input additionally requires confirm_local_upload=true and is copied to a private Kaggle Dataset."""
        if confirm_remote_processing is not True:
            raise ValueError("Set confirm_remote_processing=true only after approving transfer/processing of this dataset on Kaggle.")
        if local_input is not None and confirm_local_upload is not True:
            raise ValueError("Set confirm_local_upload=true to copy a local file to a private Kaggle Dataset.")
        result = run_curation(
            dataset_ref=dataset_ref,
            local_input=local_input,
            upload_local=confirm_local_upload,
            kaggle_username=kaggle_username,
            model_handle=model_handle,
            prompt_field=prompt_field,
            response_field=response_field,
            mode=mode,
            instruction=instruction,
            max_records=max_records,
            output_dir=output_dir,
        )
        return _json(result)

    @server.tool()
    def preview_dataset_export(
        run_id: str,
        exclude_near: bool = False,
        exclude_incomplete: bool = False,
        exclude_malformed: bool = True,
    ) -> str:
        """Preview how many records would be exported under the selected rules. Does not write a dataset package."""
        return _json(agent_service.preview_export(run_id, exclude_near, exclude_incomplete, exclude_malformed))

    @server.tool()
    def export_reviewed_dataset(
        run_id: str,
        output_dir: str,
        dataset_name: str,
        description: str = "",
        output_format: str = "jsonl",
        exclude_near: bool = False,
        exclude_incomplete: bool = False,
        exclude_malformed: bool = True,
        user_approved: bool = False,
    ) -> str:
        """Export a reviewed package only after the user explicitly approves, including rights review. user_approved must be true."""
        return _json(agent_service.export_reviewed(
            run_id, output_dir, dataset_name, description, output_format,
            exclude_near, exclude_incomplete, exclude_malformed, user_approved,
        ))

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP transport binds only to loopback addresses.")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    try:
        from mcp.server import MCPServer  # noqa: F401
        import ipaddress
        if args.transport == "streamable-http" and not ipaddress.ip_address(args.host).is_loopback:
            parser.error("The MCP HTTP transport may bind only to a loopback address.")
        server = build_server()
        if args.transport == "stdio":
            server.run()
        else:
            server.run(transport="streamable-http", host=args.host, port=args.port)
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

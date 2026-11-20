"""Loopback-only HTTP API and browser dashboard for DataSetCleaner agents."""

from __future__ import annotations

import argparse
import ipaddress
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from typing import Any

from .agent_service import AgentService, run_metadata_files

MAX_BODY_BYTES = 2 * 1024 * 1024
WEB_ROOT = Path(__file__).with_name("web")


class _ApiHandler(BaseHTTPRequestHandler):
    server_version = "DataSetCleanerAPI/1"

    @property
    def service(self) -> AgentService:
        return self.server.service  # type: ignore[attr-defined]

    def _reply(self, status: int, data: Any, content_type: str = "application/json; charset=utf-8") -> None:
        if content_type.startswith("application/json"):
            body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        else:
            body = data if isinstance(data, bytes) else str(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(body)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _allowed_origin(self) -> bool:
        host = self.headers.get("Host", "")
        try:
            parsed_host = urlsplit("http://" + host)
            hostname = parsed_host.hostname or ""
            port = parsed_host.port
            expected_port = self.server.server_address[1]
            if port != expected_port:
                return False
            if hostname == "localhost":
                pass
            elif not ipaddress.ip_address(hostname).is_loopback:
                return False
        except ValueError:
            return False
        origin = self.headers.get("Origin")
        if not origin:
            return True
        parsed = urlsplit(origin)
        try:
            origin_host = parsed.hostname or ""
            origin_loopback = origin_host == "localhost" or ipaddress.ip_address(origin_host).is_loopback
            return parsed.scheme == "http" and origin_loopback and parsed.port == expected_port and parsed.netloc.casefold() == host.casefold()
        except ValueError:
            return False

    @staticmethod
    def _boolean(data: dict[str, Any], key: str, default: bool) -> bool:
        value = data.get(key, default)
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a JSON boolean.")
        return value

    def _body(self) -> dict[str, Any]:
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length.") from exc
        if size <= 0 or size > MAX_BODY_BYTES:
            raise ValueError(f"Request body must be between 1 and {MAX_BODY_BYTES} bytes.")
        try:
            value = json.loads(self.rfile.read(size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Request body must be valid UTF-8 JSON.") from exc
        if not isinstance(value, dict):
            raise ValueError("Request JSON must be an object.")
        return value

    def _handle_error(self, exc: Exception) -> None:
        if isinstance(exc, KeyError):
            self._reply(404, {"error": str(exc)})
        elif isinstance(exc, PermissionError):
            self._reply(403, {"error": str(exc)})
        elif isinstance(exc, (ValueError, TypeError)):
            self._reply(400, {"error": str(exc)})
        elif isinstance(exc, RuntimeError):
            self._reply(409, {"error": str(exc)})
        else:
            logging.exception("Local DataSetCleaner API request failed")
            self._reply(500, {"error": f"Local operation failed ({type(exc).__name__})."})

    def do_GET(self) -> None:
        if not self._allowed_origin():
            self._reply(403, {"error": "Only requests from this loopback API origin are accepted."})
            return
        try:
            parsed = urlsplit(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path == "/":
                self._reply(200, (WEB_ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/health":
                self._reply(200, {"status": "ok", "local_only": True, "mcp_available": _mcp_available()})
                return
            if path == "/api/runs":
                self._reply(200, {"runs": run_metadata_files(self.service.home)})
                return
            parts = path.split("/")
            if len(parts) >= 4 and parts[1:3] == ["api", "runs"]:
                run_id = parts[3]
                query = parse_qs(parsed.query)
                if len(parts) == 4:
                    result = self.service.get_run(run_id)
                elif len(parts) == 5 and parts[4] == "findings":
                    result = self.service.list_findings(run_id, int(query.get("offset", ["0"])[0]), int(query.get("limit", ["50"])[0]), query.get("code", [""])[0])
                elif len(parts) == 5 and parts[4] == "records":
                    result = self.service.get_records(run_id, int(query.get("offset", ["0"])[0]), int(query.get("limit", ["25"])[0]))
                elif len(parts) == 5 and parts[4] == "candidates":
                    result = self.service.list_candidates(run_id)
                else:
                    self._reply(404, {"error": "Unknown route."})
                    return
                self._reply(200, result)
                return
            self._reply(404, {"error": "Unknown route."})
        except Exception as exc:
            self._handle_error(exc)

    def do_POST(self) -> None:
        if not self._allowed_origin():
            self._reply(403, {"error": "Only requests from this loopback API origin are accepted."})
            return
        try:
            data = self._body()
            path = urlsplit(self.path).path.rstrip("/")
            if path == "/api/runs":
                result = self.service.start_analysis(
                    data.get("sources", []),
                    data.get("field_map"),
                    data.get("preferred_fields"),
                    float(data.get("threshold", 0.9)),
                    self._boolean(data, "detect_near", False),
                    self._boolean(data, "check_syntax", True),
                )
                self._reply(202, result)
                return
            parts = path.split("/")
            if len(parts) == 7 and parts[1:3] == ["api", "runs"] and parts[4] == "records" and parts[6] == "include":
                result = self.service.set_record_included(parts[3], int(parts[5]), data.get("include"))
                self._reply(200, result)
                return
            if len(parts) == 5 and parts[1:3] == ["api", "runs"] and parts[4] == "candidates":
                result = self.service.stage_generated_examples(parts[3], data.get("examples", []), data.get("provider", ""), data.get("model", ""))
                self._reply(201, result)
                return
            if len(parts) == 5 and parts[1:3] == ["api", "runs"] and parts[4] == "preview-export":
                result = self.service.preview_export(
                    parts[3], self._boolean(data, "exclude_near", False), self._boolean(data, "exclude_incomplete", False), self._boolean(data, "exclude_malformed", True)
                )
                self._reply(200, result)
                return
            if len(parts) == 5 and parts[1:3] == ["api", "runs"] and parts[4] == "export":
                result = self.service.export_reviewed(
                    parts[3], data.get("output_dir", ""), data.get("dataset_name", ""), data.get("description", ""),
                    data.get("output_format", "jsonl"), self._boolean(data, "exclude_near", False),
                    self._boolean(data, "exclude_incomplete", False), self._boolean(data, "exclude_malformed", True),
                    data.get("user_approved", False),
                )
                self._reply(201, result)
                return
            self._reply(404, {"error": "Unknown route."})
        except Exception as exc:
            self._handle_error(exc)

    def do_OPTIONS(self) -> None:
        # No cross-origin browser access; the dashboard shares this origin.
        self._reply(403, {"error": "Cross-origin requests are not enabled."})

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)


class _ApiServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], service: AgentService):
        super().__init__(address, _ApiHandler)
        self.service = service


def _mcp_available() -> bool:
    try:
        import mcp  # noqa: F401
    except ImportError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Must be a loopback address; the service is not remotely accessible.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, help="Persistent local review store (default: the OS application data folder).")
    args = parser.parse_args(argv)
    try:
        if not ipaddress.ip_address(args.host).is_loopback:
            parser.error("The HTTP API can bind only to a loopback address.")
        if not 1 <= args.port <= 65535:
            parser.error("--port must be between 1 and 65535.")
        service = AgentService(args.data_dir)
        server = _ApiServer((args.host, args.port), service)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"DataSetCleaner dashboard/API: http://{args.host}:{args.port}/", flush=True)
    print(f"Agent run data: {service.home}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

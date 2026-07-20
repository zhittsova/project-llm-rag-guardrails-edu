from __future__ import annotations

import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .reconciliation_ui import RECONCILIATION_UI_HTML
from .review_reconciliation import ReconciliationStore


class ReconciliationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        store: ReconciliationStore,
    ) -> None:
        self.store = store
        super().__init__(server_address, ReconciliationRequestHandler)


class ReconciliationRequestHandler(BaseHTTPRequestHandler):
    server: ReconciliationHTTPServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(RECONCILIATION_UI_HTML, "text/html; charset=utf-8")
            return
        if path == "/api/state":
            self._send_json(
                {
                    "schema_version": 1,
                    "sections": self.server.store.sections(),
                    "progress": self.server.store.progress(),
                }
            )
            return
        if path.startswith("/api/sections/"):
            try:
                self._send_json(
                    self.server.store.section(
                        unquote(path.removeprefix("/api/sections/"))
                    )
                )
            except KeyError as exc:
                self._send_error(HTTPStatus.NOT_FOUND, str(exc))
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_PATCH(self) -> None:
        path = urlparse(self.path).path
        if not path.startswith("/api/adjudications/"):
            self._send_error(HTTPStatus.NOT_FOUND, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 64_000:
                raise ValueError("request body is too large")
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            item_id = unquote(path.removeprefix("/api/adjudications/"))
            self._send_json(
                self.server.store.save_adjudication(item_id, payload)
            )
        except KeyError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(
        self,
        payload: dict[str, object],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self._send(
            json.dumps(payload, ensure_ascii=False),
            "application/json; charset=utf-8",
            status,
        )

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)

    def _send(
        self,
        content: str,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        encoded = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'unsafe-inline'; "
            "script-src 'unsafe-inline'; connect-src 'self'; "
            "frame-ancestors 'none'",
        )
        self.end_headers()
        self.wfile.write(encoded)


def serve_reconciliation_ui(
    *,
    study_dir: Path,
    port: int = 8770,
    section_size: int = 10,
    open_browser: bool = False,
) -> None:
    store = ReconciliationStore(study_dir, section_size=section_size)
    server = ReconciliationHTTPServer(("127.0.0.1", port), store)
    host, bound_port = server.server_address
    url = f"http://{host}:{bound_port}"
    print(f"Judge reconciliation UI: {url}")
    print("Press Ctrl-C to stop.")
    if open_browser:
        threading.Timer(0.2, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

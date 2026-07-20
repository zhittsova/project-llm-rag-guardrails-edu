from __future__ import annotations

import json
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .policy_manager import PolicyManager
from .policy_ui import POLICY_UI_HTML


class PolicyHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        manager: PolicyManager,
        *,
        max_body_bytes: int,
    ) -> None:
        self.manager = manager
        self.max_body_bytes = max_body_bytes
        super().__init__(server_address, PolicyRequestHandler)


class PolicyRequestHandler(BaseHTTPRequestHandler):
    server: PolicyHTTPServer

    def do_GET(self) -> None:
        if not self._request_is_local():
            self._send_error(HTTPStatus.FORBIDDEN, "non-local request rejected")
            return
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(POLICY_UI_HTML)
            return
        if path == "/api/state":
            self._send_json(self.server.manager.state())
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        if not self._request_is_local():
            self._send_error(HTTPStatus.FORBIDDEN, "non-local request rejected")
            return
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/draft":
                document = payload.get("document")
                if not isinstance(document, dict):
                    raise ValueError("document must be an object")
                self._send_json(self.server.manager.save_draft(document))
                return
            if path == "/api/simulate":
                self._send_json(
                    self.server.manager.simulate(
                        payload.get("text", ""),
                        stage=payload.get("stage", "input"),
                    )
                )
                return
            if path == "/api/coverage":
                self._send_json(self.server.manager.run_coverage())
                return
            if path == "/api/publish":
                self._send_json(self.server.manager.publish())
                return
            if path == "/api/rollback":
                version_id = payload.get("version_id")
                if not isinstance(version_id, int):
                    raise ValueError("version_id must be an integer")
                self._send_json(self.server.manager.rollback(version_id))
                return
            if path == "/api/reload":
                self._send_json(self.server.manager.reload_source())
                return
        except _PayloadTooLarge as exc:
            self._send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(exc))
            return
        except KeyError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
            return
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _read_json(self) -> dict[str, object]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if content_length > self.server.max_body_bytes:
            raise _PayloadTooLarge("request body is too large")
        raw = self.rfile.read(content_length)
        payload = json.loads(raw or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("request body must be an object")
        return payload

    def _send_html(self, html: str) -> None:
        body = html.encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(
        self,
        payload: dict[str, object],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)

    def _request_is_local(self) -> bool:
        host = self.headers.get("Host", "").split(":", 1)[0]
        if host not in {"127.0.0.1", "localhost"}:
            return False
        origin = self.headers.get("Origin")
        if not origin:
            return True
        return urlparse(origin).hostname in {"127.0.0.1", "localhost"}


class _PayloadTooLarge(ValueError):
    pass


def create_policy_server(
    manager: PolicyManager,
    *,
    host: str = "127.0.0.1",
    port: int = 8770,
    max_body_bytes: int = 1_000_000,
) -> PolicyHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("policy manager must bind to localhost")
    if max_body_bytes <= 0:
        raise ValueError("max_body_bytes must be greater than zero")
    return PolicyHTTPServer(
        (host, port),
        manager,
        max_body_bytes=max_body_bytes,
    )


def serve_policy_ui(
    manager: PolicyManager,
    *,
    host: str = "127.0.0.1",
    port: int = 8770,
    open_browser: bool = False,
) -> None:
    server = create_policy_server(manager, host=host, port=port)
    actual_host, actual_port = server.server_address
    url = f"http://{actual_host}:{actual_port}"
    print(f"Guardrail policy manager: {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

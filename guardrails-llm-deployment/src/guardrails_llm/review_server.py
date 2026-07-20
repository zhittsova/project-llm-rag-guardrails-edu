from __future__ import annotations

import json
import threading
import webbrowser
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .judge_study_audit import audit_judge_study
from .review_store import ReviewStore
from .review_ui import REVIEW_UI_HTML


class ReviewHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        store: ReviewStore,
        quality_report: dict[str, object] | None = None,
    ) -> None:
        self.store = store
        self.quality_report = quality_report
        super().__init__(server_address, ReviewRequestHandler)


class ReviewRequestHandler(BaseHTTPRequestHandler):
    server: ReviewHTTPServer

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(REVIEW_UI_HTML)
            return
        if path == "/api/state":
            self._send_json(_state_payload(self.server))
            return
        if path.startswith("/api/sections/"):
            section_id = unquote(path.removeprefix("/api/sections/"))
            try:
                self._send_json(_section_payload(self.server.store, section_id))
            except KeyError as exc:
                self._send_error(HTTPStatus.NOT_FOUND, str(exc))
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not found")

    def do_PATCH(self) -> None:
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/annotator":
                annotator_id = payload.get("annotator_id")
                if not isinstance(annotator_id, str):
                    raise ValueError("annotator_id must be a string")
                self._send_json(self.server.store.set_annotator_id(annotator_id))
                return
            if path.startswith("/api/items/"):
                item_id = unquote(path.removeprefix("/api/items/"))
                self._send_json(self.server.store.save_draft(item_id, payload))
                return
        except KeyError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
            return
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._send_error(HTTPStatus.NOT_FOUND, "not found")

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, object]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid content length") from exc
        if length > 64_000:
            raise ValueError("request body is too large")
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_html(self, content: str) -> None:
        encoded = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._security_headers("text/html; charset=utf-8", len(encoded))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_json(
        self,
        payload: dict[str, object],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._security_headers("application/json; charset=utf-8", len(encoded))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)

    def _security_headers(self, content_type: str, length: int) -> None:
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'unsafe-inline'; "
            "script-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'",
        )


def create_review_server(
    store: ReviewStore,
    *,
    port: int = 8765,
    quality_report: dict[str, object] | None = None,
) -> ReviewHTTPServer:
    return ReviewHTTPServer(("127.0.0.1", port), store, quality_report)


def serve_review_ui(
    *,
    study_dir: Path,
    reviewer: str,
    port: int = 8765,
    section_size: int = 10,
    open_browser: bool = False,
) -> None:
    quality_report = audit_judge_study(study_dir)
    if not quality_report["quality_gates_passed"]:
        failed = [
            name
            for name, passed in quality_report["quality_gates"].items()
            if not passed
        ]
        raise ValueError(
            "judge study failed quality gates: " + ", ".join(failed)
        )
    store = ReviewStore(study_dir, reviewer, section_size=section_size)
    server = create_review_server(store, port=port, quality_report=quality_report)
    host, bound_port = server.server_address
    url = f"http://{host}:{bound_port}"
    print(f"Human judge review UI: {url}")
    print(f"Reviewer: {reviewer}")
    print("Press Ctrl-C to stop.")
    if open_browser:
        threading.Timer(0.2, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _state_payload(server: ReviewHTTPServer) -> dict[str, object]:
    items = server.store.items()
    annotator_id = server.store.draft(str(items[0]["item_id"]))["annotator_id"]
    return {
        "schema_version": 1,
        "reviewer": server.store.reviewer,
        "annotator_id": annotator_id,
        "progress": server.store.progress(),
        "sections": server.store.sections(),
        "study_quality": server.quality_report,
    }


def _section_payload(store: ReviewStore, section_id: str) -> dict[str, object]:
    section = next(
        (section for section in store.sections() if section["section_id"] == section_id),
        None,
    )
    if section is None:
        raise KeyError(f"unknown review section: {section_id}")
    item_ids = set(section["item_ids"])
    groups: OrderedDict[str, list[dict[str, object]]] = OrderedDict()
    for item in store.items():
        item_id = str(item["item_id"])
        if item_id not in item_ids:
            continue
        visible_item = dict(item)
        visible_item["draft"] = store.draft(item_id)
        groups.setdefault(str(item.get("question", "")), []).append(visible_item)
    return {
        "section": section,
        "question_groups": [
            {"question": question, "items": items}
            for question, items in groups.items()
        ],
    }

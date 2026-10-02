"""Local web server for the DataSI UI (standard library only).

Security model: binds to 127.0.0.1 by default, rejects requests whose Host header is
not the bound loopback address (DNS-rebinding protection), sends no CORS headers,
caps upload size, and deletes uploaded files as soon as they are analysed. Nothing is
sent anywhere else.
"""

from __future__ import annotations

import json
import mimetypes
import os
import tempfile
import threading
import webbrowser
from email.parser import BytesParser
from email.policy import HTTP
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from datasi._version import __version__

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_SUFFIXES = {".csv", ".tsv", ".txt", ".json", ".jsonl", ".ndjson", ".parquet", ".pq"}
MAX_UPLOAD_BYTES = int(os.environ.get("DATASI_MAX_UPLOAD_MB", "1024")) * 1024 * 1024

FALLBACK_PAGE = """<!doctype html><meta charset=utf-8><title>DataSI</title>
<body style="font-family:system-ui;max-width:640px;margin:40px auto;line-height:1.5">
<h1>DataSI server is running</h1>
<p>The web UI assets are not built in this installation. Build them with
<code>npm --prefix web ci &amp;&amp; npm --prefix web run build</code>, or use the API:</p>
<pre>curl -F file=@data.csv http://127.0.0.1:PORT/api/inspect</pre>
</body>"""


class UploadError(ValueError):
    pass


def parse_multipart(content_type: str, body: bytes) -> dict[str, tuple[str | None, bytes]]:
    """Parse multipart/form-data into {field: (filename, payload)}."""
    msg = BytesParser(policy=HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    if not msg.is_multipart():
        raise UploadError("expected multipart/form-data")
    fields: dict[str, tuple[str | None, bytes]] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True)
        fields[str(name)] = (part.get_filename(), payload if isinstance(payload, bytes) else b"")
    return fields


def _safe_suffix(filename: str | None) -> str:
    if not filename:
        raise UploadError("uploaded file has no name")
    suffixes = [s.lower() for s in Path(filename).suffixes]
    if len(suffixes) >= 2 and suffixes[-1] == ".gz" and suffixes[-2] in ALLOWED_SUFFIXES:
        return suffixes[-2] + ".gz"
    if suffixes and suffixes[-1] in ALLOWED_SUFFIXES:
        return suffixes[-1]
    raise UploadError(f"unsupported file type {filename!r}; use CSV, TSV, JSON(L) or Parquet")


def _with_temp_file(field: tuple[str | None, bytes], fn: Any) -> Any:
    filename, payload = field
    suffix = _safe_suffix(filename)
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="datasi-upload-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        return fn(path, Path(filename or "upload").name)
    finally:
        os.unlink(path)


def _text(fields: dict[str, tuple[str | None, bytes]], key: str) -> str | None:
    if key not in fields:
        return None
    value = fields[key][1].decode("utf-8").strip()
    return value or None


def _config(fields: dict[str, tuple[str | None, bytes]]) -> dict[str, Any] | None:
    raw = _text(fields, "config")
    if raw is None:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise UploadError("config must be a JSON object")
    return data


def handle_inspect(fields: dict[str, tuple[str | None, bytes]]) -> dict[str, Any]:
    from datasi.core.pipeline import investigate

    if "file" not in fields:
        raise UploadError("missing 'file' field")
    config = _config(fields)
    target = _text(fields, "target")
    time_column = _text(fields, "time_column")

    def run(path: str, name: str) -> dict[str, Any]:
        return investigate(
            path, config=config, target=target, time_column=time_column, name=name
        ).to_dict()

    report: dict[str, Any] = _with_temp_file(fields["file"], run)
    report["detector_docs"] = detector_docs()
    return report


def handle_compare(fields: dict[str, tuple[str | None, bytes]]) -> dict[str, Any]:
    from datasi.core.compare import inspect_train_test

    for key in ("reference", "current"):
        if key not in fields:
            raise UploadError(f"missing '{key}' field")
    config = _config(fields)
    target = _text(fields, "target")

    def run_ref(ref_path: str, ref_name: str) -> dict[str, Any]:
        def run_cur(cur_path: str, cur_name: str) -> dict[str, Any]:
            return inspect_train_test(
                ref_path,
                cur_path,
                config=config,
                target=target,
                reference_name=ref_name,
                current_name=cur_name,
            ).to_dict()

        result: dict[str, Any] = _with_temp_file(fields["current"], run_cur)
        return result

    report: dict[str, Any] = _with_temp_file(fields["reference"], run_ref)
    report["detector_docs"] = detector_docs()
    return report


def detector_docs() -> dict[str, dict[str, str]]:
    from datasi.core.compare import DriftDetector
    from datasi.detectors import registry

    docs = {name: cls.info() for name, cls in registry().items()}
    docs["drift"] = DriftDetector.info()
    return docs


def render(report_json: bytes, fmt: str) -> tuple[str, str]:
    from datasi.reports.model import Report

    data = json.loads(report_json)
    data.pop("detector_docs", None)
    report = Report.from_json(json.dumps(data))
    if fmt in ("md", "markdown"):
        return report.to_markdown(), "text/markdown; charset=utf-8"
    return report.to_html(), "text/html; charset=utf-8"


class Handler(BaseHTTPRequestHandler):
    server_version = f"DataSI/{__version__}"
    allowed_hosts: frozenset[str] = frozenset()

    def log_message(self, format: str, *args: Any) -> None:  # quieter default logging
        if os.environ.get("DATASI_SERVER_LOG"):
            super().log_message(format, *args)

    def _host_ok(self) -> bool:
        if not self.allowed_hosts:
            return True
        return (self.headers.get("Host") or "") in self.allowed_hosts

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def do_GET(self) -> None:
        if not self._host_ok():
            self._json(HTTPStatus.FORBIDDEN, {"error": "host not allowed"})
            return
        path = urlparse(self.path).path
        if path == "/api/health":
            self._json(HTTPStatus.OK, {"status": "ok", "version": __version__})
            return
        if path == "/api/detectors":
            from datasi.configuration import Config

            self._json(
                HTTPStatus.OK,
                {"detectors": detector_docs(), "default_config": Config().model_dump(mode="json")},
            )
            return
        self._static(path)

    def _static(self, path: str) -> None:
        if not (STATIC_DIR / "index.html").is_file():
            port = str(self.server.server_address[1])  # type: ignore[index]
            self._send(
                HTTPStatus.OK,
                FALLBACK_PAGE.replace("PORT", port).encode(),
                "text/html; charset=utf-8",
            )
            return
        rel = path.lstrip("/") or "index.html"
        target = (STATIC_DIR / rel).resolve()
        if not target.is_relative_to(STATIC_DIR.resolve()) or not target.is_file():
            target = STATIC_DIR / "index.html"  # SPA fallback; never escapes STATIC_DIR
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(HTTPStatus.OK, target.read_bytes(), ctype)

    def do_POST(self) -> None:
        if not self._host_ok():
            self._json(HTTPStatus.FORBIDDEN, {"error": "host not allowed"})
            return
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "empty request body"})
            return
        if length > MAX_UPLOAD_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "upload too large"})
            return
        body = self.rfile.read(length)
        try:
            if url.path == "/api/render":
                fmt = parse_qs(url.query).get("format", ["html"])[0]
                text, ctype = render(body, fmt)
                self._send(HTTPStatus.OK, text.encode("utf-8"), ctype)
                return
            fields = parse_multipart(self.headers.get("Content-Type") or "", body)
            if url.path == "/api/inspect":
                self._json(HTTPStatus.OK, handle_inspect(fields))
            elif url.path == "/api/compare":
                self._json(HTTPStatus.OK, handle_compare(fields))
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        except (
            ValueError,
            KeyError,
        ) as exc:  # bad input: LoadError, UploadError, ConfigError, JSON
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})


def make_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    loopback = host in {"127.0.0.1", "localhost", "::1"}
    server = ThreadingHTTPServer((host, port), Handler)
    actual = server.server_address[1]
    hosts = (
        frozenset({f"127.0.0.1:{actual}", f"localhost:{actual}", f"[::1]:{actual}"})
        if loopback
        else frozenset()
    )
    server.RequestHandlerClass = type("BoundHandler", (Handler,), {"allowed_hosts": hosts})
    return server


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    server = make_server(host, port)
    url = f"http://{'127.0.0.1' if host in {'127.0.0.1', 'localhost'} else host}:{server.server_address[1]}/"
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            f"warning: listening on {host}; anyone who can reach it can upload data for analysis."
        )
    print(f"DataSI UI on {url} (Ctrl+C to stop). Data stays on this machine.")
    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

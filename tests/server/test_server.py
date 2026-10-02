from __future__ import annotations

import http.client
import json
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from datasi.server import make_server, parse_multipart

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def server() -> Iterator[int]:
    srv = make_server("127.0.0.1", 0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


def multipart(fields: dict[str, tuple[str | None, bytes]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, (filename, payload) in fields.items():
        disp = f'form-data; name="{name}"' + (f'; filename="{filename}"' if filename else "")
        parts.append(
            f"--{boundary}\r\nContent-Disposition: {disp}\r\n\r\n".encode() + payload + b"\r\n"
        )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def request(
    port: int,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    conn.request(method, path, body=body, headers=headers or {})
    resp = conn.getresponse()
    return resp.status, resp.read()


def test_health_and_detectors(server: int) -> None:
    status, body = request(server, "GET", "/api/health")
    assert status == 200 and json.loads(body)["status"] == "ok"
    status, body = request(server, "GET", "/api/detectors")
    assert "missingness" in json.loads(body)["detectors"]


def test_inspect_upload(server: int) -> None:
    body, ctype = multipart(
        {
            "file": ("orders.csv", (DATA / "dataset_duplicates.csv").read_bytes()),
            "config": (None, b'{"reference_time": "2026-10-01"}'),
        }
    )
    status, resp = request(server, "POST", "/api/inspect", body, {"Content-Type": ctype})
    assert status == 200, resp
    data = json.loads(resp)
    assert data["dataset"]["name"] == "orders.csv"
    assert any(f["code"] == "duplicates.exact" for f in data["findings"])
    assert "detector_docs" in data


def test_compare_upload(server: int) -> None:
    body, ctype = multipart(
        {
            "reference": ("train.csv", (DATA / "dataset_drift_train.csv").read_bytes()),
            "current": ("test.csv", (DATA / "dataset_drift_test.csv").read_bytes()),
        }
    )
    status, resp = request(server, "POST", "/api/compare", body, {"Content-Type": ctype})
    assert status == 200
    data = json.loads(resp)
    assert data["kind"] == "comparison" and data["reference"]["name"] == "train.csv"


def test_render_export(server: int) -> None:
    body, ctype = multipart({"file": ("t.csv", (DATA / "dataset_types.csv").read_bytes())})
    _, resp = request(server, "POST", "/api/inspect", body, {"Content-Type": ctype})
    status, html = request(
        server, "POST", "/api/render?format=html", resp, {"Content-Type": "application/json"}
    )
    assert status == 200 and html.startswith(b"<!doctype html>")
    status, md = request(
        server, "POST", "/api/render?format=md", resp, {"Content-Type": "application/json"}
    )
    assert status == 200 and b"# DataSI" in md


def test_rejects_bad_input(server: int) -> None:
    body, ctype = multipart({"file": ("evil.exe", b"MZ")})
    status, resp = request(server, "POST", "/api/inspect", body, {"Content-Type": ctype})
    assert status == 400 and b"unsupported" in resp
    status, _ = request(server, "POST", "/api/inspect", b"x", {"Content-Type": "text/plain"})
    assert status == 400
    status, _ = request(server, "POST", "/api/nope", body, {"Content-Type": ctype})
    assert status == 404


def test_dns_rebinding_protection(server: int) -> None:
    status, _ = request(server, "GET", "/api/health", headers={"Host": "attacker.example:80"})
    assert status == 403


def test_static_path_traversal(server: int) -> None:
    status, body = request(server, "GET", "/../../../../etc/passwd")
    assert status == 200
    assert b"root:" not in body


def test_parse_multipart() -> None:
    body, ctype = multipart({"a": (None, b"1"), "f": ("x.csv", b"a,b\n1,2\n")})
    fields = parse_multipart(ctype, body)
    assert fields["a"] == (None, b"1")
    assert fields["f"] == ("x.csv", b"a,b\n1,2\n")

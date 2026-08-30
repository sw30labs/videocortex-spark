"""Tests for the loopback command deck.

Three concerns, same split as the other house decks:

* **assets**   — the static page ships and talks to real endpoints
* **api**      — JSON contracts the SPA polls, and the job lifecycle
* **security** — Host pin, loopback bind, POST validation, path containment

Jobs run through ``runner.set_job_handler`` so nothing here loads TRIBE.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest
import numpy as np

from videocortex_spark.web import runner
from videocortex_spark.web.server import (
    STATIC_DIR,
    DeckHandler,
    _is_loopback_bind_host,
    _split_host_header,
)

# 1x1 PNG — enough to prove /media/ serves real bytes.
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000a49444154789c6300010000050001"
    "0d0a2db40000000049454e44ae426082"
)


@pytest.fixture(autouse=True)
def _reset_runner():
    runner.reset_for_tests()
    yield
    runner.reset_for_tests()


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), DeckHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def run_tree(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    run = root / "demo"
    (run / "frames").mkdir(parents=True)
    (run / "contact_sheet.png").write_bytes(_PNG)
    (run / "frames" / "frame_00000.png").write_bytes(_PNG)
    (run / "predictions.npy").write_bytes(b"not-a-real-npy")
    (run / "overlay.mp4").write_bytes(b"fake-mp4-bytes-for-range")
    (run / "manifest.json").write_text(
        json.dumps({
            "run": {"video": "/tmp/clip.mp4"},
            "result": {
                "n_timesteps": 12,
                "n_vertices": 20484,
                "n_frames": 6,
                "seconds": 3.2,
                "tr_s": 1.49,
            },
        }),
        encoding="utf-8",
    )
    (tmp_path / "clip.mp4").write_bytes(b"video")
    runner.set_runs_dir(root)
    return {"root": root, "run": run, "video": tmp_path / "clip.mp4"}


def _wait_idle(timeout: float = 8.0) -> None:
    deadline = time.time() + timeout
    while runner.busy() and time.time() < deadline:
        time.sleep(0.02)
    assert not runner.busy()


def get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return r.status, json.loads(r.read()), dict(r.headers)


def get_raw(base: str, path: str, headers: dict | None = None):
    req = urllib.request.Request(base + path, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def delete(base: str, path: str):
    req = urllib.request.Request(base + path, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw.decode(errors="replace")}


def post(base: str, path: str, payload, content_type: str = "application/json"):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path, data=body, headers={"Content-Type": content_type}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw.decode(errors="replace")}


def _raw_http(base: str, request: bytes) -> bytes:
    parsed = urllib.parse.urlsplit(base)
    with socket.create_connection((parsed.hostname, parsed.port), timeout=5) as sock:
        sock.sendall(request)
        chunks = []
        while True:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)
    return b"".join(chunks)


# -- assets ----------------------------------------------------------------


class TestWebAssets:
    def test_static_files_ship(self):
        assert (STATIC_DIR / "index.html").is_file()
        assert (STATIC_DIR / "favicon.svg").is_file()

    def test_index_is_self_contained(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        assert "<script" in html
        assert not re.search(r'<script[^>]+src=["\']https?://', html)
        assert not re.search(r'<link[^>]+href=["\']https?://', html)

    def test_index_calls_only_real_endpoints(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        called = set(re.findall(r"/api/([a-z]+)", html))
        served = {"health", "doctor", "defaults", "runs", "jobs"}
        assert called <= served, f"SPA calls unknown endpoints: {called - served}"

    def test_index_says_encoding_not_decoding(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        assert "encoding, not decoding" in html.lower()
        assert "VIDEOCORTEX" in html

    def test_index_serves_over_http(self, server):
        status, body, headers = get_raw(server, "/")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert b"VIDEOCORTEX" in body

    def test_favicon_serves(self, server):
        status, body, headers = get_raw(server, "/favicon.svg")
        assert status == 200
        assert headers["Content-Type"] == "image/svg+xml"
        assert b"<svg" in body


# -- API -------------------------------------------------------------------


class TestWebAPI:
    def test_health(self, server):
        status, body, _ = get(server, "/api/health")
        assert status == 200
        assert body["ok"] is True
        assert body["busy"] is False
        assert body["version"]
        assert "runs_dir" in body
        assert isinstance(body.get("can_encode"), bool)
        assert isinstance(body.get("tribev2"), bool)

    def test_defaults_match_configs(self, server):
        _, body, _ = get(server, "/api/defaults")
        assert "standard" in body["views"]
        assert body["overlay"]["position"] == "top-right"
        assert body["overlay"]["dps"] == 24.0
        assert body["render"]["max_frames"] == 60
        assert body["device"] == "auto"

    def test_doctor_renderer_scope(self, server):
        _, body, _ = get(server, "/api/doctor?scope=renderer")
        assert body["scope"] == "renderer"
        names = [c["name"] for c in body["checks"]]
        assert names == ["python", "render stack", "ffmpeg", "fsaverage5"]
        assert "torch" not in names
        assert body["ok"] is True

    def test_runs_and_detail(self, server, run_tree):
        _, body, _ = get(server, "/api/runs")
        ids = [r["id"] for r in body["runs"]]
        assert "demo" in ids
        demo = next(r for r in body["runs"] if r["id"] == "demo")
        assert demo["n_timesteps"] == 12
        assert demo["has_overlay"] is True
        assert demo["has_predictions"] is True

        _, detail, _ = get(server, "/api/runs/demo")
        assert detail["media"]["contact_sheet"] == "/media/runs/demo/contact_sheet.png"
        assert detail["media"]["overlay"] == "/media/runs/demo/overlay.mp4"
        assert detail["manifest"]["result"]["n_timesteps"] == 12

    def test_unknown_run_is_404(self, server, run_tree):
        status, body, _ = get_raw(server, "/api/runs/nope")
        assert status == 404
        assert json.loads(body)["error"] == "unknown run"

    def test_delete_run_removes_the_folder(self, server, run_tree):
        folder = run_tree["run"]
        assert folder.is_dir()
        status, body = delete(server, "/api/runs/demo")
        assert status == 200
        assert body == {"id": "demo", "deleted": True}
        assert not folder.exists()
        _, listing, _ = get(server, "/api/runs")
        assert "demo" not in [r["id"] for r in listing["runs"]]
        status404, err, _ = get_raw(server, "/api/runs/demo")
        assert status404 == 404
        assert json.loads(err)["error"] == "unknown run"

    def test_delete_unknown_run_is_404(self, server, run_tree):
        status, body = delete(server, "/api/runs/ghost")
        assert status == 404
        assert body["error"] == "unknown run"

    def test_delete_refuses_a_run_the_active_job_is_using(self, server, run_tree):
        gate = threading.Event()

        def handler(kind, params):
            gate.wait(timeout=5)
            return {"ok": True}

        runner.set_job_handler(handler)
        try:
            status, _ = post(server, "/api/jobs/overlay", {"run": "demo"})
            assert status == 202
            status2, body2 = delete(server, "/api/runs/demo")
            assert status2 == 409
            assert "in use" in body2["error"]
            assert run_tree["run"].is_dir()
        finally:
            gate.set()
            _wait_idle()

    def test_media_png(self, server, run_tree):
        status, body, headers = get_raw(server, "/media/runs/demo/contact_sheet.png")
        assert status == 200
        assert headers["Content-Type"] == "image/png"
        assert body == _PNG

    def test_media_range(self, server, run_tree):
        status, body, headers = get_raw(
            server,
            "/media/runs/demo/overlay.mp4",
            headers={"Range": "bytes=0-3"},
        )
        assert status == 206
        assert body == b"fake"
        assert headers["Content-Range"].startswith("bytes 0-3/")
        assert headers["Accept-Ranges"] == "bytes"

    def test_render_job_lifecycle(self, server, run_tree):
        seen = {}

        def handler(kind, params):
            seen["kind"] = kind
            seen["params"] = params
            return {"out_dir": "runs/demo", "n_timesteps": 3}

        runner.set_job_handler(handler)
        status, body = post(
            server,
            "/api/jobs/render",
            {"video": str(run_tree["video"]), "max_frames": 12},
        )
        assert status == 202
        job_id = body["job_id"]
        _wait_idle()
        _, job, _ = get(server, f"/api/jobs/{job_id}")
        assert job["status"] == "ok"
        assert job["kind"] == "render"
        assert seen["kind"] == "render"
        assert seen["params"]["kind"] == "video"
        assert seen["params"]["max_frames"] == 12

    def test_overlay_job_uses_spin_dest(self, server, run_tree):
        captured = {}

        def handler(kind, params):
            captured.update(params)
            return {"out": params["out"], "n_cards": 4, "spin": params["spin"]}

        runner.set_job_handler(handler)
        status, body = post(
            server,
            "/api/jobs/overlay",
            {"run": "demo", "spin": True, "dps": 18},
        )
        assert status == 202
        _wait_idle()
        assert captured["spin"] is True
        assert captured["dps"] == 18.0
        assert captured["out"].endswith("overlay_spin.mp4")

    def test_second_job_is_409(self, server, run_tree):
        gate = threading.Event()

        def handler(kind, params):
            gate.wait(timeout=5)
            return {"ok": True}

        runner.set_job_handler(handler)
        try:
            status, _ = post(server, "/api/jobs/render", {"video": str(run_tree["video"])})
            assert status == 202
            status2, body2 = post(server, "/api/jobs/overlay", {"run": "demo"})
            assert status2 == 409
            assert "running" in body2["error"]
        finally:
            gate.set()
            _wait_idle()

    def test_render_requires_existing_file(self, server, run_tree):
        status, body = post(
            server, "/api/jobs/render", {"video": "/no/such/clip.mp4"}
        )
        assert status == 400
        assert "not a file" in body["error"]

    def test_render_refuses_without_model_stack(self, server, run_tree, monkeypatch):
        monkeypatch.setattr(runner, "_model_stack_error", lambda: "torch is not installed")
        status, body = post(
            server, "/api/jobs/render", {"video": str(run_tree["video"])}
        )
        assert status == 400
        assert "torch" in body["error"]

    def test_overlay_unknown_run(self, server, run_tree):
        status, body = post(server, "/api/jobs/overlay", {"run": "ghost"})
        assert status == 400
        assert "unknown run" in body["error"]

    def test_post_rejects_non_json(self, server):
        status, body = post(server, "/api/jobs/render", b"nope", content_type="text/plain")
        assert status == 415


class TestExportAPI:
    """brain.html: the self-contained 3-D viewer, straight from the deck.

    ``regions: false`` keeps these off the Destrieux download — the bundled
    fsaverage5 mesh is the real thing, the atlas is not.
    """

    @pytest.fixture
    def real_run(self, server, run_tree):
        preds = np.random.default_rng(0).normal(0, 1, (4, 20484)).astype("float32")
        np.save(run_tree["run"] / "predictions.npy", preds)
        np.save(run_tree["run"] / "timestamps.npy", np.arange(4, dtype=float))
        return run_tree

    def test_export_unknown_run_is_404(self, server, run_tree):
        status, body = post(server, "/api/runs/ghost/export", {})
        assert status == 404
        assert "unknown run" in body["error"]

    def test_export_needs_predictions(self, server, run_tree):
        (run_tree["run"] / "predictions.npy").unlink()
        status, body = post(server, "/api/runs/demo/export", {})
        assert status == 400
        assert "predictions" in body["error"]

    def test_export_builds_caches_and_serves(self, server, real_run):
        status, body = post(server, "/api/runs/demo/export", {"regions": False})
        assert status == 200
        assert body["brain"] == "/media/runs/demo/brain.html"
        assert body["cached"] is False
        assert (real_run["run"] / "brain.html").is_file()

        # unchanged run → cache hit, no repack
        status, body = post(server, "/api/runs/demo/export", {})
        assert status == 200 and body["cached"] is True

        # the file serves as real html over /media/
        status, raw, headers = get_raw(server, "/media/runs/demo/brain.html")
        assert status == 200
        assert headers["Content-Type"].startswith("text/html")
        assert b"vcx-data" in raw

        # and the runs listing now flags it
        _, runs, _ = get(server, "/api/runs")
        demo = next(r for r in runs["runs"] if r["id"] == "demo")
        assert demo["has_brain"] is True

    def test_export_force_rebuilds(self, server, real_run):
        post(server, "/api/runs/demo/export", {"regions": False})
        status, body = post(server, "/api/runs/demo/export", {"force": True, "regions": False})
        assert status == 200 and body["cached"] is False

    def test_export_uses_manifest_colour_defaults(self, server, real_run):
        man_path = real_run["run"] / "manifest.json"
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["render"] = {"cmap": "cold_hot", "percentile": 90.0, "threshold_frac": 0.4}
        man_path.write_text(json.dumps(man), encoding="utf-8")
        status, _ = post(server, "/api/runs/demo/export", {"regions": False})
        assert status == 200
        html = (real_run["run"] / "brain.html").read_text(encoding="utf-8")
        data = json.loads(re.search(r'id="vcx-data">(.*?)</script>', html, re.S).group(1))
        assert data["percentile"] == 90.0
        assert data["threshold"] == pytest.approx(data["vmax"] * 0.4)


# -- security --------------------------------------------------------------


class TestWebSecurity:
    def test_loopback_bind_helper(self):
        assert _is_loopback_bind_host("127.0.0.1") is True
        assert _is_loopback_bind_host("localhost") is True
        assert _is_loopback_bind_host("::1") is True
        assert _is_loopback_bind_host("0.0.0.0") is False
        assert _is_loopback_bind_host("192.168.1.5") is False

    def test_split_host_header(self):
        assert _split_host_header("127.0.0.1:8730") == ("127.0.0.1", "8730")
        assert _split_host_header("[::1]:8730") == ("::1", "8730")
        assert _split_host_header("localhost") == ("localhost", "")

    def test_bad_host_is_421(self, server):
        parsed = urllib.parse.urlsplit(server)
        raw = _raw_http(
            server,
            (
                f"GET /api/health HTTP/1.1\r\n"
                f"Host: evil.example:{parsed.port}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode(),
        )
        assert raw.startswith(b"HTTP/1.1 421")

    def test_missing_host_is_421(self, server):
        raw = _raw_http(
            server,
            b"GET /api/health HTTP/1.0\r\nConnection: close\r\n\r\n",
        )
        assert raw.startswith(b"HTTP/1.0 421") or raw.startswith(b"HTTP/1.1 421")

    def test_delete_path_traversal(self, server, run_tree):
        parsed = urllib.parse.urlsplit(server)
        host = f"127.0.0.1:{parsed.port}"
        for path in (
            "/api/runs/../demo",
            "/api/runs/demo/../demo",
            "/api/runs/%2e%2e",
        ):
            raw = _raw_http(
                server,
                f"DELETE {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode(),
            )
            status_line = raw.split(b"\r\n", 1)[0]
            assert b"404" in status_line, status_line
        assert run_tree["run"].is_dir()

    def test_media_path_traversal(self, server, run_tree):
        parsed = urllib.parse.urlsplit(server)
        host = f"127.0.0.1:{parsed.port}"
        for path in (
            "/media/runs/demo/../../pyproject.toml",
            "/media/runs/../demo/contact_sheet.png",
            "/media/runs/demo/frames/../../../contact_sheet.png",
        ):
            raw = _raw_http(
                server,
                f"GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode(),
            )
            status_line = raw.split(b"\r\n", 1)[0]
            assert b"404" in status_line, status_line

    def test_media_refuses_npy(self, server, run_tree):
        status, _, _ = get_raw(server, "/media/runs/demo/predictions.npy")
        assert status == 404

    def test_non_loopback_host_rejected_by_serve(self):
        from videocortex_spark.web.server import serve

        with pytest.raises(SystemExit, match="loopback"):
            serve(host="0.0.0.0", port=1, open_browser=False)


class TestQuietErrors:
    """Safari speculation / macOS preconnect RST sockets before sending a
    request line. The stdlib prints a traceback per event; the deck must
    stay silent for connection noise and still print real handler bugs."""

    def test_handle_error_swallows_connection_noise(self):
        from videocortex_spark.web.server import QuietHTTPServer

        srv = QuietHTTPServer.__new__(QuietHTTPServer)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            try:
                raise ConnectionResetError(54, "Connection reset by peer")
            except ConnectionResetError:
                srv.handle_error(None, ("127.0.0.1", 1))
        assert err.getvalue() == ""

    def test_handle_error_prints_real_bugs(self):
        from videocortex_spark.web.server import QuietHTTPServer

        srv = QuietHTTPServer.__new__(QuietHTTPServer)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            try:
                raise ValueError("a real handler bug")
            except ValueError:
                srv.handle_error(None, ("127.0.0.1", 1))
        assert "ValueError" in err.getvalue()

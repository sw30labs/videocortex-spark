"""Loopback command deck for videocortex-spark.

Stdlib HTTP, no FastAPI. The socket binds loopback, every ``/api/`` and
``/media/`` route also refuses a non-loopback peer, and Host is pinned so a
DNS-rebinding page cannot read your runs. One job at a time — a render is
minutes of GB10, not something to queue behind.

Run:
    videocortex-spark serve                 # http://127.0.0.1:8730
    python -m videocortex_spark.web --port 9000
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import ipaddress
import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from videocortex_spark import __version__
from videocortex_spark.web import DEFAULT_PORT, runner

__all__ = ["main", "serve", "DeckHandler", "STATIC_DIR", "DEFAULT_PORT"]

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_REQUEST_BYTES = 64_000
_LOOPBACK_NAMES = frozenset({"localhost", "127.0.0.1", "::1"})
_CHUNK = 64 * 1024


def _is_loopback_bind_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _client_is_loopback(handler: BaseHTTPRequestHandler) -> bool:
    host = handler.client_address[0]
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host in _LOOPBACK_NAMES


def _split_host_header(value: str) -> tuple[str, str]:
    """``Host`` into ``(hostname, port)``, including the ``[::1]:8730`` form."""
    if value.startswith("["):
        hostname, _, rest = value.partition("]")
        return hostname[1:], rest.lstrip(":")
    hostname, _, port = value.partition(":")
    return hostname, port


def _host_header_ok(handler: BaseHTTPRequestHandler) -> bool:
    """Pin Host to this listener.

    Peer-address alone is not enough: under DNS rebinding the browser *is*
    the peer, so a page on any domain that re-resolves to 127.0.0.1 becomes
    same-origin with the deck. Pinning Host is what closes that hole.
    """
    hostname, port = _split_host_header(handler.headers.get("Host", ""))
    if hostname.lower() not in _LOOPBACK_NAMES:
        return False
    expected = handler.server.server_address[1]
    if not port:
        return expected == 80
    try:
        return int(port) == expected
    except ValueError:
        return False


class QuietHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that shrugs off browser socket churn.

    Safari speculation and macOS preconnect open sockets to the deck and
    RST them before sending a request line; the stdlib prints a full
    traceback per event. For a loopback dev server that is pure noise —
    swallow connection/timeout errors, keep printing real handler bugs.
    """

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return
        super().handle_error(request, client_address)


class DeckHandler(BaseHTTPRequestHandler):
    server_version = f"VideoCortexDeck/{__version__}"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch(self._get)

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch(self._post)

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch(self._delete)

    def do_HEAD(self) -> None:  # noqa: N802
        self._dispatch(self._get)

    def _dispatch(self, handler) -> None:
        if not _host_header_ok(self):
            self._send_json({"error": "unrecognised Host header"}, status=421)
            return
        try:
            handler()
        except Exception as exc:
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, status=500)

    def _get(self) -> None:
        path = self.path.split("?", 1)[0]
        query = parse_qs(urlparse(self.path).query)
        guarded = path.startswith("/api/") or path.startswith("/media/")
        if guarded and not _client_is_loopback(self):
            self._send_json(
                {"error": "deck APIs are restricted to loopback"}, status=403
            )
            return

        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif path == "/favicon.svg":
            self._send_file(STATIC_DIR / "favicon.svg", "image/svg+xml")
        elif path == "/api/health":
            self._send_json(runner.health())
        elif path == "/api/defaults":
            self._send_json(runner.defaults())
        elif path == "/api/doctor":
            scope = (query.get("scope") or ["renderer"])[0]
            offline = (query.get("offline") or ["0"])[0] in ("1", "true", "yes")
            self._send_json(
                runner.doctor_report(model=scope == "model", network=not offline)
            )
        elif path == "/api/runs":
            self._send_json({"runs": runner.list_runs(), "busy": runner.busy()})
        elif path.startswith("/api/runs/"):
            run_id = unquote(path[len("/api/runs/"):])
            detail = runner.get_run(run_id)
            if detail is None:
                self._send_json({"error": "unknown run"}, status=404)
            else:
                self._send_json(detail)
        elif path == "/api/jobs":
            self._send_json(
                {
                    "jobs": runner.list_jobs(),
                    "busy": runner.busy(),
                    "active_job_id": runner.active_job_id(),
                }
            )
        elif path.startswith("/api/jobs/"):
            job = runner.get_job(path.rsplit("/", 1)[-1])
            if job:
                self._send_json(job)
            else:
                self._send_json({"error": "unknown job"}, status=404)
        elif path.startswith("/media/runs/"):
            self._send_media(unquote(path[len("/media/runs/"):]))
        else:
            self.send_error(404, "Not found")

    def _post(self) -> None:
        path = self.path.split("?", 1)[0]
        export_run = None
        if path.startswith("/api/runs/") and path.endswith("/export"):
            export_run = unquote(path[len("/api/runs/"):-len("/export")])
        if path not in ("/api/jobs/render", "/api/jobs/overlay", "/api/jobs/sonify") and export_run is None:
            self.send_error(404, "Not found")
            return
        if not _client_is_loopback(self):
            self._send_json(
                {"error": "run requests are restricted to the local deck"},
                status=403,
            )
            return
        body = self._read_json_body()
        if body is None:
            return
        if export_run is not None:
            # A few seconds of CPU, no model — synchronous like /api/doctor.
            payload, error = runner.export_brain(
                export_run,
                force=bool(body.get("force")),
                regions=runner._as_bool(body.get("regions"), True),
            )
            if error:
                status = 404 if error == "unknown run" else 400
                self._send_json({"error": error}, status=status)
            else:
                self._send_json(payload, status=200)
            return
        if path.endswith("/render"):
            job_id, error = runner.start_render_job(body)
        elif path.endswith("/sonify"):
            job_id, error = runner.start_sonify_job(body)
        else:
            job_id, error = runner.start_overlay_job(body)
        if error:
            status = 409 if "running" in error else 400
            self._send_json({"error": error}, status=status)
        else:
            self._send_json({"job_id": job_id}, status=202)

    def _delete(self) -> None:
        path = self.path.split("?", 1)[0]
        if not path.startswith("/api/runs/"):
            self.send_error(404, "Not found")
            return
        if not _client_is_loopback(self):
            self._send_json(
                {"error": "deck APIs are restricted to loopback"}, status=403
            )
            return
        run_id = unquote(path[len("/api/runs/"):])
        payload, error = runner.delete_run(run_id)
        if error:
            if error == "unknown run":
                status = 404
            elif "in use" in error:
                status = 409
            else:
                status = 500
            self._send_json({"error": error}, status=status)
            return
        self._send_json(payload)

    def _send_media(self, rest: str) -> None:
        run_id, _, rel = rest.partition("/")
        target = runner.resolve_media(run_id, rel)
        if target is None:
            self.send_error(404, "Not found")
            return
        content_type = runner.MEDIA_TYPES[target.suffix.lower()]
        self._send_path_ranged(target, content_type)

    def _read_json_body(self) -> dict[str, Any] | None:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].lower()
        if content_type != "application/json":
            self._send_json(
                {"error": "Content-Type must be application/json"}, status=415
            )
            return None
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._send_json({"error": "Content-Length is required"}, status=411)
            return None
        try:
            length = int(raw_length)
        except ValueError:
            self._send_json({"error": "invalid Content-Length"}, status=400)
            return None
        if length < 0:
            self._send_json({"error": "invalid Content-Length"}, status=400)
            return None
        if length > MAX_REQUEST_BYTES:
            self._send_json({"error": "request body is too large"}, status=413)
            return None
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "invalid JSON body"}, status=400)
            return None
        if not isinstance(body, dict):
            self._send_json({"error": "JSON body must be an object"}, status=400)
            return None
        return body

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            data = path.read_bytes()
        except OSError:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _send_path_ranged(self, path: Path, content_type: str) -> None:
        try:
            size = path.stat().st_size
        except OSError:
            self.send_error(404, "Not found")
            return
        start, end, status = 0, size - 1, 200
        range_h = self.headers.get("Range")
        if range_h and range_h.startswith("bytes=") and size > 0:
            spec = range_h[6:].split(",", 1)[0].strip()
            left, _, right = spec.partition("-")
            try:
                if left and right:
                    start, end = int(left), int(right)
                elif left:
                    start, end = int(left), size - 1
                elif right:
                    start, end = max(0, size - int(right)), size - 1
                else:
                    start, end = 0, size - 1
            except ValueError:
                start, end = 0, size - 1
            else:
                end = min(end, size - 1)
                if start > end or start < 0:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = 206
        length = end - start + 1 if size else 0
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "private, max-age=60")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if self.command == "HEAD" or length <= 0:
            return
        try:
            with path.open("rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            return

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    runs_dir: Path | None = None,
) -> None:
    """Start the dashboard (blocks until Ctrl-C)."""
    if not _is_loopback_bind_host(host):
        raise SystemExit(
            "--host must be a loopback address or localhost; "
            "the deck is intentionally local-only"
        )
    os.environ.setdefault("MPLBACKEND", "Agg")
    if runs_dir is not None:
        runner.set_runs_dir(runs_dir)

    try:
        server = QuietHTTPServer((host, port), DeckHandler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            print(f"Port {port} is already in use — a deck instance is most likely")
            print(f"already running at http://{host}:{port}")
            print(f"(or pick another port: videocortex-spark serve --port {port + 1})")
            raise SystemExit(1) from None
        raise

    url = f"http://{host}:{port}"
    print(f"videocortex-spark command deck → {url}  (Ctrl+C to stop)", flush=True)
    print(f"  runs: {runner.runs_dir()}", flush=True)
    print("  encoding, not decoding — stimulus in, predicted cortex out", flush=True)
    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="videocortex-spark loopback command deck")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--runs", type=Path, default=None, help="runs directory")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser tab on start.",
    )
    args = parser.parse_args(argv)
    serve(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        runs_dir=args.runs,
    )


if __name__ == "__main__":
    main()

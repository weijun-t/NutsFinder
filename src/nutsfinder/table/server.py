"""The poker table's HTTP server: one HTML page and a small JSON API (stdlib only).

Endpoints (all JSON unless noted; errors are ``{"error": "..."}`` with a 4xx/5xx status)::

    GET  /                       the page (index.html)
    GET  /api/bots               the bots on offer (CFRBot only if a trained model loaded)
    POST /api/session            {} -> {"session": id}  (a fresh session id)
    POST /api/play/new           {session, bot, seed?}   new opponent, deals hand 1
    POST /api/play/next          {session, expected_step}          deal the next hand
    POST /api/play/act           {session, action, amount?, expected_step}
                                 action: fold|check|call|bet|raise
    POST /api/play/live          {session, on}           reasoning shown live (peeking)
    POST /api/play/reveal        {session}               show the bot's cards after a hand
    POST /api/play/end           {session}               end the session, reveal its seed/secret
    GET  /api/play/state?session=ID
    POST /api/replay/start       {session, bot_a, bot_b, seed?, hands}
    POST /api/replay/cancel      {session}
    GET  /api/replay/status?session=ID
    GET  /api/replay/hand?session=ID&index=N
    GET  /api/replay/export?session=ID   (a JSON download)

Rules every request goes through:

- ``Host`` must be ``127.0.0.1:<port>`` or ``localhost:<port>`` (else 403): a
  page on another site cannot reach the API through DNS rebinding.
- A POST must have ``Content-Type: application/json`` (else 403), so a plain
  cross-site form cannot post to it.
- ``expected_step`` (the ``step`` of the state the browser acted on) is
  required for ``act`` and ``next``: a duplicate or stale request gets 409.
- Mutating play requests take the session's lock without waiting: a second
  click while the bot thinks gets ``409 busy``. ``GET /api/play/state``
  returns the snapshot of the last completed change and never waits for the bot.

State lives in memory per session id (single user is the intended use; the
oldest sessions are dropped beyond :data:`MAX_SESSIONS`).
"""

from __future__ import annotations

import json
import secrets
import socket
import sys
import threading
import traceback
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from nutsfinder.table.session import (
    MAX_REPLAY_HANDS,
    BotCatalog,
    PlaySession,
    ReplayJob,
    SessionError,
    StaleRequest,
    parse_digits,
)

__all__ = ["TableApp", "make_server", "serve", "MAX_SESSIONS"]

MAX_SESSIONS = 32
MAX_BODY = 64 * 1024
PAGE = Path(__file__).with_name("index.html")


class HttpError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class _Session:
    def __init__(self) -> None:
        self.play: PlaySession | None = None
        self.replay: ReplayJob | None = None


def _user_seed(value: Any) -> int | None:
    """A user-given seed (int or ASCII digit string, at most 20 digits), or ``None`` for random."""
    if value is None or value == "":
        return None
    try:
        return parse_digits(value, "seed")
    except SessionError as err:
        raise HttpError(400, str(err)) from None


class TableApp:
    """The server's state: the bot catalog and the sessions. Request handling is in :class:`_Handler`."""

    def __init__(self, catalog: BotCatalog) -> None:
        self.catalog = catalog
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._lock = threading.Lock()

    # --- sessions -------------------------------------------------------------

    def new_session(self) -> str:
        sid = secrets.token_hex(8)
        with self._lock:
            self._sessions[sid] = _Session()
            while len(self._sessions) > MAX_SESSIONS:
                _, old = self._sessions.popitem(last=False)
                if old.replay is not None:
                    old.replay.cancel()
        return sid

    def session(self, sid: Any) -> _Session:
        if not isinstance(sid, str) or not sid:
            raise HttpError(400, "missing session id")
        with self._lock:
            s = self._sessions.get(sid)
            if s is None:
                raise HttpError(404, "unknown session (the server may have restarted): start a new one")
            self._sessions.move_to_end(sid)
            return s

    def _play(self, body: dict[str, Any]) -> PlaySession:
        p = self.session(body.get("session")).play
        if p is None:
            raise HttpError(409, "no game yet: choose an opponent and start")
        return p

    def _locked(self, play: PlaySession, fn, expected_step: Any = None, check_step: bool = False) -> dict[str, Any]:
        if not play.lock.acquire(blocking=False):
            raise HttpError(409, "busy: the bot is still thinking")
        try:
            if check_step:
                play.check_step(expected_step)
            fn()
            return play.refresh_snapshot()
        finally:
            play.lock.release()

    def _replay(self, sid: Any) -> ReplayJob:
        r = self.session(sid).replay
        if r is None:
            raise HttpError(409, "no replay yet: start one")
        return r

    # --- routes -----------------------------------------------------------------

    def get(self, path: str, query: dict[str, str]) -> Any:
        match path:
            case "/api/bots":
                return {"bots": self.catalog.listing(), "max_replay_hands": MAX_REPLAY_HANDS,
                        "turn_tables_loaded": self.catalog.turn_tables_loaded}
            case "/api/play/state":
                return self._play({"session": query.get("session")}).snapshot()
            case "/api/replay/status":
                return self._replay(query.get("session")).status_payload()
            case "/api/replay/hand":
                index = parse_digits(query.get("index", ""), "hand index", 6)
                return self._replay(query.get("session")).hand_payload(index)
            case "/api/replay/export":
                return self._replay(query.get("session")).export()
        raise HttpError(404, f"no such endpoint: GET {path}")

    def post(self, path: str, body: dict[str, Any]) -> Any:
        match path:
            case "/api/session":
                return {"session": self.new_session()}
            case "/api/play/new":
                s = self.session(body.get("session"))
                key = body.get("bot")
                if not isinstance(key, str):
                    raise HttpError(400, "bot must be a bot key, see /api/bots")
                ok, why = self.catalog.available(key)
                if not ok:
                    raise HttpError(400, f"bot {key!r} is not available: {why}")
                play = PlaySession.create(self.catalog, key, _user_seed(body.get("seed")))
                s.play = play
                return self._locked(play, play.start_hand)
            case "/api/play/next":
                play = self._play(body)
                return self._locked(play, play.start_hand, body.get("expected_step"), check_step=True)
            case "/api/play/act":
                play = self._play(body)
                return self._locked(play, lambda: play.act(body.get("action"), body.get("amount")),
                                    body.get("expected_step"), check_step=True)
            case "/api/play/live":
                play = self._play(body)
                return self._locked(play, lambda: play.set_live(body.get("on")))
            case "/api/play/reveal":
                play = self._play(body)
                return self._locked(play, play.reveal_bot_cards)
            case "/api/play/end":
                play = self._play(body)
                return self._locked(play, play.end_session)
            case "/api/replay/start":
                s = self.session(body.get("session"))
                a, b = body.get("bot_a"), body.get("bot_b")
                for key in (a, b):
                    if not isinstance(key, str):
                        raise HttpError(400, "bot_a and bot_b must be bot keys, see /api/bots")
                    ok, why = self.catalog.available(key)
                    if not ok:
                        raise HttpError(400, f"bot {key!r} is not available: {why}")
                hands = parse_digits(body.get("hands"), "hands", 4)
                seed = _user_seed(body.get("seed"))
                if seed is None:  # replays have no hidden information: a 64-bit random seed, shown
                    seed = secrets.randbits(64)
                job = ReplayJob(self.catalog, a, b, seed, hands)
                if s.replay is not None:
                    s.replay.cancel()
                s.replay = job
                job.start()
                return job.status_payload()
            case "/api/replay/cancel":
                job = self._replay(body.get("session"))
                job.cancel()
                return job.status_payload()
        raise HttpError(404, f"no such endpoint: POST {path}")


class _Handler(BaseHTTPRequestHandler):
    server_version = "NutsFinderTable/1.0"
    app: TableApp  # set on the subclass made by make_server
    quiet = True

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        if not self.quiet:
            super().log_message(format, *args)

    def _send(self, status: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, data: Any, extra: dict[str, str] | None = None) -> None:
        body = json.dumps(data, allow_nan=False, separators=(",", ":")).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8", extra)

    def _check_host(self) -> None:
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = (self.headers.get("Host") or "").strip().lower()
        if host not in allowed:
            raise HttpError(403, f"forbidden Host {host!r}: use http://127.0.0.1:{port}/")

    def _handle(self, method: str) -> None:
        try:
            self._check_host()
            url = urlparse(self.path)
            path = url.path.rstrip("/") or "/"
            if method == "GET" and path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                return
            if method == "GET" and path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
                return
            if method == "GET":
                query = {k: v[-1] for k, v in parse_qs(url.query).items()}
                data = self.app.get(path, query)
                extra = None
                if path == "/api/replay/export":
                    extra = {"Content-Disposition": 'attachment; filename="nutsfinder-replay.json"'}
                self._json(200, data, extra)
            else:
                self._json(200, self.app.post(path, self._body()))
        except HttpError as err:
            self._json(err.status, {"error": err.message})
        except StaleRequest as err:
            self._json(409, {"error": str(err)})
        except SessionError as err:
            self._json(400, {"error": str(err)})
        except (ConnectionError, BrokenPipeError):
            pass
        except Exception as err:  # never let one request take the server down
            traceback.print_exc(file=sys.stderr)
            try:
                self._json(500, {"error": f"internal error: {type(err).__name__}: {err}"})
            except Exception:
                pass

    def _body(self) -> dict[str, Any]:
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype != "application/json":
            raise HttpError(403, "POST requests must have Content-Type: application/json")
        length = (self.headers.get("Content-Length") or "0").strip()
        if not (length.isascii() and length.isdigit() and len(length) <= 9):
            raise HttpError(400, "bad Content-Length")
        n = int(length)
        if n > MAX_BODY:
            raise HttpError(413, "request body too large")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except (ValueError, RecursionError):  # bad UTF-8/JSON, ints over the digit limit, deep nesting
            raise HttpError(400, "body must be valid JSON") from None
        if not isinstance(data, dict):
            raise HttpError(400, "body must be a JSON object")
        return data

    def do_GET(self) -> None:  # noqa: N802 - stdlib name
        self._handle("GET")

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")


class _Server(ThreadingHTTPServer):
    """A threading server that owns its port exclusively.

    On Windows ``SO_REUSEADDR`` lets a second process bind a port that is
    already listening (requests then go to either one), so it is off there and
    ``SO_EXCLUSIVEADDRUSE`` is set instead.
    """

    daemon_threads = True
    allow_reuse_address = sys.platform != "win32"

    def server_bind(self) -> None:
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def make_server(catalog: BotCatalog, host: str = "127.0.0.1", port: int = 8765,
                verbose: bool = False) -> ThreadingHTTPServer:
    """A ready-to-serve server (call ``serve_forever``). ``port=0`` picks a free port."""
    app = TableApp(catalog)
    handler = type("Handler", (_Handler,), {"app": app, "quiet": not verbose})
    server = _Server((host, port), handler)
    server.app = app  # type: ignore[attr-defined]
    return server


def serve(catalog: BotCatalog, host: str = "127.0.0.1", port: int = 8765, verbose: bool = False) -> None:
    """Serve until Ctrl+C, printing the URL."""
    server = make_server(catalog, host, port, verbose)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"NutsFinder table running at {url}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        server.server_close()

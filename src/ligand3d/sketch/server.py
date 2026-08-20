"""Serve the browser sketcher.

`ligand3d sketch` starts a localhost server and opens a browser. The page hosts
a 2D editor, the build settings, and a run log. You draw, build, read what
happened, clear the canvas, and draw the next one — the server stays up until
you stop it, so nothing needs reloading between molecules.

The editor is [JSME](https://jsme-editor.github.io/): a 1 MB zip whose single
`jsme.nocache.js` loader runs entirely in the browser, fetched on first use and
offline thereafter. If it cannot be fetched the page falls back to a box that
accepts a pasted SMILES or molblock, so the command still works offline.

Everything is bound to 127.0.0.1 and nothing leaves the machine.
"""

from __future__ import annotations

import http.server
import io
import json
import shutil
import socket
import socketserver
import threading
import urllib.parse
import urllib.request
import webbrowser
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import Ligand3DError
from .session import (
    JobStore,
    backend_catalog,
    inspect_target,
    next_filename,
    run_job,
    slurm_status,
    solvent_catalog,
)

JSME_RELEASE = "JSME_2024-04-29"
JSME_URL = (
    "https://raw.githubusercontent.com/jsme-editor/jsme-editor.github.io/"
    f"master/downloads/{JSME_RELEASE}.zip"
)

_HERE = Path(__file__).parent
_STATIC = _HERE / "static"
_MAX_BODY = 8 * 1024 * 1024


APP_PAGE = _STATIC / "app.html"
MODELS_PAGE = _STATIC / "models.html"
"""The only page. It handles the no-editor case itself with a paste box, so the
settings panel and the run log are available even when JSME cannot be fetched."""


@dataclass(frozen=True)
class Engine:
    """A sketcher whose assets are served from `root`."""

    name: str
    root: Path


def _extract(payload: bytes, target: Path, strip_prefix: str | None = None) -> None:
    """Unpack a zip, optionally dropping one leading path component.

    Skips the `__MACOSX` metadata these archives carry, and refuses members whose
    path would escape the target directory.
    """
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            name = member.filename
            if name.startswith("__MACOSX") or "/._" in name or name.endswith("/._"):
                continue
            if strip_prefix:
                if not name.startswith(strip_prefix):
                    continue
                name = name[len(strip_prefix) :]
            if not name or name.endswith("/"):
                continue
            destination = (target / name).resolve()
            try:
                destination.relative_to(root)
            except ValueError:
                continue  # zip-slip attempt
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, open(destination, "wb") as out:
                shutil.copyfileobj(source, out)


def ensure_jsme(quiet: bool = False) -> Path | None:
    """Download and unpack JSME if it isn't cached already."""
    from ..config import jsme_dir

    target = jsme_dir()
    if (target / "jsme" / "jsme.nocache.js").exists():
        return target

    try:
        if not quiet:
            print(f"fetching the JSME sketcher ({JSME_RELEASE}, ~1 MB, one time) ...")
        with urllib.request.urlopen(JSME_URL, timeout=180) as response:
            payload = response.read()
        _extract(payload, target, strip_prefix=f"{JSME_RELEASE}/")
    except Exception as exc:
        if not quiet:
            print(f"  could not fetch JSME: {exc}")
        return None

    return target if (target / "jsme" / "jsme.nocache.js").exists() else None


def choose_engine(quiet: bool = False) -> Engine | None:
    """Return the JSME engine, fetching it if need be, or None to fall back."""
    jsme = ensure_jsme(quiet=quiet)
    if jsme is None:
        return None
    return Engine(name="jsme", root=jsme)


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the app, the sketcher assets, and the build API."""

    engine: Engine | None = None
    jobs: JobStore
    defaults: dict[str, Any] = {}

    def log_message(self, *args) -> None:  # silence the default access log
        pass

    # ---- plumbing --------------------------------------------------------
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the tab went away mid-response

    def _json(self, payload: Any, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > _MAX_BODY:
            return {}
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"molblock": raw}
        return parsed if isinstance(parsed, dict) else {}

    # ---- GET -------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = urllib.parse.unquote(self.path.split("?", 1)[0])
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

        if path in ("/", "/index.html"):
            self._send(200, APP_PAGE.read_bytes(), "text/html; charset=utf-8")
            return

        if path in ("/models", "/models.html"):
            self._send(200, MODELS_PAGE.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/api/models":
            from ..catalog import summarize

            self._json(summarize().to_json())
            return

        if path == "/api/config":
            self._json(
                {
                    "engine": self.engine.name if self.engine else None,
                    "backends": backend_catalog(),
                    "solvents": solvent_catalog(),
                    "defaults": self.defaults,
                    "slurm": slurm_status(),
                }
            )
            return

        if path == "/api/templates":
            from ..resolve import opsin_available, template_list

            self._json(
                {
                    "templates": template_list(),
                    "opsin": opsin_available(),
                }
            )
            return

        if path == "/api/next-name":
            directory = (query.get("directory") or [self.defaults.get("directory", ".")])[0]
            stem = (query.get("stem") or ["sketch"])[0]
            self._json({"filename": next_filename(directory, stem=stem)})
            return

        if path.startswith("/api/job/"):
            try:
                job_id = int(path.rsplit("/", 1)[1])
            except ValueError:
                self._json({"error": "bad job id"}, 400)
                return
            job = self.jobs.get(job_id)
            if job is None:
                self._json({"error": f"no job {job_id}"}, 404)
                return
            self._json(job.to_json())
            return

        if path.startswith("/sketcher/") and self.engine:
            self._serve_asset(path[len("/sketcher/") :] or "index.html")
            return

        self._send(404, b"not found", "text/plain")

    def _serve_asset(self, relative: str) -> None:
        """Serve a file from the engine's directory, and nothing outside it."""
        root = self.engine.root.resolve()
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            self._send(403, b"forbidden", "text/plain")
            return
        if not target.is_file():
            self._send(404, b"not found", "text/plain")
            return
        self._send(200, target.read_bytes(), self.guess_type(str(target)))

    # ---- POST ------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0]

        if path == "/api/check-path":
            data = self._read_json()
            info = inspect_target(
                data.get("directory", "."),
                data.get("filename", "sketch0"),
                formats=data.get("formats"),
            )
            self._json(info.to_json())
            return

        if path == "/api/preview":
            self._preview(self._read_json())
            return

        if path == "/api/build":
            self._start_build(self._read_json())
            return

        if path == "/api/resolve":
            self._resolve(self._read_json())
            return

        self._send(404, b"not found", "text/plain")

    def _resolve(self, data: dict) -> None:
        """Turn a typed name, SMILES, InChI, or CID into something to edit.

        A failed lookup is an ordinary outcome — a typo, a name PubChem does not
        carry, no network — so it comes back as a message to read rather than an
        error state.
        """
        from ..resolve import ResolveError, resolve, template, template_list

        query = str(data.get("query") or "").strip()
        if not query:
            self._json({"error": "nothing to look up"}, 400)
            return

        try:
            known = {entry["name"] for entry in template_list()}
            found = (
                template(query)
                if query.lower() in known
                else resolve(query, allow_network=not data.get("offline"))
            )
            self._json(found.to_json())
        except ResolveError as exc:
            self._json({"error": str(exc)}, 404)
        except Exception as exc:  # a genuine bug, not a failed lookup
            self._json({"error": f"unexpected {type(exc).__name__}: {exc}"}, 500)

    def _preview(self, data: dict) -> None:
        """Render how the drawing is being read, with atom indices.

        Called on every edit, so a half-finished structure must produce an
        explanation rather than a stack trace or a blank panel.
        """
        from ..depict import depict_molblock

        molblock = data.get("molblock") or ""
        if not molblock.strip():
            self._json({"empty": True})
            return
        try:
            depiction = depict_molblock(
                molblock,
                width=int(data.get("width") or 480),
                height=int(data.get("height") or 300),
                dark=bool(data.get("dark")),
                show_indices=data.get("show_indices", True),
            )
        except Ligand3DError as exc:
            self._json({"error": str(exc)})
            return
        except Exception as exc:  # a mid-edit structure can be anything
            self._json({"error": f"{type(exc).__name__}: {exc}"})
            return

        payload = depiction.to_json()
        try:
            from ..molecule import classify_undefined_stereo, from_molblock

            molecule = from_molblock(molblock)
            payload["smiles"] = molecule.smiles
            payload["formula"] = molecule.formula
            payload["mw"] = round(molecule.molecular_weight, 2)
            payload["charge"] = molecule.formal_charge
            payload["zwitterion"] = molecule.is_zwitterion
            payload["n_fragments"] = molecule.n_fragments
            payload["advice"] = classify_undefined_stereo(molecule)
            payload["double_bonds"] = [
                {"begin": r.begin, "end": r.end, "cip": r.cip, "cis_trans": r.cis_trans}
                for r in __import__(
                    "ligand3d.molecule", fromlist=["describe_double_bonds"]
                ).describe_double_bonds(molecule)
            ]
        except Exception:
            pass
        self._json(payload)

    def _start_build(self, data: dict) -> None:
        molblock = data.get("molblock") or ""
        if not molblock.strip():
            self._json({"error": "nothing was drawn"}, 400)
            return

        settings = data.get("settings") or {}
        info = inspect_target(
            settings.get("directory", "."),
            settings.get("filename", "sketch0"),
            formats=settings.get("formats"),
        )
        if info.error:
            self._json({"error": info.error}, 400)
            return
        if info.would_overwrite and not data.get("overwrite"):
            # The page asks the user and retries with overwrite set.
            self._json(
                {
                    "needs_confirmation": "overwrite",
                    "target": info.to_json(),
                },
                409,
            )
            return

        job = self.jobs.create()
        if settings.get("slurm"):
            # An empty options dict still means "queue this" — falling back to
            # a local run because a client sent no options would quietly do the
            # opposite of what was asked.
            options = settings.get("slurm_options")
            job.slurm_options = options if isinstance(options, dict) else {}
        thread = threading.Thread(
            target=run_job, args=(job, molblock, settings, info), daemon=True
        )
        thread.start()
        self._json({"job_id": job.id, "target": info.to_json()})


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """Threaded so the page can poll a job's log while that job is running.

    `allow_reuse_address` has to be a class attribute: HTTPServer binds inside
    __init__, so setting it on the instance afterwards is too late and a second
    `ligand3d sketch` on the same port fails with EADDRINUSE.
    """

    allow_reuse_address = True
    daemon_threads = True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _warm_catalog() -> None:
    """Touch the things the first page load needs, off the request path.

    Availability probing stats weight directories that may be on a network
    filesystem. Doing it here means the latency lands before anyone is looking
    rather than in front of the editor.
    """
    try:
        backend_catalog()
        slurm_status()
    except Exception:  # warming is an optimization; never let it break startup
        pass


def serve(
    port: int = 0,
    open_browser: bool = True,
    quiet: bool = False,
    defaults: dict[str, Any] | None = None,
    block: bool = True,
) -> tuple[_Server, int]:
    """Start the sketcher session. Runs until interrupted.

    Returns the server and its port. With `block=False` the caller is
    responsible for shutting it down, which is what the tests do.
    """
    engine = choose_engine(quiet=quiet)
    if engine is None and not quiet:
        print("no sketcher available; serving the paste-box fallback")

    port = port or _free_port()
    handler = type(
        "_BoundHandler",
        (_Handler,),
        {"engine": engine, "jobs": JobStore(), "defaults": dict(defaults or {})},
    )

    try:
        server = _Server(("127.0.0.1", port), handler)
    except OSError as exc:
        raise Ligand3DError(
            f"could not bind port {port}: {exc}. "
            "Pass --port to choose another, or omit it to have one picked."
        ) from exc

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Work out what the backends can do while the browser is still fetching the
    # page, so the first /api/config is answered from a warm cache rather than
    # probing the filesystem with someone waiting on it.
    threading.Thread(target=_warm_catalog, daemon=True).start()

    url = f"http://127.0.0.1:{port}/"
    if not quiet:
        label = engine.name if engine else "paste box"
        print(f"ligand3d sketcher ({label}) running at {url}")
        print("draw, set the options, and press Build. Ctrl-C to stop.")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    if block:
        try:
            thread.join()
        except KeyboardInterrupt:
            if not quiet:
                print("\nstopping")
        finally:
            server.shutdown()
            server.server_close()
    return server, port

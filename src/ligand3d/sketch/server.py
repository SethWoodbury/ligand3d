"""Serve a browser sketcher and wait for a molecule.

The flow is deliberately blunt: start a localhost HTTP server, open a browser at
it, block until the page POSTs a molblock back, shut down, and hand the molblock
to the same pipeline the command line uses. No persistent daemon, no state.

Three engines, tried in order:

- **JSME** — the default. One 1 MB zip containing a single `jsme.nocache.js`
  loader that runs entirely in the browser. It is fetched on first use and then
  works offline forever.
- **Ketcher** — nicer to use, but EPAM ships it as an npm library rather than a
  servable page: `ketcher-standalone.zip` contains `index.js` and `main.js` and
  no HTML at all, so it cannot simply be unzipped and served. Point
  `LIGAND3D_KETCHER_DIR` at a directory containing a built `index.html` and this
  will use it in preference to JSME.
- **A plain paste box** — the fallback when nothing can be fetched, so
  `ligand3d sketch` still does something useful on an air-gapped machine.
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

from ..errors import Ligand3DError

JSME_RELEASE = "JSME_2024-04-29"
JSME_URL = (
    "https://raw.githubusercontent.com/jsme-editor/jsme-editor.github.io/"
    f"master/downloads/{JSME_RELEASE}.zip"
)

_HERE = Path(__file__).parent
_STATIC = _HERE / "static"


@dataclass(frozen=True)
class Engine:
    """A sketcher that can be served from disk."""

    name: str
    root: Path
    page: str

    @property
    def bridge(self) -> Path:
        return _STATIC / self.page


def _extract(payload: bytes, target: Path, strip_prefix: str | None = None) -> None:
    """Unpack a zip, optionally dropping one leading path component.

    Skips the `__MACOSX` metadata directories that these archives carry.
    """
    target.mkdir(parents=True, exist_ok=True)
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
            destination = target / name
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


def ketcher_is_available() -> bool:
    from ..config import ketcher_dir

    return (ketcher_dir() / "index.html").exists()


def choose_engine(quiet: bool = False) -> Engine | None:
    """Pick the best sketcher available, fetching JSME if need be."""
    from ..config import ketcher_dir

    ketcher = ketcher_dir()
    if (ketcher / "index.html").exists():
        return Engine(name="ketcher", root=ketcher, page="bridge_ketcher.html")

    jsme = ensure_jsme(quiet=quiet)
    if jsme is not None:
        return Engine(name="jsme", root=jsme, page="bridge_jsme.html")

    return None


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the bridge page, the sketcher assets, and accepts one submission."""

    result: dict = {}
    done: threading.Event
    engine: Engine | None = None

    def log_message(self, *args) -> None:  # silence the default access log
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = urllib.parse.unquote(self.path.split("?", 1)[0])

        if path in ("/", "/index.html"):
            page = self.engine.bridge if self.engine else _STATIC / "fallback.html"
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/status":
            body = {"engine": self.engine.name if self.engine else None}
            self._send(200, json.dumps(body).encode(), "application/json")
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

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.split("?", 1)[0] != "/submit":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"molblock": raw}

        self.result["molblock"] = payload.get("molblock", "")
        self.result["smiles"] = payload.get("smiles", "")
        self._send(200, json.dumps({"ok": True}).encode(), "application/json")
        self.done.set()


class _Server(socketserver.TCPServer):
    """TCP server that can rebind a port still in TIME_WAIT.

    `allow_reuse_address` has to be a class attribute: TCPServer binds inside
    __init__, so setting it on the instance afterwards is too late and a second
    `ligand3d sketch` on the same port fails with EADDRINUSE.
    """

    allow_reuse_address = True
    daemon_threads = True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def sketch_molecule(
    port: int = 0,
    open_browser: bool = True,
    timeout: float = 1800.0,
    quiet: bool = False,
) -> str | None:
    """Serve the sketcher and block until a molecule comes back.

    Returns a molblock, or None if the page was closed without submitting.
    """
    engine = choose_engine(quiet=quiet)
    if engine is None and not quiet:
        print("no sketcher available; falling back to a paste box")

    port = port or _free_port()
    done = threading.Event()

    handler = type(
        "_BoundHandler", (_Handler,), {"result": {}, "done": done, "engine": engine}
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

    url = f"http://127.0.0.1:{port}/"
    if not quiet:
        label = engine.name if engine else "paste box"
        print(f"sketcher ({label}) running at {url}")
        print("draw a molecule, then press 'Use this molecule'. Ctrl-C to cancel.")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        done.wait(timeout=timeout)
    except KeyboardInterrupt:
        return None
    finally:
        server.shutdown()
        server.server_close()

    return _to_molblock(handler.result)


def _to_molblock(result: dict) -> str | None:
    """Prefer the molblock; fall back to building one from a SMILES string.

    The molblock is returned verbatim. Its first line is the molecule-name
    field, which is routinely empty, so stripping leading whitespace shifts
    every subsequent line up one and the counts line lands where the header
    belongs — producing a file no parser will accept. Strip only to test
    whether anything was sent.
    """
    molblock = result.get("molblock") or ""
    if molblock.strip():
        return molblock

    smiles = (result.get("smiles") or "").strip()
    if not smiles:
        return None

    from rdkit import Chem
    from rdkit.Chem import AllChem

    from ..molecule import rdkit_quiet

    with rdkit_quiet():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        AllChem.Compute2DCoords(mol)
        return Chem.MolToMolBlock(mol)

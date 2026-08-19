"""Serve a browser sketcher and wait for a molecule.

The flow is deliberately blunt: start a localhost HTTP server, open a browser at
it, block until the page POSTs a molblock back, shut down, and hand the molblock
to the same pipeline the command line uses. No persistent daemon, no state.

Ketcher's prebuilt bundle is 35 MB, which is too much to commit, so it is fetched
on demand into the cache. If it cannot be fetched — no network, or a locked-down
machine — the server falls back to a plain paste box that accepts SMILES or a
molblock, so `ligand3d sketch` still does something useful.
"""

from __future__ import annotations

import http.server
import io
import json
import shutil
import socket
import socketserver
import threading
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

from ..errors import Ligand3DError

KETCHER_RELEASE = "v3.17.0"
KETCHER_URL = (
    f"https://github.com/epam/ketcher/releases/download/{KETCHER_RELEASE}/"
    "ketcher-standalone.zip"
)

_HERE = Path(__file__).parent
_STATIC = _HERE / "static"


def ketcher_is_available() -> bool:
    from ..config import ketcher_dir

    return (ketcher_dir() / "index.html").exists()


def ensure_ketcher(quiet: bool = False) -> Path | None:
    """Download and unpack the Ketcher bundle if it isn't cached already."""
    from ..config import ketcher_dir

    target = ketcher_dir()
    if (target / "index.html").exists():
        return target

    target.mkdir(parents=True, exist_ok=True)
    try:
        if not quiet:
            print(f"fetching Ketcher {KETCHER_RELEASE} (~35 MB, one time) ...")
        with urllib.request.urlopen(KETCHER_URL, timeout=120) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(target)
    except Exception:
        return None

    if (target / "index.html").exists():
        return target
    # Some releases nest everything one directory deep.
    for candidate in target.iterdir():
        if candidate.is_dir() and (candidate / "index.html").exists():
            for item in candidate.iterdir():
                shutil.move(str(item), str(target / item.name))
            candidate.rmdir()
            return target
    return None


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves the bridge page, the Ketcher bundle, and accepts one submission."""

    result: dict = {}
    done: threading.Event
    ketcher_root: Path | None = None

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
        path = self.path.split("?", 1)[0]

        if path in ("/", "/index.html"):
            page = _STATIC / ("bridge.html" if self.ketcher_root else "fallback.html")
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/status":
            self._send(
                200,
                json.dumps({"ketcher": bool(self.ketcher_root)}).encode(),
                "application/json",
            )
            return

        if path.startswith("/ketcher/") and self.ketcher_root:
            relative = path[len("/ketcher/") :] or "index.html"
            target = (self.ketcher_root / relative).resolve()
            try:
                target.relative_to(self.ketcher_root.resolve())
            except ValueError:
                self._send(403, b"forbidden", "text/plain")
                return
            if not target.is_file():
                self._send(404, b"not found", "text/plain")
                return
            ctype = self.guess_type(str(target))
            self._send(200, target.read_bytes(), ctype)
            return

        self._send(404, b"not found", "text/plain")

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

    Returns a molblock, or None if the user closed the page without submitting.
    """
    ketcher_root = ensure_ketcher(quiet=quiet)
    if ketcher_root is None and not quiet:
        print(
            "could not fetch Ketcher; falling back to a paste box "
            "(set LIGAND3D_KETCHER_DIR to use a local build)"
        )

    port = port or _free_port()
    done = threading.Event()

    handler = type(
        "_BoundHandler",
        (_Handler,),
        {"result": {}, "done": done, "ketcher_root": ketcher_root},
    )

    try:
        server = socketserver.TCPServer(("127.0.0.1", port), handler)
    except OSError as exc:
        raise Ligand3DError(f"could not bind port {port}: {exc}") from exc

    server.allow_reuse_address = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/"
    if not quiet:
        print(f"sketcher running at {url}")
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

    molblock = handler.result.get("molblock") or ""
    if not molblock.strip():
        smiles = handler.result.get("smiles", "").strip()
        if smiles:
            from rdkit import Chem

            from ..molecule import rdkit_quiet

            with rdkit_quiet():
                mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                from rdkit.Chem import AllChem

                AllChem.Compute2DCoords(mol)
                return Chem.MolToMolBlock(mol)
        return None
    return molblock

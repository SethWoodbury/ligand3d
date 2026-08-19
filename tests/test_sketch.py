"""The browser sketcher, driven the way a browser drives it."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from ligand3d.molecule import from_molblock
from ligand3d.sketch import server as srv


@pytest.fixture
def running_server(monkeypatch):
    """Serve the fallback page so the test needs no 35 MB Ketcher download.

    Each test gets its own port; sharing one across tests races against the
    previous server's shutdown.
    """
    monkeypatch.setattr(srv, "ensure_ketcher", lambda quiet=False: None)

    port = srv._free_port()
    captured: dict[str, str | None] = {"base": f"http://127.0.0.1:{port}"}

    def serve():
        captured["molblock"] = srv.sketch_molecule(
            port=port, open_browser=False, timeout=30, quiet=True
        )

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    for _ in range(50):  # wait for the port to answer
        try:
            urllib.request.urlopen(f"{captured['base']}/status", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("sketch server never came up")

    yield captured

    if thread.is_alive():
        try:
            _post(captured["base"], {"smiles": "C"})  # release the blocking wait
        except Exception:
            pass
    thread.join(timeout=10)


def _post(base: str, payload: dict) -> str:
    request = urllib.request.Request(
        f"{base}/submit",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=5).read().decode()


def test_serves_a_page(running_server):
    page = urllib.request.urlopen(f"{running_server['base']}/", timeout=5).read().decode()
    assert "Paste a structure" in page


def test_status_reports_no_ketcher(running_server):
    body = urllib.request.urlopen(f"{running_server['base']}/status", timeout=5).read().decode()
    assert json.loads(body) == {"ketcher": False}


@pytest.mark.parametrize(
    "path",
    [
        "/ketcher/../../../../etc/passwd",
        "/../../etc/passwd",
        "/etc/passwd",
        "/ketcher/..%2f..%2fetc%2fpasswd",
    ],
)
def test_path_traversal_is_refused(running_server, path):
    """The server reads files off disk, so this guard has to hold."""
    try:
        response = urllib.request.urlopen(f"{running_server['base']}{path}", timeout=5)
    except urllib.error.HTTPError as exc:
        assert exc.code in (403, 404)
        return
    assert b"root:" not in response.read()[:512]


def test_unknown_post_target_is_refused(running_server):
    request = urllib.request.Request(
        f"{running_server['base']}/anything", data=b"{}", method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request, timeout=5)
    assert exc.value.code == 404


def test_submitted_smiles_comes_back_with_stereo_intact(running_server):
    """A drawn stereocenter must survive the trip through the browser bridge."""
    assert json.loads(_post(running_server["base"], {"smiles": "C[C@H](N)C(=O)O", "molblock": ""}))["ok"]

    for _ in range(100):
        if "molblock" in running_server:
            break
        time.sleep(0.1)

    molblock = running_server.get("molblock")
    assert molblock, "server did not return a molblock"
    assert from_molblock(molblock).smiles == "C[C@H](N)C(=O)O"


def test_submitted_molblock_is_passed_through(running_server):
    molblock = """
  Mrv

  3  2  0  0  0  0            999 V2000
    0.0000    0.0000    0.0000 C   0  0
    1.5000    0.0000    0.0000 C   0  0
    2.2500    1.2990    0.0000 O   0  0
  1  2  1  0
  2  3  1  0
M  END
"""
    assert json.loads(_post(running_server["base"], {"molblock": molblock, "smiles": ""}))["ok"]

    for _ in range(100):
        if "molblock" in running_server:
            break
        time.sleep(0.1)

    assert from_molblock(running_server["molblock"]).smiles == "CCO"

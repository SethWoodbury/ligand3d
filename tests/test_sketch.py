"""The browser sketcher, driven the way a browser drives it."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from ligand3d.molecule import from_molblock
from ligand3d.sketch import server as srv


def molblock_for(smiles: str) -> str:
    """A 2D molblock with wedge bonds — what a sketcher actually produces."""
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    Chem.WedgeMolBonds(mol, mol.GetConformer())
    return Chem.MolToMolBlock(mol, kekulize=True)


class TestMolblockHandling:
    """`_to_molblock` must not touch the bytes it was given.

    A MOL file's first line is the molecule-name field and is routinely empty.
    Stripping leading whitespace shifts every later line up one, so the counts
    line lands where the header belongs and nothing can parse the result.
    """

    def test_leading_blank_line_is_preserved(self):
        original = molblock_for("C[C@H](N)C(=O)O")
        assert original.startswith("\n"), "test premise: RDKit emits an empty name line"

        returned = srv._to_molblock({"molblock": original})
        assert returned == original
        assert from_molblock(returned).smiles == "C[C@H](N)C(=O)O"

    def test_smiles_fallback_builds_a_parseable_molblock(self):
        returned = srv._to_molblock({"molblock": "", "smiles": "C[C@H](N)C(=O)O"})
        assert from_molblock(returned).smiles == "C[C@H](N)C(=O)O"

    def test_molblock_wins_over_smiles(self):
        returned = srv._to_molblock(
            {"molblock": molblock_for("CCO"), "smiles": "c1ccccc1"}
        )
        assert from_molblock(returned).smiles == "CCO"

    def test_nothing_submitted_gives_none(self):
        assert srv._to_molblock({"molblock": "   ", "smiles": ""}) is None
        assert srv._to_molblock({}) is None

    def test_unparseable_smiles_fallback_gives_none(self):
        assert srv._to_molblock({"molblock": "", "smiles": "not a molecule"}) is None


@pytest.fixture
def running_server(monkeypatch):
    """Serve the paste-box fallback so the test needs no download.

    Each test gets its own port; sharing one races against the previous
    server's shutdown.
    """
    monkeypatch.setattr(srv, "choose_engine", lambda quiet=False: None)

    port = srv._free_port()
    captured: dict[str, str | None] = {"base": f"http://127.0.0.1:{port}"}

    def serve():
        captured["molblock"] = srv.sketch_molecule(
            port=port, open_browser=False, timeout=30, quiet=True
        )

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()

    for _ in range(50):
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


def _wait_for_molblock(captured: dict) -> str:
    for _ in range(100):
        if captured.get("molblock"):
            return captured["molblock"]
        time.sleep(0.1)
    pytest.fail("server never returned a molblock")


class TestServer:
    def test_serves_the_fallback_page(self, running_server):
        page = (
            urllib.request.urlopen(f"{running_server['base']}/", timeout=5).read().decode()
        )
        assert "Paste a structure" in page

    def test_status_reports_no_engine(self, running_server):
        body = (
            urllib.request.urlopen(f"{running_server['base']}/status", timeout=5)
            .read()
            .decode()
        )
        assert json.loads(body) == {"engine": None}

    @pytest.mark.parametrize(
        "path",
        [
            "/sketcher/../../../../etc/passwd",
            "/../../etc/passwd",
            "/etc/passwd",
            "/sketcher/..%2f..%2fetc%2fpasswd",
        ],
    )
    def test_path_traversal_is_refused(self, running_server, path):
        """The server reads files off disk, so this guard has to hold."""
        try:
            response = urllib.request.urlopen(f"{running_server['base']}{path}", timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code in (403, 404)
            return
        assert b"root:" not in response.read()[:512]

    def test_unknown_post_target_is_refused(self, running_server):
        request = urllib.request.Request(
            f"{running_server['base']}/anything", data=b"{}", method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(request, timeout=5)
        assert exc.value.code == 404

    def test_submitted_smiles_keeps_its_stereo(self, running_server):
        assert json.loads(
            _post(running_server["base"], {"smiles": "C[C@H](N)C(=O)O", "molblock": ""})
        )["ok"]
        assert from_molblock(_wait_for_molblock(running_server)).smiles == "C[C@H](N)C(=O)O"

    def test_submitted_molblock_keeps_its_wedge_stereo(self, running_server):
        """A drawn wedge bond must arrive as a real stereocenter."""
        assert json.loads(
            _post(running_server["base"], {"molblock": molblock_for("C[C@@H](N)C(=O)O")})
        )["ok"]

        molecule = from_molblock(_wait_for_molblock(running_server))
        assert molecule.smiles == "C[C@@H](N)C(=O)O"
        assert dict(molecule.stereo.assigned_centers) == {1: "R"}

    def test_malformed_json_body_is_treated_as_a_raw_molblock(self, running_server):
        raw = molblock_for("CCO")
        request = urllib.request.Request(
            f"{running_server['base']}/submit", data=raw.encode(), method="POST"
        )
        urllib.request.urlopen(request, timeout=5).read()
        assert from_molblock(_wait_for_molblock(running_server)).smiles == "CCO"


class TestEngineSelection:
    def test_ketcher_is_preferred_when_the_user_supplied_a_build(self, tmp_path, monkeypatch):
        build = tmp_path / "ketcher"
        build.mkdir()
        (build / "index.html").write_text("<html></html>")
        monkeypatch.setenv("LIGAND3D_KETCHER_DIR", str(build))

        from ligand3d import config

        config.load_config.cache_clear()
        engine = srv.choose_engine(quiet=True)
        assert engine is not None and engine.name == "ketcher"
        assert engine.bridge.exists()

    def test_bridge_pages_all_exist(self):
        for page in ("bridge_jsme.html", "bridge_ketcher.html", "fallback.html"):
            assert (srv._STATIC / page).is_file(), f"missing {page}"

    def test_jsme_bridge_links_the_loader_it_serves(self):
        """The page and the asset route have to agree on the path."""
        page = (srv._STATIC / "bridge_jsme.html").read_text()
        assert "/sketcher/jsme/jsme.nocache.js" in page

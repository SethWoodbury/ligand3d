"""The browser session: the API the page actually calls."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from ligand3d.sketch import server as srv
from ligand3d.sketch.session import (
    backend_catalog,
    describe_stereo,
    inspect_target,
    next_filename,
    normalize_formats,
)


def molblock_for(smiles: str) -> str:
    """A 2D molblock with wedge bonds — what a sketcher actually produces."""
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    Chem.WedgeMolBonds(mol, mol.GetConformer())
    return Chem.MolToMolBlock(mol, kekulize=True)


class TestTargetInspection:
    """The page shows where files will land before writing anything.

    The filename field is a *base* name now, because one build can write an
    mmCIF, a PDB, an SDF, a trajectory, and a params set.
    """

    def test_resolves_a_base_name_per_format(self, tmp_path):
        info = inspect_target(str(tmp_path), "thing", formats=["cif", "pdb", "sdf"])
        assert info.stem == "thing"
        assert info.formats == ("cif", "pdb", "sdf")
        assert info.will_write == [
            str(tmp_path / "thing.cif"),
            str(tmp_path / "thing.pdb"),
            str(tmp_path / "thing.sdf"),
        ]
        assert info.primary == str(tmp_path / "thing.cif")
        assert info.error is None

    def test_a_typed_extension_is_not_doubled(self, tmp_path):
        """Typing "lig.cif" must not produce "lig.cif.cif"."""
        info = inspect_target(str(tmp_path), "lig.cif", formats=["cif"])
        assert info.stem == "lig"
        assert info.will_write == [str(tmp_path / "lig.cif")]

    def test_an_unknown_extension_is_kept_as_part_of_the_name(self, tmp_path):
        info = inspect_target(str(tmp_path), "my.thing", formats=["cif"])
        assert info.will_write == [str(tmp_path / "my.thing.cif")]

    def test_defaults_when_no_formats_are_given(self, tmp_path):
        assert inspect_target(str(tmp_path), "x").formats == ("cif", "sdf")

    def test_flags_a_directory_that_would_be_created(self, tmp_path):
        info = inspect_target(str(tmp_path / "new" / "deeper"), "a")
        assert info.will_create_directory
        assert info.writable
        assert info.error is None

    def test_reports_existing_files_as_an_overwrite(self, tmp_path):
        (tmp_path / "a.cif").write_text("x")
        info = inspect_target(str(tmp_path), "a", formats=["cif", "sdf"])
        assert info.would_overwrite
        assert str(tmp_path / "a.cif") in info.existing
        assert str(tmp_path / "a.sdf") not in info.existing

    def test_only_selected_formats_count_as_a_clash(self, tmp_path):
        (tmp_path / "a.pdb").write_text("x")
        assert not inspect_target(str(tmp_path), "a", formats=["cif"]).would_overwrite
        assert inspect_target(str(tmp_path), "a", formats=["pdb"]).would_overwrite

    def test_unwritable_location_is_an_error(self):
        assert inspect_target("/proc/nope/deeper", "a").error is not None

    def test_a_file_where_a_directory_should_be_is_an_error(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        assert inspect_target(str(blocker), "a").error is not None

    def test_empty_filename_gets_a_default(self, tmp_path):
        assert inspect_target(str(tmp_path), "   ").stem == "sketch0"

    def test_path_components_in_the_filename_are_stripped(self, tmp_path):
        info = inspect_target(str(tmp_path), "../../escape", formats=["cif"])
        assert info.stem == "escape"
        assert info.will_write == [str(tmp_path / "escape.cif")]


class TestFormatNormalization:
    def test_mmcif_is_an_alias_for_cif(self):
        assert normalize_formats("mmcif") == ("cif",)

    def test_unknown_formats_are_dropped(self):
        assert normalize_formats(["cif", "xyz", "pdb"]) == ("cif", "pdb")

    def test_order_is_preserved_and_repeats_removed(self):
        assert normalize_formats(["pdb", "cif", "pdb"]) == ("pdb", "cif")

    def test_a_missing_key_falls_back_to_the_default(self):
        assert normalize_formats(None) == ("cif", "sdf")

    def test_an_explicitly_empty_list_is_a_dry_run(self):
        """Unticking every format is a request, not an omission."""
        assert normalize_formats([]) == ()
        assert normalize_formats("") == ()

    def test_only_unknown_formats_is_also_empty(self):
        assert normalize_formats(["xyz", "mol2"]) == ()


class TestFilenameSequence:
    """Successive builds should not need the user to invent names."""

    def test_starts_at_zero(self, tmp_path):
        assert next_filename(str(tmp_path)) == "sketch0"

    def test_skips_names_already_taken(self, tmp_path):
        for n in (0, 1, 2, 4):
            (tmp_path / f"sketch{n}.cif").write_text("x")
        assert next_filename(str(tmp_path)) == "sketch3"

    def test_any_known_extension_counts_as_taken(self, tmp_path):
        """A name is taken if any format we write already uses it."""
        (tmp_path / "sketch0.pdb").write_text("x")
        (tmp_path / "sketch1.sdf").write_text("x")
        assert next_filename(str(tmp_path)) == "sketch2"

    def test_honours_a_custom_stem(self, tmp_path):
        (tmp_path / "lig0.cif").write_text("x")
        assert next_filename(str(tmp_path), stem="lig") == "lig1"

    def test_missing_directory_starts_at_zero(self, tmp_path):
        assert next_filename(str(tmp_path / "absent")) == "sketch0"


class TestStereoNarration:
    """What the run log says about stereochemistry."""

    def test_reports_each_centre_with_its_cip_code(self):
        from ligand3d.molecule import from_smiles

        lines = describe_stereo(from_smiles("C[C@@H](O)[C@H](N)C(=O)O"))
        joined = " ".join(lines)
        assert "2 stereocenter(s) defined" in joined
        assert "= R" in joined and "= S" in joined

    def test_reports_a_genuine_ambiguity_as_such(self):
        from ligand3d.molecule import from_smiles

        joined = " ".join(describe_stereo(from_smiles("CC(N)C(=O)O")))
        assert "left undefined" in joined and "ambiguous" in joined

    def test_does_not_call_constrained_bridgeheads_undefined(self):
        """3-quinuclidinone's bridgeheads look stereogenic but cannot vary."""
        from ligand3d.molecule import from_smiles

        joined = " ".join(describe_stereo(from_smiles("O=C1CN2CCC1CC2")))
        assert "fixed by the ring system" in joined
        assert "left undefined" not in joined

    def test_reports_double_bond_geometry_as_e_z_with_cis_trans(self):
        from ligand3d.molecule import from_smiles

        joined = " ".join(describe_stereo(from_smiles(r"C/C=C/C(=O)O")))
        assert "double bond(s) with defined geometry" in joined
        assert "= E (trans)" in joined

        joined = " ".join(describe_stereo(from_smiles(r"OC(=O)/C=C\\C(=O)O")))
        assert "= Z (cis)" in joined

    def test_refuses_cis_trans_where_it_does_not_apply(self):
        """Tamoxifen is unambiguously (Z) but tetrasubstituted, so "cis" is
        meaningless — older literature calls the same geometry "trans"."""
        from ligand3d.molecule import from_smiles

        joined = " ".join(describe_stereo(from_smiles(
            r"CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1")))
        assert "= Z" in joined
        assert "cis/trans does not apply" in joined
        assert "(cis)" not in joined


def test_backend_catalog_reports_capabilities():
    catalog = {b["id"]: b for b in backend_catalog()}
    assert "mmff94" in catalog and catalog["mmff94"]["ready"]
    # The capability that drives the whole registry.
    assert catalog["gfn2"]["takes_charge"] and catalog["gfn2"]["supports_solvation"]
    assert not catalog["mace-off"]["takes_charge"]
    for entry in catalog.values():
        assert entry["ready"] or entry["reason"], f"{entry['id']} unavailable with no reason"


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A running session server, serving the paste-box fallback."""
    monkeypatch.setattr(srv, "choose_engine", lambda quiet=False: None)

    server, port = srv.serve(
        port=0,
        open_browser=False,
        quiet=True,
        block=False,
        defaults={"directory": str(tmp_path), "filename": "sketch0",
                  "backend": "mmff94", "threads": 2},
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            urllib.request.urlopen(f"{base}/api/config", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    else:
        server.shutdown(); server.server_close()
        pytest.fail("session server never came up")

    yield base, tmp_path

    server.shutdown()
    server.server_close()


def _get(base: str, path: str):
    return json.load(urllib.request.urlopen(base + path, timeout=20))


def _post(base: str, path: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=120)
        return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def _run(base: str, smiles: str, settings: dict, overwrite: bool = False) -> dict:
    code, data = _post(
        base,
        "/api/build",
        {"molblock": molblock_for(smiles), "settings": settings, "overwrite": overwrite},
    )
    if code != 200:
        return {"http": code, **data}
    for _ in range(600):
        job = _get(base, f"/api/job/{data['job_id']}")
        if job["state"] in ("done", "error"):
            return {"http": 200, **job}
        time.sleep(0.1)
    pytest.fail("job never finished")


def _settings(tmp_path, **overrides) -> dict:
    base = {
        "directory": str(tmp_path),
        "filename": "sketch0",
        "formats": ["cif", "sdf"],
        "backend": "mmff94",
        "protonation": "as-drawn",
        "n_confs": 1,
        "conf_method": "rdkit",
        "stereo_mode": "require",
        "threads": 2,
    }
    base.update(overrides)
    return base


class TestSessionAPI:
    def test_config_carries_the_defaults(self, session):
        base, tmp_path = session
        cfg = _get(base, "/api/config")
        assert cfg["defaults"]["directory"] == str(tmp_path)

    def test_backends_are_served_separately_from_the_defaults(self, session):
        """The two halves used to be one call, which made the editor wait on a
        network filesystem. See TestTheEditorNeverWaitsOnTheFilesystem."""
        base, _ = session
        data = _get(base, "/api/backends")
        assert any(b["id"] == "mmff94" for b in data["backends"])

    def test_serves_the_app_page_even_with_no_editor(self, session):
        """One page for every case: with no editor it shows a paste box, and the
        settings panel and run log still work."""
        base, _ = session
        page = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "Run log" in page and "Backend chain" in page
        assert "pasteBox" in page

    def test_a_pasted_smiles_builds(self, session):
        """The fallback path: no molblock, just a SMILES string in the box."""
        base, tmp_path = session
        code, data = _post(
            base,
            "/api/build",
            {"molblock": "C[C@H](N)C(=O)O", "settings": _settings(tmp_path)},
        )
        assert code == 200, data
        for _ in range(600):
            job = _get(base, f"/api/job/{data['job_id']}")
            if job["state"] in ("done", "error"):
                break
            time.sleep(0.1)
        assert job["state"] == "done", job
        assert any("pasted SMILES" in e["text"] for e in job["log"])
        assert len(job["result"]["stereocenters"]) == 1

    def test_next_name_endpoint(self, session):
        base, tmp_path = session
        (tmp_path / "sketch0.cif").write_text("x")
        query = urllib.parse.urlencode({"directory": str(tmp_path), "stem": "sketch"})
        assert _get(base, f"/api/next-name?{query}")["filename"] == "sketch1"

    def test_check_path_endpoint(self, session):
        base, tmp_path = session
        _, info = _post(base, "/api/check-path",
                        {"directory": str(tmp_path / "sub"), "filename": "a",
                         "formats": ["cif", "pdb"]})
        assert info["will_create_directory"]
        assert len(info["will_write"]) == 2

    def test_build_writes_the_files_and_logs_stereo(self, session):
        base, tmp_path = session
        job = _run(base, "C[C@@H](O)[C@H](N)C(=O)O", _settings(tmp_path))

        assert job["state"] == "done", job
        assert (tmp_path / "sketch0.cif").exists()
        assert (tmp_path / "sketch0.sdf").exists()

        text = " ".join(entry["text"] for entry in job["log"])
        assert "2 stereocenter(s) defined" in text
        assert str(tmp_path / "sketch0.cif") in text
        assert job["result"]["n_conformers"] == 1
        assert len(job["result"]["stereocenters"]) == 2

    def test_two_builds_in_one_session(self, session):
        base, tmp_path = session
        first = _run(base, "CCO", _settings(tmp_path, filename="sketch0"))
        second = _run(base, "O=C1CN2CCC1CC2", _settings(tmp_path, filename="sketch1"))
        assert first["state"] == "done" and second["state"] == "done"
        assert (tmp_path / "sketch0.cif").exists()
        assert (tmp_path / "sketch1.cif").exists()

    def test_overwrite_needs_confirmation_then_proceeds(self, session):
        base, tmp_path = session
        assert _run(base, "CCO", _settings(tmp_path))["state"] == "done"

        blocked = _run(base, "CCC", _settings(tmp_path))
        assert blocked["http"] == 409
        assert blocked["needs_confirmation"] == "overwrite"
        assert blocked["target"]["existing"]

        confirmed = _run(base, "CCC", _settings(tmp_path), overwrite=True)
        assert confirmed["state"] == "done"

    def test_empty_submission_is_rejected(self, session):
        base, _ = session
        code, data = _post(base, "/api/build", {"molblock": "   ", "settings": {}})
        assert code == 400 and "error" in data

    def test_two_fragments_are_reported_in_the_log(self, session):
        base, tmp_path = session
        job = _run(base, "CCO.CCO", _settings(tmp_path))
        assert job["state"] == "error"
        text = " ".join(entry["text"] for entry in job["log"])
        assert "2 separate fragments" in text
        assert "disconnected fragments" in job["error"]

    def test_undefined_stereo_is_reported_in_the_log(self, session):
        base, tmp_path = session
        job = _run(base, "CC(N)C(=O)O", _settings(tmp_path))
        assert job["state"] == "error"
        # The message names the kind of ambiguity and points at the preview,
        # because a bare atom index is not actionable on its own.
        assert "stereochemistry is undefined" in job["error"]
        assert "stereocenter with no configuration" in job["error"]
        assert "preview" in job["error"]

    def test_protonation_mode_reaches_the_pipeline(self, session):
        base, tmp_path = session
        pytest.importorskip("dimorphite_dl")
        job = _run(
            base,
            "NCC1(CC(=O)O)CCCCC1",
            _settings(tmp_path, protonation="ph", ph=7.4),
        )
        assert job["state"] == "done", job
        built = next(iter(Chem.SDMolSupplier(str(tmp_path / "sketch0.sdf"), removeHs=False)))
        charges = [a.GetFormalCharge() for a in built.GetAtoms()]
        assert any(c > 0 for c in charges) and any(c < 0 for c in charges)

    def test_conformer_count_reaches_the_pipeline(self, session):
        base, tmp_path = session
        job = _run(base, "CCCCCCCCO", _settings(tmp_path, n_confs=6))
        assert job["state"] == "done"
        assert job["result"]["n_conformers"] > 1

    def test_unknown_backend_is_an_error_not_a_crash(self, session):
        base, tmp_path = session
        job = _run(base, "CCO", _settings(tmp_path, backend="not-a-backend"))
        assert job["state"] == "error"
        assert "unknown backend" in job["error"]

    def test_directory_is_created_when_missing(self, session):
        base, tmp_path = session
        target = tmp_path / "made" / "here"
        job = _run(base, "CCO", _settings(tmp_path, directory=str(target)))
        assert job["state"] == "done"
        assert (target / "sketch0.cif").exists()

    @pytest.mark.parametrize(
        "path",
        [
            "/sketcher/../../../../etc/passwd",
            "/../../etc/passwd",
            "/etc/passwd",
            "/sketcher/..%2f..%2fetc%2fpasswd",
        ],
    )
    def test_path_traversal_is_refused(self, session, path):
        base, _ = session
        try:
            response = urllib.request.urlopen(base + path, timeout=5)
        except urllib.error.HTTPError as exc:
            assert exc.code in (403, 404)
            return
        assert b"root:" not in response.read()[:512]

    def test_unknown_endpoints_404(self, session):
        base, _ = session
        for method, path in (("GET", "/nope"), ("POST", "/api/nope")):
            request = urllib.request.Request(
                base + path, data=b"{}" if method == "POST" else None, method=method
            )
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(request, timeout=5)
            assert exc.value.code == 404

    def test_missing_job_404s(self, session):
        base, _ = session
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/api/job/999999", timeout=5)
        assert exc.value.code == 404


class TestStaticAssets:
    def test_app_page_exists_and_wires_the_loader(self):
        page = (srv._STATIC / "app.html").read_text()
        assert "/sketcher/jsme/jsme.nocache.js" in page
        # The page must explain how to draw a dashed bond, not just a wedge.
        assert "dashed" in page.lower()
        for endpoint in ("/api/config", "/api/build", "/api/check-path", "/api/next-name"):
            assert endpoint in page, f"page never calls {endpoint}"

    def test_no_reference_to_a_removed_editor(self):
        """Ketcher was removed; nothing should still advertise it.

        Matched on a word boundary rather than as a substring: "sketcher" —
        the name of the thing this page *is* — contains "ketcher", so a plain
        `in` check passed only for as long as the page never said the word.
        """
        import re

        text = srv.APP_PAGE.read_text()
        hits = [m.group() for m in re.finditer(r"\bketcher\b", text, re.IGNORECASE)]
        assert hits == [], f"the removed Ketcher editor is still referenced: {hits}"


class TestAppJavaScript:
    """The page's logic is not exercised by the Python tests, so at minimum it
    must parse. A typo in here is otherwise only visible in a browser console."""

    @staticmethod
    def _inline_script() -> str:
        import re

        html = (srv._STATIC / "app.html").read_text()
        blocks = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S)
        assert blocks, "app.html has no inline script"
        return blocks[-1]

    def test_parses(self, tmp_path):
        import shutil
        import subprocess

        node = shutil.which("node") or shutil.which("nodejs")
        if node is None:
            pytest.skip("node not available to parse the script")

        script = tmp_path / "app.js"
        script.write_text(self._inline_script())
        result = subprocess.run(
            [node, "--check", str(script)], capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, result.stderr

    def test_every_referenced_element_id_exists(self):
        """`el("typo")` returns null and fails silently at runtime."""
        import re

        html = (srv._STATIC / "app.html").read_text()
        defined = set(re.findall(r'\bid="([A-Za-z0-9_]+)"', html))
        referenced = set(re.findall(r'\bel\("([A-Za-z0-9_]+)"\)', self._inline_script()))
        # pasteBox is injected by script, not present in the markup.
        injected = {"pasteBox"}
        missing = referenced - defined - injected
        assert not missing, f"app.html references undefined ids: {sorted(missing)}"


class TestTheEditorNeverWaitsOnTheFilesystem:
    """The sketcher took six seconds to show a canvas, twice.

    Both times the cause was the same shape: the page cannot mount the editor
    until /api/config answers, and /api/config was doing work that touches the
    model store. That store is on a network filesystem, so resolving twenty-odd
    checkpoints costs seconds whenever its attribute cache is cold — which is
    once per login, exactly when someone is opening the page.

    A wall-clock assertion alone would not have caught it: on a warm cache the
    same code returns in milliseconds and the test passes while the user waits.
    So the guarantee here is structural. /api/config must not resolve weights
    at all, and the timing test below makes weight resolution genuinely slow to
    prove the split holds rather than trusting a fast machine.
    """

    EDITOR_BUDGET = 1.0

    def test_config_does_not_resolve_weights(self, session, monkeypatch):
        from ligand3d import config as cfg

        def explode(key):
            raise AssertionError(f"/api/config resolved weights for {key!r}")

        monkeypatch.setattr(cfg, "resolve_weights", explode)
        base, _ = session
        payload = _get(base, "/api/config")
        assert "defaults" in payload

    def test_config_is_fast_even_when_the_model_store_is_slow(
        self, session, monkeypatch
    ):
        """Simulates a cold network filesystem, which is the real failure."""
        from ligand3d import config as cfg

        real = cfg.resolve_weights

        def crawl(key):
            time.sleep(0.5)
            return real(key)

        monkeypatch.setattr(cfg, "resolve_weights", crawl)
        base, _ = session

        started = time.perf_counter()
        _get(base, "/api/config")
        elapsed = time.perf_counter() - started
        assert elapsed < self.EDITOR_BUDGET, (
            f"/api/config took {elapsed:.2f}s with a slow model store; the editor "
            "cannot mount until this returns"
        )

    def test_config_carries_what_the_editor_needs_and_no_more(self, session):
        """Nothing that touches the filesystem may be served from here.

        Named rather than an exact-set match: the point is that the expensive
        keys stay out, not that the cheap set never grows. Probing for weights
        or for sbatch costs seconds on a cold network filesystem, and this is
        the call the editor waits on before it can be drawn in.
        """
        base, _ = session
        payload = _get(base, "/api/config")
        forbidden = {"backends", "solvents", "slurm", "container", "templates", "models"}
        assert not (forbidden & set(payload)), (
            f"{forbidden & set(payload)} moved back onto the editor's critical path"
        )
        assert {"engine", "defaults"} <= set(payload)

    def test_the_expensive_half_is_still_served(self, session):
        base, _ = session
        payload = _get(base, "/api/backends")
        assert payload["backends"] and "solvents" in payload and "slurm" in payload

    def test_config_is_quick_in_absolute_terms(self, session):
        """Generous, because it shares a machine with everything else."""
        base, _ = session
        _get(base, "/api/config")  # warm any first-call cost
        started = time.perf_counter()
        for _ in range(5):
            _get(base, "/api/config")
        per_call = (time.perf_counter() - started) / 5
        assert per_call < 0.25, f"/api/config averaged {per_call:.3f}s"


class TestWeightResolutionIsCached:
    """Resolving a checkpoint stats a network filesystem; doing it per request
    is what made the page slow. Checkpoints do not move mid-session."""

    def test_repeat_lookups_do_not_hit_the_filesystem_again(self):
        from ligand3d import config as cfg

        cfg.resolve_weights.cache_clear()
        first = cfg.resolve_weights("mace-off")
        info = cfg.resolve_weights.cache_info()
        assert info.misses == 1

        second = cfg.resolve_weights("mace-off")
        assert cfg.resolve_weights.cache_info().hits >= 1
        assert second is first

    def test_the_catalog_resolves_each_model_once(self):
        from ligand3d import config as cfg
        from ligand3d.catalog import build_catalog

        cfg.resolve_weights.cache_clear()
        build_catalog()
        after_first = cfg.resolve_weights.cache_info().misses
        build_catalog()
        assert cfg.resolve_weights.cache_info().misses == after_first

    def test_the_cache_can_be_cleared_if_a_checkpoint_moves(self):
        from ligand3d import config as cfg

        cfg.resolve_weights("mace-off")
        cfg.resolve_weights.cache_clear()
        assert cfg.resolve_weights.cache_info().currsize == 0


class TestWhereABuildRuns:
    """The page picks one of three places to run. The wire format is a single
    value rather than two booleans, because "in a container" and "on a GPU
    node" are alternatives and setting both is not a thing."""

    def test_local_is_the_default(self, session):
        base, tmp_path = session
        job = _run(base, "CCO", _settings(tmp_path))
        assert job["state"] == "done"
        assert job.get("slurm_job_id") is None
        assert not (job.get("result") or {}).get("container")

    def test_an_unknown_place_falls_back_to_local_rather_than_failing(self, session):
        """A client sending something unexpected should still get a build."""
        base, tmp_path = session
        job = _run(base, "CCO", _settings(tmp_path, where="somewhere-else"))
        assert job["state"] == "done"

    def test_asking_for_a_container_without_apptainer_fails_loudly(
        self, session, monkeypatch
    ):
        """It must not quietly run locally and pretend it honoured the request:
        the whole point of choosing a container is that the local environment
        cannot run the method."""
        import ligand3d.container as container_mod

        monkeypatch.setattr(container_mod, "apptainer_available", lambda: False)
        base, tmp_path = session
        job = _run(base, "CCO", _settings(tmp_path, where="container"))
        assert job["state"] == "error"
        assert "apptainer" in (job["error"] or "")

    def test_backends_report_what_a_container_could_run(self, session):
        """Without this the menu would keep greying out exactly the models the
        container exists for."""
        base, _ = session
        data = _get(base, "/api/backends")
        assert set(data["container"]) == {"available", "images"}
        assert all("in_container" in b for b in data["backends"])

def test_free_port_prefers_a_stable_number(monkeypatch):
    """A random port means re-making the SSH tunnel every session."""
    import socket as socket_module

    from ligand3d.sketch import server

    with socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        known_free = probe.getsockname()[1]

    monkeypatch.setattr(server, "PREFERRED_PORTS", (known_free,))
    assert server._free_port() == known_free


def test_free_port_falls_through_when_the_preferred_one_is_taken(monkeypatch):
    import socket as socket_module

    from ligand3d.sketch import server

    with socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        taken = held.getsockname()[1]
        monkeypatch.setattr(server, "PREFERRED_PORTS", (taken,))
        assert server._free_port() != taken


def test_ssh_hint_only_appears_in_a_remote_session(monkeypatch):
    from ligand3d.sketch.server import _ssh_hint

    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    assert _ssh_hint(8765) is None

    monkeypatch.setenv("SSH_CONNECTION", "10.0.0.1 22 10.0.0.2 22")
    hint = _ssh_hint(8765)
    assert hint is not None
    # The port has to appear on both sides, or the forward lands nowhere.
    assert "-L 8765:127.0.0.1:8765" in hint

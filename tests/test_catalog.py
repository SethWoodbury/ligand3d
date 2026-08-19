"""The method catalog, MACE-POLAR registration, and packaging sanity."""

from __future__ import annotations

import pytest

from ligand3d.catalog import build_catalog, summarize
from ligand3d.config import MODELS, MODELS_BY_KEY
from ligand3d.minimize import all_backends, get_backend


class TestCatalog:
    def test_covers_every_registered_backend(self):
        catalog = {m.id for m in build_catalog()}
        registered = {b.caps.name for b in all_backends()}
        assert catalog == registered

    def test_classical_methods_are_described_too(self):
        """The catalog is not just the neural potentials."""
        by_id = {m.id: m for m in build_catalog()}
        for name in ("mmff94", "uff", "gfn1", "gfn2", "gfnff"):
            entry = by_id[name]
            assert entry.speed, f"{name} has no speed"
            assert entry.accuracy, f"{name} has no accuracy note"
            assert entry.family and entry.family != entry.kind

    def test_charge_handling_is_explicit_or_implicit(self):
        for method in build_catalog():
            assert method.charge in ("explicit", "implicit")

    def test_charge_matches_the_backend_capability(self):
        by_id = {m.id: m for m in build_catalog()}
        for backend in all_backends():
            expected = "explicit" if backend.caps.takes_charge else "implicit"
            assert by_id[backend.caps.name].charge == expected

    def test_solvent_column_matches_capability(self):
        by_id = {m.id: m for m in build_catalog()}
        assert by_id["gfn2"].solvent == "ALPB"
        assert by_id["mace-off"].solvent == "no"

    def test_checkpointed_models_report_their_file(self):
        for method in build_catalog():
            if method.id in MODELS_BY_KEY:
                assert method.weights_file, f"{method.id} has no weights filename"
                assert method.repo, f"{method.id} has no upstream repo"

    def test_resolved_paths_point_at_the_named_file(self):
        for method in build_catalog():
            if method.weights_path:
                assert method.weights_path.endswith(method.weights_file)

    def test_unavailable_methods_always_give_a_reason(self):
        for method in build_catalog():
            assert method.ready or method.reason, f"{method.id} unavailable with no reason"

    def test_summary_reports_one_weight_root(self):
        report = summarize()
        assert report.weight_roots
        for method in report.methods:
            if method.weights_path:
                assert any(method.weights_path.startswith(r) for r in report.weight_roots)

    def test_json_is_serializable(self):
        import json

        json.dumps(summarize().to_json())


class TestMacePolar:
    """MACE-POLAR is real: it loads and evaluates once its fork is installed.

    It was previously listed as unloadable. What it actually needs is a patched
    MACE fork that installs *as* mace-torch, so it is a third mutually exclusive
    environment rather than something that sits beside stock mace.
    """

    @pytest.mark.parametrize("key", ["mace-polar-s", "mace-polar", "mace-polar-l"])
    def test_registered(self, key):
        assert key in MODELS_BY_KEY
        backend = get_backend(key)
        assert backend.caps.kind == "mlff"

    def test_no_longer_listed_as_unloadable(self):
        from ligand3d.config import UNSUPPORTED_MODELS

        assert "mace-polar" not in UNSUPPORTED_MODELS

    def test_availability_explains_the_fork_requirement(self):
        import importlib.util

        backend = get_backend("mace-polar")
        availability = backend.available()
        if importlib.util.find_spec("graph_longrange") is None:
            assert not availability
            assert "graph_longrange" in availability.reason
            assert "separate venv" in availability.hint

    def test_weights_resolve_on_this_machine(self):
        from ligand3d.config import find_model_weights

        path = find_model_weights("mace-polar")
        if path is None:
            pytest.skip("POLAR weights not present here")
        assert path.name == "MACE-POLAR-1-M.model"
        assert "mace-polar-1-beta" in str(path)


class TestModelMetadata:
    def test_every_model_names_its_upstream_and_training(self):
        for spec in MODELS:
            assert spec.repo, f"{spec.key} has no repo"
            assert spec.training, f"{spec.key} says nothing about training data"

    def test_filename_is_the_last_pattern(self):
        for spec in MODELS:
            assert spec.filename and "/" not in spec.filename

    def test_spin_awareness_is_recorded(self):
        """OMol25-trained fairchem models take spin; MACE-OFF does not."""
        assert MODELS_BY_KEY["esen"].spin_aware
        assert MODELS_BY_KEY["uma-s"].spin_aware
        assert not MODELS_BY_KEY["mace-off"].spin_aware

    def test_charge_handling_string(self):
        assert MODELS_BY_KEY["mace-omol"].charge_handling == "explicit"
        assert MODELS_BY_KEY["mace-off"].charge_handling == "implicit"


class TestPackaging:
    """The extras are mutually exclusive; uv has to be told, or CI fails."""

    @staticmethod
    def _pyproject() -> dict:
        import pathlib
        import tomllib

        root = pathlib.Path(__file__).resolve().parents[1]
        return tomllib.loads((root / "pyproject.toml").read_text())

    def test_the_conflict_is_declared(self):
        conflicts = self._pyproject()["tool"]["uv"]["conflicts"]
        pairs = {frozenset(entry["extra"] for entry in group) for group in conflicts}
        assert frozenset({"mace", "fairchem"}) in pairs

    def test_fairchem_is_gated_by_python_version(self):
        """fairchem-core supports 3.11-3.13 while ligand3d runs on 3.10+."""
        extras = self._pyproject()["project"]["optional-dependencies"]
        entry = next(d for d in extras["fairchem"] if d.startswith("fairchem-core"))
        assert "python_version" in entry

    def test_ci_never_lets_uv_re_resolve(self):
        """`uv run` re-syncs from pyproject and tries to satisfy every extra."""
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "ci.yml").read_text()
        assert "uv run pytest" not in workflow
        assert "UV_NO_SYNC" in workflow


class TestModelsPage:
    def test_the_page_exists_and_calls_the_api(self):
        from ligand3d.sketch import server as srv

        assert srv.MODELS_PAGE.is_file()
        page = srv.MODELS_PAGE.read_text()
        assert "/api/models" in page
        assert "Seth M. Woodbury" in page

    def test_the_app_links_to_it(self):
        from ligand3d.sketch import server as srv

        assert '/models' in srv.APP_PAGE.read_text()

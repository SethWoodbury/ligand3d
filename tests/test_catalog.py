"""The method catalog, MACE-POLAR registration, and packaging sanity."""

from __future__ import annotations

import re
from pathlib import Path

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
    """MACE-POLAR is real: it loads and evaluates, and needs one extra package.

    It was once listed as unloadable, then as needing a patched MACE fork that
    installed *as* mace-torch — a third mutually exclusive environment. Neither
    is true any more: mace-torch 0.3.16 ships PolarMACE upstream, so POLAR sits
    beside every other MACE model and only `graph_longrange`, which is not on
    PyPI, is still separate.
    """

    @pytest.mark.parametrize("key", ["mace-polar-s", "mace-polar", "mace-polar-l"])
    def test_registered(self, key):
        assert key in MODELS_BY_KEY
        backend = get_backend(key)
        assert backend.caps.kind == "mlff"

    def test_no_longer_listed_as_unloadable(self):
        from ligand3d.config import UNSUPPORTED_MODELS

        assert "mace-polar" not in UNSUPPORTED_MODELS

    def test_availability_names_the_package_that_is_missing(self):
        """However it is unavailable, the hint must say what to install.

        The reason varies with what is missing: with no torch at all the module
        check fires first and says so, and only once mace is present does the
        graph_longrange check get a chance. The hint is the POLAR-specific part
        and is what actually tells someone what to do.
        """
        import importlib.util

        backend = get_backend("mace-polar")
        availability = backend.available()
        if importlib.util.find_spec("graph_longrange") is not None:
            return  # installed here; nothing to explain
        assert not availability
        assert availability.reason
        assert "graph_longrange" in availability.hint

    def test_the_hint_does_not_still_demand_a_separate_environment(self):
        """The fork was merged upstream, so that advice is now wrong.

        Worth pinning: this text told people to build a third virtualenv, which
        is a real afternoon of work to follow for no reason.
        """
        hint = get_backend("mace-polar").install_hint()
        assert "separate venv" not in hint
        assert "fork" not in hint
        assert "--no-deps" in hint
        assert "graph_longrange" in hint

    def test_graph_longrange_is_the_reason_once_mace_is_present(self):
        """With stock mace installed, the specific missing piece is named."""
        import importlib.util

        if importlib.util.find_spec("mace") is None:
            pytest.skip("mace not installed")
        if importlib.util.find_spec("graph_longrange") is not None:
            pytest.skip("the POLAR fork is installed here")
        availability = get_backend("mace-polar").available()
        assert "graph_longrange" in availability.reason

    def test_weights_resolve_on_this_machine(self):
        from ligand3d.config import find_model_weights

        path = find_model_weights("mace-polar")
        if path is None:
            pytest.skip("POLAR weights not present here")
        assert path.name == "MACE-POLAR-1-M.model"
        assert "mace-polar-1" in str(path)


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
    def _text() -> str:
        """Read pyproject as text.

        Deliberately not parsed: tomllib only exists from 3.11, and this suite
        runs on 3.10 too. These are presence checks, so text is enough.
        """
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        return (root / "pyproject.toml").read_text()

    def test_the_conflict_is_declared(self):
        text = self._text()
        assert "[tool.uv]" in text
        assert "conflicts" in text
        assert 'extra = "mace"' in text and 'extra = "fairchem"' in text

    def test_fairchem_is_gated_by_python_version(self):
        """fairchem-core supports 3.11-3.13 while ligand3d runs on 3.10+."""
        text = self._text()
        line = next(
            ln for ln in text.splitlines() if ln.strip().startswith('"fairchem-core')
        )
        assert "python_version" in line

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


class TestTimingHonesty:
    """Speeds must be marked measured or estimated, and never silently guessed.

    The first version of this table carried estimates presented as measurements
    and had MACE-POLAR at 4/8/16 s when it actually takes 23/57/114 s — wrong by
    six- to seven-fold, in the flattering direction. The `measured` flag exists
    so that mistake is visible rather than plausible.
    """

    def test_every_model_declares_a_speed(self):
        for spec in MODELS:
            assert spec.speed, f"{spec.key} has no speed"

    def test_estimates_say_so_in_the_text(self):
        for spec in MODELS:
            if not spec.measured:
                assert "estimate" in spec.speed.lower(), (
                    f"{spec.key} is not measured but does not say so: {spec.speed!r}"
                )

    def test_measured_values_are_not_hedged(self):
        for spec in MODELS:
            if spec.measured:
                assert "estimate" not in spec.speed.lower()
                assert "~" not in spec.speed, f"{spec.key} claims measured but hedges"

    def test_models_that_run_here_are_measured(self):
        """Anything installable in this environment should have real numbers."""
        for spec in MODELS:
            if spec.family in ("mace", "mace-polar"):
                assert spec.measured, f"{spec.key} could be timed but was not"

    def test_polar_is_recorded_as_much_slower_than_plain_mace(self):
        """The long-range electrostatics cost roughly an order of magnitude."""
        plain = MODELS_BY_KEY["mace-off"]
        polar = MODELS_BY_KEY["mace-polar"]
        assert plain.measured and polar.measured

        def seconds(text: str) -> float:
            import re

            value = float(re.search(r"[\d.]+", text).group())
            return value / 1000 if "ms" in text else value

        assert seconds(polar.speed) > 5 * seconds(plain.speed)

    def test_the_catalog_carries_the_flag(self):
        by_id = {m.id: m for m in build_catalog()}
        assert by_id["mace-off"].measured
        assert by_id["mmff94"].measured

    def test_every_model_is_measured_now(self):
        """The fairchem models used to be the exception, timed only by guess
        because they cannot run in this virtualenv. They can be timed in the
        container that runs them, and were: nothing in the table is a guess.

        If a new model is added, this fails until it has been run — which is
        the point. An estimate that reads like a measurement is worse than a
        blank.
        """
        unmeasured = [spec.key for spec in MODELS if not spec.measured]
        assert unmeasured == [], (
            f"{unmeasured} have estimated speeds. Time them with "
            f"--container, or mark the speed as an estimate."
        )

    def test_slow_loading_models_record_it(self):
        """aimnet2 spends 14 s constructing itself and 0.6 s minimizing."""
        by_id = {m.id: m for m in build_catalog()}
        assert by_id["aimnet2"].load_seconds > 5


class TestTheReadmeMatchesTheCode:
    """Documentation drifts silently, and a wrong number in a README is worse
    than no number — it gets believed and quoted.

    These caught three real drifts: the backend table listed gfn2 as slower
    than gfn1 (backwards, and 3x off), the intro said builds write a `.pdb`
    long after mmCIF became the default, and two flags were documented that no
    longer exist.
    """

    ROOT = Path(__file__).resolve().parents[1]
    README = ROOT / "README.md"
    #: The speed table moved here when the README was split. A guard that does
    #: not follow its subject stops being a guard.
    METHODS_DOC = ROOT / "docs" / "methods.md"

    def test_the_backend_speed_table_matches_the_catalog(self):
        text = self.METHODS_DOC.read_text()
        speeds = {m.id: m.speed for m in build_catalog()}
        for backend in ("mmff94", "uff", "gfnff", "gfn2", "gfn1"):
            row = re.search(rf"^\| `{re.escape(backend)}` \|.*$", text, re.M)
            assert row, f"{backend} has no row in the backends table"
            quoted = row.group(0).rsplit("|", 2)[1].strip()
            assert quoted == speeds[backend], (
                f"README says {backend} takes {quoted}, catalog says {speeds[backend]}"
            )

    def test_every_documented_flag_exists(self):
        """A README flag that no longer exists fails as soon as it is copied.

        Introspects the command objects rather than parsing `--help`. The help
        text is rendered by rich and wraps to the terminal, so parsing it
        passes on a wide terminal and collapses on a narrow one — which is
        exactly what this test did on CI the first time.
        """
        import typer

        from ligand3d.cli import app

        command = typer.main.get_command(app)
        known = {
            option
            for sub in command.commands.values()
            for param in sub.params
            for option in param.opts
            if option.startswith("--")
        }
        assert len(known) > 40, f"introspection found only {len(known)} options"

        # Flags belonging to other tools that legitimately appear in examples.
        foreign = {
            "--index-url", "--local-dir", "--no-deps", "--strict", "--cfg",
            "--with-test-dataset", "--keep-names", "--help",
        }
        text = self.README.read_text()
        used = set()
        for block in re.findall(r"```bash\n(.*?)```", text, re.S):
            for line in block.replace("\\\n", " ").splitlines():
                if line.strip().startswith("ligand3d "):
                    used |= set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", line))
        unknown = sorted(used - known - foreign)
        assert unknown == [], f"documented but not real options: {unknown}"

    def test_the_flag_check_would_notice_a_bad_flag(self):
        """The check above is only worth having if it can fail."""
        import typer

        from ligand3d.cli import app

        command = typer.main.get_command(app)
        known = {
            option
            for sub in command.commands.values()
            for param in sub.params
            for option in param.opts
        }
        assert "--confs" in known
        assert "--no-such-flag" not in known

    def test_the_intro_does_not_claim_pdb_is_the_default(self):
        """mmCIF and SDF are what a bare `build` writes; PDB is opt-in."""
        head = self.README.read_text()[:1400]
        assert "mmCIF" in head and "default" in head
        assert "PDB by default" not in head


def test_polar_models_have_a_charge_channel():
    """MACE-POLAR reads atoms.info['charge'] — measured, not assumed.

    Setting a charge on ethanol moves the MACE-POLAR-1-M energy by +1.41 eV
    (-1) and +11.71 eV (+1), while mace-mh under the same test does not move
    at all. Flagging it False made ligand3d refuse carboxylates and
    zwitterions on the one MACE model built for exactly that case.
    """
    from ligand3d.config import MODELS

    polar = [m for m in MODELS if m.family == "mace-polar"]
    assert polar, "the polar models went missing"
    assert all(m.takes_charge for m in polar)


def test_polar_weights_live_in_the_released_directory():
    """The checkpoints are the public release, not the old -beta staging dir."""
    from ligand3d.config import MODELS

    for spec in (m for m in MODELS if m.family == "mace-polar"):
        assert "mace-polar-1-beta" not in str(spec.patterns)
        assert spec.repo == "ACEsuit/mace-polar-1"

"""Running a build inside an Apptainer image, on this machine.

The eSEN/UMA/AllScAIP models cannot share a virtualenv with MACE — the e3nn pin
is a genuine conflict, not a missing install. The images used for GPU
submission each carry one side of the split, so the model that is unavailable
here is available in a container already on disk. These tests cover the routing
and the refusals; actually invoking apptainer is left to the marked test at the
bottom, since not every machine has it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ligand3d.container import (
    ContainerError,
    available,
    image_for,
    would_help,
)
from ligand3d.molecule import from_smiles
from ligand3d.pipeline import Settings
from ligand3d.slurm import QUANTUM_CHEM_SIF, UMA_SIF

needs_apptainer = pytest.mark.skipif(
    shutil.which("apptainer") is None and shutil.which("singularity") is None,
    reason="apptainer not installed",
)


class TestImageChoice:
    """Same split as the SLURM path, and for the same reason."""

    def test_fairchem_models_get_the_uma_image(self):
        for name in ("esen", "uma-s", "allscaip"):
            assert image_for(name) == UMA_SIF, name

    def test_mace_models_get_the_quantum_chem_image(self):
        for name in ("mace-off", "mace-polar", "aimnet2"):
            assert image_for(name) == QUANTUM_CHEM_SIF, name

    def test_a_chain_mixing_the_families_is_refused(self):
        """No single image can satisfy both, so say so rather than fail inside."""
        with pytest.raises(ContainerError, match="incompatible e3nn"):
            image_for("mace-off,esen")

    def test_a_classical_prefix_follows_the_potential(self):
        assert image_for("mmff94,esen") == UMA_SIF


class TestWouldHelp:
    """Only worth suggesting when it turns an unusable chain into a usable one."""

    def test_not_suggested_for_something_that_already_runs(self):
        assert would_help("mmff94") is False

    def test_not_suggested_for_a_chain_it_could_not_fix(self):
        assert would_help("mace-off,esen") is False

    def test_not_suggested_for_an_unknown_backend(self):
        assert would_help("not-a-backend") is False

    def test_says_no_without_apptainer(self, monkeypatch):
        monkeypatch.setattr("ligand3d.container.apptainer_available", lambda: False)
        assert would_help("esen") is False


class TestRefusals:
    def test_without_apptainer_it_says_so_and_points_at_slurm(self, monkeypatch, tmp_path):
        monkeypatch.setattr("ligand3d.container.apptainer_available", lambda: False)
        from ligand3d.container import run

        with pytest.raises(ContainerError, match="apptainer was not found"):
            run(from_smiles("CCO"), Settings(), tmp_path / "o.cif")

    def test_a_missing_image_names_the_override(self, monkeypatch, tmp_path):
        monkeypatch.setattr("ligand3d.container.apptainer_available", lambda: True)
        from ligand3d.container import run

        with pytest.raises(ContainerError, match="LIGAND3D_SIF"):
            run(
                from_smiles("CCO"), Settings(), tmp_path / "o.cif",
                image=tmp_path / "nope.sif",
            )


class TestNoiseFilter:
    """torch narrates a great deal on import, none of it about the molecule."""

    @pytest.mark.parametrize(
        "line",
        [
            "cuequivariance or cuequivariance_torch is not available.",
            "  warnings.warn(",
            "/x/e3nn/o3/_wigner.py:10: UserWarning: TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
            "",
        ],
    )
    def test_noise_is_hidden(self, line):
        from ligand3d.container import _worth_showing

        assert _worth_showing(line) is False

    @pytest.mark.parametrize(
        "line",
        [
            "  · minimization time: esen 5.62s",
            "  · lowest energy -253084.1597 kcal/mol (esen), 1 conformer(s)",
            "  · neural potential ran on CPU",
        ],
    )
    def test_the_actual_result_is_shown(self, line):
        from ligand3d.container import _worth_showing

        assert _worth_showing(line) is True


@needs_apptainer
@pytest.mark.slow
class TestForReal:
    """Actually runs a build in a container. Marked slow; needs the image."""

    def test_a_fairchem_model_that_cannot_run_here_runs_there(self, tmp_path):
        from ligand3d.container import run
        from ligand3d.minimize import get_backend

        if not UMA_SIF.exists():
            pytest.skip("the uma image is not on this machine")
        assert not get_backend("esen").available(), (
            "esen is importable here, so this test is not proving anything"
        )

        result = run(
            from_smiles("O=C1CN2CCC1CC2"),
            Settings(backend="esen", sample=1, trace=False, formats=("cif",)),
            tmp_path / "quin.cif",
        )
        assert (tmp_path / "quin.cif").exists()
        assert result["formula"] == "C7H11NO"


class TestTheMenuKnowsWhatAContainerCouldRun:
    """The method menu greys out what cannot run. Without a per-backend flag the
    models the container exists for would stay greyed out, so the checkbox would
    be there and the thing it enables would not be selectable."""

    def test_checkpointed_models_are_flagged(self):
        from ligand3d.container import runnable_in_container

        ids = ["esen", "uma-s", "mace-off", "mace-polar", "aimnet2"]
        flagged = runnable_in_container(ids)
        if not available():
            assert flagged == set()
            return
        from ligand3d.container import image_status

        present = image_status()
        if present["fairchem"]:
            assert {"esen", "uma-s"} <= flagged
        if present["mace"]:
            assert {"mace-off", "mace-polar"} <= flagged

    def test_methods_needing_no_image_are_not_flagged(self):
        from ligand3d.container import runnable_in_container

        assert runnable_in_container(["mmff94", "uff", "gfn2", "gfnff"]) == set()

    def test_nothing_is_flagged_without_apptainer(self, monkeypatch):
        from ligand3d.container import runnable_in_container

        monkeypatch.setattr("ligand3d.container.apptainer_available", lambda: False)
        assert runnable_in_container(["esen", "mace-off"]) == set()

    def test_family_routing(self):
        from ligand3d.container import family_of

        assert family_of("esen") == "fairchem"
        assert family_of("uma-m") == "fairchem"
        assert family_of("mace-off") == "mace"
        assert family_of("mace-polar") == "mace"
        assert family_of("mmff94") is None

    def test_the_catalog_carries_the_flag(self):
        from ligand3d.sketch.session import backend_catalog

        catalog = {b["id"]: b for b in backend_catalog()}
        assert "in_container" in catalog["esen"]
        assert catalog["mmff94"]["in_container"] is False

    def test_image_status_is_a_single_answer_for_both_families(self):
        from ligand3d.container import image_status

        assert set(image_status()) == {"mace", "fairchem"}

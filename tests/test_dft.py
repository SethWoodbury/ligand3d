"""The ORCA backend, and what adding it proved about the architecture.

DFT is the first backend that is not a force field or a fitted potential, and
adding it needed one new module plus three one-line edits — a `Literal`, a
module list, and a sort order. Everything else it gets from declaring its
capabilities: charge and spin are passed to it, the zwitterion check applies,
implicit solvent is routed, and it slots into a chain as the expensive last
link.

The tests that need ORCA itself are marked slow, because a DFT optimisation is
tens of seconds even for methanol.
"""

from __future__ import annotations

import pytest

from ligand3d.minimize import get_backend, list_backends
from ligand3d.minimize.dft import (
    COMPOSITE_METHODS,
    CPCM_SOLVENTS,
    DEFAULT_METHOD,
    OrcaOptions,
    _cpcm_name,
    _is_quantum_orca_available,
)


class TestItIsRegisteredLikeAnyOtherBackend:
    def test_it_appears_in_the_registry(self):
        assert "orca" in list_backends()

    def test_it_declares_what_it_can_do(self):
        caps = get_backend("orca").caps
        assert caps.kind == "dft"
        assert caps.takes_charge, "DFT takes a charge; the pipeline must pass it"
        assert caps.spin_aware, "DFT takes a multiplicity"
        assert caps.supports_solvation, "CPCM is available"
        assert not caps.fixed_topology, "a proton can move, so solvation matters"

    def test_its_energy_is_a_total_not_a_strain(self):
        """Reporting a DFT total as a strain energy would invite comparing it
        to an MMFF94 number, which means nothing."""
        assert get_backend("orca").caps.energy_kind == "total"

    def test_the_catalog_covers_it(self):
        from ligand3d.catalog import build_catalog

        entry = next(m for m in build_catalog() if m.id == "orca")
        assert entry.kind == "dft"
        assert entry.charge == "explicit"


class TestTheKeywordLine:
    def test_a_composite_method_brings_its_own_basis(self):
        line = OrcaOptions(method="B97-3c").simple_input(None)
        assert line.startswith("B97-3c")
        assert "def2" not in line, "a composite method must not be given a basis"

    def test_a_functional_can_be_paired_with_a_basis(self):
        line = OrcaOptions(method="PBE0", basis="def2-SVP").simple_input(None)
        assert "PBE0 def2-SVP" in line

    def test_it_asks_for_a_gradient_not_an_optimisation(self):
        """ligand3d drives L-BFGS itself so that every backend produces the same
        trace and the same convergence criterion. Letting ORCA optimise
        internally would make this one method behave differently from the rest."""
        assert "ENGRAD" in OrcaOptions().simple_input(None)
        assert "OPT" not in OrcaOptions().simple_input(None).upper().split()

    def test_solvent_becomes_cpcm(self):
        assert "CPCM(water)" in OrcaOptions().simple_input("water")

    def test_solvent_aliases_are_translated(self):
        assert _cpcm_name("ch2cl2") == "dichloromethane"
        assert _cpcm_name("h2o") == "water"

    def test_threads_only_appear_when_asked_for(self):
        assert OrcaOptions(threads=1).blocks() == ""
        assert "nprocs 8" in OrcaOptions(threads=8).blocks()

    def test_the_default_is_a_composite_method(self):
        """A DFT optimisation is minutes; the default should not be a
        large-basis hybrid nobody asked for."""
        assert DEFAULT_METHOD in COMPOSITE_METHODS


class TestTheScreenReaderProblem:
    """`orca` on PATH here is the GNOME screen reader, not the quantum
    chemistry program. Resolution has to tell them apart or a DFT run would
    invoke an accessibility tool and fail in a way that mentions neither."""

    def test_a_small_script_is_not_the_quantum_program(self, tmp_path):
        from ligand3d.config import _is_quantum_orca

        fake = tmp_path / "orca"
        fake.write_text("#!/usr/bin/python3\n# screen reader\n")
        fake.chmod(0o755)
        assert _is_quantum_orca(fake) is False

    def test_a_real_install_is_recognised(self, tmp_path):
        from ligand3d.config import _is_quantum_orca

        real = tmp_path / "orca"
        real.write_bytes(b"\0" * 2_000_000)
        real.chmod(0o755)
        (tmp_path / "orca_scf").write_text("")
        assert _is_quantum_orca(real) is True

    def test_resolution_skips_the_one_on_path(self):
        """On this cluster the screen reader is on PATH and the real program is
        under /net/software, so a correct answer proves the check works."""
        from ligand3d.config import resolve_orca

        resolution = resolve_orca()
        if not resolution.found:
            pytest.skip("no ORCA on this machine")
        assert "/net/software" in str(resolution.path) or resolution.via != "PATH"


class TestSolvents:
    def test_cpcm_is_kept_separate_from_alpb(self):
        """The xtb tier uses ALPB with a different list. Pretending one mapping
        covers both would silently accept a solvent ORCA does not know."""
        from ligand3d.solvents import SOLVENTS

        assert CPCM_SOLVENTS != set(SOLVENTS)

    def test_an_unknown_solvent_is_refused_with_the_alternatives(self):
        from ligand3d.errors import BackendUnavailable
        from ligand3d.minimize.base import MinimizeJob
        from ligand3d.molecule import from_smiles
        from ligand3d.embed import embed

        if not _is_quantum_orca_available():
            pytest.skip("no ORCA on this machine")
        backend = get_backend("orca")
        job = MinimizeJob(mol=embed(from_smiles("CCO")), solvent="furane")
        with pytest.raises(BackendUnavailable, match="CPCM"):
            backend.make_calculator(job)


@pytest.mark.slow
class TestARealCalculation:
    """Needs ORCA. Tens of seconds even for a tiny molecule."""

    def _skip_without_orca(self):
        if not _is_quantum_orca_available():
            pytest.skip("no ORCA on this machine")

    def test_methanol_gives_a_sensible_total_energy(self):
        from ligand3d.molecule import from_smiles
        from ligand3d.pipeline import Settings, build

        self._skip_without_orca()
        outcome = build(from_smiles("CO"), Settings(backend="orca", sample=1, trace=False))
        hartree = outcome.best_energy / 627.5095
        # Methanol at a 3c composite level sits near -115.7 Hartree. A wide
        # window, because the point is to catch a wrong unit or a failed parse,
        # not to pin the functional.
        assert -116.5 < hartree < -115.0, f"{hartree} Hartree looks wrong"

    def test_water_geometry_beats_the_force_field(self):
        """Experiment: 0.958 A, 104.5 degrees."""
        from rdkit.Chem import rdMolTransforms as transforms

        from ligand3d.molecule import from_smiles
        from ligand3d.pipeline import Settings, build

        self._skip_without_orca()
        outcome = build(from_smiles("O"), Settings(backend="orca", sample=1, trace=False))
        mol = outcome.mol_3d
        conformer = mol.GetConformer()
        oxygen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "O")
        hydrogens = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "H"]

        length = transforms.GetBondLength(conformer, oxygen, hydrogens[0])
        angle = transforms.GetAngleDeg(conformer, hydrogens[0], oxygen, hydrogens[1])
        assert abs(length - 0.958) < 0.03, length
        assert abs(angle - 104.5) < 4.0, angle

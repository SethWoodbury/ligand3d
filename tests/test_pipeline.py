"""End-to-end: the two target molecules, and the guarantees along the way."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from rdkit import Chem

from ligand3d.errors import BackendMismatch, StereoError
from ligand3d.minimize import get_backend
from ligand3d.molecule import from_smiles
from ligand3d.pipeline import Settings, build, run

QUINUCLIDINONE = "O=C1CN2CCC1CC2"
GABAPENTIN = "NCC1(CC(=O)O)CCCCC1"


def backend_available(name: str) -> bool:
    try:
        return bool(get_backend(name).available())
    except Exception:
        return False


needs_xtb = pytest.mark.skipif(not backend_available("gfn2"), reason="tblite not installed")
needs_dimorphite = pytest.mark.skipif(
    importlib.util.find_spec("dimorphite_dl") is None, reason="dimorphite-dl not installed"
)


class TestTargetMolecules:
    """The two molecules this tool was built to handle first."""

    @pytest.mark.parametrize(
        "smiles,formula,n_atoms",
        [(QUINUCLIDINONE, "C7H11NO", 20), (GABAPENTIN, "C9H17NO2", 29)],
    )
    def test_builds_from_smiles(self, smiles, formula, n_atoms, tmp_path):
        molecule = from_smiles(smiles)
        assert molecule.formula == formula

        outcomes = run(molecule, Settings(), output=tmp_path / "out.pdb")
        assert len(outcomes) == 1

        outcome = outcomes[0]
        assert outcome.mol_3d.GetNumAtoms() == n_atoms
        assert outcome.pdb_path.exists()
        assert outcome.sdf_path.exists()
        assert outcome.best_energy is not None

    def test_quinuclidinone_is_not_rejected_for_phantom_stereocenters(self):
        """Its bridgeheads look stereogenic to a graph analysis but are not."""
        build(from_smiles(QUINUCLIDINONE), Settings())

    def test_output_is_reloadable(self, tmp_path):
        outcome = run(
            from_smiles(QUINUCLIDINONE), Settings(), output=tmp_path / "q.pdb"
        )[0]
        back = Chem.MolFromPDBFile(str(outcome.pdb_path), removeHs=False, sanitize=False)
        assert back is not None
        assert back.GetNumAtoms() == outcome.mol_3d.GetNumAtoms()


class TestStereoPolicy:
    def test_undefined_stereo_is_rejected_by_default(self):
        with pytest.raises(StereoError):
            build(from_smiles("CC(N)C(=O)O"), Settings())

    def test_any_mode_proceeds(self):
        outcome = build(from_smiles("CC(N)C(=O)O"), Settings(stereo_mode="any"))
        assert outcome.mol_3d.GetNumConformers() == 1

    def test_enumerate_mode_builds_every_isomer(self, tmp_path):
        outcomes = run(
            from_smiles("CC(N)C(=O)O"),
            Settings(stereo_mode="enumerate"),
            output=tmp_path / "ala.pdb",
        )
        assert len(outcomes) == 2
        paths = {o.pdb_path for o in outcomes}
        assert len(paths) == 2, "each isomer needs its own file"
        assert all(p.exists() for p in paths)

    def test_specified_stereo_survives_the_whole_pipeline(self, tmp_path):
        molecule = from_smiles("C[C@@H](O)[C@H](N)C(=O)O")
        outcome = run(molecule, Settings(), output=tmp_path / "thr.pdb")[0]

        from ligand3d.embed import perceive_stereo_3d

        recovered = perceive_stereo_3d(outcome.mol_3d)
        assert dict(recovered.assigned_centers) == dict(molecule.stereo.assigned_centers)


class TestConformers:
    def test_multiple_conformers_are_generated_and_ranked(self, tmp_path):
        outcome = run(
            from_smiles(GABAPENTIN), Settings(n_confs=10), output=tmp_path / "g.pdb"
        )[0]
        assert outcome.mol_3d.GetNumConformers() > 1

        energies = [r.energy for r in outcome.records]
        assert energies == sorted(energies), "conformers must come back best-first"

    def test_conformers_are_distinct(self, tmp_path):
        outcome = run(
            from_smiles("CCCCCCCCO"), Settings(n_confs=5), output=tmp_path / "o.pdb"
        )[0]
        positions = [
            np.array(c.GetPositions()) for c in outcome.mol_3d.GetConformers()
        ]
        for i in range(len(positions) - 1):
            assert not np.allclose(positions[i], positions[i + 1])

    def test_energy_window_filters(self, tmp_path):
        wide = run(
            from_smiles("CCCCCCCCO"), Settings(n_confs=20), output=tmp_path / "w.pdb"
        )[0]
        narrow = run(
            from_smiles("CCCCCCCCO"),
            Settings(n_confs=20, energy_window=1.0),
            output=tmp_path / "n.pdb",
        )[0]
        assert len(narrow.records) <= len(wide.records)

        best = narrow.records[0].energy
        assert all(r.energy - best <= 1.0 + 1e-6 for r in narrow.records)


class TestBackendSelection:
    def test_chained_backends_report_the_last_one(self, tmp_path):
        if not backend_available("gfn2"):
            pytest.skip("tblite not installed")
        outcome = run(
            from_smiles(QUINUCLIDINONE),
            Settings(backend="mmff94,gfn2"),
            output=tmp_path / "q.pdb",
        )[0]
        assert outcome.records[0].backend == "gfn2"

    def test_incompatible_backend_fails_before_doing_work(self):
        with pytest.raises(BackendMismatch):
            build(from_smiles("CC(=O)[O-]"), Settings(backend="mace-off"))

    def test_unavailable_backend_gives_an_actionable_error(self):
        from ligand3d.errors import Ligand3DError

        if backend_available("mace-off"):
            pytest.skip("mace-off is installed here")
        with pytest.raises(Ligand3DError, match="not available"):
            build(from_smiles("CCO"), Settings(backend="mace-off"))


@needs_dimorphite
class TestProtonationInPipeline:
    def test_default_keeps_the_structure_as_drawn(self, tmp_path):
        outcome = run(from_smiles(GABAPENTIN), Settings(), output=tmp_path / "g.pdb")[0]
        assert outcome.molecule.formal_charge == 0
        assert not outcome.molecule.is_zwitterion

    @needs_xtb
    def test_ph_produces_the_zwitterion_and_it_survives(self, tmp_path):
        outcome = run(
            from_smiles(GABAPENTIN),
            Settings(ph=7.4, backend="mmff94,gfn2"),
            output=tmp_path / "g.pdb",
        )[0]
        assert outcome.molecule.is_zwitterion
        assert any("solvation" in note for note in outcome.notes)

        positions = np.array(outcome.mol_3d.GetConformer(0).GetPositions())
        symbols = [a.GetSymbol() for a in outcome.mol_3d.GetAtoms()]
        nitrogen = symbols.index("N")
        n_h = sum(
            1
            for i, s in enumerate(symbols)
            if s == "H" and np.linalg.norm(positions[i] - positions[nitrogen]) < 1.25
        )
        assert n_h == 3, "the ammonium lost a proton during minimization"

    def test_enumerate_states_writes_one_file_each(self, tmp_path):
        outcomes = run(
            from_smiles(GABAPENTIN),
            Settings(ph=7.4, enumerate_states=True),
            output=tmp_path / "g.pdb",
        )
        assert len(outcomes) > 1
        assert len({o.pdb_path for o in outcomes}) == len(outcomes)


class TestDeterminism:
    def test_same_seed_gives_identical_coordinates(self, tmp_path):
        first = run(
            from_smiles(GABAPENTIN), Settings(seed=7), output=tmp_path / "a.pdb"
        )[0]
        second = run(
            from_smiles(GABAPENTIN), Settings(seed=7), output=tmp_path / "b.pdb"
        )[0]
        assert np.allclose(
            first.mol_3d.GetConformer(0).GetPositions(),
            second.mol_3d.GetConformer(0).GetPositions(),
        )

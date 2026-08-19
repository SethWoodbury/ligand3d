"""Protonation state assignment, ordering stability, and integrity checking."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from rdkit import Chem

from ligand3d.embed import embed
from ligand3d.errors import ProtonationError
from ligand3d.molecule import from_smiles
from ligand3d.protonate import (
    _rank_key,
    ProtonationState,
    assert_protonation_intact,
    enumerate_states,
    hydrogen_partners,
    protonate,
    suggest_solvent,
)

needs_dimorphite = pytest.mark.skipif(
    importlib.util.find_spec("dimorphite_dl") is None,
    reason="dimorphite-dl not installed",
)

GABAPENTIN = "NCC1(CC(=O)O)CCCCC1"


class TestSolventSuggestion:
    def test_neutral_molecule_needs_no_solvent(self):
        assert suggest_solvent(from_smiles(GABAPENTIN)) is None

    def test_charged_molecule_gets_water(self):
        assert suggest_solvent(from_smiles("CC(=O)[O-]")) == "water"

    def test_zwitterion_gets_water_despite_zero_net_charge(self):
        assert suggest_solvent(from_smiles("[NH3+]CC1(CC(=O)[O-])CCCCC1")) == "water"


class TestStateRanking:
    """Ordering must not depend on dictionary iteration order.

    dimorphite-dl returns states in an order that varies with Python's per-process
    string hash seed. Taking its first element meant the same command could build a
    different molecule on a different run, silently.
    """

    def test_neutral_outranks_charged(self):
        zwitterion = ProtonationState("[NH3+]CC(=O)[O-]", charge=0, n_charged_atoms=2)
        anion = ProtonationState("NCC(=O)[O-]", charge=-1, n_charged_atoms=1)
        assert _rank_key(zwitterion) < _rank_key(anion)

    def test_ionized_outranks_unionized_at_equal_net_charge(self):
        zwitterion = ProtonationState("[NH3+]CC(=O)[O-]", charge=0, n_charged_atoms=2)
        neutral = ProtonationState("NCC(=O)O", charge=0, n_charged_atoms=0)
        assert _rank_key(zwitterion) < _rank_key(neutral)

    def test_ordering_is_total_and_stable(self):
        states = [
            ProtonationState("b", charge=1, n_charged_atoms=1),
            ProtonationState("a", charge=0, n_charged_atoms=0),
            ProtonationState("c", charge=0, n_charged_atoms=2),
        ]
        assert [s.smiles for s in sorted(states, key=_rank_key)] == ["c", "a", "b"]


@needs_dimorphite
class TestProtonation:
    def test_gabapentin_at_ph_7_4_is_the_zwitterion(self):
        """Gabapentin is predominantly zwitterionic at physiological pH."""
        molecule = protonate(from_smiles(GABAPENTIN), ph=7.4)[0]
        assert molecule.is_zwitterion
        assert molecule.formal_charge == 0

    def test_gabapentin_is_cationic_at_low_ph(self):
        molecule = protonate(from_smiles(GABAPENTIN), ph=2.0)[0]
        assert molecule.formal_charge == 1

    def test_gabapentin_is_anionic_at_high_ph(self):
        molecule = protonate(from_smiles(GABAPENTIN), ph=11.0)[0]
        assert molecule.formal_charge == -1

    def test_selection_is_reproducible(self):
        results = {protonate(from_smiles(GABAPENTIN), ph=7.4)[0].smiles for _ in range(8)}
        assert len(results) == 1

    def test_multiple_states_are_reported_to_the_user(self):
        molecule = protonate(from_smiles(GABAPENTIN), ph=7.4)[0]
        assert any("2 states" in note for note in molecule.notes)

    def test_enumerate_returns_every_state(self):
        molecules = protonate(from_smiles(GABAPENTIN), ph=7.4, enumerate_all=True)
        assert len(molecules) > 1
        assert len({m.smiles for m in molecules}) == len(molecules)

    def test_states_are_sorted_deterministically(self):
        first = [s.smiles for s in enumerate_states(from_smiles(GABAPENTIN), ph=7.4)]
        second = [s.smiles for s in enumerate_states(from_smiles(GABAPENTIN), ph=7.4)]
        assert first == second

    def test_quinuclidinone_protonates_at_the_bridgehead_nitrogen(self):
        molecule = protonate(from_smiles("O=C1CN2CCC1CC2"), ph=2.0)[0]
        assert molecule.formal_charge == 1


class TestIntegrityCheck:
    """Detecting that a proton moved during minimization."""

    def test_intact_molecule_passes(self):
        mol_3d = embed(from_smiles("CC(=O)O"))
        assert_protonation_intact(mol_3d, conf_id=0)

    def test_moved_proton_is_caught(self):
        """Physically relocate a hydrogen and confirm the guard fires."""
        molecule = from_smiles("[NH3+]CC1(CC(=O)[O-])CCCCC1")
        mol_3d = embed(molecule)
        conformer = mol_3d.GetConformer(0)
        positions = np.array(conformer.GetPositions())
        symbols = [a.GetSymbol() for a in mol_3d.GetAtoms()]

        nitrogen = symbols.index("N")
        oxygens = [i for i, s in enumerate(symbols) if s == "O"]
        ammonium_h = next(
            i
            for i, s in enumerate(symbols)
            if s == "H" and np.linalg.norm(positions[i] - positions[nitrogen]) < 1.25
        )
        # Park it 1.0 A from a carboxylate oxygen: a covalent O-H bond.
        target = positions[oxygens[0]] + np.array([1.0, 0.0, 0.0])
        conformer.SetAtomPosition(ammonium_h, target.tolist())

        with pytest.raises(ProtonationError, match="protonation state changed"):
            assert_protonation_intact(mol_3d, conf_id=0)

    def test_hydrogen_partners_matches_the_bond_table_for_a_clean_structure(self):
        mol_3d = embed(from_smiles("CCO"))
        geometric = hydrogen_partners(mol_3d, conf_id=0)
        topological = {
            a.GetIdx(): a.GetNeighbors()[0].GetIdx()
            for a in mol_3d.GetAtoms()
            if a.GetAtomicNum() == 1
        }
        assert geometric == topological

    def test_molecule_without_hydrogens_is_trivially_intact(self):
        mol = Chem.MolFromSmiles("[Cl-]")
        from rdkit.Chem import AllChem

        AllChem.Compute2DCoords(mol)
        assert_protonation_intact(mol, conf_id=0)

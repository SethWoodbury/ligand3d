"""Reading and auditing input molecules."""

from __future__ import annotations

import pytest

from ligand3d.errors import InputError, StereoError
from ligand3d.molecule import (
    from_smiles,
    has_real_stereo_ambiguity,
    read_input,
    require_defined_stereo,
)


def test_reads_smiles():
    mol = from_smiles("O=C1CN2CCC1CC2")
    assert mol.formula == "C7H11NO"
    assert mol.formal_charge == 0
    assert not mol.is_zwitterion


def test_rejects_garbage_smiles():
    with pytest.raises(InputError):
        from_smiles("this is not a molecule")


def test_rejects_empty_smiles():
    with pytest.raises(InputError):
        from_smiles("   ")


def test_detects_assigned_stereocenters():
    mol = from_smiles("C[C@@H](O)[C@H](N)C(=O)O")
    codes = dict(mol.stereo.assigned_centers)
    assert len(codes) == 2
    assert set(codes.values()) <= {"R", "S"}


def test_detects_double_bond_geometry():
    mol = from_smiles(r"CC/C(=C(\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1")
    assert len(mol.stereo.assigned_bonds) == 1
    assert mol.stereo.assigned_bonds[0][2] in ("E", "Z")


def test_zwitterion_detected_despite_zero_net_charge():
    mol = from_smiles("[NH3+]CC1(CC(=O)[O-])CCCCC1")
    assert mol.formal_charge == 0
    assert mol.is_zwitterion


class TestStereoAmbiguity:
    """Telling a real undefined stereocenter from a constrained one.

    RDKit flags bridgehead atoms of caged systems as potential stereocenters
    because a graph-only analysis cannot see the ring constraint. Rejecting
    3-quinuclidinone for "undefined stereochemistry" would be wrong; failing to
    reject unspecified alanine would be worse.
    """

    @pytest.mark.parametrize(
        "smiles",
        [
            "O=C1CN2CCC1CC2",  # 3-quinuclidinone: bridgeheads are constrained
            "C1CN2CCC1CC2",  # quinuclidine
            "C1C2CC3CC1CC(C2)C3",  # adamantane
            "C1CC2CCC1C2",  # norbornane
            "NCC1(CC(=O)O)CCCCC1",  # gabapentin: genuinely achiral
        ],
    )
    def test_constrained_centers_are_not_ambiguous(self, smiles):
        mol = from_smiles(smiles)
        assert not has_real_stereo_ambiguity(mol)
        require_defined_stereo(mol)  # must not raise

    @pytest.mark.parametrize(
        "smiles",
        [
            "CC(N)C(=O)O",  # alanine, alpha carbon unspecified
            "C[C@@H](O)C(N)C(=O)O",  # threonine, one of two unspecified
            "C1CCC2CCCCC2C1",  # decalin: cis and trans really are different
            "OCC(O)C(O)C(O)C(O)C=O",  # open-chain glucose
        ],
    )
    def test_real_ambiguity_is_caught(self, smiles):
        mol = from_smiles(smiles)
        assert has_real_stereo_ambiguity(mol)
        with pytest.raises(StereoError):
            require_defined_stereo(mol)

    def test_fully_specified_molecule_passes(self):
        morphine = from_smiles("CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@H](O)C=C[C@H]3[C@H]1C5")
        assert len(morphine.stereo.assigned_centers) == 5
        require_defined_stereo(morphine)


def test_read_input_prefers_existing_file(tmp_path):
    path = tmp_path / "mol.smi"
    path.write_text("CCO ethanol\n")
    mol = read_input(str(path))
    assert mol.smiles == "CCO"


def test_read_input_falls_back_to_smiles():
    assert read_input("CCO").smiles == "CCO"

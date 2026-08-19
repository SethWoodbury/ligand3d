"""The test that matters most: stereochemistry must survive 3D generation.

A mirrored stereocenter produces a structure that looks entirely reasonable and
is the wrong molecule. These tests exist so that failure is loud.
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from ligand3d.embed import EmbedOptions, embed, embed_multi, perceive_stereo_3d, verify_stereo
from ligand3d.errors import StereoError
from ligand3d.molecule import from_smiles

CHIRAL_MOLECULES = [
    pytest.param("C[C@H](N)C(=O)O", id="L-alanine"),
    pytest.param("C[C@@H](O)[C@H](N)C(=O)O", id="threonine-2-centers"),
    pytest.param("C[C@H](C(=O)O)c1ccc2cc(OC)ccc2c1", id="S-naproxen"),
    pytest.param(
        "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@H](O)C=C[C@H]3[C@H]1C5", id="morphine-5-centers"
    ),
    pytest.param("OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O", id="alpha-D-glucose"),
    pytest.param("C[C@@H]1CC[C@H](C(C)C)CC1", id="cis-ring-substituents"),
]


@pytest.mark.parametrize("smiles", CHIRAL_MOLECULES)
def test_stereocenters_survive_embedding(smiles):
    """SMILES -> 3D -> re-perceived CIP labels must match the input exactly."""
    molecule = from_smiles(smiles)
    assert molecule.stereo.assigned_centers, "test molecule should have stereocenters"

    mol_3d = embed(molecule)
    recovered = perceive_stereo_3d(mol_3d)

    assert dict(recovered.assigned_centers) == dict(molecule.stereo.assigned_centers)


def test_double_bond_geometry_survives_embedding():
    """E/Z must round-trip.

    Guards a specific trap: RDKit labels bond stereo as both CIS/TRANS (relative
    to reference atoms) and E/Z (CIP). Comparing the two families directly gives
    a false mismatch, so both sides normalize to E/Z.
    """
    molecule = from_smiles(r"CC/C(=C(\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1")
    mol_3d = embed(molecule)
    recovered = perceive_stereo_3d(mol_3d)

    want = {(a, b): label for a, b, label in molecule.stereo.assigned_bonds}
    got = {(a, b): label for a, b, label in recovered.assigned_bonds}
    assert want and want == {k: got.get(k) for k in want}


def test_enantiomers_embed_to_different_structures():
    """The obvious sanity check: R and S must not produce the same answer."""
    r_form = perceive_stereo_3d(embed(from_smiles("C[C@H](N)C(=O)O")))
    s_form = perceive_stereo_3d(embed(from_smiles("C[C@@H](N)C(=O)O")))
    assert dict(r_form.assigned_centers) != dict(s_form.assigned_centers)


def test_verify_stereo_raises_on_inverted_center():
    """Deliberately corrupt a structure and confirm the guard fires."""
    molecule = from_smiles("C[C@H](N)C(=O)O")
    mol_3d = embed(molecule)

    mirrored = Chem.Mol(mol_3d)
    conformer = mirrored.GetConformer()
    for i in range(mirrored.GetNumAtoms()):
        pos = conformer.GetAtomPosition(i)
        conformer.SetAtomPosition(i, (-pos.x, pos.y, pos.z))

    with pytest.raises(StereoError, match="stereochemistry changed"):
        verify_stereo(mirrored, molecule.stereo)


def test_verify_stereo_is_silent_when_nothing_was_specified():
    molecule = from_smiles("CCO")
    verify_stereo(embed(molecule), molecule.stereo)


@pytest.mark.parametrize(
    "smiles",
    ["O=C1CN2CCC1CC2", "NCC1(CC(=O)O)CCCCC1", "C1C2CC3CC1CC(C2)C3"],
)
def test_strained_cages_embed(smiles):
    """Bridged and caged systems are where naive embedding fails."""
    mol_3d = embed(from_smiles(smiles))
    assert mol_3d.GetNumConformers() == 1
    assert mol_3d.GetConformer().Is3D()


def test_embedding_is_deterministic():
    molecule = from_smiles("C[C@@H](O)[C@H](N)C(=O)O")
    first = embed(molecule, EmbedOptions(seed=1234)).GetConformer().GetPositions()
    second = embed(molecule, EmbedOptions(seed=1234)).GetConformer().GetPositions()
    assert (first == second).all()


def test_different_seeds_give_different_structures():
    molecule = from_smiles("CCCCCCCCO")
    first = embed(molecule, EmbedOptions(seed=1)).GetConformer().GetPositions()
    second = embed(molecule, EmbedOptions(seed=99)).GetConformer().GetPositions()
    assert not (first == second).all()


def test_multi_embed_produces_several_conformers():
    mol_3d = embed_multi(from_smiles("CCCCCCCCO"), n_confs=10)
    assert mol_3d.GetNumConformers() > 1


def test_hydrogens_are_added():
    molecule = from_smiles("O=C1CN2CCC1CC2")
    mol_3d = embed(molecule)
    assert mol_3d.GetNumAtoms() == 20  # C7H11NO
    assert any(a.GetAtomicNum() == 1 for a in mol_3d.GetAtoms())

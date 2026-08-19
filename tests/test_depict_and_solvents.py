"""The 2D preview, the stereo explanation, and the solvent table."""

from __future__ import annotations

import pytest

from ligand3d.depict import depict, depict_molblock
from ligand3d.errors import BackendMismatch, StereoError
from ligand3d.molecule import (
    classify_undefined_stereo,
    from_smiles,
    require_defined_stereo,
)

# A 1,3-disubstituted cyclobutane: the substituents can be cis or trans, which
# is real stereochemistry that looks nothing like a wedge-on-a-carbon.
RING_CIS_TRANS = "C1CC(C2CC(C3CC4(CCC4)C3)C2)C1"


class TestStereoExplanation:
    """Saying *what kind* of stereochemistry is missing, not just where."""

    def test_ring_pairs_are_described_as_cis_trans(self):
        advice = " ".join(classify_undefined_stereo(from_smiles(RING_CIS_TRANS)))
        assert "same 4-membered ring" in advice
        assert "cis" in advice and "trans" in advice
        # "stereocenter" alone sends people hunting for a tetrahedral carbon.
        assert "same face" in advice

    def test_a_plain_stereocenter_is_described_as_one(self):
        advice = " ".join(classify_undefined_stereo(from_smiles("CC(N)C(=O)O")))
        assert "stereocenter with no configuration" in advice
        assert "acyclic" in advice

    def test_constrained_centers_produce_no_advice(self):
        """Quinuclidinone's bridgeheads cannot vary, so there is nothing to say."""
        assert classify_undefined_stereo(from_smiles("O=C1CN2CCC1CC2")) == []

    def test_a_fully_specified_molecule_produces_no_advice(self):
        assert classify_undefined_stereo(from_smiles("C[C@H](N)C(=O)O")) == []

    def test_the_error_points_at_the_preview(self):
        with pytest.raises(StereoError) as exc:
            require_defined_stereo(from_smiles(RING_CIS_TRANS))
        message = str(exc.value)
        assert "4-membered ring" in message
        assert "preview" in message
        assert "--stereo enumerate" in message


class TestDepiction:
    def test_renders_svg_with_atom_indices(self):
        result = depict(from_smiles(RING_CIS_TRANS))
        assert result.svg.startswith("<?xml") or "<svg" in result.svg
        assert result.n_atoms == 15
        # The whole point: every atom carries the number the messages use.
        assert result.svg.count("class='note'") + result.svg.count('class="note"') > 0

    def test_indices_can_be_turned_off(self):
        with_indices = depict(from_smiles("CCO"), show_indices=True).svg
        without = depict(from_smiles("CCO"), show_indices=False).svg
        assert len(without) < len(with_indices)

    def test_highlights_undefined_centers(self):
        result = depict(from_smiles(RING_CIS_TRANS))
        assert result.undefined_centers == (3, 5)
        assert result.defined_centers == ()
        assert "<ellipse" in result.svg
        assert any("amber" in note for note in result.notes)

    def test_marks_defined_centers_separately(self):
        result = depict(from_smiles("C[C@@H](O)[C@H](N)C(=O)O"))
        assert dict(result.defined_centers) == {1: "R", 3: "S"}
        assert result.undefined_centers == ()

    def test_marks_ring_constrained_centers_as_such(self):
        result = depict(from_smiles("O=C1CN2CCC1CC2"))
        assert result.constrained_centers == (3, 6)
        assert result.undefined_centers == ()
        assert any("fixed by the ring" in note for note in result.notes)

    def test_dark_palette_differs_from_light(self):
        molecule = from_smiles("CCO")
        assert depict(molecule, dark=True).svg != depict(molecule, dark=False).svg

    def test_uses_the_supplied_layout(self):
        """The picture should match what the user drew, not a fresh layout."""
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles("c1ccccc1C(=O)O")
        AllChem.Compute2DCoords(mol)
        molblock = Chem.MolToMolBlock(mol)
        result = depict_molblock(molblock)
        assert result.n_atoms == mol.GetNumAtoms()

    def test_json_is_serializable(self):
        import json

        json.dumps(depict(from_smiles("CCO")).to_json())


class TestSolvents:
    def test_the_documented_names_resolve(self):
        from ligand3d.solvents import SOLVENTS, resolve, validate

        assert len(SOLVENTS) >= 20
        for entry in SOLVENTS:
            assert validate(entry.name) == entry.name
            for alias in entry.aliases:
                assert resolve(alias) is entry

    @pytest.mark.parametrize(
        "alias,canonical",
        [("h2o", "water"), ("ch2cl2", "dichloromethane"),
         ("chcl3", "chloroform"), ("ether", "diethylether"), ("cs2", "carbondisulfide")],
    )
    def test_aliases_map_to_canonical_names(self, alias, canonical):
        from ligand3d.solvents import validate

        assert validate(alias) == canonical

    def test_case_and_whitespace_are_forgiven(self):
        from ligand3d.solvents import validate

        assert validate("  DMSO  ") == "dmso"

    def test_an_unparameterized_solvent_suggests_a_stand_in(self):
        """Cyclohexane is the obvious thing to try and ALPB does not have it."""
        from ligand3d.solvents import validate

        with pytest.raises(BackendMismatch, match="hexane"):
            validate("cyclohexane")

    def test_an_unknown_solvent_says_where_to_look(self):
        from ligand3d.solvents import validate

        with pytest.raises(BackendMismatch, match="ligand3d solvents"):
            validate("unobtainium")

    def test_the_pipeline_rejects_a_bad_solvent_before_working(self):
        from ligand3d.pipeline import Settings, build

        with pytest.raises(BackendMismatch):
            build(from_smiles("CCO"), Settings(backend="gfn2", solvent="cyclohexane"))

    @pytest.mark.parametrize("solvent", ["water", "dmso", "chloroform", "toluene"])
    def test_tblite_accepts_every_name_we_advertise(self, solvent):
        """The table is only useful if the calculator agrees with it."""
        from ligand3d.embed import embed
        from ligand3d.minimize import MinimizeJob, get_backend

        backend = get_backend("gfn2")
        if not backend.available():
            pytest.skip("tblite not installed")
        mol = embed(from_smiles("CCO"))
        result = backend.minimize(
            MinimizeJob(mol=mol, conf_id=0, solvent=solvent, max_steps=5)
        )
        assert result.energy < 0

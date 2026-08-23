"""The annotated mmCIF that RFdiffusion4-Proteina reads.

An annotated CIF is an ordinary mmCIF with extra `_atom_site` columns saying
what the model should hold fixed. Three of the format's failure modes are
silent — a misspelled tag is dropped with no diagnostic, a mask written as 1/0
parses as an integer and fails much later, and a bond left generically aromatic
is cast to single — so the tests here mostly pin things that would otherwise go
wrong quietly.

`TestAgainstTheRealValidator` runs RFD4's own `check_annotated_cif.py`, which
executes the actual inference pipeline. It is skipped where that repo is not
checked out, and it is the only test here that proves the file is acceptable
rather than merely well-formed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from rdkit import Chem

from ligand3d.annotated import (
    ANNOTATION_TAGS,
    COORDINATE_MASK,
    SEQUENCE_MASK,
    SEQUENCE_VALUE,
    DesignSegment,
    parse_segment,
    to_annotated_cif_string,
    write_annotated_cif,
)
from ligand3d.embed import embed
from ligand3d.errors import Ligand3DError
from ligand3d.molecule import from_smiles

RFD4 = Path("/home/woodbuse/git/RFD4-Proteina-dev")
needs_validator = pytest.mark.skipif(
    not (RFD4 / "scripts/json2cif/check_annotated_cif.py").exists()
    or not (RFD4 / ".venv/bin/python").exists(),
    reason="RFD4-Proteina-dev is not checked out here",
)


def annotated(smiles: str, **kwargs) -> str:
    return to_annotated_cif_string(
        embed(from_smiles(smiles)), resname="LIG", smiles=smiles, **kwargs
    )


def atom_rows(text: str) -> list[list[str]]:
    return [
        line.split() for line in text.splitlines()
        if line.startswith(("ATOM ", "HETATM "))
    ]


def column(text: str, tag: str) -> list[str]:
    tags = [t.strip() for t in text.splitlines() if t.strip().startswith("_atom_site.")]
    index = [t.split(".", 1)[1] for t in tags].index(tag)
    return [row[index] for row in atom_rows(text)]


class TestTheColumnsAreSpelledRight:
    """A misspelled tag is silently dropped — no warning, no error, no
    annotation. The names come from the format's registry and are worth pinning
    exactly, because nothing downstream will complain if they drift."""

    @pytest.mark.parametrize("tag", ANNOTATION_TAGS)
    def test_every_tag_appears_as_a_column(self, tag):
        text = annotated("CCO", design=parse_segment("100-155"))
        assert f"_atom_site.{tag}" in text

    def test_the_exact_registry_names(self):
        assert COORDINATE_MASK == "mask_coordinate_1_atom"
        assert SEQUENCE_MASK == "mask_sequence_1_residue"
        assert SEQUENCE_VALUE == "condition_sequence_1_residue"


class TestMasksAreBooleanStrings:
    """Dtype inference tries int before bool, so a mask written as 1/0 becomes
    an int64 array and fails later with "Mask must be a boolean array"."""

    def test_true_and_false_are_written_as_words(self):
        text = annotated("CCO")
        for tag in (COORDINATE_MASK, SEQUENCE_MASK):
            assert set(column(text, tag)) <= {"True", "False"}

    def test_never_one_or_zero(self):
        text = annotated("CCO", design=parse_segment("100-155"))
        for tag in ANNOTATION_TAGS:
            if "mask_" in tag:
                assert not (set(column(text, tag)) & {"1", "0"}), tag


class TestWhatTheFileSays:
    def test_the_ligand_is_coordinate_conditioned_by_default(self):
        """The pose is what ligand3d exists to produce; pinning it is the point."""
        assert set(column(annotated("CCO"), COORDINATE_MASK)) == {"True"}

    def test_the_pose_can_be_left_free(self):
        text = annotated("CCO", fix_coordinates=False)
        assert set(column(text, COORDINATE_MASK)) == {"False"}

    def test_the_ligand_is_always_sequence_conditioned(self):
        """Non-polymers are atomized, and an atomized token that is not
        sequence-conditioned is rejected. It is not an option."""
        for kwargs in ({}, {"fix_coordinates": False}):
            assert set(column(annotated("CCO", **kwargs), SEQUENCE_MASK)) == {"True"}

    def test_coordinate_conditioning_never_appears_without_sequence(self):
        """Every ligand atom is non-backbone, and coordinate conditioning on a
        non-backbone atom requires sequence conditioning. The combination the
        validator rejects is not expressible here."""
        text = annotated("c1ccccc1C(=O)[O-]")
        coords = column(text, COORDINATE_MASK)
        seqs = column(text, SEQUENCE_MASK)
        assert all(s == "True" for c, s in zip(coords, seqs) if c == "True")

    def test_the_residue_name_is_carried(self):
        assert set(column(annotated("CCO"), SEQUENCE_VALUE)) == {"LIG"}


class TestTheDesignSegment:
    def test_a_bare_number_is_a_fixed_length(self):
        assert parse_segment("120") == DesignSegment(120, 120)

    def test_a_range_is_resampled(self):
        assert parse_segment("100-155") == DesignSegment(100, 155)

    @pytest.mark.parametrize("bad", ["", "abc", "0", "50-10", "-5"])
    def test_nonsense_is_refused(self, bad):
        with pytest.raises(Ligand3DError):
            parse_segment(bad)

    def test_the_sentinel_is_one_extra_atom(self):
        without = atom_rows(annotated("CCO"))
        with_seg = atom_rows(annotated("CCO", design=parse_segment("100-155")))
        assert len(with_seg) == len(without) + 1

    def test_the_sentinel_carries_the_bounds_collapsed(self):
        """Collapsed, not realized: a segment written out as residues is frozen
        to one length and will not be resampled per replicate."""
        text = annotated("CCO", design=parse_segment("100-155"))
        row = atom_rows(text)[-1]
        assert "100-155" in row          # residue name is the range itself
        assert row[2] == "X" and "UNK" in row
        assert column(text, "annotation_expsegmin_1_residue")[-1] == "100"
        assert column(text, "annotation_expsegmax_1_residue")[-1] == "155"

    def test_the_sentinel_has_no_coordinates(self):
        """A real coordinate here would place the designed chain."""
        text = annotated("CCO", design=parse_segment("100-155"))
        assert atom_rows(text)[-1].count("nan") >= 3

    def test_ligand_atoms_carry_no_segment(self):
        text = annotated("CCO", design=parse_segment("100-155"))
        assert set(column(text, "annotation_expsegmin_1_residue")[:-1]) == {"-1"}

    def test_the_segment_is_declared_a_polypeptide(self):
        """Without this it reads as a non-polymer and is rejected with
        "Only polypeptide expandable segments are currently supported"."""
        text = annotated("CCO", design=parse_segment("100-155"))
        assert "polypeptide(L)" in text
        assert "_entity_poly" in text


class TestFormalCharges:
    """`?` means unknown, and charges reach the model. A neutral atom is not an
    atom of unknown charge."""

    def test_neutral_atoms_say_zero_not_question_mark(self):
        assert set(column(annotated("CCO"), "pdbx_formal_charge")) == {"0"}

    def test_a_real_charge_is_carried(self):
        charges = column(annotated("c1ccccc1C(=O)[O-]"), "pdbx_formal_charge")
        assert "-1" in charges
        assert charges.count("-1") == 1


class TestOnePosePerFile:
    def test_several_conformers_are_refused(self):
        """The annotation says "use exactly this geometry"; several models in
        one file would silently condition on the first."""
        from ligand3d.conformers import ConformerOptions, generate

        molecule = from_smiles("CCCCO")
        many = generate(molecule, ConformerOptions(n_confs=3, seed=1))
        if many.GetNumConformers() < 2:
            pytest.skip("embedding produced a single conformer")
        with pytest.raises(Ligand3DError, match="one pose"):
            to_annotated_cif_string(many, resname="LIG")


class TestKekulization:
    """A bond left generically aromatic is cast to single, which would tell the
    model an aromatic ring is saturated."""

    def test_benzene_is_not_six_single_bonds(self):
        text = annotated("c1ccccc1")
        ring = [
            line.split() for line in text.splitlines()
            if line.startswith("LIG ") and line.rstrip().endswith(" Y")
        ]
        orders = [row[3] for row in ring]
        assert orders.count("DOUB") == 3, orders
        assert orders.count("SING") == 3, orders

    def test_the_aromatic_flag_survives_kekulization(self):
        text = annotated("c1ccccc1")
        assert sum(1 for line in text.splitlines()
                   if line.startswith("LIG ") and line.rstrip().endswith(" Y")) == 6


class TestWriting:
    def test_writes_the_file(self, tmp_path):
        target = write_annotated_cif(
            tmp_path / "x.annotated.cif", embed(from_smiles("CCO")), resname="LIG"
        )
        assert target.exists() and target.read_text().startswith("data_")

    def test_it_is_still_readable_as_an_mmcif(self, tmp_path):
        """A plain mmCIF is a valid annotated CIF, and the reverse holds too:
        the annotations are extra columns, not a different format."""
        gemmi = pytest.importorskip("gemmi")
        target = write_annotated_cif(
            tmp_path / "x.annotated.cif",
            embed(from_smiles("c1ccccc1C(=O)[O-]")),
            resname="LIG",
            design=parse_segment("100-155"),
        )
        block = gemmi.cif.read(str(target)).sole_block()
        assert block.find_loop("_atom_site.id") is not None


@needs_validator
@pytest.mark.slow
class TestAgainstTheRealValidator:
    """RFD4's own checker, which runs the actual inference pipeline.

    A pass here means the model will accept the file; nothing else in this
    module proves that.
    """

    def _check(self, path: Path) -> str:
        env = {
            **os.environ,
            "CCD_MIRROR_PATH": "/projects/ml/frozen_pdb_copies/2026_01_06_ccd",
            "PDB_MIRROR_PATH": "/projects/ml/frozen_pdb_copies/2026_01_06_pdb",
            "CLUSTER": "digs",
            "ALLOW_BIOTITE_CCD": "True",
            "PYTHONPATH": f"{RFD4}/src:{RFD4}/lib/atomworks/src",
        }
        result = subprocess.run(
            [str(RFD4 / ".venv/bin/python"),
             str(RFD4 / "scripts/json2cif/check_annotated_cif.py"), str(path)],
            capture_output=True, text=True, cwd=RFD4, env=env, timeout=1800,
        )
        return result.stdout + result.stderr

    @pytest.mark.parametrize(
        "smiles, length",
        [
            ("c1ccccc1C(=O)[O-]", "100-155"),       # charged, aromatic
            ("O=C1CN2CCC1CC2", "120"),              # bridged bicyclic
            ("[NH3+]CC1(CC(=O)[O-])CCCCC1", "80"),  # zwitterion
            ("Cc1ccc(cc1)S(=O)(=O)Nc1ccccn1", "150"),  # heteroaromatic, sulfonamide
            ("CC(=O)Oc1ccccc1C(=O)O", None),        # ligand alone, no segment
        ],
    )
    def test_the_validator_reports_no_errors(self, tmp_path, smiles, length):
        target = write_annotated_cif(
            tmp_path / "check.annotated.cif",
            embed(from_smiles(smiles)),
            resname="LIG",
            smiles=smiles,
            design=parse_segment(length) if length else None,
        )
        report = self._check(target)
        errors = re.search(r"(\d+) error\(s\)", report)
        assert errors, f"validator produced no summary:\n{report[-3000:]}"
        assert errors.group(1) == "0", report[-3000:]


@needs_validator
@pytest.mark.slow
class TestAromaticBondsSurviveAsAromatic:
    """Kekulization is not an alternative to AROMATIC_SINGLE/AROMATIC_DOUBLE —
    it is how you express them in mmCIF.

    The pair (`value_order`, `pdbx_aromatic_flag`) is the encoding: a kekulized
    order with the flag set reads back as the aromatic bond type carrying that
    order. This is the actual contract, and it is worth asserting on the parsed
    types rather than on the strings we wrote, because the strings are only
    half of it.

    Measured, for benzene:

        kekulized SING/DOUB + flag Y   ->  AROMATIC_SINGLE x3, AROMATIC_DOUBLE x3
        all SING + flag Y              ->  AROMATIC_SINGLE x6   (orders lost)
        AROM + flag Y                  ->  AROMATIC x6          (no defined order)

    The last is the failure the format spec warns about: `BondType.AROMATIC` is
    deliberately excluded from atomworks' bond-order table, because the order
    is not well defined. So there is no useful "write generic aromatic" option
    — every alternative to kekulizing is strictly worse.
    """

    def _bond_types(self, path: Path) -> dict[str, int]:
        script = (
            "import warnings; warnings.filterwarnings('ignore')\n"
            "import json\n"
            "from collections import Counter\n"
            "from biotite.structure import BondType\n"
            "import biotite.structure.io.pdbx as pdbx\n"
            f"arr = pdbx.get_structure(pdbx.CIFFile.read({str(path)!r}), "
            "model=1, include_bonds=True)\n"
            "print(json.dumps(dict(Counter("
            "BondType(int(b[2])).name for b in arr.bonds.as_array()))))\n"
        )
        env = {**os.environ, "PYTHONPATH": f"{RFD4}/src:{RFD4}/lib/atomworks/src",
               "ALLOW_BIOTITE_CCD": "True"}
        result = subprocess.run(
            [str(RFD4 / ".venv/bin/python"), "-c", script],
            capture_output=True, text=True, cwd=RFD4, env=env, timeout=600,
        )
        import json as _json

        assert result.returncode == 0, result.stderr[-2000:]
        return _json.loads(result.stdout.strip().splitlines()[-1])

    def test_benzene_reads_back_as_alternating_aromatic_bonds(self, tmp_path):
        target = write_annotated_cif(
            tmp_path / "benzene.annotated.cif",
            embed(from_smiles("c1ccccc1")),
            resname="LIG",
        )
        types = self._bond_types(target)
        assert types.get("AROMATIC_SINGLE") == 3, types
        assert types.get("AROMATIC_DOUBLE") == 3, types
        # Never the generic type, whose order atomworks does not define.
        assert "AROMATIC" not in types, types

    def test_a_fused_heteroaromatic_keeps_its_orders(self, tmp_path):
        target = write_annotated_cif(
            tmp_path / "indole.annotated.cif",
            embed(from_smiles("c1ccc2[nH]ccc2c1")),
            resname="LIG",
        )
        types = self._bond_types(target)
        assert types.get("AROMATIC_DOUBLE", 0) >= 3, types
        assert "AROMATIC" not in types, types

    def test_non_aromatic_bonds_stay_plain(self, tmp_path):
        """The flag must not be sprayed over everything: a C-H is not aromatic."""
        target = write_annotated_cif(
            tmp_path / "benzene.annotated.cif",
            embed(from_smiles("c1ccccc1")),
            resname="LIG",
        )
        types = self._bond_types(target)
        assert types.get("SINGLE") == 6, types   # the six C-H bonds

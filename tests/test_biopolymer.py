"""Building peptides and nucleic acids from a sequence.

The load-bearing tests are the ones that check the residue library against
RDKit. A hand-written amino-acid SMILES with an inverted stereocentre produces
a molecule that looks entirely reasonable and is the wrong enantiomer, and a
misdrawn sugar gives a nucleic acid nobody would spot by eye. RDKit builds the
canonical alphabets itself, so it can be asked whether the library agrees.
"""

from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from ligand3d.biopolymer import (
    KINDS,
    SequenceError,
    available_residues,
    build,
    build_nucleic,
    build_peptide,
    parse_sequence,
    residue_name,
)

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"

# RDKit's MolFromSequence flavors: 0 = L-peptide, 2 = RNA, 6 = DNA, all
# without terminal phosphates, which is what this module builds.
FLAVOR = {"peptide": 0, "rna": 2, "dna": 6}


def canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unreadable: {smiles}"
    return Chem.MolToSmiles(mol)


def reference(sequence: str, kind: str) -> str:
    mol = Chem.MolFromSequence(sequence, flavor=FLAVOR[kind])
    assert mol is not None, f"RDKit would not build {sequence!r} as {kind}"
    return Chem.MolToSmiles(mol)


class TestAgainstRDKit:
    """The library must reproduce RDKit exactly for everything RDKit can build."""

    @pytest.mark.parametrize("code", list(CANONICAL_AA))
    def test_every_canonical_amino_acid(self, code):
        assert canonical(build_peptide([code])) == reference(code, "peptide")

    @pytest.mark.parametrize(
        "sequence",
        ["PLYNGSG", CANONICAL_AA, "WWW", "PPP", "GG", "CFC", "HHH", "RRR", "MKV"],
    )
    def test_peptide_chains(self, sequence):
        """Also pins that ring-closure digits may be reused between residues.

        Tryptophan and proline each open rings inside their own fragment, so
        concatenating them is only safe because each closes what it opens.
        """
        assert canonical(build_peptide(list(sequence))) == reference(sequence, "peptide")

    @pytest.mark.parametrize("code", list("ACGT"))
    def test_every_dna_monomer(self, code):
        assert canonical(build_nucleic([code], "dna")) == reference(code, "dna")

    @pytest.mark.parametrize("code", list("ACGU"))
    def test_every_rna_monomer(self, code):
        assert canonical(build_nucleic([code], "rna")) == reference(code, "rna")

    @pytest.mark.parametrize("sequence", ["AA", "AC", "GGCAT", "ATGCATGC", "TTTT"])
    def test_dna_chains(self, sequence):
        assert canonical(build_nucleic(list(sequence), "dna")) == reference(
            sequence, "dna"
        )

    @pytest.mark.parametrize("sequence", ["AA", "AUGGC", "ACGU", "GGGG"])
    def test_rna_chains(self, sequence):
        """RNA is the case that can go wrong quietly.

        Ribose carries a 2'-OH as well as the 3'-OH, so linking to the wrong
        one produces a 2'-5' chain — a real but different molecule.
        """
        assert canonical(build_nucleic(list(sequence), "rna")) == reference(
            sequence, "rna"
        )


class TestParsing:
    def test_plain_letters(self):
        assert parse_sequence("PLY") == ["P", "L", "Y"]

    def test_bracketed_codes_are_one_residue(self):
        assert parse_sequence("GS(KCX)PL") == ["G", "S", "KCX", "P", "L"]

    def test_case_is_normalized(self):
        assert parse_sequence("ggcat") == ["G", "G", "C", "A", "T"]

    @pytest.mark.parametrize("text", ["GG CAT", "GG-CAT", "GG.CAT", "G G C A T"])
    def test_separators_people_actually_paste(self, text):
        assert parse_sequence(text) == ["G", "G", "C", "A", "T"]

    def test_residue_numbering_is_ignored(self):
        assert parse_sequence("1 GGC 4 AT") == ["G", "G", "C", "A", "T"]

    def test_an_empty_bracket_is_refused(self):
        with pytest.raises(SequenceError, match="empty"):
            parse_sequence("GS()PL")

    def test_an_empty_sequence_is_refused(self):
        with pytest.raises(SequenceError, match="empty"):
            parse_sequence("   ")

    def test_junk_is_refused(self):
        with pytest.raises(SequenceError, match="not something a sequence"):
            parse_sequence("GG@AT")


class TestModifiedResidues:
    def test_the_motivating_example(self):
        """GS(KCX)PL: carboxylated lysine in the middle of a chain."""
        result = build("GS(KCX)PL", "peptide")
        mol = Chem.MolFromSmiles(result.smiles)
        # Five free residues condensed with four losses of water.
        assert rdMolDescriptors.CalcMolFormula(mol) == "C23H40N6O9"
        assert result.residues == ["G", "S", "KCX", "P", "L"]

    def test_a_modified_residue_is_reported_not_silent(self):
        notes = " ".join(build("GS(KCX)PL", "peptide").notes)
        assert "KCX = N6-carboxylysine" in notes

    def test_phosphoserine(self):
        mol = Chem.MolFromSmiles(build("A(SEP)G", "peptide").smiles)
        assert rdMolDescriptors.CalcMolFormula(mol) == "C8H16N3O8P"

    def test_inosine_in_dna(self):
        result = build("GGCIT", "dna")
        assert "2'-deoxyinosine" in " ".join(result.notes)

    def test_inosine_in_rna(self):
        assert "inosine" in " ".join(build("AUGGI", "rna").notes)

    def test_pseudouridine_is_an_isomer_of_uridine(self):
        """Ψ is uracil joined through carbon instead of nitrogen.

        Same formula, different molecule — so the formula check confirms the
        composition and the SMILES difference confirms it is not just uridine.
        """
        psu = build("A(PSU)GGC", "rna")
        uridine = build("AUGGC", "rna")
        formula = lambda s: rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(s))
        assert formula(psu.smiles) == formula(uridine.smiles)
        assert psu.smiles != uridine.smiles

    def test_the_extended_one_letter_codes(self):
        """U and O are selenocysteine and pyrrolysine in the expanded alphabet."""
        result = build("UGO", "peptide")
        assert "selenocysteine" in " ".join(result.notes)
        assert "pyrrolysine" in " ".join(result.notes)
        assert "Se" in result.smiles

    def test_every_library_entry_is_a_readable_structure(self):
        """A typo in a stored SMILES should fail here, not in front of a user."""
        for kind in ("peptide", "dna", "rna"):
            for code in available_residues(kind):
                tokens = [code]
                smiles = (
                    build_peptide(tokens) if kind == "peptide"
                    else build_nucleic(tokens, kind)
                )
                assert Chem.MolFromSmiles(smiles) is not None, f"{kind} {code}"

    # Known formulas for every nucleoside in the library. Parsing proves the
    # SMILES is well-formed; only this proves it is the right compound.
    NUCLEOSIDES = {
        ("dna", "A"): "C10H13N5O3", ("dna", "C"): "C9H13N3O4",
        ("dna", "G"): "C10H13N5O4", ("dna", "T"): "C10H14N2O5",
        ("dna", "U"): "C9H12N2O5", ("dna", "I"): "C10H12N4O4",
        ("dna", "5MC"): "C10H15N3O4", ("dna", "8OG"): "C10H13N5O5",
        ("dna", "BRU"): "C9H11BrN2O5",
        ("rna", "A"): "C10H13N5O4", ("rna", "C"): "C9H13N3O5",
        ("rna", "G"): "C10H13N5O5", ("rna", "U"): "C9H12N2O6",
        ("rna", "T"): "C10H14N2O6", ("rna", "I"): "C10H12N4O5",
        ("rna", "PSU"): "C9H12N2O6", ("rna", "5MC"): "C10H15N3O5",
        ("rna", "6MA"): "C11H15N5O4", ("rna", "7MG"): "C11H16N5O5+",
    }

    @pytest.mark.parametrize("key,formula", sorted(NUCLEOSIDES.items()))
    def test_nucleoside_formulas(self, key, formula):
        kind, code = key
        mol = Chem.MolFromSmiles(build_nucleic([code], kind))
        assert rdMolDescriptors.CalcMolFormula(mol) == formula

    def test_the_nucleoside_table_covers_the_whole_library(self):
        """So adding a residue without a formula check fails here."""
        for kind in ("dna", "rna"):
            for code in available_residues(kind):
                assert (kind, code) in self.NUCLEOSIDES, f"{kind} {code} unverified"

    def test_every_amino_acid_has_a_free_amine_and_acid(self):
        """Whatever the side chain, the backbone has to be able to chain."""
        for code in available_residues("peptide"):
            mol = Chem.MolFromSmiles(build_peptide([code]))
            assert mol.HasSubstructMatch(
                Chem.MolFromSmarts("[NX3;H1,H2][CX4][CX3](=O)[OX2H1]")
            ), f"{code} has no free amino-acid backbone"

    def test_every_code_has_a_name(self):
        for kind in ("peptide", "dna", "rna"):
            for code in available_residues(kind):
                assert residue_name(code, kind) != code, f"{kind} {code} has no name"

    def test_modified_residues_chain_like_canonical_ones(self):
        """A modified residue mid-chain must still be one residue, not a break."""
        plain = Chem.MolFromSmiles(build("GSKPL", "peptide").smiles)
        modified = Chem.MolFromSmiles(build("GS(KCX)PL", "peptide").smiles)
        assert len(Chem.GetMolFrags(modified)) == 1
        # One CO2 heavier than the unmodified peptide, and nothing else.
        assert modified.GetNumAtoms() == plain.GetNumAtoms() + 3


class TestRefusals:
    @pytest.mark.parametrize("code", ["N", "R", "Y", "W", "B", "V"])
    def test_iupac_ambiguity_codes_are_explained(self, code):
        """N means "any base": a set of sequences, not a molecule."""
        with pytest.raises(SequenceError, match="set of sequences"):
            build(f"GG{code}AT", "dna")

    def test_an_unknown_residue_lists_the_known_ones(self):
        with pytest.raises(SequenceError, match="Available:"):
            build("AGZ", "peptide")

    def test_an_unknown_bracketed_code(self):
        with pytest.raises(SequenceError, match="FOO"):
            build("A(FOO)G", "peptide")

    def test_an_unknown_kind(self):
        with pytest.raises(SequenceError, match="unknown sequence kind"):
            build("AG", "protein-ish")

    def test_t_is_not_a_plain_rna_base_but_is_offered(self):
        """Ribothymidine is real, so T builds — as the modified residue it is."""
        assert "ribothymidine" in " ".join(build("AGT", "rna").notes)


class TestReporting:
    def test_kinds_are_what_the_module_says_they_are(self):
        assert KINDS == ("peptide", "dna", "rna")

    def test_describes_what_was_built(self):
        assert build("PLYNGSG", "peptide").description == "7-residue peptide"
        assert build("GGCAT", "dna").description == "5-base DNA"
        assert build("AUGGC", "rna").description == "5-base RNA"

    def test_says_the_chain_direction(self):
        assert "5' to 3'" in " ".join(build("GGCAT", "dna").notes)
        assert "N terminus first" in " ".join(build("PLY", "peptide").notes)

    def test_without_a_ph_it_says_the_structure_is_neutral(self):
        notes = " ".join(build("DEKR", "peptide", ph=None).notes)
        assert "neutral" in notes and "assign at pH" in notes

    def test_with_a_ph_it_says_what_will_happen(self):
        notes = " ".join(build("DEKR", "peptide", ph=7.4).notes)
        assert "pH 7.4" in notes

    def test_nothing_is_actually_protonated_here(self):
        """The pipeline does protonation; doing it twice would be worse."""
        assert build("DEKR", "peptide", ph=7.4).smiles == build(
            "DEKR", "peptide", ph=None
        ).smiles

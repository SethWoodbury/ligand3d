"""Build peptides and nucleic acids from a sequence.

Typing `PLYNGSG` is a much better way to get a heptapeptide than drawing 50
atoms, and a DNA or RNA oligo is not realistically drawable at all.

RDKit already builds the canonical alphabets — `MolFromSequence` with the right
flavor covers the twenty amino acids and A/C/G/T/U — and it is used directly
wherever it can be. What it will not do is anything modified: a phosphoserine,
a carboxylated lysine, an inosine. Those are most of the reason to want this,
so residues are also assembled here from a monomer library, which is what makes
`GS(KCX)PL` possible.

The library is checked against RDKit rather than trusted. Every canonical
residue and every canonical chain built here is asserted to be the same
molecule RDKit produces, so a hand-written SMILES with an inverted stereocentre
fails the test suite instead of quietly producing the wrong enantiomer. That
matters: a peptide built with one D residue among the L residues looks entirely
plausible and is the wrong compound.

Naming follows the PDB Chemical Component Dictionary, so `(SEP)` here is the
same residue as `SEP` in a PDB file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rdkit import Chem

from .errors import Ligand3DError


class SequenceError(Ligand3DError):
    """A sequence could not be read or built."""


# --- amino acids -------------------------------------------------------
#
# A residue contributes `N[C@@H](side)C(=O)` to the chain, and the last one
# gets an -OH to close the C terminus. Ring-closure digits are reused per
# residue, which is legal because each residue's rings open and close inside
# its own fragment.

_L_UNIT = "N[C@@H]({side})C(=O)"

# code -> (name, side chain). Glycine and proline are shaped differently and
# carry a full unit instead; see _SPECIAL.
_SIDE_CHAINS: dict[str, tuple[str, str]] = {
    "A": ("alanine", "C"),
    "R": ("arginine", "CCCNC(=N)N"),
    "N": ("asparagine", "CC(N)=O"),
    "D": ("aspartate", "CC(=O)O"),
    "C": ("cysteine", "CS"),
    "E": ("glutamate", "CCC(=O)O"),
    "Q": ("glutamine", "CCC(N)=O"),
    "H": ("histidine", "Cc1c[nH]cn1"),
    "I": ("isoleucine", "[C@@H](C)CC"),
    "L": ("leucine", "CC(C)C"),
    "K": ("lysine", "CCCCN"),
    "M": ("methionine", "CCSC"),
    "F": ("phenylalanine", "Cc1ccccc1"),
    "S": ("serine", "CO"),
    "T": ("threonine", "[C@@H](C)O"),
    "W": ("tryptophan", "Cc1c[nH]c2ccccc12"),
    "Y": ("tyrosine", "Cc1ccc(O)cc1"),
    "V": ("valine", "C(C)C"),
}

# Residues whose backbone is not the standard N-CA(-side)-C=O.
_SPECIAL: dict[str, tuple[str, str]] = {
    "G": ("glycine", "NCC(=O)"),
    "P": ("proline", "N1CCC[C@H]1C(=O)"),
}

# Non-canonical residues and post-translational modifications, by PDB code.
# Written as side chains where the backbone is standard.
_MODIFIED_SIDE: dict[str, tuple[str, str]] = {
    "KCX": ("N6-carboxylysine", "CCCCNC(=O)O"),
    "ALY": ("N6-acetyllysine", "CCCCNC(C)=O"),
    "MLZ": ("N6-methyllysine", "CCCCNC"),
    "MLY": ("N6,N6-dimethyllysine", "CCCCN(C)C"),
    "M3L": ("N6,N6,N6-trimethyllysine", "CCCC[N+](C)(C)C"),
    "SEP": ("phosphoserine", "COP(=O)(O)O"),
    "TPO": ("phosphothreonine", "[C@@H](C)OP(=O)(O)O"),
    "PTR": ("phosphotyrosine", "Cc1ccc(OP(=O)(O)O)cc1"),
    "SEC": ("selenocysteine", "C[SeH]"),
    "MSE": ("selenomethionine", "CC[Se]C"),
    "CSO": ("S-hydroxycysteine", "CSO"),
    "CSD": ("S-cysteinesulfinic acid", "CS(=O)O"),
    "CIR": ("citrulline", "CCCNC(N)=O"),
    "ORN": ("ornithine", "CCCN"),
    "NLE": ("norleucine", "CCCC"),
    "NVA": ("norvaline", "CCC"),
    "ABA": ("2-aminobutyrate", "CC"),
    "DAB": ("2,4-diaminobutyrate", "CCN"),
    "HCS": ("homocysteine", "CCS"),
    "HSE": ("homoserine", "CCO"),
    "TYS": ("sulfotyrosine", "Cc1ccc(OS(=O)(=O)O)cc1"),
    "NIY": ("3-nitrotyrosine", "Cc1ccc(O)c([N+](=O)[O-])c1"),
    "OCS": ("cysteic acid", "CS(=O)(=O)O"),
    "PFF": ("4-fluorophenylalanine", "Cc1ccc(F)cc1"),
    "AZF": ("4-azidophenylalanine", "Cc1ccc(N=[N+]=[N-])cc1"),
    "BIF": ("4-benzoylphenylalanine", "Cc1ccc(C(=O)c2ccccc2)cc1"),
}

_MODIFIED_SPECIAL: dict[str, tuple[str, str]] = {
    "HYP": ("4-hydroxyproline", "N1C[C@H](O)C[C@H]1C(=O)"),
    "PCA": ("pyroglutamate", "N1C(=O)CC[C@H]1C(=O)"),
    "AIB": ("2-aminoisobutyrate", "NC(C)(C)C(=O)"),
    "SAR": ("sarcosine (N-methylglycine)", "N(C)CC(=O)"),
    "DAL": ("D-alanine", "N[C@H](C)C(=O)"),
    "DVA": ("D-valine", "N[C@H](C(C)C)C(=O)"),
    "DPR": ("D-proline", "N1CCC[C@@H]1C(=O)"),
}

# The two one-letter codes beyond the twenty, from the expanded genetic code.
_EXTENDED_ONE_LETTER = {"U": "SEC", "O": "PYL"}
_MODIFIED_SPECIAL["PYL"] = (
    "pyrrolysine",
    "N[C@@H](CCCCNC(=O)[C@@H]1CC=N[C@@H]1C)C(=O)",
)


def _peptide_unit(code: str) -> tuple[str, str]:
    """(display name, SMILES fragment) for one residue."""
    if code in _SPECIAL:
        return _SPECIAL[code]
    if code in _SIDE_CHAINS:
        name, side = _SIDE_CHAINS[code]
        return name, _L_UNIT.format(side=side)
    if code in _MODIFIED_SPECIAL:
        return _MODIFIED_SPECIAL[code]
    if code in _MODIFIED_SIDE:
        name, side = _MODIFIED_SIDE[code]
        return name, _L_UNIT.format(side=side)
    raise SequenceError(_unknown_residue_message(code, "peptide"))


# --- nucleic acids -----------------------------------------------------
#
# A chain is assembled monomer by monomer, joining residue i's 3'-OH to
# residue i+1's 5'-OH through a phosphate. The sugars below are RDKit's own,
# taken from `MolFromSequence`, so the stereochemistry is not hand-written.

_DNA_SUGAR = "[C@H]1C[C@H](O)[C@@H](CO)O1"
_RNA_SUGAR = "[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O"


def _sugar(template: str, ring: int) -> str:
    """The sugar with its ring-closure digit renumbered, for nesting."""
    return template.replace("1", str(ring))


_DNA_RESIDUES: dict[str, tuple[str, str]] = {
    "A": ("2'-deoxyadenosine", "Nc1ncnc2c1ncn2" + _DNA_SUGAR),
    "C": ("2'-deoxycytidine", f"Nc1ccn({_sugar(_DNA_SUGAR, 2)})c(=O)n1"),
    "G": ("2'-deoxyguanosine", f"Nc1nc2c(ncn2{_sugar(_DNA_SUGAR, 2)})c(=O)[nH]1"),
    "T": ("thymidine", f"Cc1cn({_sugar(_DNA_SUGAR, 2)})c(=O)[nH]c1=O"),
    "U": ("2'-deoxyuridine", f"O=c1ccn({_sugar(_DNA_SUGAR, 2)})c(=O)[nH]1"),
    "I": ("2'-deoxyinosine", "O=c1[nH]cnc2c1ncn2" + _DNA_SUGAR),
    "5MC": ("5-methyl-2'-deoxycytidine",
            f"Cc1cn({_sugar(_DNA_SUGAR, 2)})c(=O)nc1N"),
    "8OG": ("8-oxo-2'-deoxyguanosine",
            f"Nc1nc2c([nH]c(=O)n2{_sugar(_DNA_SUGAR, 2)})c(=O)[nH]1"),
    "BRU": ("5-bromo-2'-deoxyuridine",
            f"O=c1[nH]c(=O)n({_sugar(_DNA_SUGAR, 2)})cc1Br"),
}

_RNA_RESIDUES: dict[str, tuple[str, str]] = {
    "A": ("adenosine", "Nc1ncnc2c1ncn2" + _RNA_SUGAR),
    "C": ("cytidine", f"Nc1ccn({_sugar(_RNA_SUGAR, 2)})c(=O)n1"),
    "G": ("guanosine", f"Nc1nc2c(ncn2{_sugar(_RNA_SUGAR, 2)})c(=O)[nH]1"),
    "U": ("uridine", f"O=c1ccn({_sugar(_RNA_SUGAR, 2)})c(=O)[nH]1"),
    "T": ("ribothymidine", f"Cc1cn({_sugar(_RNA_SUGAR, 2)})c(=O)[nH]c1=O"),
    "I": ("inosine", "O=c1[nH]cnc2c1ncn2" + _RNA_SUGAR),
    "PSU": ("pseudouridine", f"O=c1[nH]cc({_sugar(_RNA_SUGAR, 2)})c(=O)[nH]1"),
    "5MC": ("5-methylcytidine", f"Cc1cn({_sugar(_RNA_SUGAR, 2)})c(=O)nc1N"),
    "6MA": ("N6-methyladenosine", "CNc1ncnc2c1ncn2" + _RNA_SUGAR),
    # Quaternising N7 leaves the imidazole cationic, which is why the mRNA cap
    # carries a positive charge. Writing it neutral does not kekulize.
    "7MG": ("7-methylguanosine (cation)",
            f"Nc1nc2c([n+](C)cn2{_sugar(_RNA_SUGAR, 2)})c(=O)[nH]1"),
}

# IUPAC codes that stand for "one of several", which is a set of molecules
# rather than a molecule.
_AMBIGUITY_CODES = {
    "N": "any base", "R": "A or G", "Y": "C or T/U", "W": "A or T/U",
    "S": "G or C", "K": "G or T/U", "M": "A or C", "B": "C, G or T/U",
    "D": "A, G or T/U", "H": "A, C or T/U", "V": "A, C or G",
}

KINDS = ("peptide", "dna", "rna")


@dataclass
class SequenceResult:
    """A built biopolymer and what was worth saying about it."""

    smiles: str
    kind: str
    residues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def description(self) -> str:
        unit = {"peptide": "residue", "dna": "base", "rna": "base"}[self.kind]
        label = {"peptide": "peptide", "dna": "DNA", "rna": "RNA"}[self.kind]
        return f"{len(self.residues)}-{unit} {label}"


_TOKEN = re.compile(r"\(([^)]*)\)|([A-Za-z])|(\s+)|(.)")


def parse_sequence(text: str) -> list[str]:
    """Split a sequence into residue codes.

    Single letters are residues; anything in parentheses is one residue given
    by its full code, which is how `GS(KCX)PL` names a modified lysine in the
    middle of a chain. Whitespace and the digits of a numbered listing are
    ignored, so a sequence pasted out of a paper mostly works.
    """
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.strip()):
        bracketed, letter, space, other = match.groups()
        if space:
            continue
        if bracketed is not None:
            code = bracketed.strip().upper()
            if not code:
                raise SequenceError("empty () in the sequence")
            tokens.append(code)
        elif letter:
            tokens.append(letter.upper())
        elif other and other.isdigit():
            continue  # residue numbering pasted along with the sequence
        elif other in ("-", "*", ".", ",", ":"):
            continue  # separators people put between residues
        else:
            raise SequenceError(
                f"{other!r} is not something a sequence can contain. Use one-letter "
                "codes, or (CODE) for a modified residue."
            )
    if not tokens:
        raise SequenceError("the sequence is empty")
    return tokens


def _unknown_residue_message(code: str, kind: str) -> str:
    if kind in ("dna", "rna") and code in _AMBIGUITY_CODES:
        return (
            f"{code!r} means {_AMBIGUITY_CODES[code]}, which is a set of sequences "
            "rather than one molecule. Pick the base you want."
        )
    known = ", ".join(sorted(available_residues(kind)))
    return f"{code!r} is not a {kind} residue ligand3d knows. Available: {known}"


def available_residues(kind: str) -> list[str]:
    """Every code that can be used for this kind of sequence."""
    if kind == "peptide":
        return sorted(
            set(_SIDE_CHAINS) | set(_SPECIAL) | set(_MODIFIED_SIDE)
            | set(_MODIFIED_SPECIAL) | set(_EXTENDED_ONE_LETTER)
        )
    if kind == "dna":
        return sorted(_DNA_RESIDUES)
    if kind == "rna":
        return sorted(_RNA_RESIDUES)
    raise SequenceError(f"unknown sequence kind {kind!r}")


def residue_name(code: str, kind: str) -> str:
    """The chemical name behind a code, for reporting what was built."""
    code = _EXTENDED_ONE_LETTER.get(code, code) if kind == "peptide" else code
    table = {
        "peptide": {**_SIDE_CHAINS, **_SPECIAL, **_MODIFIED_SIDE, **_MODIFIED_SPECIAL},
        "dna": _DNA_RESIDUES,
        "rna": _RNA_RESIDUES,
    }[kind]
    entry = table.get(code)
    return entry[0] if entry else code


def _is_modified(code: str, kind: str) -> bool:
    if kind == "peptide":
        return code in _MODIFIED_SIDE or code in _MODIFIED_SPECIAL or (
            code in _EXTENDED_ONE_LETTER
        )
    canonical = {"dna": set("ACGT"), "rna": set("ACGU")}[kind]
    return code not in canonical


def build_peptide(tokens: list[str]) -> str:
    """SMILES for a peptide, N terminus first, free amine and free acid."""
    parts = []
    for code in tokens:
        resolved = _EXTENDED_ONE_LETTER.get(code, code)
        _, fragment = _peptide_unit(resolved)
        parts.append(fragment)
    return "".join(parts) + "O"


def _termini(mol: Chem.Mol) -> tuple[int, int]:
    """(5'-O, 3'-O) atom indices of one nucleoside.

    The 5'-OH hangs off an exocyclic CH2. Ribose also carries a 2'-OH on a ring
    carbon, so "on a ring carbon" does not identify the 3'-OH by itself; being
    adjacent to C4', the ring carbon bearing that CH2, does.
    """
    five = three = -1
    c4 = -1
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "O" and atom.GetDegree() == 1:
            carbon = atom.GetNeighbors()[0]
            if carbon.GetSymbol() == "C" and not carbon.IsInRing():
                five = atom.GetIdx()
                c4 = next(
                    (n.GetIdx() for n in carbon.GetNeighbors()
                     if n.GetSymbol() == "C" and n.IsInRing()),
                    -1,
                )
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "O" and atom.GetDegree() == 1:
            carbon = atom.GetNeighbors()[0]
            if carbon.GetSymbol() == "C" and carbon.IsInRing():
                if any(n.GetIdx() == c4 for n in carbon.GetNeighbors()):
                    three = atom.GetIdx()
    if five < 0 or three < 0:
        raise SequenceError("could not find the 5' and 3' hydroxyls of a nucleoside")
    return five, three


def _link(chain: Chem.Mol, three: int, nxt: Chem.Mol, five: int) -> Chem.Mol:
    """Join two residues with a phosphodiester: 3'-O-P(=O)(OH)-O-5'."""
    offset = chain.GetNumAtoms()
    rw = Chem.RWMol(Chem.CombineMols(chain, nxt))
    phosphorus = rw.AddAtom(Chem.Atom(15))
    rw.AddBond(three, phosphorus, Chem.BondType.SINGLE)
    rw.AddBond(phosphorus, rw.AddAtom(Chem.Atom(8)), Chem.BondType.DOUBLE)
    rw.AddBond(phosphorus, rw.AddAtom(Chem.Atom(8)), Chem.BondType.SINGLE)
    rw.AddBond(phosphorus, five + offset, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def build_nucleic(tokens: list[str], kind: str) -> str:
    """SMILES for a DNA or RNA chain, written 5' to 3', no terminal phosphate."""
    table = _DNA_RESIDUES if kind == "dna" else _RNA_RESIDUES
    monomers = []
    for code in tokens:
        entry = table.get(code)
        if entry is None:
            raise SequenceError(_unknown_residue_message(code, kind))
        mol = Chem.MolFromSmiles(entry[1])
        if mol is None:  # pragma: no cover - a library typo, caught by tests
            raise SequenceError(f"the stored structure for {code!r} is not readable")
        monomers.append(mol)

    chain = monomers[0]
    _, three = _termini(chain)
    for nxt in monomers[1:]:
        next_five, next_three = _termini(nxt)
        offset = chain.GetNumAtoms()
        chain = _link(chain, three, nxt, next_five)
        # CombineMols appends, so an index in `nxt` just shifts by the offset.
        three = next_three + offset
    return Chem.MolToSmiles(chain)


def build(text: str, kind: str, ph: float | None = None) -> SequenceResult:
    """Build a sequence of the given kind into a molecule.

    `ph` is only used to say what will happen to the ionizable groups; nothing
    is protonated here. The pipeline does that, and doing it twice would be
    worse than not saying anything.
    """
    if kind not in KINDS:
        raise SequenceError(f"unknown sequence kind {kind!r}; use one of {KINDS}")

    tokens = parse_sequence(text)
    smiles = (
        build_peptide(tokens) if kind == "peptide" else build_nucleic(tokens, kind)
    )
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise SequenceError("the assembled sequence is not a readable structure")

    notes: list[str] = []
    modified = [c for c in tokens if _is_modified(c, kind)]
    if modified:
        named = ", ".join(
            f"{c} = {residue_name(c, kind)}" for c in dict.fromkeys(modified)
        )
        notes.append(f"non-standard residue(s): {named}")

    if kind == "peptide":
        notes.append(
            "built N terminus first, with a free amine and a free acid"
        )
        notes.append(_ph_note(ph, "the termini and any Asp, Glu, Lys, Arg or His"))
    else:
        label = "DNA" if kind == "dna" else "RNA"
        notes.append(
            f"{label} written 5' to 3', with free hydroxyls at both ends and no "
            "terminal phosphate"
        )
        notes.append(_ph_note(ph, "the phosphate backbone"))

    return SequenceResult(
        smiles=Chem.MolToSmiles(mol), kind=kind, residues=tokens, notes=notes
    )


def _ph_note(ph: float | None, groups: str) -> str:
    """What the current protonation setting will do to the ionizable groups."""
    if ph is None:
        return (
            f"drawn neutral: {groups} are drawn uncharged. Set Protonation to "
            "'assign at pH' if you want the charge state at a given pH."
        )
    return (
        f"{groups} will be set to their state at pH {ph:g} when this is built, "
        "because Protonation is set to assign at that pH."
    )

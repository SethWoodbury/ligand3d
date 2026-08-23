"""Write an annotated mmCIF, the native input format for RFdiffusion4-Proteina.

An annotated CIF is an ordinary mmCIF with extra columns on the `_atom_site`
loop. There is no new format and no sidecar — a plain mmCIF is already a valid
annotated CIF, one where every annotation takes its default. The columns say
what the model should *do* with each atom.

The mental model that prevents most mistakes:

    mask = True means CONDITIONED. There is no "diffuse this" flag.
    Diffusion is the absence of conditioning.

So this writes what to **hold fixed**, not what to build.

What ligand3d contributes is a ligand with a real conformer, which is exactly
the thing a binder- or enzyme-design run needs to be given rather than asked to
invent. The file it writes says: here is the ligand, in this pose, with this
identity — now design a protein around it.

Three rules from the format make the whole difference, and all three fail
silently if broken:

**Masks must be the literal strings `True`/`False`.** Dtype inference tries int
before bool, so `1`/`0` parses as int64 and fails much later with "Mask must be
a boolean array".

**A misspelled tag is dropped with no diagnostic at all.** Tag names here are
spelled against the registry in the format spec and pinned by tests.

**Bond orders must be Kekulé.** A bond left generically aromatic is cast to
single, which would tell the model an aromatic ring is saturated. `write.py`
kekulizes for this reason.

Two invariants govern a ligand specifically. Non-polymers are *atomized*, and
atomized tokens must be sequence-conditioned — they are context, never
generated. And coordinate conditioning on any non-backbone atom requires
sequence conditioning, because the atom37 slot occupancy fingerprints the
residue anyway. Every ligand atom is non-backbone, so both point the same way:
a coordinate-conditioned ligand is always sequence-conditioned too, and this
module cannot emit the combination that would be rejected.

Verified against the real validator, `check_annotated_cif.py` from
RFD4-Proteina-dev, not against this description.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem

from .errors import Ligand3DError
from .write import _MAX_RESNAME, to_cif_string

#: Written for every atom, in this order. Names come from the annotation
#: registry: {prefix}_{name}_{n_body}_{level}. A typo produces a column the
#: parser silently ignores, so these are pinned by tests rather than trusted.
COORDINATE_MASK = "mask_coordinate_1_atom"
SEQUENCE_MASK = "mask_sequence_1_residue"
SEQUENCE_VALUE = "condition_sequence_1_residue"
EXPSEG_MIN_MASK = "mask_expsegmin_1_residue"
EXPSEG_MAX_MASK = "mask_expsegmax_1_residue"
EXPSEG_MIN = "annotation_expsegmin_1_residue"
EXPSEG_MAX = "annotation_expsegmax_1_residue"

ANNOTATION_TAGS = (
    COORDINATE_MASK,
    SEQUENCE_MASK,
    SEQUENCE_VALUE,
    EXPSEG_MIN_MASK,
    EXPSEG_MAX_MASK,
    EXPSEG_MIN,
    EXPSEG_MAX,
)

#: The value that means "no expandable segment here".
NO_SEGMENT = "-1"

#: Booleans must be these exact strings. `1`/`0` parses as int and breaks later.
TRUE, FALSE = "True", "False"


@dataclass(frozen=True)
class DesignSegment:
    """A stretch of protein for the model to build, given as a length range.

    Written as a single sentinel atom rather than as residues, which is what
    lets the model sample a different length for every replicate. A segment
    realized to a concrete length is frozen and will not be resampled, so the
    collapsed form is the useful one.
    """

    minimum: int
    maximum: int
    chain: str = "B"

    def __post_init__(self) -> None:
        if self.minimum < 1 or self.maximum < self.minimum:
            raise Ligand3DError(
                f"a design segment needs 1 <= min <= max, got {self.minimum}-{self.maximum}"
            )

    @property
    def label(self) -> str:
        """`min-max`, which is what goes in the sentinel's residue name."""
        return f"{self.minimum}-{self.maximum}"


def parse_segment(text: str, chain: str = "B") -> DesignSegment:
    """Read `120` or `100-155` as a design segment."""
    cleaned = str(text).strip()
    try:
        if "-" in cleaned:
            low, high = (int(part) for part in cleaned.split("-", 1))
        else:
            low = high = int(cleaned)
    except ValueError as exc:
        raise Ligand3DError(
            f"could not read {text!r} as a length. Use 120, or a range like 100-155."
        ) from exc
    return DesignSegment(low, high, chain=chain)


def to_annotated_cif_string(
    mol: Chem.Mol,
    records=None,
    resname: str = "LIG",
    smiles: str | None = None,
    name: str = "ligand3d",
    fix_coordinates: bool = True,
    design: DesignSegment | None = None,
) -> str:
    """Render one conformer as an annotated mmCIF.

    `fix_coordinates` pins the ligand where ligand3d put it. Turning it off
    hands the model the ligand's identity but lets it choose the pose, which is
    the right choice when the conformer is a guess rather than a measurement.
    """
    try:
        import gemmi
    except ImportError as exc:  # pragma: no cover - guarded by cif_available
        raise Ligand3DError(
            "writing mmCIF needs gemmi. Install it with: pip install gemmi"
        ) from exc

    if mol.GetNumConformers() > 1:
        # Each replicate conditions on one pose. Several models in one file
        # would silently condition on the first, so say so instead.
        raise Ligand3DError(
            f"an annotated CIF describes one pose, but this has "
            f"{mol.GetNumConformers()} conformers. Write them separately, or "
            f"build with --confs 1."
        )

    text = to_cif_string(
        mol, records=records, resname=resname, smiles=smiles, name=name
    )
    document = gemmi.cif.read_string(text)
    block = document.sole_block()

    loop = block.find_loop("_atom_site.id").get_loop()
    if loop is None:  # pragma: no cover - gemmi always writes a loop here
        raise Ligand3DError("the mmCIF has no _atom_site loop to annotate")

    comp = (resname or "LIG").upper()[:_MAX_RESNAME]
    n_atoms = loop.length()
    width = loop.width()
    existing = list(loop.values)

    # gemmi's `loop.values` is a copy, so assigning into it changes nothing.
    # The whole table is rebuilt column-wise and handed to set_all_values,
    # which is also what lets the sentinel row be appended in one step.
    columns = [
        [existing[row * width + col] for row in range(n_atoms)]
        for col in range(width)
    ]

    per_atom = {
        COORDINATE_MASK: TRUE if fix_coordinates else FALSE,
        # Not optional: a non-polymer is atomized, and an atomized token that
        # is not sequence-conditioned is rejected. It is also what the
        # coordinate rule demands, since no ligand atom is a backbone atom.
        SEQUENCE_MASK: TRUE,
        SEQUENCE_VALUE: comp,
        EXPSEG_MIN_MASK: FALSE,
        EXPSEG_MAX_MASK: FALSE,
        EXPSEG_MIN: NO_SEGMENT,
        EXPSEG_MAX: NO_SEGMENT,
    }
    tags = list(loop.tags) + [f"_atom_site.{tag}" for tag in ANNOTATION_TAGS]
    for tag in ANNOTATION_TAGS:
        columns.append([per_atom[tag]] * n_atoms)

    if design is not None:
        for index, tag in enumerate(tags):
            field = tag.split(".", 1)[1]
            maker = _SENTINEL.get(field)
            columns[index].append(maker(design, n_atoms) if maker else ".")

    loop.add_columns([f"_atom_site.{tag}" for tag in ANNOTATION_TAGS], value="?")
    loop.set_all_values(columns)

    _fill_formal_charges(block, mol)
    if design is not None:
        _declare_polymer(block, design, comp)
    return document.as_string()


def _fill_formal_charges(block, mol: Chem.Mol) -> None:
    """Replace `?` charges with the real value, which is usually zero.

    `?` means *unknown*, and the charges reach the model. A neutral atom is not
    an atom of unknown charge, and the validator says so — mixed known and
    unknown "looks deliberate but isn't".
    """
    loop = block.find_loop("_atom_site.id").get_loop()
    tag = "_atom_site.pdbx_formal_charge"
    if loop is None or tag not in loop.tags:
        return

    width, column = loop.width(), loop.tags.index(tag)
    values = list(loop.values)
    charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    for row in range(loop.length()):
        # Rows past the molecule are the sentinel, whose charge is genuinely
        # not a number.
        if row < len(charges) and values[row * width + column] == "?":
            values[row * width + column] = str(charges[row])
    loop.set_all_values(
        [[values[r * width + c] for r in range(loop.length())] for c in range(width)]
    )


def _declare_polymer(block, design: DesignSegment, comp: str) -> None:
    """Declare the sentinel's chain as a polypeptide entity.

    Without this the segment is read as a non-polymer and rejected with "Only
    polypeptide expandable segments are currently supported" — the sentinel row
    alone does not say what kind of chain is being asked for.
    """
    entities = block.find_loop("_entity.id").get_loop()
    if entities is not None and "_entity.type" in entities.tags:
        width = entities.width()
        values = list(entities.values)
        ids = entities.tags.index("_entity.id")
        types = entities.tags.index("_entity.type")
        rows = [
            [values[r * width + c] for c in range(width)]
            for r in range(entities.length())
        ]
        row = ["." for _ in range(width)]
        row[ids], row[types] = "2", "polymer"
        rows.append(row)
        entities.set_all_values(
            [[r[c] for r in rows] for c in range(width)]
        )

    poly = block.init_mmcif_loop(
        "_entity_poly.",
        ["entity_id", "type", "nstd_linkage", "nstd_monomer",
         "pdbx_seq_one_letter_code", "pdbx_seq_one_letter_code_can",
         "pdbx_strand_id", "pdbx_target_identifier"],
    )
    poly.add_row(["2", "polypeptide(L)", "no", "yes", "X", "X", design.chain, "?"])

    seq = block.init_mmcif_loop(
        "_entity_poly_seq.", ["entity_id", "hetero", "mon_id", "num"]
    )
    seq.add_row(["2", "n", design.label, "1"])

    # The segment label is a residue name like any other, so it needs a
    # chem_comp entry beside the ligand's.
    comps = block.find_loop("_chem_comp.id").get_loop()
    if comps is not None:
        width = comps.width()
        values = list(comps.values)
        rows = [
            [values[r * width + c] for c in range(width)]
            for r in range(comps.length())
        ]
        row = ["?" for _ in range(width)]
        row[comps.tags.index("_chem_comp.id")] = design.label
        if "_chem_comp.type" in comps.tags:
            row[comps.tags.index("_chem_comp.type")] = "NON-POLYMER"
        rows.append(row)
        comps.set_all_values([[r[c] for r in rows] for c in range(width)])


#: What the sentinel row holds, by `_atom_site` field. Anything not listed gets
#: `.`, the mmCIF "not applicable".
_SENTINEL = {
    "group_PDB": lambda d, n: "HETATM",
    "id": lambda d, n: str(n + 1),
    "type_symbol": lambda d, n: "X",
    "label_atom_id": lambda d, n: "UNK",
    "auth_atom_id": lambda d, n: "UNK",
    "label_alt_id": lambda d, n: ".",
    "label_comp_id": lambda d, n: d.label,
    "auth_comp_id": lambda d, n: d.label,
    "label_asym_id": lambda d, n: d.chain,
    "auth_asym_id": lambda d, n: d.chain,
    "label_entity_id": lambda d, n: "2",
    "label_seq_id": lambda d, n: "1",
    "auth_seq_id": lambda d, n: "1",
    "pdbx_PDB_ins_code": lambda d, n: ".",
    # nan, not 0: a real coordinate here would place the designed chain.
    "Cartn_x": lambda d, n: "nan",
    "Cartn_y": lambda d, n: "nan",
    "Cartn_z": lambda d, n: "nan",
    "occupancy": lambda d, n: "0.0",
    "B_iso_or_equiv": lambda d, n: "nan",
    "pdbx_formal_charge": lambda d, n: "?",
    "pdbx_PDB_model_num": lambda d, n: "1",
    COORDINATE_MASK: lambda d, n: FALSE,
    SEQUENCE_MASK: lambda d, n: FALSE,
    SEQUENCE_VALUE: lambda d, n: "<M>",   # unquoted: the "designed" sentinel
    EXPSEG_MIN_MASK: lambda d, n: TRUE,
    EXPSEG_MAX_MASK: lambda d, n: TRUE,
    EXPSEG_MIN: lambda d, n: str(d.minimum),
    EXPSEG_MAX: lambda d, n: str(d.maximum),
}


def write_annotated_cif(
    path: str | Path,
    mol: Chem.Mol,
    records=None,
    resname: str = "LIG",
    smiles: str | None = None,
    name: str = "ligand3d",
    fix_coordinates: bool = True,
    design: DesignSegment | None = None,
) -> Path:
    """Write an annotated mmCIF for RFdiffusion4."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        to_annotated_cif_string(
            mol, records=records, resname=resname, smiles=smiles, name=name,
            fix_coordinates=fix_coordinates, design=design,
        )
    )
    return target


def describe(fix_coordinates: bool, design: DesignSegment | None) -> list[str]:
    """What the file says, in the terms the model will read it in."""
    lines = [
        "ligand is sequence-conditioned: its identity is given, never designed"
    ]
    lines.append(
        "ligand coordinates are fixed: the pose ligand3d built is the pose used"
        if fix_coordinates
        else "ligand coordinates are free: the model chooses the pose"
    )
    if design is not None:
        lines.append(
            f"chain {design.chain} is an expandable segment of {design.label} "
            "residues, resampled per replicate"
        )
    else:
        lines.append(
            "no design segment: the file conditions on the ligand only, so pair "
            "it with a contig or a condition spec that says what to build"
        )
    return lines

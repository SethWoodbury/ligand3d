"""Reading, standardizing, and auditing the input molecule.

The audit is the interesting part. Before any 3D work happens we record exactly
which stereocentres and double bonds carry a definite assignment, so that after
embedding we can prove the 3D structure encodes the stereochemistry that was
drawn rather than an arbitrary one.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdCIPLabeler, rdMolDescriptors

from .errors import InputError, StereoError


@contextlib.contextmanager
def rdkit_quiet():
    """Silence RDKit's C++ logger.

    Needed around dimorphite-dl, which builds transiently invalid molecules and
    lets RDKit complain about them on stderr.
    """
    RDLogger.DisableLog("rdApp.*")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.*")


@dataclass(frozen=True)
class StereoAudit:
    """What stereochemistry the input specifies, and what it leaves open."""

    assigned_centers: tuple[tuple[int, str], ...] = ()
    unassigned_centers: tuple[int, ...] = ()
    assigned_bonds: tuple[tuple[int, int, str], ...] = ()
    unassigned_bonds: tuple[tuple[int, int], ...] = ()

    @property
    def has_undefined(self) -> bool:
        return bool(self.unassigned_centers or self.unassigned_bonds)

    @property
    def n_defined(self) -> int:
        return len(self.assigned_centers) + len(self.assigned_bonds)

    def describe(self) -> str:
        bits = []
        if self.assigned_centers:
            bits.append(
                "stereocenters " + ", ".join(f"{i}{lab}" for i, lab in self.assigned_centers)
            )
        if self.assigned_bonds:
            bits.append(
                "double bonds " + ", ".join(f"{a}-{b}{lab}" for a, b, lab in self.assigned_bonds)
            )
        if not bits:
            return "no defined stereochemistry"
        return "; ".join(bits)


def _cip_labels(mol: Chem.Mol) -> tuple[dict[int, str], list[int]]:
    """Return {atom_idx: CIP code} for assigned centres, plus unassigned indices.

    Uses the accurate CIP labeller rather than the legacy implementation, which
    gets fused and bridged ring systems wrong.
    """
    assigned: dict[int, str] = {}
    unassigned: list[int] = []
    with rdkit_quiet():
        rdCIPLabeler.AssignCIPLabels(mol)
        found = Chem.FindMolChiralCenters(
            mol, useLegacyImplementation=False, includeUnassigned=True
        )
    for idx, code in found:
        if code == "?":
            unassigned.append(idx)
        else:
            assigned[idx] = code
    return assigned, unassigned


def _bond_stereo(mol: Chem.Mol) -> tuple[list[tuple[int, int, str]], list[tuple[int, int]]]:
    """Return assigned and unassigned stereogenic double bonds.

    Normalized to CIP E/Z. RDKit also uses STEREOCIS/STEREOTRANS, which are
    defined relative to a pair of reference atoms and therefore are NOT
    comparable between a 2D input and a structure re-perceived from 3D. Comparing
    them directly is a real trap; this normalization is what avoids it.
    """
    assigned: list[tuple[int, int, str]] = []
    unassigned: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        st = bond.GetStereo()
        i, j = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        name = str(st)
        if st == Chem.BondStereo.STEREONONE:
            continue
        if st == Chem.BondStereo.STEREOANY:
            unassigned.append((i, j))
        elif name in ("STEREOE", "STEREOZ"):
            assigned.append((i, j, name[-1]))
        else:
            # Everything else — STEREOCIS/STEREOTRANS, and the atropisomer
            # labels — reaches here. Callers always run
            # AssignStereochemistry(cleanIt=True, force=True) first, which
            # rewrites CIS/TRANS into E/Z, so in practice this is only the
            # atropisomer path.
            #
            # CIS/TRANS is defined relative to a pair of reference atoms while
            # E/Z is CIP, so the two families must never be compared against
            # each other. Prefer the CIP code when the labeller set one, and
            # otherwise keep the raw label, which still compares like for like
            # between the input and the structure perceived from 3D. Dropping
            # it instead would silently stop checking a defined stereo element.
            code = bond.GetPropsAsDict().get("_CIPCode")
            assigned.append((i, j, code if code in ("E", "Z") else name))
    return assigned, unassigned


def audit_stereo(mol: Chem.Mol) -> StereoAudit:
    """Record the stereochemistry a molecule currently specifies."""
    work = Chem.Mol(mol)
    with rdkit_quiet():
        Chem.AssignStereochemistry(work, cleanIt=True, force=True)
        rdCIPLabeler.AssignCIPLabels(work)
    assigned, unassigned = _cip_labels(work)
    bonds_a, bonds_u = _bond_stereo(work)
    return StereoAudit(
        assigned_centers=tuple(sorted(assigned.items())),
        unassigned_centers=tuple(sorted(unassigned)),
        assigned_bonds=tuple(sorted(bonds_a)),
        unassigned_bonds=tuple(sorted(bonds_u)),
    )


@dataclass
class Molecule:
    """An input molecule plus everything we learned while reading it."""

    mol: Chem.Mol
    source: str
    name: str = "LIG"
    stereo: StereoAudit = field(default_factory=StereoAudit)
    notes: list[str] = field(default_factory=list)
    """Things the user should know about how this molecule was derived."""

    @property
    def smiles(self) -> str:
        return Chem.MolToSmiles(Chem.RemoveHs(self.mol))

    @property
    def formal_charge(self) -> int:
        return Chem.GetFormalCharge(self.mol)

    @property
    def formula(self) -> str:
        return rdMolDescriptors.CalcMolFormula(self.mol)

    @property
    def is_zwitterion(self) -> bool:
        """Net-neutral but carrying both a positive and a negative site.

        Worth knowing on its own: zwitterions need implicit solvation to survive
        minimization even though their net charge is zero, so a net-charge check
        alone would miss them.
        """
        if self.formal_charge != 0:
            return False
        charges = [a.GetFormalCharge() for a in self.mol.GetAtoms()]
        return any(c > 0 for c in charges) and any(c < 0 for c in charges)

    @property
    def elements(self) -> frozenset[int]:
        return frozenset(a.GetAtomicNum() for a in self.mol.GetAtoms())

    @property
    def fragments(self) -> tuple[str, ...]:
        """Canonical SMILES of each disconnected component."""
        return tuple(sorted(self.smiles.split(".")))

    @property
    def n_fragments(self) -> int:
        return len(Chem.GetMolFrags(self.mol))


def require_single_fragment(molecule: Molecule) -> None:
    """Refuse disconnected inputs such as a salt or a solvate.

    Distance geometry has no restraints between disconnected components, so
    ETKDG places them on top of one another — measured inter-fragment distances
    of 0.0 Å, and a force field with a fixed bond list cannot pull them apart.
    The result looks like a structure and is physically impossible.

    Rather than emit that, or guess at a placement nobody asked for, say so.
    """
    if molecule.n_fragments <= 1:
        return
    parts = " + ".join(molecule.fragments)
    raise InputError(
        f"input has {molecule.n_fragments} disconnected fragments ({parts}). "
        "ligand3d builds one molecule at a time, and 3D embedding would stack "
        "the fragments on top of each other. Build the component you want on "
        "its own, or pass --largest-fragment to keep the biggest one."
    )


def largest_fragment(molecule: Molecule) -> Molecule:
    """Keep the largest disconnected component, discarding counterions."""
    frags = Chem.GetMolFrags(molecule.mol, asMols=True, sanitizeFrags=True)
    if len(frags) <= 1:
        return molecule
    biggest = max(frags, key=lambda m: (m.GetNumHeavyAtoms(), m.GetNumAtoms()))
    kept = Molecule(
        mol=biggest,
        source=f"{molecule.source} [largest of {len(frags)} fragments]",
        name=molecule.name,
        stereo=audit_stereo(biggest),
        notes=list(molecule.notes),
    )
    dropped = sorted(
        Chem.MolToSmiles(f) for f in frags if f is not biggest
    )
    kept.notes.append(f"discarded {len(frags) - 1} smaller fragment(s): {', '.join(dropped)}")
    return kept


def from_smiles(smiles: str, name: str = "LIG") -> Molecule:
    """Parse a SMILES string, keeping any stereo annotations it carries."""
    smiles = smiles.strip()
    if not smiles:
        raise InputError("empty SMILES string")
    with rdkit_quiet():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InputError(
            f"could not parse SMILES {smiles!r}. "
            "Check valences and ring closures; RDKit rejected it outright."
        )
    return _finish(mol, source=f"smiles:{smiles}", name=name)


def from_molblock(block: str, name: str = "LIG") -> Molecule:
    """Parse a MOL/SDF block, e.g. what the sketcher posts back."""
    with rdkit_quiet():
        mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
    if mol is None:
        raise InputError("could not parse molblock; RDKit rejected it")
    # A molblock drawn in 2D carries wedge/hash bonds. Perceive stereo from those
    # before we throw the 2D coordinates away.
    with rdkit_quiet():
        if mol.GetNumConformers() and not mol.GetConformer().Is3D():
            Chem.AssignChiralTypesFromBondDirs(mol)
            Chem.AssignStereochemistryFrom3D  # noqa: B018 - documented no-op guard
            Chem.DetectBondStereoChemistry(mol, mol.GetConformer())
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return _finish(mol, source="molblock", name=name)


def from_file(path: str | os.PathLike[str], name: str | None = None) -> Molecule:
    """Read the first molecule from a .mol, .sdf, .smi, or .smiles file."""
    p = Path(path)
    if not p.exists():
        raise InputError(f"no such file: {p}")
    suffix = p.suffix.lower()
    stem = name or p.stem[:3].upper() or "LIG"

    if suffix in (".mol", ".sdf", ".mdl"):
        with rdkit_quiet():
            supplier = Chem.SDMolSupplier(str(p), sanitize=True, removeHs=False)
            mols = [m for m in supplier if m is not None]
        if not mols:
            raise InputError(f"no readable molecule in {p}")
        mol = mols[0]
        if len(mols) > 1:
            # Not an error, but the user should know we ignored the rest.
            mol.SetProp("_ligand3d_note", f"read 1 of {len(mols)} records in {p.name}")
        return _finish(mol, source=str(p), name=stem)

    if suffix in (".smi", ".smiles", ".txt"):
        first = ""
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                first = line
                break
        if not first:
            raise InputError(f"no SMILES found in {p}")
        parts = first.split()
        return from_smiles(parts[0], name=(parts[1][:3].upper() if len(parts) > 1 else stem))

    raise InputError(
        f"unrecognized file type {suffix!r}. Supported: .mol .sdf .smi .smiles"
    )


def read_input(spec: str, name: str | None = None) -> Molecule:
    """Read from whatever the user typed: a path if it exists, else a SMILES."""
    candidate = Path(spec)
    if candidate.exists() and candidate.is_file():
        return from_file(candidate, name=name)
    return from_smiles(spec, name=name or "LIG")


def _finish(mol: Chem.Mol, source: str, name: str) -> Molecule:
    """Sanitize, normalize hydrogens, perceive stereo, and audit.

    Shared tail of every reader.
    """
    try:
        with rdkit_quiet():
            Chem.SanitizeMol(mol)
            # Collapse explicit hydrogens into implicit counts.
            #
            # This is what keeps atom indices meaningful. Stereo is audited here
            # and re-checked later against a structure that has been through
            # AddHs and RemoveHs, and RemoveHs yields heavy-atom-only numbering.
            # A .mol/.sdf file that lists its hydrogens first — legal and common
            # — would otherwise audit as "stereocenter at atom 8" and verify as
            # "stereocenter at atom 1", and every such file would be rejected as
            # having lost its stereochemistry. AddHs re-adds them at the end, so
            # heavy-atom indices survive the round trip.
            #
            # RemoveHs deliberately keeps isotope-labelled hydrogens, so
            # deuterium is not silently discarded.
            mol = Chem.RemoveHs(mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception as exc:  # RDKit raises bare exceptions here
        raise InputError(f"molecule failed sanitization: {exc}") from exc
    if mol.GetNumAtoms() == 0:
        raise InputError("molecule has no atoms")
    return Molecule(mol=mol, source=source, name=(name or "LIG")[:3].upper(),
                    stereo=audit_stereo(mol))


@dataclass(frozen=True)
class DoubleBondReport:
    """A stereogenic double bond, described for a human."""

    begin: int
    end: int
    cip: str
    """E or Z, from CIP priorities. Always well defined."""
    cis_trans: str | None
    """"cis" or "trans", or None when the term does not apply.

    See `describe_double_bonds` for why this is often None.
    """
    n_hydrogens: tuple[int, int] = (0, 0)

    def describe(self) -> str:
        core = f"bond {self.begin}-{self.end} = {self.cip}"
        if self.cis_trans:
            return f"{core} ({self.cis_trans})"
        return f"{core} (cis/trans not applicable: {self._why}) "

    @property
    def _why(self) -> str:
        left, right = self.n_hydrogens
        if left == 0 and right == 0:
            return "tetrasubstituted"
        return "trisubstituted"


def describe_double_bonds(molecule: "Molecule") -> list[DoubleBondReport]:
    """Report each stereogenic double bond as CIP E/Z, plus cis/trans if valid.

    E/Z and cis/trans are not synonyms. E/Z is defined by CIP priority of the
    substituents on each alkene carbon and is always unambiguous. cis/trans
    describes two *reference* substituents being on the same or opposite side,
    which is only meaningful when it is obvious which two atoms are meant.

    When each alkene carbon carries exactly one hydrogen — the common
    1,2-disubstituted case — there is exactly one substituent per carbon to
    compare, that comparison is what CIP ranks, and the two systems coincide:
    Z is cis and E is trans.

    For a tri- or tetrasubstituted alkene they do not coincide, because "cis to
    what?" has no single answer. Tamoxifen is the standard example: it is
    unambiguously (Z) by CIP, both alkene carbons are fully substituted, and
    older literature describes the same geometry as "trans" with respect to the
    two phenyl rings. So cis/trans is reported only where it is defensible, and
    E/Z is always reported.
    """
    audit = molecule.stereo
    if not audit.assigned_bonds:
        return []

    work = Chem.Mol(molecule.mol)
    with rdkit_quiet():
        Chem.AssignStereochemistry(work, cleanIt=True, force=True)

    counts: dict[tuple[int, int], tuple[int, int]] = {}
    for bond in work.GetBonds():
        key = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        counts[key] = (
            bond.GetBeginAtom().GetTotalNumHs(),
            bond.GetEndAtom().GetTotalNumHs(),
        )

    reports = []
    for begin, end, cip in audit.assigned_bonds:
        hydrogens = counts.get((begin, end), (0, 0))
        unambiguous = hydrogens[0] == 1 and hydrogens[1] == 1
        cis_trans = None
        if unambiguous and cip in ("E", "Z"):
            cis_trans = "cis" if cip == "Z" else "trans"
        reports.append(
            DoubleBondReport(
                begin=begin, end=end, cip=cip,
                cis_trans=cis_trans, n_hydrogens=hydrogens,
            )
        )
    return reports


def count_embeddable_isomers(mol: Chem.Mol, limit: int = 2) -> int:
    """Count distinct, geometrically realizable isomers of the undefined centers.

    This is what separates a real ambiguity from a bookkeeping artifact. RDKit's
    stereo perception flags the bridgehead atoms of bicyclo[2.2.2] systems as
    potential stereocenters because a graph-based analysis cannot see that the
    cage constrains them. 3-quinuclidinone would otherwise be rejected as having
    two undefined stereocenters when it is in fact achiral.

    Enumerating with `tryEmbedding=True` discards combinations that cannot exist
    in three dimensions, which resolves those cases correctly while still
    catching genuine ambiguity such as an unspecified alanine alpha carbon.
    """
    from rdkit.Chem.EnumerateStereoisomers import (
        EnumerateStereoisomers,
        StereoEnumerationOptions,
    )

    opts = StereoEnumerationOptions(
        onlyUnassigned=True, unique=True, maxIsomers=limit, tryEmbedding=True
    )
    with rdkit_quiet():
        seen = set()
        for isomer in EnumerateStereoisomers(mol, options=opts):
            seen.add(Chem.MolToSmiles(isomer))
            if len(seen) >= limit:
                break
    return len(seen)


def has_real_stereo_ambiguity(molecule: Molecule) -> bool:
    """True if the undefined stereo elements genuinely give different molecules."""
    if not molecule.stereo.has_undefined:
        return False
    return count_embeddable_isomers(molecule.mol, limit=2) > 1


def require_defined_stereo(molecule: Molecule) -> None:
    """Raise unless every stereogenic element carries an assignment.

    Called by default so that an ambiguous drawing never silently becomes one
    arbitrary diastereomer. Candidate centers that the ring system already
    constrains are not an ambiguity and do not raise.
    """
    audit = molecule.stereo
    if not audit.has_undefined:
        return
    if not has_real_stereo_ambiguity(molecule):
        # Flagged as potential stereocenters, but only one isomer can exist.
        return
    parts = []
    if audit.unassigned_centers:
        atoms = ", ".join(
            f"atom {i} ({molecule.mol.GetAtomWithIdx(i).GetSymbol()})"
            for i in audit.unassigned_centers
        )
        parts.append(f"undefined stereocenters: {atoms}")
    if audit.unassigned_bonds:
        parts.append(
            "undefined double-bond geometry: "
            + ", ".join(f"{a}-{b}" for a, b in audit.unassigned_bonds)
        )
    raise StereoError(
        "; ".join(parts)
        + ". Specify it in the drawing or SMILES, or pass '--stereo any' to let RDKit "
        "pick one arbitrarily, or '--stereo enumerate' to build every isomer."
    )


def enumerate_stereoisomers(molecule: Molecule, max_isomers: int = 32) -> list[Molecule]:
    """Expand undefined stereochemistry into concrete isomers."""
    from rdkit.Chem.EnumerateStereoisomers import (
        EnumerateStereoisomers,
        StereoEnumerationOptions,
    )

    # tryEmbedding discards combinations that cannot exist in 3D, so a bridged
    # bicyclic does not produce phantom diastereomers.
    opts = StereoEnumerationOptions(
        onlyUnassigned=True, maxIsomers=max_isomers, unique=True, tryEmbedding=True
    )
    with rdkit_quiet():
        isomers = list(EnumerateStereoisomers(molecule.mol, options=opts))
    out = []
    for n, iso in enumerate(isomers, start=1):
        out.append(
            Molecule(
                mol=iso,
                source=f"{molecule.source} [stereoisomer {n}/{len(isomers)}]",
                name=molecule.name,
                stereo=audit_stereo(iso),
            )
        )
    return out


def read_3d(path: str | os.PathLike[str]) -> Chem.Mol:
    """Load a molecule that already has 3D coordinates, keeping them and its Hs.

    Distinct from `from_file`, which normalizes hydrogens away because it is
    preparing an input for embedding. Here the coordinates *are* the payload —
    this is what the per-step commands (`minimize`, `conformers`, `params`)
    operate on — so nothing is stripped, and every conformer in the file is
    kept.
    """
    p = Path(path)
    if not p.exists():
        raise InputError(f"no such file: {p}")

    suffix = p.suffix.lower()
    mol: Chem.Mol | None = None

    with rdkit_quiet():
        if suffix in (".sdf", ".mol", ".mdl"):
            supplier = Chem.SDMolSupplier(str(p), sanitize=True, removeHs=False)
            mols = [m for m in supplier if m is not None]
            if mols:
                mol = mols[0]
                # Extra records are treated as further conformers of the first,
                # which is how a conformer ensemble is normally stored.
                for extra in mols[1:]:
                    if extra.GetNumAtoms() == mol.GetNumAtoms() and extra.GetNumConformers():
                        mol.AddConformer(
                            Chem.Conformer(extra.GetConformer()), assignId=True
                        )
        elif suffix == ".pdb":
            mol = Chem.MolFromPDBFile(str(p), sanitize=True, removeHs=False)
        elif suffix in (".cif", ".mmcif"):
            mol = _mol_from_cif(p)
        else:
            raise InputError(
                f"cannot read 3D coordinates from {suffix!r}. "
                "Supported: .sdf .mol .pdb .cif"
            )

    if mol is None:
        raise InputError(f"could not parse {p}")
    if not mol.GetNumConformers():
        raise InputError(f"{p} has no coordinates")
    if not mol.GetConformer().Is3D():
        raise InputError(f"{p} holds 2D coordinates; embed it first")
    return mol


def _mol_from_cif(path: Path) -> Chem.Mol | None:
    """Read an mmCIF by converting it to PDB with gemmi first.

    RDKit has no mmCIF reader, and gemmi is already a dependency of the writer,
    so a round trip through PDB is the shortest correct path.
    """
    try:
        import gemmi
    except ImportError as exc:
        raise InputError(
            "reading mmCIF needs gemmi. Install it with: pip install gemmi"
        ) from exc
    try:
        structure = gemmi.read_structure(str(path))
    except Exception as exc:
        raise InputError(f"gemmi could not read {path}: {exc}") from exc

    structure.setup_entities()

    # Build from model 0 alone. Handing RDKit the whole multi-model PDB makes it
    # read every MODEL as a conformer, and the loop below would then add them a
    # second time — four conformers in, seven out.
    mol = Chem.MolFromPDBBlock(
        _single_model_pdb(structure, 0), sanitize=False, removeHs=False
    )
    if mol is None:
        return None

    # Going via PDB loses bond orders, but an mmCIF we wrote carries them in a
    # `_chem_comp_bond` loop. Without this the double bond of a ketone comes
    # back as a single bond and the molecule is quietly wrong.
    _apply_cif_bond_orders(mol, path)

    with rdkit_quiet():
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            raise InputError(f"{path} did not sanitize after reading: {exc}") from exc
        Chem.AssignStereochemistryFrom3D(mol)

    # Later models are conformers of the same molecule.
    for index in range(1, len(structure)):
        extra = Chem.MolFromPDBBlock(
            _single_model_pdb(structure, index), sanitize=False, removeHs=False
        )
        if extra is not None and extra.GetNumAtoms() == mol.GetNumAtoms():
            mol.AddConformer(Chem.Conformer(extra.GetConformer()), assignId=True)
    return mol


def _single_model_pdb(structure, index: int) -> str:
    """PDB text for exactly one model of a gemmi structure."""
    import gemmi

    one = gemmi.Structure()
    one.add_model(structure[index])
    one.setup_entities()
    return one.make_pdb_string()


_CIF_ORDER_TO_BOND = {
    "SING": Chem.BondType.SINGLE,
    "DOUB": Chem.BondType.DOUBLE,
    "TRIP": Chem.BondType.TRIPLE,
    "QUAD": Chem.BondType.QUADRUPLE,
}


def _apply_cif_bond_orders(mol: Chem.Mol, path: Path) -> None:
    """Restore bond orders from an mmCIF `_chem_comp_bond` loop, if present.

    Matches on atom name, which is why the writer assigns unique names. Silent
    no-op for an mmCIF that has no such loop — a coordinates-only file from
    elsewhere still reads, just without bond orders, exactly like a PDB.
    """
    import gemmi

    try:
        block = gemmi.cif.read(str(path)).sole_block()
        table = block.find(
            "_chem_comp_bond.",
            ["atom_id_1", "atom_id_2", "value_order", "?pdbx_aromatic_flag"],
        )
    except Exception:
        return
    if not len(table):
        return

    by_name: dict[str, int] = {}
    for atom in mol.GetAtoms():
        info = atom.GetMonomerInfo()
        if info is not None:
            by_name[info.GetName().strip()] = atom.GetIdx()

    for row in table:
        first, second = by_name.get(row.str(0)), by_name.get(row.str(1))
        if first is None or second is None:
            continue
        bond = mol.GetBondBetweenAtoms(first, second)
        if bond is None:
            continue
        aromatic = row.has(3) and row.str(3).upper().startswith("Y")
        if aromatic:
            bond.SetBondType(Chem.BondType.AROMATIC)
            bond.SetIsAromatic(True)
            mol.GetAtomWithIdx(first).SetIsAromatic(True)
            mol.GetAtomWithIdx(second).SetIsAromatic(True)
        else:
            bond.SetBondType(
                _CIF_ORDER_TO_BOND.get(row.str(2).upper(), Chem.BondType.SINGLE)
            )

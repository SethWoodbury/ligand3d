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
        if st in (Chem.BondStereo.STEREOANY,):
            unassigned.append((i, j))
        elif name in ("STEREOE", "STEREOZ"):
            assigned.append((i, j, name[-1]))
        elif name in ("STEREOCIS", "STEREOTRANS"):
            # Resolve to E/Z via the CIP labeller, which sets the bond's
            # _CIPCode property when it can.
            code = bond.GetPropsAsDict().get("_CIPCode")
            if code in ("E", "Z"):
                assigned.append((i, j, code))
            else:
                unassigned.append((i, j))
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
    """Sanitize, perceive stereo, and audit. Shared tail of every reader."""
    try:
        with rdkit_quiet():
            Chem.SanitizeMol(mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception as exc:  # RDKit raises bare exceptions here
        raise InputError(f"molecule failed sanitization: {exc}") from exc
    if mol.GetNumAtoms() == 0:
        raise InputError("molecule has no atoms")
    return Molecule(mol=mol, source=source, name=(name or "LIG")[:3].upper(),
                    stereo=audit_stereo(mol))


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
        + ". Specify it in the input, or pass --any-stereo to let RDKit pick one, "
        "or --enumerate-stereo to build every isomer."
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

"""Protonation states, and proving they survived.

Two separate jobs live here.

The first is assignment: given a molecule and a pH, work out which ionizable
groups are protonated. That is delegated to dimorphite-dl, which does it with
curated SMARTS and empirical pKa ranges.

The second is verification, and it is the one that earns its keep. A minimizer
does not know it is supposed to preserve your protonation state. Optimize
gabapentin's zwitterion in gas phase and the ammonium proton transfers back to
the carboxylate, leaving a neutral molecule that looks like a perfectly good
result. `assert_protonation_intact` catches that by checking which heavy atom
each hydrogen is actually bonded to after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from .errors import ProtonationError
from .molecule import Molecule, audit_stereo, rdkit_quiet

DEFAULT_PH = 7.4

_H_COVALENT_RADIUS = 0.31  # Angstrom
_BOND_TOLERANCE = 1.30
"""How far past the sum of covalent radii still counts as a bond.

Generous on purpose. The check exists to catch a proton that jumped to a
different heavy atom, which moves it by an angstrom or more; being strict here
buys nothing and starts rejecting long but perfectly real bonds.
"""


def _bond_cutoff(z_a: int, z_b: int) -> float:
    """Maximum distance at which two elements count as bonded.

    Derived from RDKit's periodic table rather than a hand-written dict, which
    silently defaulted heavier elements to a cutoff shorter than their actual
    X-H bond length: Se-H is 1.46 A and As-H 1.52 A, so `C[SeH]` and `[AsH3]`
    were reported as having lost a hydrogen that had never moved.
    """
    table = Chem.GetPeriodicTable()
    return (table.GetRcovalent(z_a) + table.GetRcovalent(z_b)) * _BOND_TOLERANCE


@dataclass(frozen=True)
class ProtonationState:
    """One protonation state produced for a molecule."""

    smiles: str
    charge: int
    n_charged_atoms: int = 0
    label: str = ""

    def describe(self) -> str:
        sign = f"{self.charge:+d}" if self.charge else "neutral"
        return f"{self.smiles} ({sign})"


def _rank_key(state: ProtonationState) -> tuple[int, int, str]:
    """Deterministic, chemically-motivated ordering for protonation states.

    dimorphite-dl returns states in an order that depends on Python's per-process
    string hash seed, so the same command can hand back a different molecule on a
    different run. Taking the first element of that list is not reproducible, and
    the failure is silent. This imposes our own order instead.

    In water at moderate pH the dominant species is normally the one with no net
    charge, and among those the one whose ionizable groups are actually ionized —
    which is why gabapentin at pH 7.4 should be the zwitterion rather than the
    neutral amino acid. Canonical SMILES breaks any remaining tie so the result
    is stable run to run.
    """
    return (abs(state.charge), -state.n_charged_atoms, state.smiles)


def enumerate_states(
    molecule: Molecule,
    ph: float = DEFAULT_PH,
    ph_window: float = 0.0,
    max_variants: int = 16,
) -> list[ProtonationState]:
    """Enumerate plausible protonation states at a pH using dimorphite-dl.

    A single pH can legitimately yield more than one state — gabapentin at 7.4
    gives both the zwitterion and the neutral-amine carboxylate — so this always
    returns a list and lets the caller decide.
    """
    try:
        from dimorphite_dl import protonate_smiles
    except ImportError as exc:
        raise ProtonationError(
            "pH-based protonation needs dimorphite-dl. "
            "Install it with: pip install 'ligand3d[protonation]'"
        ) from exc

    smiles = molecule.smiles
    with rdkit_quiet():  # dimorphite-dl builds transiently invalid intermediates
        try:
            variants = protonate_smiles(
                smiles, ph_min=ph - ph_window, ph_max=ph + ph_window
            )
        except Exception as exc:
            raise ProtonationError(
                f"dimorphite-dl could not protonate {smiles!r}: {exc}"
            ) from exc

    states: list[ProtonationState] = []
    seen: set[str] = set()
    for variant in variants[:max_variants]:
        with rdkit_quiet():
            m = Chem.MolFromSmiles(variant)
        if m is None:
            continue
        canonical = Chem.MolToSmiles(m)
        if canonical in seen:
            continue
        seen.add(canonical)
        states.append(
            ProtonationState(
                smiles=canonical,
                charge=Chem.GetFormalCharge(m),
                n_charged_atoms=sum(1 for a in m.GetAtoms() if a.GetFormalCharge() != 0),
            )
        )
    if not states:
        raise ProtonationError(
            f"dimorphite-dl returned no valid state for {smiles!r} at pH {ph}"
        )
    return sorted(states, key=_rank_key)


def apply_state(molecule: Molecule, state: ProtonationState) -> Molecule:
    """Build a new Molecule for one protonation state, keeping the name.

    Stereochemistry is re-audited rather than copied: dimorphite-dl round-trips
    through SMILES, and atom indices do not survive that.
    """
    with rdkit_quiet():
        mol = Chem.MolFromSmiles(state.smiles)
    if mol is None:
        raise ProtonationError(f"protonated SMILES did not parse: {state.smiles!r}")
    return Molecule(
        mol=mol,
        source=f"{molecule.source} [pH state {state.describe()}]",
        name=molecule.name,
        stereo=audit_stereo(mol),
        notes=list(molecule.notes),
    )


def protonate(
    molecule: Molecule,
    ph: float = DEFAULT_PH,
    enumerate_all: bool = False,
    ph_window: float = 0.0,
) -> list[Molecule]:
    """Return protonated variants of a molecule.

    With `enumerate_all=False` only the first (most likely) state is returned, so
    the common case still produces exactly one output file.
    """
    states = enumerate_states(molecule, ph=ph, ph_window=ph_window)
    chosen = states if enumerate_all else states[:1]
    out = [apply_state(molecule, s) for s in chosen]
    if len(states) > 1 and not enumerate_all:
        others = ", ".join(s.describe() for s in states[1:])
        out[0].notes.append(
            f"pH {ph:g} allows {len(states)} states; used {states[0].describe()}. "
            f"Also plausible: {others}. Pass --enumerate-states to build them all."
        )
    return out


def hydrogen_partners(mol: Chem.Mol, conf_id: int = -1) -> dict[int, int]:
    """Map each hydrogen index to the heavy atom it is actually bonded to in 3D.

    Geometry, not topology: this is what detects a proton that moved during
    minimization while the RDKit bond table still says otherwise.
    """
    conf = mol.GetConformer(conf_id)
    positions = np.asarray(conf.GetPositions(), dtype=float)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    if not heavy:
        return {}
    heavy_pos = positions[heavy]
    heavy_z = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in heavy]

    partners: dict[int, int] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        h = atom.GetIdx()
        distances = np.linalg.norm(heavy_pos - positions[h], axis=1)
        nearest = int(np.argmin(distances))
        cutoff = _bond_cutoff(1, heavy_z[nearest])
        partners[h] = heavy[nearest] if distances[nearest] <= cutoff else -1
    return partners


def topological_hydrogen_partners(mol: Chem.Mol) -> dict[int, int]:
    """Map each hydrogen to its bonded heavy atom according to the bond table.

    Hydrogens bonded only to other hydrogens (H2) have no heavy partner and are
    left out entirely, so they are not compared against a geometric map that
    cannot contain them either.
    """
    partners: dict[int, int] = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        heavy_neighbors = [
            n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() != 1
        ]
        if heavy_neighbors:
            partners[atom.GetIdx()] = heavy_neighbors[0]
    return partners


def assert_protonation_intact(
    mol_3d: Chem.Mol, conf_id: int = -1, context: str = "minimization"
) -> None:
    """Raise if any hydrogen ended up on a different heavy atom than it started on.

    This is the check that catches gas-phase zwitterion collapse. It compares the
    bond table (what we asked for) against the geometry (what we got).
    """
    expected = topological_hydrogen_partners(mol_3d)
    if not expected:
        return
    actual = hydrogen_partners(mol_3d, conf_id=conf_id)

    moved = []
    for h, want in expected.items():
        got = actual.get(h, -1)
        if got == want:
            continue
        want_label = _atom_label(mol_3d, want)
        got_label = "no heavy atom within bonding distance" if got < 0 else _atom_label(mol_3d, got)
        moved.append(f"  H{h} was bonded to {want_label} but is now on {got_label}")

    if moved:
        raise ProtonationError(
            f"protonation state changed during {context}:\n"
            + "\n".join(moved)
            + "\n\nThis usually means a proton transferred in gas phase. Add implicit "
            "solvation (--solvent water with --backend gfn2), or pass "
            "--allow-proton-transfer if the transfer is what you wanted to study."
        )


def _atom_label(mol: Chem.Mol, idx: int) -> str:
    if idx < 0:
        return "nothing"
    atom = mol.GetAtomWithIdx(idx)
    return f"{atom.GetSymbol()}{idx}"


def assert_connectivity_intact(
    mol_3d: Chem.Mol, conf_id: int = -1, context: str = "minimization"
) -> None:
    """Raise if any heavy-atom bond was stretched past breaking, or one formed.

    The hydrogen check catches a proton hopping between heteroatoms, which is
    the common failure. It does not catch a potential that breaks a C-C bond or
    closes a ring that was not there — semi-empirical methods and machine-learned
    potentials both work from positions alone and can do either. Bond orders are
    fixed in the RDKit molecule, so a broken bond leaves no trace in the output
    except two atoms implausibly far apart.
    """
    conf = mol_3d.GetConformer(conf_id)
    positions = np.asarray(conf.GetPositions(), dtype=float)

    broken: list[str] = []
    for bond in mol_3d.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        z_i = mol_3d.GetAtomWithIdx(i).GetAtomicNum()
        z_j = mol_3d.GetAtomWithIdx(j).GetAtomicNum()
        if z_i == 1 or z_j == 1:
            continue  # hydrogens are covered by the protonation check
        distance = float(np.linalg.norm(positions[i] - positions[j]))
        cutoff = _bond_cutoff(z_i, z_j)
        if distance > cutoff:
            broken.append(
                f"  {_atom_label(mol_3d, i)}-{_atom_label(mol_3d, j)} is "
                f"{distance:.2f} A apart (bonded, expected under {cutoff:.2f} A)"
            )

    if broken:
        raise ProtonationError(
            f"heavy-atom connectivity changed during {context}:\n"
            + "\n".join(broken)
            + "\n\nThe optimizer pulled a bond apart. Try a different backend, or "
            "check the input geometry is sane."
        )


def suggest_solvent(molecule: Molecule) -> str | None:
    """Recommend implicit solvation when gas phase would be actively misleading.

    Charged and zwitterionic species need a dielectric to stay put. Neutral
    molecules do not, and paying for solvation there would be waste.
    """
    if molecule.formal_charge != 0 or molecule.is_zwitterion:
        return "water"
    return None

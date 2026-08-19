"""Conformer generation, pruning, and clustering.

Two methods with very different cost profiles. The RDKit method embeds many
ETKDG conformers at once, minimizes each with a cheap force field, and clusters
what survives — seconds, and enough for most purposes. The CREST method runs
metadynamics at the GFN level and finds far more: about 145 unique gabapentin
conformers in five wall-clock minutes, versus a few dozen from ETKDG.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolAlign

from .embed import EmbedOptions, embed_multi
from .errors import BackendUnavailable, MinimizationError
from .molecule import Molecule, rdkit_quiet

KCAL_PER_HARTREE = 627.5094740631


@dataclass(frozen=True)
class ConformerOptions:
    n_confs: int = 1
    method: str = "rdkit"
    prune_rms: float = 0.5
    """Heavy-atom RMSD below which two conformers count as the same."""
    energy_window: float | None = None
    """Discard conformers more than this many kcal/mol above the best."""
    max_keep: int | None = None
    seed: int = 0xF00D
    n_threads: int = 1


def generate(molecule: Molecule, opts: ConformerOptions) -> Chem.Mol:
    """Generate conformers by the configured method."""
    if opts.method == "rdkit":
        return _generate_rdkit(molecule, opts)
    if opts.method == "crest":
        return _generate_crest(molecule, opts)
    raise BackendUnavailable(
        f"unknown conformer method {opts.method!r}. Available: rdkit, crest"
    )


def _generate_rdkit(molecule: Molecule, opts: ConformerOptions) -> Chem.Mol:
    """ETKDG multi-embed with RMS pruning done inside RDKit."""
    embed_opts = EmbedOptions(
        seed=opts.seed,
        prune_rms=opts.prune_rms if opts.n_confs > 1 else -1.0,
        n_threads=opts.n_threads,
    )
    # Oversample: pruning discards duplicates, so asking for exactly n_confs
    # reliably returns fewer than n_confs.
    target = opts.n_confs if opts.n_confs == 1 else min(int(opts.n_confs * 2), 2000)
    return embed_multi(molecule, n_confs=target, opts=embed_opts)


def _generate_crest(molecule: Molecule, opts: ConformerOptions) -> Chem.Mol:
    """Shell out to CREST for metadynamics-based conformer sampling."""
    from .config import find_crest_binary
    from .minimize.xtb import _xtb_env

    binary = find_crest_binary()
    if binary is None:
        raise BackendUnavailable(
            "crest executable not found. Set LIGAND3D_CREST_BIN, put crest on PATH, "
            "or add [binaries].crest to ~/.config/ligand3d/config.toml. "
            "Source: https://github.com/crest-lab/crest"
        )

    # CREST needs a starting geometry, so embed one first.
    start = embed_multi(molecule, n_confs=1, opts=EmbedOptions(seed=opts.seed))
    from .minimize import MinimizeJob, get_backend

    get_backend("mmff94").minimize(MinimizeJob(mol=start, conf_id=0))

    with tempfile.TemporaryDirectory(prefix="ligand3d-crest-") as tmp:
        work = Path(tmp)
        Chem.MolToXYZFile(start, str(work / "input.xyz"), confId=0)
        cmd = [
            binary, "input.xyz",
            "--gfn2",
            "--chrg", str(molecule.formal_charge),
            "--T", str(max(1, opts.n_threads)),
            "--niceprint",
        ]
        if opts.energy_window:
            cmd += ["--ewin", str(opts.energy_window)]
        try:
            proc = subprocess.run(
                cmd, cwd=work, capture_output=True, text=True,
                timeout=7200, env=_xtb_env(),
            )
        except subprocess.TimeoutExpired as exc:
            raise MinimizationError("crest timed out after 2 hours") from exc

        ensemble = work / "crest_conformers.xyz"
        if not ensemble.exists():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
            raise MinimizationError(
                "crest produced no conformer ensemble.\n" + "\n".join(tail)
            )
        return _load_ensemble(ensemble, start, max_keep=opts.max_keep or opts.n_confs)


def _load_ensemble(path: Path, template: Chem.Mol, max_keep: int) -> Chem.Mol:
    """Read a multi-structure XYZ into conformers on the template molecule."""
    from rdkit.Geometry import Point3D

    lines = path.read_text().splitlines()
    n_atoms = template.GetNumAtoms()
    out = Chem.Mol(template)
    out.RemoveAllConformers()

    cursor = 0
    kept = 0
    while cursor < len(lines) and kept < max_keep:
        try:
            count = int(lines[cursor].split()[0])
        except (ValueError, IndexError):
            break
        if count != n_atoms:
            raise MinimizationError(
                f"crest ensemble has {count} atoms but the molecule has {n_atoms}"
            )
        conf = Chem.Conformer(n_atoms)
        for i, line in enumerate(lines[cursor + 2 : cursor + 2 + n_atoms]):
            _, x, y, z = line.split()[:4]
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        conf.Set3D(True)
        out.AddConformer(conf, assignId=True)
        kept += 1
        cursor += n_atoms + 2

    if out.GetNumConformers() == 0:
        raise MinimizationError(f"no conformers parsed from {path}")
    return out


def heavy_atom_rmsd(mol: Chem.Mol, i: int, j: int) -> float:
    """Symmetry-aware heavy-atom RMSD between two conformers.

    GetBestRMS accounts for automorphisms, so a molecule with a freely rotating
    methyl or a symmetric ring is not reported as two different conformers just
    because atom labels permuted.
    """
    with rdkit_quiet():
        probe = Chem.RemoveHs(Chem.Mol(mol))
        return float(rdMolAlign.GetBestRMS(probe, probe, prbId=i, refId=j))


def prune(
    mol: Chem.Mol,
    energies: dict[int, float],
    rms_threshold: float = 0.5,
    energy_window: float | None = None,
    max_keep: int | None = None,
) -> list[int]:
    """Rank conformers by energy and drop duplicates and high-energy outliers.

    Returns the conformer ids to keep, best first.
    """
    # Only conformers that actually minimized are candidates. Ranking a failed
    # one as +inf still leaves it in the list whenever there is room, and the
    # caller then looks up an energy that was never recorded.
    conf_ids = [c.GetId() for c in mol.GetConformers() if c.GetId() in energies]
    ranked = sorted(conf_ids, key=lambda c: energies[c])
    if not ranked:
        return []

    best = energies[ranked[0]]

    if energy_window is not None:
        ranked = [c for c in ranked if energies[c] - best <= energy_window]

    kept: list[int] = []
    for cid in ranked:
        if max_keep is not None and len(kept) >= max_keep:
            break
        if rms_threshold > 0 and any(
            heavy_atom_rmsd(mol, cid, other) < rms_threshold for other in kept
        ):
            continue
        kept.append(cid)
    return kept


def subset(mol: Chem.Mol, conf_ids: list[int]) -> Chem.Mol:
    """Return a copy of `mol` carrying only the listed conformers, in order."""
    out = Chem.Mol(mol)
    conformers = {c.GetId(): Chem.Conformer(c) for c in mol.GetConformers()}
    out.RemoveAllConformers()
    for cid in conf_ids:
        if cid in conformers:
            out.AddConformer(conformers[cid], assignId=True)
    return out

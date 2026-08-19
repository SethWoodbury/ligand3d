"""2D to 3D: distance-geometry embedding, and proof that stereo survived.

Embedding is the cheap part. The part worth writing carefully is the check
afterwards: re-perceive stereochemistry from the 3D coordinates and confirm it
matches what the input specified. Without that, a silently mirrored stereocenter
looks exactly like a correct result.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import rdDistGeom

from .errors import EmbedError, StereoError
from .molecule import Molecule, StereoAudit, audit_stereo, rdkit_quiet

DEFAULT_SEED = 0xF00D


@dataclass(frozen=True)
class EmbedOptions:
    """Knobs for ETKDGv3. Defaults are chosen to be deterministic and robust."""

    seed: int = DEFAULT_SEED
    max_attempts: int = 0  # 0 lets RDKit decide, escalating internally
    use_random_coords_fallback: bool = True
    prune_rms: float = -1.0  # only meaningful for multi-conformer embedding
    n_threads: int = 1


def _params(opts: EmbedOptions) -> rdDistGeom.EmbedParameters:
    ps = rdDistGeom.ETKDGv3()
    ps.randomSeed = opts.seed
    # Both of these matter for the molecules this tool targets: small bridged
    # rings (quinuclidine) and macrocycles embed badly without them.
    ps.useSmallRingTorsions = True
    ps.useMacrocycleTorsions = True
    ps.enforceChirality = True
    ps.maxIterations = opts.max_attempts
    ps.numThreads = opts.n_threads
    if opts.prune_rms > 0:
        ps.pruneRmsThresh = opts.prune_rms
    return ps


def embed(molecule: Molecule, opts: EmbedOptions | None = None) -> Chem.Mol:
    """Generate one 3D conformer with hydrogens, preserving stereochemistry.

    Returns a new Mol with explicit Hs and exactly one conformer.
    """
    opts = opts or EmbedOptions()
    with rdkit_quiet():
        mol_h = Chem.AddHs(molecule.mol, addCoords=True)
        cid = rdDistGeom.EmbedMolecule(mol_h, _params(opts))
        if cid < 0 and opts.use_random_coords_fallback:
            # Random-coordinate start rescues cages and highly constrained rings
            # that the default ETKDG start cannot satisfy.
            ps = _params(opts)
            ps.useRandomCoords = True
            ps.maxIterations = max(opts.max_attempts, 200)
            cid = rdDistGeom.EmbedMolecule(mol_h, ps)
    if cid < 0:
        raise EmbedError(
            f"could not embed {molecule.smiles!r} in 3D after fallback. "
            "This usually means an over-constrained ring system or an impossible "
            "stereochemistry assignment."
        )
    verify_stereo(mol_h, molecule.stereo, context="embedding")
    return mol_h


def embed_multi(
    molecule: Molecule, n_confs: int, opts: EmbedOptions | None = None
) -> Chem.Mol:
    """Generate up to `n_confs` conformers in one pass.

    RMS pruning happens inside RDKit here, which is much cheaper than embedding
    everything and clustering afterwards.
    """
    opts = opts or EmbedOptions()
    if n_confs < 1:
        raise EmbedError(f"n_confs must be >= 1, got {n_confs}")
    with rdkit_quiet():
        mol_h = Chem.AddHs(molecule.mol, addCoords=True)
        cids = rdDistGeom.EmbedMultipleConfs(mol_h, numConfs=n_confs, params=_params(opts))
        if not len(cids):
            ps = _params(opts)
            ps.useRandomCoords = True
            ps.maxIterations = max(opts.max_attempts, 200)
            cids = rdDistGeom.EmbedMultipleConfs(mol_h, numConfs=n_confs, params=ps)
    if not len(cids):
        raise EmbedError(
            f"could not embed any conformer of {molecule.smiles!r}. "
            "Try a smaller --confs, or check the stereochemistry is satisfiable."
        )
    verify_stereo(mol_h, molecule.stereo, context="embedding", conf_id=int(cids[0]))
    return mol_h


def perceive_stereo_3d(mol_3d: Chem.Mol, conf_id: int = -1) -> StereoAudit:
    """Read stereochemistry back out of 3D coordinates."""
    work = Chem.Mol(mol_3d)
    if conf_id >= 0:
        # Isolate the conformer of interest; AssignStereochemistryFrom3D reads
        # whichever conformer it is handed, and defaults are easy to get wrong.
        #
        # GetConformer hands back a reference owned by the molecule, so it must
        # be copied before RemoveAllConformers frees it — otherwise AddConformer
        # reads freed memory and RDKit dies with a MemoryError.
        keep = Chem.Conformer(work.GetConformer(conf_id))
        work.RemoveAllConformers()
        work.AddConformer(keep, assignId=True)
    with rdkit_quiet():
        Chem.AssignStereochemistryFrom3D(work)
        flat = Chem.RemoveHs(work)
    return audit_stereo(flat)


def verify_stereo(
    mol_3d: Chem.Mol,
    expected: StereoAudit,
    context: str = "3D generation",
    conf_id: int = -1,
) -> None:
    """Raise if the 3D structure does not encode the expected stereochemistry.

    Only elements the input actually specified are checked. An input that left a
    centre undefined has nothing to violate.
    """
    if expected.n_defined == 0:
        return
    got = perceive_stereo_3d(mol_3d, conf_id=conf_id)

    want_centers = dict(expected.assigned_centers)
    got_centers = dict(got.assigned_centers)
    bad_centers = [
        (idx, code, got_centers.get(idx, "undefined"))
        for idx, code in want_centers.items()
        if got_centers.get(idx) != code
    ]

    want_bonds = {(a, b): lab for a, b, lab in expected.assigned_bonds}
    got_bonds = {(a, b): lab for a, b, lab in got.assigned_bonds}
    bad_bonds = [
        (key, lab, got_bonds.get(key, "undefined"))
        for key, lab in want_bonds.items()
        if got_bonds.get(key) != lab
    ]

    if not bad_centers and not bad_bonds:
        return

    lines = [f"stereochemistry changed during {context}:"]
    for idx, want, got_code in bad_centers:
        lines.append(f"  atom {idx}: expected {want}, got {got_code}")
    for (a, b), want, got_code in bad_bonds:
        lines.append(f"  bond {a}-{b}: expected {want}, got {got_code}")
    raise StereoError("\n".join(lines))

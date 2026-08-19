"""The pipeline: input to minimized 3D structure.

Everything the CLI does goes through `run`. Keeping the orchestration in one
readable function — rather than spread across the CLI — means the Python API and
the command line cannot drift apart, and the order of operations is auditable in
one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem

from . import conformers as conf_mod
from . import protonate as proton_mod
from .embed import EmbedOptions, verify_stereo
from .errors import Ligand3DError, MinimizationError
from .minimize import MinimizeJob, MinimizeResult, check_compatible, get_backend, parse_chain
from .molecule import (
    Molecule,
    enumerate_stereoisomers,
    has_real_stereo_ambiguity,
    largest_fragment,
    require_defined_stereo,
    require_single_fragment,
)
from .write import ConformerRecord, verify_pdb_roundtrip, write_pdb, write_sdf


@dataclass
class Settings:
    """Everything the pipeline needs to know. Mirrors the CLI flags."""

    backend: str = "mmff94"
    n_confs: int = 1
    conf_method: str = "rdkit"
    prune_rms: float = 0.5
    energy_window: float | None = None
    max_keep: int | None = None

    ph: float | None = None
    enumerate_states: bool = False

    stereo_mode: str = "require"  # require | any | enumerate
    largest_fragment: bool = False
    """Keep only the biggest disconnected component instead of refusing."""

    solvent: str | None = None
    auto_solvent: bool = True
    allow_charge_mismatch: bool = False
    allow_proton_transfer: bool = False

    max_steps: int = 500
    fmax: float = 0.05
    seed: int = 0xF00D
    n_threads: int = 1
    resname: str | None = None
    write_sdf: bool = True


@dataclass
class Outcome:
    """What one input molecule produced."""

    molecule: Molecule
    mol_3d: Chem.Mol
    records: list[ConformerRecord]
    pdb_path: Path | None = None
    sdf_path: Path | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def best_energy(self) -> float | None:
        energies = [r.energy for r in self.records if r.energy is not None]
        return min(energies) if energies else None


def expand_inputs(molecule: Molecule, settings: Settings) -> list[Molecule]:
    """Apply fragment, stereo, and protonation policy, yielding what to build."""
    molecule = _resolve_fragments(molecule, settings)

    if settings.stereo_mode == "require":
        require_defined_stereo(molecule)
        molecules = [molecule]
    elif settings.stereo_mode == "enumerate":
        molecules = (
            enumerate_stereoisomers(molecule)
            if has_real_stereo_ambiguity(molecule)
            else [molecule]
        )
    else:  # "any" — let ETKDG pick whatever satisfies the remaining constraints
        molecules = [molecule]

    if settings.ph is None:
        return molecules

    expanded: list[Molecule] = []
    for m in molecules:
        expanded.extend(
            proton_mod.protonate(m, ph=settings.ph, enumerate_all=settings.enumerate_states)
        )
    return expanded


def _resolve_fragments(molecule: Molecule, settings: Settings) -> Molecule:
    if settings.largest_fragment:
        return largest_fragment(molecule)
    require_single_fragment(molecule)
    return molecule


def resolve_solvent(molecule: Molecule, settings: Settings, supports_solvation: bool) -> str | None:
    """Decide whether to use implicit solvent for this molecule.

    Explicit beats automatic. Automatic turns solvation on exactly when gas phase
    would be misleading — a net charge or a zwitterion — and only if the backend
    can honour it.
    """
    if settings.solvent:
        return settings.solvent
    if not settings.auto_solvent or not supports_solvation:
        return None
    return proton_mod.suggest_solvent(molecule)


def build(molecule: Molecule, settings: Settings) -> Outcome:
    """Build and minimize one molecule. The core of the tool."""
    # `run` already screens inputs, but `build` is public and callers reach for
    # it directly. Re-checking here means the guarantee holds for the Python API
    # too, and it is idempotent for anything that came through `expand_inputs`.
    #
    # Screening comes first because it can annotate the molecule — discarding a
    # counterion adds a note the user needs to see.
    molecule = _resolve_fragments(molecule, settings)
    if settings.stereo_mode == "require":
        require_defined_stereo(molecule)

    notes: list[str] = list(molecule.notes)

    chain = parse_chain(settings.backend)
    backends = [get_backend(name) for name in chain]

    # Gate before doing any expensive work: a mismatch found after a five-minute
    # conformer search is a five-minute waste.
    solvation_capable = any(b.caps.supports_solvation for b in backends)
    solvent = resolve_solvent(molecule, settings, solvation_capable)
    for backend in backends:
        check_compatible(
            backend.caps,
            charge=molecule.formal_charge,
            elements=molecule.elements,
            solvent=solvent if backend.caps.supports_solvation else None,
            is_zwitterion=molecule.is_zwitterion,
            allow_charge_mismatch=settings.allow_charge_mismatch,
        )
        availability = backend.available()
        if not availability:
            raise Ligand3DError(
                f"backend {backend.caps.name!r} is not available: {availability.reason}"
                + (f"\n  {availability.hint}" if availability.hint else "")
            )
    if solvent and solvation_capable:
        notes.append(f"implicit solvation: ALPB {solvent}")

    conf_opts = conf_mod.ConformerOptions(
        n_confs=settings.n_confs,
        method=settings.conf_method,
        prune_rms=settings.prune_rms,
        energy_window=settings.energy_window,
        max_keep=settings.max_keep,
        seed=settings.seed,
        n_threads=settings.n_threads,
    )
    mol_3d = conf_mod.generate(molecule, conf_opts)
    notes.append(f"embedded {mol_3d.GetNumConformers()} conformer(s) via {settings.conf_method}")

    energies: dict[int, float] = {}
    results: dict[int, MinimizeResult] = {}
    failures: list[str] = []

    for conformer in list(mol_3d.GetConformers()):
        cid = conformer.GetId()
        last: MinimizeResult | None = None
        try:
            for backend in backends:
                job = MinimizeJob(
                    mol=mol_3d,
                    conf_id=cid,
                    charge=molecule.formal_charge,
                    max_steps=settings.max_steps,
                    fmax=settings.fmax,
                    solvent=solvent if backend.caps.supports_solvation else None,
                    n_threads=settings.n_threads,
                )
                last = backend.minimize(job)
        except Ligand3DError as exc:
            failures.append(f"conformer {cid}: {exc}")
            continue
        if last is not None:
            energies[cid] = last.energy
            results[cid] = last

    if not results:
        raise MinimizationError(
            "every conformer failed to minimize:\n  " + "\n  ".join(failures[:5])
        )
    if failures:
        notes.append(f"{len(failures)} conformer(s) failed to minimize and were dropped")

    keep = conf_mod.prune(
        mol_3d,
        energies,
        rms_threshold=settings.prune_rms,
        energy_window=settings.energy_window,
        max_keep=settings.max_keep or (settings.n_confs if settings.n_confs > 1 else 1),
    )
    final = conf_mod.subset(mol_3d, keep)
    if len(keep) < len(results):
        notes.append(f"kept {len(keep)} of {len(results)} minimized conformers after pruning")

    # Verify only after minimization: this is the point where a wrong answer
    # would otherwise escape. Every conformer is checked, not just the first —
    # an optimizer can invert one conformer and leave the rest alone, and
    # checking the default conformer would ship the other five as the
    # enantiomer of what was asked for.
    for conformer in final.GetConformers():
        cid = conformer.GetId()
        verify_stereo(final, molecule.stereo, context="minimization", conf_id=cid)
        if not settings.allow_proton_transfer:
            proton_mod.assert_protonation_intact(final, conf_id=cid)
            proton_mod.assert_connectivity_intact(final, conf_id=cid)

    records = [
        ConformerRecord(
            conf_id=new_conf.GetId(),
            energy=results[old_cid].energy,
            energy_unit=results[old_cid].energy_unit,
            energy_kind=results[old_cid].energy_kind,
            backend=results[old_cid].backend,
            converged=results[old_cid].converged,
        )
        for new_conf, old_cid in zip(final.GetConformers(), keep)
    ]

    return Outcome(molecule=molecule, mol_3d=final, records=records, notes=notes)


def run(
    molecule: Molecule,
    settings: Settings,
    output: Path | None = None,
) -> list[Outcome]:
    """Expand the input, build each variant, and write the files."""
    variants = expand_inputs(molecule, settings)
    outcomes: list[Outcome] = []
    multiple = len(variants) > 1

    for n, variant in enumerate(variants, start=1):
        outcome = build(variant, settings)
        if output is not None:
            path = _numbered(output, n) if multiple else output
            resname = settings.resname or variant.name
            outcome.pdb_path = write_pdb(
                path,
                outcome.mol_3d,
                records=outcome.records,
                resname=resname,
                smiles=variant.smiles,
                extra_remarks=[f"SOURCE {variant.source}"[:68]],
            )
            # Confirm what we just wrote is readable and its atom names are
            # unique, rather than trusting the writer.
            verify_pdb_roundtrip(outcome.pdb_path, outcome.mol_3d)
            if settings.write_sdf:
                outcome.sdf_path = write_sdf(
                    path.with_suffix(".sdf"),
                    outcome.mol_3d,
                    records=outcome.records,
                    name=resname,
                )
        outcomes.append(outcome)
    return outcomes


def _numbered(path: Path, n: int) -> Path:
    return path.with_name(f"{path.stem}_{n}{path.suffix}")

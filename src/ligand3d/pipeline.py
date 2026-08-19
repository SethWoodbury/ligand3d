"""The pipeline: input to minimized 3D structure.

Everything the CLI does goes through `run`. Keeping the orchestration in one
readable function — rather than spread across the CLI — means the Python API and
the command line cannot drift apart, and the order of operations is auditable in
one place.
"""

from __future__ import annotations

import time
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
from .write import (
    ConformerRecord,
    verify_cif_roundtrip,
    verify_pdb_roundtrip,
    write_cif,
    write_pdb,
    write_sdf,
    write_trajectory,
)


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

    formats: tuple[str, ...] = ("cif", "sdf")
    """Which representations to write.

    mmCIF is the default because it carries everything PDB does plus the bond
    orders PDB cannot, and it is what current structural tools prefer. The SDF
    rides along because it is the format RDKit itself round-trips perfectly.
    PDB is written on request; nothing in the params path needs it, since
    molfile_to_params reads the SDF directly.
    """
    trace: bool = False
    """Record energy at every optimizer step. Costs time; off by default."""
    trajectory: bool = False
    """Keep every step's coordinates and write them as a multi-model PDB."""

    params: bool = False
    """Also generate a Rosetta params file."""
    params_code: str | None = None
    allow_code_conflict: bool = False


@dataclass
class Outcome:
    """What one input molecule produced."""

    molecule: Molecule
    mol_3d: Chem.Mol
    records: list[ConformerRecord]
    cif_path: Path | None = None
    pdb_path: Path | None = None
    sdf_path: Path | None = None
    trajectory_path: Path | None = None
    params_result: object | None = None
    notes: list[str] = field(default_factory=list)
    trace: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    wall_seconds: float = 0.0

    @property
    def best_energy(self) -> float | None:
        energies = [r.energy for r in self.records if r.energy is not None]
        return min(energies) if energies else None

    @property
    def primary_path(self) -> Path | None:
        """The file a user most likely means when they say "the output"."""
        return self.cif_path or self.pdb_path or self.sdf_path

    def written(self) -> list[Path]:
        paths = [
            p for p in (self.cif_path, self.pdb_path, self.sdf_path, self.trajectory_path)
            if p is not None
        ]
        if self.params_result is not None:
            paths.extend(self.params_result.paths())
        return paths


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
        from .solvents import validate

        # Checked here rather than at the calculator, which rejects an unknown
        # name with a message that does not say what the valid ones are.
        return validate(settings.solvent)
    if not settings.auto_solvent or not supports_solvation:
        return None
    return proton_mod.suggest_solvent(molecule)


def build(molecule: Molecule, settings: Settings) -> Outcome:
    """Build and minimize one molecule. The core of the tool."""
    started = time.perf_counter()
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

    # Tracing every conformer would produce an unreadable pile of curves, so
    # only the first is traced; it is the one the graph shows.
    trace_conformer = mol_3d.GetConformers()[0].GetId() if mol_3d.GetNumConformers() else -1
    traces: dict[int, list] = {}
    frames: dict[int, list] = {}
    timings: list[tuple[str, float]] = []

    for conformer in list(mol_3d.GetConformers()):
        cid = conformer.GetId()
        last: MinimizeResult | None = None
        observe = cid == trace_conformer
        try:
            for stage, backend in enumerate(backends):
                job = MinimizeJob(
                    mol=mol_3d,
                    conf_id=cid,
                    charge=molecule.formal_charge,
                    max_steps=settings.max_steps,
                    fmax=settings.fmax,
                    solvent=solvent if backend.caps.supports_solvation else None,
                    n_threads=settings.n_threads,
                    trace=settings.trace and observe,
                    trajectory=settings.trajectory and observe,
                    stage=stage,
                )
                last = backend.minimize(job)
                timings.append((backend.caps.name, last.wall_seconds))
                if observe:
                    traces.setdefault(cid, []).extend(last.trace)
                    frames.setdefault(cid, []).extend(last.frames)
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

    per_backend: dict[str, float] = {}
    for name, seconds in timings:
        per_backend[name] = per_backend.get(name, 0.0) + seconds
    if per_backend:
        breakdown = ", ".join(f"{name} {sec:.2f}s" for name, sec in per_backend.items())
        notes.append(f"minimization time: {breakdown}")

    elapsed = time.perf_counter() - started
    notes.append(f"total time {elapsed:.2f}s")

    return Outcome(
        molecule=molecule,
        mol_3d=final,
        records=records,
        notes=notes,
        trace=traces.get(trace_conformer, []),
        frames=frames.get(trace_conformer, []),
        wall_seconds=elapsed,
    )


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
            base = _numbered(output, n) if multiple else output
            _write_outputs(outcome, variant, settings, base)
        outcomes.append(outcome)
    return outcomes


def _write_outputs(
    outcome: Outcome, variant: Molecule, settings: Settings, base: Path
) -> None:
    """Write every requested representation of one built molecule."""
    resname = settings.resname or variant.name
    remarks = [f"SOURCE {variant.source}"[:68]]
    formats = settings.formats

    if "cif" in formats:
        outcome.cif_path = write_cif(
            base.with_suffix(".cif"),
            outcome.mol_3d,
            records=outcome.records,
            resname=resname,
            smiles=variant.smiles,
            extra_remarks=remarks,
        )
        verify_cif_roundtrip(outcome.cif_path, outcome.mol_3d)

    if "pdb" in formats:
        outcome.pdb_path = write_pdb(
            base.with_suffix(".pdb"),
            outcome.mol_3d,
            records=outcome.records,
            resname=resname,
            smiles=variant.smiles,
            extra_remarks=remarks,
        )
        # Confirm what we just wrote is readable and its atom names are unique,
        # rather than trusting the writer.
        verify_pdb_roundtrip(outcome.pdb_path, outcome.mol_3d)

    if "sdf" in formats:
        outcome.sdf_path = write_sdf(
            base.with_suffix(".sdf"),
            outcome.mol_3d,
            records=outcome.records,
            name=resname,
        )

    if settings.trajectory and outcome.frames:
        outcome.trajectory_path = write_trajectory(
            base.with_name(f"{base.stem}_traj.pdb"),
            outcome.mol_3d,
            outcome.frames,
            resname=resname,
            energies=[step.energy for step in outcome.trace] or None,
            stage_labels=[f"stage {s.stage}: {s.backend}" for s in outcome.trace] or None,
        )
        outcome.notes.append(
            f"trajectory: {len(outcome.frames)} frames over "
            f"{len({step.stage for step in outcome.trace}) or 1} stage(s)"
        )

    if settings.params:
        from . import params as params_mod

        code = params_mod.normalize_code(settings.params_code or resname)
        outcome.params_result = params_mod.generate(
            outcome.mol_3d,
            code=code,
            out_dir=base.parent,
            conformers=outcome.mol_3d.GetNumConformers() > 1,
            allow_code_conflict=settings.allow_code_conflict,
        )
        # Only the commentary; the paths come from written().
        outcome.notes.extend(outcome.params_result.notes)


def _numbered(path: Path, n: int) -> Path:
    return path.with_name(f"{path.stem}_{n}{path.suffix}")

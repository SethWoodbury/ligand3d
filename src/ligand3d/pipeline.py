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
    """How many conformers to write out."""
    sample: int | None = None
    """How many to generate and search before keeping `n_confs`.

    None means pick from the molecule's flexibility. This is separate from
    `n_confs` on purpose: a single output structure should still be the best of
    a real search, not one arbitrary local minimum. Embedding and force-field
    minimizing a few dozen conformers costs well under a second, and the answer
    for a flexible molecule moves by many kcal/mol.
    """
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
    """Which representations to write. Empty means write nothing.

    An empty tuple is a dry run: everything is built, checked and reported, and
    no file is created. Useful for testing a molecule before committing a name.
    """
    """Which representations to write.

    mmCIF is the default because it carries everything PDB does plus the bond
    orders PDB cannot, and it is what current structural tools prefer. The SDF
    rides along because it is the format RDKit itself round-trips perfectly.
    PDB is written on request; nothing in the params path needs it, since
    molfile_to_params reads the SDF directly.
    """
    trace: bool = True
    """Record energy at every optimizer step, with the change from the previous.

    On by default: it is what makes a minimization inspectable rather than a
    black box. The cost is real but small, and the geometry is unaffected.
    """
    trajectory: bool = False
    trajectory_every: int = 10
    """Keep every Nth optimizer frame. 500 individual structures is not a
    trajectory anyone opens twice; every tenth still shows the path."""
    trace_all_conformers: bool = True
    """Give every kept conformer its own curve on the energy plot."""
    max_traced_conformers: int = 12
    """Above this, the plot stops being readable and the extra curves are
    dropped rather than drawn on top of each other."""
    split_conformers: bool = False
    """Write each conformer to its own file, ordered by energy, rather than
    one file with several models."""
    align_conformers: bool = True
    """Superimpose the kept conformers before writing. Without it they sit
    wherever the optimizer left them, and opening the file shows a scattered
    cloud rather than a comparison."""
    """Keep every step's coordinates and write them as a multi-model PDB."""

    params: bool = False
    """Also generate a Rosetta params file."""

    rfd_design_length: str | None = None
    """Length of the protein RFdiffusion4 should build around the ligand.

    `120`, or a range like `100-155` which is resampled per replicate. None
    writes the ligand alone, for pairing with a contig or condition spec that
    says what to build.
    """
    rfd_fix_coordinates: bool = True
    """Pin the ligand's pose in the annotated CIF.

    On, because a pose is the thing ligand3d exists to produce. Off hands the
    model the ligand's identity and lets it choose the geometry, which is right
    when the conformer is a guess rather than a measurement.
    """

    def effective_sample(self, molecule) -> int:
        """How many conformers to generate for the search.

        Scaled by rotatable-bond count, the usual proxy for how many distinct
        shapes a molecule has. A rigid cage needs a handful; a peptide-like
        chain needs hundreds. CREST does its own sampling, so it is left alone.
        """
        if self.conf_method != "rdkit":
            return self.n_confs
        if self.sample is not None:
            return max(self.sample, self.n_confs)

        from rdkit.Chem import rdMolDescriptors

        rotatable = rdMolDescriptors.CalcNumRotatableBonds(molecule.mol)
        if rotatable <= 2:
            budget = 20
        elif rotatable <= 5:
            budget = 60
        elif rotatable <= 9:
            budget = 150
        else:
            budget = 300
        return max(budget, self.n_confs)
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
    annotated_path: Path | None = None
    trajectory_path: Path | None = None
    conformer_paths: list[Path] = field(default_factory=list)
    """Per-conformer files, when split_conformers is on. Empty otherwise."""
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
            p for p in (self.cif_path, self.pdb_path, self.sdf_path,
                        self.annotated_path, self.trajectory_path)
            if p is not None
        ]
        paths.extend(self.conformer_paths)
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

    sample = settings.effective_sample(molecule)
    conf_opts = conf_mod.ConformerOptions(
        n_confs=sample,
        method=settings.conf_method,
        prune_rms=settings.prune_rms,
        energy_window=settings.energy_window,
        max_keep=settings.max_keep,
        seed=settings.seed,
        n_threads=settings.n_threads,
    )
    mol_3d = conf_mod.generate(molecule, conf_opts)
    generated = mol_3d.GetNumConformers()
    if sample > settings.n_confs:
        notes.append(
            f"searched {generated} conformer(s) via {settings.conf_method}, "
            f"keeping the best {settings.n_confs}"
        )
    else:
        notes.append(f"embedded {generated} conformer(s) via {settings.conf_method}")

    # Which conformers get a trace. One curve was the old default because a
    # hundred is unreadable; but with a handful, seeing them separate is the
    # point — that is how you notice one conformer taking a different path.
    # Capped so that a large search cannot produce an unreadable plot.
    _all_ids = [c.GetId() for c in mol_3d.GetConformers()]
    if not generated:
        traced_ids: list[int] = []
    elif settings.trace_all_conformers:
        traced_ids = _all_ids[: settings.max_traced_conformers]
    else:
        traced_ids = _all_ids[:1]
    traced_id = traced_ids[0] if traced_ids else -1
    traces: dict[int, list] = []
    traces = {}
    frames: dict[int, list] = {}
    timings: list[tuple[str, float]] = []
    failures: list[str] = []

    def run_stage(stage: int, backend, conf_ids: list[int]):
        """Minimize a set of conformers with one backend."""
        energies: dict[int, float] = {}
        results: dict[int, MinimizeResult] = {}
        traced_set = set(traced_ids)
        for cid in conf_ids:
            observe = cid in traced_set
            try:
                result = backend.minimize(
                    MinimizeJob(
                        mol=mol_3d,
                        conf_id=cid,
                        charge=molecule.formal_charge,
                        max_steps=settings.max_steps,
                        fmax=settings.fmax,
                        solvent=solvent if backend.caps.supports_solvation else None,
                        n_threads=settings.n_threads,
                        trace=settings.trace and observe,
                        # Trajectory frames stay on the primary conformer.
                        # Curves are cheap to overlay; a step-by-step file per
                        # conformer per method is a pile nobody asked for.
                        trajectory=settings.trajectory and cid == traced_id,
                        trajectory_every=settings.trajectory_every,
                        stage=stage,
                    )
                )
            except Ligand3DError as exc:
                failures.append(f"conformer {cid} on {backend.caps.name}: {exc}")
                continue
            timings.append((backend.caps.name, result.wall_seconds))
            if observe:
                traces.setdefault(cid, []).extend(result.trace)
                frames.setdefault(cid, []).extend(result.frames)
            energies[cid] = result.energy
            results[cid] = result
        return energies, results

    # --- search with the cheapest method, refine only the survivors ---------
    #
    # Running every backend on every sampled conformer is the obvious
    # implementation and the wrong one: searching broadly is only affordable
    # because the first method is cheap. A hundred GFN2 minimizations to find
    # the same handful of shapes MMFF94 would have found is minutes wasted.
    all_ids = [c.GetId() for c in mol_3d.GetConformers()]
    energies, results = run_stage(0, backends[0], all_ids)
    if not results:
        raise MinimizationError(
            "every conformer failed to minimize:\n  " + "\n  ".join(failures[:5])
        )

    if len(backends) > 1:
        survivors = conf_mod.prune(
            mol_3d,
            energies,
            rms_threshold=settings.prune_rms,
            energy_window=settings.energy_window,
            max_keep=settings.max_keep or settings.n_confs,
        )
        notes.append(
            f"{backends[0].caps.name} narrowed {len(results)} to {len(survivors)}; "
            f"refining with {', '.join(b.caps.name for b in backends[1:])}"
        )
        # Follow the survivors. The conformer traced through the search is
        # usually pruned away, and tracing an id that is no longer being
        # minimized silently produces an empty curve for every later stage.
        # The traced set has to follow the narrowing, or the refinement stage
        # records nothing: the conformers being traced were dropped, and the
        # ones being refined are not traced. Keeping traced_ids in step is the
        # whole reason stage 1 appears on the plot at all.
        if survivors:
            traced_ids = [c for c in traced_ids if c in survivors] or list(survivors)
            traced_ids = traced_ids[: settings.max_traced_conformers]
            if traced_id not in survivors:
                traced_id = traced_ids[0]
        for stage, backend in enumerate(backends[1:], start=1):
            energies, results = run_stage(stage, backend, survivors)
            if not results:
                raise MinimizationError(
                    f"every conformer failed on {backend.caps.name}:\n  "
                    + "\n  ".join(failures[:5])
                )
            survivors = list(results)

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

    # Superimpose before anything is written. The optimizer leaves each
    # conformer wherever it happened to converge, so a multi-model file opens
    # in PyMOL as a scattered cloud; aligned, the differences are the point.
    # Purely a rigid-body move: no coordinate is changed relative to the
    # others within a conformer, so energies and stereo are untouched.
    if settings.align_conformers and final.GetNumConformers() > 1:
        try:
            from rdkit.Chem import AllChem

            AllChem.AlignMolConformers(final)
            notes.append(f"aligned {final.GetNumConformers()} conformers for viewing")
        except Exception as exc:  # never lose a good structure to a cosmetic step
            notes.append(f"could not align conformers ({type(exc).__name__}); "
                         "they are written as optimized")
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
        # Worth stating outright: a neural potential silently falling back to
        # CPU looks exactly like one running on a GPU, only slower, and that is
        # the difference between a queued job being worth submitting or not.
        hardware = _mlff_device(per_backend)
        if hardware:
            notes.append(f"neural potential ran on {hardware}")

    elapsed = time.perf_counter() - started
    notes.append(f"total time {elapsed:.2f}s")

    return Outcome(
        molecule=molecule,
        mol_3d=final,
        records=records,
        notes=notes,
        trace=[step for steps in traces.values() for step in steps],
        frames=frames.get(traced_id, []) or next(iter(frames.values()), []),
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


def _mlff_device(per_backend: dict[str, float]) -> str | None:
    """Which processor the neural potentials used, if any ran.

    Returns None when the chain was entirely classical, so nothing is claimed
    about a run that never touched torch.
    """
    ran_mlff = False
    for name in per_backend:
        try:
            ran_mlff = ran_mlff or get_backend(name).caps.kind == "mlff"
        except Ligand3DError:
            continue
    if not ran_mlff:
        return None

    try:
        import torch

        if not torch.cuda.is_available():
            return "CPU"
        return f"GPU ({torch.cuda.get_device_name(0)})"
    except Exception:
        return None


def _conformer_slices(outcome: Outcome):
    """One single-conformer molecule per kept conformer, lowest energy first.

    The records are already energy-ordered by `prune`, so index 0 is the best
    one — which is what makes `_conf_0` a promise rather than a filename.
    """
    from rdkit import Chem

    slices = []
    for index, record in enumerate(outcome.records):
        single = Chem.Mol(outcome.mol_3d)
        keep = Chem.Conformer(outcome.mol_3d.GetConformer(record.conf_id))
        single.RemoveAllConformers()
        single.AddConformer(keep, assignId=True)
        slices.append((index, record, single))
    return slices


def _write_split_conformers(
    outcome: Outcome, variant: Molecule, settings: Settings, base: Path,
    resname: str, remarks: list[str],
) -> None:
    """Write each conformer to its own file, `<stem>_conf_<n>.<ext>`.

    Numbered by energy rather than by RDKit's conformer id, so `_conf_0` is
    always the lowest — the one you would open first. The ids are an internal
    detail and are not stable across a prune.
    """
    formats = settings.formats
    written: list[Path] = []

    for index, record, single in _conformer_slices(outcome):
        stem = f"{base.name}_conf_{index}"
        target = base.with_name(stem)
        one = [record.__class__(**{**record.__dict__,
                                   "conf_id": single.GetConformer().GetId()})]
        note = remarks + [f"CONFORMER {index} of {len(outcome.records)}"[:68]]

        if "cif" in formats:
            written.append(write_cif(target.with_suffix(".cif"), single,
                                     records=one, resname=resname,
                                     smiles=variant.smiles, extra_remarks=note))
        if "pdb" in formats:
            written.append(write_pdb(target.with_suffix(".pdb"), single,
                                     records=one, resname=resname,
                                     smiles=variant.smiles, extra_remarks=note))
        if "sdf" in formats:
            written.append(write_sdf(target.with_suffix(".sdf"), single,
                                     records=one, name=f"{resname}_conf_{index}"))

    outcome.conformer_paths = written
    if written:
        outcome.notes.append(
            f"wrote {len(outcome.records)} conformer(s) to separate files, "
            f"_conf_0 being the lowest energy"
        )


def _write_outputs(
    outcome: Outcome, variant: Molecule, settings: Settings, base: Path
) -> None:
    """Write every requested representation of one built molecule."""
    resname = settings.resname or variant.name
    remarks = [f"SOURCE {variant.source}"[:68]]
    formats = settings.formats

    # Separate files as well as the combined one, not instead of it: the
    # multi-model file is still the thing to open when comparing, and losing
    # it would be a surprise for anyone who has scripted against it.
    if settings.split_conformers and len(outcome.records) > 1:
        _write_split_conformers(outcome, variant, settings, base, resname, remarks)

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

    if "annotated" in formats:
        from .annotated import parse_segment, write_annotated_cif

        # One pose per file: the annotation says "use exactly this geometry",
        # and several models in one file would silently condition on the first.
        single = Chem.Mol(outcome.mol_3d)
        # Copy before removing: GetConformer returns a reference into the
        # molecule, so RemoveAllConformers frees it and using it afterwards is
        # a use-after-free that surfaces as MemoryError.
        keep = Chem.Conformer(single.GetConformer(outcome.records[0].conf_id))
        single.RemoveAllConformers()
        single.AddConformer(keep, assignId=True)

        outcome.annotated_path = write_annotated_cif(
            base.with_name(f"{base.stem}.annotated.cif"),
            single,
            records=outcome.records[:1],
            resname=resname,
            smiles=variant.smiles,
            fix_coordinates=settings.rfd_fix_coordinates,
            design=(
                parse_segment(settings.rfd_design_length)
                if settings.rfd_design_length
                else None
            ),
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

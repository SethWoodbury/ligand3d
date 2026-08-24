"""Command-line interface."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .errors import Ligand3DError
from .molecule import read_input
from .protonate import DEFAULT_PH

app = typer.Typer(
    name="ligand3d",
    help="Turn a 2D molecule into a minimized 3D structure.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


def _fail(exc: Exception) -> None:
    err_console.print(f"[bold red]error:[/bold red] {exc}")
    raise typer.Exit(code=1)


@app.command()
def build(
    molecule: str = typer.Argument(
        ..., help="SMILES string, or a path to a .mol/.sdf/.smi file."
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output .pdb path. Defaults to <name>.pdb."
    ),
    backend: str = typer.Option(
        "mmff94",
        "--backend",
        "-b",
        help="Minimization backend, or a comma-separated chain such as 'mmff94,gfn2'.",
    ),
    confs: int = typer.Option(1, "--confs", "-n", min=1, help="Number of conformers to keep."),
    sample: Optional[int] = typer.Option(
        None,
        "--sample",
        help="How many conformers to generate and search before keeping --confs. "
        "Defaults to 20-300 by rotatable-bond count. Use 1 to skip the search.",
    ),
    conf_method: str = typer.Option(
        "rdkit", "--conf-method", help="Conformer search method: rdkit or crest."
    ),
    ph: Optional[float] = typer.Option(
        None,
        "--ph",
        help="Assign the protonation state at this pH, e.g. --ph 7.4. "
        "Omit it to keep the structure exactly as drawn.",
    ),
    protonate: bool = typer.Option(
        False,
        "--protonate",
        help=f"Shorthand for --ph {DEFAULT_PH}.",
    ),
    enumerate_states: bool = typer.Option(
        False, "--enumerate-states", help="Write one file per plausible protonation state."
    ),
    stereo: str = typer.Option(
        "require",
        "--stereo",
        help="Undefined stereocenters: 'require' to error, 'any' to let RDKit pick, "
        "'enumerate' to build every isomer.",
    ),
    largest_fragment: bool = typer.Option(
        False,
        "--largest-fragment",
        help="For a salt or solvate, keep the biggest component instead of refusing.",
    ),
    solvent: Optional[str] = typer.Option(
        None,
        "--solvent",
        help="Implicit solvent (ALPB) for backends that support it. "
        "Run 'ligand3d solvents' for the list.",
    ),
    no_auto_solvent: bool = typer.Option(
        False,
        "--no-auto-solvent",
        help="Do not enable implicit solvation automatically for charged species.",
    ),
    allow_charge_mismatch: bool = typer.Option(
        False,
        "--allow-charge-mismatch",
        help="Permit a charged molecule on a backend with no charge channel.",
    ),
    allow_proton_transfer: bool = typer.Option(
        False,
        "--allow-proton-transfer",
        help="Do not fail if a proton moved to a different heavy atom during minimization.",
    ),
    prune_rms: float = typer.Option(
        0.5, "--prune-rms", help="Heavy-atom RMSD below which conformers are duplicates."
    ),
    energy_window: Optional[float] = typer.Option(
        None, "--energy-window", help="Discard conformers this far (kcal/mol) above the best."
    ),
    max_steps: int = typer.Option(500, "--max-steps", help="Optimizer step limit."),
    seed: int = typer.Option(0xF00D, "--seed", help="Random seed for embedding."),
    threads: int = typer.Option(1, "--threads", "-j", min=1, help="Threads for backends that use them."),
    resname: Optional[str] = typer.Option(
        None, "--resname", help="PDB residue name (3 characters). Defaults to LIG."
    ),
    formats: str = typer.Option(
        "cif,sdf",
        "--format",
        "-f",
        help="Comma-separated outputs: cif, pdb, sdf, annotated. mmCIF is the "
        "default because it carries the bond orders PDB cannot; 'annotated' adds "
        "an RFdiffusion4 annotated CIF alongside it.",
    ),
    no_sdf: bool = typer.Option(False, "--no-sdf", help="Do not write the .sdf sidecar."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Build, check and report, but write no files.",
    ),
    no_trace: bool = typer.Option(
        False,
        "--no-trace",
        help="Skip the per-step energy log. Tracing is on by default.",
    ),
    trajectory: bool = typer.Option(
        False,
        "--trajectory",
        help="Save the geometry at every step as <name>_traj.pdb, one MODEL per step.",
    ),
    make_params: bool = typer.Option(
        False, "--params", help="Also generate a Rosetta params file."
    ),
    annotated: bool = typer.Option(
        False,
        "--annotated",
        help="Also write an annotated mmCIF for RFdiffusion4-Proteina, with the "
        "ligand's pose and identity given to the model as context.",
    ),
    rfd_length: Optional[str] = typer.Option(
        None,
        "--rfd-length",
        help="Protein length RFdiffusion4 should build around the ligand: 120, "
        "or a range like 100-155 resampled per replicate.",
    ),
    rfd_free_pose: bool = typer.Option(
        False,
        "--rfd-free-pose",
        help="Let the model choose the ligand's pose instead of pinning the one "
        "ligand3d built. Its identity is still given.",
    ),
    params_code: Optional[str] = typer.Option(
        None,
        "--params-code",
        help="Three-character Rosetta ligand code. Defaults to the residue name.",
    ),
    allow_code_conflict: bool = typer.Option(
        False,
        "--allow-code-conflict",
        help="Generate params even if the code already exists in Rosetta.",
    ),
    container: bool = typer.Option(
        False,
        "--container",
        help="Run inside the Apptainer image that has this backend, on this "
        "machine. This is how to use eSEN, UMA and AllScAIP, which cannot "
        "share a virtualenv with MACE.",
    ),
    slurm: bool = typer.Option(
        False,
        "--slurm",
        help="Submit to SLURM on a GPU node instead of running here (IPD). "
        "Only worth it for the neural potentials.",
    ),
    slurm_wait: bool = typer.Option(
        False, "--slurm-wait", help="With --slurm, block until the job finishes."
    ),
    slurm_partition: str = typer.Option(
        "gpu", "--slurm-partition", help="SLURM partition: gpu, gpu-bf, cpu, cpu-bf."
    ),
    slurm_gpu: str = typer.Option(
        "small", "--slurm-gpu", help="GPU class to request: small, large, or h200."
    ),
    slurm_time: str = typer.Option(
        "01:00:00", "--slurm-time", help="Walltime. The scheduler rejects under 5 minutes."
    ),
    slurm_cpus: int = typer.Option(4, "--slurm-cpus", min=1, help="CPUs per task."),
    slurm_mem: str = typer.Option("16G", "--slurm-mem", help="Memory for the job."),
    slurm_dir: Optional[Path] = typer.Option(
        None,
        "--slurm-dir",
        help="Where to keep the job script and logs. Must be on shared storage "
        "(/home, /net, /mnt) — a compute node's /tmp is its own.",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print the output path."),
) -> None:
    """Build a minimized 3D structure. Writes mmCIF by default."""
    from .pipeline import Settings, run

    if stereo not in ("require", "any", "enumerate"):
        _fail(ValueError(f"--stereo must be require, any, or enumerate (got {stereo!r})"))

    if protonate and ph is not None and ph != DEFAULT_PH:
        _fail(ValueError(f"--protonate means --ph {DEFAULT_PH}; do not also pass --ph {ph}"))
    if protonate:
        ph = DEFAULT_PH
    if enumerate_states and ph is None:
        _fail(ValueError("--enumerate-states needs a pH; add --ph <value> or --protonate"))

    chosen = () if dry_run else _parse_formats(formats, want_sdf=not no_sdf)
    if annotated and not dry_run and "annotated" not in chosen:
        chosen = (*chosen, "annotated")

    settings = Settings(
        backend=backend,
        n_confs=confs,
        sample=sample,
        conf_method=conf_method,
        prune_rms=prune_rms,
        energy_window=energy_window,
        ph=ph,
        enumerate_states=enumerate_states,
        stereo_mode=stereo,
        largest_fragment=largest_fragment,
        solvent=solvent,
        auto_solvent=not no_auto_solvent,
        allow_charge_mismatch=allow_charge_mismatch,
        allow_proton_transfer=allow_proton_transfer,
        max_steps=max_steps,
        seed=seed,
        n_threads=threads,
        resname=resname,
        formats=chosen,
        trace=not no_trace,
        trajectory=trajectory,
        params=make_params,
        rfd_design_length=rfd_length,
        rfd_fix_coordinates=not rfd_free_pose,
        params_code=params_code,
        allow_code_conflict=allow_code_conflict,
    )

    try:
        mol = read_input(molecule)
        target = output or Path(f"{mol.name.lower()}.{chosen[0] if chosen else 'cif'}")
        # An explicit extension is a format request, so honour it — but not
        # under --dry-run, where the empty format list is the whole point and
        # the extension is only there to name a file that is never written.
        suffix = target.suffix.lower().lstrip(".")
        if not dry_run and suffix in ("cif", "pdb", "sdf") and suffix not in settings.formats:
            settings.formats = (suffix, *settings.formats)

        if slurm and container:
            _fail(ValueError("--container runs here and --slurm runs on a node; pick one"))
            return

        if slurm:
            _submit_to_slurm(
                mol, settings, target,
                partition=slurm_partition, gpu_class=slurm_gpu, walltime=slurm_time,
                cpus=slurm_cpus, memory=slurm_mem, workdir=slurm_dir,
                wait=slurm_wait, quiet=quiet,
            )
            return

        if container:
            _run_in_container(mol, settings, target, quiet=quiet)
            return

        outcomes = run(mol, settings, output=target)
    except Ligand3DError as exc:
        _fail(exc)
        return

    if quiet:
        for outcome in outcomes:
            if outcome.primary_path:
                console.print(str(outcome.primary_path))
        return

    console.print(f"[bold]{mol.formula}[/bold]  {mol.smiles}")
    _print_stereo(mol)
    for outcome in outcomes:
        console.print()
        if len(outcomes) > 1:
            console.print(f"  [cyan]{outcome.molecule.source}[/cyan]")
        for note in outcome.notes:
            console.print(f"  · {note}")
        energy = outcome.best_energy
        if energy is not None:
            record = outcome.records[0]
            unit, kind, backend_used = record.energy_unit, record.energy_kind, record.backend
            label = "total electronic energy" if kind == "total" else "strain energy"
            console.print(
                f"  · lowest {label} {energy:.4f} {unit} ({backend_used}), "
                f"{len(outcome.records)} conformer(s)"
            )
            if len(outcome.records) > 1:
                # Absolute totals are unreadable and only differences mean
                # anything, so show the spread rather than a column of them.
                spread = [
                    r.energy - energy for r in outcome.records if r.energy is not None
                ]
                console.print(
                    f"  · relative energies: "
                    + ", ".join(f"{d:+.2f}" for d in spread[:8])
                    + (" ..." if len(spread) > 8 else "")
                    + f" {unit}"
                )
        if outcome.trace:
            _print_trace(outcome.trace)
        _print_written(outcome.written())


def _print_stereo(mol) -> None:
    """Report stereochemistry: centres with CIP codes, bonds with E/Z."""
    from .molecule import describe_double_bonds

    audit = mol.stereo
    if audit.assigned_centers:
        detail = ", ".join(f"atom {i} = {code}" for i, code in audit.assigned_centers)
        console.print(f"  {len(audit.assigned_centers)} stereocenter(s): {detail}")
    else:
        console.print("  no stereocenters defined")

    for report in describe_double_bonds(mol):
        if report.cis_trans:
            gloss = f"[bold]{report.cip}[/bold] ({report.cis_trans})"
        else:
            gloss = (
                f"[bold]{report.cip}[/bold] "
                f"[dim](cis/trans does not apply: {report._why} alkene)[/dim]"
            )
        console.print(f"  double bond {report.begin}-{report.end}: {gloss}")

    if audit.unassigned_centers:
        from .molecule import (
            describe_resonance_centers,
            has_real_stereo_ambiguity,
            resonance_averaged_centers,
        )

        averaged = set(resonance_averaged_centers(mol))
        if averaged:
            console.print(f"  [dim]{escape(describe_resonance_centers(mol))}[/dim]")

        remaining = [i for i in audit.unassigned_centers if i not in averaged]
        if remaining and has_real_stereo_ambiguity(mol):
            atoms = ", ".join(str(i) for i in remaining)
            console.print(f"  [yellow]undefined stereocenter(s): atom {atoms}[/yellow]")
        elif remaining:
            console.print(
                f"  [dim]{len(remaining)} atom(s) look stereogenic but "
                f"are fixed by the ring system[/dim]"
            )


def _parse_formats(spec: str, want_sdf: bool = True) -> tuple[str, ...]:
    """Validate and normalize a --format list."""
    known = ("cif", "pdb", "sdf", "annotated")
    chosen = [f.strip().lower() for f in spec.split(",") if f.strip()]
    chosen = ["cif" if f == "mmcif" else f for f in chosen]
    bad = [f for f in chosen if f not in known]
    if bad:
        _fail(ValueError(f"unknown format(s) {', '.join(bad)}. Choose from: {', '.join(known)}"))
    if not want_sdf:
        chosen = [f for f in chosen if f != "sdf"]
    if not chosen:
        _fail(ValueError("no output formats left to write"))
    # Preserve order but drop repeats; the first entry is the primary output.
    return tuple(dict.fromkeys(chosen))


def _print_written(paths) -> None:
    """List output paths, one bare path per line.

    The path goes on its own line with nothing before it so a double-click
    selects the whole thing and it pastes straight into a shell. Prefixing each
    with "wrote" means every copy needs the mouse.
    """
    paths = list(paths)
    if not paths:
        console.print("  [dim]nothing written (no output formats selected)[/dim]")
        return
    console.print(f"  [green]wrote {len(paths)} file(s):[/green]")
    for path in paths:
        console.print(str(path), highlight=False, soft_wrap=True)


def _print_trace(trace: list, limit: int = 12) -> None:
    """Show the energy path, one block per method in the chain."""
    by_stage: dict[int, list] = {}
    for step in trace:
        by_stage.setdefault(step.stage, []).append(step)

    for stage in sorted(by_stage):
        steps = by_stage[stage]
        name = steps[0].backend
        unit = steps[0].energy_unit
        kind = "total" if steps[0].energy_kind == "total" else "strain"
        total = steps[-1].energy - steps[0].energy
        console.print(
            f"  [bold]{name}[/bold] — {len(steps)} step(s), {kind} energy in {unit}, "
            f"net {total:+.4f}"
        )
        shown = steps if len(steps) <= limit else [*steps[: limit // 2], None, *steps[-limit // 2 :]]
        for step in shown:
            if step is None:
                console.print(f"      [dim]… {len(steps) - limit} step(s) omitted …[/dim]")
                continue
            delta = "        —" if step.delta is None else f"{step.delta:+9.4f}"
            console.print(f"      step {step.step:4d}  E {step.energy:14.5f}  dE {delta}")


def _run_in_container(mol, settings, target: Path, quiet: bool = False) -> None:
    """Build inside the image that has this backend, on this machine."""
    from .container import ContainerError, image_for, run

    try:
        image = image_for(settings.backend)
        if not quiet:
            console.print(f"  running in [dim]{Path(image).name}[/dim]")
        result = run(
            mol, settings, target, image=image,
            echo=None if quiet else (lambda line: console.print(f"  {escape(line)}")),
        )
    except ContainerError as exc:
        _fail(exc)
        return

    if quiet:
        for path in result.get("outputs", []):
            console.print(str(path), highlight=False, soft_wrap=True)
        return

    written = result.get("outputs", [])
    if written:
        console.print(f"  [green]wrote {len(written)} file(s):[/green]")
        for path in written:
            console.print(str(path), highlight=False, soft_wrap=True)
    else:
        console.print("  [dim]nothing written (no output formats selected)[/dim]")


def _job_workdir(target: Path, requested: Optional[Path]) -> Path:
    """Where the script and logs go. Never reuses a live directory."""
    if requested is not None:
        return Path(requested).expanduser()
    base = target.expanduser().resolve().parent / f"{target.stem}.slurm"
    if not base.exists():
        return base
    n = 2
    while (candidate := base.with_name(f"{base.name}{n}")).exists():
        n += 1
    return candidate


def _submit_to_slurm(
    mol, settings, target: Path, *, partition: str, gpu_class: str, walltime: str,
    cpus: int, memory: str, workdir: Optional[Path], wait: bool, quiet: bool,
) -> None:
    """Queue the build instead of running it here."""
    from .slurm import (
        SlurmConfig, build_payload, container_for, job_name_for, needs_gpu, submit,
        wait_for,
    )

    target = target.expanduser().resolve()
    config = SlurmConfig(
        partition=partition, gpu_class=gpu_class, walltime=walltime,
        cpus=cpus, memory=memory, job_name=job_name_for(mol.name),
    )
    if not needs_gpu(settings.backend) and config.is_gpu:
        console.print(
            f"  [yellow]note[/yellow] {settings.backend} is not a neural potential, so a GPU "
            "will not help. It will still run, but queueing costs more than the calculation."
        )

    payload = build_payload(mol, settings, target)
    job = submit(payload, _job_workdir(target, workdir), config)

    if quiet:
        console.print(str(job.job_id))
    else:
        console.print(
            f"  submitted job [bold]{job.job_id}[/bold] to {config.partition}"
            + (f" on {config.gres()}" if config.is_gpu else "")
        )
        console.print(f"  container [dim]{Path(container_for(settings.backend)).name}[/dim]")
        console.print("  [green]log:[/green]")
        console.print(str(job.stdout), highlight=False, soft_wrap=True)
        console.print("  [green]output will be:[/green]")
        console.print(str(target), highlight=False, soft_wrap=True)

    if not wait:
        if not quiet:
            console.print(f"\n  check on it with [bold]ligand3d slurm --job {job.job_id}[/bold]")
        return

    if not quiet:
        console.print("\n  waiting…")
        state = wait_for(job.job_id, on_state=lambda s: console.print(f"  [dim]{s}[/dim]"))
    else:
        state = wait_for(job.job_id)

    # A job the scheduler cannot account for still counts if it left the file
    # behind, so the output decides rather than the state alone.
    wrote = target.exists()
    colour = "green" if wrote else "red"
    console.print(f"  job {job.job_id} finished: [{colour}]{state}[/{colour}]")
    if not wrote:
        console.print(f"  [dim]no output was written; see {job.stderr}[/dim]")
        raise typer.Exit(1)
    if state != "COMPLETED":
        console.print(
            f"  [yellow]note[/yellow] the scheduler reported {state}, "
            "but the job left a complete result"
        )
    console.print("  [green]wrote:[/green]")
    console.print(str(target), highlight=False, soft_wrap=True)


@app.command(hidden=True)
def slurm_run(payload: Path = typer.Argument(..., help="A job.json written by --slurm.")) -> None:
    """Run a submitted build. This is what executes inside the container."""
    import json

    from .sketch.session import collect_outputs, outcomes_to_result
    from .slurm import run_payload

    try:
        molecule, outcomes = run_payload(payload)
    except Ligand3DError as exc:
        _fail(exc)
        return

    # The submitter reads this back, so a queued build shows the same energy
    # plot and stereo summary as one that ran locally.
    outputs, trace = collect_outputs(outcomes)
    Path(payload).parent.joinpath("result.json").write_text(
        json.dumps(outcomes_to_result(molecule, outcomes, outputs, trace), indent=1)
    )

    for outcome in outcomes:
        for note in outcome.notes:
            console.print(f"  · {note}")
        if outcome.best_energy is not None:
            record = outcome.records[0]
            console.print(
                f"  · lowest energy {outcome.best_energy:.4f} {record.energy_unit} "
                f"({record.backend}), {len(outcome.records)} conformer(s)"
            )
        _print_written(outcome.written())


@app.command()
def slurm(
    job: Optional[int] = typer.Option(None, "--job", help="Report on one job id."),
) -> None:
    """Check whether GPU submission will work here, or report on a job.

    This is an IPD-specific convenience. Everything else in ligand3d runs
    without SLURM.
    """
    import shutil as _shutil

    from .slurm import (
        QUANTUM_CHEM_SIF, UMA_SIF, apptainer_available, job_state, slurm_available,
    )

    if job is not None:
        try:
            console.print(f"job {job}: [bold]{job_state(job)}[/bold]")
        except Ligand3DError as exc:
            _fail(exc)
        return

    ok = "[green]yes[/green]"
    no = "[red]no[/red]"
    console.print(f"  sbatch on PATH     {ok if slurm_available() else no}")
    console.print(f"  apptainer on PATH  {ok if apptainer_available() else no}")
    console.print("\n  containers (these supply the CUDA torch this venv lacks):")
    for label, sif in (("MACE, MACE-POLAR, AIMNet2", QUANTUM_CHEM_SIF),
                       ("eSEN, UMA, AllScAIP", UMA_SIF)):
        console.print(f"  {ok if Path(sif).exists() else no}  {label}")
        console.print(f"     [dim]{sif}[/dim]", highlight=False, soft_wrap=True)

    if not slurm_available():
        console.print(
            "\n  [yellow]No scheduler here.[/yellow] Submit from a login node, or drop "
            "--slurm and run locally."
        )
        return
    if _shutil.which("squeue"):
        console.print("\n  your queue:")
        subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-o", "%.10i %.9P %.24j %.8T %.10M %R"],
            check=False,
        )


@app.command()
def backends() -> None:
    """List minimization backends and what each one supports."""
    from .minimize import all_backends

    table = Table(title="ligand3d backends", header_style="bold")
    table.add_column("id")
    table.add_column("kind")
    table.add_column("charge", justify="center")
    table.add_column("solvent", justify="center")
    table.add_column("status")
    table.add_column("description")

    for backend in all_backends():
        caps = backend.caps
        try:
            availability = backend.available()
            status = (
                "[green]ready[/green]"
                if availability
                else f"[yellow]{escape(availability.reason)}[/yellow]"
            )
        except Exception as exc:  # a broken backend should still be listed
            status = f"[red]{exc}[/red]"
        table.add_row(
            caps.name,
            caps.kind,
            "yes" if caps.takes_charge else "no",
            "yes" if caps.supports_solvation else "no",
            status,
            caps.description,
        )
    console.print(table)


@app.command()
def doctor() -> None:
    """Report what is installed, what is missing, and where it looked."""
    from . import config as cfg
    from .minimize import all_backends

    console.print(f"[bold]ligand3d {__version__}[/bold]  (python {sys.version.split()[0]})")
    console.print(f"config: {cfg.CONFIG_PATH}" + ("" if cfg.CONFIG_PATH.exists() else "  [dim](not present)[/dim]"))
    console.print(f"cache:  {cfg.CACHE_DIR}")

    console.print("\n[bold]backends[/bold]")
    for backend in all_backends():
        try:
            availability = backend.available()
        except Exception as exc:
            console.print(f"  [red]✗[/red] {backend.caps.name}: {exc}")
            continue
        if availability:
            console.print(f"  [green]✓[/green] {backend.caps.name}")
        else:
            console.print(
                f"  [yellow]✗[/yellow] {backend.caps.name}: {escape(availability.reason)}"
            )
            if availability.hint:
                console.print(f"      [dim]{escape(availability.hint)}[/dim]")

    console.print("\n[bold]external binaries[/bold]")
    for resolution in (cfg.resolve_xtb(), cfg.resolve_crest()):
        if resolution.found:
            console.print(f"  [green]✓[/green] {resolution.key}: {resolution.path} [dim](via {resolution.via})[/dim]")
        else:
            console.print(f"  [yellow]✗[/yellow] {resolution.key}: not found")
            for attempt in resolution.tried:
                console.print(f"      [dim]tried {attempt}[/dim]")

    console.print("\n[bold]model weights[/bold]")
    found, missing = [], []
    for spec in cfg.MODELS:
        resolution = cfg.resolve_weights(spec.key)
        (found if resolution.found else missing).append((spec, resolution))

    # Group by the directory they were found in: on a cluster they all live in
    # one place, and repeating a 90-character path 19 times is unreadable.
    by_dir: dict[str, list] = {}
    for spec, resolution in found:
        by_dir.setdefault(str(resolution.path.parent.parent), []).append((spec, resolution))
    for directory, entries in by_dir.items():
        console.print(f"  in [cyan]{directory}[/cyan]:")
        for spec, resolution in entries:
            size = f"{spec.approx_mb} MB" if spec.approx_mb else ""
            charge = "charge-aware" if spec.takes_charge else ""
            console.print(
                f"    [green]✓[/green] {spec.key:<16} "
                f"[dim]{resolution.path.name}  {size}  {charge}[/dim]"
            )
    if missing:
        console.print(
            f"  [yellow]✗[/yellow] not found: "
            f"[dim]{', '.join(spec.key for spec, _ in missing)}[/dim]"
        )
    if not found:
        console.print(
            "  [dim]On this cluster the checkpoints live in "
            "/net/databases/huggingface/mlFF_models/, which is probed automatically.[/dim]"
        )

    if cfg.UNSUPPORTED_MODELS:
        console.print("\n[bold]known but not loadable[/bold]")
        for key, why in cfg.UNSUPPORTED_MODELS.items():
            console.print(f"  [yellow]-[/yellow] {key}: [dim]{escape(why)}[/dim]")

    console.print("\n[bold]optional python packages[/bold]")
    for module, why in (
        ("tblite", "GFN1/GFN2-xTB"),
        ("ase", "optimizers for all non-RDKit backends"),
        ("dimorphite_dl", "pH-based protonation"),
        ("torch", "machine-learned potentials"),
        ("mace", "MACE potentials"),
        ("graph_longrange", "MACE-POLAR long-range electrostatics"),
        ("aimnet", "AIMNet2 potential"),
        ("py2opsin", "offline IUPAC name lookup for 'fetch'"),
    ):
        import importlib.util

        present = importlib.util.find_spec(module) is not None
        mark = "[green]✓[/green]" if present else "[yellow]✗[/yellow]"
        console.print(f"  {mark} {module} [dim]— {why}[/dim]")

    console.print("\n[bold]molecule lookup[/bold]")
    from .resolve import opsin_available

    if opsin_available():
        console.print(
            "  [green]✓[/green] systematic names, offline [dim]— OPSIN and java are both here[/dim]"
        )
    else:
        console.print(
            "  [yellow]✗[/yellow] systematic names [dim]— needs 'pip install ligand3d[names]' "
            "and a java on PATH[/dim]"
        )
    console.print(
        "  [dim]trivial and trade names are looked up in PubChem, which needs "
        "the network.[/dim]"
    )

    # Inside the launcher's core image the neural potentials are genuinely
    # absent, but they are not out of reach — the launcher sends those backends
    # to another image. Without this, `doctor` tells someone to pip install
    # torch into a read-only container to get something that already works.
    if os.environ.get("LIGAND3D_LAUNCHER"):
        console.print(
            "\n[bold]launcher[/bold]\n"
            "  [green]✓[/green] running through the ligand3d launcher\n"
            "  [dim]The neural potentials above show as unavailable because this "
            "image does not carry torch. Ask for one anyway — the launcher runs "
            "MACE and AIMNet2 in the quantum_chem image and eSEN/UMA/AllScAIP in "
            "the uma image. Ignore the pip hints; they do not apply here.[/dim]"
        )

    # Only mentioned where it would work, so this stays quiet off the cluster.
    from .sketch.session import slurm_status

    if slurm_status()["available"]:
        console.print(
            "\n[bold]gpu offload[/bold]\n"
            "  [green]✓[/green] this host can submit to SLURM "
            "[dim]— add --slurm to a build, or run 'ligand3d slurm' for detail[/dim]"
        )


@app.command()
def config(
    init: bool = typer.Option(False, "--init", help="Write a starter config file."),
    show: bool = typer.Option(False, "--show", help="Print the active config."),
) -> None:
    """Inspect or create the configuration file."""
    from . import config as cfg

    if init:
        if cfg.CONFIG_PATH.exists():
            console.print(f"[yellow]{cfg.CONFIG_PATH} already exists; not overwriting.[/yellow]")
            raise typer.Exit(code=1)
        path = cfg.write_default_config()
        console.print(f"[green]wrote[/green] {path}")
        return
    if show or True:
        if not cfg.CONFIG_PATH.exists():
            console.print(f"[dim]no config at {cfg.CONFIG_PATH}; run 'ligand3d config --init'[/dim]")
            return
        console.print(cfg.CONFIG_PATH.read_text())


@app.command()
def sketch(
    port: int = typer.Option(0, "--port", help="Port to serve on. 0 picks a free one."),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Print the URL instead of opening a browser."
    ),
    directory: Optional[Path] = typer.Option(
        None, "--directory", "-d", help="Where built structures go. Defaults to the cwd."
    ),
    backend: str = typer.Option("mmff94", "--backend", "-b", help="Backend preselected in the page."),
    threads: int = typer.Option(4, "--threads", "-j", min=1, help="Threads preselected in the page."),
) -> None:
    """Draw molecules in the browser and build them, one after another.

    The server stays up: draw, build, read the log, clear, draw the next one.
    Every option in the page maps to a `ligand3d build` flag.
    """
    from .sketch.server import serve
    from .sketch.session import next_filename

    target = str((directory or Path.cwd()).expanduser().resolve())
    defaults = {
        "directory": target,
        "filename": next_filename(target),
        "backend": backend,
        "threads": threads,
    }
    try:
        serve(port=port, open_browser=not no_browser, defaults=defaults)
    except Ligand3DError as exc:
        _fail(exc)


@app.command()
def models(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Include training data, notes, and weight paths."
    ),
    available_only: bool = typer.Option(
        False, "--available", help="Only methods that can run here."
    ),
) -> None:
    """Describe every minimization method: cost, capabilities, and weights."""
    from .catalog import summarize

    report = summarize()
    methods = [m for m in report.methods if m.ready or not available_only]

    table = Table(title="ligand3d methods", header_style="bold", expand=True)
    table.add_column("id", no_wrap=True)
    table.add_column("family", no_wrap=True)
    table.add_column("charge", justify="center", no_wrap=True)
    table.add_column("spin", justify="center", no_wrap=True)
    table.add_column("solv", justify="center", no_wrap=True)
    table.add_column("speed", justify="right", no_wrap=True)
    table.add_column("memory", justify="right", no_wrap=True)
    table.add_column("status", overflow="fold")

    for method in methods:
        # escape() goes around the *reason*, which is arbitrary text that may
        # contain brackets; the colour tags around it are ours and must survive.
        status = (
            "[green]ready[/green]" if method.ready
            else f"[yellow]{escape(method.reason)}[/yellow]"
        )
        charge = (
            "[green]explicit[/green]" if method.charge == "explicit"
            else f"[dim]{method.charge}[/dim]"
        )
        table.add_row(
            method.id, method.family, charge,
            method.spin, method.solvent,
            method.speed if method.measured else f"[dim]{method.speed}[/dim]",
            method.memory, status,
        )
    console.print(table)
    console.print(
        "\n[dim]charge: explicit = the model consumes total charge; implicit = it only "
        "sees atoms and positions, so it cannot tell a carboxylate from a neutral acid "
        "that lost a proton.[/dim]"
    )
    console.print(
        "[dim]speed is a full minimization of gabapentin (29 atoms) on CPU with 8 threads. "
        "Dimmed values are estimates for models that cannot run in this environment; "
        "everything else was timed here.[/dim]"
    )
    slow = [m for m in methods if m.load_seconds >= 5]
    if slow:
        console.print(
            "[dim]first-call load cost, separate from the minimization: "
            + ", ".join(f"{m.id} {m.load_seconds:.0f}s" for m in slow)
            + "[/dim]"
        )

    if verbose:
        for method in methods:
            console.print(f"\n[bold]{method.id}[/bold]  {method.description}")
            for label, value in (
                ("family", method.family),
                ("trained on", method.training),
                ("reproduces", method.reference),
                ("error", method.error),
                ("accuracy", method.accuracy),
                ("elements", method.elements),
                ("notes", method.notes),
                ("upstream", method.repo),
                ("weights", method.weights_file),
                ("resolved", method.weights_path or "not found"),
                ("aliases", ", ".join(method.aliases)),
            ):
                if value:
                    console.print(f"    {label:12s} {escape(str(value))}")
            if not method.ready and method.hint:
                console.print(f"    [dim]{escape(method.hint)}[/dim]")

        console.print(
            "\n[dim]'error' is a literature figure against the reference above it, not a "
            "measurement made here, and it is always in-domain: the error on held-out "
            "data from the same distribution as the training set. A molecule unlike "
            "anything in that distribution can be far worse with no warning.[/dim]"
        )

    if report.weight_roots:
        console.print("\n[bold]weights read from[/bold]")
        for root in report.weight_roots:
            console.print(f"  {root}")
    if report.unsupported:
        console.print("\n[bold]known but not loadable[/bold]")
        for key, why in report.unsupported.items():
            console.print(f"  [yellow]-[/yellow] {key}: [dim]{escape(why)}[/dim]")


@app.command()
def fetch(
    query: Optional[str] = typer.Argument(
        None,
        help="A name, SMILES, InChI, or PubChem CID. Prefix with smiles:, inchi:, "
        "name:, or cid: to force one route.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write the 2D structure to a .sdf or .mol file."
    ),
    smiles_only: bool = typer.Option(
        False, "--smiles", help="Print only the SMILES, for piping into build."
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Do not query PubChem; names must be systematic."
    ),
    kind: str = typer.Option(
        "auto",
        "--type",
        "-t",
        help="How to read the query: auto, smiles, inchi, name, cid, peptide, "
        "dna, rna. Sequences are never auto-detected.",
    ),
    ph: Optional[float] = typer.Option(
        None,
        "--ph",
        help="For a sequence, the pH the build will use. Only reported here; "
        "nothing is protonated by fetch.",
    ),
    templates: bool = typer.Option(
        False, "--templates", help="List the built-in scaffolds and exit."
    ),
    residues: bool = typer.Option(
        False, "--residues", help="List the residue codes for --type peptide/dna/rna."
    ),
    timeout: float = typer.Option(12.0, "--timeout", help="Seconds to wait on PubChem."),
) -> None:
    """Look up a molecule by name and get a structure to start from.

    Systematic names are derived offline by OPSIN; trivial and trade names are
    looked up in PubChem, which needs the network.

        ligand3d fetch "3-Cyano-7-ethoxycoumarin"
        ligand3d build "$(ligand3d fetch aspirin --smiles)" -o aspirin.cif
    """
    from .resolve import (
        KINDS, ResolveError, opsin_available, resolve, template, template_list,
    )

    if kind not in KINDS:
        _fail(ValueError(f"--type must be one of: {', '.join(KINDS)}"))
        return

    if residues:
        from .biopolymer import available_residues, residue_name

        kinds = [kind] if kind in ("peptide", "dna", "rna") else ["peptide", "dna", "rna"]
        for one in kinds:
            table = Table(title=f"{one} residue codes", header_style="bold")
            table.add_column("code")
            table.add_column("residue")
            for code in available_residues(one):
                table.add_row(code, residue_name(code, one))
            console.print(table)
        console.print(
            "\n[dim]One-letter codes go straight in the sequence; longer codes go "
            "in parentheses, as in GS(KCX)PL.[/dim]"
        )
        return

    if templates:
        table = Table(title="built-in scaffolds", header_style="bold")
        table.add_column("name")
        table.add_column("SMILES")
        table.add_column("what it is")
        for entry in template_list():
            table.add_row(entry["name"], entry["smiles"], entry["note"])
        console.print(table)
        console.print("\n[dim]ligand3d fetch <name> loads one of these too.[/dim]")
        return

    if not query:
        _fail(ValueError("give a name, SMILES, InChI, CID, or sequence — or use --templates"))
        return

    try:
        known = {entry["name"] for entry in template_list()}
        use_template = kind == "auto" and query.strip().lower() in known
        found = template(query) if use_template else resolve(
            query, allow_network=not offline, timeout=timeout, kind=kind, ph=ph
        )
    except ResolveError as exc:
        _fail(exc)
        return

    if smiles_only:
        console.print(found.smiles, highlight=False, soft_wrap=True)
        return

    console.print(f"[bold]{found.formula}[/bold]  {escape(found.smiles)}")
    if found.name and found.name.lower() != found.query.strip().lower():
        console.print(f"  · known as [cyan]{escape(found.name)}[/cyan]")
    console.print(f"  · {found.provenance}")
    for note in found.notes:
        console.print(f"  · {note}")
    if found.url:
        console.print(f"  · {found.url}", highlight=False, soft_wrap=True)
    if found.source == "opsin":
        console.print(
            "  [dim]derived from the name itself, so it is what the name says — "
            "not a database record of a real compound.[/dim]"
        )
    elif found.source == "pubchem":
        console.print(
            "  [dim]a database match on the name you typed. Check the name above "
            "is the compound you meant.[/dim]"
        )
    if offline and not opsin_available():
        console.print(
            "  [yellow]note[/yellow] --offline with no OPSIN leaves only SMILES and InChI."
        )

    if output is not None:
        from .molecule import from_smiles
        from .write import write_2d

        path = write_2d(from_smiles(found.smiles, name=found.name or "LIG"), output)
        console.print("  [green]wrote:[/green]")
        console.print(str(path), highlight=False, soft_wrap=True)
    else:
        console.print(
            f"\n  [dim]build it with[/dim] ligand3d build {escape(found.smiles)!r}"
        )


@app.command()
def solvents() -> None:
    """List the implicit solvents ALPB is parameterized for."""
    from .solvents import NOT_PARAMETERIZED, SOLVENTS

    table = Table(title="ALPB implicit solvents", header_style="bold")
    table.add_column("name")
    table.add_column("aliases")
    table.add_column("dielectric", justify="right")
    table.add_column("note")
    for entry in SOLVENTS:
        table.add_row(
            entry.name,
            ", ".join(entry.aliases),
            f"{entry.dielectric:.1f}" if entry.dielectric else "",
            entry.note,
        )
    console.print(table)
    console.print(
        "\n[dim]Available on gfn1, gfn2, and gfnff. The machine-learned potentials "
        "have no implicit solvent model at all.[/dim]"
    )
    console.print(
        "[dim]Not parameterized (nearest stand-in shown): "
        + ", ".join(f"{k} -> {v}" for k, v in list(NOT_PARAMETERIZED.items())[:6])
        + ", ...[/dim]"
    )


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


# --------------------------------------------------------------------------
# Individual pipeline steps, for scripting one stage at a time
# --------------------------------------------------------------------------


@app.command()
def stereo(
    molecule: str = typer.Argument(..., help="SMILES, or a .mol/.sdf/.pdb/.cif file."),
) -> None:
    """Report the stereochemistry of a molecule and exit.

    Stereocenters with CIP codes, double bonds with E/Z, and cis/trans where
    that term actually applies.
    """
    try:
        mol = read_input(molecule)
    except Ligand3DError as exc:
        _fail(exc)
        return
    console.print(f"[bold]{mol.formula}[/bold]  {mol.smiles}")
    _print_stereo(mol)


@app.command()
def embed(
    molecule: str = typer.Argument(..., help="SMILES, or a .mol/.sdf file."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path."),
    confs: int = typer.Option(1, "--confs", "-n", min=1, help="How many conformers."),
    formats: str = typer.Option("cif,sdf", "--format", "-f", help="cif, pdb, sdf."),
    stereo_mode: str = typer.Option("require", "--stereo", help="require | any | enumerate"),
    seed: int = typer.Option(0xF00D, "--seed", help="Random seed."),
    resname: Optional[str] = typer.Option(None, "--resname", help="PDB residue name."),
) -> None:
    """2D to 3D only: embed coordinates and write them, with no minimization.

    Useful when you want the raw ETKDG geometry, or want to minimize separately.
    """
    from .pipeline import Settings, run

    chosen = _parse_formats(formats)
    settings = Settings(
        backend="mmff94", n_confs=confs, stereo_mode=stereo_mode, seed=seed,
        resname=resname, formats=chosen, max_steps=0,
    )
    try:
        mol = read_input(molecule)
        target = output or Path(f"{mol.name.lower()}.{chosen[0] if chosen else 'cif'}")
        outcomes = run(mol, settings, output=target)
    except Ligand3DError as exc:
        _fail(exc)
        return
    console.print(f"[bold]{mol.formula}[/bold]  {mol.smiles}")
    _print_stereo(mol)
    for outcome in outcomes:
        console.print(f"  embedded {outcome.mol_3d.GetNumConformers()} conformer(s)")
        _print_written(outcome.written())


@app.command()
def minimize(
    structure: Path = typer.Argument(..., help="A 3D .sdf/.mol/.pdb/.cif file."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path."),
    backend: str = typer.Option("mmff94", "--backend", "-b", help="Backend or chain."),
    formats: str = typer.Option("cif,sdf", "--format", "-f", help="cif, pdb, sdf."),
    solvent: Optional[str] = typer.Option(None, "--solvent", help="Implicit solvent."),
    max_steps: int = typer.Option(500, "--max-steps", help="Optimizer step limit."),
    no_trace: bool = typer.Option(False, "--no-trace", help="Skip the per-step energy log."),
    trajectory: bool = typer.Option(False, "--trajectory", help="Save the step geometries."),
    threads: int = typer.Option(1, "--threads", "-j", min=1, help="Threads."),
    resname: Optional[str] = typer.Option(None, "--resname", help="PDB residue name."),
) -> None:
    """Minimize a structure that already has 3D coordinates.

    Every conformer in the input file is minimized.
    """
    import time

    from .minimize import MinimizeJob, get_backend, parse_chain
    from .molecule import read_3d
    from .write import ConformerRecord

    chosen = _parse_formats(formats)
    try:
        mol = read_3d(structure)
        backends = [get_backend(name) for name in parse_chain(backend)]
    except Ligand3DError as exc:
        _fail(exc)
        return

    from rdkit import Chem

    charge = Chem.GetFormalCharge(mol)
    console.print(
        f"[bold]{structure}[/bold]: {mol.GetNumAtoms()} atoms, "
        f"{mol.GetNumConformers()} conformer(s), charge {charge:+d}"
    )

    started = time.perf_counter()
    records, all_trace, frames = [], [], []
    try:
        for index, conformer in enumerate(mol.GetConformers()):
            cid = conformer.GetId()
            result = None
            for stage, chosen_backend in enumerate(backends):
                job = MinimizeJob(
                    mol=mol, conf_id=cid, charge=charge, max_steps=max_steps,
                    solvent=solvent, n_threads=threads,
                    trace=(not no_trace) and index == 0,
                    trajectory=trajectory and index == 0,
                    stage=stage,
                )
                result = chosen_backend.minimize(job)
                if index == 0:
                    all_trace.extend(result.trace)
                    frames.extend(result.frames)
            if result is not None:
                records.append(
                    ConformerRecord(
                        conf_id=cid, energy=result.energy, energy_unit=result.energy_unit,
                        energy_kind=result.energy_kind, backend=result.backend,
                        converged=result.converged,
                    )
                )
    except Ligand3DError as exc:
        _fail(exc)
        return

    if all_trace:
        _print_trace(all_trace)
    if records:
        best = min(r.energy for r in records)
        kind = "total electronic" if records[0].energy_kind == "total" else "strain"
        console.print(
            f"  lowest {kind} energy {best:.4f} {records[0].energy_unit} "
            f"({records[0].backend})"
        )
    console.print(f"  total time {time.perf_counter() - started:.2f}s")

    base = output or structure.with_name(f"{structure.stem}_min.{chosen[0]}")
    _write_plain(base, mol, records, chosen, resname or "LIG", frames if trajectory else [],
                 all_trace)


def _write_plain(base, mol, records, formats, resname, frames, trace) -> None:
    """Write a bare molecule (no pipeline Outcome) in the chosen formats."""
    from .write import write_cif, write_pdb, write_sdf, write_trajectory

    writers = {"cif": write_cif, "pdb": write_pdb, "sdf": write_sdf}
    written = []
    for fmt in formats:
        writer = writers[fmt]
        if fmt == "sdf":
            written.append(writer(base.with_suffix(".sdf"), mol, records=records, name=resname))
        else:
            written.append(
                writer(base.with_suffix(f".{fmt}"), mol, records=records, resname=resname)
            )
    if frames:
        written.append(write_trajectory(
            base.with_name(f"{base.stem}_traj.pdb"), mol, frames, resname=resname,
            energies=[s.energy for s in trace] or None,
            stage_labels=[f"stage {s.stage}: {s.backend}" for s in trace] or None,
        ))
    _print_written(written)


@app.command()
def conformers(
    molecule: str = typer.Argument(..., help="SMILES, or a 3D/2D structure file."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output path."),
    confs: int = typer.Option(20, "--confs", "-n", min=1, help="Conformers to keep."),
    method: str = typer.Option("rdkit", "--method", help="rdkit or crest."),
    backend: str = typer.Option("mmff94", "--backend", "-b", help="Backend for ranking."),
    prune_rms: float = typer.Option(0.5, "--prune-rms", help="Duplicate RMSD threshold."),
    energy_window: Optional[float] = typer.Option(None, "--energy-window", help="kcal/mol."),
    formats: str = typer.Option("cif,sdf", "--format", "-f", help="cif, pdb, sdf."),
    threads: int = typer.Option(1, "--threads", "-j", min=1, help="Threads."),
    resname: Optional[str] = typer.Option(None, "--resname", help="PDB residue name."),
) -> None:
    """Conformer search, ranked by energy and de-duplicated by RMSD."""
    from .pipeline import Settings, run

    chosen = _parse_formats(formats)
    settings = Settings(
        backend=backend, n_confs=confs, conf_method=method, prune_rms=prune_rms,
        energy_window=energy_window, formats=chosen, n_threads=threads, resname=resname,
    )
    try:
        mol = read_input(molecule)
        target = output or Path(f"{mol.name.lower()}_confs.{chosen[0]}")
        outcomes = run(mol, settings, output=target)
    except Ligand3DError as exc:
        _fail(exc)
        return
    for outcome in outcomes:
        for note in outcome.notes:
            console.print(f"  · {note}")
        energies = [r.energy for r in outcome.records if r.energy is not None]
        if energies:
            best = min(energies)
            spread = ", ".join(f"{e - best:+.2f}" for e in energies[:10])
            console.print(f"  {len(energies)} conformer(s); relative energies: {spread}")
        _print_written(outcome.written())


@app.command()
def protonate(
    molecule: str = typer.Argument(..., help="SMILES, or a .mol/.sdf file."),
    ph: float = typer.Option(7.4, "--ph", help="pH at which to assign states."),
    all_states: bool = typer.Option(False, "--all", help="List every plausible state."),
) -> None:
    """Report protonation states at a pH, without building anything in 3D."""
    from .protonate import enumerate_states

    try:
        mol = read_input(molecule)
        states = enumerate_states(mol, ph=ph)
    except Ligand3DError as exc:
        _fail(exc)
        return

    console.print(f"[bold]{mol.formula}[/bold]  {mol.smiles}  at pH {ph:g}")
    shown = states if all_states else states[:1]
    for index, state in enumerate(shown):
        marker = "[green]->[/green]" if index == 0 else "  "
        console.print(f"  {marker} {state.smiles}  (charge {state.charge:+d})")
    if not all_states and len(states) > 1:
        console.print(
            f"  [dim]{len(states) - 1} other state(s) are plausible; pass --all to see them[/dim]"
        )


@app.command()
def params(
    structure: Path = typer.Argument(..., help="A 3D .sdf/.mol/.pdb/.cif file."),
    code: str = typer.Option("LIG", "--code", "-c", help="Three-character ligand code."),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", "-d", help="Where to write."),
    no_conformers: bool = typer.Option(
        False, "--no-conformers", help="Skip the rotamer library."
    ),
    allow_code_conflict: bool = typer.Option(
        False, "--allow-code-conflict", help="Proceed even if the code exists in Rosetta."
    ),
) -> None:
    """Generate a Rosetta params file from an existing 3D structure.

    Extra conformers in the input become the rotamer library.
    """
    from . import params as params_mod
    from .molecule import read_3d

    try:
        mol = read_3d(structure)
        result = params_mod.generate(
            mol,
            code=code,
            out_dir=out_dir or structure.parent,
            conformers=not no_conformers and mol.GetNumConformers() > 1,
            allow_code_conflict=allow_code_conflict,
        )
    except Ligand3DError as exc:
        _fail(exc)
        return

    console.print(
        f"[bold]{structure}[/bold]: {mol.GetNumAtoms()} atoms, "
        f"{mol.GetNumConformers()} conformer(s)"
    )
    for line in result.notes:
        console.print(f"  · {line}")
    _print_written(result.paths())


@app.command()
def convert(
    structure: Path = typer.Argument(..., help="A 3D .sdf/.mol/.pdb/.cif file."),
    output: Path = typer.Argument(..., help="Output path; the suffix picks the format."),
    resname: Optional[str] = typer.Option(None, "--resname", help="PDB residue name."),
) -> None:
    """Convert between mmCIF, PDB, and SDF, keeping bond orders where possible."""
    from .molecule import read_3d

    suffix = output.suffix.lower().lstrip(".")
    suffix = "cif" if suffix == "mmcif" else suffix
    if suffix not in ("cif", "pdb", "sdf"):
        _fail(ValueError(f"cannot write {output.suffix!r}; choose .cif, .pdb, or .sdf"))
    try:
        mol = read_3d(structure)
    except Ligand3DError as exc:
        _fail(exc)
        return
    console.print(
        f"[bold]{structure}[/bold] -> {output}  "
        f"({mol.GetNumAtoms()} atoms, {mol.GetNumConformers()} conformer(s))"
    )
    _write_plain(output, mol, None, (suffix,), resname or "LIG", [], [])


# Must stay last: `python -m ligand3d.cli` executes this at the point it
# appears, so any @app.command() defined below it would never be
# registered. Seven of them were, and were invisible to `-m` — which is
# exactly how the container and SLURM paths invoke this module.
if __name__ == "__main__":  # pragma: no cover
    app()

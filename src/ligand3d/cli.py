"""Command-line interface."""

from __future__ import annotations

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
    conf_method: str = typer.Option(
        "rdkit", "--conf-method", help="Conformer search method: rdkit or crest."
    ),
    ph: Optional[float] = typer.Option(
        None,
        "--ph",
        help="Assign protonation state at this pH (default 7.4 if the flag is given "
        "without a value). Omit the flag entirely to keep the structure as drawn.",
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
    solvent: Optional[str] = typer.Option(
        None, "--solvent", help="Implicit solvent for backends that support it, e.g. water."
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
    no_sdf: bool = typer.Option(False, "--no-sdf", help="Do not write the .sdf sidecar."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print the output path."),
) -> None:
    """Build a minimized 3D structure and write it as PDB."""
    from .pipeline import Settings, run

    if stereo not in ("require", "any", "enumerate"):
        _fail(ValueError(f"--stereo must be require, any, or enumerate (got {stereo!r})"))

    settings = Settings(
        backend=backend,
        n_confs=confs,
        conf_method=conf_method,
        prune_rms=prune_rms,
        energy_window=energy_window,
        ph=ph,
        enumerate_states=enumerate_states,
        stereo_mode=stereo,
        solvent=solvent,
        auto_solvent=not no_auto_solvent,
        allow_charge_mismatch=allow_charge_mismatch,
        allow_proton_transfer=allow_proton_transfer,
        max_steps=max_steps,
        seed=seed,
        n_threads=threads,
        resname=resname,
        write_sdf=not no_sdf,
    )

    try:
        mol = read_input(molecule)
        target = output or Path(f"{mol.name.lower()}.pdb")
        if target.suffix.lower() != ".pdb":
            target = target.with_suffix(".pdb")
        outcomes = run(mol, settings, output=target)
    except Ligand3DError as exc:
        _fail(exc)
        return

    if quiet:
        for outcome in outcomes:
            console.print(str(outcome.pdb_path))
        return

    console.print(f"[bold]{mol.formula}[/bold]  {mol.smiles}")
    console.print(f"  stereo: {mol.stereo.describe()}")
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
        console.print(f"  [green]wrote[/green] {outcome.pdb_path}")
        if outcome.sdf_path:
            console.print(f"  [green]wrote[/green] {outcome.sdf_path}")


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
    for key in ("mace-off", "mace-off-small", "mace-off-large", "mace-mp"):
        resolution = cfg.resolve_weights(key)
        if resolution.found:
            console.print(f"  [green]✓[/green] {key}: {resolution.path} [dim](via {resolution.via})[/dim]")
        else:
            console.print(f"  [yellow]✗[/yellow] {key}: not found")

    console.print("\n[bold]optional python packages[/bold]")
    for module, why in (
        ("tblite", "GFN1/GFN2-xTB"),
        ("ase", "optimizers for all non-RDKit backends"),
        ("dimorphite_dl", "pH-based protonation"),
        ("torch", "machine-learned potentials"),
        ("mace", "MACE potentials"),
        ("aimnet", "AIMNet2 potential"),
    ):
        import importlib.util

        present = importlib.util.find_spec(module) is not None
        mark = "[green]✓[/green]" if present else "[yellow]✗[/yellow]"
        console.print(f"  {mark} {module} [dim]— {why}[/dim]")


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
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write the drawn molecule's 3D structure here."
    ),
    backend: str = typer.Option("mmff94", "--backend", "-b", help="Minimization backend."),
    smiles_only: bool = typer.Option(
        False, "--smiles-only", help="Print the SMILES and exit without building 3D."
    ),
) -> None:
    """Draw a molecule in the browser, then build it."""
    from .sketch.server import sketch_molecule

    try:
        molblock = sketch_molecule(port=port, open_browser=not no_browser)
    except Ligand3DError as exc:
        _fail(exc)
        return
    if molblock is None:
        console.print("[yellow]no molecule was submitted[/yellow]")
        raise typer.Exit(code=1)

    from .molecule import from_molblock

    try:
        mol = from_molblock(molblock)
    except Ligand3DError as exc:
        _fail(exc)
        return

    console.print(f"[bold]drew:[/bold] {mol.smiles}")
    if smiles_only:
        return

    from .pipeline import Settings, run

    target = output or Path("sketch.pdb")
    try:
        outcomes = run(mol, Settings(backend=backend), output=target)
    except Ligand3DError as exc:
        _fail(exc)
        return
    for outcome in outcomes:
        console.print(f"  [green]wrote[/green] {outcome.pdb_path}")


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()

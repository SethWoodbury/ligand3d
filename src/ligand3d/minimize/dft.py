"""DFT through ORCA.

This is the first backend that is not a force field or a fitted potential, and
it differs from them in ways the rest of the tool already has vocabulary for:
it consumes total charge *and* spin multiplicity, it has an implicit solvent
model, and it does not hold the bond list fixed. Declaring those in
`Capabilities` is all it takes for the pipeline to route it correctly — the
zwitterion check, the charge-channel refusal and the solvent handling come for
free.

What it does not share with the others is cost. MMFF94 is four milliseconds and
MACE-OFF is seven seconds; a DFT geometry optimisation is minutes to hours, and
scales steeply with size. So the default here is a *composite* method rather
than a functional and basis set chosen separately: `B97-3c` is built for
geometries at a fraction of the cost of a large-basis hybrid, and it is what
you want unless you have a specific reason otherwise.

The realistic use is as the last link in a chain — `mmff94,gfn2,orca` searches
cheaply, narrows, and spends the expensive method only on what survived.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from dataclasses import dataclass

from ..errors import BackendUnavailable
from .ase_bridge import ASEBackend
from .base import Availability, Capabilities, MinimizeJob, register

#: Composite methods, which pair a functional, a basis and the corrections that
#: make the pairing behave. Cheaper than assembling the parts yourself and
#: harder to get wrong.
COMPOSITE_METHODS = ("HF-3c", "PBEh-3c", "B97-3c", "r2SCAN-3c")

DEFAULT_METHOD = "B97-3c"

#: ORCA's CPCM knows these by name. Not the same list as ALPB, so the two are
#: kept apart rather than pretending one mapping covers both.
CPCM_SOLVENTS = frozenset({
    "water", "acetone", "acetonitrile", "ammonia", "benzene", "chloroform",
    "ch2cl2", "dichloromethane", "dmf", "dmso", "ethanol", "hexane",
    "methanol", "octanol", "pyridine", "thf", "toluene",
})


@dataclass
class OrcaOptions:
    """What to put on ORCA's keyword line."""

    method: str = DEFAULT_METHOD
    basis: str = ""
    """Left empty for a composite method, which brings its own."""
    threads: int = 1
    extra: str = ""

    def simple_input(self, solvent: str | None) -> str:
        parts = [self.method]
        if self.basis:
            parts.append(self.basis)
        # ORCA's own optimiser is not used: ligand3d drives L-BFGS through ASE
        # so that every backend produces the same trace and the same
        # convergence criterion. ENGRAD is one gradient per step, which is
        # exactly what the optimiser asks for.
        parts.append("ENGRAD")
        if solvent:
            parts.append(f"CPCM({_cpcm_name(solvent)})")
        if self.extra:
            parts.append(self.extra)
        return " ".join(parts)

    def blocks(self) -> str:
        return f"%pal nprocs {self.threads} end" if self.threads > 1 else ""


def _is_quantum_orca_available() -> bool:
    """True if the quantum chemistry ORCA is resolvable here."""
    from ..config import resolve_orca

    return resolve_orca().found


def _cpcm_name(solvent: str) -> str:
    alias = {"ch2cl2": "dichloromethane", "h2o": "water", "chcl3": "chloroform"}
    return alias.get(solvent.lower(), solvent.lower())


class OrcaBackend(ASEBackend):
    """DFT geometry optimisation driven through ORCA's gradients."""

    caps = Capabilities(
        name="orca",
        kind="dft",
        description=(
            f"DFT via ORCA, {DEFAULT_METHOD} by default. Charge- and spin-aware, "
            "with CPCM implicit solvent. Minutes to hours, not seconds."
        ),
        takes_charge=True,
        supports_solvation=True,
        spin_aware=True,
        fixed_topology=False,
        # A total electronic energy, like the semi-empirical and neural
        # tiers — not a strain energy. Only differences between
        # conformers of the same molecule mean anything.
        energy_kind="total",
        requires=("ase",),
    )

    def __init__(self) -> None:
        self._scratch_dirs: list[str] = []
        atexit.register(self._clean_up)

    def _clean_up(self) -> None:
        for path in self._scratch_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self._scratch_dirs.clear()

    def install_hint(self) -> str:
        return (
            "ORCA is a separate download from https://orcaforum.kofo.mpg.de (free "
            "for academic use). Point LIGAND3D_ORCA_BIN at the binary, or put it "
            "under [binaries] in ~/.config/ligand3d/config.toml.\n"
            "On this cluster it is already at /net/software/orca/latest/orca."
        )

    def extra_availability(self) -> Availability | None:
        from ..config import resolve_orca

        resolution = resolve_orca()
        if not resolution.found:
            return Availability(
                ok=False,
                reason="the ORCA binary was not found",
                hint=self.install_hint(),
            )
        return None

    def make_calculator(self, job: MinimizeJob):
        from ase.calculators.orca import ORCA, OrcaProfile

        from ..config import resolve_orca

        resolution = resolve_orca()
        if not resolution.found:  # pragma: no cover - guarded by availability
            raise BackendUnavailable(self.install_hint())

        options = _options_from(job)
        if job.solvent and _cpcm_name(job.solvent) not in CPCM_SOLVENTS:
            raise BackendUnavailable(
                f"ORCA's CPCM has no entry for {job.solvent!r}. Available: "
                + ", ".join(sorted(CPCM_SOLVENTS))
            )

        # ORCA scatters a dozen files per gradient call. They go somewhere
        # temporary rather than into whatever directory the command was run
        # from, which would otherwise fill with .gbw and .engrad debris.
        scratch = tempfile.mkdtemp(prefix="ligand3d-orca-")
        self._scratch_dirs.append(scratch)

        return ORCA(
            profile=OrcaProfile(command=str(resolution.path)),
            directory=scratch,
            charge=job.charge,
            mult=job.multiplicity,
            orcasimpleinput=options.simple_input(job.solvent),
            orcablocks=options.blocks(),
        )


def _options_from(job: MinimizeJob) -> OrcaOptions:
    """Read the keyword line from the environment.

    A functional is a scientific choice rather than a plumbing one, so it is
    exposed where someone can set it per run without editing code. The default
    is deliberately a composite method; see the module docstring.
    """
    return OrcaOptions(
        method=os.environ.get("LIGAND3D_ORCA_METHOD", DEFAULT_METHOD),
        basis=os.environ.get("LIGAND3D_ORCA_BASIS", ""),
        threads=max(1, getattr(job, "n_threads", 1) or 1),
        extra=os.environ.get("LIGAND3D_ORCA_EXTRA", ""),
    )


register("orca", OrcaBackend)

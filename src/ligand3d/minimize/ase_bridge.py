"""Shared plumbing for every backend that speaks ASE.

Semi-empirical and machine-learned potentials all expose ASE calculators, so
they differ only in how the calculator is constructed. That construction is the
one method a subclass overrides; everything else — coordinate transfer, the
optimizer, unit conversion, writing results back into the RDKit conformer —
lives here exactly once.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from rdkit.Chem import rdchem

from ..errors import BackendUnavailable, MinimizationError
from .base import (
    Availability,
    Capabilities,
    MinimizeJob,
    MinimizeResult,
    TraceStep,
    modules_present,
)

if TYPE_CHECKING:  # pragma: no cover
    from ase import Atoms

EV_TO_KCAL = 23.060547830618307


def mol_to_atoms(mol: rdchem.Mol, conf_id: int = -1) -> "Atoms":
    """Copy one RDKit conformer into an ASE Atoms object."""
    from ase import Atoms

    conf = mol.GetConformer(conf_id)
    return Atoms(
        numbers=[a.GetAtomicNum() for a in mol.GetAtoms()],
        positions=np.asarray(conf.GetPositions(), dtype=float),
    )


def atoms_to_conformer(atoms: "Atoms", mol: rdchem.Mol, conf_id: int = -1) -> None:
    """Write optimized coordinates back into the RDKit conformer in place."""
    from rdkit.Geometry import Point3D

    conf = mol.GetConformer(conf_id)
    positions = atoms.get_positions()
    if len(positions) != mol.GetNumAtoms():
        raise MinimizationError(
            f"optimizer returned {len(positions)} atoms for a molecule with "
            f"{mol.GetNumAtoms()}"
        )
    for i, (x, y, z) in enumerate(positions):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))


class ASEBackend:
    """Base class for ASE-calculator backends.

    Subclasses implement `make_calculator` and set `caps`.
    """

    caps: Capabilities

    def make_calculator(self, job: MinimizeJob):  # pragma: no cover - abstract
        raise NotImplementedError

    def prepare_atoms(self, atoms: "Atoms", job: MinimizeJob) -> None:
        """Annotate the Atoms before the calculator sees it.

        The OMol25-trained potentials read total charge and spin off
        `atoms.info` rather than taking them as constructor arguments, so this
        is where a charge-aware backend passes them along. Default is a no-op.
        """
        return None

    def extra_availability(self) -> Availability | None:
        """Hook for checks beyond module imports, e.g. a missing weights file."""
        return None

    def available(self) -> Availability:
        ok, missing = modules_present(self.caps.requires)
        if not ok:
            return Availability(
                ok=False,
                reason=f"missing Python module(s): {', '.join(missing)}",
                hint=self.install_hint(),
            )
        extra = self.extra_availability()
        if extra is not None and not extra.ok:
            return extra
        return Availability(ok=True)

    def install_hint(self) -> str:
        return ""

    def minimize(self, job: MinimizeJob) -> MinimizeResult:
        avail = self.available()
        if not avail:
            raise BackendUnavailable(
                f"{self.caps.name} is not available: {avail.reason}"
                + (f"\n  {avail.hint}" if avail.hint else "")
            )

        from ase.optimize import LBFGS

        atoms = mol_to_atoms(job.mol, job.conf_id)
        self.prepare_atoms(atoms, job)
        atoms.calc = self.make_calculator(job)

        trace: list[TraceStep] = []
        frames: list = []
        started = time.perf_counter()

        try:
            opt = LBFGS(atoms, logfile=None)
            if job.trace or job.trajectory:
                # ASE calls attached observers once per step, including step 0,
                # which is what makes the first delta meaningful.
                opt.attach(
                    lambda: self._observe(atoms, opt, job, trace, frames), interval=1
                )
            converged = bool(opt.run(fmax=job.fmax, steps=job.max_steps))
            energy_ev = float(atoms.get_potential_energy())
            n_steps = int(opt.get_number_of_steps())
        except Exception as exc:
            raise MinimizationError(
                f"{self.caps.name} failed during optimization: {exc}"
            ) from exc

        atoms_to_conformer(atoms, job.mol, job.conf_id)

        return MinimizeResult(
            energy=energy_ev * EV_TO_KCAL,
            converged=converged,
            n_steps=n_steps,
            backend=self.caps.name,
            energy_unit="kcal/mol",
            energy_kind=self.caps.energy_kind,
            note="" if converged else f"did not converge in {job.max_steps} steps",
            trace=trace,
            frames=frames,
            wall_seconds=time.perf_counter() - started,
        )

    def _observe(self, atoms, opt, job: MinimizeJob, trace: list, frames: list) -> None:
        """Record one optimizer step. Attached to the ASE optimizer."""
        if job.trajectory:
            frames.append(atoms.get_positions().copy())
        if not job.trace:
            return
        energy = float(atoms.get_potential_energy()) * EV_TO_KCAL
        previous = trace[-1].energy if trace else None
        trace.append(
            TraceStep(
                stage=job.stage,
                backend=self.caps.name,
                step=len(trace),
                energy=energy,
                energy_unit="kcal/mol",
                energy_kind=self.caps.energy_kind,
                delta=None if previous is None else energy - previous,
            )
        )

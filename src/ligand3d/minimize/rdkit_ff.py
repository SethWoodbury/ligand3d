"""MMFF94 and UFF, via RDKit's own optimizer.

These stay off the ASE path because RDKit's minimizer operates directly on the
conformer and is roughly a thousand times faster than round-tripping coordinates
through ASE for what is a five-millisecond job.
"""

from __future__ import annotations

import time

from rdkit.Chem import AllChem, rdForceFieldHelpers

from ..errors import MinimizationError
from ..molecule import rdkit_quiet
from .base import (
    Availability,
    Capabilities,
    MinimizeJob,
    MinimizeResult,
    TraceStep,
    register,
)


class _RDKitForceField:
    """Shared implementation; the variant differs only in which FF is built."""

    variant: str = "mmff94"

    def __init__(self, caps: Capabilities) -> None:
        self.caps = caps

    def available(self) -> Availability:
        return Availability(ok=True)

    def _build(self, job: MinimizeJob):
        mol = job.mol
        with rdkit_quiet():
            if self.variant == "uff":
                if not rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
                    raise MinimizationError(
                        "UFF has no parameters for this molecule. Try --backend mmff94 "
                        "or a semi-empirical backend such as gfn2."
                    )
                return AllChem.UFFGetMoleculeForceField(mol, confId=job.conf_id)

            if not rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
                raise MinimizationError(
                    "MMFF94 has no parameters for this molecule (this is common for "
                    "unusual oxidation states, boron, and most metals). Try "
                    "--backend uff for a rough geometry, or --backend gfn2 for a good one."
                )
            props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            return AllChem.MMFFGetMoleculeForceField(mol, props, confId=job.conf_id)

    def minimize(self, job: MinimizeJob) -> MinimizeResult:
        started = time.perf_counter()
        ff = self._build(job)
        with rdkit_quiet():
            ff.Initialize()
            if job.trace or job.trajectory:
                status, steps, trace, frames = self._minimize_stepwise(ff, job)
            else:
                # Minimize returns 0 on convergence, 1 if it hit the cap.
                status = ff.Minimize(maxIts=job.max_steps)
                steps = job.max_steps if status != 0 else -1
                trace, frames = [], []
            energy = ff.CalcEnergy()
        # The converged geometry, whatever the sampling interval landed on. Its
        # own import and conformer lookup, because the untraced branch above
        # defines neither — a trajectory ending ten steps short of the answer
        # is misleading in exactly the way a trajectory should prevent.
        if job.trajectory:
            import numpy as _np

            final = _np.asarray(
                job.mol.GetConformer(job.conf_id).GetPositions(), dtype=float
            ).copy()
            if not frames or not _np.allclose(frames[-1], final, rtol=0, atol=1e-9):
                frames.append(final)

        return MinimizeResult(
            energy=float(energy),
            converged=(status == 0),
            n_steps=steps,
            backend=self.caps.name,
            energy_unit=self.caps.energy_unit,
            energy_kind=self.caps.energy_kind,
            note="" if status == 0 else f"did not converge in {job.max_steps} steps",
            trace=trace,
            frames=frames,
            wall_seconds=time.perf_counter() - started,
        )

    def _minimize_stepwise(self, ff, job: MinimizeJob):
        """Drive the force field one iteration at a time to observe each step.

        RDKit's `Minimize` runs to convergence inside C++ with no callback, so
        the only way to see the path is to ask for one iteration at a time.

        This is not free, and not merely slower: each call restarts the
        optimizer's internal state, so the descent is less efficient and takes
        many more iterations to reach the same place. Measured on a handful of
        drug-sized molecules, single-stepping needed 1300-2000 iterations where
        an uninterrupted run converged well inside the default budget, and
        sometimes hit the cap without converging.

        The energies themselves agree to about 1e-4 kcal/mol, so the trace is
        faithful. But the geometry a user gets must never depend on whether they
        asked for a log, so once the trace is collected this hands control back
        to RDKit for an uninterrupted run to convergence, and records where that
        landed as a final point. Tracing therefore costs time and nothing else.
        """
        import numpy as np

        trace: list[TraceStep] = []
        frames: list = []
        conf = job.mol.GetConformer(job.conf_id)
        status = 1

        every = max(1, int(getattr(job, "trajectory_every", 1) or 1))

        def record(step: int) -> None:
            # Subsampled like the ASE path, and for the same reason: a file of
            # five hundred structures is not a trajectory anyone opens twice.
            # The final geometry is appended after the loop regardless.
            if job.trajectory and step % every == 0:
                frames.append(np.asarray(conf.GetPositions(), dtype=float).copy())
            if not job.trace:
                return
            energy = float(ff.CalcEnergy())
            previous = trace[-1].energy if trace else None
            trace.append(
                TraceStep(
                    stage=job.stage,
                    conf_id=job.conf_id,
                    backend=self.caps.name,
                    step=step,
                    energy=energy,
                    energy_unit=self.caps.energy_unit,
                    energy_kind=self.caps.energy_kind,
                    delta=None if previous is None else energy - previous,
                )
            )

        record(0)
        steps = 0
        for step in range(1, job.max_steps + 1):
            status = ff.Minimize(maxIts=1)
            steps = step
            record(step)
            if status == 0:
                break

        if status != 0:
            # Hand back to RDKit uninterrupted so the geometry we return is the
            # one an untraced run would have produced.
            status = ff.Minimize(maxIts=job.max_steps)
            record(steps + 1)
        return status, steps, trace, frames


def _mmff94() -> _RDKitForceField:
    be = _RDKitForceField(
        Capabilities(
            name="mmff94",
            kind="ff",
            description="MMFF94s classical force field (RDKit). Milliseconds, no extra deps.",
            takes_charge=False,
            supports_solvation=False,
            fixed_topology=True,
            elements=None,
            requires=(),
            energy_unit="kcal/mol",
        )
    )
    be.variant = "mmff94"
    return be


def _uff() -> _RDKitForceField:
    be = _RDKitForceField(
        Capabilities(
            name="uff",
            kind="ff",
            description="Universal Force Field (RDKit). Broad element coverage, low accuracy.",
            takes_charge=False,
            supports_solvation=False,
            fixed_topology=True,
            elements=None,
            requires=(),
            energy_unit="kcal/mol",
        )
    )
    be.variant = "uff"
    return be


register("mmff94", _mmff94, aliases=("mmff", "mmff94s"))
register("uff", _uff)

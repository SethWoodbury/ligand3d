"""MMFF94 and UFF, via RDKit's own optimizer.

These stay off the ASE path because RDKit's minimizer operates directly on the
conformer and is roughly a thousand times faster than round-tripping coordinates
through ASE for what is a five-millisecond job.
"""

from __future__ import annotations

from rdkit.Chem import AllChem, rdForceFieldHelpers

from ..errors import MinimizationError
from ..molecule import rdkit_quiet
from .base import (
    Availability,
    Capabilities,
    MinimizeJob,
    MinimizeResult,
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
        ff = self._build(job)
        with rdkit_quiet():
            ff.Initialize()
            # Minimize returns 0 on convergence, 1 if it hit the iteration cap.
            status = ff.Minimize(maxIts=job.max_steps)
            energy = ff.CalcEnergy()
        return MinimizeResult(
            energy=float(energy),
            converged=(status == 0),
            n_steps=job.max_steps if status != 0 else -1,
            backend=self.caps.name,
            energy_unit=self.caps.energy_unit,
            energy_kind=self.caps.energy_kind,
            note="" if status == 0 else f"did not converge in {job.max_steps} steps",
        )


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

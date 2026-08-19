"""Machine-learned interatomic potentials.

The important difference between these, and the reason `Capabilities` exists at
all: AIMNet2 takes the total molecular charge as an input and MACE does not.
Handing a carboxylate to MACE-OFF gets you a confident answer to a question you
did not ask, so the registry blocks that pairing unless explicitly overridden.

None of these have an implicit solvent model, which makes them the wrong tool
for zwitterions regardless of charge handling.
"""

from __future__ import annotations

import os

from .ase_bridge import ASEBackend
from .base import (
    AIMNET2_ELEMENTS,
    ORGANIC_ELEMENTS,
    Availability,
    Capabilities,
    MinimizeJob,
    register,
)


def _disable_torch_compile() -> None:
    """Turn off TorchDynamo before torch is imported.

    AIMNet2 calls torch.compile, which shells out to a C++ compiler and needs
    Python development headers. On machines without python3-dev that fails with
    'Python.h: No such file or directory' partway through the first energy
    evaluation. Eager mode is a little slower and always works.
    """
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


class MACEBackend(ASEBackend):
    """MACE potentials loaded from a local checkpoint."""

    def __init__(self, model_key: str, name: str, description: str,
                 elements: frozenset[int] | None) -> None:
        self.model_key = model_key
        self.caps = Capabilities(
            name=name,
            kind="mlff",
            description=description,
            takes_charge=False,
            supports_solvation=False,
            elements=elements,
            requires=("mace", "torch", "ase"),
            energy_unit="kcal/mol",
            energy_kind="total",
        )
        self._calc = None

    def install_hint(self) -> str:
        return (
            "pip install torch --index-url https://download.pytorch.org/whl/cpu && "
            "pip install 'ligand3d[mlff]'"
        )

    def extra_availability(self) -> Availability | None:
        from ..config import find_model_weights

        path = find_model_weights(self.model_key)
        if path is None:
            return Availability(
                ok=False,
                reason=f"no weights found for {self.model_key}",
                hint=(
                    f"Set LIGAND3D_{self.model_key.upper().replace('-', '_')} to a .model "
                    f"file, or add a [weights] entry to ~/.config/ligand3d/config.toml. "
                    f"Run 'ligand3d doctor' to see where it looked."
                ),
            )
        return Availability(ok=True)

    def make_calculator(self, job: MinimizeJob):
        _disable_torch_compile()
        from ..config import find_model_weights

        if self._calc is None:
            from mace.calculators import MACECalculator

            path = find_model_weights(self.model_key)
            self._calc = MACECalculator(
                model_paths=str(path), device=_device(), default_dtype="float64"
            )
        return self._calc


class AIMNet2Backend(ASEBackend):
    """AIMNet2 — charge-aware, and fast enough to be a practical default."""

    def __init__(self) -> None:
        self.caps = Capabilities(
            name="aimnet2",
            kind="mlff",
            description="AIMNet2 neural potential. Charge-aware, ~0.5 s for a drug-sized molecule.",
            takes_charge=True,
            supports_solvation=False,
            elements=AIMNET2_ELEMENTS,
            requires=("aimnet", "torch", "ase"),
            energy_unit="kcal/mol",
            energy_kind="total",
        )
        self._calc = None

    def install_hint(self) -> str:
        return (
            "pip install torch --index-url https://download.pytorch.org/whl/cpu && "
            "pip install 'aimnet @ git+https://github.com/isayevlab/aimnetcentral.git'"
        )

    def make_calculator(self, job: MinimizeJob):
        _disable_torch_compile()
        from ..config import aimnet2_model_name

        if self._calc is None:
            from aimnet.calculators import AIMNet2ASE

            self._calc = AIMNet2ASE(
                aimnet2_model_name(), charge=int(job.charge), mult=int(job.multiplicity)
            )
        # The calculator is reused across conformers and across protonation
        # states, so the charge must be set per job rather than only at build.
        self._calc.set_charge(int(job.charge))
        return self._calc


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


register(
    "mace-off",
    lambda: MACEBackend(
        "mace-off",
        "mace-off",
        "MACE-OFF23 (medium) for neutral organic molecules. No charge channel.",
        ORGANIC_ELEMENTS,
    ),
    aliases=("mace-off23", "maceoff"),
)
register(
    "mace-off-small",
    lambda: MACEBackend(
        "mace-off-small",
        "mace-off-small",
        "MACE-OFF23 (small). Faster, slightly less accurate.",
        ORGANIC_ELEMENTS,
    ),
)
register(
    "mace-off-large",
    lambda: MACEBackend(
        "mace-off-large",
        "mace-off-large",
        "MACE-OFF23 (large). Slowest and most accurate of the MACE-OFF set.",
        ORGANIC_ELEMENTS,
    ),
)
register(
    "mace-mp",
    lambda: MACEBackend(
        "mace-mp",
        "mace-mp",
        "MACE-MP-0 universal potential. Broad element coverage, no charge channel.",
        None,
    ),
    aliases=("macemp",),
)
register("aimnet2", AIMNet2Backend, aliases=("aimnet",))

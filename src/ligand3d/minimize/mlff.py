"""Machine-learned interatomic potentials.

The distinction that matters, and the reason `Capabilities` exists: some of
these consume the total molecular charge and some do not. Handing a carboxylate
to a model with no charge channel gets you a confident answer to a question you
did not ask, so the registry blocks that pairing unless explicitly overridden.

Measured on this machine, `atoms.info["charge"]` changes the energy for
MACE-omol, eSEN, UMA, AllScAIP, and AIMNet2, and does not for MACE-OFF,
MACE-MP, or the multi-head MACE models. Those measurements are what the
`takes_charge` flags in `config.MODELS` record.

None of them have an implicit solvent model, which makes the whole family the
wrong tool for zwitterions no matter how they handle charge.
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
    """Turn off TorchDynamo before torch does anything with it.

    MACE, fairchem, and AIMNet2 all reach for torch.compile, which shells out to
    a C++ compiler and needs Python development headers. On a machine without
    python3-dev that fails with 'Python.h: No such file or directory' partway
    through the first energy evaluation, which is a confusing place to discover
    a missing system package. Eager mode is a little slower and always works.
    """
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    try:
        import torch

        torch._dynamo.config.suppress_errors = True
    except Exception:  # torch not installed, or an API that moved
        pass


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _e3nn_version() -> tuple[int, ...] | None:
    try:
        import importlib.metadata as metadata

        return tuple(int(p) for p in metadata.version("e3nn").split(".")[:2] if p.isdigit())
    except Exception:
        return None


_E3NN_CONFLICT = (
    "mace-torch pins e3nn==0.4.4 and fairchem-core requires e3nn>=0.5, so the two "
    "cannot share one environment. Whichever was installed last wins, and the other "
    "fails while deserializing its checkpoint. Keep them in separate virtualenvs: "
    "pip install 'ligand3d[mace]' in one and 'ligand3d[fairchem]' in the other."
)


def _check_e3nn(needs_legacy: bool) -> Availability | None:
    """Detect the mace/fairchem e3nn incompatibility before it becomes a crash.

    Without this the failure surfaces as `ValueError: too many values to unpack`
    from deep inside e3nn's codegen pickling, which says nothing about the real
    problem.
    """
    version = _e3nn_version()
    if version is None or len(version) < 2:
        return None
    is_legacy = version < (0, 5)
    if needs_legacy and not is_legacy:
        return Availability(
            ok=False,
            reason=f"e3nn {'.'.join(map(str, version))} is too new for mace-torch",
            hint=_E3NN_CONFLICT,
        )
    if not needs_legacy and is_legacy:
        return Availability(
            ok=False,
            reason=f"e3nn {'.'.join(map(str, version))} is too old for fairchem-core",
            hint=_E3NN_CONFLICT,
        )
    return None


class _WeightsBackend(ASEBackend):
    """Shared behaviour for potentials loaded from a local checkpoint."""

    def __init__(self, spec, requires: tuple[str, ...], kind: str = "mlff") -> None:
        self.spec = spec
        self.caps = Capabilities(
            name=spec.key,
            kind=kind,
            description=spec.description,
            takes_charge=spec.takes_charge,
            supports_solvation=False,
            elements=ORGANIC_ELEMENTS if spec.organic_only else None,
            requires=requires,
            energy_unit="kcal/mol",
            energy_kind="total",
        )
        self._calc = None

    def weights_path(self):
        from ..config import find_model_weights

        return find_model_weights(self.spec.key)

    def extra_availability(self) -> Availability | None:
        if self.weights_path() is None:
            env = "LIGAND3D_" + self.spec.key.upper().replace("-", "_")
            return Availability(
                ok=False,
                reason=f"no weights found for {self.spec.key}",
                hint=(
                    f"Set {env} to the checkpoint, or add it under [weights] in "
                    f"~/.config/ligand3d/config.toml. Run 'ligand3d doctor' to see "
                    f"every location that was tried."
                ),
            )
        return None


class MACEBackend(_WeightsBackend):
    """MACE potentials, including the multi-head models."""

    def __init__(self, spec) -> None:
        super().__init__(spec, requires=("mace", "torch", "ase"))

    def install_hint(self) -> str:
        return (
            "pip install torch --index-url https://download.pytorch.org/whl/cpu && "
            "pip install 'ligand3d[mace]'"
        )

    def extra_availability(self) -> Availability | None:
        conflict = _check_e3nn(needs_legacy=True)
        if conflict is not None:
            return conflict
        return super().extra_availability()

    def make_calculator(self, job: MinimizeJob):
        _disable_torch_compile()
        if self._calc is None:
            from mace.calculators import MACECalculator

            kwargs = {}
            if self.spec.head:
                # Multi-head checkpoints refuse to load without a head selected,
                # and none of them names a head "default".
                kwargs["head"] = self.spec.head
            self._calc = MACECalculator(
                model_paths=str(self.weights_path()),
                device=_device(),
                default_dtype="float64",
                **kwargs,
            )
        return self._calc

    def prepare_atoms(self, atoms, job: MinimizeJob) -> None:
        if self.spec.takes_charge:
            atoms.info["charge"] = int(job.charge)
            atoms.info["spin"] = int(job.multiplicity)


class FairChemBackend(_WeightsBackend):
    """eSEN, UMA, and AllScAIP through fairchem-core.

    All are trained on OMol25 and read charge and spin off `atoms.info`, so the
    molecular task is what we want in every case.
    """

    def __init__(self, spec) -> None:
        super().__init__(spec, requires=("fairchem", "torch", "ase"))

    def install_hint(self) -> str:
        return (
            "pip install torch --index-url https://download.pytorch.org/whl/cpu && "
            "pip install 'ligand3d[fairchem]'"
        )

    def extra_availability(self) -> Availability | None:
        conflict = _check_e3nn(needs_legacy=False)
        if conflict is not None:
            return conflict
        return super().extra_availability()

    def make_calculator(self, job: MinimizeJob):
        _disable_torch_compile()
        if self._calc is None:
            from fairchem.core.calculate import pretrained_mlip
            from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

            unit = pretrained_mlip.load_predict_unit(
                str(self.weights_path()), device=_device()
            )
            self._calc = FAIRChemCalculator(unit, task_name="omol")
        return self._calc

    def prepare_atoms(self, atoms, job: MinimizeJob) -> None:
        atoms.info["charge"] = int(job.charge)
        atoms.info["spin"] = int(job.multiplicity)


class AIMNet2Backend(ASEBackend):
    """AIMNet2 — charge-aware, and fast enough to be a practical default."""

    def __init__(self) -> None:
        self.caps = Capabilities(
            name="aimnet2",
            kind="mlff",
            description=(
                "AIMNet2 neural potential. Charge-aware, ~0.5 s for a drug-sized molecule."
            ),
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
        # The calculator is reused across conformers and protonation states, so
        # the charge must be set per job rather than only at construction.
        self._calc.set_charge(int(job.charge))
        return self._calc


def _register_from_config() -> None:
    """Register one backend per checkpoint described in config.MODELS."""
    from ..config import MODELS

    families = {"mace": MACEBackend, "fairchem": FairChemBackend}
    aliases = {
        "mace-off": ("mace-off23", "maceoff"),
        "mace-mp": ("macemp",),
        "esen": ("esen-sm", "esen-sm-conserving"),
        "uma-s": ("uma",),
    }
    for spec in MODELS:
        factory = families.get(spec.family)
        if factory is None:
            continue
        register(
            spec.key,
            (lambda s=spec, f=factory: f(s)),
            aliases=aliases.get(spec.key, ()),
        )


_register_from_config()
register("aimnet2", AIMNet2Backend, aliases=("aimnet",))

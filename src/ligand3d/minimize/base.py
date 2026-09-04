"""The backend registry.

Backends differ in ways the pipeline has to reason about. GFN2-xTB consumes the
total molecular charge and offers implicit solvation; MACE-MP consumes neither;
AIMNet2 takes charge but has no solvent model; MMFF94 has no concept of either
but does have its own parameter coverage limits.

Rather than special-casing names at the call site, each backend declares what it
can do and the registry refuses combinations that would silently produce a wrong
answer. Adding a new potential means adding one file and one registration.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol, runtime_checkable

from rdkit import Chem

from ..errors import BackendMismatch, BackendUnavailable

#: "hf" is separate from "dft" because HF-3c is Hartree-Fock — a wavefunction
#: method with no exchange-correlation functional at all. Filing it under DFT
#: put a plainly wrong label on a result, which is the one thing a method name
#: exists to prevent.
BackendKind = Literal["ff", "semiempirical", "mlff", "dft", "hf"]

# Elements covered by the organic-chemistry MLFFs, as trained.
ORGANIC_ELEMENTS = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})
AIMNET2_ELEMENTS = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53})


@dataclass(frozen=True)
class Capabilities:
    """What a backend can and cannot do."""

    name: str
    kind: BackendKind
    description: str = ""
    takes_charge: bool = False
    """True if the model consumes the total molecular charge.

    This is the distinction that matters most in practice. A potential without a
    charge channel cannot tell a carboxylate from a carboxylic acid whose proton
    has wandered off, so handing it a charged species produces a confident and
    wrong answer.
    """
    supports_solvation: bool = False
    spin_aware: bool = False
    """Consumes spin multiplicity as an input, not just total charge."""
    fixed_topology: bool = False
    """True if the method holds the bond list fixed and so cannot move a proton.

    MMFF94 and UFF build their terms from an explicit bond table, so a hydrogen
    physically cannot migrate to a different heavy atom no matter what the
    geometry does. Semi-empirical and machine-learned potentials work from
    positions alone and will happily relocate a proton — which is exactly how a
    gas-phase zwitterion collapses. Only the second group needs a dielectric to
    keep a charge-separated species intact.
    """
    elements: frozenset[int] | None = None
    """Supported atomic numbers, or None for unrestricted."""
    requires: tuple[str, ...] = ()
    """Python modules that must be importable."""
    energy_unit: str = "kcal/mol"
    energy_kind: Literal["strain", "total"] = "strain"
    """What the reported energy means.

    Classical force fields report a strain energy near zero; semi-empirical and
    ML potentials report a total electronic energy in the tens of thousands.
    Printing one next to the other without saying which is which invites a
    reader to compare numbers that are not comparable, and energies from
    different backends are never comparable to each other at all.
    """


@dataclass(frozen=True)
class Availability:
    """Whether a backend can run, and if not, what to do about it."""

    ok: bool
    reason: str = ""
    hint: str = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class TraceStep:
    """One optimizer step, for the energy trace."""

    stage: int
    """Position in a chained backend spec, so `mmff94,gfn2` stays separable."""
    backend: str
    step: int
    energy: float
    conf_id: int = -1
    """Which conformer this step belongs to.

    Without it, tracing more than one conformer yields one series that jumps
    between them — a curve describing nothing. The plot groups on
    (stage, conf_id) for exactly this reason.
    """
    energy_unit: str = "kcal/mol"
    energy_kind: str = "strain"
    delta: float | None = None
    """Change from the previous step of the same stage.

    Deliberately not carried across a stage boundary: energies from two
    different methods are not comparable, so a delta spanning them would be
    a meaningless number that looks meaningful.
    """

    def to_json(self) -> dict:
        return {
            "stage": self.stage,
            "conf_id": self.conf_id,
            "backend": self.backend,
            "step": self.step,
            "energy": self.energy,
            "energy_unit": self.energy_unit,
            "energy_kind": self.energy_kind,
            "delta": self.delta,
        }


@dataclass
class MinimizeJob:
    """Everything a backend needs to minimize one conformer."""

    mol: Chem.Mol
    conf_id: int = -1
    charge: int = 0
    multiplicity: int = 1
    max_steps: int = 500
    """Per backend, per conformer — not a budget shared across a chain. A
    three-method chain on five conformers may take fifteen times this."""
    fmax: float = 0.05
    """Force convergence threshold in eV/Angstrom (ASE backends only)."""
    solvent: str | None = None
    n_threads: int = 1

    trace: bool = False
    """Record the energy at every optimizer step.

    Off by default because it forces the RDKit force fields to be driven one
    step at a time, which is slower than letting them run to convergence
    internally.
    """
    trajectory: bool = False
    trajectory_every: int = 1
    """Keep every Nth frame. 1 keeps all of them."""
    """Keep the coordinates at every step so the path can be written out."""
    stage: int = 0
    """Which link of a backend chain this job is; recorded on each trace step."""


@dataclass
class MinimizeResult:
    """Outcome of minimizing one conformer."""

    energy: float
    converged: bool
    n_steps: int
    backend: str
    energy_unit: str = "kcal/mol"
    energy_kind: str = "strain"
    note: str = ""
    trace: list[TraceStep] = field(default_factory=list)
    frames: list = field(default_factory=list)
    """Coordinates per step, as (n_atoms, 3) arrays, when trajectory is on."""
    wall_seconds: float = 0.0


@runtime_checkable
class Backend(Protocol):
    caps: Capabilities

    def available(self) -> Availability: ...

    def minimize(self, job: MinimizeJob) -> MinimizeResult: ...


_REGISTRY: dict[str, Callable[[], Backend]] = {}
_ALIASES: dict[str, str] = {}


def register(name: str, factory: Callable[[], Backend], aliases: tuple[str, ...] = ()) -> None:
    """Register a backend factory under `name`."""
    _REGISTRY[name] = factory
    for alias in aliases:
        _ALIASES[alias] = name


def resolve_name(name: str) -> str:
    key = name.strip().lower()
    return _ALIASES.get(key, key)


def get_backend(name: str) -> Backend:
    """Instantiate a backend by name or alias."""
    key = resolve_name(name)
    if key not in _REGISTRY:
        known = ", ".join(sorted(set(_REGISTRY) | set(_ALIASES)))
        raise BackendUnavailable(f"unknown backend {name!r}. Available: {known}")
    return _REGISTRY[key]()


def list_backends() -> list[str]:
    return sorted(_REGISTRY)


def all_backends() -> list[Backend]:
    out = []
    for name in list_backends():
        try:
            out.append(_REGISTRY[name]())
        except Exception:  # a broken backend should not hide the working ones
            continue
    return out


def modules_present(modules: tuple[str, ...]) -> tuple[bool, list[str]]:
    """Check importability without importing. Keeps `doctor` fast."""
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    return (not missing, missing)


def check_compatible(
    caps: Capabilities,
    *,
    charge: int,
    elements: frozenset[int],
    solvent: str | None,
    is_zwitterion: bool = False,
    allow_charge_mismatch: bool = False,
) -> None:
    """Refuse molecule/backend pairings that would silently be wrong."""
    if caps.elements is not None:
        unsupported = elements - caps.elements
        if unsupported:
            symbols = ", ".join(
                sorted(Chem.GetPeriodicTable().GetElementSymbol(z) for z in unsupported)
            )
            raise BackendMismatch(
                f"{caps.name} does not support {symbols}. "
                f"Supported elements: "
                f"{', '.join(sorted(Chem.GetPeriodicTable().GetElementSymbol(z) for z in caps.elements))}"
            )

    # A fixed-topology force field reads formal charges off the atom typing, so
    # it already knows the molecule is charged; only the potentials that infer
    # everything from positions need to be told.
    if charge != 0 and not caps.takes_charge and not caps.fixed_topology and not allow_charge_mismatch:
        raise BackendMismatch(
            f"molecule has net charge {charge:+d} but {caps.name} has no charge channel, "
            f"so it would treat it as neutral and give a confidently wrong geometry. "
            f"Use a charge-aware backend (gfn2, gfn1, gfnff, aimnet2), neutralize the "
            f"input, or pass --allow-charge-mismatch if you know what you are doing."
        )

    if solvent and not caps.supports_solvation:
        raise BackendMismatch(
            f"{caps.name} has no implicit solvent model, but solvation was requested. "
            f"Drop --solvent, or use gfn2/gfn1/gfnff which support ALPB."
        )

    if (
        is_zwitterion
        and not caps.supports_solvation
        and not caps.fixed_topology
        and not allow_charge_mismatch
    ):
        raise BackendMismatch(
            f"input is a zwitterion and {caps.name} has no implicit solvent model. "
            f"In gas phase the proton transfers back and you get a different molecule "
            f"than the one you asked for. Use --backend gfn2 (which defaults to ALPB "
            f"water here), or pass --allow-charge-mismatch to override."
        )


@dataclass
class ChainStep:
    """One stage of a chained minimization such as `mmff94,gfn2`."""

    backend: Backend
    result: MinimizeResult | None = None


def parse_chain(spec: str) -> list[str]:
    """Split a `--backend a,b,c` specification into resolved names."""
    names = [part.strip() for part in spec.split(",") if part.strip()]
    if not names:
        raise BackendUnavailable("empty backend specification")
    return [resolve_name(n) for n in names]


def load_builtin_backends() -> None:
    """Import the backend modules so their registrations run.

    Import errors are swallowed on purpose: a missing optional dependency should
    make one backend unavailable, not break the program.
    """
    from . import rdkit_ff  # noqa: F401  (registers mmff94, uff)

    for module in ("xtb", "mlff", "dft"):
        try:
            __import__(f"{__package__}.{module}")
        except Exception:  # pragma: no cover - depends on optional deps
            continue

"""One description of every method, for `ligand3d models` and the web page.

Choosing a backend means weighing several things at once — is it charge aware,
how long will it take, does it have a solvent model, will it even run here — and
that information was previously scattered between the registry, the model table,
and a README. This assembles it in one place so both the terminal and the
browser show the same answer.

The classical and semi-empirical methods are described here rather than in
`config.MODELS`, which only covers checkpointed neural potentials.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class MethodInfo:
    """Everything worth knowing about one backend before choosing it."""

    id: str
    kind: str
    family: str
    description: str = ""

    charge: str = "n/a"
    """explicit (consumes total charge), implicit (infers from atoms), or n/a."""
    spin: str = "no"
    solvent: str = "no"

    speed: str = ""
    memory: str = ""
    accuracy: str = ""
    training: str = ""
    elements: str = "unrestricted"
    notes: str = ""

    repo: str = ""
    weights_file: str = ""
    weights_path: str | None = None
    weights_mb: int = 0

    ready: bool = False
    reason: str = ""
    hint: str = ""
    requires: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["requires"] = list(self.requires)
        data["aliases"] = list(self.aliases)
        return data


# The methods that are not checkpointed neural potentials. Speeds are wall time
# for gabapentin (29 atoms) on this machine, single conformer.
_CLASSICAL: dict[str, dict[str, Any]] = {
    "mmff94": dict(
        family="classical force field",
        charge="implicit",
        speed="~5 ms",
        memory="negligible",
        accuracy="good bond lengths and angles; poor for unusual electronics",
        training="MMFF94s parameter set (Halgren), fitted to HF and experiment",
        notes="bond list is fixed, so it cannot move a proton. No extra dependencies.",
        elements="wherever MMFF94 has parameters; fails loudly otherwise",
    ),
    "uff": dict(
        family="classical force field",
        charge="implicit",
        speed="~5 ms",
        memory="negligible",
        accuracy="rough; use when MMFF94 has no parameters",
        training="Universal Force Field, rule-based from element and hybridization",
        notes="bond list is fixed. Very broad element coverage, low accuracy.",
    ),
    "gfnff": dict(
        family="generic force field (xtb)",
        charge="explicit",
        solvent="ALPB",
        speed="~0.1 s",
        memory="~100 MB",
        accuracy="between a classical FF and GFN2; good geometries, very fast",
        training="GFN-FF, parameterized across the periodic table",
        notes="runs in the xtb binary, which optimizes internally, so it cannot be traced.",
    ),
    "gfn1": dict(
        family="semi-empirical tight binding",
        charge="explicit",
        solvent="ALPB",
        speed="~0.8 s",
        memory="~200 MB",
        accuracy="good; superseded by GFN2 for most purposes",
        training="GFN1-xTB",
    ),
    "gfn2": dict(
        family="semi-empirical tight binding",
        charge="explicit",
        solvent="ALPB",
        speed="~0.9 s",
        memory="~200 MB",
        accuracy="near-QM geometries; the best value here for charged species",
        training="GFN2-xTB, fitted to DFT reference data",
        notes="the only tier with both a charge channel and implicit solvent.",
    ),
    "aimnet2": dict(
        family="neural potential",
        charge="explicit",
        speed="~0.5 s",
        memory="~500 MB",
        accuracy="high for organics; the fastest charge-aware option",
        training="wB97M-D3 on a broad organic dataset",
        notes="weights download themselves and cache in ~/.cache/aimnet.",
        repo="isayevlab/aimnetcentral",
    ),
}


def _element_summary(caps) -> str:
    from rdkit import Chem

    if caps.elements is None:
        return "unrestricted"
    table = Chem.GetPeriodicTable()
    return ", ".join(sorted(table.GetElementSymbol(z) for z in caps.elements))


def build_catalog() -> list[MethodInfo]:
    """Every registered backend, described and with its availability resolved."""
    from .config import MODELS_BY_KEY, resolve_weights
    from .minimize import all_backends
    from .minimize.base import _ALIASES

    aliases: dict[str, list[str]] = {}
    for alias, target in _ALIASES.items():
        aliases.setdefault(target, []).append(alias)

    catalog: list[MethodInfo] = []
    for backend in all_backends():
        caps = backend.caps
        try:
            availability = backend.available()
            ready, reason = bool(availability), availability.reason
            hint = availability.hint
        except Exception as exc:
            ready, reason, hint = False, f"{type(exc).__name__}: {exc}", ""

        info = MethodInfo(
            id=caps.name,
            kind=caps.kind,
            family=caps.kind,
            description=caps.description,
            charge="explicit" if caps.takes_charge else "implicit",
            spin="yes" if caps.spin_aware else "no",
            solvent="ALPB" if caps.supports_solvation else "no",
            elements=_element_summary(caps),
            ready=ready,
            reason=reason,
            hint=hint,
            requires=tuple(caps.requires),
            aliases=tuple(sorted(aliases.get(caps.name, ()))),
        )

        extra = _CLASSICAL.get(caps.name)
        if extra:
            for key, value in extra.items():
                setattr(info, key, value)

        spec = MODELS_BY_KEY.get(caps.name)
        if spec is not None:
            resolution = resolve_weights(spec.key)
            info.family = {
                "mace": "MACE",
                "mace-polar": "MACE-POLAR",
                "fairchem": "fairchem / OMol25",
                "aimnet2": "AIMNet2",
            }.get(spec.family, spec.family)
            info.repo = spec.repo
            info.training = spec.training
            info.speed = spec.speed
            info.memory = spec.memory
            info.accuracy = spec.accuracy
            info.notes = spec.notes
            info.spin = "yes" if spec.spin_aware else "no"
            info.weights_file = spec.filename
            info.weights_mb = spec.approx_mb
            info.weights_path = str(resolution.path) if resolution.found else None

        catalog.append(info)

    order = {"ff": 0, "semiempirical": 1, "mlff": 2}
    catalog.sort(key=lambda m: (order.get(m.kind, 3), not m.ready, m.id))
    return catalog


@dataclass
class CatalogSummary:
    """The catalog plus the context needed to read it."""

    methods: list[MethodInfo] = field(default_factory=list)
    weight_roots: list[str] = field(default_factory=list)
    unsupported: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "methods": [m.to_json() for m in self.methods],
            "weight_roots": self.weight_roots,
            "unsupported": self.unsupported,
        }


def summarize() -> CatalogSummary:
    """The catalog, plus where weights are being read from."""
    from .config import UNSUPPORTED_MODELS, _WEIGHT_PROBE_DIRS

    import os
    from pathlib import Path

    methods = build_catalog()
    # Collapse to the store the checkpoints actually live under. Listing every
    # per-model subdirectory is sixteen near-identical paths and tells nobody
    # anything they did not already know.
    found = [Path(m.weights_path) for m in methods if m.weights_path]
    roots: list[str] = []
    if found:
        common = os.path.commonpath([str(p.parent) for p in found])
        roots = [common]
    if not roots:
        roots = [str(d) for d in _WEIGHT_PROBE_DIRS]
    return CatalogSummary(
        methods=methods, weight_roots=roots, unsupported=dict(UNSUPPORTED_MODELS)
    )

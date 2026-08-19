"""Finding model weights and external binaries.

Weights are hundreds of megabytes and binaries are platform-specific, so neither
belongs in the repo. Every resource resolves through the same ordered search:

    1. an explicit environment variable
    2. ~/.config/ligand3d/config.toml
    3. a probe of locations where these things are conventionally installed
    4. the local cache at ~/.cache/ligand3d/

`ligand3d doctor` prints each step's outcome, so a miss is diagnosable without
reading this file.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

ENV_PREFIX = "LIGAND3D_"

CONFIG_PATH = Path(
    os.environ.get("LIGAND3D_CONFIG", Path.home() / ".config" / "ligand3d" / "config.toml")
)
CACHE_DIR = Path(
    os.environ.get("LIGAND3D_CACHE", Path.home() / ".cache" / "ligand3d")
)

# Conventional install locations probed as a last resort before the cache.
# These are deliberately additive: a hit is a convenience, a miss is not an error.
_WEIGHT_PROBE_DIRS = [
    Path("/net/databases/huggingface/mlFF_models"),
    Path("/mnt/projects/ml/mlff/models"),
    CACHE_DIR / "weights",
]


@dataclass(frozen=True)
class ModelSpec:
    """One machine-learned potential checkpoint.

    `patterns` are tried in order inside each probe directory, so the
    HuggingFace-style layout used on the cluster is found first and a flat
    directory of `.model` files still works.
    """

    key: str
    family: str  # "mace" | "mace-polar" | "fairchem" | "aimnet2"
    patterns: tuple[str, ...]
    description: str = ""
    takes_charge: bool = False
    head: str | None = None  # multi-head MACE models need one selected
    organic_only: bool = False
    approx_mb: int = 0

    # --- reference information, for `ligand3d models` and the web page -----
    repo: str = ""
    """Upstream identifier, e.g. ACEsuit/mace-off-23."""
    training: str = ""
    """What it was trained on, which is what actually bounds its accuracy."""
    spin_aware: bool = False
    """Consumes spin multiplicity as an input, not just total charge."""
    speed: str = ""
    """Rough wall time for a drug-sized molecule on CPU, measured here."""
    memory: str = ""
    accuracy: str = ""
    notes: str = ""

    def name_padded(self, width: int = 16) -> str:
        return f"{self.key}:".ljust(width + 1)

    @property
    def filename(self) -> str:
        """The checkpoint filename, without any directory."""
        return self.patterns[-1] if self.patterns else ""

    @property
    def charge_handling(self) -> str:
        """How the model learns about charge.

        "explicit" means the total charge is an input the model consumes.
        "implicit" means it only sees atoms and positions, so a carboxylate and
        a neutral acid missing a proton look identical to it.
        """
        return "explicit" if self.takes_charge else "implicit"


def _hf(repo: str, filename: str) -> tuple[str, ...]:
    """Both the HuggingFace cache layout and a plain filename."""
    return (f"models--{repo}/{filename}", filename)


# Every checkpoint ligand3d knows how to load. Presence is discovered at
# runtime; listing one here does not assume it exists on this machine.
MODELS: tuple[ModelSpec, ...] = (
    # --- MACE, organic molecules ------------------------------------------
    ModelSpec("mace-off", "mace", _hf("ACEsuit--mace-off-23", "MACE-OFF23_medium.model"),
              "MACE-OFF23 medium. Neutral organic molecules.", organic_only=True, approx_mb=18,
              repo="ACEsuit/mace-off-23", training="SPICE, neutral organics (wB97M-D3)", speed="~7 s", memory="~1 GB", accuracy="very good for neutral organics", notes="no total-charge input"),
    ModelSpec("mace-off-small", "mace", _hf("ACEsuit--mace-off-23", "MACE-OFF23_small.model"),
              "MACE-OFF23 small. Fastest of the OFF23 set.", organic_only=True, approx_mb=7,
              repo="ACEsuit/mace-off-23", training="SPICE, neutral organics", speed="~3 s", memory="~0.8 GB", accuracy="a little below medium", notes="fastest MACE-OFF"),
    ModelSpec("mace-off-large", "mace", _hf("ACEsuit--mace-off-23", "MACE-OFF23_large.model"),
              "MACE-OFF23 large. Most accurate of the OFF23 set.", organic_only=True, approx_mb=55,
              repo="ACEsuit/mace-off-23", training="SPICE, neutral organics", speed="~15 s", memory="~1.5 GB", accuracy="best of the OFF23 set", notes="slowest MACE-OFF"),
    ModelSpec("mace-off-24", "mace", _hf("ACEsuit--mace-off-24", "MACE-OFF24_medium.model"),
              "MACE-OFF24 medium. Successor to OFF23.", organic_only=True, approx_mb=18,
              repo="ACEsuit/mace-off-24", training="SPICE v2, wider coverage", speed="~7 s", memory="~1 GB", accuracy="successor to OFF23", notes="no total-charge input"),
    # --- MACE, charge-aware -----------------------------------------------
    ModelSpec("mace-omol", "mace",
              _hf("ACEsuit--mace-omol-0", "MACE-omol-0-extra-large-1024.model"),
              "MACE trained on OMol25. Consumes total charge.",
              takes_charge=True, approx_mb=422,
              repo="ACEsuit/mace-omol-0", training="OMol25 (charged and open-shell)", speed="~20 s", memory="~3 GB", accuracy="high; the charge-aware MACE", notes="extra-large 1024-channel; slow on CPU"),
    # --- MACE, multi-head -------------------------------------------------
    ModelSpec("mace-mh", "mace", _hf("ACEsuit--mace-mh-0", "mace-mh-0.model"),
              "MACE multi-head (omol head). Fast and general, ignores charge.",
              head="omol", approx_mb=40,
              repo="ACEsuit/mace-mh-0", training="multi-head: omol, SPICE, MatPES, OMat, OC20", speed="~2 s", memory="~1 GB", accuracy="good general purpose", notes="omol head; ignores charge despite the head name"),
    ModelSpec("mace-mh-1", "mace", _hf("ACEsuit--mace-mh-1", "mace-mh-1.model"),
              "MACE multi-head v1 (omol head).", head="omol", approx_mb=59,
              repo="ACEsuit/mace-mh-1", training="multi-head v1", speed="~2 s", memory="~1 GB", accuracy="good general purpose", notes="omol head selected"),
    ModelSpec("mace-mh-spice", "mace", _hf("ACEsuit--mace-mh-0", "mace-mh-0.model"),
              "MACE multi-head, SPICE wB97M head. Organic chemistry.",
              head="spice_wB97M", approx_mb=40,
              repo="ACEsuit/mace-mh-0", training="multi-head, SPICE wB97M head", speed="~2 s", memory="~1 GB", accuracy="organic chemistry", notes="spice_wB97M head selected"),
    # --- MACE, materials --------------------------------------------------
    ModelSpec("mace-mp", "mace",
              _hf("ACEsuit--mace-mp-0", "MACE-matpes-r2scan-omat-ft.model"),
              "MACE-MP-0 universal potential. Broad elements, ignores charge.",
              approx_mb=79,
              repo="ACEsuit/mace-mp-0", training="MatPES r2SCAN + OMat fine-tune", speed="~5 s", memory="~1 GB", accuracy="built for periodic solids, not molecules", notes="broadest elements; for inorganics"),
    # --- fairchem / OMol25, all charge- and spin-aware ---------------------
    ModelSpec("esen", "fairchem",
              _hf("facebook--esen-sm-conserving-all-omol", "esen_sm_conserving_all.pt"),
              "eSEN small, conserving. OMol25; excellent accuracy per second.",
              takes_charge=True, approx_mb=51,
              repo="facebook/esen-sm-conserving-all-omol", training="OMol25 (wB97M-V)", spin_aware=True, speed="~1 s", memory="~1 GB", accuracy="best accuracy per second here", notes="energy-conserving: forces are exact gradients"),
    ModelSpec("esen-sm-direct", "fairchem",
              _hf("facebook--esen-sm-direct-all-omol", "esen_sm_direct_all.pt"),
              "eSEN small, direct force prediction. Faster, not energy-conserving.",
              takes_charge=True, approx_mb=51,
              repo="facebook/esen-sm-direct-all-omol", training="OMol25", spin_aware=True, speed="~0.7 s", memory="~1 GB", accuracy="close to conserving, a little faster", notes="direct forces; not energy-conserving"),
    ModelSpec("esen-md-direct", "fairchem",
              _hf("facebook--esen-md-direct-all-omol", "esen_md_direct_all.pt"),
              "eSEN medium, direct.", takes_charge=True, approx_mb=406,
              repo="facebook/esen-md-direct-all-omol", training="OMol25", spin_aware=True, speed="~3 s", memory="~2 GB", accuracy="better than the small models", notes="direct forces"),
    ModelSpec("uma-s", "fairchem",
              _hf("facebook--fairchem-uma-s-1p1", "uma-s-1p1.pt"),
              "UMA small 1.1. Universal model; omol task.",
              takes_charge=True, approx_mb=1170,
              repo="facebook/fairchem-uma-s-1p1", training="UMA multi-domain, omol task", spin_aware=True, speed="~12 s", memory="~4 GB", accuracy="strong across chemistry and materials", notes="first load is slow"),
    ModelSpec("uma-s-1p2", "fairchem",
              _hf("facebook--fairchem-uma-s-1p2", "uma-s-1p2.pt"),
              "UMA small 1.2.", takes_charge=True, approx_mb=2330,
              repo="facebook/fairchem-uma-s-1p2", training="UMA 1.2", spin_aware=True, speed="~12 s", memory="~6 GB", accuracy="newer UMA small"),
    ModelSpec("uma-sm", "fairchem",
              _hf("facebook--fairchem-uma-sm", "uma_sm.pt"),
              "UMA sm.", takes_charge=True, approx_mb=1170,
              repo="facebook/fairchem-uma-sm", training="UMA sm", spin_aware=True, speed="~12 s", memory="~4 GB"),
    ModelSpec("uma-m", "fairchem",
              _hf("facebook--fairchem-uma-m-1p1", "uma-m-1p1.pt"),
              "UMA medium 1.1. Largest and slowest; needs real memory.",
              takes_charge=True, approx_mb=11170,
              repo="facebook/fairchem-uma-m-1p1", training="UMA medium", spin_aware=True, speed="minutes", memory="~24 GB", accuracy="most accurate UMA", notes="11 GB of weights; needs real memory"),
    ModelSpec("allscaip", "fairchem",
              _hf("facebook--allscaip-omol102m-md-cons", "AllScAIP-OMol102M-md-cons.pt"),
              "AllScAIP OMol102M, conserving.", takes_charge=True, approx_mb=688,
              repo="facebook/allscaip-omol102m-md-cons", training="OMol25 102M", spin_aware=True, speed="~7 s", memory="~3 GB", accuracy="high", notes="energy-conserving"),
    # --- MACE-POLAR: real, but needs the patched fork ----------------------
    ModelSpec("mace-polar-s", "mace-polar",
              _hf("ACEsuit--mace-polar-1-beta", "MACE-POLAR-1-S.model"),
              "MACE-POLAR small. Explicit long-range electrostatics.",
              approx_mb=33, repo="ACEsuit/mace-polar-1-beta",
              training="polarizable dataset with explicit long-range terms",
              speed="~4 s", memory="~1 GB",
              accuracy="models the long-range electrostatics other MACE models omit",
              notes="needs the patched MACE fork plus graph_longrange"),
    ModelSpec("mace-polar", "mace-polar",
              _hf("ACEsuit--mace-polar-1-beta", "MACE-POLAR-1-M.model"),
              "MACE-POLAR medium. Explicit long-range electrostatics.",
              approx_mb=68, repo="ACEsuit/mace-polar-1-beta",
              training="polarizable dataset with explicit long-range terms",
              speed="~8 s", memory="~1.5 GB", accuracy="the usual POLAR pick",
              notes="needs the patched MACE fork plus graph_longrange"),
    ModelSpec("mace-polar-l", "mace-polar",
              _hf("ACEsuit--mace-polar-1-beta", "MACE-POLAR-1-L.model"),
              "MACE-POLAR large. Explicit long-range electrostatics.",
              approx_mb=130, repo="ACEsuit/mace-polar-1-beta",
              training="polarizable dataset with explicit long-range terms",
              speed="~16 s", memory="~2 GB", accuracy="best of the POLAR set",
              notes="needs the patched MACE fork plus graph_longrange"),
    ModelSpec("allscaip-direct", "fairchem",
              _hf("facebook--allscaip-omol102m-md-d", "AllScAIP-OMol102M-md-d.pt"),
              "AllScAIP OMol102M, direct.", takes_charge=True, approx_mb=695,
              repo="facebook/allscaip-omol102m-md-d", training="OMol25 102M", spin_aware=True, speed="~5 s", memory="~3 GB", accuracy="high", notes="direct forces"),
)

MODELS_BY_KEY: dict[str, ModelSpec] = {m.key: m for m in MODELS}

# Models present on the cluster that ligand3d cannot currently load, kept here
# so `doctor` can explain the gap rather than staying silent about it.
UNSUPPORTED_MODELS: dict[str, str] = {
    "so3lr": (
        "SO3LR is a JAX model and needs jax, orbax, and the so3lr package; "
        "ligand3d's backends are all torch or ASE based."
    ),
    "orb-mol": (
        "orb-models 0.7 removed orb_models.forcefield.calculator, so there is no "
        "ASE calculator to attach; pin orb-models<0.7 and open an issue if needed."
    ),
}

_WEIGHT_PATTERNS: dict[str, tuple[str, ...]] = {m.key: m.patterns for m in MODELS}

_XTB_PROBE = [
    Path("/home/woodbuse/conda/envs/qcb-xtb/bin/xtb"),
    CACHE_DIR / "xtb" / "bin" / "xtb",
]
_CREST_PROBE = [
    Path("/home/woodbuse/conda/envs/qcb-xtb/bin/crest"),
    CACHE_DIR / "crest" / "bin" / "crest",
]
_XTB_LIB_PROBE = [
    Path("/home/woodbuse/conda/envs/qcb-xtb/lib"),
]

# Rosetta's molfile_to_params.py. It imports a sibling `rosetta_py` package via
# sys.path[0], so it must be invoked at its real location rather than copied.
_MOLFILE_TO_PARAMS_PROBE = [
    Path("/net/software/rosetta/main/source/scripts/python/public/molfile_to_params.py"),
    Path("/net/software/rosetta/latest/source/scripts/python/public/molfile_to_params.py"),
]
_ROSETTA_RESIDUE_TYPES_PROBE = [
    Path(
        "/net/software/rosetta/main/database/chemical/residue_type_sets/"
        "fa_standard/residue_types.txt"
    ),
    Path(
        "/net/software/rosetta/latest/database/chemical/residue_type_sets/"
        "fa_standard/residue_types.txt"
    ),
]


@dataclass
class Resolution:
    """Where a resource was found, and what was tried along the way."""

    key: str
    path: Path | None = None
    via: str = ""
    tried: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.path is not None


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Read ~/.config/ligand3d/config.toml, or {} if there isn't one."""
    if tomllib is None or not CONFIG_PATH.exists():
        return {}
    try:
        return tomllib.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def _env_key(key: str) -> str:
    return ENV_PREFIX + key.upper().replace("-", "_")


def resolve_weights(key: str) -> Resolution:
    """Locate the weights file for a logical model name."""
    res = Resolution(key=key)

    env = os.environ.get(_env_key(key))
    res.tried.append(f"${_env_key(key)}")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            res.path, res.via = p, f"${_env_key(key)}"
            return res

    cfg = load_config().get("weights", {})
    res.tried.append(f"{CONFIG_PATH}:[weights].{key}")
    if key in cfg:
        p = Path(str(cfg[key])).expanduser()
        if p.exists():
            res.path, res.via = p, "config.toml"
            return res

    for base in _WEIGHT_PROBE_DIRS:
        for pattern in _WEIGHT_PATTERNS.get(key, ()):
            candidate = base / pattern
            res.tried.append(str(candidate))
            if candidate.exists():
                res.path, res.via = candidate, "probe"
                return res
    return res


def find_model_weights(key: str) -> Path | None:
    return resolve_weights(key).path


def resolve_binary(name: str, probes: list[Path]) -> Resolution:
    """Locate an external executable."""
    res = Resolution(key=name)

    env_name = _env_key(f"{name}_bin")
    res.tried.append(f"${env_name}")
    env = os.environ.get(env_name)
    if env:
        p = Path(env).expanduser()
        if p.exists() and os.access(p, os.X_OK):
            res.path, res.via = p, f"${env_name}"
            return res

    cfg = load_config().get("binaries", {})
    res.tried.append(f"{CONFIG_PATH}:[binaries].{name}")
    if name in cfg:
        p = Path(str(cfg[name])).expanduser()
        if p.exists() and os.access(p, os.X_OK):
            res.path, res.via = p, "config.toml"
            return res

    res.tried.append("$PATH")
    on_path = shutil.which(name)
    if on_path:
        res.path, res.via = Path(on_path), "PATH"
        return res

    for candidate in probes:
        res.tried.append(str(candidate))
        if candidate.exists() and os.access(candidate, os.X_OK):
            res.path, res.via = candidate, "probe"
            return res
    return res


def resolve_file(name: str, probes: list[Path], config_section: str) -> Resolution:
    """Locate a data file (not an executable) through the usual search order."""
    res = Resolution(key=name)

    env_name = _env_key(name)
    res.tried.append(f"${env_name}")
    env = os.environ.get(env_name)
    if env:
        p = Path(env).expanduser()
        if p.exists():
            res.path, res.via = p, f"${env_name}"
            return res

    cfg = load_config().get(config_section, {})
    res.tried.append(f"{CONFIG_PATH}:[{config_section}].{name}")
    if name in cfg:
        p = Path(str(cfg[name])).expanduser()
        if p.exists():
            res.path, res.via = p, "config.toml"
            return res

    for candidate in probes:
        res.tried.append(str(candidate))
        if candidate.exists():
            res.path, res.via = candidate, "probe"
            return res
    return res


def resolve_molfile_to_params() -> Resolution:
    return resolve_file("molfile_to_params", _MOLFILE_TO_PARAMS_PROBE, "rosetta")


def resolve_rosetta_residue_types() -> Resolution:
    return resolve_file("rosetta_residue_types", _ROSETTA_RESIDUE_TYPES_PROBE, "rosetta")


def find_molfile_to_params() -> Path | None:
    return resolve_molfile_to_params().path


def find_rosetta_residue_types() -> Path | None:
    return resolve_rosetta_residue_types().path


def resolve_xtb() -> Resolution:
    return resolve_binary("xtb", _XTB_PROBE)


def resolve_crest() -> Resolution:
    return resolve_binary("crest", _CREST_PROBE)


def find_xtb_binary() -> str | None:
    r = resolve_xtb()
    return str(r.path) if r.path else None


def find_crest_binary() -> str | None:
    r = resolve_crest()
    return str(r.path) if r.path else None


def xtb_library_paths() -> list[str]:
    """Extra LD_LIBRARY_PATH entries the xtb binary may need.

    A conda-built xtb links against libraries in its own env's lib/ directory,
    which is not on the loader path when the binary is invoked by absolute path.
    """
    cfg = load_config().get("binaries", {})
    out: list[str] = []
    if "xtb_lib" in cfg:
        out.append(str(Path(str(cfg["xtb_lib"])).expanduser()))
    env = os.environ.get(_env_key("xtb_lib"))
    if env:
        out.append(env)
    binary = resolve_xtb().path
    if binary is not None:
        sibling = binary.parent.parent / "lib"
        if sibling.is_dir():
            out.append(str(sibling))
    for probe in _XTB_LIB_PROBE:
        if probe.is_dir():
            out.append(str(probe))
    seen: set[str] = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def aimnet2_model_name() -> str:
    """Which AIMNet2 checkpoint to request.

    AIMNet2 resolves its own weights by name and caches them, so this is a model
    identifier rather than a path. Overridable for pinning a specific variant.
    """
    return os.environ.get(_env_key("aimnet2_model"), "aimnet2")


def jsme_dir() -> Path:
    """Where the JSME sketcher lives once fetched."""
    env = os.environ.get(_env_key("jsme_dir"))
    if env:
        return Path(env).expanduser()
    cfg = load_config().get("sketch", {})
    if "jsme_dir" in cfg:
        return Path(str(cfg["jsme_dir"])).expanduser()
    return CACHE_DIR / "jsme"


def write_default_config(path: Path | None = None) -> Path:
    """Write a commented starter config the user can edit."""
    target = path or CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        """# ligand3d configuration
#
# Every entry is optional. Anything omitted falls back to environment
# variables, then $PATH, then the built-in probe locations, then ~/.cache.
# Run `ligand3d doctor` to see what was found and where it looked.

[weights]
# mace-off       = "/path/to/MACE-OFF23_medium.model"
# mace-off-small = "/path/to/MACE-OFF23_small.model"
# mace-off-large = "/path/to/MACE-OFF23_large.model"
# mace-mp        = "/path/to/MACE-matpes-r2scan-omat-ft.model"

[binaries]
# xtb     = "/path/to/xtb"
# crest   = "/path/to/crest"
# xtb_lib = "/path/to/env/lib"   # extra LD_LIBRARY_PATH for a conda-built xtb

[rosetta]
# molfile_to_params      = "/path/to/rosetta/.../public/molfile_to_params.py"
# rosetta_residue_types  = "/path/to/rosetta/.../fa_standard/residue_types.txt"

[sketch]
# jsme_dir = "/path/to/JSME"   # auto-downloaded if absent
"""
    )
    return target

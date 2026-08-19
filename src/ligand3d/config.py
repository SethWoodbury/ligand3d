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

# Filename patterns for each logical model, tried in order within each probe dir.
_WEIGHT_PATTERNS: dict[str, tuple[str, ...]] = {
    "mace-off": (
        "models--ACEsuit--mace-off-23/MACE-OFF23_medium.model",
        "mace_off/MACE-OFF23_medium.model",
        "MACE-OFF23_medium.model",
    ),
    "mace-off-small": (
        "models--ACEsuit--mace-off-23/MACE-OFF23_small.model",
        "mace_off/MACE-OFF23_small.model",
        "MACE-OFF23_small.model",
    ),
    "mace-off-large": (
        "models--ACEsuit--mace-off-23/MACE-OFF23_large.model",
        "mace_off/MACE-OFF23_large.model",
        "MACE-OFF23_large.model",
    ),
    "mace-mp": (
        "models--ACEsuit--mace-mp-0/MACE-matpes-r2scan-omat-ft.model",
        "mace_mp/MACE-matpes-r2scan-omat-ft.model",
        "MACE-matpes-r2scan-omat-ft.model",
    ),
}

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


def ketcher_dir() -> Path:
    """Where the Ketcher bundle lives once fetched."""
    env = os.environ.get(_env_key("ketcher_dir"))
    if env:
        return Path(env).expanduser()
    cfg = load_config().get("sketch", {})
    if "ketcher_dir" in cfg:
        return Path(str(cfg["ketcher_dir"])).expanduser()
    return CACHE_DIR / "ketcher"


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

[sketch]
# ketcher_dir = "/path/to/ketcher/standalone/build"
"""
    )
    return target

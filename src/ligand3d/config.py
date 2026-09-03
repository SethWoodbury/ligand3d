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
import re
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
    """Wall time for a full minimization of gabapentin (29 atoms) on CPU.

    See `measured`: some of these are timed on this machine and some are
    estimates for models that cannot run in this environment.
    """
    measured: bool = False
    """True if `speed` was timed here rather than extrapolated.

    Worth tracking separately. The first estimates in this table were scaled
    from model size and were wrong by up to seven-fold — MACE-POLAR medium was
    guessed at 8 s and actually takes 57 s, because the long-range electrostatics
    it adds are not free.
    """
    load_seconds: float = 0.0
    """One-off cost of building the calculator, separate from the minimization."""
    memory: str = ""
    accuracy: str = ""
    notes: str = ""

    reference: str = ""
    """The level of theory this method reproduces, which is its accuracy ceiling.

    A fitted method cannot be more right than what it was fitted to. Quoting the
    reference alongside the error is the only way an error bar means anything:
    "1 kcal/mol" against DFT and "1 kcal/mol" against CCSD(T) are very different
    claims.
    """
    error: str = ""
    """Reported error against that reference. Literature values, not measured here.

    Always in-domain — the held-out part of the training distribution. A molecule
    unlike anything in that distribution can be wrong by far more, with no
    warning, which is the failure mode that matters most for neural potentials.
    """

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
              repo="ACEsuit/mace-off-23", training="SPICE, neutral organics (wB97M-D3)", speed="6.8 s", measured=True, load_seconds=0.5, memory="~1 GB", accuracy="very good for neutral organics", notes="no total-charge input"),
    ModelSpec("mace-off-small", "mace", _hf("ACEsuit--mace-off-23", "MACE-OFF23_small.model"),
              "MACE-OFF23 small. Fastest of the OFF23 set.", organic_only=True, approx_mb=7,
              repo="ACEsuit/mace-off-23", training="SPICE, neutral organics", speed="2.0 s", measured=True, load_seconds=3.6, memory="~0.8 GB", accuracy="a little below medium", notes="fastest MACE-OFF"),
    ModelSpec("mace-off-large", "mace", _hf("ACEsuit--mace-off-23", "MACE-OFF23_large.model"),
              "MACE-OFF23 large. Most accurate of the OFF23 set.", organic_only=True, approx_mb=55,
              repo="ACEsuit/mace-off-23", training="SPICE, neutral organics", speed="31 s", measured=True, load_seconds=9.3, memory="~1.5 GB", accuracy="best of the OFF23 set", notes="slowest MACE-OFF"),
    ModelSpec("mace-off-24", "mace", _hf("ACEsuit--mace-off-24", "MACE-OFF24_medium.model"),
              "MACE-OFF24 medium. Successor to OFF23.", organic_only=True, approx_mb=18,
              repo="ACEsuit/mace-off-24", training="SPICE v2, wider coverage", speed="7.0 s", measured=True, load_seconds=0.5, memory="~1 GB", accuracy="successor to OFF23", notes="no total-charge input"),
    # --- MACE, charge-aware -----------------------------------------------
    ModelSpec("mace-omol", "mace",
              _hf("ACEsuit--mace-omol-0", "MACE-omol-0-extra-large-1024.model"),
              "MACE trained on OMol25. Consumes total charge.",
              takes_charge=True, approx_mb=422,
              repo="ACEsuit/mace-omol-0", training="OMol25 (charged and open-shell)", speed="28 s", measured=True, load_seconds=5.2, memory="~3 GB", accuracy="high; the charge-aware MACE", notes="extra-large 1024-channel; slow on CPU"),
    # --- MACE, multi-head -------------------------------------------------
    ModelSpec("mace-mh", "mace", _hf("ACEsuit--mace-mh-0", "mace-mh-0.model"),
              "MACE multi-head (omol head). Fast and general, ignores charge.",
              head="omol", approx_mb=40,
              repo="ACEsuit/mace-mh-0", training="multi-head: omol, SPICE, MatPES, OMat, OC20", speed="8.4 s", measured=True, load_seconds=0.4, memory="~1 GB", accuracy="good general purpose", notes="omol head; ignores charge despite the head name"),
    ModelSpec("mace-mh-1", "mace", _hf("ACEsuit--mace-mh-1", "mace-mh-1.model"),
              "MACE multi-head v1 (omol head).", head="omol", approx_mb=59,
              repo="ACEsuit/mace-mh-1", training="multi-head v1", speed="12 s", measured=True, load_seconds=1.5, memory="~1 GB", accuracy="good general purpose", notes="omol head selected"),
    ModelSpec("mace-mh-spice", "mace", _hf("ACEsuit--mace-mh-0", "mace-mh-0.model"),
              "MACE multi-head, SPICE wB97M head. Organic chemistry.",
              head="spice_wB97M", approx_mb=40,
              repo="ACEsuit/mace-mh-0", training="multi-head, SPICE wB97M head", speed="7.9 s", measured=True, load_seconds=0.5, memory="~1 GB", accuracy="organic chemistry", notes="spice_wB97M head selected"),
    # --- MACE, materials --------------------------------------------------
    ModelSpec("mace-mp", "mace",
              _hf("ACEsuit--mace-mp-0", "MACE-matpes-r2scan-omat-ft.model"),
              "MACE-MP-0 universal potential. Broad elements, ignores charge.",
              approx_mb=79,
              repo="ACEsuit/mace-mp-0", training="MatPES r2SCAN + OMat fine-tune", speed="7.8 s", measured=True, load_seconds=1.1, memory="~1 GB", accuracy="built for periodic solids, not molecules", notes="broadest elements; for inorganics"),
    # --- fairchem / OMol25, all charge- and spin-aware ---------------------
    ModelSpec("esen", "fairchem",
              _hf("facebook--esen-sm-conserving-all-omol", "esen_sm_conserving_all.pt"),
              "eSEN small, conserving. OMol25; excellent accuracy per second.",
              takes_charge=True, approx_mb=51,
              repo="facebook/esen-sm-conserving-all-omol", training="OMol25 (wB97M-V)", spin_aware=True, speed="7.9 s", measured=True, load_seconds=5.3, memory="~1 GB", accuracy="best accuracy per second here", notes="energy-conserving: forces are exact gradients"),
    ModelSpec("esen-sm-direct", "fairchem",
              _hf("facebook--esen-sm-direct-all-omol", "esen_sm_direct_all.pt"),
              "eSEN small, direct force prediction. Faster, not energy-conserving.",
              takes_charge=True, approx_mb=51,
              repo="facebook/esen-sm-direct-all-omol", training="OMol25", spin_aware=True, speed="2.8 s", measured=True, load_seconds=0.6, memory="~1 GB", accuracy="close to conserving, a little faster", notes="direct forces; not energy-conserving"),
    ModelSpec("esen-md-direct", "fairchem",
              _hf("facebook--esen-md-direct-all-omol", "esen_md_direct_all.pt"),
              "eSEN medium, direct.", takes_charge=True, approx_mb=406,
              repo="facebook/esen-md-direct-all-omol", training="OMol25", spin_aware=True, speed="16 s", measured=True, load_seconds=3.9, memory="~2 GB", accuracy="better than the small models", notes="direct forces"),
    ModelSpec("uma-s", "fairchem",
              _hf("facebook--fairchem-uma-s-1p1", "uma-s-1p1.pt"),
              "UMA small 1.1. Universal model; omol task.",
              takes_charge=True, approx_mb=1170,
              repo="facebook/fairchem-uma-s-1p1", training="UMA multi-domain, omol task", spin_aware=True, speed="15 s", measured=True, load_seconds=9.7, memory="~4 GB", accuracy="strong across chemistry and materials", notes="first load is slow"),
    ModelSpec("uma-s-1p2", "fairchem",
              _hf("facebook--fairchem-uma-s-1p2", "uma-s-1p2.pt"),
              "UMA small 1.2.", takes_charge=True, approx_mb=2330,
              repo="facebook/fairchem-uma-s-1p2", training="UMA 1.2", spin_aware=True, speed="36 s", measured=True, load_seconds=2.0, memory="~6 GB", accuracy="newer UMA small"),
    # Timed 2026-09-01 in the fairchem container, same molecule and machine as
    # the rest of the table: 35.2 s end to end over three runs (35.2/35.3/38.7).
    # load_seconds is carried over from uma-s-1p2 because the two were measured
    # back to back under identical conditions and came out equal — 31.1 s vs
    # 31.5 s on a stopwatch around make_calculator, which is a different scale
    # from what this column records but establishes that they load alike.
    ModelSpec("uma-s-1p2p1", "fairchem",
              _hf("facebook--fairchem-uma-s-1p2p1", "uma-s-1p2p1.pt"),
              "UMA small 1.2.1.", takes_charge=True, approx_mb=2330,
              repo="facebook/UMA", training="UMA 1.2.1", spin_aware=True,
              speed="35 s", measured=True, load_seconds=2.0, memory="~6 GB",
              accuracy="newest UMA small; FAIR recommends it over 1.2"),
    ModelSpec("uma-m", "fairchem",
              _hf("facebook--fairchem-uma-m-1p1", "uma-m-1p1.pt"),
              "UMA medium 1.1. Largest and slowest; needs real memory.",
              takes_charge=True, approx_mb=11170,
              repo="facebook/fairchem-uma-m-1p1", training="UMA medium", spin_aware=True, speed="63 s", measured=True, load_seconds=14.1, memory="~24 GB", accuracy="most accurate UMA", notes="11 GB of weights. Needs real memory: it was OOM-killed on a 31 GB workstation and ran fine on a node with 96 GB."),
    ModelSpec("allscaip", "fairchem",
              _hf("facebook--allscaip-omol102m-md-cons", "AllScAIP-OMol102M-md-cons.pt"),
              "AllScAIP OMol102M, conserving.", takes_charge=True, approx_mb=688,
              repo="facebook/allscaip-omol102m-md-cons", training="OMol25 102M", spin_aware=True, speed="23 s", measured=True, load_seconds=5.9, memory="~3 GB", accuracy="high", notes="energy-conserving"),
    # --- MACE-POLAR-1: public since 2026-02-23, ASL (academic use only) ----
    # Released through GitHub, not Hugging Face; the local directory keeps the
    # HF-style name only so it sits with the rest of the store.
    # Mainline mace-torch >= 0.3.16 carries the PolarMACE code, but the
    # checkpoints reference graph_longrange at unpickle time and mace does not
    # depend on it — hence `graph_electrostatics` alongside.
    ModelSpec("mace-polar-s", "mace-polar",
              _hf("ACEsuit--mace-polar-1", "MACE-POLAR-1-S.model"),
              "MACE-POLAR small. Explicit long-range electrostatics.",
              takes_charge=True, approx_mb=33, repo="ACEsuit/mace-polar-1",
              training="polarizable dataset with explicit long-range terms",
              speed="23 s", measured=True, load_seconds=0.2, memory="~1 GB",
              accuracy="models the long-range electrostatics other MACE models omit",
              notes="needs graph_electrostatics (MIT, GitHub only) for its graph_longrange module; the reciprocal-space electrostatics make it several times slower than plain MACE"),
    ModelSpec("mace-polar", "mace-polar",
              _hf("ACEsuit--mace-polar-1", "MACE-POLAR-1-M.model"),
              "MACE-POLAR medium. Explicit long-range electrostatics.",
              takes_charge=True, approx_mb=68, repo="ACEsuit/mace-polar-1",
              training="polarizable dataset with explicit long-range terms",
              speed="57 s", measured=True, load_seconds=0.2, memory="~1.5 GB", accuracy="the usual POLAR pick",
              notes="needs graph_electrostatics (MIT, GitHub only) for its graph_longrange module; the reciprocal-space electrostatics make it several times slower than plain MACE"),
    ModelSpec("mace-polar-l", "mace-polar",
              _hf("ACEsuit--mace-polar-1", "MACE-POLAR-1-L.model"),
              "MACE-POLAR large. Explicit long-range electrostatics.",
              takes_charge=True, approx_mb=130, repo="ACEsuit/mace-polar-1",
              training="polarizable dataset with explicit long-range terms",
              speed="114 s", measured=True, load_seconds=0.3, memory="~2 GB", accuracy="best of the POLAR set",
              notes="needs graph_electrostatics (MIT, GitHub only) for its graph_longrange module; the reciprocal-space electrostatics make it several times slower than plain MACE"),
    ModelSpec("allscaip-direct", "fairchem",
              _hf("facebook--allscaip-omol102m-md-d", "AllScAIP-OMol102M-md-d.pt"),
              "AllScAIP OMol102M, direct.", takes_charge=True, approx_mb=695,
              repo="facebook/allscaip-omol102m-md-d", training="OMol25 102M", spin_aware=True, speed="11 s", measured=True, load_seconds=6.2, memory="~3 GB", accuracy="high", notes="direct forces"),
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
#: Newest first. The IPD's lab install is a wrapper script that sets PATH and
#: LD_LIBRARY_PATH before exec'ing the real binary with an absolute path, which
#: is what a shared ORCA build and its OpenMPI need; the version it selects is
#: recorded in that tree's registry/installations.toml. The older
#: /net/software/orca/latest is ORCA 4.1.1 from 2019 and stays as a fallback,
#: so a machine with only that still works — with fewer methods, which the
#: per-method version gate reports rather than discovering at run time.
_ORCA_PROBE = [
    Path("/net/software/lab/quantum_chem/bin/orca"),
    Path("/net/software/orca/latest/orca"),
    Path("/software/orca/latest/orca"),
    CACHE_DIR / "orca" / "orca",
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


@lru_cache(maxsize=None)
def resolve_weights(key: str) -> Resolution:
    """Locate the weights file for a logical model name.

    Cached because it is not cheap and it is asked repeatedly. The probe
    directories are on a network filesystem, and resolving all twenty-odd
    models costs seconds when the attribute cache is cold — which is once per
    login, exactly when someone is opening the page and waiting.

    Checkpoints do not move mid-session. `resolve_weights.cache_clear()` is
    there if one ever does.
    """
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


def resolve_binary(name: str, probes: list[Path], validate=None) -> Resolution:
    """Locate an external executable.

    `validate` rejects a candidate that has the right name and is the wrong
    program, which is not hypothetical: `orca` on PATH here is the GNOME screen
    reader, and running a geometry through it would fail in a way that mentions
    neither chemistry nor accessibility.
    """
    res = Resolution(key=name)

    env_name = _env_key(f"{name}_bin")
    res.tried.append(f"${env_name}")
    env = os.environ.get(env_name)
    if env:
        p = Path(env).expanduser()
        if p.exists() and os.access(p, os.X_OK) and (validate is None or validate(p)):
            res.path, res.via = p, f"${env_name}"
            return res

    cfg = load_config().get("binaries", {})
    res.tried.append(f"{CONFIG_PATH}:[binaries].{name}")
    if name in cfg:
        p = Path(str(cfg[name])).expanduser()
        if p.exists() and os.access(p, os.X_OK) and (validate is None or validate(p)):
            res.path, res.via = p, "config.toml"
            return res

    res.tried.append("$PATH")
    on_path = shutil.which(name)
    if on_path and (validate is None or validate(Path(on_path))):
        res.path, res.via = Path(on_path), "PATH"
        return res

    for candidate in probes:
        res.tried.append(str(candidate))
        if (candidate.exists() and os.access(candidate, os.X_OK)
                and (validate is None or validate(candidate))):
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


#: g-xTB ships as its own patched xtb build — the stock binary has no --gxtb —
#: so it is resolved separately rather than assumed to be the xtb on PATH.
_GXTB_PROBE = [
    Path("/opt/gxtb/bin/xtb"),  # where the container puts it
    Path("/net/software/xtb/gxtb/bin/xtb"),
    Path.home() / ".local/share/ligand3d/gxtb/bin/xtb",
]


def resolve_gxtb() -> Resolution:
    return resolve_binary("gxtb", _GXTB_PROBE)


def resolve_crest() -> Resolution:
    return resolve_binary("crest", _CREST_PROBE)


#: Executables that only the quantum chemistry ORCA ships. ORCA 6 dropped
#: orca_scf, so more than one name is checked — a single name silently stopped
#: recognising a whole major version.
_ORCA_SIBLINGS = ("orca_scf", "orca_gtoint", "orca_scfgrad", "orca_main", "orca_cpscf")


def _is_quantum_orca(path: Path) -> bool:
    """True if this `orca` is the quantum chemistry program.

    GNOME ships a screen reader by the same name, and on this cluster it is the
    one on PATH. Telling them apart by execution means launching a screen
    reader; the quantum program instead ships a family of sibling executables,
    and a 13 KB Python script is not a compiled SCF driver.
    """
    try:
        if path.is_symlink():
            path = path.resolve()

        # A launcher script is the other legitimate shape. A shared-library
        # ORCA needs LD_LIBRARY_PATH set and its own absolute path passed for
        # parallel runs, so the sanctioned entry point at the IPD is a small
        # wrapper that does both — and a size check alone would reject it for
        # looking exactly like the screen reader. Read it instead: a wrapper
        # for the real thing names an ORCA tree that has the sibling binaries.
        if path.stat().st_size < 1_000_000:
            try:
                text = path.read_text(errors="replace")
            except OSError:
                return False
            if "orca" not in text.lower():
                return False
            for token in re.findall(r"/[\w./+-]+", text):
                candidate = Path(token)
                root = candidate if candidate.is_dir() else candidate.parent
                if any((root / sibling).exists() for sibling in _ORCA_SIBLINGS):
                    return True
            return False

        return any((path.parent / sibling).exists() for sibling in _ORCA_SIBLINGS)
    except OSError:
        return False


def resolve_orca() -> Resolution:
    return resolve_binary("orca", _ORCA_PROBE, validate=_is_quantum_orca)


def find_orca_binary() -> str | None:
    resolution = resolve_orca()
    return str(resolution.path) if resolution.path else None


def find_xtb_binary() -> str | None:
    r = resolve_xtb()
    return str(r.path) if r.path else None



def find_gxtb_binary() -> str | None:
    r = resolve_gxtb()
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

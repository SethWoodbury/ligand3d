"""DFT through ORCA.

This is the first backend that is not a force field or a fitted potential, and
it differs from them in ways the rest of the tool already has vocabulary for:
it consumes total charge *and* spin multiplicity, it has an implicit solvent
model, and it does not hold the bond list fixed. Declaring those in
`Capabilities` is all it takes for the pipeline to route it correctly — the
zwitterion check, the charge-channel refusal and the solvent handling come for
free.

What it does not share with the others is cost. MMFF94 is four milliseconds and
MACE-OFF is seven seconds; a DFT geometry optimisation is minutes to hours, and
scales steeply with size. So the default here is a *composite* method rather
than a functional and basis set chosen separately: `B97-3c` is built for
geometries at a fraction of the cost of a large-basis hybrid, and it is what
you want unless you have a specific reason otherwise.

The realistic use is as the last link in a chain — `mmff94,gfn2,orca` searches
cheaply, narrows, and spends the expensive method only on what survived.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..errors import BackendUnavailable
from .ase_bridge import ASEBackend
from .base import Availability, Capabilities, MinimizeJob, register

#: Composite methods, which pair a functional, a basis and the corrections that
#: make the pairing behave. Cheaper than assembling the parts yourself and
#: harder to get wrong.
COMPOSITE_METHODS = ("HF-3c", "PBEh-3c", "B97-3c", "r2SCAN-3c")

DEFAULT_METHOD = "B97-3c"


#: ORCA's CPCM knows these by name. Not the same list as ALPB, so the two are
#: kept apart rather than pretending one mapping covers both.
CPCM_SOLVENTS = frozenset({
    "water", "acetone", "acetonitrile", "ammonia", "benzene", "chloroform",
    "ch2cl2", "dichloromethane", "dmf", "dmso", "ethanol", "hexane",
    "methanol", "octanol", "pyridine", "thf", "toluene",
})


@dataclass(frozen=True)
class Method:
    """One level of theory, exposed under its own name.

    "DFT" is not a method, and a result labelled that way cannot be written up
    or reproduced: the functional, the basis and the dispersion correction are
    the answer. Each entry here names all three, so the run, the trace and the
    model table say what was actually done.
    """

    alias: str
    keywords: str
    """What goes on ORCA's keyword line, minus ENGRAD and solvation."""
    summary: str
    rung: str
    kind: str = "dft"
    """Almost all of these are DFT. HF-3c is not: it is Hartree-Fock, with no
    functional at all, and calling it DFT would be simply wrong."""
    """Where it sits: composite, GGA, meta-GGA, hybrid, range-separated."""
    min_orca: tuple[int, int] = (4, 1)
    """Oldest ORCA that knows these keywords.

    Every entry was checked by running it, against both ORCAs on this cluster:
    4.1.1 from 2019, and 6.1.1. r2SCAN, r2SCAN-3c, D4, wB97X-3c and M06-2X all
    fail on 4.1.1 and pass on 6.1.1. Without this the failure arrives as
    `exit status 4`, with nothing naming the cause, minutes into a job.

    The bounds are conservative: they say "verified to work at or above this",
    not "provably fails below it". A method may well run on an ORCA between the
    two, and pointing LIGAND3D_ORCA_BIN at it is the way to find out.
    """


#: Ordered cheapest first. Composites lead because they are what you want for a
#: geometry unless you have a specific reason otherwise: a functional, a basis
#: and the corrections that make the pairing behave, tuned together and far
#: cheaper than assembling the parts by hand.
METHODS: tuple[Method, ...] = (
    Method("orca-hf3c", "HF-3c",
           "HF-3c: Hartree-Fock, minimal basis, three corrections. Not DFT — "
           "there is no functional. The cheapest thing here: a sanity geometry, "
           "not energetics.", "composite", kind="hf"),
    Method("orca-pbeh3c", "PBEh-3c",
           "PBEh-3c: hybrid GGA composite on def2-mSVP. Good geometries, modest "
           "cost.", "composite"),
    Method("orca-b973c", "B97-3c",
           "B97-3c: GGA composite on a triple-zeta basis, built for geometries. "
           "The default, and the right first choice.", "composite"),
    Method("orca-r2scan3c", "r2SCAN-3c",
           "r2SCAN-3c: meta-GGA composite. Better than B97-3c on non-covalent "
           "interactions and barriers.", "composite", min_orca=(5, 0)),
    Method("orca-bp86", "BP86 def2-SVP D3BJ",
           "BP86/def2-SVP with D3(BJ). A plain GGA — fast, and the reference "
           "point for what a composite buys you.", "GGA"),
    Method("orca-tpss", "TPSS def2-TZVP D3BJ",
           "TPSS/def2-TZVP with D3(BJ). Meta-GGA, a step up from the composites "
           "when geometry matters more than wall clock.", "meta-GGA"),
    Method("orca-b3lyp", "B3LYP def2-TZVP D3BJ",
           "B3LYP/def2-TZVP with D3(BJ). Not the best hybrid here, but the one "
           "most published organic geometries were computed with.", "hybrid"),
    Method("orca-pbe0", "PBE0 def2-TZVP D3BJ",
           "PBE0/def2-TZVP with D3(BJ). A parameter-free hybrid, generally "
           "better behaved than B3LYP.", "hybrid"),
    Method("orca-wb97x", "wB97X-D3 def2-TZVP",
           "wB97X-D3/def2-TZVP. Range-separated hybrid, accurate for organic "
           "molecules and priced accordingly. Runs on any ORCA here.",
           "range-separated hybrid"),
    # --- everything below needs an ORCA newer than 2019 -------------------
    Method("orca-wb97x3c", "wB97X-3c",
           "wB97X-3c: a range-separated hybrid composite on its own vDZP basis. "
           "The best geometry-per-second on offer. Note its basis carries "
           "effective core potentials, so its total energies sit on a different "
           "scale from every all-electron method here (methanol: -24 Eh, not "
           "-115.6) and are not comparable across methods. Conformer ranking "
           "within a run is unaffected.",
           "composite", min_orca=(6, 0)),
    Method("orca-r2scan", "r2SCAN def2-TZVP D4",
           "r2SCAN/def2-TZVP with D4 dispersion. A modern meta-GGA; D4 is a "
           "real improvement over D3 for charged and polar systems.",
           "meta-GGA", min_orca=(5, 0)),
    Method("orca-wb97xd4", "wB97X-D4 def2-TZVP",
           "wB97X-D4/def2-TZVP. The wB97X above, with the newer dispersion "
           "correction.", "range-separated hybrid", min_orca=(5, 0)),
    Method("orca-wb97mv", "wB97M-V def2-TZVPD",
           "wB97M-V/def2-TZVPD. Among the most accurate functionals in routine "
           "use for organic chemistry, and the level g-xTB is fit to reproduce "
           "— so it is the reference to check g-xTB against.",
           "range-separated hybrid", min_orca=(5, 0)),
    Method("orca-m062x", "M062X def2-TZVP",
           "M06-2X/def2-TZVP. Heavily parameterised, still widely used for "
           "thermochemistry and barriers. Note ORCA spells it M062X.",
           "hybrid meta-GGA", min_orca=(5, 0)),
)

METHODS_BY_ALIAS = {m.alias: m for m in METHODS}


#: The IPD's quantum-chemistry tree records what it installed, and whether the
#: install was actually verified. Reading that is both faster and better
#: evidence than re-deriving the version from a banner.
ORCA_REGISTRY = Path("/net/software/lab/quantum_chem/registry/installations.toml")


def _version_from_registry(binary: Path) -> tuple[int, int] | None:
    """The recorded version for this binary, if a registry describes it.

    Only trusted when the resolved binary actually lives in the tree the
    registry describes — otherwise a registry on one path would be asserting
    the version of an ORCA somewhere else entirely.
    """
    if not ORCA_REGISTRY.exists():
        return None
    try:
        import tomllib

        data = tomllib.loads(ORCA_REGISTRY.read_text())
    except Exception:
        return None

    resolved = str(binary.resolve())
    root = str(ORCA_REGISTRY.parent.parent)
    if not resolved.startswith(root):
        return None

    best: tuple[int, int] | None = None
    for entry in (data.get("orca") or {}).values():
        version = str(entry.get("version", ""))
        found = re.match(r"(\d+)\.(\d+)", version)
        if not found:
            continue
        # A recorded install that never passed its own smoke test is not
        # evidence of anything; fall through to asking the binary.
        if entry.get("serial_smoke_test") not in (None, "passed"):
            continue
        pair = (int(found.group(1)), int(found.group(2)))
        if best is None or pair > best:
            best = pair
    return best


@lru_cache(maxsize=1)
def orca_version() -> tuple[int, int] | None:
    """The installed ORCA's version, or None if it cannot be determined.

    Prefers the site registry, which is authoritative and free. Falls back to
    running ORCA on a one-atom input and reading its banner, since ORCA does
    not answer --version. Cached either way: it never changes within a run.
    """
    from ..config import resolve_orca

    resolution = resolve_orca()
    if not resolution.found:
        return None

    recorded = _version_from_registry(Path(resolution.path))
    if recorded is not None:
        return recorded
    with tempfile.TemporaryDirectory(prefix="ligand3d-orca-v") as scratch:
        probe = Path(scratch) / "v.inp"
        probe.write_text("! HF\n* xyz 0 1\nH 0 0 0\n*\n")
        try:
            out = subprocess.run(
                [str(resolution.path), str(probe)],
                capture_output=True, text=True, timeout=120, cwd=scratch,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
    found = re.search(r"Program Version\s+(\d+)\.(\d+)", out)
    return (int(found.group(1)), int(found.group(2))) if found else None


@dataclass
class OrcaOptions:
    """What to put on ORCA's keyword line."""

    method: str = DEFAULT_METHOD
    basis: str = ""
    """Left empty for a composite method, which brings its own."""
    threads: int = 1
    extra: str = ""

    def simple_input(self, solvent: str | None) -> str:
        parts = [self.method]
        if self.basis:
            parts.append(self.basis)
        # ORCA's own optimiser is not used: ligand3d drives L-BFGS through ASE
        # so that every backend produces the same trace and the same
        # convergence criterion. ENGRAD is one gradient per step, which is
        # exactly what the optimiser asks for.
        parts.append("ENGRAD")
        if solvent:
            parts.append(f"CPCM({_cpcm_name(solvent)})")
        if self.extra:
            parts.append(self.extra)
        return " ".join(parts)

    def blocks(self) -> str:
        return f"%pal nprocs {self.threads} end" if self.threads > 1 else ""


def _is_quantum_orca_available() -> bool:
    """True if the quantum chemistry ORCA is resolvable here."""
    from ..config import resolve_orca

    return resolve_orca().found


def _cpcm_name(solvent: str) -> str:
    alias = {"ch2cl2": "dichloromethane", "h2o": "water", "chcl3": "chloroform"}
    return alias.get(solvent.lower(), solvent.lower())


class OrcaBackend(ASEBackend):
    """DFT geometry optimisation driven through ORCA's gradients.

    One subclass per level of theory, so that what ran is in the name rather
    than in an environment variable somebody has to remember to record.
    """

    #: Overridden per method; the bare class keeps the historical default.
    method: Method = METHODS_BY_ALIAS["orca-b973c"]

    caps = Capabilities(
        name="orca",
        kind="dft",
        description=(
            f"DFT via ORCA, {DEFAULT_METHOD} by default. Charge- and spin-aware, "
            "with CPCM implicit solvent. Minutes to hours, not seconds."
        ),
        takes_charge=True,
        supports_solvation=True,
        spin_aware=True,
        fixed_topology=False,
        # A total electronic energy, like the semi-empirical and neural
        # tiers — not a strain energy. Only differences between
        # conformers of the same molecule mean anything.
        energy_kind="total",
        requires=("ase",),
    )

    def __init__(self) -> None:
        self._scratch_dirs: list[str] = []
        atexit.register(self._clean_up)

    def _clean_up(self) -> None:
        for path in self._scratch_dirs:
            shutil.rmtree(path, ignore_errors=True)
        self._scratch_dirs.clear()

    def install_hint(self) -> str:
        return (
            "ORCA is a separate download from https://orcaforum.kofo.mpg.de (free "
            "for academic use). Point LIGAND3D_ORCA_BIN at the binary, or put it "
            "under [binaries] in ~/.config/ligand3d/config.toml.\n"
            "On this cluster it is already at /net/software/orca/latest/orca."
        )

    def extra_availability(self) -> Availability | None:
        from ..config import resolve_orca

        resolution = resolve_orca()
        if not resolution.found:
            return Availability(
                ok=False,
                reason="the ORCA binary was not found",
                hint=self.install_hint(),
            )

        # An unknown keyword makes ORCA exit 4 with no explanation the caller
        # can act on, several minutes into a job. Better to refuse up front and
        # name the reason.
        have = orca_version()
        if have is not None and have < self.method.min_orca:
            want = ".".join(str(n) for n in self.method.min_orca)
            return Availability(
                ok=False,
                reason=(
                    f"{self.method.keywords} needs ORCA {want}+; this one is "
                    f"{have[0]}.{have[1]}"
                ),
                hint=(
                    "Use a composite that this version knows — orca-b973c, "
                    "orca-pbeh3c or orca-hf3c — or point LIGAND3D_ORCA_BIN at a "
                    "newer ORCA."
                ),
            )
        return None

    def make_calculator(self, job: MinimizeJob):
        from ase.calculators.orca import ORCA, OrcaProfile

        from ..config import resolve_orca

        resolution = resolve_orca()
        if not resolution.found:  # pragma: no cover - guarded by availability
            raise BackendUnavailable(self.install_hint())

        options = _options_from(job, self.method)
        if job.solvent and _cpcm_name(job.solvent) not in CPCM_SOLVENTS:
            raise BackendUnavailable(
                f"ORCA's CPCM has no entry for {job.solvent!r}. Available: "
                + ", ".join(sorted(CPCM_SOLVENTS))
            )

        # ORCA scatters a dozen files per gradient call. They go somewhere
        # temporary rather than into whatever directory the command was run
        # from, which would otherwise fill with .gbw and .engrad debris.
        scratch = tempfile.mkdtemp(prefix="ligand3d-orca-")
        self._scratch_dirs.append(scratch)

        return ORCA(
            profile=OrcaProfile(command=str(resolution.path)),
            directory=scratch,
            charge=job.charge,
            mult=job.multiplicity,
            orcasimpleinput=options.simple_input(job.solvent),
            orcablocks=options.blocks(),
        )


def _options_from(job: MinimizeJob, method: Method | None = None) -> OrcaOptions:
    """Build the keyword line for this job.

    The method normally comes from which backend was asked for, so the name of
    the run records the level of theory. LIGAND3D_ORCA_METHOD still overrides
    it, for a functional that has no alias here.
    """
    keywords = (method or OrcaBackend.method).keywords
    return OrcaOptions(
        method=os.environ.get("LIGAND3D_ORCA_METHOD", keywords),
        basis=os.environ.get("LIGAND3D_ORCA_BASIS", ""),
        threads=max(1, getattr(job, "n_threads", 1) or 1),
        extra=os.environ.get("LIGAND3D_ORCA_EXTRA", ""),
    )


def _make_backend(method: Method, name: str | None = None) -> type[OrcaBackend]:
    """One registered backend per level of theory.

    `name` lets the bare `orca` alias keep its own identity rather than
    reporting itself as orca-b973c, which would give the catalog a duplicate
    entry and no entry for `orca` at all.
    """
    return type(
        f"Orca_{(name or method.alias).replace('-', '_')}",
        (OrcaBackend,),
        {
            "method": method,
            "caps": Capabilities(
                name=name or method.alias,
                kind=method.kind,
                description=f"{method.summary} Charge- and spin-aware, CPCM solvent.",
                takes_charge=True,
                supports_solvation=True,
                spin_aware=True,
                fixed_topology=False,
                energy_kind="total",
                requires=("ase",),
            ),
        },
    )


for _method in METHODS:
    register(_method.alias, _make_backend(_method))

#: `orca` stays the name of the sensible default rather than becoming ambiguous.
register("orca", _make_backend(METHODS_BY_ALIAS["orca-b973c"], name="orca"))

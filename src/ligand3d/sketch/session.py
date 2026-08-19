"""Build jobs for the browser session.

The command line runs one molecule and exits. The sketcher is a session: you
draw, build, read the log, clear, and draw the next one without reloading. That
needs three things the CLI does not: a job that runs on a background thread so
the page can poll it, a log the page can display, and target-path inspection so
the page can warn before it overwrites anything.
"""

from __future__ import annotations

import itertools
import os
import re
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import Ligand3DError

_JOB_IDS = itertools.count(1)


@dataclass
class TargetInfo:
    """What a chosen directory and filename resolve to, and what it would cost."""

    directory: str
    filename: str
    pdb: str
    sdf: str
    directory_exists: bool
    will_create_directory: bool
    writable: bool
    existing: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def would_overwrite(self) -> bool:
        return bool(self.existing)

    def to_json(self) -> dict[str, Any]:
        return {
            "directory": self.directory,
            "filename": self.filename,
            "pdb": self.pdb,
            "sdf": self.sdf,
            "directory_exists": self.directory_exists,
            "will_create_directory": self.will_create_directory,
            "writable": self.writable,
            "existing": self.existing,
            "would_overwrite": self.would_overwrite,
            "error": self.error,
        }


def _normalize_filename(filename: str) -> str:
    """Force a .pdb suffix and strip anything that is not a bare filename."""
    name = Path(filename.strip()).name or "sketch0.pdb"
    if not name.lower().endswith(".pdb"):
        name = f"{Path(name).stem or 'sketch0'}.pdb"
    return name


def inspect_target(directory: str, filename: str, write_sdf: bool = True) -> TargetInfo:
    """Resolve a target and report whether writing there is safe.

    Deliberately does not create anything: the page shows this to the user first
    and only then asks to proceed.
    """
    name = _normalize_filename(filename)
    try:
        base = Path(directory.strip() or ".").expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        return TargetInfo(
            directory=directory, filename=name, pdb="", sdf="",
            directory_exists=False, will_create_directory=False, writable=False,
            error=f"cannot resolve that path: {exc}",
        )

    pdb = base / name
    sdf = pdb.with_suffix(".sdf")

    exists = base.is_dir()
    # Walk up to the nearest existing ancestor to judge whether we could create
    # the directory, and whether that ancestor is writable at all.
    probe = base
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    writable = os.access(probe, os.W_OK) if probe.exists() else False

    info = TargetInfo(
        directory=str(base),
        filename=name,
        pdb=str(pdb),
        sdf=str(sdf),
        directory_exists=exists,
        will_create_directory=not exists,
        writable=writable,
        existing=[str(p) for p in ((pdb, sdf) if write_sdf else (pdb,)) if p.exists()],
    )
    if base.exists() and not base.is_dir():
        info.error = f"{base} exists but is not a directory"
    elif not writable:
        info.error = f"{probe} is not writable"
    return info


def next_filename(directory: str, stem: str = "sketch") -> str:
    """First unused `sketch0.pdb`, `sketch1.pdb`, ... in the directory.

    Only used to seed the field; the user can always type something else, and
    the overwrite check still runs on whatever they end up with.
    """
    try:
        base = Path(directory.strip() or ".").expanduser().resolve()
    except (OSError, RuntimeError):
        return f"{stem}0.pdb"
    if not base.is_dir():
        return f"{stem}0.pdb"

    used = set()
    pattern = re.compile(rf"^{re.escape(stem)}(\d+)\.pdb$", re.IGNORECASE)
    for entry in base.iterdir():
        match = pattern.match(entry.name)
        if match:
            used.add(int(match.group(1)))
    n = 0
    while n in used:
        n += 1
    return f"{stem}{n}.pdb"


@dataclass
class Job:
    """One build, its log, and its result."""

    id: int
    state: str = "queued"  # queued | running | done | error
    log: list[dict[str, str]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def say(self, text: str, level: str = "info") -> None:
        with self._lock:
            self.log.append({"level": level, "text": text})

    def to_json(self) -> dict[str, Any]:
        with self._lock:
            return {
                "id": self.id,
                "state": self.state,
                "log": list(self.log),
                "result": self.result,
                "error": self.error,
            }


class JobStore:
    """Keeps the last few jobs so the page can poll them."""

    def __init__(self, keep: int = 40) -> None:
        self._jobs: dict[int, Job] = {}
        self._order: list[int] = []
        self._keep = keep
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(id=next(_JOB_IDS))
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > self._keep:
                self._jobs.pop(self._order.pop(0), None)
        return job

    def get(self, job_id: int) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


def describe_stereo(molecule) -> list[str]:
    """Human-readable lines about the stereochemistry that was drawn."""
    audit = molecule.stereo
    lines: list[str] = []

    if audit.assigned_centers:
        detail = ", ".join(f"atom {i} = {code}" for i, code in audit.assigned_centers)
        lines.append(
            f"{len(audit.assigned_centers)} stereocenter(s) defined: {detail}"
        )
    else:
        lines.append("no stereocenters defined")

    if audit.assigned_bonds:
        detail = ", ".join(f"bond {a}-{b} = {code}" for a, b, code in audit.assigned_bonds)
        lines.append(f"{len(audit.assigned_bonds)} double bond(s) with geometry: {detail}")

    if audit.unassigned_centers:
        from ..molecule import has_real_stereo_ambiguity

        if has_real_stereo_ambiguity(molecule):
            lines.append(
                f"{len(audit.unassigned_centers)} stereocenter(s) left undefined: "
                + ", ".join(f"atom {i}" for i in audit.unassigned_centers)
                + " — this is ambiguous and will be refused unless you change the "
                "stereo policy"
            )
        else:
            lines.append(
                f"{len(audit.unassigned_centers)} atom(s) look stereogenic but are "
                "fixed by the ring system, so nothing is ambiguous"
            )
    return lines


def run_job(
    job: Job,
    molblock: str,
    settings_json: dict[str, Any],
    target: TargetInfo,
) -> None:
    """Execute one build, narrating into the job log. Never raises."""
    from ..molecule import from_molblock
    from ..pipeline import Settings, run

    job.state = "running"
    try:
        job.say(f"reading the drawn structure ({len(molblock.splitlines())} molblock lines)")
        molecule = from_molblock(molblock)
        job.say(f"formula {molecule.formula}, SMILES {molecule.smiles}")

        n_frags = molecule.n_fragments
        if n_frags > 1:
            job.say(
                f"the drawing contains {n_frags} separate fragments: "
                + " + ".join(molecule.fragments),
                "warn",
            )

        for line in describe_stereo(molecule):
            job.say(line)

        settings = _settings_from_json(settings_json)
        job.say(
            f"backend {settings.backend}"
            + (f", pH {settings.ph:g}" if settings.ph is not None else ", protonation as drawn")
            + (f", {settings.n_confs} conformers via {settings.conf_method}"
               if settings.n_confs > 1 else ", single conformer")
        )

        Path(target.directory).mkdir(parents=True, exist_ok=True)
        if target.will_create_directory:
            job.say(f"created directory {target.directory}")

        outcomes = run(molecule, settings, output=Path(target.pdb))

        outputs = []
        for outcome in outcomes:
            for note in outcome.notes:
                job.say(note)
            energy = outcome.best_energy
            if energy is not None:
                record = outcome.records[0]
                kind = "total electronic" if record.energy_kind == "total" else "strain"
                job.say(
                    f"lowest {kind} energy {energy:.4f} {record.energy_unit} "
                    f"({record.backend}), {len(outcome.records)} conformer(s) kept"
                )
            job.say(f"wrote {outcome.pdb_path}", "ok")
            outputs.append(str(outcome.pdb_path))
            if outcome.sdf_path:
                job.say(f"wrote {outcome.sdf_path}", "ok")
                outputs.append(str(outcome.sdf_path))

        job.result = {
            "smiles": molecule.smiles,
            "formula": molecule.formula,
            "outputs": outputs,
            "n_conformers": sum(len(o.records) for o in outcomes),
            "stereocenters": [
                {"atom": i, "code": code} for i, code in molecule.stereo.assigned_centers
            ],
        }
        job.state = "done"

    except Ligand3DError as exc:
        # An expected refusal: undefined stereo, two ligands, a bad backend
        # pairing. The message is written for a human, so show it as-is.
        job.say(str(exc), "error")
        job.error = str(exc)
        job.state = "error"
    except Exception as exc:  # a genuine bug; keep the traceback for the log
        job.say(f"unexpected {type(exc).__name__}: {exc}", "error")
        job.say(traceback.format_exc().strip().splitlines()[-1], "error")
        job.error = f"{type(exc).__name__}: {exc}"
        job.state = "error"


def _settings_from_json(data: dict[str, Any]):
    """Translate the page's form values into pipeline Settings."""
    from ..pipeline import Settings
    from ..protonate import DEFAULT_PH

    def num(key, default, cast=float):
        value = data.get(key, default)
        if value in (None, "", "null"):
            return default
        try:
            return cast(value)
        except (TypeError, ValueError):
            return default

    mode = str(data.get("protonation", "as-drawn"))
    ph = None
    enumerate_states = False
    if mode == "ph":
        ph = num("ph", DEFAULT_PH)
    elif mode == "enumerate":
        ph = num("ph", DEFAULT_PH)
        enumerate_states = True

    solvent = (data.get("solvent") or "").strip() or None

    return Settings(
        backend=str(data.get("backend") or "mmff94"),
        n_confs=int(num("n_confs", 1, int)),
        conf_method=str(data.get("conf_method") or "rdkit"),
        prune_rms=num("prune_rms", 0.5),
        energy_window=(num("energy_window", None) if data.get("energy_window") else None),
        ph=ph,
        enumerate_states=enumerate_states,
        stereo_mode=str(data.get("stereo_mode") or "require"),
        largest_fragment=bool(data.get("largest_fragment")),
        solvent=solvent,
        auto_solvent=bool(data.get("auto_solvent", True)),
        allow_charge_mismatch=bool(data.get("allow_charge_mismatch")),
        allow_proton_transfer=bool(data.get("allow_proton_transfer")),
        max_steps=int(num("max_steps", 500, int)),
        seed=int(num("seed", 0xF00D, int)),
        n_threads=int(num("threads", 1, int)),
        resname=(data.get("resname") or None),
        write_sdf=bool(data.get("write_sdf", True)),
    )


def backend_catalog() -> list[dict[str, Any]]:
    """Every registered backend with its capabilities and whether it can run."""
    from ..minimize import all_backends

    catalog = []
    for backend in all_backends():
        caps = backend.caps
        try:
            availability = backend.available()
            ready, reason, hint = bool(availability), availability.reason, availability.hint
        except Exception as exc:
            ready, reason, hint = False, f"{type(exc).__name__}: {exc}", ""
        catalog.append(
            {
                "id": caps.name,
                "kind": caps.kind,
                "description": caps.description,
                "takes_charge": caps.takes_charge,
                "supports_solvation": caps.supports_solvation,
                "ready": ready,
                "reason": reason,
                "hint": hint,
            }
        )
    return catalog

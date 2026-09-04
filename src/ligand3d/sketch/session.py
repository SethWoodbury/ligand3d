"""Build jobs for the browser session.

The command line runs one molecule and exits. The sketcher is a session: you
draw, build, read the log, clear, and draw the next one without reloading. That
needs three things the CLI does not: a job that runs on a background thread so
the page can poll it, a log the page can display, and target-path inspection so
the page can warn before it overwrites anything.
"""

from __future__ import annotations

import itertools
import json
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
    """What a chosen directory and base name resolve to, and what it would cost."""

    directory: str
    stem: str
    formats: tuple[str, ...]
    will_write: list[str] = field(default_factory=list)
    directory_exists: bool = True
    will_create_directory: bool = False
    writable: bool = True
    existing: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def would_overwrite(self) -> bool:
        return bool(self.existing)

    @property
    def primary(self) -> str:
        return self.will_write[0] if self.will_write else ""

    def to_json(self) -> dict[str, Any]:
        return {
            "directory": self.directory,
            "stem": self.stem,
            "formats": list(self.formats),
            "will_write": self.will_write,
            "directory_exists": self.directory_exists,
            "will_create_directory": self.will_create_directory,
            "writable": self.writable,
            "existing": self.existing,
            "would_overwrite": self.would_overwrite,
            "error": self.error,
        }


VALID_FORMATS = ("cif", "pdb", "sdf", "annotated")

#: File suffix per format. The annotated CIF keeps `.cif` so anything that
#: reads mmCIF still opens it — it *is* an mmCIF — while the double extension
#: says at a glance which of the two files carries the conditioning.
FORMAT_SUFFIX = {
    "cif": "cif", "pdb": "pdb", "sdf": "sdf", "annotated": "annotated.cif",
}


def normalize_formats(formats, default=("cif", "sdf")) -> tuple[str, ...]:
    """Coerce whatever the page sent into a valid, ordered format tuple.

    `formats=[]` is meaningful — it is the dry-run request — so an explicitly
    empty list is honoured, while a missing key falls back to the default.
    """
    if formats is None:
        return tuple(default)
    if isinstance(formats, str):
        formats = [f.strip() for f in formats.split(",")]
    cleaned = [str(f).strip().lower() for f in formats if str(f).strip()]
    cleaned = ["cif" if f == "mmcif" else f for f in cleaned]
    return tuple(f for f in dict.fromkeys(cleaned) if f in VALID_FORMATS)


def _normalize_stem(filename: str) -> str:
    """Reduce whatever was typed to a bare base name with no extension.

    The page asks for a base name because one build can write several files.
    Path components are stripped so a name can never escape the chosen
    directory, and a known suffix is dropped so typing "lig.cif" does not
    produce "lig.cif.cif".
    """
    name = Path(str(filename).strip()).name
    # ".annotated.cif" is two suffixes, so strip that pair before the single ones.
    for suffix in sorted(FORMAT_SUFFIX.values(), key=len, reverse=True):
        if name.lower().endswith("." + suffix):
            name = name[: -(len(suffix) + 1)]
            break
    stem = name
    stem = stem.strip() or "sketch0"
    return stem


def inspect_target(directory: str, filename: str, formats=None) -> TargetInfo:
    """Resolve a target and report whether writing there is safe.

    Deliberately creates nothing: the page shows this first and only then asks
    to proceed.
    """
    chosen = normalize_formats(formats)
    stem = _normalize_stem(filename)
    try:
        base = Path(str(directory).strip() or ".").expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        return TargetInfo(
            directory=str(directory), stem=stem, formats=chosen,
            directory_exists=False, will_create_directory=False, writable=False,
            error=f"cannot resolve that path: {exc}",
        )

    targets = [base / f"{stem}.{FORMAT_SUFFIX[fmt]}" for fmt in chosen]
    exists = base.is_dir()

    # Walk up to the nearest existing ancestor to judge whether the directory
    # could be created, and whether that ancestor is writable at all.
    probe = base
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    writable = os.access(probe, os.W_OK) if probe.exists() else False

    info = TargetInfo(
        directory=str(base),
        stem=stem,
        formats=chosen,
        will_write=[str(p) for p in targets],
        directory_exists=exists,
        will_create_directory=not exists,
        writable=writable,
        existing=[str(p) for p in targets if p.exists()],
    )
    if base.exists() and not base.is_dir():
        info.error = f"{base} exists but is not a directory"
    elif not writable:
        info.error = f"{probe} is not writable"
    return info


def next_filename(directory: str, stem: str = "sketch") -> str:
    """First unused `sketch0`, `sketch1`, ... base name in the directory.

    Considers every extension ligand3d can write, so `sketch0` is skipped if a
    `sketch0.cif` exists even when only `.pdb` is currently selected.
    """
    try:
        base = Path(str(directory).strip() or ".").expanduser().resolve()
    except (OSError, RuntimeError):
        return f"{stem}0"
    if not base.is_dir():
        return f"{stem}0"

    used: set[int] = set()
    pattern = re.compile(
        rf"^{re.escape(stem)}(\d+)\.(?:{'|'.join(re.escape(s) for s in FORMAT_SUFFIX.values())})$",
        re.IGNORECASE,
    )
    for entry in base.iterdir():
        match = pattern.match(entry.name)
        if match:
            used.add(int(match.group(1)))
    n = 0
    while n in used:
        n += 1
    return f"{stem}{n}"


@dataclass
class Job:
    """One build, its log, and its result."""

    id: int
    state: str = "queued"  # queued | running | done | error
    log: list[dict[str, str]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    slurm_options: dict[str, Any] | None = None
    """Set when the page asked for the build to run on a compute node."""
    slurm_job_id: int | None = None
    in_container: bool = False
    """Set when the page asked for the build to run in an Apptainer image."""
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
                "slurm_job_id": self.slurm_job_id,
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
        from ..molecule import describe_double_bonds

        reports = describe_double_bonds(molecule)
        lines.append(f"{len(reports)} double bond(s) with defined geometry:")
        for report in reports:
            if report.cis_trans:
                lines.append(
                    f"    bond {report.begin}-{report.end} = {report.cip} "
                    f"({report.cis_trans})"
                )
            else:
                lines.append(
                    f"    bond {report.begin}-{report.end} = {report.cip} "
                    f"(cis/trans does not apply: {report._why} alkene)"
                )

    if audit.unassigned_centers:
        from ..molecule import (
            describe_resonance_centers,
            has_real_stereo_ambiguity,
            resonance_averaged_centers,
        )

        averaged = resonance_averaged_centers(molecule)
        if averaged:
            lines.append(describe_resonance_centers(molecule))

        remaining = [i for i in audit.unassigned_centers if i not in set(averaged)]
        if remaining and has_real_stereo_ambiguity(molecule):
            lines.append(
                f"{len(remaining)} stereocenter(s) left undefined: "
                + ", ".join(f"atom {i}" for i in remaining)
                + " — this is ambiguous and will be refused unless you change the "
                "stereo policy"
            )
        elif remaining:
            lines.append(
                f"{len(remaining)} atom(s) look stereogenic but are "
                "fixed by the ring system, so nothing is ambiguous"
            )
    return lines


def summarize_trace(trace: list) -> list[str]:
    """One line per method in the chain: steps taken and the net energy change."""
    by_stage: dict[int, list] = {}
    for step in trace:
        by_stage.setdefault(step.stage, []).append(step)

    lines = []
    for stage in sorted(by_stage):
        steps = by_stage[stage]
        net = steps[-1].energy - steps[0].energy
        kind = "total" if steps[0].energy_kind == "total" else "strain"
        lines.append(
            f"{steps[0].backend}: {len(steps)} step(s), {kind} energy "
            f"{steps[0].energy:.4f} -> {steps[-1].energy:.4f} "
            f"{steps[0].energy_unit} (net {net:+.4f})"
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
        molecule = _read_structure(molblock, job)
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
        parts = [f"backend {settings.backend}"]
        parts.append(
            f"pH {settings.ph:g}" if settings.ph is not None else "protonation as drawn"
        )
        parts.append(
            f"{settings.n_confs} conformers via {settings.conf_method}"
            if settings.n_confs > 1 else "single conformer"
        )
        parts.append(
            "formats " + "/".join(settings.formats) if settings.formats
            else "dry run, writing nothing"
        )
        if settings.trace:
            parts.append("tracing every step")
        if settings.trajectory:
            parts.append("saving trajectory")
        if settings.params:
            parts.append("Rosetta params")
        job.say(", ".join(parts))

        Path(target.directory).mkdir(parents=True, exist_ok=True)
        if target.will_create_directory:
            job.say(f"created directory {target.directory}")

        if settings.formats:
            base = Path(target.directory) / f"{target.stem}.{settings.formats[0]}"
        else:
            job.say("dry run: building and checking, writing no files", "warn")
            base = Path(target.directory) / target.stem

        # `is not None`, not truthiness: an empty options dict means "queue
        # this with the defaults", and running locally instead would silently
        # do the opposite of what was asked.
        if job.slurm_options is not None:
            _run_on_slurm(job, molecule, settings, base)
            return

        if job.in_container:
            _run_in_container(job, molecule, settings, base)
            return

        outcomes = run(molecule, settings, output=base)

        outputs: list[str] = []
        trace: list[dict] = []
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
            if outcome.trace:
                for line in summarize_trace(outcome.trace):
                    job.say(line)
                trace.extend(step.to_json() for step in outcome.trace)
            written = outcome.written()
            if written:
                job.say(f"wrote {len(written)} file(s):", "ok")
                for path in written:
                    job.say(str(path), "ok")
                    outputs.append(str(path))
            else:
                job.say("no files written (dry run)", "warn")

        job.result = outcomes_to_result(molecule, outcomes, outputs, trace)
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


def outcomes_to_result(
    molecule, outcomes, outputs: list[str], trace: list[dict]
) -> dict[str, Any]:
    """What the page needs to draw a finished build.

    Shared with the SLURM path, which runs on a compute node and writes this
    same structure to disk. Keeping one serializer means a queued build gets
    the energy plot and the stereo summary, not a lesser view.
    """
    from .. import molecule as mol_mod

    return {
        "smiles": molecule.smiles,
        "formula": molecule.formula,
        "outputs": outputs,
        "n_conformers": sum(len(o.records) for o in outcomes),
        "seconds": round(sum(o.wall_seconds for o in outcomes), 3),
        "stereocenters": [
            {"atom": i, "code": code} for i, code in molecule.stereo.assigned_centers
        ],
        "double_bonds": [
            {"begin": r.begin, "end": r.end, "cip": r.cip, "cis_trans": r.cis_trans}
            for r in mol_mod.describe_double_bonds(molecule)
        ],
        "trace": trace,
    }


def collect_outputs(outcomes) -> tuple[list[str], list[dict]]:
    """The written paths and the flattened energy trace across all outcomes."""
    outputs: list[str] = []
    trace: list[dict] = []
    for outcome in outcomes:
        outputs.extend(str(p) for p in outcome.written())
        if outcome.trace:
            trace.extend(step.to_json() for step in outcome.trace)
    return outputs, trace


def _run_in_container(job: Job, molecule, settings, base: Path) -> None:
    """Build inside the Apptainer image that has this backend, on this machine.

    This is what makes eSEN, UMA and AllScAIP reachable from the browser: they
    cannot share a virtualenv with MACE, so the server's own interpreter will
    never be able to run them.
    """
    from ..container import ContainerError, image_for, run

    try:
        image = image_for(settings.backend)
    except ContainerError as exc:
        raise Ligand3DError(str(exc)) from exc

    job.say(f"running in {Path(image).name}", "ok")
    try:
        result = run(molecule, settings, base, image=image, echo=job.say)
    except ContainerError as exc:
        raise Ligand3DError(str(exc)) from exc

    job.result = result
    job.result["container"] = Path(image).name
    for path in result.get("outputs", []):
        job.say(str(path), "ok")
    job.state = "done"


def _run_on_slurm(job: Job, molecule, settings, base: Path) -> None:
    """Queue the build on a GPU node and follow it, instead of running here.

    The page stays responsive because this is already on a background thread;
    all that changes is what the thread is waiting on.
    """
    from ..slurm import (
        SlurmConfig, build_payload, container_for, job_name_for, needs_gpu, submit,
        wait_for,
    )

    options = job.slurm_options or {}
    partition = str(options.get("partition") or "gpu")
    config = SlurmConfig(
        partition=partition,
        gpu_class=str(options.get("gpu_class") or "small"),
        walltime=str(options.get("walltime") or "01:00:00"),
        cpus=int(options.get("cpus") or 4),
        memory=str(options.get("memory") or "16G"),
        # A cpu destination asks for no GRES. SlurmConfig.is_gpu already
        # refuses to emit one off a gpu partition, so this is belt and braces
        # — but an explicit 0 is what the caller meant.
        gpus=int(options.get("gpus", 1)),
        account=str(options.get("account") or "IPD"),
        job_name=job_name_for(molecule.name),
    )
    if not needs_gpu(settings.backend) and config.is_gpu:
        # Only worth saying when a GPU was actually requested. Choosing a cpu
        # node for quantum chemistry is the right answer, not a mistake to
        # warn about.
        job.say(
            f"{settings.backend} is not a neural potential, so a GPU will not help — "
            "queueing costs more than the calculation does",
            "warn",
        )

    workdir = base.parent / f"{base.stem}.slurm"
    n = 2
    while workdir.exists():
        workdir = base.parent / f"{base.stem}.slurm{n}"
        n += 1

    submitted = submit(build_payload(molecule, settings, base), workdir, config)
    job.slurm_job_id = submitted.job_id
    job.say(
        f"submitted SLURM job {submitted.job_id} to {config.partition}"
        + (f" on {config.gres()}" if config.is_gpu else ""),
        "ok",
    )
    job.say(f"container {Path(container_for(settings.backend)).name}")
    job.say(f"log {submitted.stdout}")

    def narrate(state: str) -> None:
        job.say(f"job {submitted.job_id} is {state}")
        if state == "PENDING":
            job.say("waiting for a node — this can take a while when the cluster is busy")

    state = wait_for(submitted.job_id, poll_seconds=5.0, on_state=narrate)
    result_path = workdir / "result.json"

    if state != "COMPLETED" and not result_path.exists():
        # A job the scheduler cannot account for still counts as finished if it
        # left a result behind, so only complain when there is nothing to show.
        for line in _tail(submitted.stderr, 12):
            job.say(line, "error")
        raise Ligand3DError(
            f"SLURM job {submitted.job_id} ended as {state}. Full log: {submitted.stderr}"
        )
    if not result_path.exists():
        raise Ligand3DError(
            f"job {submitted.job_id} completed but wrote no result. See {submitted.stderr}"
        )
    if state != "COMPLETED":
        job.say(
            f"the scheduler reported {state}, but the job left a complete result",
            "warn",
        )
    job.result = json.loads(result_path.read_text())
    job.result["slurm_job_id"] = submitted.job_id
    for path in job.result.get("outputs", []):
        job.say(str(path), "ok")
    job.state = "done"


def _tail(path: Path, lines: int) -> list[str]:
    try:
        return path.read_text(errors="replace").strip().splitlines()[-lines:]
    except OSError:
        return []


def _read_structure(text: str, job: Job):
    """Read a molblock, or a bare SMILES string from the paste-box fallback."""
    from ..molecule import from_molblock, from_smiles

    stripped = text.strip()
    # A molblock always has a counts line, so it is never a single short line.
    if "\n" not in stripped and len(stripped) < 400:
        job.say(f"reading the pasted SMILES {stripped}")
        return from_smiles(stripped)
    job.say(f"reading the drawn structure ({len(text.splitlines())} molblock lines)")
    return from_molblock(text)


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

    # An explicitly empty list is the dry-run request and must survive; only a
    # missing key falls back to the default.
    formats = normalize_formats(data.get("formats"))

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
        split_conformers=bool(data.get("split_conformers")),
        trajectory_every=max(1, int(num("trajectory_every", 10, int))),
        align_conformers=data.get("align_conformers", True) is not False,
        seed=int(num("seed", 0xF00D, int)),
        n_threads=int(num("threads", 1, int)),
        resname=(data.get("resname") or None),
        formats=tuple(formats),
        trace=bool(data.get("trace", True)),
        trajectory=bool(data.get("trajectory")),
        params=bool(data.get("params")),
        rfd_design_length=(str(data.get("rfd_design_length")).strip() or None)
        if data.get("rfd_design_length") else None,
        rfd_fix_coordinates=bool(data.get("rfd_fix_coordinates", True)),
        params_code=(data.get("params_code") or None),
        allow_code_conflict=bool(data.get("allow_code_conflict")),
    )


def slurm_status() -> dict[str, Any]:
    """Whether this host can hand a build to a GPU node.

    The page hides the option entirely when it cannot, so nobody outside the
    IPD cluster is offered a checkbox that could only fail.
    """
    from ..slurm import (
        QUANTUM_CHEM_SIF, UMA_SIF, apptainer_available, slurm_available,
    )

    has_sbatch = slurm_available()
    has_apptainer = apptainer_available()
    containers = {
        "mace": Path(QUANTUM_CHEM_SIF).exists(),
        "fairchem": Path(UMA_SIF).exists(),
    }
    return {
        "available": has_sbatch and has_apptainer and any(containers.values()),
        "sbatch": has_sbatch,
        "apptainer": has_apptainer,
        "containers": containers,
    }


def solvent_catalog() -> list[dict[str, Any]]:
    """The ALPB solvent table, for the page's dropdown."""
    from ..solvents import SOLVENTS

    return [
        {
            "name": s.name,
            "aliases": list(s.aliases),
            "dielectric": s.dielectric,
            "note": s.note,
        }
        for s in SOLVENTS
    ]


def backend_catalog() -> list[dict[str, Any]]:
    """Every registered backend with its capabilities and whether it can run."""
    from ..container import runnable_in_container
    from ..minimize import all_backends

    backends = all_backends()
    # Worked out once for the whole list: it stats a network filesystem, and
    # doing that per backend is how this endpoint got slow before.
    in_container = runnable_in_container(b.caps.name for b in backends)

    catalog = []
    for backend in backends:
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
                # The functional and basis, where the backend has one, so the
                # menu can say what will run rather than only which id.
                "level": getattr(getattr(backend, "method", None), "keywords", None),
                "description": caps.description,
                "takes_charge": caps.takes_charge,
                "supports_solvation": caps.supports_solvation,
                "ready": ready,
                "reason": reason,
                "hint": hint,
                # Whether picking "in a container" makes this selectable. The
                # menu greys out what cannot run, and without this the models
                # the container exists for would stay greyed out.
                "in_container": caps.name in in_container,
            }
        )
    return catalog

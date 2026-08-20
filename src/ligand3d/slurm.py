"""Submit a build to SLURM, for GPU-backed machine-learned force fields.

This is an IPD-specific convenience and is entirely optional: nothing else in
ligand3d imports it, and the tool works exactly as before without SLURM.

The point is narrow. The neural potentials are the only backends slow enough on
CPU to be worth queueing — MACE-OFF large takes 31 s per conformer here, and a
50-conformer refinement is half an hour. Everything else finishes faster than a
job would start.

Two details make this more than "shell out to sbatch":

**The local environment is CPU-only.** `torch` here is the `+cpu` build, so
submitting the current interpreter to a GPU node would allocate a GPU and ignore
it. The job therefore runs inside an Apptainer image that has a CUDA torch, with
this source tree bind-mounted and put on PYTHONPATH — no install step, and the
code that runs is the code you have checked out.

**The right image depends on the backend.** The mace/fairchem e3nn split exists
inside the containers too: the quantum_chem image carries e3nn 0.4.4 for MACE and
AIMNet2, the uma image carries e3nn 0.5.9 for eSEN, UMA, and AllScAIP. Picking
the wrong one fails the same way it does locally, so the image is chosen from the
requested backend unless told otherwise.

Conventions here follow the ones already in use on this cluster: write a script
and `sbatch --parsable` it rather than `--wrap`, bind `/home /net /mnt`, `--nv`
for GPU, and `gpu:<class>:<count>` GRES where class is small, large, or h200.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import Ligand3DError


class SlurmError(Ligand3DError):
    """Submitting to, or querying, SLURM failed."""


# Images that carry a CUDA torch plus the potentials. Overridable per-field via
# the environment so this file is not the only place a path can live.
QUANTUM_CHEM_SIF = Path(
    os.environ.get(
        "LIGAND3D_SIF_MACE",
        "/net/software/containers/users/woodbuse/quantum_chem/quantum_chem-20260604.sif",
    )
)
UMA_SIF = Path(
    os.environ.get(
        "LIGAND3D_SIF_FAIRCHEM",
        "/net/software/containers/users/woodbuse/quantum_chem/uma-20260527.sif",
    )
)

STANDARD_BINDS: tuple[str, ...] = ("/home", "/net", "/mnt")

# The scheduler rejects very short jobs outright: "The job time limit is too
# short; run longer jobs!". Five minutes is accepted.
MIN_WALLTIME_MINUTES = 5

TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY",
}


def job_name_for(molecule_name: str) -> str:
    """A SLURM job name built from a molecule name.

    The molecule name comes from a filename or a drawing, so it can contain
    anything; only the characters SLURM is happy with survive.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_.+-]", "", molecule_name.lower())[:20]
    return f"l3d-{cleaned}" if cleaned else "l3d"


def slurm_available() -> bool:
    """True if this host can submit jobs."""
    return shutil.which("sbatch") is not None


def apptainer_available() -> bool:
    return shutil.which("apptainer") is not None or shutil.which("singularity") is not None


def container_for(backend_spec: str) -> Path:
    """Pick the image that can actually run this backend chain.

    The e3nn incompatibility between MACE and fairchem exists inside the
    containers as well, so a chain mixing the two cannot be satisfied by either
    image and is refused here rather than failing on a compute node ten minutes
    later.
    """
    from .config import MODELS_BY_KEY
    from .minimize import resolve_name

    families = set()
    for name in (part.strip() for part in backend_spec.split(",")):
        spec = MODELS_BY_KEY.get(resolve_name(name))
        if spec is not None:
            families.add("fairchem" if spec.family == "fairchem" else "mace")

    if families == {"mace", "fairchem"}:
        raise SlurmError(
            f"backend chain {backend_spec!r} mixes MACE and fairchem models. They pin "
            "incompatible e3nn versions, so no single container can run both. Split the "
            "run, or pick models from one family."
        )
    return UMA_SIF if families == {"fairchem"} else QUANTUM_CHEM_SIF


def needs_gpu(backend_spec: str) -> bool:
    """True if any link of the chain is a neural potential.

    Only those are worth a GPU. Queueing an MMFF94 job that finishes in four
    milliseconds is strictly slower than running it here.
    """
    from .minimize import get_backend, resolve_name

    for name in (part.strip() for part in backend_spec.split(",")):
        try:
            if get_backend(resolve_name(name)).caps.kind == "mlff":
                return True
        except Ligand3DError:
            continue
    return False


@dataclass
class SlurmConfig:
    """Resources and placement for one submission."""

    partition: str = "gpu"
    gpu_class: str = "small"
    """small, large, or h200. The old a4000/b4000 GRES names no longer schedule."""
    gpus: int = 1
    cpus: int = 4
    memory: str = "16G"
    walltime: str = "01:00:00"
    account: str = field(default_factory=lambda: os.environ.get("LIGAND3D_SLURM_ACCOUNT", "IPD"))
    job_name: str = "ligand3d"
    container: Path | None = None
    binds: tuple[str, ...] = STANDARD_BINDS
    constraint: str | None = None
    exclude: str | None = None
    extra_sbatch: tuple[str, ...] = ()
    email: str | None = None

    @property
    def is_gpu(self) -> bool:
        return self.partition.startswith("gpu") and self.gpus > 0

    def gres(self) -> str | None:
        if not self.is_gpu:
            return None
        return f"gpu:{self.gpu_class}:{self.gpus}"

    def check(self) -> list[str]:
        """Problems that would make the job fail, waste an allocation, or worse.

        These values reach a shell script, and one of them is a free-text box
        on a web page. A newline in any of them would end the `#SBATCH` comment
        and leave the rest sitting in the script as a command, so the shapes are
        checked rather than trusted — and a value that cannot be parsed is
        refused here instead of raising somewhere less helpful.
        """
        problems: list[str] = []

        for label, value, pattern in (
            ("partition", self.partition, r"[A-Za-z0-9_.-]+"),
            ("gpu class", self.gpu_class, r"[A-Za-z0-9_.-]+"),
            ("memory", self.memory, r"\d+[KMGT]?B?"),
            ("account", self.account, r"[A-Za-z0-9_.-]*"),
            ("job name", self.job_name, r"[A-Za-z0-9_.+-]+"),
            ("walltime", self.walltime, r"(\d+-)?\d+(:\d+){0,2}"),
        ):
            if not re.fullmatch(pattern, value or ""):
                problems.append(f"{label} {value!r} is not a valid SLURM value")

        if not problems:  # only parseable once the shape is known good
            if _walltime_minutes(self.walltime) < MIN_WALLTIME_MINUTES:
                problems.append(
                    f"walltime {self.walltime} is below the {MIN_WALLTIME_MINUTES}-minute "
                    "minimum this scheduler enforces"
                )
        if self.is_gpu and self.gpu_class not in ("small", "large", "h200"):
            problems.append(
                f"gpu class {self.gpu_class!r} is not one of small, large, h200"
            )
        if self.cpus < 1 or self.gpus < 0:
            problems.append(f"cannot ask for {self.cpus} cpus and {self.gpus} gpus")
        return problems


def _walltime_minutes(walltime: str) -> float:
    """Minutes in a SLURM walltime.

    Six shapes are accepted, and what a field means depends on how many there
    are: bare `30` is thirty minutes, but `30` after a day count is thirty
    hours. Reading right-to-left as h:m:s gets the common cases wrong.
    """
    text = walltime.strip()
    days = 0
    dashed = "-" in text
    if dashed:
        day_part, _, text = text.partition("-")
        days = int(day_part)

    parts = [float(p) for p in text.split(":")] if text else [0.0]
    if dashed:
        # days-hours[:minutes[:seconds]]
        hours, minutes, seconds = (parts + [0.0, 0.0])[:3]
    elif len(parts) == 1:
        hours, minutes, seconds = 0.0, parts[0], 0.0
    elif len(parts) == 2:
        hours, minutes, seconds = 0.0, parts[0], parts[1]
    else:
        hours, minutes, seconds = parts[:3]
    return days * 1440 + hours * 60 + minutes + seconds / 60


def _is_shared(path: Path) -> bool:
    """True if the path looks like it is on shared storage.

    A compute node has its own `/tmp`, so a job whose output goes there writes
    to a filesystem that vanishes when the allocation ends. This is not
    hypothetical: the first probe job for this feature ran fine, exited 0, and
    left nothing behind, because its `--output` pointed at the submit host's
    `/tmp`.
    """
    resolved = str(path.resolve())
    return any(resolved.startswith(root) for root in ("/home", "/net", "/mnt", "/projects"))


def _snapshot_source(source_root: Path, workdir: Path) -> Path:
    """Copy the package into the job directory and return the new import root.

    A job can sit in the queue for hours. Bind-mounting the working tree would
    mean it runs whatever the source happens to be when it finally starts, so
    editing the repo could change or break a job already submitted. Copying is
    half a megabyte and makes the run reproducible: the job directory holds the
    exact code that produced its output.
    """
    destination = workdir / "src"
    package = source_root / "ligand3d"
    if not package.is_dir():
        # Installed rather than checked out; import from wherever it lives.
        return source_root
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(
        package,
        destination / "ligand3d",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "sketcher"),
    )
    return destination


@dataclass
class SubmittedJob:
    """A queued job and where to find its results."""

    job_id: int
    workdir: Path
    script: Path
    stdout: Path
    stderr: Path
    config: SlurmConfig
    backend: str = ""

    def to_json(self) -> dict[str, Any]:
        data = {
            "job_id": self.job_id,
            "workdir": str(self.workdir),
            "script": str(self.script),
            "stdout": str(self.stdout),
            "stderr": str(self.stderr),
            "backend": self.backend,
            "config": asdict(self.config),
        }
        data["config"]["container"] = str(self.config.container) if self.config.container else None
        data["config"]["binds"] = list(self.config.binds)
        data["config"]["extra_sbatch"] = list(self.config.extra_sbatch)
        return data


def render_script(
    workdir: Path,
    config: SlurmConfig,
    container: Path,
    source_root: Path,
    payload: Path,
) -> str:
    """Build the sbatch script.

    Short-flag directives, `set -euo pipefail`, and
    `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` follow what is already used on this
    cluster so the scripts read like their neighbours.
    """
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {config.job_name}",
        f"#SBATCH -p {config.partition}",
        f"#SBATCH -c {config.cpus}",
        f"#SBATCH --mem={config.memory}",
        f"#SBATCH -t {config.walltime}",
        "#SBATCH -N 1",
        f"#SBATCH -o {workdir / 'job.out'}",
        f"#SBATCH -e {workdir / 'job.err'}",
    ]
    gres = config.gres()
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    if config.account:
        lines.append(f"#SBATCH -A {config.account}")
    if config.constraint:
        lines.append(f"#SBATCH --constraint='{config.constraint}'")
    if config.exclude:
        lines.append(f"#SBATCH --exclude={config.exclude}")
    if config.email:
        lines.append(f"#SBATCH --mail-user={config.email}")
        lines.append("#SBATCH --mail-type=END,FAIL")
    lines.extend(config.extra_sbatch)

    binds = " ".join(f"--bind {b}:{b}" for b in config.binds if Path(b).exists())
    nv = "--nv " if config.is_gpu else ""

    lines += [
        "",
        "set -euo pipefail",
        'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"',
        "",
        'echo "host=$(hostname) job=${SLURM_JOB_ID:-none}"',
    ]
    if config.is_gpu:
        lines.append(
            'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader '
            '|| echo "no nvidia-smi on this node"'
        )
    lines += [
        "",
        "# The source tree is bind-mounted and put on PYTHONPATH rather than",
        "# installed, so the job runs exactly the code that was submitted.",
        f"apptainer exec {nv}{binds} \\",
        f"  --env PYTHONPATH={source_root} \\",
        # Without this the job compiles the snapshot as it imports it, leaving
        # a quarter of a megabyte of __pycache__ in what should be a record of
        # exactly what was submitted.
        "  --env PYTHONDONTWRITEBYTECODE=1 \\",
        "  --env HF_HOME=/net/databases/huggingface \\",
        "  --env TORCHDYNAMO_DISABLE=1 \\",
        f"  {container} \\",
        f"  python -m ligand3d.cli slurm-run {payload}",
        "",
        'echo "ligand3d job finished"',
    ]
    return "\n".join(lines) + "\n"


def submit(
    payload: dict[str, Any],
    workdir: Path,
    config: SlurmConfig | None = None,
    source_root: Path | None = None,
    dry_run: bool = False,
) -> SubmittedJob:
    """Queue one build. `payload` is what `slurm-run` will execute."""
    config = config or SlurmConfig()
    workdir = Path(workdir).expanduser().resolve()

    if not dry_run and not slurm_available():
        raise SlurmError(
            "sbatch was not found, so this host cannot submit jobs. Run without "
            "--slurm, or submit from a login node."
        )
    if not dry_run and not apptainer_available():
        raise SlurmError("apptainer was not found; the job needs it to get a CUDA torch.")

    problems = config.check()
    if problems:
        raise SlurmError("; ".join(problems))

    if not _is_shared(workdir):
        raise SlurmError(
            f"{workdir} does not look like shared storage. A compute node has its own "
            f"/tmp, so results written there are lost when the job ends. Choose a "
            f"directory under /home, /net, or /mnt."
        )

    backend = str(payload.get("settings", {}).get("backend", ""))
    container = config.container or container_for(backend)
    if not dry_run and not Path(container).exists():
        raise SlurmError(
            f"container {container} does not exist. Set LIGAND3D_SIF_MACE or "
            f"LIGAND3D_SIF_FAIRCHEM, or pass one explicitly."
        )

    workdir.mkdir(parents=True, exist_ok=True)
    source_root = _snapshot_source(
        Path(source_root or Path(__file__).resolve().parents[1]), workdir
    )

    payload_path = workdir / "job.json"
    payload_path.write_text(json.dumps(payload, indent=1))

    script_path = workdir / "job.sbatch"
    script_path.write_text(
        render_script(workdir, config, Path(container), source_root, payload_path)
    )
    script_path.chmod(0o755)

    job = SubmittedJob(
        job_id=0,
        workdir=workdir,
        script=script_path,
        stdout=workdir / "job.out",
        stderr=workdir / "job.err",
        config=SlurmConfig(**{**asdict(config), "container": Path(container)}),
        backend=backend,
    )
    if dry_run:
        return job

    try:
        proc = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SlurmError(f"could not run sbatch: {exc}") from exc
    if proc.returncode != 0:
        raise SlurmError(f"sbatch refused the job: {(proc.stderr or proc.stdout).strip()}")

    try:
        job.job_id = int(proc.stdout.split(";")[0].strip())
    except (ValueError, IndexError) as exc:
        raise SlurmError(f"could not read a job id from sbatch: {proc.stdout!r}") from exc

    (workdir / "job.meta.json").write_text(json.dumps(job.to_json(), indent=1))
    return job


def job_state(job_id: int) -> str:
    """Current state, from squeue if it is still queued, else sacct."""
    try:
        proc = subprocess.run(
            ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass

    try:
        proc = subprocess.run(
            ["sacct", "-j", str(job_id), "--format=State", "--noheader", "-P"],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SlurmError(f"could not query job {job_id}: {exc}") from exc
    for line in proc.stdout.splitlines():
        state = line.strip().split()[0] if line.strip() else ""
        if state:
            return state
    return "UNKNOWN"


def wait_for(job_id: int, poll_seconds: float = 15.0, timeout: float | None = None) -> str:
    """Block until the job leaves the queue. Returns its final state.

    Transient failures of squeue and sacct are tolerated: a scheduler too busy
    to answer is not the same as a job that died, and treating it that way
    would abandon a run that is going fine.
    """
    started = time.monotonic()
    delay = poll_seconds
    failures = 0
    while True:
        try:
            state = job_state(job_id)
            failures = 0
        except SlurmError:
            failures += 1
            if failures > 20:
                raise
            state = "UNKNOWN"

        if state in TERMINAL_STATES:
            return state
        if timeout is not None and time.monotonic() - started > timeout:
            raise SlurmError(f"job {job_id} still {state} after {timeout:g}s")
        time.sleep(delay)
        delay = min(delay * 1.25, 60.0)


def build_payload(molecule, settings, output: Path) -> dict[str, Any]:
    """Everything `slurm-run` needs, as plain JSON.

    The molecule travels as a molblock and the settings as a dict, so the job
    never has to reconstruct a Python object from a command line — which is
    also how quoting problems are avoided entirely. A molblock carries the
    stereochemistry and any existing conformer, so what the node builds is what
    was submitted, not a re-parse of the original SMILES.
    """
    data = asdict(settings)
    data["formats"] = list(settings.formats)
    return {
        "version": 1,
        "molblock": molecule.molblock,
        "name": molecule.name,
        "settings": data,
        "output": str(output),
    }


def run_payload(payload_path: Path) -> tuple[Any, list]:
    """Execute a payload. This is what runs inside the container.

    Returns the molecule alongside the outcomes because the caller serializes
    both back to the submitter.
    """
    from .molecule import from_molblock
    from .pipeline import Settings, run

    data = json.loads(Path(payload_path).read_text())
    settings_data = dict(data["settings"])
    settings_data["formats"] = tuple(settings_data.get("formats") or ())
    settings = Settings(**settings_data)

    molecule = from_molblock(data["molblock"], name=data.get("name", "LIG"))
    return molecule, run(molecule, settings, output=Path(data["output"]))

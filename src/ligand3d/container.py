"""Run a build inside an Apptainer image, on this machine.

The eSEN, UMA and AllScAIP models cannot run in this virtualenv and never will:
`mace-torch` pins `e3nn==0.4.4` and `fairchem-core` needs `e3nn>=0.5`, so one
environment cannot hold both. That is a genuine conflict, not a missing
install, and no amount of pip will resolve it.

It is also already solved. The images used for GPU submission each carry one
side of the split, so the model that is unavailable here is available a few
hundred milliseconds away in a container that is already on disk. This runs a
build in one without involving the scheduler, which is what you want for a
single small molecule where queueing costs more than the calculation.

The mechanism is the one `--slurm` already uses: the settings and the molecule
go into a JSON payload, a copy of the package goes alongside it, and the
container's interpreter is pointed at both. Nothing is installed, and the code
that runs is the code you have checked out.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .errors import Ligand3DError
from .slurm import (
    STANDARD_BINDS,
    SlurmError,
    _snapshot_source,
    apptainer_available,
    build_payload,
    container_for,
)


class ContainerError(Ligand3DError):
    """Running a build inside a container failed."""


def available() -> bool:
    """True if a build can be run in a container on this machine."""
    return apptainer_available()


def image_for(backend_spec: str) -> Path:
    """The image that can run this backend chain."""
    try:
        return container_for(backend_spec)
    except SlurmError as exc:  # the mace/fairchem split, described once
        raise ContainerError(str(exc)) from exc


def would_help(backend_spec: str) -> bool:
    """True if running in a container turns an unusable chain into a usable one.

    Only true when something in the chain cannot run here but the image exists,
    which is the case worth mentioning to somebody staring at "unavailable".
    """
    from .minimize import get_backend, resolve_name

    if not available():
        return False
    blocked = False
    for name in (part.strip() for part in backend_spec.split(",")):
        try:
            backend = get_backend(resolve_name(name))
        except Ligand3DError:
            return False
        try:
            if not backend.available():
                blocked = True
        except Exception:
            blocked = True
    if not blocked:
        return False
    try:
        return Path(image_for(backend_spec)).exists()
    except ContainerError:
        return False


def run(
    molecule,
    settings,
    output: Path,
    image: Path | None = None,
    workdir: Path | None = None,
    timeout: float = 3600.0,
    echo=None,
) -> dict[str, Any]:
    """Build `molecule` inside a container and return the result payload.

    `echo` is called with each line the container prints, so a long
    minimization is not a silent wait.
    """
    if not available():
        raise ContainerError(
            "apptainer was not found, so a container cannot be used here. "
            "Install it, or use --slurm to run on a node that has it."
        )

    image = Path(image or image_for(settings.backend))
    if not image.exists():
        raise ContainerError(
            f"container {image} does not exist. Set LIGAND3D_SIF_MACE or "
            f"LIGAND3D_SIF_FAIRCHEM to point at one."
        )

    output = Path(output).expanduser().resolve()
    temporary = workdir is None
    root = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ligand3d-"))
    root.mkdir(parents=True, exist_ok=True)

    try:
        source_root = _snapshot_source(
            Path(__file__).resolve().parents[1], root
        )
        payload_path = root / "job.json"
        payload_path.write_text(
            json.dumps(build_payload(molecule, settings, output), indent=1)
        )

        binds: list[str] = []
        for path in (*STANDARD_BINDS, str(output.parent), str(root)):
            if Path(path).exists():
                binds += ["--bind", f"{path}:{path}"]

        command = [
            "apptainer", "exec", *binds,
            "--env", f"PYTHONPATH={source_root}",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "HF_HOME=/net/databases/huggingface",
            "--env", "TORCHDYNAMO_DISABLE=1",
            str(image),
            "python", "-m", "ligand3d.cli", "slurm-run", str(payload_path),
        ]

        lines = _stream(command, timeout, echo)
        result_path = root / "result.json"
        if not result_path.exists():
            tail = "\n".join(lines[-15:])
            raise ContainerError(
                f"the build in {image.name} produced no result.\n{tail}"
            )
        return json.loads(result_path.read_text())
    finally:
        if temporary:
            shutil.rmtree(root, ignore_errors=True)


def _stream(command: list[str], timeout: float, echo) -> list[str]:
    """Run the container, passing its output through as it arrives."""
    lines: list[str] = []
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
    except OSError as exc:
        raise ContainerError(f"could not start apptainer: {exc}") from exc

    try:
        assert process.stdout is not None
        listing = False
        for line in process.stdout:
            line = line.rstrip("\n")
            lines.append(line)
            # The run inside the container ends by listing what it wrote. The
            # caller reports that from the result payload, on its own lines for
            # copy-paste, so echoing it here would print it all twice.
            listing = listing or line.strip().startswith("wrote ")
            if echo is not None and not listing and _worth_showing(line):
                echo(line)
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise ContainerError(f"the container run exceeded {timeout:g}s") from exc

    if code != 0:
        tail = "\n".join(lines[-15:])
        raise ContainerError(f"the container exited {code}.\n{tail}")
    return lines


# torch and its dependencies narrate a great deal on import, none of which is
# about the molecule.
_NOISE = (
    "cuequivariance", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "warnings.warn",
    "FutureWarning", "UserWarning", "DeprecationWarning", "_Jd,",
    "torch.load", "TRANSFORMERS_CACHE",
)


def _worth_showing(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return not any(marker in line for marker in _NOISE)

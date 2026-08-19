"""Semi-empirical tier: GFN1/GFN2-xTB via tblite, GFN-FF via the xtb binary.

Two different integration strategies, for a reason. tblite ships manylinux
wheels and exposes an ASE calculator, so GFN1/GFN2 ride the shared ASE path.
GFN-FF exists only in the xtb program, and xtb has a perfectly good internal
optimizer, so driving it from ASE would mean one process launch per optimizer
step. That backend shells out once and lets xtb do the whole relaxation.

Both support ALPB implicit solvation, which is what keeps zwitterions from
collapsing.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path

from rdkit import Chem

from ..errors import BackendUnavailable, MinimizationError
from .ase_bridge import ASEBackend
from .base import (
    Availability,
    Capabilities,
    MinimizeJob,
    MinimizeResult,
    register,
)

HARTREE_TO_KCAL = 627.5094740631


class TBLiteBackend(ASEBackend):
    """GFN1-xTB or GFN2-xTB through the tblite Python wheel."""

    def __init__(self, method: str, name: str, description: str) -> None:
        self.method = method
        self.caps = Capabilities(
            name=name,
            kind="semiempirical",
            description=description,
            takes_charge=True,
            supports_solvation=True,
            elements=None,  # tblite covers Z=1-86
            requires=("tblite", "ase"),
            energy_unit="kcal/mol",
            energy_kind="total",
        )

    def install_hint(self) -> str:
        return "pip install 'ligand3d[xtb]'   (tblite ships manylinux wheels; no compiler needed)"

    def make_calculator(self, job: MinimizeJob):
        from tblite.ase import TBLite

        kwargs = {
            "method": self.method,
            "charge": float(job.charge),
            "multiplicity": int(job.multiplicity),
            "verbosity": 0,
        }
        if job.solvent:
            # tblite exposes ALPB but not CPCM; asking for CPCM raises.
            kwargs["solvation"] = ("alpb", job.solvent)
        return TBLite(**kwargs)


class XTBBinaryBackend:
    """GFN-FF through the xtb executable, using xtb's own optimizer."""

    def __init__(self) -> None:
        self.caps = Capabilities(
            name="gfnff",
            kind="ff",
            description="GFN-FF generic force field (xtb binary). Fast, charge-aware.",
            takes_charge=True,
            supports_solvation=True,
            elements=None,
            requires=(),
            energy_unit="kcal/mol",
            energy_kind="total",
        )

    def _binary(self) -> str | None:
        from ..config import find_xtb_binary

        return find_xtb_binary()

    def available(self) -> Availability:
        binary = self._binary()
        if binary is None:
            return Availability(
                ok=False,
                reason="xtb executable not found",
                hint=(
                    "Set LIGAND3D_XTB_BIN to an xtb binary, put xtb on PATH, or download "
                    "one from https://github.com/grimme-lab/xtb/releases"
                ),
            )
        return Availability(ok=True)

    def minimize(self, job: MinimizeJob) -> MinimizeResult:
        started = time.perf_counter()
        binary = self._binary()
        if binary is None:
            raise BackendUnavailable(self.available().reason)

        with tempfile.TemporaryDirectory(prefix="ligand3d-xtb-") as tmp:
            work = Path(tmp)
            xyz = work / "input.xyz"
            Chem.MolToXYZFile(job.mol, str(xyz), confId=job.conf_id)

            cmd = [
                binary,
                xyz.name,
                "--gfnff",
                "--opt",
                "--chrg", str(job.charge),
                "--uhf", str(job.multiplicity - 1),
                "--parallel", str(max(1, job.n_threads)),
            ]
            if job.solvent:
                cmd += ["--alpb", job.solvent]

            env = _xtb_env()
            try:
                proc = subprocess.run(
                    cmd, cwd=work, capture_output=True, text=True, timeout=1800, env=env
                )
            except subprocess.TimeoutExpired as exc:
                raise MinimizationError("xtb timed out after 30 minutes") from exc

            optimized = work / "xtbopt.xyz"
            if not optimized.exists():
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
                raise MinimizationError(
                    "xtb did not produce an optimized geometry.\n" + "\n".join(tail)
                )

            energy_h, converged = _parse_xtbopt(optimized, proc.stdout)
            _load_xyz_into_conformer(optimized, job.mol, job.conf_id)

        note = "" if converged else "xtb reported the optimization did not converge"
        if job.trace or job.trajectory:
            # xtb's own optimizer runs inside the subprocess; it writes an
            # xtbopt.log trajectory but no per-step energies we can rely on
            # across versions, so say so rather than inventing a trace.
            note = (note + " " if note else "") + (
                "per-step tracing is not available for gfnff: xtb optimizes "
                "internally. Use gfn2 for a traced semi-empirical run."
            )
        return MinimizeResult(
            energy=energy_h * HARTREE_TO_KCAL,
            converged=converged,
            n_steps=-1,
            backend=self.caps.name,
            energy_unit="kcal/mol",
            energy_kind=self.caps.energy_kind,
            note=note.strip(),
            wall_seconds=time.perf_counter() - started,
        )


def _xtb_env() -> dict[str, str]:
    """Environment for the xtb subprocess, including any configured lib path."""
    import os

    from ..config import xtb_library_paths

    env = dict(os.environ)
    libs = xtb_library_paths()
    if libs:
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join([*libs, existing]) if existing else ":".join(libs)
    env.setdefault("OMP_NUM_THREADS", env.get("OMP_NUM_THREADS", "4"))
    return env


def _parse_xtbopt(path: Path, stdout: str) -> tuple[float, bool]:
    """Pull the final energy out of xtbopt.xyz's comment line."""
    lines = path.read_text().splitlines()
    energy = 0.0
    if len(lines) > 1:
        for token in lines[1].replace(":", " ").split():
            try:
                energy = float(token)
                break
            except ValueError:
                continue
    converged = "GEOMETRY OPTIMIZATION CONVERGED" in stdout.upper() or (
        "FAILED TO CONVERGE" not in stdout.upper()
    )
    return energy, converged


def _load_xyz_into_conformer(path: Path, mol: Chem.Mol, conf_id: int) -> None:
    from rdkit.Geometry import Point3D

    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    if n != mol.GetNumAtoms():
        raise MinimizationError(
            f"xtb returned {n} atoms for a molecule with {mol.GetNumAtoms()}"
        )
    conf = mol.GetConformer(conf_id)
    for i, line in enumerate(lines[2 : 2 + n]):
        _, x, y, z = line.split()[:4]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))


register(
    "gfn2",
    lambda: TBLiteBackend(
        "GFN2-xTB",
        "gfn2",
        "GFN2-xTB semi-empirical tight binding (tblite). ~1 s, near-QM geometries.",
    ),
    aliases=("gfn2-xtb", "xtb"),
)
register(
    "gfn1",
    lambda: TBLiteBackend(
        "GFN1-xTB",
        "gfn1",
        "GFN1-xTB semi-empirical tight binding (tblite).",
    ),
    aliases=("gfn1-xtb",),
)
register("gfnff", XTBBinaryBackend, aliases=("gfn-ff",))

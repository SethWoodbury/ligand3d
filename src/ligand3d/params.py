"""Rosetta params files.

Rosetta needs a per-ligand topology file: atom types, charges, bonds, and
optionally a rotamer library. The canonical generator is Rosetta's own
`molfile_to_params.py`, and this calls it rather than reimplementing it — the
atom typing is a large table of chemistry-specific rules that would be wrong in
subtle ways if rewritten.

What this module adds around it:

- Feeding it a multi-conformer SDF written straight from RDKit, so the rotamer
  library is the conformer ensemble ligand3d already generated.
- The conformer-file fixup. `molfile_to_params.py` writes conformer 1 to
  `NAME.pdb` and conformers 2..N to `NAME_conformers.pdb`, so the rotamer
  library referenced by `PDB_ROTAMERS` is missing its first member unless
  `NAME.pdb` is prepended onto it. Every in-house wrapper does this; getting it
  wrong silently costs you one rotamer.
- Checking the three-letter code against Rosetta's `residue_types.txt` before
  spending any work, because a collision there is a confusing runtime failure
  much later.

Aromaticity was a worry and turned out not to be one: molfile_to_params derives
ring aromaticity itself, and a Kekulized RDKit molfile and an aromatic-bond one
produce byte-identical `aroC`/`Haro` typing. Plain `Chem.SDWriter` output is
therefore fine.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from rdkit import Chem

from .errors import Ligand3DError, ResourceNotFound
from .molecule import rdkit_quiet
from .write import assign_pdb_names


class ParamsError(Ligand3DError):
    """Generating a Rosetta params file failed."""


@dataclass
class ParamsResult:
    """What a params run produced."""

    params: Path
    pdb: Path | None = None
    conformers: Path | None = None
    code: str = "LIG"
    n_conformers: int = 1
    notes: list[str] = field(default_factory=list)

    def paths(self) -> list[Path]:
        return [p for p in (self.params, self.pdb, self.conformers) if p is not None]


def normalize_code(code: str) -> str:
    """Coerce a ligand code to Rosetta's three alphanumeric characters."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", code or "")
    if not cleaned:
        raise ParamsError(
            f"{code!r} has no alphanumeric characters; a Rosetta ligand code must "
            "be exactly three, such as LIG or Z01."
        )
    return cleaned[:3].upper()


def code_conflict(code: str) -> str | None:
    """Return a description if `code` already exists in Rosetta, else None.

    A plain text search of `residue_types.txt`, which is what the in-house
    scripts do. It is deliberately conservative: a false positive costs you one
    rename, a false negative costs you a baffling Rosetta error much later.
    """
    from .config import find_rosetta_residue_types

    listing = find_rosetta_residue_types()
    if listing is None:
        return None

    code = normalize_code(code)
    try:
        text = listing.read_text(errors="replace")
    except OSError:
        return None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if re.search(rf"(^|/){re.escape(code)}\.params$", stripped):
            return f"{code} is already a Rosetta residue type ({stripped})"
    return None


def write_conformer_sdf(mol: Chem.Mol, path: Path, code: str) -> int:
    """Write every conformer to one SDF, with the atom names we assigned.

    molfile_to_params takes topology from the first entry and coordinates from
    the rest, which is exactly the shape of an RDKit multi-conformer molecule.
    """
    named = assign_pdb_names(mol, resname=code)
    conf_ids = [c.GetId() for c in named.GetConformers()]
    if not conf_ids:
        raise ParamsError("molecule has no conformers, so there is nothing to write")

    path.parent.mkdir(parents=True, exist_ok=True)
    with rdkit_quiet(), Chem.SDWriter(str(path)) as writer:
        for conf_id in conf_ids:
            work = Chem.Mol(named)
            work.SetProp("_Name", code)
            writer.write(work, confId=conf_id)
    return len(conf_ids)


def generate(
    mol: Chem.Mol,
    code: str = "LIG",
    out_dir: str | Path = ".",
    conformers: bool = True,
    allow_code_conflict: bool = False,
    keep_names: bool = True,
    root_atom: int | None = None,
    nbr_atom: int | None = None,
    timeout: float = 600.0,
) -> ParamsResult:
    """Generate a Rosetta params file for a 3D molecule.

    `mol` must already carry 3D coordinates and explicit hydrogens; extra
    conformers become the rotamer library when `conformers` is true.
    """
    from .config import find_molfile_to_params

    script = find_molfile_to_params()
    if script is None:
        raise ResourceNotFound(
            "Rosetta's molfile_to_params.py was not found. Set "
            "LIGAND3D_MOLFILE_TO_PARAMS to it, or add [rosetta].molfile_to_params "
            "to ~/.config/ligand3d/config.toml. Run 'ligand3d doctor' to see where "
            "it looked."
        )

    code = normalize_code(code)
    notes: list[str] = []

    conflict = code_conflict(code)
    if conflict:
        if not allow_code_conflict:
            raise ParamsError(
                f"{conflict}. Rosetta would load its own definition instead of "
                f"yours. Pick another code, or pass --allow-code-conflict to "
                f"override."
            )
        notes.append(f"{conflict} — proceeding because the conflict was allowed")

    if mol.GetNumAtoms() == 1:
        raise ParamsError(
            "molfile_to_params is not meant for single-atom ligands such as a bare "
            "metal ion; Rosetta has native types for those."
        )

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ligand3d-params-") as tmp:
        work = Path(tmp)
        sdf = work / f"{code}.sdf"
        n_written = write_conformer_sdf(mol, sdf, code)
        if not conformers and n_written > 1:
            n_written = 1
            single = Chem.Mol(mol)
            keep = Chem.Conformer(single.GetConformer(single.GetConformers()[0].GetId()))
            single.RemoveAllConformers()
            single.AddConformer(keep, assignId=True)
            write_conformer_sdf(single, sdf, code)

        command = [sys.executable, str(script), "--name", code, "--clobber"]
        if keep_names:
            # Without this Rosetta renames the atoms, which breaks any constraint
            # file or alignment that refers to them by the names we wrote.
            command.append("--keep-names")
        if conformers and n_written > 1:
            command.append("--conformers-in-one-file")
        if root_atom is not None:
            command.append(f"--root_atom={root_atom}")
        if nbr_atom is not None:
            command.append(f"--nbr_atom={nbr_atom}")
        command.append(sdf.name)

        try:
            proc = subprocess.run(
                command, cwd=work, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise ParamsError(f"molfile_to_params timed out after {timeout:g}s") from exc

        params = work / f"{code}.params"
        if not params.exists():
            tail = (proc.stdout + proc.stderr).strip().splitlines()[-15:]
            raise ParamsError(
                "molfile_to_params did not produce a params file.\n" + "\n".join(tail)
            )

        # Do not guess the structure file's name. With
        # --conformers-in-one-file it is NAME.pdb, and without it the single
        # conformer comes out as NAME_0001.pdb, so a hardcoded NAME.pdb silently
        # collects nothing for a one-conformer run.
        produced_confs = work / f"{code}_conformers.pdb"
        candidates = sorted(
            p for p in work.glob(f"{code}*.pdb") if p != produced_confs
        )
        produced_pdb = candidates[0] if candidates else work / f"{code}.pdb"
        if len(candidates) > 1:
            notes.append(
                "molfile_to_params wrote several structure files "
                f"({', '.join(p.name for p in candidates)}); kept {produced_pdb.name}"
            )

        if conformers and n_written > 1 and produced_confs.exists() and produced_pdb.exists():
            # Conformer 1 lives in NAME.pdb, not in NAME_conformers.pdb, so the
            # rotamer library is short by one until it is prepended.
            merged = produced_pdb.read_text().rstrip("\n") + "\n" + produced_confs.read_text()
            produced_confs.write_text(merged)
            notes.append(
                f"prepended {code}.pdb onto {code}_conformers.pdb so all "
                f"{n_written} conformers are in the rotamer library"
            )

        result = ParamsResult(
            params=out_dir / params.name, code=code, n_conformers=n_written, notes=notes
        )
        shutil.copy2(params, result.params)
        if produced_pdb.exists():
            result.pdb = out_dir / produced_pdb.name
            shutil.copy2(produced_pdb, result.pdb)
        if produced_confs.exists():
            result.conformers = out_dir / produced_confs.name
            shutil.copy2(produced_confs, result.conformers)

    _verify(result, mol.GetNumAtoms())
    return result


def count_rotamers(path: Path, n_atoms: int) -> int:
    """Count conformers in a rotamer library written by molfile_to_params.

    It separates conformers with `TER`, not `MODEL`/`ENDMDL`, so counting MODEL
    records returns zero for a perfectly good file. Counting coordinate lines
    and dividing by the atom count is what actually works, and it also catches a
    truncated final block.
    """
    text = path.read_text(errors="replace")
    atom_lines = sum(1 for line in text.splitlines() if line.startswith(("ATOM", "HETATM")))
    if n_atoms <= 0:
        return 0
    if atom_lines % n_atoms:
        raise ParamsError(
            f"{path} holds {atom_lines} coordinate lines, which is not a whole "
            f"number of {n_atoms}-atom conformers"
        )
    return atom_lines // n_atoms


def _verify(result: ParamsResult, n_atoms: int) -> None:
    """Sanity-check the params file rather than trusting the exit code."""
    text = result.params.read_text(errors="replace")
    if "NAME" not in text or not re.search(r"^ATOM ", text, re.M):
        raise ParamsError(f"{result.params} does not look like a params file")

    n_typed = len(re.findall(r"^ATOM ", text, re.M))
    if n_typed != n_atoms:
        raise ParamsError(
            f"{result.params} types {n_typed} atoms but the molecule has {n_atoms}"
        )

    if result.n_conformers <= 1:
        return

    if re.search(r"^PDB_ROTAMERS\s+(\S+)", text, re.M) is None:
        result.notes.append(
            "warning: more than one conformer was supplied but the params file has "
            "no PDB_ROTAMERS line, so Rosetta will not use the ensemble"
        )
        return

    if result.conformers is None:
        result.notes.append(
            "warning: PDB_ROTAMERS is set but no conformers file was produced"
        )
        return

    found = count_rotamers(result.conformers, n_atoms)
    if found != result.n_conformers:
        raise ParamsError(
            f"rotamer library {result.conformers} holds {found} conformer(s) but "
            f"{result.n_conformers} were supplied. Conformer 1 lives in "
            f"{result.code}.pdb and has to be prepended; that step looks to have "
            f"gone wrong."
        )


def summarize(result: ParamsResult) -> list[str]:
    """Human-readable lines about what was produced."""
    lines = [f"wrote {result.params}"]
    if result.pdb:
        lines.append(f"wrote {result.pdb}")
    if result.conformers:
        lines.append(
            f"wrote {result.conformers} ({result.n_conformers} rotamer(s))"
        )
    lines.extend(result.notes)
    return lines

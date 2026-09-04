# Output formats

mmCIF, SDF, PDB, Rosetta params, annotated CIF for RFdiffusion4, and minimization trajectories.

[← back to the README](../README.md)

## Output

**mmCIF is the default.** It carries everything a PDB does *plus* the bond orders and
aromaticity a PDB cannot, in a `_chem_comp_bond` loop, so a `.cif` from ligand3d reads
back as the same molecule — double bonds and all. Multiple conformers become models
tagged with `pdbx_PDB_model_num`. Provenance goes in a `_ligand3d` category.

An `.sdf` rides along by default because RDKit round-trips it perfectly. PDB is written
on request:

```bash
ligand3d build "<smiles>" -f cif,pdb,sdf -o thing    # all three
ligand3d build "<smiles>" -o thing.pdb               # an explicit suffix selects a format
ligand3d build "<smiles>" -f cif,annotated -o thing  # + thing.annotated.cif for RFdiffusion4
```

The four formats are `cif`, `sdf`, `pdb` and `annotated` — the last writes
`<name>.annotated.cif` beside the ordinary one and is
[described below](#annotated-cif-for-rfdiffusion4-proteina). It keeps the `.cif`
extension because it *is* an mmCIF; the double extension only says which of the two
carries the conditioning.

Nothing in the params path needs a PDB — `molfile_to_params` reads the SDF directly.

`--dry-run` (or unticking every format in the browser) builds, minimizes, checks and
reports without writing anything, which is the quick way to try a molecule before
committing to a filename.

Charges survive everywhere: the mmCIF carries `pdbx_formal_charge` per atom, so a
zwitterion reads back as `[NH3+]CC1(CC(=O)[O-])CCCCC1` with both sites intact, and a
nitro group as `[N+](=O)[O-]` with the charge separation on the right two atoms.
Neutral atoms are written as `0` rather than `?`, because `?` means *unknown* and a
neutral atom is not an atom of unknown charge.

**Aromatic bonds are kekulized**, three `SING` and three `DOUB` around benzene with
`pdbx_aromatic_flag Y`, the way the PDB Chemical Component Dictionary does it. Writing
every aromatic bond as `SING` — which is what a naive mapping produces — says the ring is
saturated to anything that reads bond orders rather than re-perceiving aromaticity.

PDB output has unique atom names, CONECT records, formal charges, and provenance REMARKs.
Every file written is read back and checked before the command returns.


## Rosetta params

```bash
ligand3d build "<smiles>" --confs 20 --params --params-code LIG
ligand3d params ensemble.sdf --code LIG -d params/
```

This drives Rosetta's own `molfile_to_params.py` rather than reimplementing its atom
typing. On top of it:

- **Conformers become the rotamer library.** The ensemble ligand3d already generated is
  fed in as a multi-entry SDF, and `PDB_ROTAMERS` is emitted.
- **The conformer file is repaired.** `molfile_to_params` puts conformer 1 in `NAME.pdb`
  and conformers 2..N in `NAME_conformers.pdb`, so the rotamer library is short by one
  until the first is prepended. ligand3d does that and then *counts* the result — the
  library is separated by `TER`, not `MODEL`, so the obvious check silently passes on an
  empty file.
- **The three-letter code is checked** against Rosetta's `residue_types.txt` before any
  work happens. `--allow-code-conflict` overrides it. (`BZO` and `ALA` are both taken,
  for instance.)
- **Atom names are preserved.** ligand3d always passes `--keep-names` to
  `molfile_to_params.py`, so a constraint file that refers to them keeps working.

`ligand3d doctor` reports where it found `molfile_to_params.py`; override with
`LIGAND3D_MOLFILE_TO_PARAMS` or `[rosetta]` in the config.


## Annotated CIF for RFdiffusion4-Proteina

```bash
ligand3d build "<smiles>" -o lig.cif --annotated --rfd-length 100-155
```

Writes `lig.annotated.cif` beside the ordinary one, or tick **annotated CIF
(RFdiffusion4)** in the sketcher. It is the model's native input format: an ordinary
mmCIF with extra `_atom_site` columns saying what to hold fixed.

The mental model that prevents most mistakes:

> **`mask = True` means CONDITIONED. There is no "diffuse this" flag. Diffusion is the
> absence of conditioning.**

So the file says *here is the ligand, in this pose, with this identity* — and leaves
everything else to be generated. What ligand3d contributes is the pose, which is exactly
the thing a binder- or enzyme-design run needs given rather than invented.

| what it writes | why |
|---|---|
| `mask_coordinate_1_atom = True` | pins the conformer ligand3d built. `--rfd-free-pose` turns this off and lets the model choose the geometry |
| `mask_sequence_1_residue = True` | **not optional.** A non-polymer is *atomized*, and an atomized token that is not sequence-conditioned is rejected — ligands are context, never generated |
| `condition_sequence_1_residue` | the residue name |
| an expandable-segment sentinel | one fake atom whose residue name is `100-155`, standing for the protein to build. Collapsed, not realized, so the model samples a length per replicate |

Coordinate conditioning on any non-backbone atom *requires* sequence conditioning, because
the atom37 slot occupancy fingerprints the residue anyway. Every ligand atom is
non-backbone, so both rules point the same way and the combination the validator rejects is
not expressible here.

Omit `--rfd-length` to condition on the ligand alone and supply the contig yourself.

**Verified against the real thing.** Every file is checked with RFD4's own
`check_annotated_cif.py`, which runs the actual inference pipeline — a pass means the model
will accept it. Charged, aromatic, heteroaromatic, zwitterionic and bare-ligand cases all
report **0 errors**, matching the reference files shipped with the format spec.

Two format traps are handled rather than documented, because both fail silently:

- **Masks are the literal strings `True`/`False`.** Dtype inference tries int before bool,
  so `1`/`0` becomes an int64 array and fails much later with "Mask must be a boolean array".
- **Bond orders are Kekulé.** A bond left generically aromatic is cast to single, which
  would tell the model an aromatic ring is saturated. Benzene is written as three `SING` and
  three `DOUB` with `pdbx_aromatic_flag Y`, the way the CCD does it.

Kekulizing is not a way of *discarding* aromaticity, and there is no reason to want it off.
The order and the flag together are how mmCIF expresses an aromatic bond of a given order.
Read back through biotite — which is what RFdiffusion4 uses — the three encodings give:

| what is written | what the reader gets |
|---|---|
| kekulized `SING`/`DOUB` + flag `Y` | `AROMATIC_SINGLE` ×3, `AROMATIC_DOUBLE` ×3 |
| all `SING` + flag `Y` | `AROMATIC_SINGLE` ×6 — the alternation is gone |
| generic `AROM` | `BondType.AROMATIC` ×6 — **no defined order** |

So `AROMATIC_SINGLE` / `AROMATIC_DOUBLE` is very much still the target; kekulizing is how
you hit it. The last row is the one the format spec warns about: atomworks deliberately
omits `BondType.AROMATIC` from its bond-order table because the order is not well defined.
Measured on benzene, not inferred, and pinned by a test.

Formal charges are written as real numbers rather than `?`. Charges reach the model, and a
neutral atom is not an atom of unknown charge.


## Watching the minimization

Tracing is **on by default** — it is what makes a minimization inspectable rather than a
black box, and it does not change the geometry. Turn it off with `--no-trace`.

```bash
ligand3d build "<smiles>" --backend mmff94,gfn2 --trajectory
ligand3d build "<smiles>" --no-trace          # quieter
ligand3d build "<smiles>" --dry-run           # build and check, write nothing
```

The trace logs the energy at every optimizer step with the change from the previous step,
kept **separate per method** — a chained `mmff94,gfn2` run reports two blocks with their
own step counts and net changes, and no delta ever bridges the boundary, because a strain
energy and a total electronic energy are not comparable. The total wall time and a
per-method breakdown are always printed.

`--trajectory` writes `<name>_traj.pdb`, one MODEL per kept frame — every tenth step
by default, set with `--trajectory-every`, and the converged geometry is always the
last frame. Each carries the energy and the
responsible method in REMARKs, so it animates in PyMOL or ChimeraX.

In the browser this becomes a graph. The x axis is the **cumulative** step count, because
the stages genuinely run one after another — a method that takes 20 steps after 502 of the
first is drawn at 502–522, not back at zero. The y axis is ΔE from **each stage's own
first step**, and the curves are deliberately *not* joined end to end: absolute energies
from two methods share no scale, so connecting them would read as one continuous descent
and imply the two drops add up. They do not. The absolute final energy of each stage is
printed under the plot instead.

One caveat worth knowing: RDKit's force fields have no per-step callback, so tracing them
means asking for one iteration at a time, which restarts the optimizer's state and
descends less efficiently. ligand3d finishes with an uninterrupted pass so the geometry
you get is identical to an untraced run (verified to 1e-3 kcal/mol); tracing costs time
and nothing else. GFN-FF cannot be traced at all — xtb optimizes inside its own process —
and says so rather than inventing a curve.

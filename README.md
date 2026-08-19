# ligand3d

Turn a 2D molecule into a minimized 3D structure.

You give it a SMILES string, a MOL/SDF file, or a structure you drew in the browser.
It builds a 3D conformer, optionally sets the protonation state, minimizes with the
level of theory you choose, optionally searches conformers, and writes a `.pdb`.

```bash
ligand3d build "O=C1CN2CCC1CC2" -o quinuclidinone          # writes .cif + .sdf
ligand3d build "NCC1(CC(=O)O)CCCCC1" --ph 7.4 --backend gfn2 -o gabapentin
ligand3d build "C[C@@H](O)[C@H](N)C(=O)O" --backend mmff94,gfn2 --confs 20 -o threonine
ligand3d build "<smiles>" --params --params-code LIG        # + a Rosetta params file
ligand3d build "<smiles>" --trace --trajectory              # energy per step, and the path
ligand3d sketch                        # draw it, then the pipeline runs on what you drew
ligand3d doctor                        # what's installed, what isn't, and why
```

Every stage is also its own command, so a script can run one step at a time:

```bash
ligand3d stereo     "C[C@@H](O)[C@H](N)C(=O)O"   # report R/S and E/Z, build nothing
ligand3d protonate  "NCC1(CC(=O)O)CCCCC1" --ph 7.4 --all
ligand3d embed      "<smiles>" -o raw            # 2D -> 3D, no minimization
ligand3d minimize   raw.sdf -o min --backend gfn2 --trace
ligand3d conformers "<smiles>" -n 50 -o ensemble
ligand3d params     ensemble.sdf --code LIG
ligand3d convert    min.cif min.pdb
```

## Why this exists

Getting from a drawing to a usable 3D ligand is a five-minute job that everyone
re-implements badly. The parts that are easy to get wrong are the ones this tool
takes seriously:

- **Stereochemistry is verified, not assumed.** After embedding, the 3D structure's
  CIP labels are re-perceived and compared against what you drew. A mirrored
  stereocenter is an error, not a surprise you find three weeks later.
- **Your protonation state survives.** Minimizing gabapentin's zwitterion in gas phase
  destroys it — the proton hops back from N to O and you silently get a different
  molecule. ligand3d turns on implicit solvation for charged species and checks the
  connectivity afterwards.
- **Backends declare what they can do.** Some potentials consume total charge
  (GFN2-xTB, AIMNet2) and some do not (MACE-OFF, MACE-MP). Handing a carboxylate to a
  model with no charge channel is refused rather than quietly answered.

## Install

```bash
uv pip install -e .                    # core: RDKit only. 2D→3D plus MMFF94/UFF
uv pip install -e ".[xtb]"             # GFN1/GFN2-xTB via tblite wheels
uv pip install -e ".[protonation]"     # pH-based protonation via dimorphite-dl
uv pip install -e ".[mace]"            # MACE potentials      (see the split below)
uv pip install -e ".[fairchem]"        # eSEN / UMA / AllScAIP (see the split below)
```

The core install is small and needs no compiler and no conda. For anything with `torch`
in it, install torch from the CPU index first unless you have a GPU, or pip pulls the
multi-gigabyte CUDA build:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[xtb,protonation,mace]"
```

`[mace]` and `[fairchem]` cannot coexist — see
[the split](#the-mace--fairchem-split--read-this-before-installing).

## Backends

| id | kind | takes charge | solvation | speed (gabapentin) |
|---|---|---|---|---|
| `mmff94` | classical FF | implicit in typing | no | 5 ms |
| `uff` | classical FF | implicit in typing | no | 5 ms |
| `gfnff` | classical FF | yes | ALPB | ~0.1 s |
| `gfn2` | semi-empirical | yes | ALPB | 0.9 s |
| `gfn1` | semi-empirical | yes | ALPB | ~0.8 s |
| ML potentials | see below | varies | no | 0.5 s and up |

Chain them with a comma — cheap first, expensive last:

```bash
ligand3d build "NCC1(CC(=O)O)CCCCC1" --backend mmff94,gfn2
```

`ligand3d models` is the full reference — cost, memory, charge and spin handling,
training data, element coverage, and the resolved path of every checkpoint:

```bash
ligand3d models              # the table
ligand3d models --available  # only what runs here
ligand3d models -v           # plus training data, notes, and weight paths
```

The browser has the same thing at `/models`, filterable, linked from the header.
`ligand3d backends` is the short version; `ligand3d doctor` diagnoses what is missing.

Chains are not limited to the presets. Any comma-separated sequence works, on the command
line or via **build my own chain** in the browser:

```bash
ligand3d build "<smiles>" --backend mmff94,mace-mh,mace-off-large
```

The first method searches the conformers and the rest refine only the survivors, so put
the cheap one first.

## Implicit solvent

`gfn1`, `gfn2`, and `gfnff` support ALPB implicit solvation with 25 parameterized
solvents. `ligand3d solvents` lists them with dielectric constants:

```bash
ligand3d build "<smiles>" --backend gfn2 --solvent dmso
ligand3d build "<smiles>" --backend gfn2 --solvent woctanol   # the logP phase
ligand3d solvents
```

water, methanol, ethanol, acetonitrile, dmso, dmf, acetone, thf, dichloromethane,
chloroform, ethylacetate, diethylether, dioxane, toluene, benzene, hexane, hexadecane,
octanol, woctanol, phenol, aniline, benzaldehyde, nitromethane, furane, carbondisulfide —
plus the aliases `h2o`, `ch2cl2`, `chcl3`, `ether`, `cs2`.

Solvent names are validated before any work starts, because tblite rejects an unknown one
with a message that does not say what the alternatives are. **Cyclohexane, octane and
heptane are not in the ALPB table** despite being obvious things to reach for, so asking
for them names the nearest stand-in instead of just failing:

```console
$ ligand3d build "CC(=O)O" --backend gfn2 --solvent cyclohexane
error: ALPB has no parameters for 'cyclohexane'. The closest available solvent
is 'hexane'. Run 'ligand3d solvents' for the full list.
```

Water is still applied automatically to charged and zwitterionic species unless you pass
`--solvent none`. The machine-learned potentials have no implicit solvent model at all,
which is why they are the wrong tool for a zwitterion regardless of charge handling.

## Machine-learned force fields

**On this cluster the checkpoints live in `/net/databases/huggingface/mlFF_models/`, and
ligand3d finds them there automatically** — for example `mace-polar` resolves to
`models--ACEsuit--mace-polar-1-beta/MACE-POLAR-1-M.model`. `ligand3d models -v` prints the
resolved path for every model so there is no guessing about which file is being loaded. That path is one of the built-in probe
locations, so nothing needs configuring — `ligand3d doctor` prints exactly which file it
resolved for each model. On any other machine, point `LIGAND3D_<MODEL>` at a checkpoint
or list it under `[weights]` in `~/.config/ligand3d/config.toml`.

The column that matters most is **charge**: a potential with no charge channel cannot
tell a carboxylate from a neutral acid, so ligand3d refuses that pairing rather than
answering it. Every flag below was determined by measurement — setting
`atoms.info["charge"]` and checking whether the energy actually moved.

| backend id | model | charge | size | notes |
|---|---|---|---|---|
| `mace-off` | MACE-OFF23 medium | no | 18 MB | neutral organics; the sensible MACE default |
| `mace-off-small` | MACE-OFF23 small | no | 7 MB | fastest MACE |
| `mace-off-large` | MACE-OFF23 large | no | 55 MB | most accurate MACE-OFF |
| `mace-off-24` | MACE-OFF24 medium | no | 18 MB | successor to OFF23 |
| `mace-omol` | MACE-omol-0 XL-1024 | **yes** | 422 MB | OMol25-trained, charge-aware MACE |
| `mace-mh` | MACE multi-head 0 | no | 40 MB | `omol` head selected |
| `mace-mh-1` | MACE multi-head 1 | no | 59 MB | `omol` head selected |
| `mace-mh-spice` | MACE multi-head 0 | no | 40 MB | `spice_wB97M` head |
| `mace-mp` | MACE-MP-0 (MatPES r2SCAN) | no | 79 MB | materials; broad element coverage |
| `esen` | eSEN sm conserving | **yes** | 51 MB | OMol25; best accuracy per second here |
| `esen-sm-direct` | eSEN sm direct | **yes** | 51 MB | faster, not energy-conserving |
| `esen-md-direct` | eSEN md direct | **yes** | 406 MB | |
| `uma-s` | UMA s 1.1 | **yes** | 1.2 GB | universal; `omol` task |
| `uma-s-1p2` | UMA s 1.2 | **yes** | 2.3 GB | |
| `uma-sm` | UMA sm | **yes** | 1.2 GB | |
| `uma-m` | UMA m 1.1 | **yes** | 11 GB | slow; needs real memory |
| `allscaip` | AllScAIP OMol102M cons | **yes** | 688 MB | |
| `allscaip-direct` | AllScAIP OMol102M direct | **yes** | 695 MB | |
| `aimnet2` | AIMNet2 | **yes** | 35 MB | fastest charge-aware option |

The multi-head MACE checkpoints refuse to load without a head selected — none of them
names one `default` — so ligand3d picks the head for you (`omol`, or `spice_wB97M` for
`mace-mh-spice`).

### The MACE / fairchem split — read this before installing

`mace-torch` pins `e3nn==0.4.4` and `fairchem-core` requires `e3nn>=0.5`. **They cannot
share one environment.** Whichever is installed second wins, and the loser dies
deserializing its checkpoint with `ValueError: too many values to unpack` from inside
e3nn's codegen — which tells you nothing about the real cause. ligand3d detects the
mismatch and says so plainly in `ligand3d doctor`.

So keep two environments:

```bash
uv pip install -e ".[xtb,protonation,mace]"      # MACE + AIMNet2 + xTB
uv pip install -e ".[xtb,protonation,fairchem]"  # eSEN / UMA / AllScAIP + xTB
```

Install torch from the CPU index first in both unless you have a GPU.

### MACE-POLAR — a third environment

MACE-POLAR-1 (S/M/L) models long-range electrostatics that the other MACE models leave
out, and it does work — all three sizes load and evaluate. It needs two things that are
not on PyPI: `graph_longrange`, and a **patched MACE fork** that installs *as*
`mace-torch` and therefore replaces the stock package. So it is a third mutually
exclusive environment, not something that sits beside `[mace]`.

On this cluster both sources are in `quantum_cowboy_biochemistry/deps/`:

```bash
uv venv --python 3.12 .venv-polar && source .venv-polar/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install <repo>/deps/mace_polar_src <repo>/deps/graph_longrange_src
uv pip install -e /path/to/ligand3d
ligand3d build "<smiles>" --backend mmff94,mace-polar
```

### Still not usable

- **SO3LR v2 beta** is a JAX model needing jax, orbax, and `so3lr`; every ligand3d
  backend is torch or ASE based.
- **orb-mol-conservative** (99 MB) loads, but `orb-models` 0.7 removed
  `orb_models.forcefield.calculator`, so there is no ASE calculator to attach.

`ligand3d doctor` and `ligand3d models` list these with the same explanation rather than
pretending they aren't there.

## Protonation

The default is **what you drew**. If you type a neutral carboxylic acid you get a
neutral carboxylic acid.

```bash
ligand3d build "NCC1(CC(=O)O)CCCCC1"                       # as drawn
ligand3d build "NCC1(CC(=O)O)CCCCC1" --protonate           # dimorphite-dl at pH 7.4
ligand3d build "NCC1(CC(=O)O)CCCCC1" --ph 2.0              # at pH 2
ligand3d build "NCC1(CC(=O)O)CCCCC1" --protonate --enumerate-states  # one file per state
```

## What it refuses to do

Three inputs are rejected rather than answered, because the answer would be wrong
in a way that looks right:

- **Undefined stereochemistry** — unless you pass `--stereo any` or
  `--stereo enumerate`. Constrained centers that only look stereogenic (the
  bridgeheads of 3-quinuclidinone, say) are not an ambiguity and do not trigger this.
- **Disconnected fragments** — a salt or solvate. Distance geometry has no restraints
  between components and stacks them on top of each other, at measured separations of
  0.0 Å. Use `--largest-fragment` to keep the biggest component and drop the counterion.
- **A charged molecule on a potential with no charge channel**, or a zwitterion on one
  with no implicit solvent. Override with `--allow-charge-mismatch` if you mean it.

After minimization it also verifies that stereochemistry, protonation state, and
heavy-atom connectivity all survived — on every conformer, not just the first.

## Conformers — and what the default actually does

**Every build searches.** Asking for one output structure does not mean one guess: a
batch of conformers is generated with ETKDG, minimized with the cheap force field, and
only the best `--confs` are kept. The count scales with rotatable bonds (20 for a rigid
cage, up to 300 for something floppy) and is overridable with `--sample`.

This matters more than it sounds. Minimizing a single ETKDG guess is a *local*
minimization, and for gabapentin the answer moved by **9.6 kcal/mol** depending only on
the random seed:

| | best energy found |
|---|---|
| one guess, 5 different seeds | −7.46, −17.03, −9.49, −8.74, −15.15 |
| searching (the default now) | −17.03, −17.03, −17.06, −17.25, −17.32 |

The cost is about half a second.

```bash
ligand3d build "<smiles>"                    # searches ~60, keeps the best 1
ligand3d build "<smiles>" --confs 20         # searches, keeps 20
ligand3d build "<smiles>" --sample 500 -n 50 # search harder
ligand3d build "<smiles>" --sample 1         # skip the search: one guess, minimized
ligand3d conformers "<smiles>" -n 50 --method crest
```

ETKDG is a genuine stochastic global sampler — independent distance-geometry starts with
torsion preferences from CSD statistics — not a walk from one structure. Survivors are
de-duplicated by symmetry-corrected heavy-atom RMSD and ranked by energy. CREST does far
more (metadynamics at the GFN level, minutes instead of seconds) and is what to reach for
when the answer really matters.

**Chained backends search cheaply and refine narrowly.** `--backend mmff94,gfn2` runs
MMFF94 over the whole sample, prunes to the survivors, and only then runs GFN2 on those:

```
searched 57 conformer(s) via rdkit, keeping the best 2
mmff94 narrowed 57 to 2; refining with gfn2
minimization time: mmff94 0.29s, gfn2 0.45s
```

Running both methods over all 57 would have cost about thirty seconds of GFN2 to rediscover
shapes MMFF94 already found.

## Drawing

`ligand3d sketch` starts a local server, opens a browser, and stays up. Draw a structure,
set the options, press **Build**, read the run log, clear the canvas, and draw the next
one — no reloading between molecules.

```bash
ligand3d sketch                          # output goes to the current directory
ligand3d sketch -d ~/ligands -b gfn2     # preselect a directory and backend
ligand3d sketch --no-browser             # print the URL (SSH port-forwarding)
```

The page is laid out as editor on the left, work area on the right: settings grouped into
four tabs (Output, Minimize, Chemistry, Rosetta) so the panel stays short, then the energy
graph, then the run log. Both the editor and the log grow to fill the window.

- **Output path and formats**, shown resolved in full before anything is written. The name
  field is a base name, since one build can produce an mmCIF, a PDB, an SDF, a trajectory
  and a params set; tick the formats you want. A directory that does not exist yet is
  flagged and created on build; an unwritable location is an error before you spend a
  minimization on it.
- **Auto-incrementing names** — `sketch0`, `sketch1`, skipping any base name already taken
  in *any* format. If a build would replace existing files you get a dialog listing exactly
  which, and nothing is written until you confirm.
- **Every build option**, including the Rosetta params tab and the trace and trajectory
  toggles. Each maps to a `ligand3d build` flag.
- **An energy graph** when tracing is on: one curve per method, each plotted as the change
  from its own first step (a strain energy and a total electronic energy share no scale),
  with the step count and net change in the legend.
- **A run log** reporting stereocenters with R/S, double bonds with E/Z and cis/trans,
  warnings such as more than one fragment, any error in full, per-method timing, the total
  time, and every file written. There is a Copy button.

### Seeing how your drawing is read

Under the editor is a live panel showing the molecule **as ligand3d parses it**, redrawn
by RDKit with every atom numbered, updating as you draw.

This exists because of one specific failure. When the pipeline says

```
2 stereocenter(s) left undefined: atom 4, atom 6
```

those numbers index the file your sketcher emitted, and there is no way to tell from the
canvas which atoms they are — the same molecule drawn twice can even number differently.
The panel makes the number point at something you can see. Atoms are colour-coded:

- **amber** — needs a configuration from you
- **green** — already specified, with its R/S annotation
- **grey** — looks stereogenic to a graph analysis but is fixed by the ring system, so
  there is nothing to decide

Atom numbering can be toggled off, and the panel follows your light or dark theme.

The messages also say *what kind* of stereochemistry is missing, which is often not what
you would guess. Two flagged atoms on the same ring are not two independent wedges to
draw — they are one cis/trans relationship:

```
atoms 3, 5 sit on the same 4-membered ring, so the ambiguity is whether their
substituents are on the same face (cis) or opposite faces (trans). Put a wedge on
one substituent bond and a wedge or a dash on the other to say which.
```

### Drawing stereochemistry — wedges *and* dashes

Select the wedge-bond tool, then **click the same bond repeatedly to cycle it**: solid
wedge → dashed (hashed) → plain. There is no separate dash tool; the one tool cycles. A
wedge and a dash on the same drawing give opposite configurations, which is verified in
the test suite — molfile bond flag 1 reads back as *R* and flag 6 as *S* for the same 2D
layout.

Whatever you draw is checked: after embedding and again after minimization the CIP labels
are re-perceived from the 3D coordinates and compared against your drawing, so a
stereocenter cannot silently flip.

### The editor

[JSME](https://jsme-editor.github.io/) is fetched once (about 1 MB) into
`~/.cache/ligand3d/` and works offline thereafter. Nothing is sent anywhere: the server
binds `127.0.0.1` only, and it shuts down when you stop it.

If it cannot be fetched the same page shows a paste box instead, accepting a molblock or a
SMILES string. Every other control — settings, run log, overwrite protection — works
unchanged, so an air-gapped machine loses only the drawing canvas.

[Ketcher](https://github.com/epam/ketcher) was evaluated and deliberately not used. EPAM
publishes it as an npm library rather than a servable page: `ketcher-standalone.zip`
contains `index.js`, `main.js`, and type declarations, and no HTML file anywhere, so it
cannot be unzipped and served. Supporting it would mean requiring a node toolchain to
build an editor that JSME already provides in 1 MB.

## Configuration

Model weights and external binaries are never bundled. Each resolves through environment
variables, then `~/.config/ligand3d/config.toml`, then `$PATH` and conventional install
locations, then `~/.cache/ligand3d/`. Generate a starter config with:

```bash
ligand3d config --init
```

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
```

Nothing in the params path needs a PDB — `molfile_to_params` reads the SDF directly.

`--dry-run` (or unticking every format in the browser) builds, minimizes, checks and
reports without writing anything, which is the quick way to try a molecule before
committing to a filename.

Charges survive everywhere: the mmCIF carries `pdbx_formal_charge` per atom, so a
zwitterion reads back as `[NH3+]CC1(CC(=O)[O-])CCCCC1` with both sites intact.

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
- Atom names are preserved with `--keep-names`, so a constraint file that refers to them
  keeps working.

`ligand3d doctor` reports where it found `molfile_to_params.py`; override with
`LIGAND3D_MOLFILE_TO_PARAMS` or `[rosetta]` in the config.

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

`--trajectory` writes `<name>_traj.pdb`, one MODEL per step with the energy and the
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

## Stereochemistry reporting

```console
$ ligand3d stereo "C[C@@H](O)[C@H](N)C(=O)O"
C4H9NO3  C[C@@H](O)[C@H](N)C(=O)O
  2 stereocenter(s): atom 1 = R, atom 3 = S

$ ligand3d stereo "OC(=O)/C=C\C(=O)O"
C4H4O4  O=C(O)/C=C\C(=O)O
  double bond 3-4: Z (cis)
```

**E/Z and cis/trans are not synonyms**, and ligand3d only claims the second where it is
defensible. E/Z comes from CIP priorities and is always well defined. cis/trans compares
two *reference* substituents, which is only meaningful when it is obvious which two are
meant — that is, when each alkene carbon carries exactly one hydrogen. There, Z is cis and
E is trans, and both labels are printed.

For a tri- or tetrasubstituted alkene "cis to what?" has no single answer, so only E/Z is
reported:

```console
$ ligand3d stereo "CC/C(=C(\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1"
  double bond 2-3: Z (cis/trans does not apply: tetrasubstituted alkene)
```

Tamoxifen is the standard trap: unambiguously (Z) by CIP, and described as "trans" in
older literature with respect to the two phenyls. Both statements are about the same
molecule.

## Built on

[RDKit](https://www.rdkit.org/) (ETKDGv3 embedding, MMFF94/UFF),
[tblite](https://github.com/tblite/tblite) (GFN1/GFN2-xTB),
[xtb](https://github.com/grimme-lab/xtb) and [CREST](https://github.com/crest-lab/crest),
[dimorphite-dl](https://github.com/durrantlab/dimorphite_dl) (protonation states),
[MACE](https://github.com/ACEsuit/mace) and
[AIMNet2](https://github.com/isayevlab/aimnetcentral) (ML potentials),
[ASE](https://wiki.fysik.dtu.dk/ase/) (optimizers),
[JSME](https://jsme-editor.github.io/) and
[Ketcher](https://github.com/epam/ketcher) (2D sketchers).

## Author

Seth M. Woodbury — [github.com/SethWoodbury](https://github.com/SethWoodbury)

## License

MIT.

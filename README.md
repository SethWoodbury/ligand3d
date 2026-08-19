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

`ligand3d backends` lists what is registered; `ligand3d doctor` says which ones can
actually run here and what to install for the rest.

## Machine-learned force fields

**On this cluster the checkpoints live in `/net/databases/huggingface/mlFF_models/`, and
ligand3d finds them there automatically.** That path is one of the built-in probe
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

### Present on the cluster but not usable

- **MACE-POLAR-1** (S/M/L, 191 MB) needs the `graph_longrange` package and a patched
  MACE fork (`mace.modules.extensions.PolarMACE`), neither of which is on PyPI.
- **SO3LR v2 beta** is a JAX model needing jax, orbax, and `so3lr`; every ligand3d
  backend is torch or ASE based.
- **orb-mol-conservative** (99 MB) loads, but `orb-models` 0.7 removed
  `orb_models.forcefield.calculator`, so there is no ASE calculator to attach.

`ligand3d doctor` lists these with the same explanation rather than pretending they
aren't there.

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

## Conformers

```bash
ligand3d build "<smiles>" --confs 50                 # ETKDG multi-embed, prune, cluster
ligand3d build "<smiles>" --confs 50 --conf-method crest   # CREST metadynamics
```

The RDKit method takes seconds. CREST takes minutes and finds far more — 145 unique
conformers for gabapentin in about five wall-clock minutes on eight threads — and needs
a `crest` binary, which `ligand3d doctor` will locate or tell you how to provide.

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

Both are off by default because they cost time.

```bash
ligand3d build "<smiles>" --backend mmff94,gfn2 --trace --trajectory
```

`--trace` logs the energy at every optimizer step with the change from the previous step,
kept **separate per method** — a chained `mmff94,gfn2` run reports two blocks with their
own step counts and net changes, and no delta ever bridges the boundary, because a strain
energy and a total electronic energy are not comparable. The total wall time and a
per-method breakdown are always printed.

`--trajectory` writes `<name>_traj.pdb`, one MODEL per step with the energy and the
responsible method in REMARKs, so it animates in PyMOL or ChimeraX.

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

## License

MIT.

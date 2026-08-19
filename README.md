# ligand3d

Turn a 2D molecule into a minimized 3D structure.

You give it a SMILES string, a MOL/SDF file, or a structure you drew in the browser.
It builds a 3D conformer, optionally sets the protonation state, minimizes with the
level of theory you choose, optionally searches conformers, and writes a `.pdb`.

```bash
ligand3d build "O=C1CN2CCC1CC2" -o quinuclidinone.pdb
ligand3d build "NCC1(CC(=O)O)CCCCC1" --ph 7.4 --backend gfn2 -o gabapentin.pdb
ligand3d build "C[C@@H](O)[C@H](N)C(=O)O" --backend mmff94,gfn2 --confs 20 -o threonine.pdb
ligand3d sketch                        # draw it, then the pipeline runs on what you drew
ligand3d doctor                        # what's installed, what isn't, and why
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

The page gives you:

- **Output path and filename**, shown resolved in full before anything is written. Both
  are editable. A directory that does not exist yet is flagged and created on build; an
  unwritable location is an error before you waste time on a minimization.
- **Auto-incrementing names** — `sketch0.pdb`, `sketch1.pdb`, and so on, skipping any
  already present. If a build would replace an existing file you get a dialog listing
  exactly which files, and nothing is written until you say so.
- **Every build option**: the backend chain, implicit solvent, protonation mode and pH,
  conformer count and search method, energy window, stereo policy, residue name,
  threads, and the override switches. Each one maps to a `ligand3d build` flag.
- **A run log** reporting how many stereocenters were found and their R/S assignments,
  double-bond geometry, warnings such as more than one fragment in the drawing, any
  error in full, and the path of every file written.

### Drawing stereochemistry — wedges *and* dashes

Select the wedge-bond tool, then **click the same bond repeatedly to cycle it**: solid
wedge → dashed (hashed) → plain. There is no separate dash tool; the one tool cycles. A
wedge and a dash on the same drawing give opposite configurations, which is verified in
the test suite — molfile bond flag 1 reads back as *R* and flag 6 as *S* for the same 2D
layout.

Whatever you draw is checked: after embedding and again after minimization the CIP labels
are re-perceived from the 3D coordinates and compared against your drawing, so a
stereocenter cannot silently flip.

### Editors

The default is [JSME](https://jsme-editor.github.io/), fetched once (about 1 MB) into
`~/.cache/ligand3d/` and offline thereafter. Nothing is sent anywhere: the server binds
`127.0.0.1` only.

[Ketcher](https://github.com/epam/ketcher) is a nicer editor and is supported, but EPAM
publishes it as an npm library rather than a servable page — `ketcher-standalone.zip`
contains `index.js`, `main.js`, and type declarations, and no HTML at all — so it cannot
be fetched and used automatically. Build it yourself, point `LIGAND3D_KETCHER_DIR` at the
directory holding the resulting `index.html`, and it is used in preference to JSME. The
pinned upstream source is the `vendor/ketcher` submodule, which a normal clone does not
download.

If neither can be reached the page degrades to a paste box accepting SMILES or a
molblock, so the command still works on an air-gapped machine.

## Configuration

Model weights and external binaries are never bundled. Each resolves through environment
variables, then `~/.config/ligand3d/config.toml`, then `$PATH` and conventional install
locations, then `~/.cache/ligand3d/`. Generate a starter config with:

```bash
ligand3d config --init
```

## Output

A `.pdb` with unique atom names, CONECT records, formal charges, and provenance REMARKs.
Multiple conformers become MODEL/ENDMDL records. An `.sdf` sidecar is written alongside,
because PDB does not carry bond orders reliably and you will want them back.

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

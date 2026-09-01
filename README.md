# ligand3d

Turn a 2D molecule into a minimized 3D structure.

You give it a SMILES string, a MOL/SDF file, or a structure you drew in the browser.
It builds a 3D conformer, optionally sets the protonation state, minimizes with the
level of theory you choose, optionally searches conformers, and writes mmCIF, SDF, PDB,
a Rosetta params set, or an annotated CIF for RFdiffusion4 — mmCIF by default, because it
carries the bond orders and formal charges PDB cannot.

```bash
ligand3d build "O=C1CN2CCC1CC2" -o quinuclidinone          # writes .cif + .sdf
ligand3d build "NCC1(CC(=O)O)CCCCC1" --ph 7.4 --backend gfn2 -o gabapentin
ligand3d build "C[C@@H](O)[C@H](N)C(=O)O" --backend mmff94,gfn2 --confs 20 -o threonine
ligand3d build "<smiles>" --params --params-code LIG        # + a Rosetta params file
ligand3d build "<smiles>" --trajectory                      # save the geometry at every step
ligand3d fetch "3-Cyano-7-ethoxycoumarin"  # look a molecule up by name
ligand3d sketch                        # draw it, then the pipeline runs on what you drew
ligand3d doctor                        # what's installed, what isn't, and why
```

Every stage is also its own command, so a script can run one step at a time:

```bash
ligand3d stereo     "C[C@@H](O)[C@H](N)C(=O)O"   # report R/S and E/Z, build nothing
ligand3d protonate  "NCC1(CC(=O)O)CCCCC1" --ph 7.4 --all
ligand3d embed      "<smiles>" -o raw            # 2D -> 3D, no minimization
ligand3d minimize   raw.sdf -o min --backend gfn2
ligand3d conformers "<smiles>" -n 50 -o ensemble
ligand3d params     ensemble.sdf --code LIG
ligand3d convert    min.cif min.pdb
```

## Quickstart

### At the IPD, on digs — nothing to install

The containers live on `/net` and are world-readable, so every node can already see them
and you need no environment of your own:

```bash
export PATH=/net/software/containers/users/woodbuse/ligand3d:$PATH   # put this in ~/.bashrc

ligand3d build "O=C1CN2CCC1CC2" -o quin.cif   # SMILES -> minimized 3D
ligand3d sketch                                # or draw it in a browser
ligand3d doctor                                # what is available here, and why
```

That is the whole setup — no clone, no venv, no `module load`. `/net` and `/home` are
automounted on every node, so the same line works from a login node or a compute node, and
your labmates can use it by adding the same line.

**To open the sketcher over SSH, forward the port.** The server binds `127.0.0.1` and
never listens on an external interface, so a tunnel is the only way to reach it. The port
is stable at 8765, so the same tunnel works every session:

```bash
# on digs
$ ligand3d sketch
ligand3d sketcher (jsme) running at http://127.0.0.1:8765/
draw, set the options, and press Build. Ctrl-C to stop.
  to reach it from your machine, run this locally first:
    ssh -N -L 8765:127.0.0.1:8765 <the node it names>
  then open http://127.0.0.1:8765/ there.
```

Copy that `ssh` line into a second terminal **on your laptop** — `sketch` has already
filled in the node's real hostname — then open <http://127.0.0.1:8765/> in your browser.
Leave the `ssh -N` running; it is the tunnel.

At a machine that has a browser and a display, `ligand3d sketch` just opens one and none
of this applies.

Bare `sketch` starts in the richest image present, because the backend is chosen in the
browser *after* the image has been picked — so you get 18 backends rather than the core
image's 6. `sketch -b esen` starts on the fairchem side instead.

### Anywhere else, or to work on the code

```bash
git clone git@github.com:SethWoodbury/ligand3d && cd ligand3d
uv venv && uv pip install -e ".[xtb,protonation,names]"
uv run ligand3d sketch
```

Add `,mace` **or** `,fairchem` for the neural potentials — not both; see
[the split](#the-mace--fairchem-split--read-this-before-installing). Working from a
checkout is also the only way to use `--slurm`, which needs `sbatch` and cannot run from
inside a container.

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

The [Quickstart](#quickstart) covers the two normal cases. This is the full menu of
extras, for a checkout:

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
[the split](#the-mace--fairchem-split--read-this-before-installing). `[names]` adds
offline IUPAC-name lookup for `fetch`, which also needs a JRE on `PATH`.

Verified from a clean clone: `git clone`, `uv venv`, `uv pip install -e
".[xtb,protonation,names]"`, then `uv run ligand3d build ...` and `uv run ligand3d sketch`
all work with nothing else present.

## Releasing it to the lab

Using it needs nothing but the `PATH` line in the [Quickstart](#quickstart) — this section
is for the person cutting the release.

`container/build.sh` produces a directory holding the images, a launcher, and a `VERSION`
note saying what is in them. Everything is world-readable, so a labmate needs only that
one `PATH` entry.

```bash
container/build.sh                    # into ./dist
container/build.sh /net/software/containers/users/woodbuse/ligand3d
```

A shared destination is built locally and copied at the end, each image verified by
checksum, with the release it replaced kept beside it as `.previous`. A failed build or a
bad copy leaves the live release untouched. The build context is a filtered export of the
checkout rather than the checkout itself — otherwise `%files` copies `.venv`, which is 1.6
GB of the wrong platform's wheels and turns a four-minute release into twenty.

The build runs `container/selfcheck.py` inside each image and **fails rather than producing
one that cannot do what it claims** — imports, a JRE for OPSIN, every required backend, the
family's own neural module, and a real end-to-end build. That check has earned itself
three times: a missing `libgomp1`, which made GFN1/GFN2 fail at import with a message
naming no library; `xtb` absent so GFN-FF was silently unavailable; and `graph_longrange`
missing, which would have shipped a MACE image whose POLAR models could not be unpickled.

**The source is baked into the image**, so a run is pinned to a released version rather
than to whatever happens to be checked out — a colleague's results cannot change because you were mid-edit. The
trade is that a code change needs a rebuild before it reaches anyone.

To cut a release:

```bash
container/build.sh                    # into ./dist
container/build.sh /net/software/containers/users/woodbuse/ligand3d
```

The build runs `container/selfcheck.py` inside the image and **fails rather than producing
one that cannot do what it claims** — imports, a JRE for OPSIN, every required backend, and
a real end-to-end build. That check has already earned itself twice: once on a missing
`libgomp1`, which made GFN1/GFN2 fail at import with a message naming no library, and once
on `xtb` being absent so GFN-FF was silently unavailable.

### Why there are three images, and not one or twenty

**One image per dependency conflict.** There is exactly one: `mace-torch` pins
`e3nn==0.4.4` and `fairchem-core` needs `e3nn>=0.5`. That is two groups, so:

| image | size | neural family |
|---|---|---|
| `ligand3d.sif` | 394 MB | none |
| `ligand3d-mace.sif` | 770 MB | MACE, MACE-POLAR, AIMNet2 |
| `ligand3d-fairchem.sif` | 1.1 GB | eSEN, UMA, AllScAIP |

A container per *method* would be the wrong shape: all the MACE variants share one
environment happily, and torch is most of the size, so twenty images would mean twenty
copies of torch to solve a problem that has two sides.

**Every image carries the full non-neural feature set** — tblite, protonation, name
lookup, the sketcher, params, the annotated CIF. That is the property that matters, and
it is what the earlier arrangement got wrong: borrowing the lab's general-purpose
`quantum_chem` image gave you MACE but no tblite, so `-b gfn2,mace-off` failed partway
through with an error telling you to pip install into a read-only container. Now the whole
chain runs in one place.

The launcher reads the backend and picks the image. If a family image is missing it falls
back to the lab's, warns, and says what is unavailable in it.

`container/build.sh` builds all three; `LIGAND3D_FAMILIES="core mace"` builds a subset.

Two things do not work from a container, and the launcher says so rather than failing
obscurely:

- **`--slurm`** needs `sbatch`, whose auth plugins cannot be reached from inside a
  container. Submit from a checkout.
- **`--container`** is redundant here and nested apptainer does not work anyway; the
  launcher already does that dispatch.

### Or install it normally

Nothing about the container is required. A checkout works as it always has, and is what
you want if you are editing the code or submitting to SLURM:

```bash
git clone git@github.com:SethWoodbury/ligand3d && cd ligand3d
uv venv && uv pip install -e ".[xtb,protonation,names]"
```

## Backends

| id | kind | takes charge | solvation | speed (gabapentin) |
|---|---|---|---|---|
| `mmff94` | classical FF | implicit in typing | no | 4 ms |
| `uff` | classical FF | implicit in typing | no | 4 ms |
| `gfnff` | generic FF (xtb) | yes | ALPB | 0.06 s |
| `gfn2` | semi-empirical | yes | ALPB | 0.33 s |
| `gfn1` | semi-empirical | yes | ALPB | 0.57 s |
| ML potentials | see below | varies | no | 0.5 s to a minute |
| `orca` | DFT | yes (+ spin) | CPCM | 25 s for methanol, and it climbs steeply |

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

## Adding a method

A backend declares what it can do and implements one function. The pipeline
reads the declaration and routes around it — the charge-channel refusal, the
zwitterion solvation check, the element check and the chaining all come from
`Capabilities` rather than from anything the backend does.

```python
class MyBackend(ASEBackend):
    caps = Capabilities(
        name="mymethod", kind="mlff",
        takes_charge=True, supports_solvation=False,
        fixed_topology=False, energy_kind="total",
        requires=("mypackage",),
    )

    def make_calculator(self, job):
        import mypackage
        return mypackage.ASECalculator(charge=job.charge)

register("mymethod", MyBackend)
```

That is the whole contract for anything with an ASE calculator, which is most
things. **DFT was added this way as the test of it**: one new module, plus three
one-line edits — adding `"dft"` to a `Literal`, to a module list, and to a sort
order. Charge, spin, solvent and chaining needed no work at all.

A new *checkpoint* for an existing family is smaller still: one `ModelSpec` in
`config.py` naming the file and what it was trained on. A new **classical**
method that is not ASE-shaped implements `minimize()` directly instead —
`rdkit_ff.py` and the GFN-FF path in `xtb.py` both do, because they optimise
internally rather than returning gradients.

Two things a new method should get right, because they are what the rest of the
tool reasons about:

- **`takes_charge`** decides whether a charged molecule is allowed near it. Get
  this wrong and a carboxylate is silently treated as a neutral acid.
- **`fixed_topology`** decides whether implicit solvation is forced for a
  zwitterion. A method that works from positions alone will happily move a
  proton and collapse one.

## DFT

```bash
ligand3d build "<smiles>" -b mmff94,gfn2,orca -o thing.cif
LIGAND3D_ORCA_METHOD=PBE0 LIGAND3D_ORCA_BASIS=def2-TZVP ligand3d build ... -b orca
```

ORCA, defaulting to **B97-3c** — a composite method built for geometries, which
is what you want when the alternative is minutes per gradient. `LIGAND3D_ORCA_METHOD`
and `LIGAND3D_ORCA_BASIS` override it; leave the basis empty for the other
composites (`HF-3c`, `PBEh-3c`, `r2SCAN-3c`, the last needing ORCA 5).

ligand3d drives the optimiser itself and asks ORCA only for gradients
(`ENGRAD`), so DFT produces the same trace and obeys the same convergence
criterion as every other backend rather than being a special case.

The sensible use is the last link in a chain: search with MMFF94, narrow with
GFN2, spend DFT only on what survives.

On water, against experiment (0.958 Å, 104.5°):

| | O–H | H–O–H |
|---|---|---|
| `mmff94` | 0.969 Å | 104.0° |
| `gfn2` | 0.958 Å | 107.2° |
| `orca` (B97-3c) | 0.963 Å | 103.8° |

**A trap worth knowing about:** `orca` on PATH at the IPD is the *GNOME screen
reader*, not the quantum chemistry program. Resolution checks that the binary it
found is actually ORCA — a 13 KB Python script with no `orca_scf` beside it is
not an SCF driver — and finds the real one at `/net/software/orca/latest/orca`.

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

### Where the weights come from

**At the IPD, nothing needs configuring.** The checkpoints are already on
`/net/databases/huggingface/mlFF_models/`, which is one of the built-in probe locations,
so `mace-polar` resolves to `models--ACEsuit--mace-polar-1/MACE-POLAR-1-M.model`
without being told. `ligand3d models -v` prints the resolved path for every model, and
`ligand3d doctor` shows every location it looked in, so a miss is diagnosable rather than
mysterious. That store has its own `README.md` recording who added what and from where —
read it before adding a checkpoint.

Resolution runs in this order, first hit wins:

1. `LIGAND3D_<MODEL>` — e.g. `LIGAND3D_MACE_OFF=/path/to/MACE-OFF23_medium.model`
2. `[weights]` in `~/.config/ligand3d/config.toml`
3. the probe directories: `/net/databases/huggingface/mlFF_models`,
   `/mnt/projects/ml/mlff/models`, then `~/.cache/ligand3d/weights`

**Anywhere else, put the files in `~/.cache/ligand3d/weights/`** — the last probe
directory — and they are found with no configuration, exactly as at the IPD. The
HuggingFace cache layout (`models--<org>--<name>/`) is understood as well as a flat
directory, so an existing populated `HF_HOME` works if you point a probe directory at it.

Where to actually get them differs by family, and the directory names are a *cache naming
convention*, not evidence of a HuggingFace repo:

| family | source | notes |
|---|---|---|
| `mace-off`, `mace-off-24` | [github.com/ACEsuit/mace-off](https://github.com/ACEsuit/mace-off) | **OFF24 is medium-only by design** — no small or large was ever published |
| `mace-mp` | [github.com/ACEsuit/mace-mp](https://github.com/ACEsuit/mace-mp) | |
| `mace-mh`, `mace-mh-1` | [github.com/ACEsuit/mace-foundations](https://github.com/ACEsuit/mace-foundations) releases | |
| `mace-omol` | the OMol25 release | |
| `uma-s`, `uma-s-1p2`, `uma-m` | [huggingface.co/facebook/UMA](https://huggingface.co/facebook/UMA) — **gated** | accept the FAIR Chemistry License v1, make a read token, then `HF_TOKEN=hf_… bash download_uma_models.sh` from the IPD store. FAIR publishes MD5s on the model card and the store verifies against them |
| `esen*`, `allscaip*` | [huggingface.co/facebook/OMol25](https://huggingface.co/facebook/OMol25) — **gated** | all five live in that one repo under `checkpoints/`, despite the store's per-model directory names |
| `aimnet2` | **downloads itself** | caches in `~/.cache/aimnet`; this is why its first call costs 14 s and why it needs no entry here |

Two warnings that are not about convenience:

- **MACE-POLAR-1 is public, but academic-use only.** Released 2026-02-23 through
  [GitHub](https://github.com/ACEsuit/mace-foundations/releases/tag/mace_polar_1) — not
  Hugging Face, so `huggingface-cli` will not find it. The authors are not the Baker Lab.
  The weights are under [ASL](https://github.com/gabor1/ASL), which is *not* an
  open-source licence: GPLv2-derived with a non-commercial clause. Fine for research;
  commercial use needs the licensor.
- **`uma-s-1` is archived upstream over an extensivity bug.** ligand3d does not carry that
  alias, and neither should you — `uma-s` is 1.1.

### Which model, and what it costs


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

This is a real conflict, not a missing install, so no amount of pip fixes it in place.
There are three ways round it, and on this cluster the first is usually the right one.

**1. Run in a container — `--container`.** The images used for GPU submission each carry
one side of the split, so a model that cannot run in your virtualenv is available in one
already on disk. No scheduler, no second install, no waiting in a queue:

```bash
ligand3d build "O=C1CN2CCC1CC2" -b esen -o quin.cif --container
```

ligand3d picks the image from the backend, runs the build inside it, and writes the files
where you asked. It refuses a chain that mixes the families — `mace-off,esen` — because no
single image can satisfy both.

The sketcher has the same thing: a **Run** control with *on this machine*, *in a container*,
and *on a GPU node*. Choosing a container un-greys the fairchem models in the method menu,
which otherwise sit disabled — the checkbox would be useless without that, since the thing
it enables would still not be selectable. Picking one while still set to run locally says
so, rather than letting you press Build and read about e3nn afterwards.

**2. Run on a GPU node — `--slurm`.** The same images, on a compute node. Worth it for a
large molecule or a real conformer search, and not otherwise; see
[Running on a GPU](#running-on-a-gpu-slurm-at-the-ipd). Quinuclidinone through eSEN was
**5.6 s in a local container and 8.3 s on an A4000** — the GPU loses on a 19-atom molecule,
which is the same size effect described there.

**3. Keep two virtualenvs.** Still supported, and the right answer off this cluster:

```bash
uv pip install -e ".[xtb,protonation,mace]"      # MACE + AIMNet2 + xTB
uv pip install -e ".[xtb,protonation,fairchem]"  # eSEN / UMA / AllScAIP + xTB
```

Install torch from the CPU index first in both unless you have a GPU.

All three produce the same answer. Quinuclidinone through eSEN gives
`-253084.1597 kcal/mol` in a local container, on a GPU node, and from a direct
`apptainer exec` — identical to four decimals, which is what you want from a build that
travels between environments.

`ligand3d models` names the way out in the reason itself rather than only saying
"unavailable", because a model that is one flag away should not read as out of reach.

### MACE-POLAR — one extra package, no separate environment

MACE-POLAR-1 (S/M/L) models the long-range electrostatics the other MACE models leave out.
It used to need a patched MACE fork that installed *as* `mace-torch`, which made it a third
mutually exclusive environment. **That is no longer true:** `mace-torch` 0.3.16 ships
`PolarMACE` upstream, so POLAR now sits beside every other MACE model in the same
virtualenv.

One piece is still missing from PyPI — `graph_longrange`, which supplies the
reciprocal-space sum. `mace-torch` only *soft*-imports it, so pip alone leaves you with a
checkpoint that cannot be unpickled: the dependency is recorded inside the model file, and
`torch.load` resolves it at `find_class` time. It ships in `graph_electrostatics`, MIT,
GitHub-only:

```bash
uv pip install --no-deps git+https://github.com/WillBaldwin0/graph_electrostatics.git@v0.4.0
ligand3d build "<smiles>" --backend mmff94,mace-polar
```

`--no-deps` is deliberate: its requirements (`torch`, `e3nn==0.4.4`, `numpy`, `ase`) are
already satisfied by the `[mace]` extra, and letting it re-resolve them risks moving the
e3nn pin that the whole MACE side depends on.

The `ligand3d-mace.sif` container has it baked in, so POLAR works there with nothing to
install. It is deliberately absent from the fairchem image: `graph_electrostatics` pins
`e3nn==0.4.4`, the same pin that splits the two families in the first place.

`ligand3d doctor` reports `graph_longrange` on its own line, so a missing POLAR is one
lookup rather than a puzzle.

### Still not usable

- **SO3LR v2 beta** is a JAX model needing jax, orbax, and `so3lr`; every ligand3d
  backend is torch or ASE based.
- **orb-mol-conservative** (99 MB) loads, but `orb-models` 0.7 removed
  `orb_models.forcefield.calculator`, so there is no ASE calculator to attach.

`ligand3d doctor` and `ligand3d models` list these with the same explanation rather than
pretending they aren't there.

## How a minimization actually works

Two separate things are involved, and it helps to keep them apart.

**The potential** answers one question: given these atomic positions, what is the energy,
and what is the force on each atom? That is all MMFF94, GFN2-xTB, MACE, or AIMNet2 ever
do. They differ enormously in how they answer it, and not at all in what they are asked.

**The optimizer** takes those forces and decides where to move the atoms next. ligand3d
uses **L-BFGS** for every backend except the RDKit force fields.

The loop is just:

```
positions ──▶ potential ──▶ energy + forces ──▶ L-BFGS proposes new positions ──▶ repeat
                                                        │
                                              stop when max force < fmax
```

### L-BFGS, briefly

L-BFGS is a quasi-Newton method. Steepest descent would step straight downhill along the
force, which zig-zags badly in a narrow valley — and a molecule's energy surface is full of
narrow valleys, because stretching a bond costs far more than rotating a torsion. Newton's
method would fix that using the Hessian (all second derivatives), but for a 29-atom
molecule that is an 87×87 matrix that nobody wants to compute or invert every step.

L-BFGS splits the difference: it *approximates* the inverse Hessian from the last handful
of position and gradient changes — the "limited memory" part — and never forms the matrix.
The result converges in tens of steps where steepest descent would take thousands, at
almost no extra cost per step. Every timing in this README is dominated by the potential,
not by L-BFGS.

ligand3d uses `ase.optimize.LBFGS` and stops at `fmax = 0.05 eV/Å` by default. The RDKit
force fields are the exception: they use RDKit's own C++ minimizer, because round-tripping
coordinates through ASE for a four-millisecond job costs more than the job.

### Where ACE comes in — and where it does not

ACE (Atomic Cluster Expansion) is **not** an optimizer and does not drive anything. It is
a way of *describing an atom's environment*: a systematic expansion in body order — how
this atom sits relative to one neighbour, then pairs of neighbours, then triples — built
from a basis of radial functions and spherical harmonics. Its appeal is that it is
systematic: turn up the body order and the expansion gets more expressive in a controlled
way, rather than by adding another ad-hoc term.

**MACE** uses that idea inside a message-passing neural network. Each atom carries a
feature vector; neighbours exchange messages along edges; and the many-body ACE features
are formed by symmetric contraction of tensor products of those messages. The pieces are
visible in the code — `EquivariantProductBasisBlock`, `SymmetricContraction` — and
"equivariant" is the load-bearing word: rotate the molecule and the predicted forces
rotate with it, exactly, by construction rather than by training. A model that has to
*learn* rotational symmetry from data wastes capacity on it and never gets it exactly
right.

So: ACE is the functional form of the energy. L-BFGS moves the atoms. They are unrelated
parts of the machine.

### What the other tiers are doing

- **MMFF94 / UFF** — a hand-fitted sum of springs: bond stretches, angle bends, torsions,
  van der Waals, electrostatics from fixed partial charges. The bond list is fixed at
  setup, which is why these cannot move a proton and why a zwitterion is safe with them.
  No electrons anywhere in the model.
- **GFN-FF** — the same idea, but with parameters generated across the periodic table and
  a topology perceived on the fly.
- **GFN1/GFN2-xTB** — genuinely semi-empirical *quantum* methods. They solve a simplified
  self-consistent tight-binding problem with a minimal basis, so electrons are represented
  and charge can redistribute. That is why they can break and form bonds, why they need a
  total charge, and why they are the only tier here with an implicit solvent model.
- **AIMNet2, MACE, eSEN, UMA** — neural potentials trained to reproduce DFT energies and
  forces. No electrons at run time; they have learned the *result* of the electronic
  structure calculation. Fast, and only as trustworthy as the chemistry in their training
  set.

### MACE-POLAR is a different shape

Standard MACE is strictly local: each atom sees neighbours inside a cutoff, usually 5–6 Å.
That is fine for bonded geometry and terrible for anything where a charge on one end of
the molecule matters to the other end, because Coulomb interactions fall off as 1/r and
simply do not fit inside a cutoff.

MACE-POLAR (`PolarMACE`, a subclass of the standard `ScaleShiftMACE`) adds an explicit
long-range term. It predicts atomic multipoles — charges, and optionally dipoles and
quadrupoles — then evaluates the electrostatics in reciprocal space with a k-space cutoff,
Ewald style. It can also run the field response to self-consistency, iterating until the
induced multipoles stop changing, which is what makes it *polarizable* rather than just
long-range. That machinery lives in the separate `graph_longrange` package
([graph_electrostatics](https://github.com/WillBaldwin0/graph_electrostatics)).

**This is not free**, and it is the honest answer to "does it really run that fast?": it
does not. Measured on this machine, MACE-POLAR is roughly ten times slower per L-BFGS step
than plain MACE-OFF, because every step now includes a reciprocal-space sum and possibly a
self-consistency loop:

| model | per L-BFGS step | full minimization |
|---|---|---|
| `mace-off-small` | 0.075 s | 2.0 s |
| `mace-off` | 0.25 s | 6.8 s |
| `mace-polar-s` | 0.74 s | 23 s |
| `mace-polar` | 1.8 s | 57 s |
| `mace-polar-l` | 3.6 s | 114 s |

An earlier version of this table carried estimates rather than measurements and had
MACE-POLAR at 4/8/16 s — wrong by a factor of six to seven, in the flattering direction.
Everything above was timed.

## Timings

All numbers are a **full minimization of gabapentin (29 atoms) on CPU**, no GPU involved:
a 13th-gen Intel i5-13500 with `OMP_NUM_THREADS=8`, torch's `+cpu` build. Roughly 26–32
L-BFGS steps in every case, so the per-step cost is what actually differs.

| backend | minimization | first-call load | note |
|---|---|---|---|
| `mmff94`, `uff` | 4 ms | – | no model to load |
| `gfnff` | 0.06 s | 1.5 s | xtb optimizes internally |
| `gfn2` | 0.33 s | – | tblite; the best value here |
| `gfn1` | 0.57 s | – | |
| `aimnet2` | 0.62 s | **14 s** | load dominates a single run |
| `mace-off-small` | 2.0 s | 3.6 s | |
| `mace-off` | 6.8 s | 0.5 s | |
| `mace-off-24` | 7.0 s | 0.5 s | |
| `mace-mp` | 7.8 s | 1.1 s | |
| `mace-mh-spice` | 7.9 s | 0.5 s | |
| `mace-mh` | 8.4 s | 0.4 s | |
| `mace-mh-1` | 12 s | 1.5 s | |
| `mace-polar-s` | 23 s | 0.2 s | long-range electrostatics |
| `mace-omol` | 28 s | 5.2 s | charge-aware MACE |
| `mace-off-large` | 31 s | 9.3 s | |
| `mace-polar` | 57 s | 0.2 s | |
| `mace-polar-l` | 114 s | 0.3 s | |

Every entry is now measured. The fairchem models were timed inside the container that
can run them, on the same molecule and the same machine, so they sit on the same scale as
the rest:

| backend | CPU | GPU (RTX PRO 6000) |
|---|---|---|
| `esen-sm-direct` | 2.8 s | 0.54 s |
| `esen` | 7.9 s | 1.1 s |
| `allscaip-direct` | 11 s | 1.1 s |
| `uma-s` | 15 s | 2.6 s |
| `esen-md-direct` | 16 s | 1.5 s |
| `allscaip` | 23 s | 2.5 s |
| `uma-s-1p2` | 36 s | 3.4 s |
| `uma-m` | 63 s | 14 s |

The old estimates were badly wrong in both directions — `uma-m` was guessed at "minutes"
and takes 63 s on CPU, while `uma-s` was guessed at 12 s and takes 15 s. All nine agree
on the energy to within 0.1 kcal/mol, which is the cross-check that says the numbers came
from the same calculation.

`uma-m` carries one caveat the table cannot: **11 GB of weights needs real memory.** It was
OOM-killed on a 31 GB workstation and ran without trouble on a node with 96 GB.

Two things worth noticing. `aimnet2` spends 14 seconds constructing itself and then
minimizes in 0.6 — for one molecule that is a bad trade, and for a hundred it is
irrelevant, since the calculator is built once and reused. And a conformer search
multiplies the minimization column by the number of conformers, which is exactly why
`--backend mmff94,mace-off` searches with MMFF94 and only refines the survivors: 60
conformers through MACE-OFF would be seven minutes to rediscover what MMFF94 found in a
quarter of a second.

## How accurate is any of this

`ligand3d models -v` prints, for every method, what it reproduces and how close it
reportedly gets. The methods page in the browser shows the same thing. Both are worth
reading before trusting a number, and both come with the same three caveats.

**An error bar means nothing without its reference.** A fitted method cannot be more right
than the thing it was fitted to. MACE-OFF reproduces ωB97M-D3(BJ)/def2-TZVPPD to about
1 kcal/mol — but that is agreement with *that functional*, which is itself off by a few
kcal/mol for some of the chemistry you will point this at. "1 kcal/mol against DFT" and
"1 kcal/mol against CCSD(T)" are different claims.

| method | reproduces | reported error |
|---|---|---|
| `uff` | rules, no single reference | geometries ~0.05 Å; conformer energies often several kcal/mol out |
| `mmff94` | MP2/6-31G* + experiment | bond lengths ~0.01–0.02 Å; conformer energies ~1 kcal/mol where parameterized |
| `gfnff` | GFN2-xTB geometries | heavy-atom RMSD ~0.1–0.3 Å vs DFT |
| `gfn2` | DFT (the GFN2 fit set) | bond lengths ~0.01–0.02 Å; conformer and reaction energies ~2–4 kcal/mol |
| `aimnet2` | ωB97M-D3/def2-TZVPP | ~1 kcal/mol energies, ~1–2 kcal/mol/Å forces |
| `mace-off*` | ωB97M-D3(BJ)/def2-TZVPPD (SPICE) | ~1 kcal/mol energies, ~1–2 kcal/mol/Å forces, neutral organics |
| `mace-omol`, `esen`, `uma*` | ωB97M-V/def2-TZVPD (OMol25) | sub-kcal/mol in domain; covers charged and open-shell |
| `mace-mp` | PBE / r2SCAN periodic DFT | tens of meV/Å on solids; **not fitted for molecular conformers** |
| `mace-polar*` | a polarizable set with explicit long-range terms | no settled benchmark to quote; charge-aware |

**Every one of those figures is in-domain.** They are errors on held-out data drawn from the
same distribution as the training set. That is what papers report and it is the number most
likely to mislead you, because a neural potential handed something unlike its training data
does not refuse and does not widen its error bar — it returns a confident number that can be
wrong by an order of magnitude. This is why ligand3d checks elements and total charge
*before* running a model rather than interpreting the result afterwards.

**Force error and energy error are different things**, and a method can be good at one and
poor at the other. Forces decide whether a minimization lands in the right geometry; energies
decide whether you can trust one conformer ranked above another. For picking a conformer, the
energy column binds.

The practical reading: for a **starting geometry**, everything in that table is fine and
`mmff94` is 4 ms. For **ranking conformers within a few kcal/mol**, you need `gfn2` or a
neural potential whose training set actually contains your chemistry. For anything charged,
you need a method with a charge channel at all — which is a capability question, not an
accuracy one, and ligand3d refuses those pairings rather than answering them.

## Choosing a backend

- **Default, and fine for most work** — `mmff94`. Four milliseconds, and geometry that is
  perfectly reasonable for a starting structure.
- **When the geometry matters** — `mmff94,gfn2`. Under a second, near-QM bond lengths and
  angles, and the only tier with both a charge channel and implicit solvent.
- **Charged or zwitterionic** — `gfn2` with solvation, which is applied automatically.
  Avoid the neutral-trained MLFFs entirely; ligand3d refuses those pairings anyway.
- **When you want a neural potential** — `mmff94,aimnet2` is the fastest charge-aware
  option; `mmff94,mace-off` if the molecule is neutral and organic.
- **Long-range electrostatics matter** — `mace-polar`, and budget a minute per molecule.
- **Inorganic or metal-containing** — `mace-mp`, which is the only one trained for it.

## Running on a GPU (SLURM, at the IPD)

Everything above runs on whatever machine you are sitting at. On the IPD cluster you can
hand a build to a GPU node instead:

```bash
ligand3d build "NCC1(CC(=O)O)CCCCC1" -b mace-off -n 3 --slurm
ligand3d slurm                      # can this host submit? what containers are there?
ligand3d slurm --job 18977069       # what happened to that job?
```

or tick **run this on a GPU node (SLURM)** in the sketcher. The option only appears when
the host can actually submit, so nowhere else is offered a checkbox that could only fail.

This is optional in the strongest sense: `slurm.py` is imported from exactly one place in
the CLI, nothing else in ligand3d touches it, and importing the package does not load it.
A machine with no scheduler behaves exactly as it did before the feature existed.

### Why it needs a container

**Yes, apptainer is required, and not as a convention.** The `torch` in this project's
virtualenv is the `+cpu` build. Submitting this interpreter to a GPU node would allocate
a GPU and then quietly ignore it — the slowest possible outcome, and one that looks like
success. The containers on `/net/software/containers` carry a CUDA torch, so the job runs
inside one:

| image | torch | potentials |
|---|---|---|
| `quantum_chem-20260604.sif` | 2.11.0+cu130 | MACE (all), MACE-POLAR, AIMNet2 |
| `uma-20260527.sif` | 2.8.0+cu128 | eSEN, UMA, AllScAIP |

The split is not arbitrary: it is the same e3nn incompatibility described above, which
exists inside the containers exactly as it does in a virtualenv. ligand3d picks the image
from the backend you asked for, and refuses a chain that mixes the two families *before*
submitting rather than letting it fail on a node ten minutes later.

Nothing is installed into the container. The package is copied into the job directory and
put on `PYTHONPATH`, which has a second benefit: a job that sits in the queue for an hour
runs the code that was submitted, not whatever the repo happens to contain when it
starts. The job directory ends up holding the sbatch script, the JSON payload, the exact
source that ran, both logs, and the result — enough to reproduce or explain the run later.

### Whether it is worth it

Measured with `mace-off` in the same container, one process, RTX 4000 Ada versus the four
CPU cores the job requested:

| work | CPU | GPU | speedup |
|---|---|---|---|
| gabapentin (29 atoms), 1 conformer | 4.94 s | 1.73 s | 2.9× |
| gabapentin, 10-conformer search | 163 s | 48.9 s | 3.3× |
| a 58-atom tripeptide, 1 conformer | 35.0 s | 7.65 s | 4.6× |
| the tripeptide, 10-conformer search | 749 s | 156 s | 4.8× |

**Single digits, not orders of magnitude.** That is the number worth internalizing before
building a workflow around this. A 29-atom graph does not come close to filling a GPU, and
L-BFGS is inherently sequential — every step needs the forces from the one before, so
there is nothing to overlap and kernel launch latency dominates. Conformers are minimized
one after another rather than batched, so a search multiplies the wall time rather than
hiding it.

The size dependence is the useful part: the speedup climbs from 2.9× to 4.8× between a
29-atom ligand and a 58-atom peptide, because a bigger graph gives the GPU more to do per
launch. That last row is where it starts to matter in practice — twelve and a half minutes
becomes two and a half. Extrapolate accordingly: small and rigid means run it locally, big
and floppy means the queue is worth the wait.

So the honest guidance is narrow:

- **Worth submitting** — a large or flexible molecule, a real conformer search through an
  expensive potential (`mace-off-large`, `mace-polar`, `mace-omol`), or a batch you would
  otherwise babysit.
- **Not worth submitting** — anything classical or semi-empirical, and any single small
  molecule. `mmff94` finishes in four milliseconds; you cannot beat that by waiting in a
  queue. ligand3d says so rather than silently obliging.

Every run now reports which processor the potential actually used:

```
· minimization time: mace-off 265.16s
· neural potential ran on GPU (NVIDIA RTX 4000 Ada Generation)
```

A neural potential falling back to CPU looks identical to one on a GPU, only slower, so
this line is worth reading before drawing conclusions about a timing.

### Resources and defaults

Defaults are one `gpu:small` GPU, 4 CPUs, 16 GB, one hour, on partition `gpu`, account
`IPD`. Override any of them:

```bash
ligand3d build LIG.sdf -b mace-polar --slurm \
  --slurm-partition gpu-bf --slurm-gpu large --slurm-time 04:00:00 \
  --slurm-cpus 8 --slurm-mem 32G --slurm-wait
```

GPU classes are `small`, `large`, and `h200` — the old `a4000`/`b4000` GRES names no
longer schedule, and asking for one is rejected here rather than pending forever. The
scheduler also refuses jobs under five minutes, which is checked before submission.

One trap is worth stating plainly, because it fails silently: **a compute node's `/tmp`
is its own.** A job writing there exits 0 and leaves nothing behind. Output and job
directories must be on `/home`, `/net`, or `/mnt` — exactly the paths the container
bind-mounts — and ligand3d refuses anything else, including otherwise-shared filesystems
the job would not be able to see.

Two smaller refusals, both for the same reason — SLURM reads `#SBATCH` paths literally
rather than through a shell, so it cannot be quoted around:

- A job or output path containing whitespace or shell characters is rejected rather than
  producing a script that breaks in a confusing way.
- `--slurm-dir` pointed at a directory whose `src/` is not a previous ligand3d snapshot is
  rejected, instead of deleting it to make room for one.

The default images live under `/net/software/containers/users/woodbuse/`, which is
world-readable on this cluster — anyone in the lab can submit against them, they just cannot
edit them, which is the right way round. Point at different images with `LIGAND3D_SIF_MACE`
and `LIGAND3D_SIF_FAIRCHEM`, and at a different account with `LIGAND3D_SLURM_ACCOUNT`.
`ligand3d slurm` prints which images it resolved, so a wrong path shows up before a job
does.

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

## Looking a molecule up by name

Drawing a fused polycyclic by hand is slow and easy to get subtly wrong, and most molecules
worth building already have a name. `ligand3d fetch` turns a name, SMILES, InChI, or PubChem
CID into a structure; in the sketcher there is an import box above the canvas that drops the
result straight onto it, so it becomes a scaffold you edit rather than a drawing you start
from nothing.

```bash
ligand3d fetch "3-Cyano-7-ethoxycoumarin"       # systematic name, resolved offline
ligand3d fetch aspirin                          # trivial name, via PubChem
ligand3d fetch cid:2244                         # straight to a PubChem record
ligand3d fetch "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
ligand3d fetch PLYNGSG --type peptide           # a sequence
ligand3d fetch GGCAT   --type dna
ligand3d fetch coumarin -o scaffold.sdf         # a 2D file to open in anything
ligand3d fetch --templates                      # the built-in scaffolds
ligand3d build "$(ligand3d fetch aspirin --smiles)" -o aspirin.cif
```

### Three routes, and why it matters which one answered

| route | what it handles | needs |
|---|---|---|
| parsing | SMILES, InChI | nothing |
| **OPSIN** | systematic IUPAC names | `py2opsin` and a `java` |
| **PubChem** | trivial names, trade names, CIDs | the network |
| **sequences** | peptides, DNA, RNA — see below | nothing |

These are complements, not fallbacks for each other, and `fetch` always says which one
answered:

```
$ ligand3d fetch "3-Cyano-7-ethoxycoumarin"
C12H9NO3  CCOc1ccc2cc(C#N)c(=O)oc2c1
  · derived from the name by OPSIN, offline, from IUPAC rules
```

OPSIN is a **grammar for the naming rules**, not a database. It parses names nobody has ever
catalogued, works with no network, and returns exactly what the name says — so it cannot
hand you the wrong compound, but it knows nothing about `aspirin`, because that name is not
derivable from anything.

PubChem is the opposite: a **lookup**, so it covers every trivial and trade name, and it can
quietly give you something you did not mean. Searching the name `CO` returns cobalt, not
carbon monoxide. That is why the matched title is always printed — read it.

For the record, [InChI](https://github.com/IUPAC-InChI/InChI) does not help here. It is a
structure *serialization*: it turns a structure into a canonical string and back. An InChI
string is accepted as input, but InChI cannot turn `aspirin` into a structure. OPSIN and
PubChem are the two things that do.

### Sequences: peptides, DNA and RNA

Typing `PLYNGSG` beats drawing fifty atoms, and an oligonucleotide is not
realistically drawable at all. Pick the import type — **Peptide**, **DNA** or **RNA** —
and type the sequence:

```bash
ligand3d fetch PLYNGSG   --type peptide      # N to C
ligand3d fetch GGCAT     --type dna          # 5' to 3'
ligand3d fetch AUGGC     --type rna
ligand3d fetch "GS(KCX)PL" --type peptide    # carboxylated lysine, mid-chain
ligand3d fetch --residues --type dna         # what codes exist
```

**Sequences are never auto-detected**, and that is deliberate: `GGCAT` is a perfectly
good DNA oligo and an equally good pentapeptide. Guessing would sometimes silently build
the wrong polymer, so the type has to be chosen. Auto-detect still handles everything
else, and says so when a failed lookup looks like a sequence.

RDKit builds the canonical alphabets, and is used directly for them. What it will not do
is anything modified — a phosphoserine, a carboxylated lysine, an inosine — which is most
of the reason to want this. Those come from a residue library here, and **that library is
checked against RDKit rather than trusted**: every canonical residue and chain built by
ligand3d is asserted to be the same molecule RDKit produces. A hand-written SMILES with an
inverted stereocentre gives a plausible-looking peptide of the wrong enantiomer, and a
misdrawn ribose gives a 2'-5' linked RNA; neither is something you would catch by eye, so
the test suite catches them instead. Every nucleoside is additionally checked against its
known molecular formula.

Longer codes go in parentheses, following the **PDB Chemical Component Dictionary**, so
`(SEP)` here is `SEP` in a PDB file. The sketcher lists the alphabet for whichever type is
selected, with the full residue name on hover.

| | codes |
|---|---|
| **peptide** | the 20, plus `U` (selenocysteine) and `O` (pyrrolysine) from the expanded alphabet |
| **PTMs** | `SEP` `TPO` `PTR` phospho-Ser/Thr/Tyr · `KCX` carboxy-Lys · `ALY` acetyl-Lys · `MLZ` `MLY` `M3L` mono/di/tri-methyl-Lys · `TYS` sulfo-Tyr · `HYP` hydroxyproline · `PCA` pyroglutamate · `CSO` `CSD` `OCS` oxidised Cys · `NIY` nitro-Tyr · `CIR` citrulline |
| **ncAAs** | `MSE` `SEC` `ORN` `NLE` `NVA` `ABA` `DAB` `AIB` `SAR` `HCS` `HSE` `PFF` `AZF` `BIF`, plus `DAL` `DVA` `DPR` for D residues |
| **DNA** | `A C G T`, plus `U` `I` (deoxyinosine) `5MC` `8OG` `BRU` |
| **RNA** | `A C G U`, plus `T` `I` (inosine) `PSU` (pseudouridine) `5MC` `6MA` `7MG` |

Anything non-standard is reported rather than absorbed silently:

```
$ ligand3d fetch "GS(KCX)PL" --type peptide
C23H40N6O9  CC(C)C[C@H](NC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCCNC(=O)O)NC(=O)...
  · non-standard residue(s): KCX = N6-carboxylysine
  · built N terminus first, with a free amine and a free acid
```

IUPAC ambiguity codes are refused with the reason: `N` in a DNA sequence means *any base*,
which is a set of sequences rather than a molecule.

**Protonation.** Sequences are built neutral — free amine, free acid, protonated
phosphates — and the pH from the Chemistry tab is applied when you build, not at import.
Doing it in both places would protonate twice. The import notice says which will happen:
with protonation set to *as drawn* it tells you the structure is neutral and how to change
that, and with a pH set it says the termini and ionizable side chains will be adjusted to
that pH. The default is 7.4.

Chains are built with free hydroxyls at both ends and no terminal phosphate, matching what
RDKit's uncapped flavors produce.

### Scaffolds

`ligand3d fetch --templates` lists eighteen starting points — benzene, piperidine,
morpholine, indole, coumarin, purine, adamantane, the gonane skeleton, and so on — also
available from the dropdown next to the import box. They exist for when the point is to draw
a derivative, so the list is deliberately short: a starting point, not a compound library.

Anything imported is **2D on purpose**. It is a drawing to edit, and writing it out with
3D-looking coordinates would invite someone to mistake a layout for a geometry. The
embedding happens later, in `build`, from whatever you edited it into.

## Drawing

`ligand3d sketch` starts a local server, opens a browser, and stays up. Draw a structure,
set the options, press **Build**, read the run log, clear the canvas, and draw the next
one — no reloading between molecules. There is an import box above the canvas (see
[Looking a molecule up by name](#looking-a-molecule-up-by-name)) for starting from a
named compound or a scaffold instead of an empty sheet.

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

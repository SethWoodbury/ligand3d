# Machine-learned potentials

The neural tier: which models, where their weights live, and the environment split that shapes everything else.

[← back to the README](../README.md)

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
| `uma-s`, `uma-s-1p2`, `uma-s-1p2p1`, `uma-m` | [huggingface.co/facebook/UMA](https://huggingface.co/facebook/UMA) — **gated** | accept the FAIR Chemistry License v1, make a read token, then `HF_TOKEN=hf_… bash fetch_gated.sh` from the IPD store. FAIR publishes MD5s on the model card and the store verifies against them |
| `esen*`, `allscaip*` | [huggingface.co/facebook/OMol25](https://huggingface.co/facebook/OMol25) — **gated** | all five live in that one repo under `checkpoints/`, despite the store's per-model directory names; same `fetch_gated.sh` reprovisions them |
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
| `uma-s-1p2p1` | UMA s 1.2.1 | **yes** | 2.3 GB | newest small UMA; FAIR recommends it over 1.2 |
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
[Running it on the cluster](cluster.md#running-it-on-the-cluster-slurm-at-the-ipd). Quinuclidinone through eSEN was
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

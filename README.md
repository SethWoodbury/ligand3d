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
uv pip install -e .              # core: RDKit only, does 2D→3D and MMFF94/UFF
uv pip install -e ".[xtb]"       # adds GFN1/GFN2-xTB via tblite wheels
uv pip install -e ".[protonation]"  # adds pH-based protonation via dimorphite-dl
uv pip install -e ".[mlff]"      # adds MACE / AIMNet2 (large; see note below)
```

The core install is small and has no compiler or conda requirement. Install torch from
the CPU index before `[mlff]` unless you have a GPU, or pip will pull the multi-gigabyte
CUDA build:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[mlff]"
```

## Backends

| id | kind | takes charge | solvation | speed (gabapentin) |
|---|---|---|---|---|
| `mmff94` | classical FF | no | no | 5 ms |
| `uff` | classical FF | no | no | 5 ms |
| `gfnff` | classical FF | yes | ALPB | ~0.1 s |
| `gfn2` | semi-empirical | yes | ALPB | 0.9 s |
| `gfn1` | semi-empirical | yes | ALPB | ~0.8 s |
| `aimnet2` | ML potential | yes | no | 0.5 s |
| `mace-off` | ML potential | no | no | 6.8 s |
| `mace-mp` | ML potential | no | no | varies |

Chain them with a comma — cheap first, expensive last:

```bash
ligand3d build "NCC1(CC(=O)O)CCCCC1" --backend mmff94,gfn2
```

`ligand3d backends` lists what is registered; `ligand3d doctor` says which ones can
actually run here and what to install for the rest.

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

`ligand3d sketch` starts a local web server, opens a browser, and waits. Draw a
structure — wedge and hash bonds become real stereocenters — press **Use this
molecule**, and the pipeline runs on what you drew.

It ships with [JSME](https://jsme-editor.github.io/), fetched once (about 1 MB) into
`~/.cache/ligand3d/` and thereafter working offline. Nothing is sent anywhere: the
server is bound to `127.0.0.1` and shuts down as soon as you submit.

```bash
ligand3d sketch                              # draw, build, write sketch.pdb
ligand3d sketch --backend gfn2 -o mol.pdb    # any build option applies
ligand3d sketch --no-browser                 # print the URL instead (SSH port-forwarding)
ligand3d sketch --smiles-only                # just tell me the SMILES
```

[Ketcher](https://github.com/epam/ketcher) is a nicer editor and is supported, but EPAM
publishes it as an npm library rather than a servable page — `ketcher-standalone.zip`
contains no HTML at all — so it cannot be fetched and used automatically. Build it
yourself and point `LIGAND3D_KETCHER_DIR` at the directory holding the resulting
`index.html`, and it will be used in preference to JSME. The pinned upstream source is
the `vendor/ketcher` submodule, which a normal clone does not download.

If neither can be reached, the page degrades to a paste box that accepts a SMILES
string or a molblock, so the command still works on an air-gapped machine.

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

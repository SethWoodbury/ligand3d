# Every command

`ligand3d <command> --help` is generated from the code and is always right; this page is
the map of what exists and when to reach for it.

## Building a structure

| | |
|---|---|
| `ligand3d build "<smiles>" -o out.cif` | the whole job: 2D in, minimized 3D out |
| `ligand3d embed "<smiles>" -o raw.sdf` | the first half only — ETKDG coordinates, no minimization |
| `ligand3d minimize raw.sdf -b gfn2 -o min.cif` | the second half only — minimizes every conformer in a file that already has 3D coordinates |
| `ligand3d conformers "<smiles>" -n 50 -o ensemble` | a conformer ensemble rather than one structure |

`build` is `embed` followed by `minimize`. Splitting them is worth it when the geometry
and the minimization belong to different runs — generate once, then minimize the same
starting geometry with several methods and compare like with like.

## Understanding a molecule before you build it

| | |
|---|---|
| `ligand3d fetch "aspirin"` | name, InChI, CAS or a peptide/DNA/RNA sequence to a structure |
| `ligand3d stereo "<smiles>"` | report R/S and E/Z; build nothing |
| `ligand3d protonate "<smiles>" --ph 7.4 --all` | enumerate protonation states at a pH |

## Converting and packaging

| | |
|---|---|
| `ligand3d convert min.cif min.pdb` | between mmCIF, SDF and PDB |
| `ligand3d params ensemble.sdf --code LIG` | Rosetta params from an ensemble |

## Asking what this installation can do

| | |
|---|---|
| `ligand3d models --available` | methods that will actually run on this machine |
| `ligand3d backends` | per-method capabilities: charge, spin, solvent, speed |
| `ligand3d doctor` | what is missing, and the command that fixes it |
| `ligand3d solvents` | the implicit solvents ALPB is parameterized for |
| `ligand3d config` | inspect or create the configuration file |
| `ligand3d version` | the version alone — worth quoting in a bug report |

Nothing here assumes a method is installed. What is available depends on the machine, the
container and whether weights are on disk, which is why `models --available` and `doctor`
exist rather than a fixed list.

## The browser and the cluster

| | |
|---|---|
| `ligand3d sketch` | the drawing interface; everything it does is available above |
| `ligand3d build ... --slurm` | submit instead of running here — see [cluster.md](cluster.md) |
| `ligand3d slurm --job 12345678` | check on a submitted job |

## Flags worth knowing

These are `build` flags, not global ones — checked against `--help`, not assumed:

- **`-q`** (`build` only) prints only the written paths, one per line. Parse that rather
  than the log.
- **`--dry-run`** (`build` only) does the whole calculation and writes nothing, for
  checking a plan before committing to it.
- **`--stereo any` / `--stereo enumerate`** (`build` and `embed`) — undefined
  stereocentres are refused by default rather than silently resolved. See
  [chemistry.md](chemistry.md).

`minimize` takes the ones that describe the calculation instead: `-b/--backend`,
`--solvent`, `--max-steps`, `--threads`, `--trajectory`, `--no-trace`.

### Which browser `sketch` opens

On a Chromium-family default browser, `sketch` opens a window against a profile of its
own under `~/.cache/ligand3d/browser` rather than joining your running browser.

That is deliberate. Handing a URL to `chrome` usually opens a tab, but it relies on the
singleton files in your Chrome profile naming a live process. If Chrome was killed,
crashed or was OOM-killed, those go stale and the next `chrome <url>` **takes ownership
of the profile** — the sketcher's window becomes the browser, so closing it quits Chrome
and every unrelated tab with it. A local tool should not be able to do that, so it does
not share your profile at all. Firefox and other defaults are opened normally.

Per run, from the command line:

```bash
ligand3d sketch --system-browser    # a tab in your normal browser, old behaviour
ligand3d sketch --no-browser        # print the URL, open nothing
```

Or set it once for a shell, which `--system-browser` still overrides:

| `LIGAND3D_BROWSER` | |
|---|---|
| unset / `auto` | the isolated window described above |
| `system` | hand the URL to your desktop's default browser, as before |
| `none` | open nothing; just print the URL |

Nothing is opened over SSH either way — forward the port and open it on your own machine.

Driving all of this from an agent: [AGENTS.md](../AGENTS.md).

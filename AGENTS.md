# Driving ligand3d from an agent

ligand3d turns a 2D molecule into a minimized 3D structure. Everything the browser does
is available from the CLI, so an agent can do the whole job without a person present.

## The short version

```bash
ligand3d build "CC(=O)Oc1ccccc1C(=O)O" -b mmff94,gfn2 -o aspirin.cif
```

- **Exit 0 on success, 1 on failure.** Errors go to stderr and say what to do, not just
  what went wrong.
- **`-q` prints only the written paths**, one per line — parse that rather than the log.
- **`--dry-run`** does the whole calculation and writes nothing, for checking a plan.

## Discovering what is available

Never assume a method is installed. What is usable depends on the machine, the container,
and whether weights are on disk.

```bash
ligand3d models --available    # methods that will actually run here
ligand3d doctor                # what is missing, and the command that fixes it
ligand3d backends              # capabilities: charge, spin, solvent, speed
```

## Choosing a method

Chain cheap to expensive with commas. Each link refines what the last one produced, and
only the survivors reach the expensive end:

```bash
-b mmff94                      # milliseconds; a starting geometry
-b mmff94,gfn2                 # seconds; a good default
-b mmff94,gfn2,mace-off        # neural potential, organic molecules
-b mmff94,gfn2,orca-wb97x3c    # DFT, minutes to hours
```

Tiers, cheapest first: `mmff94`/`uff` → `gfnff` → `gxtb` → `gfn1`/`gfn2` → MACE or
fairchem → `orca-*`. `ligand3d models` prints measured timings for all of them.

**Name the level of theory, not "DFT".** There are fifteen ORCA backends and they are not
interchangeable: `orca-b973c` is the sensible default, `orca-wb97x3c` the best geometry
per second, `orca-wb97mv` the most accurate. `orca` alone means `orca-b973c`.

## Things that will bite an agent

- **Undefined stereocentres are refused**, deliberately. Either draw the wedge, or pass
  `--stereo any` to let RDKit pick, or `--stereo enumerate` to build every isomer. Do not
  silently pick one for a user.
- **Charged molecules need a charge-aware method.** Handing a carboxylate to a model with
  no charge channel is refused rather than answered wrongly. `gfn1`, `gfn2`, `gfnff`,
  `gxtb`, `aimnet2`, `mace-omol`, `mace-polar`, the fairchem models and every `orca-*`
  take charge; `mace-off`, `mace-mp`, `mmff94` and `uff` do not. `ligand3d backends`
  prints the column rather than requiring you to remember this.
- **MACE and fairchem cannot mix in one chain.** They pin incompatible `e3nn` versions.
  `-b mace-off,esen` is refused at submit time rather than failing on a compute node.
- **`--ph` changes the molecule.** It enumerates protonation states; use it when the user
  asked about a pH, not by default.
- **Energies are only comparable within one method.** A strain energy near zero and a
  total electronic energy near −253000 are both "the energy". `orca-wb97x3c` is further
  apart still: its basis carries ECPs, so its totals sit on their own scale.

## Where it runs

```bash
ligand3d build ... --slurm --slurm-partition cpu    # DFT: CPU nodes, ORCA is not GPU
ligand3d build ... --slurm                          # neural potentials: GPU
ligand3d slurm --job 12345678                       # check on it
```

At the IPD the containers and all shared weights are on `/net`, so nothing downloads.
Elsewhere, see the README for what to install per tier.

## Other commands

```bash
ligand3d fetch "3-Cyano-7-ethoxycoumarin"   # name, InChI, CAS, peptide/DNA/RNA sequence
ligand3d stereo "<smiles>"                  # report R/S and E/Z, build nothing
ligand3d protonate "<smiles>" --ph 7.4 --all
ligand3d conformers "<smiles>" -n 50 -o ensemble
ligand3d params ensemble.sdf --code LIG     # Rosetta params
ligand3d convert min.cif min.pdb
```

`ligand3d <command> --help` is accurate and worth reading before guessing at flags.

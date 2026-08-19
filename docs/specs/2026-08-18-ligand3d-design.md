# ligand3d — design

**Date:** 2026-08-18
**Status:** approved, implementing

## Purpose

Turn a 2D molecule — a SMILES string, a MOL/SDF file, or something drawn in a browser
sketcher — into a minimized 3D structure written as a `.pdb` file. Stereocenters must be
respected. Minimization must be fast and must let the user choose the level of theory,
from a classical force field up to a machine-learned potential. Protonation state and
conformer search are optional stages.

The tool is a command-line program. The repo stays small: pure Python, installed with
`uv`/`pip`, no conda environment and no containers.

## Non-goals

- Protein preparation, docking, or scoring. This builds a ligand; what happens next is
  someone else's job.
- Full QM. GFN2-xTB is the ceiling for the semi-empirical tier; DFT is out of scope.
- Tautomer enumeration. Protonation states only. (Tautomers are a plausible v2.)
- Being a library first. It is a CLI first; the Python API is a byproduct and is not
  a stability commitment in v1.

## Verified ground truth

All of the following was measured on this machine (20-core CPU node, no GPU) before the
design was fixed. Timings are for a single molecule, single conformer.

| Backend | 3-quinuclidinone | gabapentin | Notes |
|---|---|---|---|
| MMFF94 (RDKit) | 7 ms | 5 ms | no dependency beyond RDKit |
| GFN2-xTB (`tblite`, pip wheel) | 0.14 s | 0.88 s | supports ALPB implicit solvation |
| AIMNet2 (local weights) | 4.3 s (incl. warm-up) | 0.45 s | consumes total charge |
| MACE-OFF23-medium (local weights) | 3.6 s | 6.8 s | no total-charge input |
| CREST 3.0.2 (conformer search) | — | 5 min 8 s → 145 conformers | 8 threads |

ETKDGv3 embeds all test molecules in under 35 ms and stereochemistry survives the
3D round-trip, verified against morphine (5 stereocenters) and tamoxifen (E/Z alkene)
as well as the two target molecules.

`Chem.MolToPDBBlock` already emits unique atom names, CONECT records, and formal charges
in columns 77-80. It does not need replacing, only wrapping.

### The zwitterion finding — drives a default

Optimizing gabapentin's zwitterion `[NH3+]CC1(CC(=O)[O-])CCCCC1` **in gas phase destroys
it**: the proton transfers back from N to O, leaving a neutral molecule with an O-H bond
of 1.00 Å. The user asked for one protonation state and silently received another.

With GFN2-xTB + ALPB water the zwitterion survives intact (3 N-H bonds, nearest O-H is
1.52 Å — a hydrogen bond, not a covalent one). AIMNet2 shows the same gas-phase collapse
and neither MLFF offers implicit solvation.

Two consequences, both binding:

1. When the input carries a net charge or is detected as a zwitterion, implicit solvation
   is enabled by default on backends that support it.
2. The pipeline **verifies the protonation state survived minimization** and fails loudly
   if it did not. Silently returning a different molecule than the one requested is the
   single worst thing this tool could do.

### Environment gotchas found

- AIMNet2 calls `torch.compile`, which needs Python dev headers this machine lacks
  (`Python.h: No such file or directory`). Setting `TORCHDYNAMO_DISABLE=1` fixes it;
  `ligand3d` sets this itself before importing the AIMNet2 calculator.
- Installing `mace-torch` pulls the CUDA torch build (3.4 GB) unless the CPU index is
  pinned. The `[mlff]` extra documents the CPU-only install.
- `tblite` exposes ALPB but not CPCM; requesting CPCM raises `TBLiteValueError`.
- `dimorphite-dl` emits RDKit valence warnings on its internal intermediates. Logging is
  suppressed around the call.

## Pipeline

```
input ─▶ standardize ─▶ protonate ─▶ embed ─▶ conformers ─▶ minimize ─▶ rank ─▶ write
```

Each stage is one module with one job, and each is usable on its own.

1. **input** (`molecule.py`) — SMILES string, `.mol`/`.sdf` file, or a molblock posted by
   the sketcher. Produces an RDKit `Mol` with explicit stereo perception.
2. **standardize** (`molecule.py`) — sanitize, assign CIP labels, record which
   stereocenters are *specified* versus *undefined*. Undefined stereocenters are an error
   by default; `--any-stereo` picks one arbitrarily, `--enumerate-stereo` fans out.
3. **protonate** (`protonate.py`) — default is **as-drawn**: what you drew is what you
   get. `--ph` runs dimorphite-dl (default 7.4 when the flag is given without a value).
   `--enumerate-states` writes one output per plausible state.
4. **embed** (`embed.py`) — ETKDGv3 with `useSmallRingTorsions` and `useMacrocycleTorsions`.
   Deterministic: a fixed seed unless `--seed` says otherwise.
5. **conformers** (`conformers.py`) — optional. `rdkit` backend (ETKDG multi-embed, RMS
   prune, FF-minimize, Butina cluster) is the default; `crest` backend shells out to the
   CREST binary when one is configured.
6. **minimize** (`minimize/`) — the pluggable stage, described below.
7. **rank + dedup** — sort by energy, drop duplicates by symmetry-corrected heavy-atom RMSD.
8. **write** (`write.py`) — `.pdb` (multi-conformer as MODEL/ENDMDL) plus an `.sdf`
   sidecar, because PDB does not carry bond orders reliably.

## The backend registry

This is the load-bearing abstraction and the part worth getting right. Backends differ in
ways the pipeline must reason about — notably, some potentials consume the total charge
and some do not — so each one declares its capabilities rather than the pipeline
special-casing names.

```python
@dataclass(frozen=True)
class Capabilities:
    name: str
    kind: Literal["ff", "semiempirical", "mlff"]
    takes_charge: bool           # model consumes total molecular charge
    supports_solvation: bool     # implicit solvent available
    elements: frozenset[int] | None   # supported Z; None = unrestricted
    requires: tuple[str, ...]    # python modules that must import

class Backend(Protocol):
    caps: Capabilities
    def available(self) -> Availability: ...          # ok, or a reason why not
    def minimize(self, job: MinimizeJob) -> MinimizeResult: ...
```

Before running, the pipeline checks the molecule against `caps` and refuses combinations
that would silently produce nonsense:

- net charge ≠ 0 on a backend with `takes_charge=False` → error, overridable with
  `--allow-charge-mismatch` (this is what makes MACE-MP different from AIMNet2)
- element outside `elements` → error naming the offending atom
- solvation requested on a backend with `supports_solvation=False` → error

Backends registered in v1:

| id | kind | takes charge | solvation | source |
|---|---|---|---|---|
| `mmff94` | ff | n/a | no | RDKit (default) |
| `uff` | ff | n/a | no | RDKit |
| `gfn2` | semiempirical | yes | ALPB | `tblite` wheel |
| `gfn1` | semiempirical | yes | ALPB | `tblite` wheel |
| `gfnff` | ff | yes | ALPB | `xtb` binary |
| `aimnet2` | mlff | yes | no | local weights |
| `mace-off` | mlff | no | no | local weights |
| `mace-mp` | mlff | no | no | local weights |

RDKit force fields use RDKit's own optimizer. Everything else converts to an ASE `Atoms`
and runs LBFGS. Both paths return the same `MinimizeResult`, so adding eSEN, UMA, or Orb
later is one new file and one registry entry.

Chaining is allowed and is the recommended pattern for the expensive backends:
`--backend mmff94,gfn2` pre-minimizes cheaply then refines.

## Resource discovery

Model weights and external binaries are large and machine-specific, so they never live in
the repo. `config.py` resolves each resource in order:

1. explicit CLI flag
2. environment variable (`LIGAND3D_MACE_OFF`, `LIGAND3D_XTB_BIN`, …)
3. user config at `~/.config/ligand3d/config.toml`
4. known-location probe (this cluster's `/net/databases/huggingface/mlFF_models/`,
   the `qcb-xtb` conda env's `bin/`)
5. upstream download into `~/.cache/ligand3d/`

`ligand3d doctor` prints what was found, what was not, and why — so a failure is
diagnosable without reading source. The same mechanism finds the Ketcher bundle.

## Sketcher

`ligand3d sketch` serves a small page from `http.server` on localhost, opens a browser,
and waits. The page hosts Ketcher (Apache-2.0); pressing "Use this molecule" POSTs a
molblock back, the server shuts down, and the molecule enters the pipeline exactly as if
it had been passed on the command line. `--no-browser` prints the URL instead, which is
what you want over SSH with a forwarded port.

Ketcher's prebuilt standalone bundle is 35 MB, so it is fetched on demand into
`~/.cache/ligand3d/ketcher/` rather than committed. The submodule under `vendor/ketcher`
pins the upstream source for provenance and for building from scratch if the release
asset ever disappears.

## Dependencies

Core, all from PyPI with pinned lower bounds: `rdkit`, `numpy`, `typer`, `rich`.
Extras: `[xtb]` → `tblite`, `ase`; `[mlff]` → `torch`, `mace-torch`, `aimnet`, `ase`;
`[protonation]` → `dimorphite-dl`; `[all]`.

The core install is ~200 MB and does everything except the semi-empirical and ML tiers.
That matters: the common case should not pay for torch.

## Testing

Property-based where the property is the point:

- **stereo round-trip** — for a battery of chiral molecules, SMILES → 3D → re-perceived
  CIP labels must equal the input's. This is the test that matters most.
- **protonation integrity** — after minimization, formal charges and heavy-atom
  connectivity must match what went in. Guards the zwitterion failure directly.
- **PDB validity** — output re-reads in RDKit with the same heavy-atom count and
  connectivity; atom names unique within the residue.
- **energy decreases** — minimization never raises the energy.
- **backend gating** — a charged molecule on a `takes_charge=False` backend errors rather
  than proceeding.
- **determinism** — same input and seed gives identical coordinates.

Backend-specific tests skip cleanly when the backend is unavailable, so the suite passes
on a laptop with only the core install.

## Repo layout

```
ligand3d/
├── pyproject.toml
├── README.md
├── LICENSE                       # MIT
├── .gitmodules                   # vendor/ketcher
├── docs/specs/
├── src/ligand3d/
│   ├── cli.py                    # build, sketch, doctor, backends
│   ├── config.py                 # resource discovery
│   ├── errors.py
│   ├── molecule.py               # parse, standardize, stereo audit
│   ├── protonate.py
│   ├── embed.py
│   ├── conformers.py
│   ├── pipeline.py
│   ├── write.py
│   ├── minimize/
│   │   ├── base.py               # Capabilities, Backend, registry
│   │   ├── rdkit_ff.py
│   │   ├── xtb.py
│   │   └── mlff.py
│   └── sketch/
├── tests/
└── examples/
```

## Build order

1. Core path: input → standardize → embed → write. MMFF94 only. Both target molecules
   producing valid PDB files.
2. Backend registry with capability gating; GFN2 via tblite; solvation defaults.
3. Protonation, including the post-minimization integrity check.
4. Conformer search: RDKit tier, then CREST.
5. MLFF backends.
6. Sketcher.

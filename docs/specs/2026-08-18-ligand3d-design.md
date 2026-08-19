# ligand3d — design

**Date:** 2026-08-18
**Status:** implemented. See the addendum at the end for what changed in practice.

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
diagnosable without reading source. The same mechanism finds the JSME bundle.

## Sketcher

`ligand3d sketch` serves a small page from `http.server` on localhost, opens a browser,
and waits. Pressing "Use this molecule" POSTs a molblock back, the server shuts down,
and the molecule enters the pipeline exactly as if it had been passed on the command
line. `--no-browser` prints the URL instead, which is what you want over SSH with a
forwarded port.

**Correction from the original plan.** The design assumed Ketcher could be fetched and
served directly. It cannot: `ketcher-standalone.zip` is the npm library — `index.js`,
`main.js`, and TypeScript declarations, with no HTML file anywhere in the archive — so
unzipping it yields nothing a browser can open. Using Ketcher means running its npm
build.

The default is therefore **JSME**, which genuinely is a drop-in: a 1 MB zip whose single
`jsme.nocache.js` loader runs entirely in the browser. It is fetched once into
`~/.cache/ligand3d/jsme/` and works offline thereafter, and it handles wedge and hash
bonds, so drawn stereocentres arrive as real stereochemistry.

If JSME cannot be fetched the page degrades to a paste box accepting SMILES or a
molblock. (Ketcher was later removed outright — see the addendum.)

One trap worth recording: the molblock must be returned to the pipeline **verbatim**.
Its first line is the molecule-name field and is routinely empty, so calling `.strip()`
on it shifts every subsequent line up one, puts the counts line where the header belongs,
and produces a file no parser accepts.

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


---

# Addendum, 2026-08-19

## The machine-learned force fields on this cluster

`/net/databases/huggingface/mlFF_models/` holds 19 model families, and that path is now
one of the built-in probe locations, so they resolve with no configuration. `config.MODELS`
is a single table describing each checkpoint — filename patterns, whether it consumes
total charge, which head to select, approximate size — and `minimize/mlff.py` derives one
backend per row from it. Adding a model is one table entry.

Eighteen of them load. Every `takes_charge` flag was set by measurement, not by reading
documentation: set `atoms.info["charge"]`, evaluate, and see whether the energy moved.

- **charge-aware:** MACE-omol, all eSEN, all UMA, both AllScAIP, AIMNet2
- **not charge-aware:** MACE-OFF23/24, MACE-MP, all multi-head MACE

That last one is worth stating plainly because it is counter-intuitive: the multi-head
MACE checkpoints have an `omol` head trained on charged data, but the calculator ignores
`atoms.info["charge"]` entirely. They are therefore registered with `takes_charge=False`
and the registry refuses to hand them an ion.

Two other discoveries about the MACE family:

- The multi-head checkpoints **refuse to load without a head selected**, and none of them
  names a head `default`. `ligand3d` picks one per registry entry (`omol`, or
  `spice_wB97M` for `mace-mh-spice`).
- MACE-POLAR-1 needs `graph_longrange` and a patched MACE fork. It cannot be installed
  from PyPI, so it is listed in `UNSUPPORTED_MODELS` with the reason, alongside SO3LR
  (JAX) and orb-mol (orb-models 0.7 removed its ASE calculator).

### The dependency conflict that shapes the install

`mace-torch` pins `e3nn==0.4.4`; `fairchem-core` requires `e3nn>=0.5`. There is no
version of either that resolves this, so **they cannot share an environment**. Installing
the second one silently breaks the first, and the failure surfaces as
`ValueError: too many values to unpack` from inside e3nn's codegen pickling — a message
that points nowhere near the cause.

`[mace]` and `[fairchem]` are therefore separate extras, and both backend families check
the installed e3nn version in `available()` and report the clash in plain language. This
is the kind of problem that has to be detected rather than documented, because nobody
reads the documentation until after the crash.

## The sketcher became a session

The original design had `ligand3d sketch` capture one molblock and exit. That is wrong for
how the tool is actually used: you draw several molecules in a row and want to see what
happened to each. It is now a persistent server with a small JSON API:

| endpoint | purpose |
|---|---|
| `GET /api/config` | backend catalog with availability, plus defaults |
| `GET /api/next-name` | first unused `sketch{N}.pdb` in the target directory |
| `POST /api/check-path` | resolve a target; report creation, overwrite, writability |
| `POST /api/build` | start a build; `409` if it would overwrite and no confirmation |
| `GET /api/job/{id}` | poll state and log |

Builds run on a background thread against a `ThreadingHTTPServer`, so the page polls the
log while the work proceeds. That matters for CREST, which takes minutes.

The overwrite flow is a deliberate two-step: the server returns `409` with the list of
files that exist, the page asks, and only a request carrying `overwrite: true` writes
anything. Nothing is clobbered without a human saying so.

The run log reports what the user asked for: how many stereocenters were found and their
R/S codes, double-bond geometry, a warning when the drawing contains more than one
fragment, the full text of any refusal, and the path of every file written.

One reporting subtlety: constrained bridgeheads must not be described as "left undefined".
3-quinuclidinone has two atoms that a graph analysis flags as stereogenic and that cannot
actually vary, and telling the user they are undefined sends them looking for a problem
that does not exist.

## Drawing dashed bonds

Verified rather than assumed: with all wedge flags cleared and exactly one bond set,
molfile bond flag 1 (wedge) reads back as *R* and flag 6 (hash) as *S* for the same 2D
layout. Both directions work, and there is a regression test.

In JSME there is no separate dash tool — the wedge tool cycles a bond wedge → hash →
plain on repeated clicks. The page says so inline, because it is not discoverable.

The JSME options string was also wrong: it passed `oldlook,star,newlook`, which asks for
the old and new look simultaneously. It is now `stereo,autoez,paste,multipart`.


## Ketcher removed, 2026-08-19

Ketcher was kept for a while as an opt-in second editor behind
`LIGAND3D_KETCHER_DIR`, with the upstream source pinned as a submodule. All of it
is gone now. The reasoning, since this reverses an earlier decision:

- The submodule reached **786 MB** on disk and contributed nothing at runtime. Its
  stated purpose was provenance and the option to build from source, but nothing in
  the project ever built from it, and anyone who wanted to build Ketcher would clone
  it from upstream themselves. Pinning a commit we had never built was ceremony.
- **Nothing tested the Ketcher path.** The one test covering it was lost when the
  sketcher was rewritten as a session, leaving an untested branch that claimed to
  work. An untested escape hatch is worse than no hatch: pointing the environment
  variable at a build and having it fail is a worse experience than the capability
  simply not being offered.
- It could not be exercised honestly in the first place, because using it requires an
  npm build that was never run here. Every claim about it was theoretical.
- JSME does the job in 1 MB and is verified end to end.

Removing it also turned up two things the audit would otherwise have missed:
`bridge_jsme.html` and `bridge_ketcher.html` had become orphans when the UI unified
on `app.html`, and `fallback.html` was **broken** — it still POSTed to `/submit`, an
endpoint that stopped existing when the server became a session. `app.html` already
handled the no-editor case with a paste box, so the fallback page was both dead and
redundant; there is now exactly one page.

If Ketcher is ever wanted, it should be added as a real feature with a real build to
test against, not restored as a stub.

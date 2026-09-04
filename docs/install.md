# Installing ligand3d

Everything beyond the one-line setup: what to install for each tier of method, and where the weights come from.

[← back to the README](../README.md)

## Install

The [Quickstart](../README.md#quickstart) covers the two normal cases. This is the full menu of
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
[the split](models.md#the-mace--fairchem-split--read-this-before-installing). `[names]` adds
offline IUPAC-name lookup for `fetch`, which also needs a JRE on `PATH`.

Verified from a clean clone: `git clone`, `uv venv`, `uv pip install -e
".[xtb,protonation,names]"`, then `uv run ligand3d build ...` and `uv run ligand3d sketch`
all work with nothing else present.


## Configuration

Model weights and external binaries are never bundled. Each resolves through environment
variables, then `~/.config/ligand3d/config.toml`, then `$PATH` and conventional install
locations, then `~/.cache/ligand3d/`. Generate a starter config with:

```bash
ligand3d config --init
```

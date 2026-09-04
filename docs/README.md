# ligand3d documentation

| | |
|---|---|
| [Installing](install.md) | what to install per tier, and where weights come from |
| [Methods](methods.md) | every level of theory, what it costs, how to choose |
| [ML potentials](models.md) | the neural tier and the environment split |
| [Chemistry](chemistry.md) | stereochemistry, protonation, conformers, name lookup |
| [Cluster](cluster.md) | SLURM, the IPD install, cutting a release |
| [Output formats](outputs.md) | mmCIF, SDF, PDB, Rosetta params, RFdiffusion4 |
| [How it works](internals.md) | the optimizer, measured accuracy, adding a method |

## Regenerating the README figures

Both are generated, so they cannot drift from what the tool does:

```bash
python docs/make_pipeline_figure.py     # docs/assets/pipeline.svg
python docs/make_structure_render.py    # docs/assets/structure.png  (needs PyMOL)
```

The first refuses to build if MMFF94 stops converging, since its last panel says it did.
The second builds through `ligand3d` itself and renders the mmCIF that run produced, so
the picture is of the actual output rather than something resembling it.

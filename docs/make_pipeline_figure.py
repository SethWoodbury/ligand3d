#!/usr/bin/env python3
"""Draw the README's pipeline figure: 2D sketch -> minimization -> 3D structure.

Generated rather than screenshotted, so it is reproducible, version-controlled,
and crisp at any size. Run it after changing anything it depicts:

    python docs/make_pipeline_figure.py

Writes docs/assets/pipeline.svg. Uses only RDKit and MMFF94, so it needs
nothing beyond the core install.
"""

from __future__ import annotations

import pathlib
import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ligand3d.embed import EmbedOptions, embed  # noqa: E402
from ligand3d.molecule import from_smiles  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "assets" / "pipeline.svg"

#: Gabapentin: small enough to read at this size, and a real ligand with a
#: zwitterion problem, which is the sort of thing this tool exists for.
SMILES = "NCC1(CC(=O)O)CCCCC1"

PANEL_W, PANEL_H = 210, 190
GAP = 46
INK = "#1a1a1a"
MUTED = "#6b7280"
ACCENT = "#2563eb"


def _draw(mol: Chem.Mol, conf_id: int = -1, wedge: bool = True) -> str:
    """One molecule as an SVG fragment, with the XML header stripped."""
    drawer = rdMolDraw2D.MolDraw2DSVG(PANEL_W, PANEL_H)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.bondLineWidth = 2
    opts.addStereoAnnotation = wedge
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, confId=conf_id, kekulize=True)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg[svg.index("<svg") :].replace("</svg>", "").split(">", 1)[1]


def build_frames() -> tuple[Chem.Mol, list[int], list[float]]:
    """Embed, then minimize while keeping a few frames along the way."""
    # ligand3d's own embedding, not a bare EmbedMolecule: the figure should
    # show what the tool does, including its ETKDGv3 settings.
    mol = embed(from_smiles(SMILES), EmbedOptions())

    keep = Chem.Mol(mol)          # frame 0, before any minimization
    keep.RemoveAllConformers()
    energies: list[float] = []

    props = AllChem.MMFFGetMoleculeProperties(mol)
    field = AllChem.MMFFGetMoleculeForceField(mol, props)
    field.Initialize()

    # Snapshots at the start, part-way, and at convergence — enough to read as
    # a trajectory without pretending three frames are the whole story.
    status = None
    for steps in (0, 12, 400):
        if steps:
            status = field.Minimize(maxIts=steps)
        conf = Chem.Conformer(mol.GetConformer())
        keep.AddConformer(conf, assignId=True)
        energies.append(field.CalcEnergy())
    # MMFF's Minimize returns 0 on convergence. The last panel says "converged",
    # so the figure should refuse to be built if that stops being true.
    if status != 0:
        raise SystemExit("MMFF94 did not converge; the figure's label would be a lie")

    # Superimpose the frames on the first one. Without this the molecule drifts
    # and tumbles between panels, and a reader cannot tell relaxation from
    # rotation — the figure would be showing motion that means nothing.
    AllChem.AlignMolConformers(keep)

    return keep, [c.GetId() for c in keep.GetConformers()], energies


def main() -> int:
    # Skeletal, as a chemist draws it. The hydrogens that appear in the next
    # panel are not a change of representation: `embed` adds them explicitly,
    # which the label says. Forcing them into the 2D panel only made it
    # unreadable.
    flat = Chem.MolFromSmiles(SMILES)
    AllChem.Compute2DCoords(flat)
    frames, ids, energies = build_frames()

    panels = [("drawn in 2D", _draw(flat), "")]
    labels = ["+H, embedded (ETKDGv3)", "MMFF94, 12 steps", "MMFF94, converged"]
    for n, (cid, label) in enumerate(zip(ids, labels)):
        panels.append((label, _draw(frames, cid, wedge=False),
                       f"{energies[n]:.1f} kcal/mol"))

    width = len(panels) * PANEL_W + (len(panels) - 1) * GAP + 40
    height = PANEL_H + 120

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="system-ui,-apple-system,'
        f'Segoe UI,Roboto,sans-serif">',
        # A light card rather than a transparent background with a dark-mode
        # media query. RDKit draws molecules in black, which all but vanishes on
        # GitHub's dark theme — and a media query keyed to the *reader's* system
        # preference does not track which GitHub theme the page is using, so it
        # can invert the text while leaving the molecules black. One background
        # renders identically everywhere.
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="#ffffff"/>',
        '<style>'
        f'.t{{fill:{INK};font-size:15px;font-weight:700;letter-spacing:.01em}}'
        f'.s{{fill:{MUTED};font-size:12px;font-weight:500}}'
        f'.a{{stroke:{ACCENT};stroke-width:2;fill:none}}'
        '</style>',
    ]

    for i, (label, body, sub) in enumerate(panels):
        x = 20 + i * (PANEL_W + GAP)
        parts.append(f'<g transform="translate({x},34)">{body}</g>')
        parts.append(f'<text class="t" x="{x + PANEL_W / 2}" y="24" '
                     f'text-anchor="middle">{label}</text>')
        if sub:
            parts.append(f'<text class="s" x="{x + PANEL_W / 2}" y="{PANEL_H + 58}" '
                         f'text-anchor="middle">{sub}</text>')
        if i < len(panels) - 1:
            ax = x + PANEL_W + 10
            ay = 34 + PANEL_H / 2
            parts.append(f'<path class="a" d="M{ax},{ay} L{ax + GAP - 20},{ay}"/>')
            parts.append(f'<path class="a" d="M{ax + GAP - 26},{ay - 5} '
                         f'l6,5 -6,5"/>')

    caption = [
        "One candidate\u2019s relaxation \u2014 the middle panels are optimizer "
        "snapshots, not different conformers.",
        "Frames are superimposed; 3D panels are projections. A default run searches "
        "many candidates and keeps the best.",
    ]
    for n, line in enumerate(caption):
        parts.append(
            f'<text class="s" x="{width / 2}" y="{height - 26 + n * 15}" '
            f'text-anchor="middle">{line}</text>'
        )
    parts.append("</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n")
    print(f"wrote {OUT.relative_to(HERE.parent)}  "
          f"({len(panels)} panels, {energies[0]:.1f} -> {energies[-1]:.1f} kcal/mol)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

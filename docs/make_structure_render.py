#!/usr/bin/env python3
"""Ball-and-stick render of the structure ligand3d actually writes.

The pipeline strip beside it is drawn with RDKit's 2D drawer, which projects a
3D geometry onto the page — accurate, but it does not look three-dimensional,
and the whole point of the tool is that a 3D structure comes out. So this
renders the real output file, in PyMOL, the way someone would open it.

    python docs/make_structure_render.py

Writes docs/assets/structure.png. Needs PyMOL on PATH; skips with a message
rather than failing if it is missing, since it is only a documentation asset.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "assets" / "structure.png"

#: The same molecule as the pipeline figure, so the two read as one story.
SMILES = "NCC1(CC(=O)O)CCCCC1"

SCRIPT = """
load {cif}, m
hide everything
show sticks, m
show spheres, m
set sphere_scale, 0.26
set stick_radius, 0.115
color grey45, elem C
color grey75, elem H
color firebrick, elem O
color marine, elem N
bg_color white
set ray_opaque_background, 1
set antialias, 2
# Outlines rather than a glossy render: this sits in a README at small size,
# where specular highlights read as noise and a black edge reads as shape.
set ray_trace_mode, 1
set ray_trace_color, black
set ray_trace_gain, 0.12
# Depth fog blurs the far side of the molecule, which at this size looks like
# a rendering fault rather than depth.
set depth_cue, 0
set ray_trace_fog, 0
set specular, 0.15
set ambient, 0.42
set direct, 0.55
orient m
zoom m, -0.3
ray 1500, 780
png {png}, dpi=200
"""


def main() -> int:
    pymol = shutil.which("pymol")
    if not pymol:
        print("pymol is not on PATH; leaving docs/assets/structure.png alone")
        return 0

    ligand3d = ROOT / ".venv" / "bin" / "ligand3d"
    if not ligand3d.exists():
        ligand3d = pathlib.Path(shutil.which("ligand3d") or "")
    if not ligand3d or not ligand3d.exists():
        print("ligand3d is not available; cannot build the structure to render")
        return 1

    with tempfile.TemporaryDirectory(prefix="ligand3d-render-") as tmp:
        work = pathlib.Path(tmp)
        # Built through ligand3d rather than RDKit directly, so the image is of
        # the file the tool ships — not of something that merely resembles it.
        built = subprocess.run(
            [str(ligand3d), "build", SMILES, "-b", "mmff94", "--stereo", "any",
             "-o", str(work / "m.cif"), "--no-trace", "-q"],
            capture_output=True, text=True, timeout=900, cwd=work,
        )
        cif = work / "m.cif"
        if built.returncode != 0 or not cif.exists():
            print("build failed:\n" + (built.stderr or built.stdout)[-800:])
            return 1

        pml = work / "render.pml"
        pml.write_text(SCRIPT.format(cif=cif, png=work / "out.png"))
        subprocess.run([pymol, "-cq", str(pml)], capture_output=True,
                       text=True, timeout=900, cwd=work)

        rendered = work / "out.png"
        if not rendered.exists():
            print("pymol produced no image")
            return 1
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_bytes(rendered.read_bytes())

    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""2D depiction of how ligand3d reads a structure.

The point of this is not decoration. When the pipeline says "atom 4 has no
configuration", that number is an index into the file you supplied, and there is
no way to tell from a drawing which atom it means. Worse, the index depends on
the order the sketcher happened to emit — the same molecule drawn twice can
number differently — so the message is unactionable on its own.

This renders the molecule *as parsed*, with every atom carrying its index and
the problematic ones highlighted, so the number in the message points at
something visible.

The layout comes from the input's own 2D coordinates whenever it has them, so
the picture matches what the user drew rather than a fresh RDKit layout of the
same graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from .molecule import Molecule, has_real_stereo_ambiguity, rdkit_quiet

# Highlights, chosen to stay legible on both light and dark backgrounds.
_UNDEFINED = (0.95, 0.55, 0.15)   # amber: needs a decision
_DEFINED = (0.25, 0.65, 0.45)     # green: already specified
_CONSTRAINED = (0.55, 0.60, 0.70) # grey: looks stereogenic but cannot vary


@dataclass
class Depiction:
    """An SVG picture plus what it is showing."""

    svg: str
    n_atoms: int
    undefined_centers: tuple[int, ...] = ()
    defined_centers: tuple[tuple[int, str], ...] = ()
    constrained_centers: tuple[int, ...] = ()
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "svg": self.svg,
            "n_atoms": self.n_atoms,
            "undefined_centers": list(self.undefined_centers),
            "defined_centers": [
                {"atom": i, "code": c} for i, c in self.defined_centers
            ],
            "constrained_centers": list(self.constrained_centers),
            "notes": self.notes,
        }


def depict(
    molecule: Molecule,
    width: int = 480,
    height: int = 360,
    dark: bool = False,
    show_indices: bool = True,
) -> Depiction:
    """Draw the molecule with atom indices and stereo highlights."""
    mol = Chem.Mol(molecule.mol)

    with rdkit_quiet():
        # Keep the user's own layout when there is one; only lay out from
        # scratch if the input had no coordinates at all.
        if not mol.GetNumConformers():
            rdDepictor.Compute2DCoords(mol)
        elif mol.GetConformer().Is3D():
            rdDepictor.Compute2DCoords(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        try:
            Chem.WedgeMolBonds(mol, mol.GetConformer())
        except Exception:
            pass

    audit = molecule.stereo
    defined = tuple(audit.assigned_centers)
    ambiguous = has_real_stereo_ambiguity(molecule)
    undefined = tuple(audit.unassigned_centers) if ambiguous else ()
    constrained = () if ambiguous else tuple(audit.unassigned_centers)

    highlights: dict[int, tuple] = {}
    for idx in undefined:
        highlights[idx] = _UNDEFINED
    for idx, _ in defined:
        highlights[idx] = _DEFINED
    for idx in constrained:
        highlights[idx] = _CONSTRAINED

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.addAtomIndices = bool(show_indices)
    options.addStereoAnnotation = True
    options.highlightRadius = 0.35
    options.bondLineWidth = 2
    if dark:
        options.setBackgroundColour((0, 0, 0, 0))
        _use_dark_palette(options)
    else:
        options.setBackgroundColour((1, 1, 1, 0))

    with rdkit_quiet():
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            highlightAtoms=list(highlights),
            highlightAtomColors=highlights,
        )
    drawer.FinishDrawing()

    notes: list[str] = []
    if undefined:
        notes.append(
            f"{len(undefined)} atom(s) highlighted in amber need a configuration"
        )
    if constrained:
        notes.append(
            f"{len(constrained)} atom(s) in grey look stereogenic but are fixed by "
            "the ring system"
        )
    if defined:
        notes.append(f"{len(defined)} stereocenter(s) in green are already specified")

    return Depiction(
        svg=drawer.GetDrawingText(),
        n_atoms=mol.GetNumAtoms(),
        undefined_centers=undefined,
        defined_centers=defined,
        constrained_centers=constrained,
        notes=notes,
    )


def _use_dark_palette(options) -> None:
    """Light atom colours for a dark page.

    RDKit's default palette assumes a white background, so carbon labels come
    out near-black and vanish.
    """
    palette = {
        6: (0.88, 0.90, 0.94),   # carbon
        7: (0.45, 0.65, 0.98),   # nitrogen
        8: (0.98, 0.45, 0.42),   # oxygen
        9: (0.55, 0.90, 0.60),   # fluorine
        15: (0.98, 0.65, 0.35),  # phosphorus
        16: (0.95, 0.85, 0.35),  # sulfur
        17: (0.55, 0.90, 0.55),  # chlorine
        35: (0.85, 0.60, 0.40),  # bromine
        53: (0.75, 0.55, 0.90),  # iodine
        0: (0.88, 0.90, 0.94),
    }
    options.updateAtomPalette(palette)
    # Three separate colour properties, and the atom indices use the one that is
    # easiest to miss: `addAtomIndices` draws through atomNoteColour, not
    # annotationColour. Setting only the latter leaves the numbers near-black on
    # a dark page — invisible, and they are the whole reason this panel exists.
    grey = (0.74, 0.78, 0.85)
    options.atomNoteColour = grey
    options.bondNoteColour = grey
    options.annotationColour = grey
    options.legendColour = grey
    options.symbolColour = (0.88, 0.90, 0.94)


def depict_molblock(
    molblock: str,
    width: int = 480,
    height: int = 360,
    dark: bool = False,
    show_indices: bool = True,
) -> Depiction:
    """Depict a molblock straight from the sketcher.

    Parses leniently: a half-finished drawing should still render rather than
    blanking the panel while the user is mid-edit.
    """
    from .molecule import from_molblock

    molecule = from_molblock(molblock)
    return depict(molecule, width=width, height=height, dark=dark,
                  show_indices=show_indices)

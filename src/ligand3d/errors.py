"""Exception hierarchy.

Every failure a user can cause should raise a Ligand3DError with a message that
says what to do about it. Anything else escaping is a bug.
"""

from __future__ import annotations


class Ligand3DError(Exception):
    """Base class for all errors this package raises deliberately."""


class InputError(Ligand3DError):
    """The input molecule could not be read or is chemically invalid."""


class StereoError(Ligand3DError):
    """Stereochemistry is undefined, or was not preserved through 3D generation."""


class EmbedError(Ligand3DError):
    """Distance-geometry embedding failed to produce a conformer."""


class BackendUnavailable(Ligand3DError):
    """A minimization backend was requested but cannot run here."""


class BackendMismatch(Ligand3DError):
    """The molecule and the chosen backend are incompatible.

    Raised for the cases where proceeding would silently produce a wrong answer:
    a charged molecule on a potential with no charge channel, an element outside
    the model's training set, or solvation on a backend that has none.
    """


class MinimizationError(Ligand3DError):
    """The optimizer ran but did not produce a usable structure."""


class ProtonationError(Ligand3DError):
    """Protonation state assignment failed, or did not survive minimization."""


class ResourceNotFound(Ligand3DError):
    """A model weight file or external binary could not be located."""

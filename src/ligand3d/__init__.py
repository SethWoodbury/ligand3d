"""ligand3d — turn a 2D molecule into a minimized 3D structure."""

from __future__ import annotations

# Kept in step with pyproject.toml by a test rather than read from package
# metadata at import time: importlib.metadata pulls in email.parser, and
# this import is on the path of every CLI invocation.
__version__ = "0.3.1"

from .errors import (
    BackendMismatch,
    BackendUnavailable,
    EmbedError,
    InputError,
    Ligand3DError,
    MinimizationError,
    ProtonationError,
    ResourceNotFound,
    StereoError,
)
from .molecule import Molecule, from_file, from_molblock, from_smiles, read_input

__all__ = [
    "__version__",
    "BackendMismatch",
    "BackendUnavailable",
    "EmbedError",
    "InputError",
    "Ligand3DError",
    "MinimizationError",
    "Molecule",
    "ProtonationError",
    "ResourceNotFound",
    "StereoError",
    "from_file",
    "from_molblock",
    "from_smiles",
    "read_input",
]

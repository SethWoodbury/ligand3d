"""Implicit solvents available through ALPB.

GFN-xTB's ALPB model ships a table of parameterized solvents; anything outside
it is rejected by tblite at run time with a message that does not list the
alternatives. This module holds the names that were verified to work here, so a
typo is caught before a minimization starts and the error can say what to use
instead.

Every entry was probed against `tblite.ase.TBLite(solvation=("alpb", name))` on
this machine. Cyclohexane, octane, and heptane are *not* in the table despite
being obvious candidates, which is exactly the kind of thing worth checking
rather than assuming.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Solvent:
    """One ALPB solvent."""

    name: str
    aliases: tuple[str, ...] = ()
    dielectric: float | None = None
    note: str = ""

    def matches(self, query: str) -> bool:
        query = query.strip().lower()
        return query == self.name or query in self.aliases


# Ordered roughly by how often a ligand person reaches for them.
SOLVENTS: tuple[Solvent, ...] = (
    Solvent("water", ("h2o",), 80.1, "the default for charged and zwitterionic species"),
    Solvent("methanol", (), 32.7),
    Solvent("ethanol", (), 24.5),
    Solvent("acetonitrile", (), 37.5),
    Solvent("dmso", (), 46.7, "dimethyl sulfoxide"),
    Solvent("dmf", (), 36.7, "N,N-dimethylformamide"),
    Solvent("acetone", (), 20.7),
    Solvent("thf", (), 7.6, "tetrahydrofuran"),
    Solvent("dichloromethane", ("ch2cl2",), 8.9),
    Solvent("chloroform", ("chcl3",), 4.8),
    Solvent("ethylacetate", (), 6.0),
    Solvent("diethylether", ("ether",), 4.3),
    Solvent("dioxane", (), 2.2),
    Solvent("toluene", (), 2.4),
    Solvent("benzene", (), 2.3),
    Solvent("hexane", (), 1.9),
    Solvent("hexadecane", (), 2.1),
    Solvent("octanol", (), 10.3, "dry 1-octanol"),
    Solvent("woctanol", (), 8.1, "water-saturated 1-octanol; the logP phase"),
    Solvent("phenol", (), 12.4),
    Solvent("aniline", (), 6.9),
    Solvent("benzaldehyde", (), 17.8),
    Solvent("nitromethane", (), 35.9),
    Solvent("furane", (), 3.0, "spelled 'furane' by xtb, not 'furan'"),
    Solvent("carbondisulfide", ("cs2",), 2.6),
)

BY_NAME: dict[str, Solvent] = {}
for _solvent in SOLVENTS:
    BY_NAME[_solvent.name] = _solvent
    for _alias in _solvent.aliases:
        BY_NAME[_alias] = _solvent

# Names xtb does not have an ALPB parameterization for, but which people try.
# Naming the nearest usable stand-in is more helpful than a bare rejection.
NOT_PARAMETERIZED: dict[str, str] = {
    "cyclohexane": "hexane",
    "chex": "hexane",
    "octane": "hexane",
    "heptane": "hexane",
    "hept": "hexane",
    "pentane": "hexane",
    "isopropanol": "ethanol",
    "propanol": "ethanol",
    "butanol": "octanol",
    "pyridine": "aniline",
    "acetic acid": "ethanol",
}


def names() -> list[str]:
    """Canonical solvent names, in table order."""
    return [s.name for s in SOLVENTS]


def resolve(query: str) -> Solvent | None:
    """Look up a solvent by name or alias."""
    return BY_NAME.get(query.strip().lower())


def validate(query: str) -> str:
    """Return the canonical name, or raise with something actionable.

    Called before any minimization so a typo costs nothing.
    """
    from .errors import BackendMismatch

    key = query.strip().lower()
    found = resolve(key)
    if found is not None:
        return found.name

    suggestion = NOT_PARAMETERIZED.get(key)
    if suggestion:
        raise BackendMismatch(
            f"ALPB has no parameters for {query!r}. The closest available solvent is "
            f"{suggestion!r}. Run 'ligand3d solvents' for the full list."
        )

    close = [n for n in names() if key in n or n in key]
    hint = f" Did you mean {close[0]!r}?" if close else ""
    raise BackendMismatch(
        f"unknown solvent {query!r}.{hint} "
        f"Run 'ligand3d solvents' to see all {len(SOLVENTS)} of them."
    )

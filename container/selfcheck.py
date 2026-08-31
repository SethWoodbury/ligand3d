"""Refuse to ship an image that cannot do what it claims.

Run inside the container at build time. Every failure here is one a labmate
would otherwise hit on their first command, at which point the image is
already distributed and the cause is somebody else's afternoon.

Also runnable against a normal install — `python container/selfcheck.py` —
which is how you check a machine has the pieces before blaming the tool.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

REQUIRED_BACKENDS = ("mmff94", "uff", "gfn1", "gfn2")
#: GFN-FF runs through the standalone xtb binary rather than tblite, so it
#: is reported but not required — an image without it is still useful.
OPTIONAL_BACKENDS = ("gfnff",)

#: The one thing that differs between the images. Without this the check is
#: family-blind: a mace image whose torch stack failed to resolve passes every
#: other line here and ships with nine backends that cannot load.
#:
#: The *import* is what gets checked, not `available()`. Weights live on /net
#: and nothing is bound during a build, so availability would report false for
#: an image that is in fact fine.
FAMILY_MODULES = {"mace": "mace", "fairchem": "fairchem"}


def main() -> int:
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if condition else 'FAIL'}  {label}{f' — {detail}' if detail and not condition else ''}")
        if not condition:
            failures.append(label)

    print("ligand3d image self-check")

    for module in ("rdkit", "numpy", "typer", "rich", "gemmi"):
        check(f"import {module}", _importable(module))

    for module, why in (
        ("tblite", "GFN1/GFN2 would be unavailable"),
        ("dimorphite_dl", "pH protonation would be unavailable"),
        ("py2opsin", "offline name lookup would be unavailable"),
    ):
        check(f"import {module}", _importable(module), why)

    # py2opsin ships the OPSIN jar but shells out to java, so the jar alone
    # proves nothing.
    check("java on PATH", shutil.which("java") is not None, "OPSIN needs a JRE")
    try:
        from ligand3d.resolve import opsin_available

        check("OPSIN usable", opsin_available())
    except Exception as exc:  # pragma: no cover - import failure already logged
        check("OPSIN usable", False, str(exc))

    try:
        from ligand3d.minimize import get_backend

        for name in REQUIRED_BACKENDS:
            availability = get_backend(name).available()
            check(f"backend {name}", bool(availability), availability.reason)
        for name in OPTIONAL_BACKENDS:
            availability = get_backend(name).available()
            print(f"  {'ok  ' if availability else 'skip'}  backend {name} "
                  f"(optional){'' if availability else ' — ' + availability.reason}")
    except Exception as exc:  # pragma: no cover
        check("backends importable", False, str(exc))

    family = os.environ.get("LIGAND3D_FAMILY", "core")
    if family in FAMILY_MODULES:
        module = FAMILY_MODULES[family]
        check(f"import {module} (the {family} image's reason to exist)", _importable(module))

    # The one that matters: a real build, end to end.
    result = subprocess.run(
        [sys.executable, "-m", "ligand3d.cli", "build", "O=C1CN2CCC1CC2",
         "--backend", "mmff94,gfn2", "--dry-run", "--no-trace"],
        capture_output=True, text=True, timeout=600,
    )
    check("a real build runs", result.returncode == 0,
          (result.stdout + result.stderr)[-400:])

    if failures:
        print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("\nall checks passed")
    return 0


def _importable(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())

"""Open the sketcher in a browser without touching the user's own session.

`webbrowser.open` ends up running `xdg-open <url>`, which for a Chromium-family
default browser means launching `chrome <url>`. That normally hands the URL to
the already-running browser, which opens a tab — harmless. It relies on the
three singleton files in the Chrome profile (`SingletonLock`, `SingletonSocket`,
`SingletonCookie`) naming a live process.

When they don't — Chrome was killed, crashed, ran out of memory, or another tool
borrowed the profile — Chrome treats the lock as stale and *takes ownership of
the profile itself*. The sketcher's window is then the browser, so closing it
quits Chrome and takes every unrelated tab with it. That is a bad way for a
local tool to behave, and it is not hypothetical: it happened here.

So for Chromium-family browsers we open a dedicated window against a profile of
our own under the ligand3d cache. It cannot take over the user's profile, cannot
be taken over by it, and closing it closes nothing else. Firefox and everything
else go through `webbrowser`, which is well behaved for them.

`LIGAND3D_BROWSER` overrides the choice:

    auto     the default described above
    system   always hand off to the desktop's default browser
    none     never open anything; just print the URL
"""

from __future__ import annotations

import os
import shutil
import subprocess
import webbrowser
from pathlib import Path

#: Checked in order. All of these share Chrome's profile-singleton design, so
#: all of them carry the same hazard.
_CHROMIUM_BINARIES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "brave-browser",
    "microsoft-edge",
)


def _profile_dir() -> Path:
    from ..config import CACHE_DIR

    return CACHE_DIR / "browser"


def _chromium() -> str | None:
    """The Chromium-family browser the desktop would actually use, if any.

    Deliberately not "whichever is installed": on a machine where Firefox is the
    default, opening the sketcher in Chrome because Chrome happens to be present
    would be its own kind of rude.
    """
    if not any(os.environ.get(v) for v in ("DISPLAY", "WAYLAND_DISPLAY")):
        return None
    try:
        default = subprocess.run(
            ["xdg-settings", "get", "default-web-browser"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().lower()
    except Exception:
        default = ""
    if default and not any(n in default for n in ("chrome", "chromium", "brave", "edge")):
        return None
    for name in _CHROMIUM_BINARIES:
        found = shutil.which(name)
        if found:
            return found
    return None


def open_url(url: str, mode: str | None = None) -> bool:
    """Show `url` to the user. True if something was launched.

    `mode` is `--system-browser` and friends: an explicit request from the
    command line, which outranks the environment. Left as None the environment
    decides, and failing that the isolated default.
    """
    mode = (mode or os.environ.get("LIGAND3D_BROWSER", "auto")).strip().lower()
    if mode == "none":
        return False
    if mode != "system":
        binary = _chromium()
        if binary:
            profile = _profile_dir()
            try:
                profile.mkdir(parents=True, exist_ok=True)
                subprocess.Popen(
                    [
                        binary,
                        f"--user-data-dir={profile}",
                        # Without these the first launch of a fresh profile
                        # opens onboarding tabs over the sketcher.
                        "--no-first-run",
                        "--no-default-browser-check",
                        "--new-window",
                        url,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return True
            except Exception:
                # Fall through: a browser we could not launch is not a reason to
                # fail to start the server.
                pass
    try:
        return webbrowser.open(url)
    except Exception:
        return False

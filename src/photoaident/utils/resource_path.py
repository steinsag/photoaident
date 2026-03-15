"""Shared helper for locating bundled resource files.

Works correctly in both development (running from the source tree) and when
packaged by PyInstaller (where ``sys._MEIPASS`` points to the bundle root).
"""

import sys
from pathlib import Path

# resource_path.py lives at src/photoaident/utils/resource_path.py.
# The project root is four parents up from this file.
_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def get_resource_path(relative_path: str) -> str:
    """Return the absolute path to a bundled resource.

    Args:
        relative_path: Path relative to the project root (e.g.
            ``"assets/icons/search.svg"``).

    Returns:
        Absolute path as a string, valid in both dev and PyInstaller bundles.
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        return str(Path(meipass) / relative_path)
    return str(_PROJECT_ROOT / relative_path)


def icon_path(name: str) -> str:
    """Return the absolute path to an icon in ``assets/icons/``.

    Args:
        name: Icon filename, e.g. ``"zoom-in.svg"``.
    """
    return get_resource_path(f"assets/icons/{name}")

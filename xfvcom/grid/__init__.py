"""
xfvcom.grid – FVCOM horizontal-grid utilities
--------------------------------------------
Public:
    * FvcomGrid
    * get_grid
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING


# ---------------------------------------------------------------
# Lazy loader: the first attribute access triggers real import
# ---------------------------------------------------------------
def _load() -> ModuleType:
    """Import xfvcom.grid.grid_obj exactly once and cache it."""
    mod = importlib.import_module(".grid_obj", __name__)
    sys.modules[f"{__name__}.grid_obj"] = mod
    return mod


if TYPE_CHECKING:  # <-- Mypy / IDE
    from .grid_obj import FvcomGrid, get_grid, read_grid  # noqa
else:

    def __getattr__(name: str):
        mod = _load()
        return getattr(mod, name)

    # ---------- public helper (calls into grid_obj lazily) -----------------
    def read_grid(path: str | Path, *, utm_zone: int, northern: bool = True):
        """
        Read a ``*_grd.dat`` file and return :class:`FvcomGrid`.
        This thin wrapper keeps the lazy-import behaviour intact.
        """
        mod = _load()  # grid_obj is now imported
        return mod.FvcomGrid.from_dat(path, utm_zone=utm_zone, northern=northern)

    def __dir__():
        return sorted({"FvcomGrid", "get_grid", "read_grid"})


__all__ = ["FvcomGrid", "get_grid", "read_grid"]

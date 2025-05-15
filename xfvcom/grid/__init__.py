"""
xfvcom.grid – FVCOM horizontal-grid utilities
--------------------------------------------
外部公開:
    * FvcomGrid
    * get_grid
"""

import importlib
import sys
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


if TYPE_CHECKING:  # <-- Mypy / IDE 用
    from .grid_obj import FvcomGrid, get_grid  # noqa
else:

    def __getattr__(name: str):
        mod = _load()
        return getattr(mod, name)

    def __dir__():
        return sorted({"FvcomGrid", "get_grid"})


__all__ = ["FvcomGrid", "get_grid"]

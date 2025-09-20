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

    def calculate_node_area(
        grid_file: str | Path,
        node_indices: list[int] | None = None,
        utm_zone: int = 54,
        index_base: int = 1,
    ) -> float:
        """
        Calculate total area of triangular elements containing specified nodes.

        This is a convenience function that loads the grid and calculates areas
        in one step.

        Parameters
        ----------
        grid_file : str | Path
            Path to FVCOM grid file (.dat format)
        node_indices : list[int] | None
            List of node indices. If None, calculates area for all nodes.
        utm_zone : int
            UTM zone for coordinate transformation (default is 54)
        index_base : int
            0 for zero-based indexing, 1 for one-based indexing (FVCOM default)

        Returns
        -------
        float
            Total area in square meters

        Examples
        --------
        >>> from xfvcom.grid import calculate_node_area
        >>> area = calculate_node_area("grid.dat", [100, 200, 300], utm_zone=54)
        >>> print(f"Total area: {area:.0f} m²")
        """
        from pathlib import Path

        grid_path = Path(grid_file)
        if not grid_path.exists():
            raise FileNotFoundError(f"Grid file not found: {grid_file}")

        mod = _load()
        grid = mod.FvcomGrid.from_dat(grid_path, utm_zone=utm_zone)
        return grid.calculate_node_area(node_indices, index_base)

    def __dir__():
        return sorted({"FvcomGrid", "get_grid", "read_grid", "calculate_node_area"})


__all__ = ["FvcomGrid", "get_grid", "read_grid", "calculate_node_area"]

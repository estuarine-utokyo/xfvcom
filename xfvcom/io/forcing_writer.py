"""Write FVCOM forcing NetCDF (surface atmosphere) with embedded grid."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from netCDF4 import Dataset, date2num, num2date
from xarray import DataArray

from ..grid.grid_obj import FvcomGrid


class ForcingNcWriter:
    """Minimal writer for surface forcing with static horizontal grid."""

    def __init__(
        self,
        grid: FvcomGrid,
        forcing: Mapping[str, DataArray],
        *,
        out_path: str | Path,
        calendar: str = "gregorian",
        units: str = "seconds since 1970-01-01 00:00:00",
    ) -> None:
        self.grid = grid
        self.forcing = forcing
        self.out_path = Path(out_path)
        self.calendar = calendar
        self.units = units

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def write(self) -> None:
        """Create NetCDF file `self.out_path`."""
        nc = Dataset(self.out_path, "w", format="NETCDF4")

        # --- dimensions
        nc.createDimension("node", self.grid.node)
        nc.createDimension("nele", self.grid.nele)
        nc.createDimension("three", 3)
        time_dim = nc.createDimension("time", None)

        # --- grid vars
        nc.createVariable("x", "f8", ("node",))[:] = self.grid.x
        nc.createVariable("y", "f8", ("node",))[:] = self.grid.y
        nc.createVariable("nv", "i4", ("nele", "three"))[:] = (
            self.grid.nv.T + 1
        )  # 1-based

        nc["x"].units = "m"
        nc["y"].units = "m"
        nc["nv"].long_name = "elements"

        # --- time
        # assume all forcing arrays share same 'time' coordinate
        time_vals = None
        for da in self.forcing.values():
            time_vals = da["time"].values
            break
        assert time_vals is not None
        tvar = nc.createVariable("time", "f8", ("time",))
        tvar.units = self.units
        tvar.calendar = self.calendar
        tvar[:] = date2num(
            time_vals.astype("M8[ms]").astype(object), self.units, self.calendar
        )

        # --- forcing vars
        for name, da in self.forcing.items():
            var = nc.createVariable(name, "f4", ("time", "node"), zlib=True)
            var[:] = da.values

        nc.close()

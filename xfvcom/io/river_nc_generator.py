from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, List

import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from .base_generator import BaseGenerator
from .rivers_nml_parser import parse_rivers_nml
from .sources.base import ConstantSource, RiverTimeSeriesSource


class RiverNetCDFGenerator(BaseGenerator):
    """Generate NetCDF-4 river forcing file from NML and constant sources."""

    def __init__(
        self,
        nml_path: Path,
        start: str,
        end: str,
        dt_seconds: int,
        default_flux: float = 0.0,
        default_temp: float = 20.0,
        default_salt: float = 0.0,
    ) -> None:
        super().__init__(nml_path)
        self.start = pd.Timestamp(start, tz="UTC")
        self.end = pd.Timestamp(end, tz="UTC")
        self.dt = dt_seconds
        self.default_flux = default_flux
        self.default_temp = default_temp
        self.default_salt = default_salt

    # --------------------------------------------------------------- #
    # Abstract-method overrides                                      #
    # --------------------------------------------------------------- #
    def load(self) -> None:
        """Parse rivers.nml and build timeline."""
        self.rivers = parse_rivers_nml(
            self.source
        )  # self.source is Path from BaseGenerator
        self.timeline = pd.date_range(
            self.start, self.end, freq=f"{self.dt}S", inclusive="both"
        )

    def validate(self) -> None:
        if not self.rivers:
            raise ValueError("No river entries found in NML.")

    def render(self) -> bytes:
        """Return NetCDF binary (bytes) compatible with BaseGenerator.write()."""
        data_vars = {}
        for r in self.rivers:
            src: RiverTimeSeriesSource = ConstantSource(
                self.default_flux, self.default_temp, self.default_salt
            )
            data_vars[f"{r}_flow"] = ("time", src.get_series("flux", self.timeline))
            data_vars[f"{r}_temp"] = ("time", src.get_series("temp", self.timeline))
            data_vars[f"{r}_salt"] = ("time", src.get_series("salt", self.timeline))

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={"time": self.timeline},
            attrs={"title": "FVCOM river forcing (constant)"},
        )

        buffer = BytesIO()
        ds.to_netcdf(buffer, engine="netcdf4", format="NETCDF4")
        return buffer.getvalue()

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from .base_generator import BaseGenerator
from .rivers_nml_parser import parse_rivers_nml  # すぐ下で実装
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

    # ---------- main ----------
    def generate(self) -> Path:
        rivers = parse_rivers_nml(self.source)
        timeline = pd.date_range(
            self.start, self.end, freq=f"{self.dt}S", inclusive="both"
        )

        data_vars = {}
        for rname in rivers:
            src: RiverTimeSeriesSource = ConstantSource(
                self.default_flux, self.default_temp, self.default_salt
            )
            data_vars[f"{rname}_flow"] = (
                ("time",),
                src.get_series("flux", timeline),
            )
            data_vars[f"{rname}_temp"] = (
                ("time",),
                src.get_series("temp", timeline),
            )
            data_vars[f"{rname}_salt"] = (
                ("time",),
                src.get_series("salt", timeline),
            )

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={"time": timeline},
            attrs={"title": "FVCOM river forcing (constant)"},
        )

        out_path = self.source.with_suffix(".nc")
        ds.to_netcdf(out_path, engine="netcdf4", format="NETCDF4")
        return out_path

# -*- coding: utf-8 -*-
"""Generate a constant-value meteorological forcing NetCDF-4 for FVCOM."""

from __future__ import annotations

import tempfile
from datetime import timezone
from io import BytesIO
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd

from .base_generator import BaseGenerator


class MetNetCDFGenerator(BaseGenerator):
    """
    Build a meteorological forcing file whose dynamic fields are spatially
    uniform and temporally constant.

    Parameters
    ----------
    grid_nc : Path
        NetCDF file that contains static mesh variables (`x`, `y`, `xc`, `yc`,
        `lon`, `lat`, `lonc`, `latc`, `nv`).
    start, end : str
        ISO-8601 UTC datetimes (e.g. '2025-01-01T00:00:00Z').
    dt_seconds : int
        Time step in seconds.
    consts : dict[str, float]
        Constant values to write (see `_DEFAULTS` for keys).
    """

    _DEFAULTS = dict(
        uwind=0.0,  # m/s      positive east → west
        vwind=0.0,  # m/s  positive north → south
        air_temp=20.0,  # deg C
        rh=80.0,  # %
        prmsl=1013.0,  # hPa
        swrad=200.0,  # W m-2
        lwrad=300.0,  # W m-2
        precip=0.0,  # kg m-2 s-1 (= mm s-1)
    )

    def __init__(
        self,
        grid_nc: Path,
        start: str,
        end: str,
        dt_seconds: int = 3600,
        **consts: float,
    ) -> None:
        super().__init__(grid_nc)
        self.start = pd.Timestamp(start, tz="UTC")
        self.end = pd.Timestamp(end, tz="UTC")
        self.dt = dt_seconds
        self.consts = {**self._DEFAULTS, **consts}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_mjd(t: pd.DatetimeIndex) -> np.ndarray:
        origin = pd.Timestamp("1858-11-17T00:00:00Z")
        return ((t - origin) / pd.Timedelta("1D")).astype("float32")

    @staticmethod
    def _itime_pair(mjd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        itime = mjd.astype("int32")
        itime2 = ((mjd - itime) * 86400000).astype("int32")
        return itime, itime2

    @staticmethod
    def _times_char(t: pd.DatetimeIndex) -> np.ndarray:
        s = t.strftime("%Y-%m-%dT%H:%M:%S.000000")
        return np.asarray([list(i.ljust(26)) for i in s], dtype="S1")

    # ------------------------------------------------------------------
    # BaseGenerator interface
    # ------------------------------------------------------------------
    def load(self) -> None:
        # Accept .dat or .nc
        if self.source.suffix.lower() == ".dat":
            from .grid_reader import GridASCII

            grid = GridASCII(self.source)
            self.mesh_ds = grid.to_xarray()
        else:  # assume NetCDF grid
            import xarray as xr

            self.mesh_ds = xr.open_dataset(self.source)

        self.timeline = pd.date_range(
            self.start, self.end, freq=f"{self.dt}s", inclusive="both", tz="UTC"
        )

    def validate(self) -> None:
        missing = [v for v in ("x", "y", "nv") if v not in self.mesh.variables]
        if missing:
            raise ValueError(f"grid file missing variables: {missing}")

    def render(self) -> bytes:
        nt = self.timeline.size
        nele = len(self.mesh.dimensions["nele"])
        node = len(self.mesh.dimensions["node"])

        mjd = self._to_mjd(self.timeline)
        itime, itime2 = self._itime_pair(mjd)

        # --- write to temp file then return bytes --------------------
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        with nc.Dataset(tmp_path, "w", format="NETCDF4_CLASSIC") as ds_out:
            # ---------------------------------------------------------
            # dimensions  (order is important for FVCOM reader)
            # ---------------------------------------------------------
            ds_out.createDimension("time", None)  # unlimited
            ds_out.createDimension("DateStrLen", 26)
            ds_out.createDimension("node", node)
            ds_out.createDimension("nele", nele)
            ds_out.createDimension("three", 3)

            # ---------------------------------------------------------
            # static mesh variables  –  copied verbatim
            # ---------------------------------------------------------
            for name in ("x", "y", "lon", "lat"):
                v_in = self.mesh.variables[name]
                v_out = ds_out.createVariable(name, v_in.dtype, ("node",))
                v_out[:] = v_in[:]
                v_out.setncatts({k: v_in.getncattr(k) for k in v_in.ncattrs()})

            for name in ("xc", "yc", "lonc", "latc"):
                v_in = self.mesh.variables[name]
                v_out = ds_out.createVariable(name, v_in.dtype, ("nele",))
                v_out[:] = v_in[:]
                v_out.setncatts({k: v_in.getncattr(k) for k in v_in.ncattrs()})

            v_nv_in = self.mesh.variables["nv"]
            v_nv_out = ds_out.createVariable("nv", v_nv_in.dtype, ("three", "nele"))
            v_nv_out[:, :] = v_nv_in[:, :]
            v_nv_out.setncatts({k: v_nv_in.getncattr(k) for k in v_nv_in.ncattrs()})

            # ---------------------------------------------------------
            # time variables
            # ---------------------------------------------------------
            v_time = ds_out.createVariable("time", "f4", ("time",))
            v_time[:] = mjd
            v_time.long_name = "time"
            v_time.units = "days since 1858-11-17 00:00:00"
            v_time.format = "modified julian day (MJD)"
            v_time.time_zone = "UTC"

            v_itime = ds_out.createVariable("Itime", "i4", ("time",))
            v_itime[:] = itime
            v_itime.units = "days since 1858-11-17 00:00:00"
            v_itime.format = "modified julian day (MJD)"
            v_itime.time_zone = "UTC"

            v_itime2 = ds_out.createVariable("Itime2", "i4", ("time",))
            v_itime2[:] = itime2
            v_itime2.units = "msec since 00:00:00"
            v_itime2.time_zone = "UTC"

            v_times = ds_out.createVariable("Times", "S1", ("time", "DateStrLen"))
            v_times[:, :] = self._times_char(self.timeline)

            # ---------------------------------------------------------
            # dynamic fields (constant in both space & time)
            # element-centred
            # ---------------------------------------------------------
            def _make(name: str, long_name: str, units: str) -> None:
                v = ds_out.createVariable(
                    name, "f4", ("time", "nele"), fill_value=np.nan
                )
                v[:, :] = self.consts[name]  # broadcast
                v.long_name = long_name
                v.units = units

            _make("uwind_speed", "eastward wind speed", "m s-1")
            _make("vwind_speed", "northward wind speed", "m s-1")
            _make("air_temperature", "surface air temperature", "Celsius")
            _make("relative_humidity", "relative humidity", "%")
            _make("surface_pressure", "mean sea level pressure", "hPa")
            _make("surface_short_wave_flux", "downward short-wave radiation", "W m-2")
            _make("surface_long_wave_flux", "downward long-wave radiation", "W m-2")
            _make("precip_rate", "precipitation rate", "kg m-2 s-1")

            # ---------------------------------------------------------
            # global attrs
            # ---------------------------------------------------------
            ds_out.type = "FVCOM METEOROLOGY FORCING FILE"
            ds_out.title = "Constant meteorological forcing (prototype)"
            ds_out.history = "generated by xfvcom"
            ds_out.info = ", ".join(f"{k}={v}" for k, v in self.consts.items())

        data = tmp_path.read_bytes()
        tmp_path.unlink()  # remove temp file
        return data

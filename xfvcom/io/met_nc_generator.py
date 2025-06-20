# -*- coding: utf-8 -*-
"""Generate a constant-value meteorological forcing NetCDF-4 for FVCOM."""

from __future__ import annotations

import datetime as dt
import tempfile
from datetime import timezone
from io import BytesIO
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import netCDF4 as nc
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..grid.grid_obj import FvcomGrid
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
        cloud=0.0,  # fraction
    )

    def __init__(
        self,
        grid_nc: Path | str | PathLike[str],
        start: str,
        end: str,
        dt_seconds: int = 3600,
        *,
        utm_zone: int | None = None,
        northern: bool = True,
        start_tz: str = "UTC",
        **consts: float,
    ) -> None:
        t0 = pd.Timestamp(start)
        t1 = pd.Timestamp(end)
        if t0.tzinfo is None:
            t0 = t0.tz_localize(start_tz)
        if t1.tzinfo is None:
            t1 = t1.tz_localize(start_tz)
        self.start = t0.tz_convert("UTC")
        self.end = t1.tz_convert("UTC")
        source = Path(grid_nc)

        # 2) 親クラスへは Path 型で渡す
        super().__init__(source)

        # 3) 以降、インスタンス属性として保持
        self.source: Path = source
        self.dt = dt_seconds
        self.utm_zone = utm_zone
        self.northern = northern
        self.consts = {**self._DEFAULTS, **consts}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_mjd(t: pd.DatetimeIndex) -> NDArray[np.float32]:
        origin = pd.Timestamp("1858-11-17T00:00:00Z")
        return ((t - origin) / pd.Timedelta("1D")).to_numpy("f4")

    @staticmethod
    def _itime_pair(mjd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        itime: NDArray[np.int32] = mjd.astype("int32")
        itime2: NDArray[np.int32] = ((mjd - itime) * 86400000).astype("int32")
        return itime, itime2

    @staticmethod
    def _times_char(t: pd.DatetimeIndex) -> np.ndarray:
        s = t.strftime("%Y-%m-%dT%H:%M:%S.000000")
        return np.asarray([list(i.ljust(26)) for i in s], dtype="S1")

    # --- crude xy→lon/lat fallback (identity) ------------------------
    @staticmethod
    def _xy_to_lonlat(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        If the grid file lacks geographic coordinates, fall back to dummy
        values that satisfy variable/attribute requirements.  Replace with a
        proper projection if lon/lat are essential for the simulation.
        """
        return x.copy(), y.copy()

    # ------------------------------------------------------------------
    # BaseGenerator interface
    # ------------------------------------------------------------------
    def load(self) -> None:
        # Accept .dat or .nc
        if self.source.suffix.lower() == ".dat":
            grid = FvcomGrid.from_dat(
                self.source,
                utm_zone=self.utm_zone,
                northern=self.northern,
            )
            ds = grid.to_xarray()
            # ------------------------------------------------------
            # lonc / latc が無ければノード平均で補完
            # ------------------------------------------------------
            if "lonc" not in ds:
                lonc = ds["lon"].values[ds["nv"].values].mean(axis=-1)
                ds["lonc"] = ("nele", lonc.astype("f8"))
            if "latc" not in ds:
                latc = ds["lat"].values[ds["nv"].values].mean(axis=-1)
                ds["latc"] = ("nele", latc.astype("f8"))
            self.mesh_ds = ds
        else:  # assume NetCDF grid
            import xarray as xr

            self.mesh_ds = xr.open_dataset(self.source)

        self.timeline = pd.date_range(
            self.start, self.end, freq=f"{self.dt}s", inclusive="both", tz="UTC"
        )

    def validate(self) -> None:
        for req in ("x", "y", "nv"):
            if req not in self.mesh_ds:
                raise ValueError(f"grid file missing variable: {req}")

    def render(self) -> bytes:
        nt = self.timeline.size
        nele: int = int(self.mesh_ds.sizes["nele"])
        node: int = int(self.mesh_ds.sizes["node"])

        mjd: NDArray[np.float32] = self._to_mjd(self.timeline)
        itime: NDArray[np.int32]
        itime2: NDArray[np.int32]
        itime, itime2 = self._itime_pair(mjd)

        # --- write to temp file then return bytes --------------------
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        with nc.Dataset(tmp_path, "w", format="NETCDF4_CLASSIC") as ds_out:
            # ---------------------------------------------------------
            # dimensions  (order is important for FVCOM reader)
            # ---------------------------------------------------------
            # order: nele, node, three, time, DateStrLen
            ds_out.createDimension("nele", nele)
            ds_out.createDimension("node", node)
            ds_out.createDimension("three", 3)
            ds_out.createDimension("time", None)
            ds_out.createDimension("DateStrLen", 26)

            # ---------------------------------------------------------
            # static mesh variables (copied verbatim, float64/int32)
            # ---------------------------------------------------------
            for name in ("x", "y", "lon", "lat"):
                if name not in self.mesh_ds:
                    continue
                v_in = self.mesh_ds[name]
                v_out = ds_out.createVariable(
                    name, v_in.dtype, ("node",), fill_value=False
                )
                v_out[:] = v_in.values
                v_out.setncatts(v_in.attrs)
                if "long_name" not in v_out.ncattrs():  # fallback
                    v_out.long_name = (
                        "nodal x-coordinate"
                        if name == "x"
                        else (
                            "nodal y-coordinate"
                            if name == "y"
                            else (
                                "nodal longitude" if name == "lon" else "nodal latitude"
                            )
                        )
                    )
                if "units" not in v_out.ncattrs():
                    v_out.units = "meters" if name in ("x", "y") else "degrees"

            for name in ("xc", "yc", "lonc", "latc"):
                if name not in self.mesh_ds:
                    continue
                v_in = self.mesh_ds[name]
                v_out = ds_out.createVariable(
                    name, v_in.dtype, ("nele",), fill_value=False
                )
                v_out[:] = v_in.values
                v_out.setncatts(v_in.attrs)
                if "long_name" not in v_out.ncattrs():
                    v_out.long_name = (
                        "zonal x-coordinate"
                        if name == "xc"
                        else (
                            "zonal y-coordinate"
                            if name == "yc"
                            else (
                                "zonal longitude"
                                if name == "lonc"
                                else "zonal latitude"
                            )
                        )
                    )
                if "units" not in v_out.ncattrs():
                    v_out.units = "meters" if name in ("xc", "yc") else "degrees"

            nv_in = self.mesh_ds["nv"].values
            v_nv_out = ds_out.createVariable(
                "nv", "i4", ("three", "nele"), fill_value=False
            )
            if nv_in.shape == (3, nele):
                v_nv_out[:, :] = nv_in
            elif nv_in.shape == (nele, 3):
                v_nv_out[:, :] = nv_in.T
            else:
                raise ValueError(f"Unexpected nv shape: {nv_in.shape}")
            v_nv_out.long_name = "nodes surrounding element"

            # ---------------------------------------------------------
            # time variables
            # ---------------------------------------------------------
            v_time = ds_out.createVariable("time", "f4", ("time",))
            v_time[:] = mjd
            v_time.long_name = "time"
            v_time.units = "days since 1858-11-17 00:00:00"
            v_time.format = "modified julian day (MJD)"
            v_time.time_zone = "UTC"

            # ---------------------------------------------------------
            # dynamic fields – constant, dims & attrs match reference
            # ---------------------------------------------------------
            def _make(
                name: str,
                key: str,
                dims: tuple[str, ...],
                long_name: str,
                units: str,
                std_name: str | None = None,
                desc: str | None = None,
            ) -> None:
                v = ds_out.createVariable(name, "f4", dims, fill_value=False)
                v[:] = self.consts[key]
                v.long_name = long_name
                if std_name:
                    v.standard_name = std_name
                if desc:
                    v.description = desc
                v.units = units
                v.grid = "fvcom_grid"
                if "node" in dims:
                    v.coordinates = "FVCOM cartesian coordinates"
                v.type = "data"

            _make(
                "uwind_speed",
                "uwind",
                ("time", "nele"),
                "Eastward Wind Speed",
                "m/s",
                "Wind Speed",
            )
            _make(
                "vwind_speed",
                "vwind",
                ("time", "nele"),
                "Northward Wind Speed",
                "m/s",
                "Wind Speed",
            )
            _make(
                "air_temperature",
                "air_temp",
                ("time", "node"),
                "Surface air temperature",
                "Celsius Degree",
            )
            _make("cloud_cover", "cloud", ("time", "node"), "Cloud cover", "-")
            _make(
                "short_wave",
                "swrad",
                ("time", "node"),
                "Downward solar shortwave radiation flux",
                "Watts meter-2",
            )
            _make(
                "long_wave",
                "lwrad",
                ("time", "node"),
                "Downward solar longwave radiation flux",
                "Watts meter-2",
            )
            _make(
                "relative_humidity",
                "rh",
                ("time", "node"),
                "surface air relative humidity",
                "percentage",
            )
            _make(
                "air_pressure", "prmsl", ("time", "node"), "Surface air pressure", "hPa"
            )
            _make(
                "Precipitation",
                "precip",
                ("time", "node"),
                "Precipitation",
                "m s-1",
                desc="Precipitation, ocean lose water is negative",
            )

            # ---------------------------------------------------------
            # global attrs
            # ---------------------------------------------------------
            ds_out.type = "FVCOM Forcing File"
            ds_out.title = "FVCOM Forcing File"
            ds_out.institution = "Sasaki Lab, The University of Tokyo"
            ds_out.source = "FVCOM grid (unstructured) surface forcing"
            ds_out.history = (
                "File created with write_FVCOM_forcing from the MATLAB " "fvcom-toolbox"
            )
            ds_out.references = (
                "http://fvcom.smast.umassd.edu, http://codfish.smast.umassd.edu"
            )
            ds_out.Conventions = "CF-1.0"
            ds_out.infos = "GWO atmospheric forcing data"
            ds_out.CoordinateSystem = "cartesian"
            ds_out.CoordinateProjection = "init=WGS84"

        data = tmp_path.read_bytes()
        tmp_path.unlink()  # remove temp file
        return data

    def write(self, dest: Path | None = None) -> Path:  # noqa: D401
        """
        NetCDF をファイルに書き出す。

        Parameters
        ----------
        dest : pathlib.Path, optional
            出力先パス。未指定なら

            1. ソースが ``*_grd.dat`` の場合 → ``*_wnd.nc``
            2. それ以外の拡張子         → ``<stem>.nc``

        Returns
        -------
        pathlib.Path
            書き出した NetCDF ファイルのパス。
        """
        if dest is None:
            stem = self.source.stem
            if stem.endswith("_grd"):
                stem = stem[:-4] + "_wnd"  # *_grd → *_wnd
            dest = self.source.with_name(f"{stem}.nc")
        return super().write(dest)

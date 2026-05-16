# -*- coding: utf-8 -*-
"""Generate meteorological forcing NetCDF-4 for FVCOM.

Supports multiple data sources:
- Constant values
- CSV/TSV time series
- GWO-AMD meteorological data (with comprehensive gap filling)
"""

from __future__ import annotations

import tempfile
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Mapping

import netCDF4 as nc
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..grid.grid_obj import FvcomGrid
from .base_generator import BaseGenerator
from .sources.base import BaseForcingSource
from .sources.timeseries import TimeSeriesSource

if TYPE_CHECKING:
    from .gwo_correlations import StationCorrelations
    from .gwo_gap_filler import GapFillResult, MissingValueInfo


class _ScalarConstantSource(BaseForcingSource):
    """Return the same constant value for exactly one variable."""

    def __init__(self, var: str, value: float) -> None:
        self._var = var
        self._val = float(value)

    def get_series(self, var_name: str, times: pd.DatetimeIndex) -> NDArray[np.float64]:  # type: ignore[override]
        if var_name != self._var:
            raise KeyError(
                f"Unsupported variable {var_name!r} (expected {self._var!r})"
            )
        return np.full(times.size, self._val, dtype=float)


class _GWOSourceAdapter(BaseForcingSource):
    """Adapter to use GWOForcingSource with MetNetCDFGenerator."""

    def __init__(self, gwo_source: object, var: str) -> None:
        # Type annotation uses object to avoid import cycle; actual type is GWOForcingSource
        self._gwo = gwo_source
        self._var = var

    def get_series(self, var_name: str, times: pd.DatetimeIndex) -> NDArray[np.float64]:  # type: ignore[override]
        if var_name != self._var:
            raise KeyError(
                f"Unsupported variable {var_name!r} (expected {self._var!r})"
            )
        return self._gwo.get_series(var_name, times)  # type: ignore[attr-defined]


def _parse_ts_spec(tokens: Iterable[str], variables: Iterable[str]) -> dict[str, str]:
    """Return mapping ``{var: path}`` from CLI --ts tokens."""

    mapping: dict[str, str] = {}
    for tok in tokens:
        path, _, vars_part = tok.partition(":")
        path = path.strip()
        vars_list = (
            [v.strip() for v in vars_part.split(",") if v.strip()] if vars_part else []
        )
        if vars_list:
            for var in vars_list:
                mapping.setdefault(var, path)
        else:
            for var in variables:
                mapping.setdefault(var, path)
    return mapping


def _choose_source(
    var: str, ts_map: Mapping[str, str], const_val: float, *, data_tz: str
) -> BaseForcingSource:
    """Return a TimeSeriesSource or constant fallback for *var*."""

    if var in ts_map:
        return TimeSeriesSource(Path(ts_map[var]), input_tz=data_tz)
    return _ScalarConstantSource(var, const_val)


class MetNetCDFGenerator(BaseGenerator):
    """
    Build a meteorological forcing file for FVCOM.

    Supports multiple data sources:
    - Constant values (default)
    - CSV/TSV time series (via --ts)
    - GWO-AMD meteorological data (via --gwo-dir)

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
    gwo_dir : Path, optional
        Base directory for GWO hourly data. If provided, enables GWO mode.
    station_map : dict[str, str], optional
        Mapping of variable names to station names for GWO data.
        Use "*" as a key for default station.
        Example: {"slht": "Tokyo", "kous": "Tokyo", "*": "Chiba"}
    wind_factor : float, optional
        Wind speed multiplier for GWO data (default: 1.8)
    max_gap_hours : int, optional
        Maximum hours to interpolate for missing GWO data (default: 6)
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
        ts_specs: list[str] | None = None,
        data_tz: str = "Asia/Tokyo",
        gwo_dir: Path | str | None = None,
        station_map: dict[str, str] | None = None,
        wind_factor: float = 1.8,
        max_gap_hours: int = 6,
        fill_gaps: bool = False,
        fallback_stations: dict[str, list[str]] | None = None,
        correlations: "StationCorrelations | None" = None,
        solar_model: str = "empirical",
        extend_boundary: bool = True,
        mpos_dir: Path | str | None = None,
        mpos_stations: tuple[str, ...] | None = None,
        mpos_idw_power: float = 2.0,
        mpos_convert_to_utc: bool = True,
        metforce_file: Path | str | None = None,
        metforce_fallback: str | None = "nearest",
        metforce_pad_trailing_bookend: bool = False,
        **consts: float,
    ) -> None:
        t0 = pd.Timestamp(start)
        t1 = pd.Timestamp(end)

        # Store whether we're in GWO mode (affects timezone handling)
        self._gwo_mode = gwo_dir is not None
        self._gwo_dir = Path(gwo_dir) if gwo_dir else None
        self._station_map = station_map
        self._wind_factor = wind_factor
        self._max_gap_hours = max_gap_hours
        self._fill_gaps = fill_gaps
        self._fallback_stations = fallback_stations
        self._correlations = correlations
        self._solar_model = solar_model
        self._extend_boundary = extend_boundary

        # MPOS spatial wind mode (mutually exclusive with --gwo-dir).
        self._mpos_mode = mpos_dir is not None
        self._mpos_dir = Path(mpos_dir) if mpos_dir else None
        self._mpos_stations = mpos_stations
        self._mpos_idw_power = mpos_idw_power
        self._mpos_convert_to_utc = mpos_convert_to_utc
        if self._mpos_mode and self._gwo_mode:
            raise ValueError(
                "--gwo-dir and --mpos-dir are mutually exclusive; choose one"
            )

        # metforce spatial-grid mode (mutually exclusive with the above).
        self._metforce_mode = metforce_file is not None
        self._metforce_file = Path(metforce_file) if metforce_file else None
        self._metforce_fallback = metforce_fallback
        self._metforce_pad_trailing_bookend = metforce_pad_trailing_bookend
        if self._metforce_mode and (self._gwo_mode or self._mpos_mode):
            raise ValueError(
                "--metforce-file is mutually exclusive with --gwo-dir / "
                "--mpos-dir; choose one"
            )

        if self._gwo_mode:
            # GWO mode: use naive timestamps (JST values labeled as UTC)
            # This matches the FVCOM convention used in reference files
            if t0.tzinfo is not None:
                t0 = t0.tz_localize(None)
            if t1.tzinfo is not None:
                t1 = t1.tz_localize(None)
            self.start = t0
            self.end = t1
        else:
            # Legacy mode: convert to UTC
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
        self._data_tz = data_tz

        # Initialize sources
        self._sources: dict[str, BaseForcingSource] = {}
        self._gwo_source: object | None = None

        if gwo_dir is not None:
            # GWO mode: use GWOForcingSource for all variables
            from .gwo_reader import GWOForcingSource

            if station_map is None:
                station_map = {"*": "Chiba"}  # Default to Chiba for all variables
            self._station_map = station_map

            gwo_source = GWOForcingSource(
                gwo_dir,
                station_map,
                wind_factor=wind_factor,
                max_gap_hours=max_gap_hours,
                input_tz=data_tz,
            )
            self._gwo_source = gwo_source

            for var in self._DEFAULTS:
                self._sources[var] = _GWOSourceAdapter(gwo_source, var)
        elif mpos_dir is not None:
            # MPOS mode: spatial wind / air-temp from MPOS observations.
            # Non-MPOS variables (rh, prmsl, swrad, lwrad, precip, cloud)
            # fall back to constants or --ts overrides, as in legacy mode.
            self._ts_map = _parse_ts_spec(ts_specs or [], self._DEFAULTS)
            for var in self._DEFAULTS:
                if var in ("uwind", "vwind", "air_temp"):
                    # Placeholder; replaced with MposWindSource in load().
                    self._sources[var] = _ScalarConstantSource(var, 0.0)
                else:
                    self._sources[var] = _choose_source(
                        var,
                        self._ts_map,
                        self.consts[var],
                        data_tz=self._data_tz,
                    )
        elif metforce_file is not None:
            # metforce mode: 8 atmospheric variables (uwind, vwind, air_temp,
            # rh, prmsl, swrad, lwrad, precip) come from the gridded OI
            # analysis bilinearly interpolated onto the FVCOM mesh. Cloud
            # cover is not produced by metforce and falls back to the
            # scalar default (or --ts override).
            self._ts_map = _parse_ts_spec(ts_specs or [], self._DEFAULTS)
            _metforce_vars = (
                "uwind",
                "vwind",
                "air_temp",
                "rh",
                "prmsl",
                "swrad",
                "lwrad",
                "precip",
            )
            for var in self._DEFAULTS:
                if var in _metforce_vars:
                    # Placeholder; replaced with MetforceGriddedSource in
                    # load() after the mesh has been parsed.
                    self._sources[var] = _ScalarConstantSource(var, 0.0)
                else:
                    self._sources[var] = _choose_source(
                        var,
                        self._ts_map,
                        self.consts[var],
                        data_tz=self._data_tz,
                    )
        else:
            # Legacy mode: use time series or constants
            self._ts_map = _parse_ts_spec(ts_specs or [], self._DEFAULTS)
            for var in self._DEFAULTS:
                self._sources[var] = _choose_source(
                    var,
                    self._ts_map,
                    self.consts[var],
                    data_tz=self._data_tz,
                )

        # Build timeline
        if self._gwo_mode:
            # GWO mode: naive timeline (JST values labeled as UTC)
            self.timeline = pd.date_range(
                self.start,
                self.end,
                freq=f"{self.dt}s",
                inclusive="both",
            )
        else:
            # Legacy mode: UTC timeline
            self.timeline = pd.date_range(
                self.start,
                self.end,
                freq=f"{self.dt}s",
                inclusive="both",
                tz="UTC",
            )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_mjd(t: pd.DatetimeIndex) -> NDArray[np.float64]:
        # Use naive origin for naive timestamps, UTC origin for tz-aware.
        # Returned as float64 so the integer hourly cadence is preserved
        # exactly; storing MJD in float32 truncates 3600 s spacings into
        # the 3375 s / 3712.5 s drift seen in pre-2026-05-16 forcing files.
        if t.tz is None:
            origin = pd.Timestamp("1858-11-17T00:00:00")
        else:
            origin = pd.Timestamp("1858-11-17T00:00:00Z")
        delta = t - origin
        seconds = np.asarray(delta.total_seconds(), dtype=np.float64)
        return seconds / 86400.0

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

        # In MPOS mode the wind / air-temperature sources need the mesh
        # coordinates and are therefore instantiated only after the grid
        # has been parsed.
        if self._mpos_mode:
            self._init_mpos_sources()

        # metforce mode follows the same pattern.
        if self._metforce_mode:
            self._init_metforce_sources()

    def _init_metforce_sources(self) -> None:
        """Instantiate :class:`MetforceGriddedSource` and patch ``self._sources``."""
        from .sources.metforce import MetforceGriddedSource

        # Pull mesh coordinates as float64 for xarray.interp.
        node_lon = np.asarray(self.mesh_ds["lon"].values, dtype=np.float64)
        node_lat = np.asarray(self.mesh_ds["lat"].values, dtype=np.float64)
        elem_lon = np.asarray(self.mesh_ds["lonc"].values, dtype=np.float64)
        elem_lat = np.asarray(self.mesh_ds["latc"].values, dtype=np.float64)

        assert self._metforce_file is not None  # narrowed by _metforce_mode
        source = MetforceGriddedSource(
            self._metforce_file,
            target_node_lon=node_lon,
            target_node_lat=node_lat,
            target_elem_lon=elem_lon,
            target_elem_lat=elem_lat,
            fallback_method=self._metforce_fallback,
            pad_trailing_bookend=self._metforce_pad_trailing_bookend,
        )
        self._metforce_source = source
        for var in source.variables:
            self._sources[var] = source

    def _init_mpos_sources(self) -> None:
        """Instantiate :class:`MposWindSource` and patch ``self._sources``."""
        from .sources.mpos_wind import MposWindSource

        # Pull mesh coordinates as float64 for IDW.
        node_lon = np.asarray(self.mesh_ds["lon"].values, dtype=np.float64)
        node_lat = np.asarray(self.mesh_ds["lat"].values, dtype=np.float64)
        elem_lon = np.asarray(self.mesh_ds["lonc"].values, dtype=np.float64)
        elem_lat = np.asarray(self.mesh_ds["latc"].values, dtype=np.float64)

        # Pad the load window by one day on each side so that JST<->UTC
        # conversion does not strip values from the edges.
        pad = pd.Timedelta(days=1)
        load_start = (self.timeline[0] - pad).tz_localize(None).isoformat()
        load_end = (self.timeline[-1] + pad).tz_localize(None).isoformat()

        source = MposWindSource(
            start=load_start,
            end=load_end,
            target_node_lon=node_lon,
            target_node_lat=node_lat,
            target_elem_lon=elem_lon,
            target_elem_lat=elem_lat,
            mpos_dir=self._mpos_dir,
            stations=self._mpos_stations,
            idw_power=self._mpos_idw_power,
            convert_to_utc=self._mpos_convert_to_utc,
        )
        self._mpos_source = source
        # Wire all three MPOS-served variables through this single source.
        for var in source.variables:
            self._sources[var] = source

    def validate(self) -> None:
        for req in ("x", "y", "nv"):
            if req not in self.mesh_ds:
                raise ValueError(f"grid file missing variable: {req}")

    def render(self) -> bytes:
        if not hasattr(self, "mesh_ds"):
            self.load()
        nt = self.timeline.size
        nele: int = int(self.mesh_ds.sizes["nele"])
        node: int = int(self.mesh_ds.sizes["node"])

        mjd: NDArray[np.float64] = self._to_mjd(self.timeline)
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
            # FvcomGrid uses 0-based indexing internally, FVCOM expects 1-based
            # Add 1 if the minimum value is 0 (indicating 0-based indexing)
            nv_offset = 1 if nv_in.min() == 0 else 0
            if nv_in.shape == (3, nele):
                v_nv_out[:, :] = nv_in + nv_offset
            elif nv_in.shape == (nele, 3):
                v_nv_out[:, :] = nv_in.T + nv_offset
            else:
                raise ValueError(f"Unexpected nv shape: {nv_in.shape}")
            v_nv_out.long_name = "nodes surrounding element"

            # ---------------------------------------------------------
            # time variables
            # ---------------------------------------------------------
            v_time = ds_out.createVariable("time", "f8", ("time",))
            v_time[:] = mjd
            v_time.long_name = "time"
            v_time.units = "days since 1858-11-17 00:00:00"
            v_time.format = "modified julian day (MJD)"
            v_time.time_zone = "UTC"

            # ---------------------------------------------------------
            # dynamic fields – from sources or constant fallback
            # ---------------------------------------------------------
            srcs: Mapping[str, BaseForcingSource] = self._sources

            def _write(
                name: str,
                key: str,
                dims: tuple[str, ...],
                long_name: str,
                units: str,
                std_name: str | None = None,
                desc: str | None = None,
            ) -> None:
                source = srcs[key]
                spatial_dim: str | None = None
                if "nele" in dims:
                    spatial_dim = "elem"
                elif "node" in dims:
                    spatial_dim = "node"
                use_spatial = (
                    spatial_dim is not None
                    and getattr(source, "is_spatial", False)
                    and hasattr(source, "get_spatial_series")
                )

                try:
                    if use_spatial:
                        data = source.get_spatial_series(  # type: ignore[attr-defined]
                            key, self.timeline, on=spatial_dim
                        )
                    else:
                        data = source.get_series(key, self.timeline)
                except ValueError as exc:
                    if str(exc).startswith("Column '"):
                        data = np.full(nt, self.consts[key], dtype=float)
                        use_spatial = False
                    else:
                        raise

                v = ds_out.createVariable(name, "f4", dims, fill_value=False)
                data_f4: NDArray[np.float32] = np.asarray(data, dtype="f4")
                if dims == ("time",):
                    v[:] = data_f4
                elif use_spatial:
                    # Already shaped (n_time, n_target).
                    v[:, :] = data_f4
                else:
                    # 1-D series broadcast over the spatial dimension.
                    v[:, :] = data_f4[:, None]
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

            _write(
                "uwind_speed",
                "uwind",
                ("time", "nele"),
                "Eastward Wind Speed",
                "m/s",
                "Wind Speed",
            )
            _write(
                "vwind_speed",
                "vwind",
                ("time", "nele"),
                "Northward Wind Speed",
                "m/s",
                "Wind Speed",
            )
            _write(
                "air_temperature",
                "air_temp",
                ("time", "node"),
                "Surface air temperature",
                "Celsius Degree",
            )
            _write("cloud_cover", "cloud", ("time", "node"), "Cloud cover", "-")
            _write(
                "short_wave",
                "swrad",
                ("time", "node"),
                "Downward solar shortwave radiation flux",
                "Watts meter-2",
            )
            _write(
                "long_wave",
                "lwrad",
                ("time", "node"),
                "Downward solar longwave radiation flux",
                "Watts meter-2",
            )
            _write(
                "relative_humidity",
                "rh",
                ("time", "node"),
                "surface air relative humidity",
                "percentage",
            )
            _write(
                "air_pressure", "prmsl", ("time", "node"), "Surface air pressure", "hPa"
            )
            _write(
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
            if self._metforce_mode and self._metforce_file is not None:
                ds_out.infos = (
                    f"metforce OI atmospheric forcing "
                    f"(source: {self._metforce_file.name})"
                )
            elif self._mpos_mode:
                ds_out.infos = "MPOS Tokyo Bay station OI atmospheric forcing"
            else:
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

    # ------------------------------------------------------------------
    # Gap filling methods (GWO mode only)
    # ------------------------------------------------------------------
    def run_gap_filling(self) -> list["GapFillResult"] | None:
        """
        Run comprehensive gap filling on GWO data.

        This method applies the 4-step gap filling process per variable,
        respecting the station_map to load data from the correct station
        for each variable:
        1. Temporal interpolation
        2. Boundary interpolation
        3. Fallback station interpolation
        4. Solar radiation estimation

        Returns
        -------
        list[GapFillResult] | None
            List of gap fill results per variable, or None if not in GWO mode.
        """
        if not self._gwo_mode or self._gwo_dir is None:
            return None

        from .gwo_gap_filler import FILLABLE_VARIABLES, GapFiller
        from .gwo_reader import GWOForcingSource, GWOReader

        reader = GWOReader(self._gwo_dir)

        # Get the default station
        default_station = (
            self._station_map.get("*", "Chiba") if self._station_map else "Chiba"
        )

        start_dt = self.start.to_pydatetime()
        end_dt = self.end.to_pydatetime()

        # Build a mapping of station -> list of variables from that station
        station_to_vars: dict[str, list[str]] = {}
        for var in FILLABLE_VARIABLES:
            if self._station_map and var in self._station_map:
                station = self._station_map[var]
            else:
                station = default_station
            if station not in station_to_vars:
                station_to_vars[station] = []
            station_to_vars[station].append(var)

        # Create gap filler
        filler = GapFiller(
            gwo_reader=reader,
            fallback_stations=self._fallback_stations,
            correlations=self._correlations,
            max_gap_hours=self._max_gap_hours,
            solar_model=self._solar_model,  # type: ignore[arg-type]
            extend_boundary=self._extend_boundary,
        )

        all_results: list["GapFillResult"] = []
        filled_data_by_station: dict[str, pd.DataFrame] = {}

        # Process each station
        for station, vars_from_station in station_to_vars.items():
            # Load raw data for this station
            df = reader.load_range(station, start_dt, end_dt)

            # Run gap filling for each variable from this station
            for var in vars_from_station:
                if var not in df.columns:
                    continue

                result = filler._fill_variable(df, var, station, start_dt, end_dt)
                all_results.append(result)

            # Store the filled data for this station
            filled_data_by_station[station] = df

        # Pass the filled data to the GWO source
        if self._gwo_source is not None:
            gwo_source = self._gwo_source
            if isinstance(gwo_source, GWOForcingSource):
                for station, filled_df in filled_data_by_station.items():
                    gwo_source.set_prefilled_data(station, filled_df)

        return all_results

    def analyze_missing_values(self) -> list["MissingValueInfo"] | None:
        """
        Analyze missing values in GWO data without filling.

        This method respects the station_map to analyze data from the correct
        station for each variable.

        Returns
        -------
        list[MissingValueInfo] | None
            List of missing value info per variable, or None if not in GWO mode.
        """
        if not self._gwo_mode or self._gwo_dir is None:
            return None

        from .gwo_gap_filler import FILLABLE_VARIABLES, GapFiller, MissingValueInfo
        from .gwo_reader import GWOReader

        reader = GWOReader(self._gwo_dir)

        # Get the default station
        default_station = (
            self._station_map.get("*", "Chiba") if self._station_map else "Chiba"
        )

        start_dt = self.start.to_pydatetime()
        end_dt = self.end.to_pydatetime()

        # Build a mapping of station -> list of variables from that station
        station_to_vars: dict[str, list[str]] = {}
        for var in FILLABLE_VARIABLES:
            if self._station_map and var in self._station_map:
                station = self._station_map[var]
            else:
                station = default_station
            if station not in station_to_vars:
                station_to_vars[station] = []
            station_to_vars[station].append(var)

        # Create gap filler (just for analysis)
        filler = GapFiller(
            gwo_reader=reader,
            max_gap_hours=self._max_gap_hours,
        )

        all_results: list["MissingValueInfo"] = []

        # Analyze each station
        for station, vars_from_station in station_to_vars.items():
            # Load raw data for this station
            df = reader.load_range(station, start_dt, end_dt)

            # Analyze missing values for each variable from this station
            for var in vars_from_station:
                if var not in df.columns:
                    continue

                rmk_col = f"{var}RMK"
                if rmk_col not in df.columns:
                    continue

                # Count missing (RMK 0, 1, 2 except for solar where 2 is valid)
                if var in ("slht", "lght"):
                    missing_rmk = {0, 1}
                else:
                    missing_rmk = {0, 1, 2}

                is_missing = df[rmk_col].isin(missing_rmk)
                missing_count = int(is_missing.sum().item())  # type: ignore[union-attr]
                total_count = len(df)

                # Find gaps
                gap_details = filler._find_gaps(is_missing)

                max_gap = max((g.duration_hours for g in gap_details), default=0)
                first_gap = gap_details[0].start if gap_details else None

                all_results.append(
                    MissingValueInfo(
                        variable=var,
                        station=station,
                        missing=missing_count,
                        total=total_count,
                        rate=(
                            100.0 * missing_count / total_count
                            if total_count > 0
                            else 0
                        ),
                        max_gap_hours=max_gap,
                        first_gap_start=first_gap,
                        gap_details=gap_details,
                    )
                )

        return all_results

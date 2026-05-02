"""MPOS (Marine/Port Observation System) data loader for Tokyo Bay.

Loads observation data from MPOS nc_structured/masked_filled NetCDF files,
performs vertical interpolation to match FVCOM model depths, and computes
daily means.

Requires the ``DATA_DIR`` environment variable to be set, pointing to the
shared data directory (e.g., ``/home/pj24001722/share/Data`` on GENKAI).
MPOS files are expected at ``$DATA_DIR/MPOS/nc_structured/masked_filled/``.

Typical usage::

    from xfvcom.io.mpos import MposLoader

    loader = MposLoader()  # uses $DATA_DIR/MPOS/nc_structured/masked_filled
    data = loader.load("temp", station="Kawasaki", target_depths=[1.4, 20.3])
    # data["surface"] → (daily_time, daily_values)
    # data["bottom"]  → (daily_time, daily_values)

    # Or load all stations at once:
    all_data = loader.load_all_stations("temp")
    # all_data["Kawasaki"]["surface"] → (daily_time, daily_values)

    # Override mpos_dir if DATA_DIR is not set:
    loader = MposLoader(mpos_dir="/custom/path/to/mpos_files")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def _default_mpos_dir() -> Path:
    """Resolve MPOS data directory from DATA_DIR environment variable."""
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        return Path(data_dir) / "MPOS" / "nc_structured" / "masked_filled"
    raise EnvironmentError(
        "DATA_DIR environment variable is not set.\n"
        "Set it to the shared data directory, e.g.:\n"
        "  export DATA_DIR=/home/pj24001722/share/Data\n"
        "Or pass mpos_dir explicitly:\n"
        "  MposLoader(mpos_dir='/path/to/mpos_files')"
    )


# =====================================================================
# Station configuration for Tokyo Bay MPOS network
# =====================================================================
@dataclass
class MposStationConfig:
    """Configuration for a single MPOS station."""

    name: str
    node_idx: int
    mpos_file: str
    surface_siglay: int
    bottom_siglay: int
    fvcom_surface_depth: float
    fvcom_bottom_depth: float


# Default station configurations for TB-FVCOM (TokyoBay18 grid)
DEFAULT_STATIONS: dict[str, MposStationConfig] = {
    "Kemigawa": MposStationConfig(
        name="Kemigawa",
        node_idx=545,
        mpos_file="masked_filled_01kemigawa.nc",
        surface_siglay=2,
        bottom_siglay=25,
        fvcom_surface_depth=0.855,
        fvcom_bottom_depth=8.726,
    ),
    "Kawasaki": MposStationConfig(
        name="Kawasaki",
        node_idx=1702,
        mpos_file="masked_filled_02kawasaki.nc",
        surface_siglay=1,
        bottom_siglay=21,
        fvcom_surface_depth=1.416,
        fvcom_bottom_depth=20.302,
    ),
    "Urayasu": MposStationConfig(
        name="Urayasu",
        node_idx=535,
        mpos_file="masked_filled_03urayasu.nc",
        surface_siglay=4,
        bottom_siglay=29,
        fvcom_surface_depth=0.981,
        fvcom_bottom_depth=6.430,
    ),
    "Chiba-1LH": MposStationConfig(
        name="Chiba-1LH",
        node_idx=1093,
        mpos_file="masked_filled_04chiba1gou.nc",
        surface_siglay=1,
        bottom_siglay=29,
        fvcom_surface_depth=0.998,
        fvcom_bottom_depth=19.630,
    ),
}

DEFAULT_STATION_ORDER = ["Kemigawa", "Kawasaki", "Urayasu", "Chiba-1LH"]

# Variable name mapping: FVCOM name → MPOS name
DEFAULT_VAR_MAP: dict[str, dict[str, Any]] = {
    "temp": {
        "fvcom_name": "temp",
        "mpos_name": "temp",
        "label": "Temperature",
        "unit": "deg C",
        "conversion": 1.0,
    },
    "salinity": {
        "fvcom_name": "salinity",
        "mpos_name": "salinity",
        "label": "Salinity",
        "unit": "PSU",
        "conversion": 1.0,
    },
    "O2": {
        "fvcom_name": "O2_o",
        "mpos_name": "DO",
        "label": "Dissolved Oxygen",
        "unit": "mg/L",
        "conversion": 0.032,  # mmol O2/m³ → g/m³ = mg/L
    },
}


@dataclass
class MposLoader:
    """Load and process MPOS observation data for model comparison.

    Parameters
    ----------
    mpos_dir : str or Path
        Directory containing MPOS NetCDF files.
    stations : dict, optional
        Station configurations. Defaults to Tokyo Bay 4-station network.
    station_order : list, optional
        Ordered list of station names for iteration.
    min_hourly : int
        Minimum number of hourly observations per day for daily mean (default: 6).
    xlim : tuple of str
        Date range for data extraction (start, end) as "YYYY-MM-DD" strings.
    """

    mpos_dir: Path = field(default_factory=_default_mpos_dir)
    stations: dict[str, MposStationConfig] = field(
        default_factory=lambda: dict(DEFAULT_STATIONS)
    )
    station_order: list[str] = field(
        default_factory=lambda: list(DEFAULT_STATION_ORDER)
    )
    min_hourly: int = 6
    xlim: tuple[str, str] = ("2020-01-16", "2020-12-31")

    def __post_init__(self) -> None:
        self.mpos_dir = Path(self.mpos_dir)

    def load(
        self,
        mpos_varname: str,
        station: str | None = None,
        target_depths: tuple[float, float] | None = None,
    ) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
        """Load MPOS observations with vertical interpolation and daily averaging.

        Parameters
        ----------
        mpos_varname : str
            MPOS variable name (e.g., "temp", "salinity", "DO").
        station : str, optional
            Load only this station. If None, loads all stations.
        target_depths : tuple of (surface_depth, bottom_depth), optional
            Override the default FVCOM depths for interpolation.

        Returns
        -------
        dict
            {station_name: {"surface": (daily_time, daily_vals),
                            "bottom": (daily_time, daily_vals)}}
        """
        t_start = np.datetime64(self.xlim[0])
        t_end = np.datetime64(self.xlim[1])
        results: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

        stations_to_load = (
            {station: self.stations[station]} if station else self.stations
        )

        for name, cfg in stations_to_load.items():
            mpos_file = self.mpos_dir / cfg.mpos_file
            if not mpos_file.exists():
                print(f"  WARNING: MPOS file not found: {mpos_file}")
                continue

            ds = xr.open_dataset(mpos_file)

            if mpos_varname not in ds:
                print(f"  {name}: '{mpos_varname}' not in MPOS file, skipping")
                ds.close()
                continue

            time = ds["time"].values
            time_mask = (time >= t_start) & (time <= t_end)
            time_sel = time[time_mask]

            if len(time_sel) == 0:
                ds.close()
                continue

            depth = ds["depth"].values[time_mask, :]
            var_data = ds[mpos_varname].values[time_mask, :]
            ds.close()

            if target_depths is not None:
                surf_depth, bot_depth = target_depths
            else:
                surf_depth = cfg.fvcom_surface_depth
                bot_depth = cfg.fvcom_bottom_depth

            var_surface, var_bottom = self._interpolate_vertical(
                time_sel, depth, var_data, surf_depth, bot_depth
            )

            daily_time, daily_surf, daily_bot = self._daily_mean(
                time_sel, var_surface, var_bottom
            )

            results[name] = {
                "surface": (daily_time, daily_surf),
                "bottom": (daily_time, daily_bot),
            }

        return results

    def load_all_stations(
        self,
        mpos_varname: str,
    ) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
        """Load all configured stations for a given variable.

        Convenience wrapper around :meth:`load` with ``station=None``.
        """
        return self.load(mpos_varname)

    def _interpolate_vertical(
        self,
        time_sel: np.ndarray,
        depth: np.ndarray,
        var_data: np.ndarray,
        surf_depth: float,
        bot_depth: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate observation profiles to target surface/bottom depths."""
        nt = len(time_sel)
        var_surface = np.full(nt, np.nan)
        var_bottom = np.full(nt, np.nan)

        for t in range(nt):
            d = depth[t, :]
            v = var_data[t, :]
            valid = np.isfinite(d) & np.isfinite(v)
            if np.sum(valid) < 2:
                continue
            d_valid = d[valid]
            v_valid = v[valid]
            sort_idx = np.argsort(d_valid)
            d_sorted = d_valid[sort_idx]
            v_sorted = v_valid[sort_idx]

            if surf_depth <= d_sorted[-1]:
                var_surface[t] = np.interp(surf_depth, d_sorted, v_sorted)
            if bot_depth <= d_sorted[-1]:
                var_bottom[t] = np.interp(bot_depth, d_sorted, v_sorted)
            elif bot_depth <= d_sorted[-1] * 1.1:
                # Bottom depth slightly beyond max observation — use deepest value
                var_bottom[t] = v_sorted[-1]

        return var_surface, var_bottom

    def _daily_mean(
        self,
        time_sel: np.ndarray,
        var_surface: np.ndarray,
        var_bottom: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute daily means requiring >= min_hourly valid values per day."""
        dates: np.ndarray = time_sel.astype("datetime64[D]")
        unique_dates = np.unique(dates)
        daily_time = unique_dates.astype("datetime64[ns]")
        daily_surf = np.full(len(unique_dates), np.nan)
        daily_bot = np.full(len(unique_dates), np.nan)

        for i, dd in enumerate(unique_dates):
            mask_d = dates == dd
            vals_s = var_surface[mask_d]
            vals_b = var_bottom[mask_d]
            if np.sum(np.isfinite(vals_s)) >= self.min_hourly:
                daily_surf[i] = np.nanmean(vals_s)
            if np.sum(np.isfinite(vals_b)) >= self.min_hourly:
                daily_bot[i] = np.nanmean(vals_b)

        return daily_time, daily_surf, daily_bot


# =====================================================================
# Meteorological loader
# =====================================================================
def _default_mpos_meteo_dir() -> Path:
    """Resolve MPOS meteorological NetCDF directory from DATA_DIR."""
    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        return Path(data_dir) / "MPOS" / "nc_meteo"
    raise EnvironmentError(
        "DATA_DIR environment variable is not set.\n"
        "Set it (e.g., 'export DATA_DIR=/home/pj24001722/share/Data') or pass\n"
        "mpos_dir explicitly to MposMeteoLoader."
    )


# Default stations to use as wind boundary forcing for Tokyo Bay FVCOM.
# 02kawasaki is excluded by default because the station is shielded by the
# adjacent "Wind Tower" structure to its northeast (per MLIT revise.pdf, 2013).
DEFAULT_METEO_STATIONS: tuple[str, ...] = (
    "01kemigawa",
    "03urayasu",
    "04chiba1gou",
)


@dataclass
class MposMeteoLoader:
    """Load multi-year MPOS meteorological observations for FVCOM forcing.

    Reads ``Mpos_K_{stn}_{year}.nc`` files written by
    ``xmpos.preprocess.mpos2nc_meteo`` and concatenates them into a single
    ``xarray.Dataset`` with dimensions ``(time, station)``.

    Parameters
    ----------
    mpos_dir : str or Path, optional
        Directory containing the yearly NetCDF files. Defaults to
        ``$DATA_DIR/MPOS/nc_meteo``.
    stations : sequence of str, optional
        Station identifiers to load (e.g. ``("01kemigawa", "04chiba1gou")``).
        Defaults to :data:`DEFAULT_METEO_STATIONS`, which excludes
        ``02kawasaki`` (see module docstring).
    convert_to_utc : bool, optional
        If True (default), shift the time axis by -9 hours to convert from
        JST (the storage convention used by xmpos) to UTC, which is what
        FVCOM forcing files expect. When False the JST values are kept
        verbatim with naive timestamps.

    Examples
    --------
    >>> loader = MposMeteoLoader()
    >>> ds = loader.load("2020-01-01", "2021-01-01")
    >>> ds["wind_speed_at_10m"].dims
    ('time', 'station')
    """

    mpos_dir: Path = field(default_factory=_default_mpos_meteo_dir)
    stations: tuple[str, ...] = DEFAULT_METEO_STATIONS
    convert_to_utc: bool = True

    def __post_init__(self) -> None:
        self.mpos_dir = Path(self.mpos_dir)

    def load(
        self,
        start: str,
        end: str,
    ) -> xr.Dataset:
        """Load meteorological data for the requested time range.

        Parameters
        ----------
        start, end : str
            ISO date or datetime strings (e.g. ``"2020-01-01"``). The
            range is interpreted in the time zone of the stored data
            (JST); when ``convert_to_utc`` is True the returned time
            axis is shifted to UTC after subsetting.

        Returns
        -------
        xarray.Dataset
            Variables: ``wind_speed_at_10m``, ``eastward_wind_at_10m``,
            ``northward_wind_at_10m``, ``air_temperature``, plus per-
            station ``latitude``, ``longitude``. Dimensions:
            ``(time, station)``.
        """
        t_start = pd.Timestamp(start)
        t_end = pd.Timestamp(end)
        years = range(t_start.year, t_end.year + 1)

        per_station: dict[str, xr.Dataset] = {}
        for stn in self.stations:
            yearly = []
            for yr in years:
                path = self.mpos_dir / f"Mpos_K_{stn}_{yr}.nc"
                if not path.exists():
                    continue
                yearly.append(xr.open_dataset(path).load())
            if not yearly:
                raise FileNotFoundError(
                    f"No MPOS meteorological NetCDF found for station "
                    f"{stn!r} between {start} and {end} in {self.mpos_dir}"
                )
            ds_stn = xr.concat(yearly, dim="time", data_vars="minimal").sortby("time")
            ds_stn = ds_stn.sel(time=slice(t_start, t_end))
            per_station[stn] = ds_stn

        wanted_vars = (
            "wind_speed_at_10m",
            "eastward_wind_at_10m",
            "northward_wind_at_10m",
            "air_temperature",
        )

        # Establish the canonical time axis from the first station.
        first_stn = self.stations[0]
        time_axis = per_station[first_stn]["time"]
        nt = time_axis.size
        nstn = len(self.stations)

        data_vars: dict[str, xr.DataArray] = {}
        for var in wanted_vars:
            arr = np.full((nt, nstn), np.nan, dtype="float32")
            attrs: dict[str, Any] = {}
            for j, stn in enumerate(self.stations):
                ds_stn = per_station[stn]
                if var in ds_stn:
                    # Reindex to common time axis to handle minor mismatches.
                    a = ds_stn[var].reindex(time=time_axis).astype("float32")
                    arr[:, j] = a.values
                    if not attrs:
                        attrs = {
                            k: v
                            for k, v in ds_stn[var].attrs.items()
                            if k not in ("caveat_use_with_caution",)
                        }
            data_vars[var] = xr.DataArray(
                arr,
                dims=("time", "station"),
                coords={"time": time_axis, "station": list(self.stations)},
                attrs=attrs,
            )

        # Per-station static coordinates. Latitude/longitude were stored as
        # scalars in each yearly file; xr.concat with data_vars="minimal"
        # keeps them scalar.
        def _scalar_coord(da: xr.DataArray) -> float:
            arr = np.atleast_1d(da.values)
            return float(arr[0])

        lats = np.array(
            [_scalar_coord(per_station[s]["latitude"]) for s in self.stations],
            dtype="float64",
        )
        lons = np.array(
            [_scalar_coord(per_station[s]["longitude"]) for s in self.stations],
            dtype="float64",
        )
        ds = xr.Dataset(
            data_vars,
            coords={
                "time": time_axis,
                "station": list(self.stations),
                "latitude": ("station", lats),
                "longitude": ("station", lons),
            },
        )
        ds["latitude"].attrs.update(
            {"standard_name": "latitude", "units": "degrees_north"}
        )
        ds["longitude"].attrs.update(
            {"standard_name": "longitude", "units": "degrees_east"}
        )

        if self.convert_to_utc:
            shifted = pd.DatetimeIndex(ds["time"].values) - pd.Timedelta(hours=9)
            ds = ds.assign_coords(time=shifted)
            ds["time"].attrs["timezone"] = "UTC"
            ds["time"].attrs[
                "conversion_note"
            ] = "Shifted from JST to UTC by subtracting 9 hours during load."
        else:
            ds["time"].attrs["timezone"] = "JST"

        ds.attrs["source"] = "MPOS K-CSV via xmpos.preprocess.mpos2nc_meteo"
        ds.attrs["stations_excluded_by_default"] = (
            "02kawasaki: shielded by the 'Wind Tower' structure to NE; per "
            "MLIT revise.pdf, NE wind speeds are systematically low and "
            "uncorrected."
        )
        return ds

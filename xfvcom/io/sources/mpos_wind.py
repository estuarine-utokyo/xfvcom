"""MPOS-based wind / air-temperature forcing source for FVCOM.

Reads the multi-station meteorological archive built by
``xmpos.preprocess.mpos2nc_meteo`` (yearly per-station CF NetCDF files in
``$DATA_DIR/MPOS/nc_meteo/``) and projects the observations onto an
unstructured FVCOM mesh via inverse-distance weighting (IDW).

The Kawasaki Artificial Island station (``02kawasaki``) is excluded by
default because its anemometer is shielded by the adjacent "Wind Tower"
structure to its NE; northeasterly winds are systematically attenuated and
not corrected upstream (see MLIT revise.pdf, section 5).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from ..mpos import DEFAULT_METEO_STATIONS, MposMeteoLoader
from ..spatial_interp import idw_apply, idw_weights
from .base import BaseForcingSource

# Mapping from FVCOM forcing-variable names (as used in MetNetCDFGenerator)
# to the corresponding MPOS NetCDF variable name.
_VAR_MAP: Final[dict[str, str]] = {
    "uwind": "eastward_wind_at_10m",
    "vwind": "northward_wind_at_10m",
    "air_temp": "air_temperature",
}


class MposWindSource(BaseForcingSource):
    """Provide spatially-varying wind / air-temperature time series from MPOS.

    The source loads station observations once on construction and caches
    them. Per-call interpolation in time is linear; in space it is IDW with
    user-controlled power.

    Parameters
    ----------
    mpos_dir : Path or str, optional
        Directory containing ``Mpos_K_{stn}_{year}.nc``. Defaults to
        ``$DATA_DIR/MPOS/nc_meteo``.
    start, end : str
        ISO datetime range (used to scope the load).
    target_node_lon, target_node_lat : ndarray
        Mesh node coordinates (``node`` dimension); used for variables on
        the ``node`` grid (e.g. ``air_temp``).
    target_elem_lon, target_elem_lat : ndarray
        Mesh element-centroid coordinates (``nele`` dimension); used for
        variables on the ``nele`` grid (``uwind`` / ``vwind``).
    stations : sequence of str, optional
        Station identifiers; defaults to :data:`DEFAULT_METEO_STATIONS`.
    idw_power : float, optional
        IDW power (default 2.0).
    convert_to_utc : bool, optional
        If True (default) convert MPOS JST timestamps to UTC during load,
        matching the FVCOM forcing convention.

    Notes
    -----
    The :meth:`get_series` interface required by :class:`BaseForcingSource`
    returns a 1-D array (mean across stations) for backward compatibility
    with callers that broadcast the result. Spatial output is exposed via
    :meth:`get_spatial_series`, which returns a ``(n_time, n_target)`` array
    suitable for direct assignment to ``(time, nele)`` or ``(time, node)``
    NetCDF variables.
    """

    def __init__(
        self,
        *,
        start: str,
        end: str,
        target_node_lon: NDArray[np.float64],
        target_node_lat: NDArray[np.float64],
        target_elem_lon: NDArray[np.float64],
        target_elem_lat: NDArray[np.float64],
        mpos_dir: Path | str | None = None,
        stations: Sequence[str] | None = None,
        idw_power: float = 2.0,
        convert_to_utc: bool = True,
    ) -> None:
        loader_kwargs: dict[str, object] = {"convert_to_utc": convert_to_utc}
        if mpos_dir is not None:
            loader_kwargs["mpos_dir"] = Path(mpos_dir)
        if stations is not None:
            loader_kwargs["stations"] = tuple(stations)
        loader = MposMeteoLoader(**loader_kwargs)  # type: ignore[arg-type]
        self._ds: xr.Dataset = loader.load(start, end)

        src_lon = np.asarray(self._ds["longitude"].values, dtype=np.float64)
        src_lat = np.asarray(self._ds["latitude"].values, dtype=np.float64)

        self._w_node = idw_weights(
            src_lon,
            src_lat,
            np.asarray(target_node_lon, dtype=np.float64),
            np.asarray(target_node_lat, dtype=np.float64),
            power=idw_power,
        )
        self._w_elem = idw_weights(
            src_lon,
            src_lat,
            np.asarray(target_elem_lon, dtype=np.float64),
            np.asarray(target_elem_lat, dtype=np.float64),
            power=idw_power,
        )
        self._stations = tuple(self._ds["station"].values.tolist())
        self._idw_power = idw_power

    @property
    def stations(self) -> tuple[str, ...]:
        """Active station identifiers (after exclusions)."""
        return self._stations

    @property
    def variables(self) -> tuple[str, ...]:
        """FVCOM-side variable names this source provides."""
        return tuple(_VAR_MAP.keys())

    @property
    def is_spatial(self) -> bool:
        """Identify this source as spatial-aware to MetNetCDFGenerator."""
        return True

    # ------------------------------------------------------------------
    # Time alignment helper
    # ------------------------------------------------------------------
    def _aligned_station_values(
        self, var_name: str, times: pd.DatetimeIndex
    ) -> NDArray[np.float64]:
        """Linearly interpolate per-station series onto *times*.

        Returns
        -------
        ndarray, shape (n_time, n_station)
        """
        if var_name not in _VAR_MAP:
            raise KeyError(
                f"Unsupported variable {var_name!r}; expected one of "
                f"{sorted(_VAR_MAP)}"
            )
        mpos_name = _VAR_MAP[var_name]

        src_times = pd.DatetimeIndex(self._ds["time"].values)
        # Make timezone-naive on both sides before interpolating.
        if getattr(src_times, "tz", None) is not None:
            src_times = src_times.tz_localize(None)
        target = pd.DatetimeIndex(times)
        if getattr(target, "tz", None) is not None:
            target = target.tz_localize(None)

        # Use a consistent ns-precision integer representation; pandas
        # 2.x date_range defaults to us-precision, while xarray-loaded
        # values are typically ns, so .asi8 cannot be compared directly.
        x_src: NDArray[np.float64] = (
            src_times.values.astype("datetime64[ns]")
            .astype(np.int64)
            .astype(np.float64)
        )
        x_tgt: NDArray[np.float64] = (
            target.values.astype("datetime64[ns]").astype(np.int64).astype(np.float64)
        )
        n_stn = self._ds.sizes["station"]
        out = np.empty((target.size, n_stn), dtype=np.float64)

        raw = np.asarray(self._ds[mpos_name].values, dtype=np.float64)
        for j in range(n_stn):
            col = raw[:, j]
            valid = np.isfinite(col)
            if not valid.any():
                out[:, j] = np.nan
                continue
            out[:, j] = np.interp(
                x_tgt,
                x_src[valid],
                col[valid],
                left=np.nan,
                right=np.nan,
            )
        return out

    # ------------------------------------------------------------------
    # BaseForcingSource interface
    # ------------------------------------------------------------------
    def get_series(  # type: ignore[override]
        self, var_name: str, times: pd.DatetimeIndex
    ) -> NDArray[np.float64]:
        """Return the across-station nanmean (1-D fallback)."""
        vals = self._aligned_station_values(var_name, times)
        with np.errstate(invalid="ignore"):
            return np.nanmean(vals, axis=1).astype(np.float64)

    # ------------------------------------------------------------------
    # Spatial-aware interface
    # ------------------------------------------------------------------
    def get_spatial_series(
        self,
        var_name: str,
        times: pd.DatetimeIndex,
        *,
        on: str = "elem",
    ) -> NDArray[np.float64]:
        """Return ``(n_time, n_target)`` array projected onto the mesh.

        Parameters
        ----------
        var_name : str
            FVCOM variable name (``"uwind"``, ``"vwind"``, ``"air_temp"``).
        times : pandas.DatetimeIndex
            Output time axis.
        on : {"elem", "node"}, optional
            Which mesh subset to interpolate to. Defaults to "elem"
            (FVCOM stores wind on element centroids).

        Returns
        -------
        ndarray, shape (len(times), n_target)
        """
        vals = self._aligned_station_values(var_name, times)
        if on == "elem":
            return idw_apply(self._w_elem, vals).astype(np.float64)
        if on == "node":
            return idw_apply(self._w_node, vals).astype(np.float64)
        raise ValueError(f"on must be 'elem' or 'node', got {on!r}")

"""metforce-based meteorological forcing source for FVCOM.

Reads the unified OI analysis built by ``metforce`` (regular MSM-S / ERA5
native grid, typically ``~/Github/metforce/analysis/fvcom_forcing_<YEAR>.nc``)
and projects it onto an unstructured FVCOM mesh by bilinear interpolation.

The metforce file is the production-default atmospheric forcing for Tokyo
Bay (see ``TB-FVCOM/hydro/docs/bc_construction_protocol.md``). It carries
eight FVCOM-relevant variables on a (time, lat, lon) regular grid:

==============  ============  =======================================
metforce var    units         description
==============  ============  =======================================
``U10``         m s-1         OI 10-m eastward wind
``V10``         m s-1         OI 10-m northward wind
``T2``          deg C         OI 2-m air temperature
``SH``          %             OI 2-m relative humidity
``SLP``         hPa           OI mean-sea-level pressure
``DSWRF``       W m-2         OI downwelling shortwave (with clear-sky cap)
``DLWRF``       W m-2         downwelling longwave (Brutsaert + Crawford)
``PRECIP``      mm h-1        OI total precipitation rate
==============  ============  =======================================

Cloud cover is not produced by metforce. Callers that need a cloud
fraction (e.g. when FVCOM derives radiation internally) should fall back
to the ``MetNetCDFGenerator`` scalar default (``cloud=0.0``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from .base import BaseForcingSource

# Map FVCOM-side variable names (the keys used by MetNetCDFGenerator) to:
#   (metforce_var, unit_factor, output_unit, fvcom_target_dim)
# ``unit_factor`` is the multiplier that converts the metforce numeric value
# to the unit FVCOM expects in the output NetCDF (see _DEFAULTS in
# ``MetNetCDFGenerator`` for FVCOM-side conventions).
#
# PRECIP: metforce reports mm h-1. FVCOM expects "m s-1" (equivalent to
# kg m-2 s-1 when assuming a density of 1000 kg m-3). The conversion is
# (1 mm / 3600 s) * (1 m / 1000 mm) = 1.0 / 3.6e6 ≈ 2.778e-7 m s-1 per mm h-1.
_VAR_MAP: Final[dict[str, tuple[str, float, str, str]]] = {
    "uwind": ("U10", 1.0, "m s-1", "elem"),
    "vwind": ("V10", 1.0, "m s-1", "elem"),
    "air_temp": ("T2", 1.0, "Celsius", "node"),
    "rh": ("SH", 1.0, "percent", "node"),
    "prmsl": ("SLP", 1.0, "hPa", "node"),
    "swrad": ("DSWRF", 1.0, "W m-2", "node"),
    "lwrad": ("DLWRF", 1.0, "W m-2", "node"),
    "precip": ("PRECIP", 1.0 / 3.6e6, "m s-1", "node"),
}


class MetforceGriddedSource(BaseForcingSource):
    """Bilinearly interpolate metforce OI analysis onto an FVCOM mesh.

    Parameters
    ----------
    metforce_file : Path or str
        Path to ``fvcom_forcing_<YEAR>.nc`` (or its ``_era5`` variant) on
        a regular ``(time, lat, lon)`` grid.
    target_node_lon, target_node_lat : ndarray
        Mesh node coordinates (``node`` dimension); used for variables on
        the ``node`` grid (everything except wind).
    target_elem_lon, target_elem_lat : ndarray
        Mesh element-centroid coordinates (``nele`` dimension); used for
        the wind components.
    fallback_method : str, optional
        Interpolation to use for FVCOM points that fall outside the
        metforce bounding box (where ``method="linear"`` returns NaN).
        ``"nearest"`` (default) reuses the closest source-grid value; pass
        ``None`` to disable the fallback (NaN will propagate, exposing the
        coverage gap immediately).
    pad_trailing_bookend : bool, optional
        When True, extend the cached source by one extra record at
        ``t_last + native_step`` whose field values are copied verbatim
        from the last source record. Used to satisfy FVCOM's inclusive
        ``END_DATE`` requirement when a caller asks for one source-hour
        past the metforce end (e.g. ``--end 2021-01-01T00:00:00`` on a
        2020 file that ends at ``2020-12-31T23:00:00``). Defaults to
        False to preserve existing behaviour.

    Notes
    -----
    * The interpolation is two-stage: spatial first (bilinear), then time
      (linear). When the target timeline coincides with the metforce
      timeline (the typical case for hourly UTC FVCOM runs starting on
      Jan 1 00:00 UTC), the time stage is a no-op.
    * The Tokyo Bay ``goto2023`` mesh sits well inside the typical
      metforce bbox (139.0–140.5 °E, 35.0–35.7 °N inside metforce's
      138.9–141.5 °E, 34.0–36.3 °N), so the nearest fallback in practice
      triggers only for stray pre-/post-year half-hour offsets, not for
      spatial coverage holes.
    """

    def __init__(
        self,
        metforce_file: Path | str,
        *,
        target_node_lon: NDArray[np.float64],
        target_node_lat: NDArray[np.float64],
        target_elem_lon: NDArray[np.float64],
        target_elem_lat: NDArray[np.float64],
        fallback_method: str | None = "nearest",
        pad_trailing_bookend: bool = False,
    ) -> None:
        path = Path(metforce_file)
        if not path.exists():
            raise FileNotFoundError(f"metforce file not found: {path}")

        self._path = path
        self._ds: xr.Dataset = xr.open_dataset(path)
        self._fallback_method = fallback_method

        # ---- sanity: required vars + canonical dims -------------------
        for vk, (mfk, _f, _u, _d) in _VAR_MAP.items():
            if mfk not in self._ds.data_vars:
                raise KeyError(
                    f"metforce file {path.name} is missing variable {mfk!r} "
                    f"(needed for FVCOM key {vk!r}); cannot use as a forcing "
                    f"source. Available data_vars: "
                    f"{sorted(self._ds.data_vars)}"
                )
        for dim in ("time", "lat", "lon"):
            if dim not in self._ds.dims:
                raise ValueError(
                    f"metforce file {path.name} is missing dimension {dim!r}"
                )

        # Cache lon/lat target arrays as float64; xarray's interp dispatch
        # is fastest when both source coords and target points share dtype.
        self._node_lon = np.asarray(target_node_lon, dtype=np.float64)
        self._node_lat = np.asarray(target_node_lat, dtype=np.float64)
        self._elem_lon = np.asarray(target_elem_lon, dtype=np.float64)
        self._elem_lat = np.asarray(target_elem_lat, dtype=np.float64)

        # Source time as naive UTC datetime64[ns] (metforce time has
        # calendar="proleptic_gregorian", units "hours since YEAR-01-01",
        # which xarray decodes as tz-naive datetime64).
        self._src_time = pd.DatetimeIndex(self._ds["time"].values)
        if self._src_time.tz is not None:
            self._src_time = self._src_time.tz_localize(None)

        if pad_trailing_bookend:
            self._append_trailing_bookend()

    def _append_trailing_bookend(self) -> None:
        """Append one extra record at ``t_last + native_step``.

        Field values are copied verbatim from the last source record, so a
        downstream linear-in-time interpolation onto the FVCOM timeline
        sees a flat extrapolation for the closing hour rather than NaN.
        """
        if self._src_time.size < 2:
            raise ValueError(
                "pad_trailing_bookend requires at least 2 source timesteps "
                "to infer the native cadence."
            )
        step = self._src_time[1] - self._src_time[0]
        new_t = self._src_time[-1] + step
        last = self._ds.isel(time=[-1]).assign_coords(
            time=np.array([np.datetime64(new_t)], dtype="datetime64[ns]")
        )
        self._ds = xr.concat([self._ds, last], dim="time")
        self._src_time = pd.DatetimeIndex(self._ds["time"].values)

    # ------------------------------------------------------------------
    # Introspection used by MetNetCDFGenerator
    # ------------------------------------------------------------------
    @property
    def is_spatial(self) -> bool:
        """Identify this source as spatial-aware to MetNetCDFGenerator."""
        return True

    @property
    def variables(self) -> tuple[str, ...]:
        """FVCOM-side variable names this source provides."""
        return tuple(_VAR_MAP.keys())

    @property
    def metforce_path(self) -> Path:
        """Path of the underlying metforce NetCDF (for provenance)."""
        return self._path

    # ------------------------------------------------------------------
    # BaseForcingSource interface
    # ------------------------------------------------------------------
    def get_series(  # type: ignore[override]
        self, var_name: str, times: pd.DatetimeIndex
    ) -> NDArray[np.float64]:
        """Return the domain-mean series (1-D fallback).

        ``MetNetCDFGenerator`` calls this only when the target dimension is
        ``time``; for ``(time, nele)`` / ``(time, node)`` variables it uses
        :meth:`get_spatial_series` instead.
        """
        if var_name not in _VAR_MAP:
            raise KeyError(
                f"Unsupported variable {var_name!r}; expected one of "
                f"{sorted(_VAR_MAP)}"
            )
        mfk, factor, _u, _d = _VAR_MAP[var_name]
        da = self._ds[mfk].mean(dim=("lat", "lon"), skipna=True)
        return self._time_align_1d(da, times) * factor

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
        """Return ``(n_time, n_target)`` bilinearly interpolated to the mesh.

        Parameters
        ----------
        var_name : str
            FVCOM-side key (``"uwind"``, ``"vwind"``, ``"air_temp"``,
            ``"rh"``, ``"prmsl"``, ``"swrad"``, ``"lwrad"``, ``"precip"``).
        times : pandas.DatetimeIndex
            Output time axis. May be naive (treated as UTC) or tz-aware
            (converted to UTC then stripped).
        on : {"elem", "node"}
            Mesh subset to target. ``"elem"`` returns ``(n_time, nele)``;
            ``"node"`` returns ``(n_time, node)``.

        Returns
        -------
        ndarray, shape (len(times), n_target), dtype float64
        """
        if var_name not in _VAR_MAP:
            raise KeyError(
                f"Unsupported variable {var_name!r}; expected one of "
                f"{sorted(_VAR_MAP)}"
            )
        mfk, factor, _u, _d = _VAR_MAP[var_name]
        if on == "elem":
            tgt_lon, tgt_lat = self._elem_lon, self._elem_lat
        elif on == "node":
            tgt_lon, tgt_lat = self._node_lon, self._node_lat
        else:
            raise ValueError(f"on must be 'elem' or 'node', got {on!r}")

        # --- 1. Spatial bilinear ---------------------------------------
        # Use a shared "point" dim so xarray.interp returns (time, point).
        tgt_lon_da = xr.DataArray(tgt_lon, dims="point")
        tgt_lat_da = xr.DataArray(tgt_lat, dims="point")
        da = self._ds[mfk]
        spatial = da.interp(lat=tgt_lat_da, lon=tgt_lon_da, method="linear")

        # Fallback for points outside the source bbox (NaN from "linear").
        # ``DataArray.sel(method="nearest")`` clips to the nearest source
        # gridpoint by default, which is exactly what we want for points
        # just outside the bbox (xr.interp(method="nearest") would also
        # return NaN there since it inherits scipy's bounds checking).
        if self._fallback_method is not None and bool(spatial.isnull().any()):
            spatial_nn = da.sel(
                lat=tgt_lat_da,
                lon=tgt_lon_da,
                method=self._fallback_method,
            )
            spatial = spatial.fillna(spatial_nn)

        # --- 2. Temporal linear ----------------------------------------
        result = self._time_align_2d(spatial, times)
        return (result * factor).astype(np.float64)

    # ------------------------------------------------------------------
    # Internals — time alignment
    # ------------------------------------------------------------------
    @staticmethod
    def _to_naive(times: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Return *times* with tz info stripped (UTC if originally tz-aware)."""
        if times.tz is not None:
            return times.tz_convert("UTC").tz_localize(None)
        return times

    def _time_align_1d(
        self, da: xr.DataArray, times: pd.DatetimeIndex
    ) -> NDArray[np.float64]:
        """Linearly interpolate a 1-D-in-time DataArray onto *times*."""
        target = self._to_naive(times)
        if target.size == self._src_time.size and bool(
            (target == self._src_time).all()
        ):
            return np.asarray(da.values, dtype=np.float64)
        out = da.interp(time=target, method="linear")
        return np.asarray(out.values, dtype=np.float64)

    def _time_align_2d(
        self, da: xr.DataArray, times: pd.DatetimeIndex
    ) -> NDArray[np.float64]:
        """Linearly interpolate a (time, point) DataArray onto *times*."""
        target = self._to_naive(times)
        if target.size == self._src_time.size and bool(
            (target == self._src_time).all()
        ):
            return np.asarray(da.values, dtype=np.float64)
        # Linear in time only; spatial axis is unchanged.
        out = da.interp(time=target, method="linear")
        return np.asarray(out.values, dtype=np.float64)


__all__ = ["MetforceGriddedSource"]

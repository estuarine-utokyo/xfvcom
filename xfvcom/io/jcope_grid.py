# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Reader for the static JCOPE-T DA grid + bathymetry archive (``basic.nc``).

The archive is produced by ``jcopetda/tools/decode_basic_dat.py`` from the
JAMSTEC ``basic.dat`` Fortran binary. It contains:

* ``lat(jm)``, ``lon(im)`` — grid axes (degrees)
* ``Z(level, lat, lon)``   — layer interface depths (m, negative downward)
* ``ZZ(level, lat, lon)``  — layer center depths   (m, negative downward)
* ``DZ(level, lat, lon)``  — layer thicknesses     (m, positive)
* ``h(lat, lon)``          — bathymetry            (m, positive)
* ``mask(lat, lon)``       — ocean mask (1 = ocean, 0 = land)
* attributes ``grid_origin_lon``, ``grid_origin_lat``, ``grid_dx_deg``,
  ``grid_dy_deg``

This module provides :class:`JcopeGrid`, a small helper that loads the
archive and lets downstream code look up depth and σ-coordinate metadata
at arbitrary (lat, lon) locations — which is what an OBC generator needs.

The vertical arrays are kept lazy (xarray-backed) so that constructing a
``JcopeGrid`` does not eagerly pull ~800 MB of float32 into memory; per-
point ``.compute()``-style slicing reads only what is asked for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr
from numpy.typing import NDArray

__all__ = ["JcopeGrid"]


class JcopeGrid:
    """Lightweight reader for the JCOPE-T DA static-grid archive.

    Parameters
    ----------
    path
        Path to ``basic.nc`` as produced by
        ``jcopetda/tools/decode_basic_dat.py``.

    Notes
    -----
    The returned object holds an open xarray Dataset until :meth:`close`
    (or context-manager exit). Eagerly cached attributes are limited to
    the small 1-D / 2-D arrays (``lat``, ``lon``, ``h``, ``mask``); the
    three 3-D layer arrays are read on demand.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser().resolve()
        self._ds: xr.Dataset = xr.open_dataset(self._path)

        # Small, eagerly cached
        self.lat: NDArray[np.float64] = np.asarray(self._ds["lat"].values)
        self.lon: NDArray[np.float64] = np.asarray(self._ds["lon"].values)
        self.h: NDArray[np.float32] = np.asarray(self._ds["h"].values)
        self.mask: NDArray[np.bool_] = np.asarray(self._ds["mask"].values, dtype=bool)

        # Grid origin / spacing (must be present)
        self.lon_origin: float = float(self._ds.attrs["grid_origin_lon"])
        self.lat_origin: float = float(self._ds.attrs["grid_origin_lat"])
        self.dx: float = float(self._ds.attrs["grid_dx_deg"])
        self.dy: float = float(self._ds.attrs["grid_dy_deg"])

        self.im: int = int(self.lon.size)
        self.jm: int = int(self.lat.size)
        self.km: int = int(self._ds.sizes["level"])

    # ---------- lifecycle ---------- #

    def close(self) -> None:
        """Release the underlying NetCDF handle."""
        self._ds.close()

    def __enter__(self) -> "JcopeGrid":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # ---------- coordinate lookup ---------- #

    def index_of(
        self,
        lat: float | Sequence[float] | NDArray,
        lon: float | Sequence[float] | NDArray,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return ``(j, i)`` indices of the nearest grid cell to ``(lat, lon)``.

        Indices are clipped to the grid extent. Inputs may be scalars or
        same-shape arrays; the output mirrors the input shape.
        """
        lat_arr = np.asarray(lat, dtype=np.float64)
        lon_arr = np.asarray(lon, dtype=np.float64)
        j = np.round((lat_arr - self.lat_origin) / self.dy).astype(np.int64)
        i = np.round((lon_arr - self.lon_origin) / self.dx).astype(np.int64)
        j = np.clip(j, 0, self.jm - 1)
        i = np.clip(i, 0, self.im - 1)
        return j, i

    def find_nearest_ocean(
        self,
        lat: float | Sequence[float] | NDArray,
        lon: float | Sequence[float] | NDArray,
        *,
        max_radius_cells: int = 20,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """For each requested point, return the nearest **ocean** cell.

        Searches outward from the nominal nearest cell up to
        ``max_radius_cells`` grid spacings. Raises ``ValueError`` if no
        ocean cell is found within the radius.
        """
        lat_arr = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon_arr = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        n = lat_arr.size

        j0_all, i0_all = self.index_of(lat_arr, lon_arr)
        out_j = np.empty(n, dtype=np.int64)
        out_i = np.empty(n, dtype=np.int64)

        for p in range(n):
            j0, i0 = int(j0_all[p]), int(i0_all[p])
            if self.mask[j0, i0]:
                out_j[p], out_i[p] = j0, i0
                continue

            found = False
            for r in range(1, max_radius_cells + 1):
                # Ring of radius r around (j0, i0)
                j_lo = max(j0 - r, 0)
                j_hi = min(j0 + r + 1, self.jm)
                i_lo = max(i0 - r, 0)
                i_hi = min(i0 + r + 1, self.im)
                box_mask = self.mask[j_lo:j_hi, i_lo:i_hi]
                if not box_mask.any():
                    continue
                # Distances within the box (in grid cells)
                jj: NDArray[np.int64]
                ii: NDArray[np.int64]
                jj, ii = np.meshgrid(
                    np.arange(j_lo, j_hi),
                    np.arange(i_lo, i_hi),
                    indexing="ij",
                )
                d2 = (jj - j0) ** 2 + (ii - i0) ** 2
                d2 = np.where(box_mask, d2, np.iinfo(np.int64).max)
                idx = np.unravel_index(np.argmin(d2), d2.shape)
                out_j[p] = j_lo + idx[0]
                out_i[p] = i_lo + idx[1]
                found = True
                break

            if not found:
                raise ValueError(
                    f"No ocean cell found within {max_radius_cells} cells "
                    f"of (lat={lat_arr[p]:.4f}, lon={lon_arr[p]:.4f})"
                )

        if np.ndim(lat) == 0:
            return out_j[0:1].reshape(()), out_i[0:1].reshape(())
        return out_j, out_i

    # ---------- depth / vertical metadata ---------- #

    def h_at(
        self,
        lat: float | Sequence[float] | NDArray,
        lon: float | Sequence[float] | NDArray,
        *,
        snap_to_ocean: bool = True,
    ) -> NDArray[np.float32]:
        """Bathymetry at the (nearest) grid cell of each requested point.

        With ``snap_to_ocean=True`` (default), land cells are resolved to
        the nearest ocean cell first.
        """
        if snap_to_ocean:
            j, i = self.find_nearest_ocean(lat, lon)
        else:
            j, i = self.index_of(lat, lon)
        return np.asarray(self.h[j, i], dtype=np.float32)

    def profile(
        self,
        kind: str,
        lat: float | Sequence[float] | NDArray,
        lon: float | Sequence[float] | NDArray,
        *,
        snap_to_ocean: bool = True,
    ) -> NDArray[np.float32]:
        """Vertical profile of ``Z``, ``ZZ``, or ``DZ`` at given points.

        Returns
        -------
        ndarray
            Shape ``(km,)`` for a single point, ``(n_points, km)`` for
            multiple. Values are float32 (Z/ZZ negative, DZ positive).
        """
        if kind not in {"Z", "ZZ", "DZ"}:
            raise ValueError(f"kind must be one of 'Z', 'ZZ', 'DZ' (got {kind!r})")
        if snap_to_ocean:
            j, i = self.find_nearest_ocean(lat, lon)
        else:
            j, i = self.index_of(lat, lon)

        # Read the (level, j, i) slice lazily; xarray fetches only what we ask.
        da = self._ds[kind]
        j_arr = np.atleast_1d(j)
        i_arr = np.atleast_1d(i)
        # Sample one (j, i) at a time — these are point lookups, cheap.
        out = np.empty((j_arr.size, self.km), dtype=np.float32)
        for n, (jn, in_) in enumerate(zip(j_arr, i_arr)):
            out[n] = da.isel(lat=int(jn), lon=int(in_)).values.astype(np.float32)
        if np.ndim(lat) == 0:
            return out[0]
        return out

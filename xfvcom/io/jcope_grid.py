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

# jsasaki 2026-11-05: review round 3 #12: absolute tolerance (degree) between the stored axes and
# origin + index * spacing; the same as the archive / region axis checks, far below half a cell
AXIS_ATOL = 1e-6


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
        # ---> jsasaki 2026-11-05: review round 5 #8: on a square grid the shape cannot tell (lat, lon)
        # from (lon, lat); the dimension names decide, and the arrays are read in (lat, lon) order
        for name in ("h", "mask"):
            dims = tuple(self._ds[name].dims)
            if sorted(dims) != ["lat", "lon"]:
                self._ds.close()
                raise ValueError(
                    f"{self._path}: {name} must have dimensions (lat, lon) (got {dims})"
                )
        self.h: NDArray[np.float32] = np.asarray(
            self._ds["h"].transpose("lat", "lon").values
        )
        self.mask: NDArray[np.bool_] = np.asarray(
            self._ds["mask"].transpose("lat", "lon").values, dtype=bool
        )
        # <--- jsasaki 2026-11-05

        # Grid origin / spacing (must be present)
        self.lon_origin: float = float(self._ds.attrs["grid_origin_lon"])
        self.lat_origin: float = float(self._ds.attrs["grid_origin_lat"])
        self.dx: float = float(self._ds.attrs["grid_dx_deg"])
        self.dy: float = float(self._ds.attrs["grid_dy_deg"])

        self.im: int = int(self.lon.size)
        self.jm: int = int(self.lat.size)
        self.km: int = int(self._ds.sizes["level"])
        # jsasaki 2026-11-05: review round 3 #12: index arithmetic uses the attributes, the payload
        # uses the axes: they must describe the same cells
        self._check_geometry()

    def _check_geometry(self) -> None:
        for name, value in (
            ("grid_origin_lon", self.lon_origin),
            ("grid_origin_lat", self.lat_origin),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{self._path}: {name} must be finite")
        for name, value in (("grid_dx_deg", self.dx), ("grid_dy_deg", self.dy)):
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"{self._path}: {name} must be finite and positive")
        for name, axis, origin, step in (
            ("lat", self.lat, self.lat_origin, self.dy),
            ("lon", self.lon, self.lon_origin, self.dx),
        ):
            if axis.ndim != 1 or axis.size == 0:
                raise ValueError(f"{self._path}: {name} must be a non-empty 1-D axis")
            want = origin + np.arange(axis.size, dtype=np.float64) * step
            if not (
                np.isfinite(axis).all()
                and np.allclose(axis, want, rtol=0, atol=AXIS_ATOL)
            ):
                raise ValueError(
                    f"{self._path}: the {name} axis is not grid_origin + index * spacing "
                    f"(tolerance {AXIS_ATOL:g} degree)"
                )
        if self.h.shape != (self.jm, self.im) or self.mask.shape != (self.jm, self.im):
            raise ValueError(
                f"{self._path}: h / mask are not (lat, lon) = ({self.jm}, {self.im})"
            )

    @staticmethod
    def _coords(lat, lon) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Finite, unmasked coordinates: two scalars or two 1-D arrays of the same length."""
        # jsasaki 2026-11-05: review round 3 #17: invalid coordinates are refused, not clipped
        if np.ma.is_masked(lat) or np.ma.is_masked(lon):
            raise ValueError("lat / lon have masked entries")
        lat_arr = np.asarray(np.ma.getdata(lat))
        lon_arr = np.asarray(np.ma.getdata(lon))
        if lat_arr.dtype.kind not in "iuf" or lon_arr.dtype.kind not in "iuf":
            raise ValueError("lat / lon must be real numbers")
        lat_arr = lat_arr.astype(np.float64)
        lon_arr = lon_arr.astype(np.float64)
        if lat_arr.ndim > 1 or lat_arr.shape != lon_arr.shape:
            raise ValueError(
                f"lat / lon must be two scalars or two 1-D arrays of the same length "
                f"(got shapes {lat_arr.shape}, {lon_arr.shape})"
            )
        if not (np.isfinite(lat_arr).all() and np.isfinite(lon_arr).all()):
            raise ValueError("lat / lon must be finite")
        return lat_arr, lon_arr

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

        Indices are clipped to the grid extent. Inputs are two finite scalars or
        two 1-D arrays of the same length; the output mirrors the input shape.
        """
        # jsasaki 2026-11-05: review round 3 #17: validated coordinates
        lat_arr, lon_arr = self._coords(lat, lon)
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

        Every ocean cell within ``max_radius_cells`` grid spacings (a square
        around the nominal nearest cell) is a candidate; the one whose centre
        is closest (Euclidean, in fractional cell units) to the REQUESTED
        position wins, ties going to the first in (j, i) order. The same rule
        applies when the nominal cell itself is ocean. Raises ``ValueError``
        if no ocean cell is found within the radius.
        """
        # ---> jsasaki 2026-11-05: review round 3 #17: validated coordinates and radius
        if (
            isinstance(max_radius_cells, (bool, np.bool_))
            or not isinstance(max_radius_cells, (int, np.integer))
            or max_radius_cells < 1
        ):
            raise ValueError(
                f"max_radius_cells must be a positive integer (got {max_radius_cells!r})"
            )
        lat0, lon0 = self._coords(lat, lon)
        lat_arr = np.atleast_1d(lat0)
        lon_arr = np.atleast_1d(lon0)
        # <--- jsasaki 2026-11-05
        n = lat_arr.size

        j0_all, i0_all = self.index_of(lat_arr, lon_arr)
        # jsasaki 2026-11-05: review round 2 #1: distances from the requested (fractional) position
        fj_all = (lat_arr - self.lat_origin) / self.dy
        fi_all = (lon_arr - self.lon_origin) / self.dx
        out_j = np.empty(n, dtype=np.int64)
        out_i = np.empty(n, dtype=np.int64)

        for p in range(n):
            j0, i0 = int(j0_all[p]), int(i0_all[p])
            # jsasaki 2026-11-05: review round 3 #20: no rounded-cell shortcut; the box search applies
            # the documented tie rule (argmin over the row-major box = first in (j, i) order)

            # jsasaki 2026-11-05: review round 2 #1: search ALL candidates within the radius instead of
            # stopping at the first non-empty square (that could pick a farther cell in a corner)
            r = int(max_radius_cells)
            j_lo = max(j0 - r, 0)
            j_hi = min(j0 + r + 1, self.jm)
            i_lo = max(i0 - r, 0)
            i_hi = min(i0 + r + 1, self.im)
            box_mask = self.mask[j_lo:j_hi, i_lo:i_hi]
            found = bool(box_mask.any())
            if found:
                jj, ii = np.meshgrid(
                    np.arange(j_lo, j_hi, dtype=np.float64),
                    np.arange(i_lo, i_hi, dtype=np.float64),
                    indexing="ij",
                )
                d2 = (jj - fj_all[p]) ** 2 + (ii - fi_all[p]) ** 2
                d2 = np.where(box_mask, d2, np.inf)
                idx = np.unravel_index(np.argmin(d2), d2.shape)
                out_j[p] = j_lo + int(idx[0])
                out_i[p] = i_lo + int(idx[1])

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

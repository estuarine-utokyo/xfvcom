# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Generate FVCOM open-boundary forcing NetCDFs from a JCOPE-T DA archive.

This module is the FVCOM-facing half of the jcopetda / xfvcom split: the
jcopetda repo produces CF-compliant archive NetCDFs (a static grid file
``basic.nc`` decoded from JAMSTEC's Fortran ``basic.dat``, and per-year
sub-region time-series ``jcopetda_region_YYYY.nc``); xfvcom consumes them
to build the OBC forcing files FVCOM actually reads.

The class :class:`JcopeObcGenerator` writes two file types, both following
the conventions already in use under ``TB-FVCOM/input/goto2023/obc/``:

* **TS OBC** (``tb_tsobc_jcope_*.nc``) — temperature and salinity at FVCOM
  σ-layers along the open boundary, used for nudging.
* **Julian elevation** (``tb_julian_obc_jcope_*.nc``) — sea-surface
  elevation at OBC nodes.

The vertical pipeline avoids the J-EGG500 approximation that
``TB-FVCOM/input/scripts/create_obc_from_jcope.py`` used to need: the
JCOPE side now reads its real bathymetry and POM-generalized σ levels
from ``basic.nc`` via :class:`xfvcom.io.JcopeGrid`. For each OBC node and
time step we therefore have:

1. Real ``h_jcope`` and real per-cell ``ZZ`` (= z of JCOPE layer centers).
2. Target FVCOM z = ``siglay * h_fvcom`` (η = 0 at OBC nudging time).
3. Linear interpolation in z-space via
   :func:`xfvcom.grid.sigma.vertical_interp`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from ..grid.sigma import vertical_interp
from .jcope_grid import JcopeGrid
from .netcdf_utils import to_mjd

__all__ = ["JcopeObcGenerator"]

# JCOPE-T DA stores 47 vertical levels in basic.dat but the per-hour
# variable files (TT/ST/UT/VT) only carry the 46 active layer centers
# (k=0..45). The 47th (deepest) ZZ entry is a POM-style padding below the
# seabed and must be ignored for interpolation.
N_JCOPE_ACTIVE_LEVELS = 46


class JcopeObcGenerator:
    """Build FVCOM OBC NetCDFs from a JCOPE-T DA grid + region archive.

    Parameters
    ----------
    grid
        Open :class:`JcopeGrid` (the ``basic.nc`` reader).
    region_nc
        Path to a ``jcopetda_region_YYYY.nc`` time-series archive whose
        domain covers all OBC nodes.
    obc_nodes
        FVCOM node IDs (1-based) along the open boundary, in the order
        they should appear in the output ``nobc`` dimension.
    obc_lat, obc_lon
        Geographic coordinates of the OBC nodes (degrees).
    obc_h_fvcom
        FVCOM bathymetry (m, positive) at each OBC node. This is the
        ``H`` value FVCOM itself uses; the generator does **not** override
        it with the JCOPE depth — instead it interpolates the JCOPE field
        onto FVCOM's σ × h column so the OBC values agree with the model's
        internal z-coordinate.
    n_siglay
        Number of FVCOM σ-layers (uniform spacing). Default 30 matches
        the current TB-FVCOM baseline.

    Notes
    -----
    All vertical interpolation is constant-extrapolated past the source
    bounds (see :func:`xfvcom.grid.sigma.vertical_interp`). This is the
    safe choice for OBC forcing where T/S vary smoothly near the
    free surface and seabed.
    """

    def __init__(
        self,
        *,
        grid: JcopeGrid,
        region_nc: str | Path,
        obc_nodes: Sequence[int],
        obc_lat: ArrayLike,
        obc_lon: ArrayLike,
        obc_h_fvcom: ArrayLike,
        n_siglay: int = 30,
    ) -> None:
        self.grid = grid
        self.region_nc_path = Path(region_nc).expanduser().resolve()
        self._region_ds: xr.Dataset = xr.open_dataset(self.region_nc_path)

        self.obc_nodes = np.asarray(obc_nodes, dtype=np.int32)
        self.obc_lat = np.asarray(obc_lat, dtype=np.float64)
        self.obc_lon = np.asarray(obc_lon, dtype=np.float64)
        self.obc_h_fvcom = np.asarray(obc_h_fvcom, dtype=np.float32)

        n_obc = self.obc_nodes.size
        if not (
            self.obc_lat.size == n_obc
            and self.obc_lon.size == n_obc
            and self.obc_h_fvcom.size == n_obc
        ):
            raise ValueError(
                "obc_nodes, obc_lat, obc_lon, obc_h_fvcom must all have the "
                f"same length (got {n_obc}, {self.obc_lat.size}, "
                f"{self.obc_lon.size}, {self.obc_h_fvcom.size})"
            )
        self.n_obc = n_obc

        if n_siglay < 2:
            raise ValueError(f"n_siglay must be ≥ 2 (got {n_siglay})")
        self.n_siglay = int(n_siglay)
        self.n_siglev = self.n_siglay + 1

        # FVCOM uniform σ
        self.siglev: NDArray[np.float64] = (
            -np.arange(self.n_siglev, dtype=np.float64) / self.n_siglay
        )
        self.siglay: NDArray[np.float64] = (
            -(np.arange(self.n_siglay, dtype=np.float64) + 0.5) / self.n_siglay
        )

        # Authoritative JCOPE depth and σ at the nearest ocean cell of each
        # OBC node. Both reads use basic.nc's mask, so they are consistent.
        self.obc_h_jcope: NDArray[np.float32] = grid.h_at(self.obc_lat, self.obc_lon)
        # ZZ profile has km values; drop the deepest one (POM padding below
        # the seabed) so it matches the 46-level time-series.
        zz_full = grid.profile("ZZ", self.obc_lat, self.obc_lon)
        if zz_full.ndim == 1:
            zz_full = zz_full[np.newaxis, :]
        self.obc_z_jcope: NDArray[np.float32] = zz_full[
            :, :N_JCOPE_ACTIVE_LEVELS
        ].astype(np.float32)

        # Map each OBC node to a (j_region, i_region) cell in region_nc.
        self._jr, self._ir = self._locate_in_region()

        # Time axis converted to MJD float64 once
        time_index = pd.DatetimeIndex(self._region_ds["time"].values)
        self._mjd: NDArray[np.float64] = to_mjd(time_index)

    # ---------- lifecycle ---------- #

    def close(self) -> None:
        """Close the region NetCDF handle."""
        self._region_ds.close()

    def __enter__(self) -> "JcopeObcGenerator":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # ---------- region indexing ---------- #

    def _locate_in_region(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Translate each OBC (lat, lon) to a (j, i) index inside region_nc.

        Uses :class:`JcopeGrid`'s global nearest-ocean lookup (anchored on
        the authoritative ``basic.nc`` mask), then shifts the result into
        the region's local axes. This guarantees the same physical cell
        is sampled for both depth/σ metadata and time-series values.
        """
        # Region's local axes
        r_lat = np.asarray(self._region_ds["lat"].values)
        r_lon = np.asarray(self._region_ds["lon"].values)

        # Region offset in the global grid: where does the region start?
        j0, i0 = self.grid.index_of(r_lat[0], r_lon[0])
        j0_int, i0_int = int(j0), int(i0)

        # Global ocean cells for OBC nodes
        jg, ig = self.grid.find_nearest_ocean(self.obc_lat, self.obc_lon)

        jr = np.asarray(jg, dtype=np.int64) - j0_int
        ir = np.asarray(ig, dtype=np.int64) - i0_int

        if (
            (jr < 0).any()
            or (ir < 0).any()
            or (jr >= r_lat.size).any()
            or (ir >= r_lon.size).any()
        ):
            raise ValueError(
                "At least one OBC node falls outside the region archive; "
                "extend the region or restrict the OBC list."
            )
        return jr, ir

    # ---------- variable readers ---------- #

    def _read_variable_profile(self, name: str) -> NDArray[np.float32]:
        """Read a 4-D field at every OBC node and time step.

        Returns
        -------
        ndarray
            Shape ``(n_time, n_level, n_obc)``, float32. Masked / NaN
            entries from the source are preserved; downstream interpolation
            (``vertical_interp``) drops them.
        """
        da = self._region_ds[name]
        if da.ndim != 4:
            raise ValueError(
                f"{name!r} is not a 4-D (time, level, lat, lon) field " f"in region_nc"
            )
        # Vectorized fancy indexing over n_obc; one array fetch per node.
        n_time = da.sizes["time"]
        n_level = da.sizes["level"]
        out = np.empty((n_time, n_level, self.n_obc), dtype=np.float32)
        for n in range(self.n_obc):
            vals = da.isel(
                lat=int(self._jr[n]),
                lon=int(self._ir[n]),
            ).values
            out[:, :, n] = np.asarray(vals, dtype=np.float32)
        return out

    def _read_variable_2d(self, name: str) -> NDArray[np.float32]:
        """Read a 3-D field (no level axis) at every OBC node and time.

        Returns ``(n_time, n_obc)`` float32.
        """
        da = self._region_ds[name]
        if da.ndim != 3:
            raise ValueError(
                f"{name!r} is not a 3-D (time, lat, lon) field in region_nc"
            )
        n_time = da.sizes["time"]
        out = np.empty((n_time, self.n_obc), dtype=np.float32)
        for n in range(self.n_obc):
            vals = da.isel(
                lat=int(self._jr[n]),
                lon=int(self._ir[n]),
            ).values
            out[:, n] = np.asarray(vals, dtype=np.float32)
        return out

    # ---------- vertical interpolation ---------- #

    def _interp_to_fvcom_sigma(
        self,
        field: NDArray,
        h_jcope: NDArray[np.float32],
        h_fvcom: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Map a ``(time, level, n_obc)`` JCOPE field onto FVCOM σ-layers.

        Returns ``(time, n_siglay, n_obc)`` float32.
        """
        n_time, n_level, n_obc = field.shape
        if n_level != N_JCOPE_ACTIVE_LEVELS:
            # Allow callers that pre-trim (or pad) — but warn via shape mismatch.
            raise ValueError(
                f"expected {N_JCOPE_ACTIVE_LEVELS} JCOPE levels, got {n_level}"
            )
        out = np.empty((n_time, self.n_siglay, n_obc), dtype=np.float32)
        # FVCOM z columns are static (η=0 assumption) — precompute once.
        # siglay is negative, h_fvcom positive, so z is negative.
        z_fvcom = (self.siglay[:, None] * h_fvcom[None, :]).astype(np.float64)
        # JCOPE z columns are also static (per OBC node).
        z_jcope = self.obc_z_jcope.astype(np.float64)  # (n_obc, n_level)

        for n in range(n_obc):
            src_z = z_jcope[n]
            dst_z = z_fvcom[:, n]
            for t in range(n_time):
                out[t, :, n] = vertical_interp(field[t, :, n], src_z, dst_z)
        return out

    # ---------- public writers ---------- #

    def write_tsobc(
        self,
        out_path: str | Path,
        *,
        variables: Iterable[str] = ("temp", "salt"),
        title: str | None = None,
    ) -> Path:
        """Write a FVCOM TIME SERIES OBC TS FILE.

        Parameters
        ----------
        out_path
            Output NetCDF path.
        variables
            Subset of ``{"temp", "salt"}`` to include. The mapping from
            FVCOM variable names to JCOPE-T DA names is::

                temp -> TT  (potential temperature, °C)
                salt -> ST  (salinity, PSU)
        title
            Optional NetCDF ``title`` attribute (defaults to a generic
            description).
        """
        var_map = {
            "temp": ("TT", "Celcius", "sea_water_temperature"),
            "salt": ("ST", "PSU", "sea_water_salinity"),
        }
        vars_list = list(variables)
        for v in vars_list:
            if v not in var_map:
                raise ValueError(
                    f"variable {v!r} is not supported by write_tsobc "
                    f"(allowed: {sorted(var_map)})"
                )

        # Interpolate each requested variable onto FVCOM σ
        fvcom_fields: dict[str, NDArray[np.float32]] = {}
        for v in vars_list:
            jcope_name, _units, _long = var_map[v]
            jcope_field = self._read_variable_profile(jcope_name)[
                :, :N_JCOPE_ACTIVE_LEVELS, :
            ]
            fvcom_fields[v] = self._interp_to_fvcom_sigma(
                jcope_field, self.obc_h_jcope, self.obc_h_fvcom
            )

        out_path = Path(out_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with nc.Dataset(out_path, "w", format="NETCDF4") as ds:
            ds.createDimension("time", None)
            ds.createDimension("nobc", self.n_obc)
            ds.createDimension("siglay", self.n_siglay)
            ds.createDimension("siglev", self.n_siglev)
            ds.createDimension("DateStrLen", 26)

            v_time = ds.createVariable("time", "f8", ("time",))
            v_time.long_name = "time"
            v_time.units = "days since 1858-11-17 00:00:00"
            v_time.format = "modified julian day (MJD)"
            v_time.time_zone = "UTC"
            v_time[:] = self._mjd

            v_nodes = ds.createVariable("obc_nodes", "i4", ("nobc",))
            v_nodes.long_name = "Open Boundary Node Number"
            v_nodes.units = "-"
            v_nodes[:] = self.obc_nodes

            v_h = ds.createVariable("obc_h", "f4", ("nobc",))
            v_h.long_name = "Open Boundary Depth"
            v_h.units = "m"
            v_h[:] = self.obc_h_fvcom

            v_siglev = ds.createVariable("siglev", "f4", ("siglev", "nobc"))
            v_siglev.long_name = "ocean_sigma/general_coordinate"
            v_siglev.units = "-"
            v_siglev[:] = np.broadcast_to(
                self.siglev[:, None], (self.n_siglev, self.n_obc)
            ).astype(np.float32)

            v_siglay = ds.createVariable("siglay", "f4", ("siglay", "nobc"))
            v_siglay.long_name = "ocean_sigma/general_coordinate"
            v_siglay.units = "-"
            v_siglay[:] = np.broadcast_to(
                self.siglay[:, None], (self.n_siglay, self.n_obc)
            ).astype(np.float32)

            for v in vars_list:
                jcope_name, units, long_name = var_map[v]
                vname = f"obc_{v if v == 'temp' else 'salinity'}"
                v_var = ds.createVariable(
                    vname,
                    "f4",
                    ("time", "siglay", "nobc"),
                )
                v_var.long_name = long_name
                v_var.units = units
                v_var[:] = fvcom_fields[v]

            ds.title = (
                title or f"Open boundary temperature and salinity from JCOPE-T DA"
            )
            ds.type = "FVCOM TIME SERIES OBC TS FILE"
            ds.source = (
                "JCOPE-T DA (JAMSTEC), interpolated to FVCOM OBC sigma "
                "layers via xfvcom.io.JcopeObcGenerator"
            )
            ds.history = (
                f"Created {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
                f" UTC from {self.region_nc_path.name}"
            )

        return out_path

    def write_elevation(
        self,
        out_path: str | Path,
        *,
        title: str | None = None,
    ) -> Path:
        """Write a FVCOM TIME SERIES ELEVATION FORCING FILE."""
        egt = self._read_variable_2d("EGT")  # (n_time, n_obc)

        out_path = Path(out_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with nc.Dataset(out_path, "w", format="NETCDF4") as ds:
            ds.createDimension("time", None)
            ds.createDimension("nobc", self.n_obc)
            ds.createDimension("DateStrLen", 26)

            v_nodes = ds.createVariable("obc_nodes", "i4", ("nobc",))
            v_nodes.long_name = "Open Boundary Node Number"
            v_nodes.units = "-"
            v_nodes[:] = self.obc_nodes

            v_time = ds.createVariable("time", "f8", ("time",))
            v_time.long_name = "time"
            v_time.units = "days since 1858-11-17 00:00:00"
            v_time.format = "modified julian day (MJD)"
            v_time.time_zone = "UTC"
            v_time[:] = self._mjd

            v_elev = ds.createVariable(
                "elevation",
                "f4",
                ("time", "nobc"),
            )
            v_elev.long_name = "Open Boundary Elevation"
            v_elev.units = "meters"
            v_elev[:] = egt

            ds.title = title or "Open boundary elevation from JCOPE-T DA"
            ds.type = "FVCOM TIME SERIES ELEVATION FORCING FILE"
            ds.source = (
                "JCOPE-T DA (JAMSTEC), at OBC node nearest ocean cell, "
                "produced by xfvcom.io.JcopeObcGenerator"
            )
            ds.history = (
                f"Created {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
                f" UTC from {self.region_nc_path.name}"
            )

        return out_path

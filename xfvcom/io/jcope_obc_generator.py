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

import os

# jsasaki 2026-11-05: review round 2 #16: unique temporary files
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike, NDArray

from ..grid.sigma import vertical_interp

# jsasaki 2026-11-04: full-domain archive source (FullDomainSource)
from .jcope_fulldomain import FullDomainSource
from .jcope_grid import JcopeGrid
from .netcdf_utils import to_mjd

__all__ = ["JcopeObcGenerator", "write_tsobc_nc", "write_elevation_nc"]

# JCOPE-T DA stores 47 vertical levels in basic.dat but the per-hour
# variable files (TT/ST/UT/VT) only carry the 46 active layer centers
# (k=0..45). The 47th (deepest) ZZ entry is a POM-style padding below the
# seabed and must be ignored for interpolation.
N_JCOPE_ACTIVE_LEVELS = 46

# FVCOM-side variable specs for the tsobc writer.
#   key  -> (jcope variable name in region.nc, FVCOM output variable name,
#           CF-ish units, long_name)
_TSOBC_VAR_MAP: dict[str, tuple[str, str, str, str]] = {
    "temp": ("TT", "obc_temp", "Celcius", "sea_water_temperature"),
    "salt": ("ST", "obc_salinity", "PSU", "sea_water_salinity"),
}


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
        the current TB-FVCOM baseline. Ignored when ``siglay_per_node`` /
        ``siglev_per_node`` are given.
    siglay_per_node, siglev_per_node
        Optional per-node vertical coordinate for a **sigma-z / SADAPT** grid:
        ``siglay_per_node`` is ``(n_siglay, n_obc)`` and ``siglev_per_node``
        ``(n_siglay + 1, n_obc)``, both negative (0 surface → −1 bed). When
        given they override the uniform σ above so the JCOPE field is
        interpolated onto each OBC node's ACTUAL layer depths. Required for a
        sigma-z run: FVCOM's OBC T/S nudging copies the file's
        ``temp(time, siglay, nobc)`` straight into model layer ``K`` with no
        vertical re-interpolation (``mod_force.F::UPDATE_OBC_TEMP``), so the
        file must already be on the model's per-node coordinate. Build these
        from the SADAPT ``sigma.dat`` via :func:`xfvcom.io.read_sigma_dat` +
        :func:`xfvcom.grid.gtsz_builder.build_gtsz` (``coord.Z`` at the OBC
        nodes = per-node siglev; layer midpoints = siglay).

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
        # jsasaki 2026-11-04: full-domain archive source (FullDomainSource)
        region_nc: "str | Path | FullDomainSource",
        obc_nodes: Sequence[int] | NDArray[np.integer],
        obc_lat: ArrayLike,
        obc_lon: ArrayLike,
        obc_h_fvcom: ArrayLike,
        n_siglay: int = 30,
        siglay_per_node: ArrayLike | None = None,
        siglev_per_node: ArrayLike | None = None,
    ) -> None:
        self.grid = grid
        # jsasaki 2026-11-04: the source may be a FullDomainSource (the daily full-domain archive)
        # instead of a jcopetda_region_YYYY.nc path; the region path is unchanged.
        self._src: FullDomainSource | None = None
        self._region_ds: xr.Dataset | None = None
        # jsasaki 2026-11-05: review round 2 #18: source_files is a property evaluated at write time
        # (a FullDomainSource reports only what was read by then)
        if isinstance(region_nc, FullDomainSource):
            self._src = region_nc
            self.region_nc_path = region_nc.root
            self.source_name = region_nc.name
        else:
            self.region_nc_path = Path(region_nc).expanduser().resolve()
            self._region_ds = xr.open_dataset(self.region_nc_path)
            self.source_name = self.region_nc_path.name
            # jsasaki 2026-11-05: review round 3 #2: the supported coordinate schema of a region file
            _validate_region_schema(self._region_ds, self.region_nc_path)

        # jsasaki 2026-11-05: review round 2 #13: node ids are validated, not silently coerced
        self.obc_nodes = _validate_obc_nodes(obc_nodes)
        # jsasaki 2026-11-05: review round 3 #3 #4: masked entries become NaN (refused below); a
        # depth beyond the float32 range becomes inf (refused below)
        self.obc_lat = _plain_float(obc_lat, "obc_lat")
        self.obc_lon = _plain_float(obc_lon, "obc_lon")
        self.obc_h_fvcom = _to_float32(_plain_float(obc_h_fvcom, "obc_h_fvcom"))
        # ---> jsasaki 2026-11-05: review round 4 #5: exactly (n_obc,), checked before anything else;
        # a (1, n_obc) depth broadcast against the layer axis and gave finite wrong profiles
        n_obc = self.obc_nodes.size
        for label, arr in (
            ("obc_lat", self.obc_lat),
            ("obc_lon", self.obc_lon),
            ("obc_h_fvcom", self.obc_h_fvcom),
        ):
            if arr.shape != (n_obc,):
                raise ValueError(
                    f"{label} must have shape ({n_obc},) like obc_nodes (got {arr.shape})"
                )
        # <--- jsasaki 2026-11-05

        # jsasaki 2026-11-04: review round 1 #10 #11: refuse physically invalid inputs
        if not (np.isfinite(self.obc_lat).all() and np.isfinite(self.obc_lon).all()):
            raise ValueError("obc_lat / obc_lon must be finite")
        # jsasaki 2026-11-05: review round 2 #14: the grid extent plus HALF a spacing (the outer
        # cells' own area), not a whole spacing
        if not (
            self.obc_lat.min() >= grid.lat.min() - 0.5 * grid.dy
            and self.obc_lat.max() <= grid.lat.max() + 0.5 * grid.dy
            and self.obc_lon.min() >= grid.lon.min() - 0.5 * grid.dx
            and self.obc_lon.max() <= grid.lon.max() + 0.5 * grid.dx
        ):
            raise ValueError("an OBC node lies outside the JCOPE grid")
        if not (np.isfinite(self.obc_h_fvcom).all() and (self.obc_h_fvcom > 0).all()):
            raise ValueError("obc_h_fvcom must be finite and positive")
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

        if siglay_per_node is not None or siglev_per_node is not None:
            # Per-node vertical coordinate (a SADAPT / sigma-z grid where the
            # sigma-layer depths vary node to node). siglay_per_node is
            # (n_siglay, n_obc), siglev_per_node (n_siglev, n_obc); both negative,
            # 0 at the surface to -1 at the bed. These are used directly as the
            # FVCOM target z = sigma * h per OBC node, so the OBC values land on
            # the model's ACTUAL layer depths. (Uniform sigma * h is WRONG for a
            # sigma-z grid, whose deep z-levels sit at fixed absolute depths and
            # whose active-layer count varies per node — see the design doc
            # docs/sigmaz_estuary_circulation_design.md §9.) FVCOM's OBC T/S
            # nudging reads the file's temp(time, siglay, nobc) DIRECTLY into the
            # model layer K (mod_force.F::UPDATE_OBC_TEMP does time interpolation
            # only, no vertical re-interpolation), so the file must already be on
            # the model coordinate.
            if siglay_per_node is None or siglev_per_node is None:
                raise ValueError(
                    "siglay_per_node and siglev_per_node must be given together"
                )
            # jsasaki 2026-11-05: review round 3 #3: masked entries become NaN (refused by _validate_sigma)
            sl = _plain_float(siglay_per_node, "siglay_per_node")
            sv = _plain_float(siglev_per_node, "siglev_per_node")
            if sl.ndim != 2 or sv.ndim != 2:
                raise ValueError("per-node siglay/siglev must be 2-D (n_sig, n_obc)")
            if sl.shape[1] != n_obc or sv.shape[1] != n_obc:
                raise ValueError(
                    f"per-node siglay/siglev second dim must be n_obc={n_obc} "
                    f"(got {sl.shape[1]}, {sv.shape[1]})"
                )
            if sv.shape[0] != sl.shape[0] + 1:
                raise ValueError(
                    f"siglev rows ({sv.shape[0]}) must be siglay rows "
                    f"({sl.shape[0]}) + 1"
                )
            if sl.shape[0] < 1:
                raise ValueError(f"n_siglay must be ≥ 1 (got {sl.shape[0]})")
            self.n_siglay = int(sl.shape[0])
            self.n_siglev = int(sv.shape[0])
            self.siglay: NDArray[np.float64] = sl
            self.siglev: NDArray[np.float64] = sv
        else:
            if n_siglay < 2:
                raise ValueError(f"n_siglay must be ≥ 2 (got {n_siglay})")
            self.n_siglay = int(n_siglay)
            self.n_siglev = self.n_siglay + 1

            # FVCOM uniform σ
            self.siglev = -np.arange(self.n_siglev, dtype=np.float64) / self.n_siglay
            self.siglay = (
                -(np.arange(self.n_siglay, dtype=np.float64) + 0.5) / self.n_siglay
            )

        # jsasaki 2026-11-04: review round 1 #10: sigma coordinates must be finite, in [-1, 0] and
        # ordered from the surface (0) to the bed (-1)
        # jsasaki 2026-11-05: review round 2 #9: endpoints 0 / -1 and each siglay between its two
        # siglev interfaces; the same check runs in write_tsobc_nc
        _validate_sigma(self.siglev, self.siglay, n_obc)
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
        # jsasaki 2026-11-05: review round 2 #10: the sampled JCOPE geometry must be usable
        _validate_jcope_geometry(
            np.atleast_1d(self.obc_h_jcope), self.obc_z_jcope, self.obc_nodes
        )

        # Map each OBC node to a (j_region, i_region) cell in region_nc.
        self._jr, self._ir = self._locate_in_region()

        # Time axis converted to MJD float64 once
        # jsasaki 2026-11-04: full-domain archive source (FullDomainSource)
        # jsasaki 2026-11-05: review round 5 #7: the region file's time must be decoded datetimes
        time_index = (
            pd.DatetimeIndex(self._src.time_index)
            if self._src is not None
            else _region_time_index(self._ds, self.region_nc_path)
        )
        self.mjd: NDArray[np.float64] = to_mjd(time_index)
        # Back-compat alias for code that referenced the underscored name.
        self._mjd = self.mjd

    @property
    def _ds(self) -> xr.Dataset:
        """The open region dataset (region-file mode only)."""
        # jsasaki 2026-11-04: _region_ds is None in full-domain mode
        assert self._region_ds is not None
        return self._region_ds

    @property
    def source_files(self) -> list[str]:
        """History lines naming the source (for an archive: what was read and repaired so far)."""
        return self.source_lines()

    def source_lines(self, jcope_names: Iterable[str] | None = None) -> list[str]:
        """History lines naming the source, restricted to ``jcope_names`` (TT ST ... EGT) for an archive."""
        if self._src is not None:
            return self._src.provenance(jcope_names)
        return [self.source_name]

    # ---------- lifecycle ---------- #

    def close(self) -> None:
        """Close the region NetCDF handle."""
        # jsasaki 2026-11-04: full-domain archive source (FullDomainSource)
        if self._region_ds is not None:
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
        # jsasaki 2026-11-04: full-domain archive source (FullDomainSource)
        if self._src is not None and not (
            self._src.lat.shape == self.grid.lat.shape
            and self._src.lon.shape == self.grid.lon.shape
            and np.allclose(self._src.lat, self.grid.lat, rtol=0, atol=1e-6)
            and np.allclose(self._src.lon, self.grid.lon, rtol=0, atol=1e-6)
        ):
            # jsasaki 2026-11-04: review round 1 #4: the day files must be on the grid file's axes
            raise ValueError(
                "the full-domain archive's lat/lon axes differ from the grid file's"
            )
        # Region's local axes (the full-domain archive's axes are the grid's own)
        if self._src is not None:
            # jsasaki 2026-11-05: review round 2 #6: the axes are Optional until a file is inspected
            assert self._src.lat is not None and self._src.lon is not None
            r_lat, r_lon = self._src.lat, self._src.lon
        else:
            r_lat = np.asarray(self._ds["lat"].values)
            r_lon = np.asarray(self._ds["lon"].values)

        # Region offset in the global grid: where does the region start?
        j0, i0 = self.grid.index_of(r_lat[0], r_lon[0])
        j0_int, i0_int = int(j0), int(i0)
        # ---> jsasaki 2026-11-05: review round 3 #1: the COMPLETE regional axes must be the contiguous
        # block of the grid's axes that starts there (not only the origin)
        if self._src is None:
            blk_lat = self.grid.lat[j0_int : j0_int + r_lat.size]
            blk_lon = self.grid.lon[i0_int : i0_int + r_lon.size]
            if not (
                blk_lat.shape == r_lat.shape
                and blk_lon.shape == r_lon.shape
                and np.allclose(r_lat, blk_lat, rtol=0, atol=AXIS_ATOL)
                and np.allclose(r_lon, blk_lon, rtol=0, atol=AXIS_ATOL)
            ):
                raise ValueError(
                    "the region file's lat/lon axes are not a contiguous block of the grid file's "
                    f"axes (tolerance {AXIS_ATOL:g} degree)"
                )
        # <--- jsasaki 2026-11-05

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
        # jsasaki 2026-11-04: full-domain archive source (FullDomainSource)
        if self._src is not None:
            return self._src.read_profile(name, self._jr, self._ir)
        da = self._ds[name]
        # jsasaki 2026-11-05: review round 3 #2: dimensions by NAME, transposed to the canonical order
        if da.ndim != 4 or set(da.dims) != {"time", "level", "lat", "lon"}:
            raise ValueError(
                f"{name!r} is not a 4-D (time, level, lat, lon) field " f"in region_nc"
            )
        da = da.transpose("time", "level", "lat", "lon")
        # Vectorized fancy indexing over n_obc; one array fetch per node.
        n_time = da.sizes["time"]
        n_level = da.sizes["level"]
        out = np.empty((n_time, n_level, self.n_obc), dtype=np.float32)
        for n in range(self.n_obc):
            vals = da.isel(
                lat=int(self._jr[n]),
                lon=int(self._ir[n]),
            ).values
            # jsasaki 2026-11-05: review round 3 #3: a masked value is NaN, never its underlying data
            out[:, :, n] = _to_float32(_plain_float(vals, name))
        return out

    def _read_variable_2d(self, name: str) -> NDArray[np.float32]:
        """Read a 3-D field (no level axis) at every OBC node and time.

        Returns ``(n_time, n_obc)`` float32.
        """
        # jsasaki 2026-11-04: full-domain archive source (FullDomainSource)
        if self._src is not None:
            return self._src.read_2d(name, self._jr, self._ir)
        da = self._ds[name]
        # jsasaki 2026-11-05: review round 3 #2: dimensions by NAME, transposed to the canonical order
        if da.ndim != 3 or set(da.dims) != {"time", "lat", "lon"}:
            raise ValueError(
                f"{name!r} is not a 3-D (time, lat, lon) field in region_nc"
            )
        da = da.transpose("time", "lat", "lon")
        n_time = da.sizes["time"]
        out = np.empty((n_time, self.n_obc), dtype=np.float32)
        for n in range(self.n_obc):
            vals = da.isel(
                lat=int(self._jr[n]),
                lon=int(self._ir[n]),
            ).values
            # jsasaki 2026-11-05: review round 3 #3: a masked value is NaN, never its underlying data
            out[:, n] = _to_float32(_plain_float(vals, name))
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
        # siglay is negative, h_fvcom positive, so z is negative. siglay is
        # (n_siglay,) for a uniform grid or (n_siglay, n_obc) for a per-node
        # (sigma-z) grid.
        if self.siglay.ndim == 2:
            z_fvcom = (self.siglay * h_fvcom[None, :]).astype(np.float64)
        else:
            z_fvcom = (self.siglay[:, None] * h_fvcom[None, :]).astype(np.float64)
        # JCOPE z columns are also static (per OBC node).
        z_jcope = self.obc_z_jcope.astype(np.float64)  # (n_obc, n_level)

        for n in range(n_obc):
            src_z = z_jcope[n]
            dst_z = z_fvcom[:, n]
            for t in range(n_time):
                out[t, :, n] = vertical_interp(field[t, :, n], src_z, dst_z)
        return out

    # ---------- array builders ---------- #

    def read_raw_profiles(
        self,
        variables: Iterable[str] = ("temp", "salt"),
    ) -> dict[str, NDArray[np.float32]]:
        """Extract the raw JCOPE profiles at the OBC cells — the EXPENSIVE step.

        Returns ``{var: (n_time, N_JCOPE_ACTIVE_LEVELS, n_obc)}`` straight from
        the region archive, BEFORE any vertical interpolation. These depend only
        on the region archive and the OBC-node (lat, lon) — NOT on the FVCOM
        depth / sigma — so they are IDENTICAL for every bathymetry variant of the
        same mesh + OBC list, and can be cached once and reused (see
        ``run_jcope_obc.py --profile-cache-dir``). Feed the result to
        :meth:`build_tsobc_arrays` via ``raw_profiles=`` to skip the region read.
        """
        out: dict[str, NDArray[np.float32]] = {}
        for v in variables:
            if v not in _TSOBC_VAR_MAP:
                raise ValueError(
                    f"variable {v!r} is not supported "
                    f"(allowed: {sorted(_TSOBC_VAR_MAP)})"
                )
            jcope_name = _TSOBC_VAR_MAP[v][0]
            out[v] = self._read_variable_profile(jcope_name)[
                :, :N_JCOPE_ACTIVE_LEVELS, :
            ]
        return out

    def build_tsobc_arrays(
        self,
        variables: Iterable[str] = ("temp", "salt"),
        raw_profiles: dict[str, NDArray[np.float32]] | None = None,
    ) -> dict[str, NDArray[np.float32]]:
        """Return interpolated FVCOM-σ fields without writing a NetCDF.

        Used directly by multi-year callers that need to concatenate one
        year's arrays onto another's before writing a single combined
        output. Single-year callers should use :meth:`write_tsobc` which
        wraps this method.

        If ``raw_profiles`` is given (e.g. loaded from a profile cache built by
        :meth:`read_raw_profiles`), its arrays are used instead of reading the
        region archive, so only the cheap depth-dependent vertical interpolation
        runs. The result is bit-identical to reading the region archive directly.
        """
        vars_list = list(variables)
        for v in vars_list:
            if v not in _TSOBC_VAR_MAP:
                raise ValueError(
                    f"variable {v!r} is not supported "
                    f"(allowed: {sorted(_TSOBC_VAR_MAP)})"
                )
        out: dict[str, NDArray[np.float32]] = {}
        for v in vars_list:
            jcope_name, _vname, _units, _long = _TSOBC_VAR_MAP[v]
            if raw_profiles is not None and v in raw_profiles:
                # ---> jsasaki 2026-11-05: review round 3 #3 #14: masked -> NaN (then refused as
                # non-finite forcing), and exactly (n_time, 46, n_obc): no time or node is dropped
                # jsasaki 2026-11-05: review round 4 #2: float32 kept as it is (no float64 round trip)
                jcope_field = _as_float32(raw_profiles[v], f"raw_profiles[{v!r}]")
                want = (self.mjd.size, N_JCOPE_ACTIVE_LEVELS, self.n_obc)
                if jcope_field.shape != want:
                    raise ValueError(
                        f"raw_profiles[{v!r}] has shape {jcope_field.shape}; expected {want}"
                    )
                # <--- jsasaki 2026-11-05
            else:
                jcope_field = self._read_variable_profile(jcope_name)[
                    :, :N_JCOPE_ACTIVE_LEVELS, :
                ]
            out[v] = self._interp_to_fvcom_sigma(
                jcope_field, self.obc_h_jcope, self.obc_h_fvcom
            )
            # jsasaki 2026-11-05: review round 2 #3: non-finite forcing is an error, not a result
            _check_finite(f"{v} ({jcope_name})", out[v], self.obc_nodes)
        return out

    def build_elevation_array(self) -> NDArray[np.float32]:
        """Return the ``(n_time, n_obc)`` SSH array, no NetCDF write."""
        # jsasaki 2026-11-05: review round 2 #3: non-finite forcing is an error, not a result
        elev = self._read_variable_2d("EGT")
        _check_finite("elevation (EGT)", elev, self.obc_nodes)
        return elev

    # ---------- public writers ---------- #

    def write_tsobc(
        self,
        out_path: str | Path,
        *,
        variables: Iterable[str] = ("temp", "salt"),
        title: str | None = None,
    ) -> Path:
        """Write a FVCOM TIME SERIES OBC TS FILE for a single year."""
        fields = self.build_tsobc_arrays(variables=variables)
        # jsasaki 2026-11-05: review round 2 #18: provenance of the variables of this file only
        names = [_TSOBC_VAR_MAP[v][0] for v in fields]
        return write_tsobc_nc(
            out_path,
            time_mjd=self.mjd,
            obc_nodes=self.obc_nodes,
            obc_h=self.obc_h_fvcom,
            siglev=self.siglev,
            siglay=self.siglay,
            fields=fields,
            title=title,
            # jsasaki 2026-11-04: source label of either archive kind
            # jsasaki 2026-11-05: review round 2 #18: evaluated after the read
            source_files=self.source_lines(names),
        )

    def write_elevation(
        self,
        out_path: str | Path,
        *,
        title: str | None = None,
    ) -> Path:
        """Write a FVCOM TIME SERIES ELEVATION FORCING FILE for one year."""
        # jsasaki 2026-11-05: review round 2 #18: read first, then the provenance of what was read
        elevation = self.build_elevation_array()
        return write_elevation_nc(
            out_path,
            time_mjd=self.mjd,
            obc_nodes=self.obc_nodes,
            elevation=elevation,
            title=title,
            # jsasaki 2026-11-04: source label of either archive kind
            source_files=self.source_lines(["EGT"]),
        )


# ---> jsasaki 2026-11-05: review round 2 #3 #9 #10 #13: input validation shared by the generator and
# the public writers
SIGMA_TOL = 1e-6


def _validate_obc_nodes(obc_nodes: ArrayLike) -> NDArray[np.int32]:
    """Non-empty, 1-D, integral, positive, unique node ids representable as int32."""
    # jsasaki 2026-11-05: review round 3 #3: a masked node id is refused, not unmasked
    if np.ma.is_masked(obc_nodes):
        raise ValueError("obc_nodes has masked entries")
    a = np.asarray(np.ma.getdata(obc_nodes))
    if a.ndim != 1 or a.size == 0:
        raise ValueError("obc_nodes must be a non-empty 1-D array")
    if a.dtype.kind not in "iuf":
        raise ValueError(f"obc_nodes must be integers (got dtype {a.dtype})")
    if a.dtype.kind == "f" and not (np.isfinite(a).all() and (a == np.rint(a)).all()):
        raise ValueError("obc_nodes must be integers")
    # ---> jsasaki 2026-11-05: review round 3 #13: the range check in a representation that keeps the
    # int32 bound exact (float64 for floats; the integer values themselves otherwise), and the
    # converted result is verified against the input
    vals = a.astype(np.float64) if a.dtype.kind == "f" else a
    if (vals < 1).any() or (vals > np.iinfo(np.int32).max).any():
        raise ValueError("obc_nodes must be positive (1-based) int32 node ids")
    if np.unique(a).size != a.size:
        raise ValueError("obc_nodes must be unique")
    out = a.astype(np.int32)
    if not np.array_equal(out.astype(np.float64), a.astype(np.float64)):
        raise ValueError("obc_nodes are not representable as int32 node ids")
    return out
    # <--- jsasaki 2026-11-05


def _validate_sigma(
    siglev: ArrayLike, siglay: ArrayLike, n_obc: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """FVCOM sigma interfaces (siglev) and layer centres (siglay), 1-D or (n_sig, n_obc).

    siglev runs from 0 (surface) to -1 (bed) without increasing; each siglay lies between its two
    interfaces. Collapsed (equal) interfaces are allowed; the layer centre then equals them.
    """
    # jsasaki 2026-11-05: review round 3 #3: masked entries become NaN (refused below)
    sv = _plain_float(siglev, "siglev")
    sl = _plain_float(siglay, "siglay")
    if sv.ndim not in (1, 2) or sl.ndim != sv.ndim:
        raise ValueError("siglev / siglay must both be 1-D or both 2-D (n_sig, n_obc)")
    if sv.ndim == 2 and (sv.shape[1] != n_obc or sl.shape[1] != n_obc):
        raise ValueError(
            f"per-node siglay/siglev second dim must be n_obc={n_obc} "
            f"(got {sl.shape[1]}, {sv.shape[1]})"
        )
    if sv.shape[0] != sl.shape[0] + 1 or sl.shape[0] < 1:
        raise ValueError(
            f"siglev rows ({sv.shape[0]}) must be siglay rows ({sl.shape[0]}) + 1, siglay rows >= 1"
        )
    if not (np.isfinite(sv).all() and np.isfinite(sl).all()):
        raise ValueError("siglev / siglay must be finite")
    if (np.abs(sv[0]) > SIGMA_TOL).any() or (np.abs(sv[-1] + 1.0) > SIGMA_TOL).any():
        raise ValueError("siglev must run from 0 at the surface to -1 at the bed")
    if (np.diff(sv, axis=0) > 0).any():
        raise ValueError(
            "siglev must not increase with layer index (0 surface -> -1 bed)"
        )
    if ((sl > sv[:-1] + SIGMA_TOL) | (sl < sv[1:] - SIGMA_TOL)).any():
        raise ValueError("each siglay must lie between its two siglev interfaces")
    return sv, sl


def _validate_jcope_geometry(
    h: NDArray, zz: NDArray, obc_nodes: NDArray[np.int32]
) -> None:
    """JCOPE depth finite positive; active layer centres finite, negative, deepening, above the bed."""
    h64 = np.asarray(h, dtype=np.float64)
    z64 = np.asarray(zz, dtype=np.float64)
    for n, node in enumerate(obc_nodes):
        if not (np.isfinite(h64[n]) and h64[n] > 0):
            raise ValueError(
                f"JCOPE depth at the cell of OBC node {int(node)} is {h64[n]} (not finite positive)"
            )
        z = z64[n]
        if not (
            np.isfinite(z).all()
            and (z < 0).all()
            and (np.diff(z) < 0).all()
            and z[-1] >= -h64[n] * (1 + 1e-3) - 1e-3
        ):
            raise ValueError(
                f"JCOPE layer-centre depths (ZZ) at the cell of OBC node {int(node)} are not finite, "
                f"negative, strictly deepening and above the bed (h={h64[n]:g}): {z[:3]} ... {z[-3:]}"
            )


def _check_finite(label: str, arr: NDArray, obc_nodes: ArrayLike) -> None:
    """Refuse a non-finite value of a (time, ..., n_obc) array, naming time index and node."""
    nodes = np.asarray(obc_nodes)
    step = 744  # one month of hours at a time: no full-size boolean temporary
    for t0 in range(0, arr.shape[0], step):
        blk = arr[t0 : t0 + step]
        ok = np.isfinite(blk)
        if not ok.all():
            first = np.unravel_index(int(np.argmin(ok)), ok.shape)
            node = int(first[-1])
            raise ValueError(
                f"{label}: non-finite value at time index {t0 + int(first[0])}, node "
                f"{int(nodes[node]) if node < nodes.size else node} (column {node})"
                + (f", layer {int(first[1])}" if len(first) == 3 else "")
            )


def _validate_time(time_mjd: ArrayLike) -> NDArray[np.float64]:
    # jsasaki 2026-11-05: review round 3 #3: masked entries become NaN (refused below)
    t = _plain_float(time_mjd, "time_mjd")
    if t.ndim != 1 or t.size == 0 or not np.isfinite(t).all():
        raise ValueError("time_mjd must be a non-empty, finite 1-D array")
    if (np.diff(t) <= 0).any():
        raise ValueError("time_mjd must be strictly increasing")
    # jsasaki 2026-11-05: review round 5 #9: time is written as f8; its fill marker reads back as missing
    if (t == F8_FILL).any():
        raise ValueError("time_mjd contains the netCDF f8 fill marker")
    return t


# <--- jsasaki 2026-11-05


# ---> jsasaki 2026-11-05: review round 3 #1 #2 #3 #4: masks, float32 conversion and the region schema
# Absolute tolerance (degree) of an axis comparison, the same as for the full-domain archive's axes.
AXIS_ATOL = 1e-6


def _plain_float(values: ArrayLike, label: str) -> NDArray[np.float64]:
    """A plain float64 ndarray; masked entries become NaN, so a mask never passes as data."""
    a = np.ma.asarray(values)
    if a.dtype.kind not in "iuf":
        raise ValueError(f"{label} must be real numbers (got dtype {a.dtype})")
    return np.asarray(np.ma.filled(a.astype(np.float64), np.nan))


def _to_float32(a: NDArray) -> NDArray[np.float32]:
    """Convert to float32; a finite value beyond the float32 range becomes inf (refused as non-finite)."""
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(a).astype(np.float32)


# ---> jsasaki 2026-11-05: review round 4 #2 #7: no float64 round trip of large forcing arrays, and the
# netCDF fill marker refused
# The fill marker of the f4 forcing variables: a finite value equal to it reads back as missing.
F4_FILL = np.float32(nc.default_fillvals["f4"])
# jsasaki 2026-11-05: review round 5 #9: the f8 marker of the time variable (read once at import)
F8_FILL = np.float64(nc.default_fillvals["f8"])


def _as_float32(values: ArrayLike, label: str) -> NDArray[np.float32]:
    """A plain float32 ndarray; masked entries become NaN, so a mask never passes as data.

    An unmasked float32 ndarray is returned as it is (no copy); another real dtype is converted once
    (a finite value beyond the float32 range becomes inf); a copy is made for a mask only when the
    mask has a set entry.
    """
    mask = np.ma.getmask(values)
    data = np.asarray(np.ma.getdata(values))
    if data.dtype.kind not in "iuf":
        raise ValueError(f"{label} must be real numbers (got dtype {data.dtype})")
    if data.dtype != np.float32:
        data = _to_float32(data)
    if mask is not np.ma.nomask and mask.any():
        if np.shares_memory(data, np.ma.getdata(values)):
            data = data.copy()
        data[np.broadcast_to(mask, data.shape)] = np.nan
    return data


def _check_not_fill(label: str, arr: NDArray, obc_nodes: ArrayLike) -> None:
    """Refuse a value equal to the netCDF f4 fill marker, blockwise along time like _check_finite."""
    nodes = np.asarray(obc_nodes)
    step = 744
    for t0 in range(0, arr.shape[0], step):
        hit = arr[t0 : t0 + step] == F4_FILL
        if hit.any():
            first = np.unravel_index(int(np.argmax(hit)), hit.shape)
            node = int(first[-1])
            raise ValueError(
                f"{label}: value {float(F4_FILL):g} (the netCDF fill marker, read back as missing) "
                f"at time index {t0 + int(first[0])}, node "
                f"{int(nodes[node]) if node < nodes.size else node} (column {node})"
                + (f", layer {int(first[1])}" if len(first) == 3 else "")
            )


def _forcing_float32(
    label: str, arr: ArrayLike, obc_nodes: ArrayLike
) -> NDArray[np.float32]:
    """The float32 array that will be written, validated finite (masked, NaN, Inf, overflow refused)
    and free of the fill marker; an unmasked float32 input is checked in place, not copied.
    """
    out = _as_float32(arr, label)
    _check_finite(f"{label} (masked, NaN, Inf or beyond float32)", out, obc_nodes)
    _check_not_fill(label, out, obc_nodes)
    return out


# <--- jsasaki 2026-11-05


def _validate_region_schema(ds: xr.Dataset, path: Path) -> None:
    """Coordinates time/level/lat/lon on their own dimensions; level exactly 0..45 ascending."""
    for c in ("time", "level", "lat", "lon"):
        if c not in ds.variables or ds[c].dims != (c,):
            raise ValueError(
                f"{path}: coordinate {c} is missing or not on dimension {c}"
            )
    lev = np.asarray(ds["level"].values)
    if lev.dtype.kind not in "iuf" or not np.array_equal(
        lev, np.arange(N_JCOPE_ACTIVE_LEVELS)
    ):
        raise ValueError(
            f"{path}: the level coordinate must be 0..{N_JCOPE_ACTIVE_LEVELS - 1} ascending "
            "(JCOPE level index, 0 = surface)"
        )
    for c in ("lat", "lon"):
        v = np.asarray(ds[c].values)
        if v.dtype.kind not in "iuf" or v.size == 0 or not np.isfinite(v).all():
            raise ValueError(f"{path}: coordinate {c} must be non-empty and finite")


# <--- jsasaki 2026-11-05


# ---> jsasaki 2026-11-05: review round 5 #7: undecoded numeric time (e.g. Unix seconds without CF
# units) was read by DatetimeIndex as nanoseconds since 1970 and passed every later check
def _region_time_index(ds: xr.Dataset, path: Path) -> pd.DatetimeIndex:
    """The region file's time coordinate as a DatetimeIndex: decoded, no NaT, strictly increasing."""
    t = np.asarray(ds["time"].values)
    if t.ndim != 1 or t.size == 0 or t.dtype.kind != "M":
        raise ValueError(
            f"{path}: the time coordinate must be a non-empty 1-D array of decoded datetimes "
            f"(got dtype {t.dtype}, shape {t.shape}); open the file with CF time decoding "
            "and give the time variable CF units"
        )
    idx = pd.DatetimeIndex(t)
    if idx.hasnans:
        raise ValueError(f"{path}: the time coordinate has missing (NaT) entries")
    if (np.diff(idx.asi8) <= 0).any():
        raise ValueError(
            f"{path}: the time coordinate must be strictly increasing "
            "(no duplicate or decreasing times)"
        )
    return idx


# <--- jsasaki 2026-11-05


# ---------- module-level NC writers ---------- #


# ---> jsasaki 2026-11-05: review round 5 #10: never change the process umask to read it. os.umask(x)
# is process-wide, so even a set-and-restore pair briefly gives other threads the wrong creation mode.
# Linux reports the umask read-only in /proc/self/status ("Umask:" line, kernel >= 4.7); elsewhere,
# or if that line is absent, the conventional 0o022 is assumed.
_DEFAULT_UMASK = 0o022


def _current_umask() -> int:
    try:
        with open("/proc/self/status", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8) & 0o777
    except (OSError, ValueError, IndexError):
        pass
    return _DEFAULT_UMASK


# <--- jsasaki 2026-11-05


def _atomic_write(out_path: str | Path, write: Callable[[Path], None]) -> Path:
    """Write via a temporary sibling and rename on success, so a failure never destroys an existing file."""
    # jsasaki 2026-11-04: review round 1 #13: the writers truncated the destination before writing
    # ---> jsasaki 2026-11-05: review round 2 #16: a unique temporary created exclusively (mkstemp)
    # in the destination directory; only that file is ever removed. Validation happens before.
    final = Path(out_path).expanduser().resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{final.name}.", suffix=".tmp", dir=final.parent
    )
    os.close(fd)
    tmp = Path(name)
    done = False
    try:
        write(tmp)
        os.chmod(tmp, 0o666 & ~_current_umask())
        os.replace(tmp, final)
        done = True
    finally:
        if not done:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
    return final
    # <--- jsasaki 2026-11-05


def _format_history(source_files: Iterable[str]) -> str:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    sources = ", ".join(source_files)
    return f"Created {when} UTC from {sources}"


def write_tsobc_nc(
    out_path: str | Path,
    *,
    time_mjd: NDArray[np.float64],
    obc_nodes: NDArray[np.int32],
    obc_h: NDArray[np.float32],
    siglev: NDArray[np.float64],
    siglay: NDArray[np.float64],
    fields: dict[str, NDArray[np.float32]],
    title: str | None = None,
    source_files: Iterable[str] = (),
) -> Path:
    """Write a FVCOM TIME SERIES OBC TS FILE from pre-computed arrays.

    ``fields`` maps short variable keys (``"temp"`` / ``"salt"``) to
    ``(n_time, n_siglay, n_obc)`` float32 arrays already interpolated to
    the FVCOM σ grid (typically produced by
    :meth:`JcopeObcGenerator.build_tsobc_arrays`). Every input is validated
    before anything is written; an existing file is replaced only on success.
    """
    # ---> jsasaki 2026-11-05: review round 2 #3 #9 #13 #16: validate everything before the
    # temporary file exists
    time_mjd = _validate_time(time_mjd)
    obc_nodes = _validate_obc_nodes(obc_nodes)
    n_time = int(time_mjd.size)
    n_obc = int(obc_nodes.size)
    # jsasaki 2026-11-05: review round 3 #3 #4: masked -> NaN, overflow -> inf, both refused below
    obc_h = _to_float32(_plain_float(obc_h, "obc_h"))
    if obc_h.shape != (n_obc,) or not (np.isfinite(obc_h).all() and (obc_h > 0).all()):
        raise ValueError(f"obc_h must be {n_obc} finite positive depths")
    # ---> jsasaki 2026-11-05: review round 5 #9: the f4 fill marker reads back as masked. siglev/siglay
    # are confined to [-1, 0] by _validate_sigma and time is checked in _validate_time.
    if (obc_h == F4_FILL).any():
        raise ValueError(
            f"obc_h: value {float(F4_FILL):g} is the netCDF fill marker (read back as missing)"
        )
    # <--- jsasaki 2026-11-05
    # siglay/siglev may be 1-D (uniform, broadcast across nodes) or 2-D
    # (n_sig, n_obc) for a per-node sigma-z grid — take the layer count from
    # the leading axis either way.
    siglev_arr, siglay_arr = _validate_sigma(siglev, siglay, n_obc)
    n_siglay = int(siglay_arr.shape[0])
    n_siglev = int(siglev_arr.shape[0])
    if not fields:
        raise ValueError("fields is empty")
    # jsasaki 2026-11-05: review round 3 #3 #4: the validated float32 arrays are the ones written
    fields32: dict[str, NDArray[np.float32]] = {}
    for v, arr in fields.items():
        if v not in _TSOBC_VAR_MAP:
            raise ValueError(
                f"variable {v!r} is not supported (allowed: {sorted(_TSOBC_VAR_MAP)})"
            )
        if np.shape(arr) != (n_time, n_siglay, n_obc):
            raise ValueError(
                f"field {v!r} has shape {np.shape(arr)}; "
                f"expected ({n_time}, {n_siglay}, {n_obc})"
            )
        fields32[v] = _forcing_float32(f"field {v!r}", arr, obc_nodes)
    sources = list(source_files)

    def _write(path: Path) -> None:
        _write_tsobc_file(
            path,
            time_mjd,
            obc_nodes,
            obc_h,
            siglev_arr,
            siglay_arr,
            n_siglev,
            n_siglay,
            fields32,
            title,
            sources,
        )

    return _atomic_write(out_path, _write)
    # <--- jsasaki 2026-11-05


def _write_tsobc_file(
    out_path: Path,
    time_mjd: NDArray[np.float64],
    obc_nodes: NDArray[np.int32],
    obc_h: NDArray[np.float32],
    siglev_arr: NDArray[np.float64],
    siglay_arr: NDArray[np.float64],
    n_siglev: int,
    n_siglay: int,
    fields: dict[str, NDArray[np.float32]],
    title: str | None,
    source_files: list[str],
) -> None:
    n_obc = int(obc_nodes.size)
    with nc.Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("nobc", n_obc)
        ds.createDimension("siglay", n_siglay)
        ds.createDimension("siglev", n_siglev)
        ds.createDimension("DateStrLen", 26)

        v_time = ds.createVariable("time", "f8", ("time",))
        v_time.long_name = "time"
        v_time.units = "days since 1858-11-17 00:00:00"
        v_time.format = "modified julian day (MJD)"
        v_time.time_zone = "UTC"
        v_time[:] = time_mjd

        v_nodes = ds.createVariable("obc_nodes", "i4", ("nobc",))
        v_nodes.long_name = "Open Boundary Node Number"
        v_nodes.units = "-"
        v_nodes[:] = obc_nodes

        v_h = ds.createVariable("obc_h", "f4", ("nobc",))
        v_h.long_name = "Open Boundary Depth"
        v_h.units = "m"
        v_h[:] = obc_h

        v_siglev = ds.createVariable("siglev", "f4", ("siglev", "nobc"))
        v_siglev.long_name = "ocean_sigma/general_coordinate"
        v_siglev.units = "-"
        v_siglev[:] = (
            siglev_arr
            if siglev_arr.ndim == 2
            else np.broadcast_to(siglev_arr[:, None], (n_siglev, n_obc))
        ).astype(np.float32)

        v_siglay = ds.createVariable("siglay", "f4", ("siglay", "nobc"))
        v_siglay.long_name = "ocean_sigma/general_coordinate"
        v_siglay.units = "-"
        v_siglay[:] = (
            siglay_arr
            if siglay_arr.ndim == 2
            else np.broadcast_to(siglay_arr[:, None], (n_siglay, n_obc))
        ).astype(np.float32)

        for v, arr in fields.items():
            _jname, vname, units, long_name = _TSOBC_VAR_MAP[v]
            v_var = ds.createVariable(vname, "f4", ("time", "siglay", "nobc"))
            v_var.long_name = long_name
            v_var.units = units
            v_var[:] = arr

        ds.title = title or "Open boundary temperature and salinity from JCOPE-T DA"
        ds.type = "FVCOM TIME SERIES OBC TS FILE"
        ds.source = (
            "JCOPE-T DA (JAMSTEC), interpolated to FVCOM OBC sigma "
            "layers via xfvcom.io.JcopeObcGenerator"
        )
        ds.history = _format_history(source_files)


def write_elevation_nc(
    out_path: str | Path,
    *,
    time_mjd: NDArray[np.float64],
    obc_nodes: NDArray[np.int32],
    elevation: NDArray[np.float32],
    title: str | None = None,
    source_files: Iterable[str] = (),
) -> Path:
    """Write a FVCOM TIME SERIES ELEVATION FORCING FILE from pre-computed arrays.

    Every input is validated before anything is written; an existing file is replaced only on success.
    """
    # ---> jsasaki 2026-11-05: review round 2 #3 #13 #16: validate everything before the temporary
    # file exists
    time_mjd = _validate_time(time_mjd)
    obc_nodes = _validate_obc_nodes(obc_nodes)
    n_time = int(time_mjd.size)
    n_obc = int(obc_nodes.size)
    if np.shape(elevation) != (n_time, n_obc):
        raise ValueError(
            f"elevation has shape {np.shape(elevation)}; expected ({n_time}, {n_obc})"
        )
    # jsasaki 2026-11-05: review round 3 #3 #4: the validated float32 array is the one written
    elevation32 = _forcing_float32("elevation", elevation, obc_nodes)
    sources = list(source_files)

    def _write(path: Path) -> None:
        _write_elevation_file(path, time_mjd, obc_nodes, elevation32, title, sources)

    return _atomic_write(out_path, _write)
    # <--- jsasaki 2026-11-05


def _write_elevation_file(
    out_path: Path,
    time_mjd: NDArray[np.float64],
    obc_nodes: NDArray[np.int32],
    elevation: NDArray[np.float32],
    title: str | None,
    source_files: list[str],
) -> None:
    n_obc = int(obc_nodes.size)
    with nc.Dataset(out_path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("nobc", n_obc)
        ds.createDimension("DateStrLen", 26)

        v_nodes = ds.createVariable("obc_nodes", "i4", ("nobc",))
        v_nodes.long_name = "Open Boundary Node Number"
        v_nodes.units = "-"
        v_nodes[:] = obc_nodes

        v_time = ds.createVariable("time", "f8", ("time",))
        v_time.long_name = "time"
        v_time.units = "days since 1858-11-17 00:00:00"
        v_time.format = "modified julian day (MJD)"
        v_time.time_zone = "UTC"
        v_time[:] = time_mjd

        v_elev = ds.createVariable("elevation", "f4", ("time", "nobc"))
        v_elev.long_name = "Open Boundary Elevation"
        v_elev.units = "meters"
        v_elev[:] = elevation

        ds.title = title or "Open boundary elevation from JCOPE-T DA"
        ds.type = "FVCOM TIME SERIES ELEVATION FORCING FILE"
        ds.source = (
            "JCOPE-T DA (JAMSTEC), at OBC node nearest ocean cell, "
            "produced by xfvcom.io.JcopeObcGenerator"
        )
        ds.history = _format_history(source_files)

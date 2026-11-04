# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Point-profile reader for the JCOPE-T DA full-domain archive (one NetCDF per day).

The jcopetda repository archives the whole 1190 x 1190 x 46 domain as daily files
``<root>/YYYY/jcopetda_YYYYMMDD.nc`` (int16 packed, 24 hourly steps, variables ``TT ST UT VT EGT``,
dimensions ``time level lat lon``; format: ``jcopetda/docs/full_domain_format.md``). This module
reads the columns an open-boundary builder needs -- a few hundred cells -- out of those files without
loading a field: the cells are grouped by HDF5 chunk and every chunk group is read once per day.

:class:`FullDomainSource` is accepted by :class:`xfvcom.io.JcopeObcGenerator` in place of a
``jcopetda_region_YYYY.nc`` path.

Records that the archive declares missing
-----------------------------------------
A day file lists the hours whose raw record could not be obtained in its global attribute
``missing_fields`` (``ST_2020121517``); those hours are fill (NaN here). ``missing="error"`` (default)
refuses to build from them, ``missing="interpolate"`` fills them by linear interpolation in time of
the columns that were read (a stand-in; it smooths whatever varies faster than the gap). The archive
tree ``$DATA_DIR/jcopetda_provisional`` already contains such interpolated day files, so for it the
default works unchanged.
"""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

__all__ = ["FullDomainSource"]

FILL = -32768  # int16 fill of the packed fields
HOURS_PER_DAY = 24
N_LEVELS = 46


class FullDomainSource:
    """Read hourly columns of the full-domain archive between two dates (inclusive).

    Parameters
    ----------
    root
        The archive root holding ``YYYY/jcopetda_YYYYMMDD.nc``.
    start, end
        First and last day (any ``pandas.Timestamp`` input); all 24 hours of both are used.
    missing
        ``"error"`` or ``"interpolate"`` (see the module docstring).

    The object keeps no file open and is picklable (it can be sent to joblib workers).
    """

    def __init__(
        self, root: str | Path, start: str, end: str, *, missing: str = "error"
    ) -> None:
        if missing not in ("error", "interpolate"):
            raise ValueError(
                f"missing must be 'error' or 'interpolate' (got {missing!r})"
            )
        self.root = Path(root).expanduser()
        self.missing = missing
        d0, d1 = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        if d1 < d0:
            raise ValueError(f"end {end} is before start {start}")
        self.start, self.end = d0, d1
        days = pd.date_range(d0, d1, freq="D")
        self.paths = [self.root / f"{d:%Y}" / f"jcopetda_{d:%Y%m%d}.nc" for d in days]
        absent = [p for p in self.paths if not p.is_file()]
        if absent:
            raise FileNotFoundError(
                f"{len(absent)} day file(s) missing under {self.root}: first {absent[0]} "
                f"(the archive covers 2018-06-04 .. 2025-12-31)"
            )
        times: list[pd.DatetimeIndex] = []
        self._missing: dict[str, list[int]] = {}
        for d, p in enumerate(self.paths):
            with nc.Dataset(p) as ds:
                if getattr(ds, "product", "") != "full_domain":
                    raise ValueError(f"{p} is not a full-domain day file")
                tv = ds.variables["time"]
                t = nc.num2date(
                    tv[:], tv.units, tv.calendar, only_use_cftime_datetimes=False
                )
                idx = pd.DatetimeIndex(
                    [pd.Timestamp(x.replace(microsecond=0)) for x in t]
                )
                if len(idx) != HOURS_PER_DAY or idx[0] != days[d]:
                    raise ValueError(
                        f"{p}: expected 24 hourly steps from {days[d]:%Y-%m-%d} 00Z"
                    )
                times.append(idx)
                if d == 0:
                    self.lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
                    self.lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
                for item in [
                    x for x in str(getattr(ds, "missing_fields", "")).split(",") if x
                ]:
                    var, stamp = item.split("_")
                    hour = pd.Timestamp(
                        stamp[:4]
                        + "-"
                        + stamp[4:6]
                        + "-"
                        + stamp[6:8]
                        + f" {stamp[8:10]}:00"
                    )
                    self._missing.setdefault(var, []).append(
                        d * HOURS_PER_DAY + int(hour.hour)
                    )
        self.time_index = times[0].append(times[1:]) if len(times) > 1 else times[0]
        self.n_time = len(self.time_index)

    # ---- identity ---- #

    @property
    def name(self) -> str:
        return f"{self.root.name}[{self.start:%Y%m%d}-{self.end:%Y%m%d}]"

    def exists(self) -> bool:
        return True

    # ---- readers ---- #

    def read_profile(
        self, name: str, jj: ArrayLike, ii: ArrayLike
    ) -> NDArray[np.float32]:
        """``(n_time, 46, n_cells)`` float32, NaN on land / fill, for a 4-D variable (TT ST UT VT)."""
        if name == "EGT":
            raise ValueError("EGT is 2-D; use read_2d")
        return self._read(name, jj, ii, True)

    def read_2d(self, name: str, jj: ArrayLike, ii: ArrayLike) -> NDArray[np.float32]:
        """``(n_time, n_cells)`` float32 for a 3-D variable (EGT)."""
        return self._read(name, jj, ii, False)

    def _read(self, name, jj, ii, is3d):
        jj = np.asarray(jj, dtype=np.int64)
        ii = np.asarray(ii, dtype=np.int64)
        n = jj.size
        gaps = sorted(self._missing.get(name, []))
        if gaps and self.missing == "error":
            when = ", ".join(f"{self.time_index[g]:%Y-%m-%d %H}Z" for g in gaps[:5])
            raise ValueError(
                f"{name}: {len(gaps)} hour(s) of {self.root} are declared missing ({when}...). Use the "
                f"archive tree $DATA_DIR/jcopetda_provisional (interpolated day files) or pass "
                f"missing='interpolate'."
            )
        shape = (self.n_time, N_LEVELS, n) if is3d else (self.n_time, n)
        out = np.full(shape, np.nan, dtype=np.float32)
        for d, path in enumerate(self.paths):
            t0 = d * HOURS_PER_DAY
            with nc.Dataset(path) as ds:
                v = ds.variables[name]
                v.set_auto_maskandscale(False)
                scale = float(v.scale_factor)
                ch = v.chunking()
                cj, ci = (
                    (ch[-2], ch[-1])
                    if ch != "contiguous"
                    else (v.shape[-2], v.shape[-1])
                )
                key = (jj // cj) * 1_000_000 + ii // ci
                for k in np.unique(key):
                    sel = np.flatnonzero(key == k)
                    j0, j1 = jj[sel].min(), jj[sel].max() + 1
                    i0, i1 = ii[sel].min(), ii[sel].max() + 1
                    blk = v[:, :, j0:j1, i0:i1] if is3d else v[:, j0:j1, i0:i1]
                    q = blk[..., jj[sel] - j0, ii[sel] - i0]  # (24, [46,] len(sel))
                    val = np.where(q == FILL, np.nan, q.astype(np.float64) * scale)
                    out[t0 : t0 + HOURS_PER_DAY, ..., sel] = val.astype(np.float32)
        if gaps:  # missing == "interpolate"
            good = np.setdiff1d(np.arange(self.n_time), gaps)
            if good.size < 2 or gaps[0] < good[0] or gaps[-1] > good[-1]:
                raise ValueError(
                    f"{name}: a missing hour has no valid neighbour on both sides"
                )
            flat = out.reshape(self.n_time, -1)
            for g in gaps:
                a = good[np.searchsorted(good, g) - 1]
                b = good[np.searchsorted(good, g)]
                w = (g - a) / (b - a)
                flat[g] = (1.0 - w) * flat[a] + w * flat[b]
        return out

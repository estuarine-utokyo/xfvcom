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
the columns that were read (a stand-in; it smooths whatever varies faster than the gap; the neighbouring
day files outside the requested period are read when a gap sits at its edge). The archive tree
``$DATA_DIR/jcopetda_provisional`` already contains such interpolated day files (global attribute
``provisional_fill``), so for it the default works unchanged; the declaration is reported in
:meth:`FullDomainSource.provenance` and ends up in the history of the files written from it.
"""

from __future__ import annotations

import re
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

__all__ = ["FullDomainSource"]

FILL = -32768  # int16 fill of the packed fields
HOURS_PER_DAY = 24
N_LEVELS = 46
FORMAT_VERSION = 1
VARIABLES = ("TT", "ST", "UT", "VT", "EGT")
_DECL = re.compile(r"^(TT|ST|UT|VT|EGT)_(\d{10})$")


def _utc_day(value) -> pd.Timestamp:
    """A calendar day as a naive UTC Timestamp (aware inputs are converted to UTC first)."""
    t = pd.Timestamp(value)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.normalize()


class FullDomainSource:
    """Read hourly columns of the full-domain archive between two UTC dates (inclusive).

    Parameters
    ----------
    root
        The archive root holding ``YYYY/jcopetda_YYYYMMDD.nc``.
    start, end
        First and last day (any ``pandas.Timestamp`` input; timezone-aware values are converted to
        UTC); all 24 hours of both are used.
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
        d0, d1 = _utc_day(start), _utc_day(end)
        if d1 < d0:
            raise ValueError(f"end {end} is before start {start}")
        self.start, self.end = d0, d1

        def path_of(d: pd.Timestamp) -> Path:
            return self.root / f"{d:%Y}" / f"jcopetda_{d:%Y%m%d}.nc"

        absent = [
            path_of(d)
            for d in pd.date_range(d0, d1, freq="D")
            if not path_of(d).is_file()
        ]
        if absent:
            raise FileNotFoundError(
                f"{len(absent)} day file(s) missing under {self.root}: first {absent[0]} "
                f"(the archive covers 2018-06-04 .. 2025-12-31)"
            )
        # In "interpolate" mode a gap at the edge of the period needs the neighbouring day.
        e0, e1 = d0, d1
        if missing == "interpolate":
            if path_of(d0 - pd.Timedelta(days=1)).is_file():
                e0 = d0 - pd.Timedelta(days=1)
            if path_of(d1 + pd.Timedelta(days=1)).is_file():
                e1 = d1 + pd.Timedelta(days=1)
        days = pd.date_range(e0, e1, freq="D")
        self.paths = [path_of(d) for d in days]
        self._i0 = int((d0 - e0).days) * HOURS_PER_DAY
        self.n_time = int((d1 - d0).days + 1) * HOURS_PER_DAY
        self.notes: list[str] = []
        self._missing: dict[str, list[int]] = {}
        self._schema: dict[str, float] = {}
        times: list[pd.DatetimeIndex] = []
        for d, p in enumerate(self.paths):
            times.append(self._inspect(p, days[d], d))
        full = times[0].append(times[1:]) if len(times) > 1 else times[0]
        self._time_full = full
        self.time_index = full[self._i0 : self._i0 + self.n_time]
        if missing == "error":  # only declarations inside the requested period matter
            lo, hi = self._i0, self._i0 + self.n_time
            for var in list(self._missing):
                self._missing[var] = [g for g in self._missing[var] if lo <= g < hi]
                if not self._missing[var]:
                    del self._missing[var]

    def _inspect(self, p: Path, day: pd.Timestamp, d: int) -> pd.DatetimeIndex:
        """Validate one day file against the supported format and collect its declarations."""
        with nc.Dataset(p) as ds:
            if getattr(ds, "product", "") != "full_domain":
                raise ValueError(f"{p} is not a full-domain day file")
            if int(np.asarray(getattr(ds, "format_version", -1))) != FORMAT_VERSION:
                raise ValueError(f"{p}: unsupported format_version")
            tv = ds.variables["time"]
            t = nc.num2date(
                tv[:], tv.units, tv.calendar, only_use_cftime_datetimes=False
            )
            idx = pd.DatetimeIndex([pd.Timestamp(x) for x in t])
            want = pd.date_range(day, periods=HOURS_PER_DAY, freq="h")
            if len(idx) != HOURS_PER_DAY or not idx.equals(want):
                raise ValueError(
                    f"{p}: the time axis is not the 24 hourly UTC steps of {day:%Y-%m-%d}"
                )
            lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
            lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
            lev = np.asarray(ds.variables["level"][:])
            if d == 0:
                self.lat, self.lon = lat, lon
            elif not (np.array_equal(lat, self.lat) and np.array_equal(lon, self.lon)):
                raise ValueError(
                    f"{p}: horizontal axes differ from the first day file's"
                )
            if not np.array_equal(lev, np.arange(N_LEVELS)):
                raise ValueError(f"{p}: level axis is not 0..{N_LEVELS - 1}")
            for name in VARIABLES:
                v = ds.variables[name]
                dims = (
                    ("time", "level", "lat", "lon")
                    if name != "EGT"
                    else ("time", "lat", "lon")
                )
                if v.dimensions != dims or v.dtype != np.int16:
                    raise ValueError(f"{p}: {name} is not int16 with dimensions {dims}")
                fill = (
                    v.getncattr("_FillValue") if "_FillValue" in v.ncattrs() else None
                )
                off = (
                    float(v.getncattr("add_offset"))
                    if "add_offset" in v.ncattrs()
                    else 0.0
                )
                if fill is None or int(fill) != FILL or off != 0.0:
                    raise ValueError(
                        f"{p}: {name} packing is not fill {FILL}, add_offset 0"
                    )
                scale = float(v.getncattr("scale_factor"))
                if self._schema.setdefault(name, scale) != scale:
                    raise ValueError(
                        f"{p}: {name} scale_factor differs between day files"
                    )
            seen: set[str] = set()
            for item in [
                x for x in str(getattr(ds, "missing_fields", "")).split(",") if x
            ]:
                m = _DECL.match(item)
                if m is None or item in seen:
                    raise ValueError(
                        f"{p}: malformed or repeated missing_fields entry {item!r}"
                    )
                seen.add(item)
                stamp = pd.Timestamp(m.group(2)[:8] + " " + m.group(2)[8:] + ":00")
                if stamp not in idx:
                    raise ValueError(
                        f"{p}: missing_fields entry {item} is not an hour of this file"
                    )
                self._missing.setdefault(m.group(1), []).append(
                    d * HOURS_PER_DAY + int(np.flatnonzero(idx == stamp)[0])
                )
            prov = str(getattr(ds, "provisional_fill", ""))
            if prov:
                self.notes.append(
                    f"{p.name}: provisional_fill {prov} (time-interpolated stand-in)"
                )
        return idx

    # ---- identity ---- #

    @property
    def name(self) -> str:
        return f"{self.root.name}[{self.start:%Y%m%d}-{self.end:%Y%m%d}]"

    def exists(self) -> bool:
        return True

    def provenance(self) -> list[str]:
        """Lines for the history of a file written from this source."""
        lines = [self.name, f"missing-record policy: {self.missing}"]
        if self.missing == "interpolate":
            for var, gaps in sorted(self._missing.items()):
                when = ", ".join(f"{self._time_full[g]:%Y-%m-%d %H}Z" for g in gaps)
                lines.append(f"{var} linearly interpolated in time at {when}")
        lines.extend(self.notes)
        return lines

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

    def _indices(self, jj: ArrayLike, ii: ArrayLike) -> tuple[NDArray, NDArray]:
        a, b = np.asarray(jj), np.asarray(ii)
        if a.ndim != 1 or a.shape != b.shape or a.size == 0:
            raise ValueError("jj and ii must be non-empty 1-D arrays of equal length")
        if not (
            np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer)
        ):
            if not (np.all(a == np.rint(a)) and np.all(b == np.rint(b))):
                raise ValueError("cell indices must be integers")
        a, b = a.astype(np.int64), b.astype(np.int64)
        if (
            a.min() < 0
            or a.max() >= self.lat.size
            or b.min() < 0
            or b.max() >= self.lon.size
        ):
            raise ValueError("a cell index lies outside the archive grid")
        return a, b

    def _read(self, name, jj, ii, is3d):
        jj, ii = self._indices(jj, ii)
        n = jj.size
        gaps = sorted(self._missing.get(name, []))
        if gaps and self.missing == "error":
            when = ", ".join(f"{self._time_full[g]:%Y-%m-%d %H}Z" for g in gaps[:5])
            raise ValueError(
                f"{name}: {len(gaps)} hour(s) of {self.root} are declared missing ({when}...). Use the "
                f"archive tree $DATA_DIR/jcopetda_provisional (interpolated day files) or pass "
                f"missing='interpolate'."
            )
        n_full = len(self._time_full)
        shape = (n_full, N_LEVELS, n) if is3d else (n_full, n)
        out = np.full(shape, np.nan, dtype=np.float32)
        scale = self._schema[name]
        for d, path in enumerate(self.paths):
            t0 = d * HOURS_PER_DAY
            with nc.Dataset(path) as ds:
                v = ds.variables[name]
                v.set_auto_maskandscale(False)
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
            good = np.setdiff1d(np.arange(n_full), gaps)
            flat = out.reshape(n_full, -1)
            for g in gaps:
                pos = np.searchsorted(good, g)
                if pos == 0 or pos == good.size:
                    raise ValueError(
                        f"{name}: a missing hour has no valid neighbour on both sides"
                    )
                a, b = good[pos - 1], good[pos]
                w = (g - a) / (b - a)
                flat[g] = (1.0 - w) * flat[a] + w * flat[b]
        out = out[self._i0 : self._i0 + self.n_time]
        # A sea cell has data at every hour: an all-NaN column means the cell is land in the archive.
        empty = np.isnan(out).all(axis=1) if is3d else np.isnan(out)
        if empty.any():
            bad = np.unique(np.nonzero(empty)[-1])
            cells = ", ".join(f"(j={jj[b]}, i={ii[b]})" for b in bad[:5])
            raise ValueError(
                f"{name}: no data at {len(bad)} requested cell(s), e.g. {cells}: land (or fill) in the "
                f"archive although the grid file may call it ocean; move the open-boundary node or "
                f"choose another cell"
            )
        return out

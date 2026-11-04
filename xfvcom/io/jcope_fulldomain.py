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
refuses to build from them, ``missing="interpolate"`` fills the declared hours INSIDE the requested
period by linear interpolation in time between the nearest valid hours (a stand-in; it smooths
whatever varies faster than the gap). When such a neighbour lies outside the period, the day files
beyond its edge are read, one day at a time and at most ``MAX_HALO_DAYS`` days out; declarations in
those extra days that the requested period does not depend on are ignored. The archive tree
``$DATA_DIR/jcopetda_provisional`` already contains such interpolated day files (global attribute
``provisional_fill``), so for it the default works unchanged; the declaration is reported in
:meth:`FullDomainSource.provenance` and ends up in the history of the files written from it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Iterable

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
# jsasaki 2026-11-05: review round 2 #7: the only supported packing steps
SCALES = {"TT": 0.01, "ST": 0.01, "UT": 0.001, "VT": 0.001, "EGT": 0.001}
# jsasaki 2026-11-05: review round 2 #7: decoding attributes this reader does not implement
UNSUPPORTED_ATTRS = (
    "_Unsigned",
    "missing_value",
    "valid_min",
    "valid_max",
    "valid_range",
)
# jsasaki 2026-11-05: review round 2 #8: how far beyond the period a valid neighbour may be sought
MAX_HALO_DAYS = 7
_DECL = re.compile(r"^(TT|ST|UT|VT|EGT)_(\d{10})$")
_HOUR = pd.Timedelta(hours=1)
_DAY = pd.Timedelta(days=1)


def _utc_day(value) -> pd.Timestamp:
    """A calendar day as a naive UTC Timestamp (aware inputs are converted to UTC first)."""
    t = pd.Timestamp(value)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.normalize()


def _scalar_attr(value) -> np.ndarray | None:
    """The attribute as a 0-d array, or None when it is not a single value."""
    a = np.asarray(value)
    if a.size != 1:
        return None
    return a.reshape(())


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
        # ---> jsasaki 2026-11-05: review round 2 #8 #18: declarations are kept as timestamps, the
        # halo days are added only while a requested gap still lacks a valid neighbour, and the
        # notes / repairs reported are those of the requested period only
        self.lat: NDArray[np.float64] | None = None
        self.lon: NDArray[np.float64] | None = None
        self._schema: dict[str, float] = {}
        self._decl: dict[str, set[pd.Timestamp]] = {}
        self._notes_by_day: dict[pd.Timestamp, list[str]] = {}
        idx_by_day: dict[pd.Timestamp, pd.DatetimeIndex] = {}
        for day in pd.date_range(d0, d1, freq="D"):
            idx_by_day[day] = self._inspect(path_of(day), day)
        lo, hi = d0, d1
        if missing == "interpolate":
            lo, hi = self._extend_halo(d0, d1, path_of, idx_by_day)
        days = pd.date_range(lo, hi, freq="D")
        self.paths = [path_of(d) for d in days]
        self._i0 = int((d0 - lo).days) * HOURS_PER_DAY
        self.n_time = int((d1 - d0).days + 1) * HOURS_PER_DAY
        full = idx_by_day[days[0]].append([idx_by_day[d] for d in days[1:]])
        self._time_full = full
        self.time_index = full[self._i0 : self._i0 + self.n_time]
        pos = {t: k for k, t in enumerate(full)}
        t_lo, t_hi = d0, d1 + _DAY
        # every declared hour of the loaded days (never a valid interpolation neighbour) ...
        self._declared: dict[str, list[int]] = {
            v: sorted(pos[t] for t in s if t in pos) for v, s in self._decl.items()
        }
        # ... and the declared hours inside the requested period (to refuse or to repair)
        self._missing: dict[str, list[int]] = {}
        for v, s in self._decl.items():
            g = sorted(pos[t] for t in s if t_lo <= t < t_hi)
            if g:
                self._missing[v] = g
        self.notes: list[str] = [
            n
            for d in pd.date_range(d0, d1, freq="D")
            for n in self._notes_by_day.get(d, [])
        ]
        self._read_vars: set[str] = set()
        self._repaired: dict[str, set[pd.Timestamp]] = {}
        # <--- jsasaki 2026-11-05

    def _extend_halo(
        self,
        d0: pd.Timestamp,
        d1: pd.Timestamp,
        path_of: Callable[[pd.Timestamp], Path],
        idx_by_day: dict[pd.Timestamp, pd.DatetimeIndex],
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Add day files beyond the period until every requested gap has valid neighbours."""
        lo, hi = d0, d1
        while True:
            need = 0
            for var, stamps in self._decl.items():
                for s in sorted(t for t in stamps if d0 <= t < d1 + _DAY):
                    for step in (-_HOUR, _HOUR):
                        t = s + step
                        while t in stamps:
                            t += step
                        if t < lo:
                            need = -1
                        elif t >= hi + _DAY:
                            need = 1
                        if need:
                            break
                    if need:
                        break
                if need:
                    break
            if not need:
                return lo, hi
            day = lo - _DAY if need < 0 else hi + _DAY
            edge = d0 if need < 0 else d1
            if abs((day - edge).days) > MAX_HALO_DAYS:
                raise ValueError(
                    f"{var}: the missing hour {s:%Y-%m-%d %H}Z has no valid neighbour within "
                    f"{MAX_HALO_DAYS} days outside the requested period"
                )
            p = path_of(day)
            if not p.is_file():
                raise ValueError(
                    f"{var}: the missing hour {s:%Y-%m-%d %H}Z has no valid neighbour on both sides "
                    f"(its interpolation needs the day file {p}, which does not exist)"
                )
            idx_by_day[day] = self._inspect(p, day)
            if need < 0:
                lo = day
            else:
                hi = day

    def _inspect(self, p: Path, day: pd.Timestamp) -> pd.DatetimeIndex:
        """Validate one day file against the supported format and collect its declarations."""
        with nc.Dataset(p) as ds:
            if getattr(ds, "product", "") != "full_domain":
                raise ValueError(f"{p} is not a full-domain day file")
            # jsasaki 2026-11-05: review round 2 #7: an integer 1, compared without truncation
            fv = _scalar_attr(getattr(ds, "format_version", None))
            if fv is None or fv.dtype.kind not in "iu" or int(fv) != FORMAT_VERSION:
                raise ValueError(f"{p}: unsupported format_version")
            # ---> jsasaki 2026-11-05: review round 2 #6: dimension lengths and coordinate dimensions
            dims = ds.dimensions
            for dname, size in (("time", HOURS_PER_DAY), ("level", N_LEVELS)):
                if dname not in dims or len(dims[dname]) != size:
                    raise ValueError(f"{p}: dimension {dname} must have length {size}")
            for cname in ("time", "level", "lat", "lon"):
                if cname not in dims or cname not in ds.variables:
                    raise ValueError(f"{p}: coordinate {cname} is missing")
                if ds.variables[cname].dimensions != (cname,):
                    raise ValueError(
                        f"{p}: coordinate variable {cname} is not on dimension {cname}"
                    )
            # <--- jsasaki 2026-11-05
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
            # jsasaki 2026-11-05: review round 2 #6: the first file read defines the axes
            if self.lat is None or self.lon is None:
                self.lat, self.lon = lat, lon
            elif not (np.array_equal(lat, self.lat) and np.array_equal(lon, self.lon)):
                raise ValueError(
                    f"{p}: horizontal axes differ from the first day file's"
                )
            if not np.array_equal(lev, np.arange(N_LEVELS)):
                raise ValueError(f"{p}: level axis is not 0..{N_LEVELS - 1}")
            for name in VARIABLES:
                v = ds.variables[name]
                dims_v = (
                    ("time", "level", "lat", "lon")
                    if name != "EGT"
                    else ("time", "lat", "lon")
                )
                # jsasaki 2026-11-05: review round 2 #6: the complete payload shape, not only names
                shape = (
                    (HOURS_PER_DAY, N_LEVELS, lat.size, lon.size)
                    if name != "EGT"
                    else (HOURS_PER_DAY, lat.size, lon.size)
                )
                if v.dimensions != dims_v or v.dtype != np.int16 or v.shape != shape:
                    raise ValueError(
                        f"{p}: {name} is not int16 with dimensions {dims_v} and shape {shape}"
                    )
                # ---> jsasaki 2026-11-05: review round 2 #7: exact packing contract
                attrs = set(v.ncattrs())
                extra = sorted(attrs.intersection(UNSUPPORTED_ATTRS))
                if extra:
                    raise ValueError(
                        f"{p}: {name} carries unsupported decoding attribute(s) {extra}"
                    )
                fill = (
                    _scalar_attr(v.getncattr("_FillValue"))
                    if "_FillValue" in attrs
                    else None
                )
                off = (
                    _scalar_attr(v.getncattr("add_offset"))
                    if "add_offset" in attrs
                    else np.asarray(0.0)
                )
                if (
                    fill is None
                    or fill.dtype != np.int16
                    or int(fill) != FILL
                    or off is None
                    or off.dtype.kind not in "iuf"
                    or float(off) != 0.0
                ):
                    raise ValueError(
                        f"{p}: {name} packing is not fill {FILL}, add_offset 0"
                    )
                scale = (
                    _scalar_attr(v.getncattr("scale_factor"))
                    if "scale_factor" in attrs
                    else None
                )
                if (
                    scale is None
                    or scale.dtype.kind != "f"
                    or float(scale) != SCALES[name]
                ):
                    raise ValueError(
                        f"{p}: {name} scale_factor must be exactly {SCALES[name]}"
                    )
                self._schema[name] = SCALES[name]
                # <--- jsasaki 2026-11-05
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
                # jsasaki 2026-11-05: review round 2 #8: kept as a timestamp (halo days may be added)
                self._decl.setdefault(m.group(1), set()).add(stamp)
            prov = str(getattr(ds, "provisional_fill", ""))
            if prov:
                # jsasaki 2026-11-05: review round 2 #18: per day (only the period's are reported),
                # with the method the tree builder recorded
                note = f"{p.name}: provisional_fill {prov} (time-interpolated stand-in)"
                method = str(getattr(ds, "provisional_method", ""))
                if method:
                    note += f"; provisional_method: {method}"
                self._notes_by_day.setdefault(day, []).append(note)
        return idx

    # ---- identity ---- #

    @property
    def name(self) -> str:
        return f"{self.root.name}[{self.start:%Y%m%d}-{self.end:%Y%m%d}]"

    def exists(self) -> bool:
        return True

    def provenance(self, variables: Iterable[str] | None = None) -> list[str]:
        """Lines for the history of a file written from this source.

        Only what was actually read (optionally restricted to ``variables``) and the hours that were
        actually interpolated inside the requested period are listed.
        """
        # ---> jsasaki 2026-11-05: review round 2 #18: report reads and repairs performed, not declarations
        lines = [self.name, f"missing-record policy: {self.missing}"]
        wanted = set(VARIABLES if variables is None else variables)
        used = [v for v in VARIABLES if v in self._read_vars and v in wanted]
        if used:
            lines.append("variables read: " + " ".join(used))
        for var in used:
            when = sorted(self._repaired.get(var, ()))
            if when:
                lines.append(
                    f"{var} linearly interpolated in time at "
                    + ", ".join(f"{t:%Y-%m-%d %H}Z" for t in when)
                )
        lines.extend(self.notes)
        return lines
        # <--- jsasaki 2026-11-05

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
        # jsasaki 2026-11-05: review round 2 #15: complex, boolean and object indices are refused
        if a.dtype.kind not in "iuf" or b.dtype.kind not in "iuf":
            raise ValueError(
                f"cell indices must be real integers (got dtypes {a.dtype}, {b.dtype})"
            )
        if not (
            np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer)
        ):
            if not (np.all(a == np.rint(a)) and np.all(b == np.rint(b))):
                raise ValueError("cell indices must be integers")
        a, b = a.astype(np.int64), b.astype(np.int64)
        assert self.lat is not None and self.lon is not None
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
                    # jsasaki 2026-11-05: review round 2 #6: never broadcast a short payload
                    want = (
                        (HOURS_PER_DAY, N_LEVELS, sel.size)
                        if is3d
                        else (HOURS_PER_DAY, sel.size)
                    )
                    if q.shape != want:
                        raise ValueError(
                            f"{path}: {name} payload has shape {q.shape}, expected {want}"
                        )
                    val = np.where(q == FILL, np.nan, q.astype(np.float64) * scale)
                    out[t0 : t0 + HOURS_PER_DAY, ..., sel] = val.astype(np.float32)
        # jsasaki 2026-11-05: review round 2 #18: record what was read
        self._read_vars.add(name)
        if gaps:  # missing == "interpolate"; only the requested hours are repaired
            # jsasaki 2026-11-05: review round 2 #8: no declared hour (halo included) is a neighbour
            good = np.setdiff1d(np.arange(n_full), self._declared.get(name, []))
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
            # jsasaki 2026-11-05: review round 2 #18: the hours actually interpolated
            self._repaired.setdefault(name, set()).update(
                self._time_full[g] for g in gaps
            )
        out = out[self._i0 : self._i0 + self.n_time]
        # A sea cell has data at every hour: an all-NaN column means the cell is land in the archive.
        # jsasaki 2026-11-05: review round 2 #17: checked one day at a time (no annual boolean array)
        empty_cells = np.zeros(n, dtype=bool)
        for t0 in range(0, self.n_time, HOURS_PER_DAY):
            nan_blk = np.isnan(out[t0 : t0 + HOURS_PER_DAY])
            if is3d:
                nan_blk = nan_blk.all(axis=1)
            empty_cells |= nan_blk.any(axis=0)
        if empty_cells.any():
            bad = np.flatnonzero(empty_cells)
            cells = ", ".join(f"(j={jj[b]}, i={ii[b]})" for b in bad[:5])
            raise ValueError(
                f"{name}: no data at {len(bad)} requested cell(s), e.g. {cells}: land (or fill) in the "
                f"archive although the grid file may call it ocean; move the open-boundary node or "
                f"choose another cell"
            )
        return out

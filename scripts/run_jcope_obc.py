#!/usr/bin/env python3
# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Drive :class:`xfvcom.io.JcopeObcGenerator` from the command line.

This is a thin runnable wrapper used to smoke-test the OBC generator
against real archive data and as a building block for project-specific
batch jobs. It loads:

* a FVCOM grid (``*_grd.dat``) for the geographic coordinates of every
  node and triangle,
* a FVCOM depth file (``*_dep.dat``) for the bathymetry at each node,
* a FVCOM OBC list (``*_obc.dat``) for the 1-based node IDs along the
  open boundary,
* the jcopetda archive (``basic.nc`` + ``jcopetda_region_YYYY.nc``),

and writes the TS-OBC and elevation NetCDFs alongside a short text
summary so the operator can sanity-check the result before driving FVCOM
with it.
"""

from __future__ import annotations

import argparse
import os

# jsasaki 2026-11-05: review round 2 #2 #4 #11 #12: strict parsers, time-axis and cache checks
# jsasaki 2026-11-05: review round 18 #4: 're' no longer needed (header regexes replaced)
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# jsasaki 2026-11-05: review round 14 #2: shared Fortran real-literal conversion
# jsasaki 2026-11-05: review round 17 #4: list-directed fields (commas separate like blanks)
# jsasaki 2026-11-05: review round 18 #1-#7: list_fields now reached through the shared scanner
# jsasaki 2026-11-05: review round 20 #6: unused fortran_real import removed (conversion is in list_read)
# jsasaki 2026-11-05: review round 19 #1 #3 #4 #5: READ-based header search and records
# jsasaki 2026-11-05: review round 21 #1: physical_lines splits records at '\n' only
from xfvcom.fvcom_input import (
    INT,
    REAL,
    ListRead,
    is_comment_only,
    locate_records,
    physical_lines,
    read_all,
    scan_count_headers,
)
from xfvcom.grid.grid_obj import FvcomGrid
from xfvcom.io import (
    FullDomainSource,
    JcopeGrid,
    JcopeObcGenerator,
    write_elevation_nc,
    write_tsobc_nc,
)

# jsasaki 2026-11-05: review round 18 #8: the generator's real-dtype guard, reused for the npz arrays
from xfvcom.io.jcope_obc_generator import _plain_float

# jsasaki 2026-11-05: review round 2 #2 #11 #12: constants of the strict parsers / time-axis check
# jsasaki 2026-11-05: review round 18 #4: the exact SCAN_FILE keys, looked up by the shared scanner
# (xfvcom/fvcom_input.py: commas, a signed count, '!' comments, the 400-byte buffer)
_OBC_KEY = "OBC Node Number"
_DEP_KEY = "Node Number"
DEP_XY_ATOL = 1e-3  # depth-file coordinates are written with 6 decimals
_HOUR = 1.0 / 24.0  # days
_STEP_TOL = 1e-6  # days (0.09 s)


def parse_year_spec(spec: str) -> list[int]:
    """Accept a single year, a comma list, or a 'YYYY-YYYY' inclusive range."""
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def check_year_list(years: list[int], allow_gaps: bool = False) -> None:
    """Refuse duplicate, unordered or (unless allowed) non-consecutive years.

    jsasaki 2026-11-04: review round 1 #2: the years are concatenated in list order, so a repeated or
    reversed year gives a non-monotonic time axis and a skipped year a silent gap inside one file.
    """
    if len(set(years)) != len(years):
        raise SystemExit(f"duplicate years in {years}")
    if years != sorted(years):
        raise SystemExit(f"years must be in increasing order (got {years})")
    if not allow_gaps and any(b != a + 1 for a, b in zip(years, years[1:])):
        raise SystemExit(
            f"years {years} are not consecutive: the output has ONE time axis, a skipped year would be an "
            f"unrepresented gap inside it (pass --allow-year-gaps to build it anyway)"
        )


def format_year_label(years: list[int]) -> str:
    if len(years) == 1:
        return str(years[0])
    return f"{min(years)}-{max(years)}"


# jsasaki 2026-11-05: review round 18 #1-#7: the count header through the shared FVCOM scanner
# jsasaki 2026-11-05: review round 19 #1 #3 #4 #5: records are FVCOM's list-directed READs, not lines
def _declared_block(
    path: Path, key: str, types: tuple[str, ...], what: str
) -> list[ListRead]:
    """The N READs of the block declared by the ``<key> = N`` header, each with its values.

    The block starts where FVCOM's header search first READs ``types`` successfully: comment,
    title and ``key = value`` lines before it are failed READs, i.e. header lines
    (:func:`xfvcom.fvcom_input.locate_records`, which also refuses a blank line that would make
    FVCOM's NHEADER rewind start inside the header).  The header is looked up as SCAN_FILE does
    (:func:`xfvcom.fvcom_input.scan_count_headers`).  A record may continue onto further lines
    and may end with ``/`` once its values are supplied; blank lines are not records.
    READ_COLDSTART_OBC_GRID / READ_COLDSTART_DEPTH count every READ that does not hit end-of-file
    as one row, a failed one included, so a READ that fails (a comment-only line inside the block:
    gfortran IOSTAT 5010, ifx 59) or leaves an item unset (a null value, an early ``/``) is refused,
    as is a count that differs from the header's.
    """
    # jsasaki 2026-11-05: review round 21 #1: not str.splitlines (it also splits at U+2028, \f, ...)
    lines = physical_lines(path)
    skipped: list[ListRead] = []
    start = locate_records(path, lines, types, block="record block", failed_out=skipped)
    counts = scan_count_headers(path, lines, {key: "n"}, start, block="record block")
    if "n" not in counts:
        raise ValueError(
            f"{path}: the header '{key} = N' was not found before the first record (FVCOM "
            f"looks up the exact, case-sensitive key '{key}')"
        )
    n = counts["n"]
    reads, eof = read_all(lines, start, types)
    for rd in reads:
        if rd.status == "error" and is_comment_only(lines[rd.end][1]):
            raise ValueError(
                f"{path}: line {rd.last_line}: comment-only line inside the record block; "
                "FVCOM's list-directed READ fails on it and counts it as one more row, so the "
                f"header declares {n} record(s) but FVCOM reads {len(reads)} -- move the "
                "comment above the first record or after a record on the same line"
            )
    if eof.values:
        raise ValueError(
            f"{path}: lines {eof.first_line}-{eof.last_line}: incomplete {what} record at the "
            f"end of the file ({len(eof.values)} of {len(types)} values)"
        )

    def malformed(rd: ListRead, note: str = "") -> ValueError:
        span = (
            f"line {rd.first_line}"
            if rd.first_line == rd.last_line
            else f"lines {rd.first_line}-{rd.last_line}"
        )
        text = " / ".join(raw.strip() for _n, raw in lines[rd.start : rd.end + 1])
        return ValueError(
            f"{path}: {span}: malformed {what} record {text!r}: {rd.message}{note}"
        )

    for rd in reads:
        if rd.status != "ok":
            raise malformed(rd)
    if len(reads) != n:
        # a READ that converted some values before failing, among the lines FVCOM's header search
        # passed over, is most likely a broken record (the search took it for a header line)
        broken = [rd for rd in skipped if rd.values]
        if broken:
            raise malformed(
                broken[-1],
                f"; FVCOM's header search passes it over as a header line, so the block "
                f"holds {len(reads)} record(s) for the {n} the header declares",
            )
        raise ValueError(
            f"{path}: the header declares {n} record(s), the file holds {len(reads)} "
            "(records counted as FVCOM's list-directed READs)"
        )
    return reads


def parse_obc_dat(path: Path) -> np.ndarray:
    """Return the 1-based OBC node IDs from a FVCOM ``*_obc.dat`` file.

    The file is ``OBC Node Number = N`` followed by exactly N records ``index node_id type``
    (three integers read list-directed: a record may continue onto the next line or end with
    ``/``; a trailing ``!`` comment and further columns are ignored); anything else is an error.
    """
    # jsasaki 2026-11-05: review round 2 #11: one recognised header, its count enforced, every
    # record (the first included) parsed strictly
    # jsasaki 2026-11-05: review round 19 #4: one record = one READ of N1,N2,N3 (FVCOM's loop)
    reads = _declared_block(Path(path), _OBC_KEY, (INT, INT, INT), "OBC")
    nodes = [int(rd.values[1]) for rd in reads]  # type: ignore[arg-type]
    if min(nodes) < 1 or len(set(nodes)) != len(nodes):
        raise ValueError(f"{path}: OBC node IDs must be unique and >= 1")
    return np.asarray(nodes, dtype=np.int32)


def parse_dep_dat(path: Path, node_xy: np.ndarray | None = None) -> np.ndarray:
    """Return the bathymetry column from a FVCOM ``*_dep.dat`` file.

    The file is ``Node Number = N`` followed by exactly N records ``x y h`` (read list-directed:
    a record may continue onto the next line or end with ``/``; a trailing ``!`` comment and
    further columns are ignored). With ``node_xy`` (``(N, 2)``, the mesh's node coordinates in the
    grid file's order) the depth file's coordinates must match them row by row (within
    ``DEP_XY_ATOL`` coordinate units), so a reordered depth file or one of another mesh is refused
    instead of putting depths on the wrong nodes.
    """
    # jsasaki 2026-11-05: review round 2 #12: the whole declared block, strictly, with coordinates
    # jsasaki 2026-11-05: review round 19 #4: one record = one READ of X1,Y1,HDEP (FVCOM's loop)
    reads = _declared_block(Path(path), _DEP_KEY, (REAL, REAL, REAL), "depth")
    rows: NDArray[np.float64] = np.array([rd.values for rd in reads], dtype=np.float64)
    if not np.isfinite(rows).all():
        raise ValueError(f"{path}: non-finite coordinate or depth")
    if node_xy is not None:
        xy = np.asarray(node_xy, dtype=np.float64)
        if xy.shape != (rows.shape[0], 2):
            raise ValueError(
                f"{path}: {rows.shape[0]} depth records for a mesh of {xy.shape[0]} nodes"
            )
        dev = np.abs(rows[:, :2] - xy).max(axis=1)
        if (dev > DEP_XY_ATOL).any():
            k = int(np.argmax(dev))
            raise ValueError(
                f"{path}: record {k + 1} sits at ({rows[k, 0]}, {rows[k, 1]}), mesh node {k + 1} at "
                f"({xy[k, 0]}, {xy[k, 1]}): the depth file does not belong to this mesh / node order"
            )
    return rows[:, 2].astype(np.float32)


def check_time_axis(
    years: list[int], mjd_by_year: dict[int, np.ndarray], allow_gaps: bool = False
) -> None:
    """Each source's times lie in its own year with hourly steps; the concatenation is hourly too.

    Only between two sources of non-consecutive years (with --allow-year-gaps) may the step be longer;
    it must still be positive. Duplicated or backward times are always refused.
    """
    # jsasaki 2026-11-05: review round 2 #2: the actual periods, not only the year list, are checked
    epoch = pd.Timestamp("1858-11-17")
    # ---> jsasaki 2026-11-05: review round 3 #18: finite, non-empty 1-D times (NaN compares false
    # with every bound below); masked entries count as missing
    checked: dict[int, np.ndarray] = {}
    for y in years:
        # jsasaki 2026-11-05: review round 4 #9: complex/object times are refused before the cast
        # (a cast to float64 discards a non-finite imaginary part)
        raw = np.ma.asarray(mjd_by_year[y])
        if raw.dtype.kind not in "iuf":
            raise SystemExit(
                f"the source for {y} has a time axis of dtype {raw.dtype}; real numbers are required"
            )
        t = np.ma.filled(raw.astype(np.float64), np.nan)
        if t.ndim != 1 or t.size == 0 or not np.isfinite(t).all():
            raise SystemExit(
                f"the source for {y} must have a non-empty, finite 1-D time axis"
            )
        checked[y] = t
    mjd_by_year = checked
    # <--- jsasaki 2026-11-05
    for y in years:
        t = mjd_by_year[y]
        lo = (pd.Timestamp(f"{y}-01-01") - epoch) / pd.Timedelta(days=1)
        hi = (pd.Timestamp(f"{y + 1}-01-01") - epoch) / pd.Timedelta(days=1)
        if t.size == 0 or t.min() < lo - _STEP_TOL or t.max() > hi - _HOUR + _STEP_TOL:
            raise SystemExit(f"the source for {y} has times outside the year {y}")
        if t.size > 1 and (np.abs(np.diff(t) - _HOUR) > _STEP_TOL).any():
            raise SystemExit(
                f"the source for {y} is not an hourly series without duplicates or gaps"
            )
    for a, b in zip(years, years[1:]):
        step = float(mjd_by_year[b][0] - mjd_by_year[a][-1])
        if abs(step - _HOUR) <= _STEP_TOL:
            continue
        if step <= 0 or not (allow_gaps and b != a + 1):
            raise SystemExit(
                f"the time axis jumps by {step * 24:g} h between {a} and {b} "
                f"(must be 1 h{'' if allow_gaps else '; --allow-year-gaps permits skipped years only'})"
            )


# ---------------------------------------------------------------------------
# Raw-profile cache. The EXPENSIVE step is extracting the JCOPE T/S profiles at
# the OBC cells (reads the ~27 GB/year region archive). Those profiles depend
# only on the region archive + OBC-node (lat, lon) — NOT on the FVCOM depth /
# sigma — so they are identical for EVERY bathymetry variant of the same mesh +
# OBC list. Cache them once per year (~50 MB) and every later variant rebuild
# reads the cache + redoes only the cheap depth-dependent vertical interpolation.
# ---------------------------------------------------------------------------
def _profile_cache_path(cache_dir, year):
    return Path(cache_dir) / f"obc_raw_profiles_{year}.npz"


def _file_identity(path) -> tuple[str, int, int]:
    p = Path(path).expanduser().resolve()
    st = p.stat()
    return str(p), int(st.st_size), int(st.st_mtime_ns)


def _profile_fingerprint(
    region_nc, basic_nc, obc_ids, obc_lat, obc_lon, cell_j, cell_i, time_mjd
) -> dict[str, np.ndarray]:
    """What the cached profiles depend on: source file, grid file, nodes, their cells and times."""
    # jsasaki 2026-11-05: review round 2 #4: the cache is bound to all of these
    sp, ss, sm = _file_identity(region_nc)
    gp, gs, gm = _file_identity(basic_nc)
    return {
        "src_path": np.asarray(sp),
        "src_size": np.asarray(ss, dtype=np.int64),
        "src_mtime_ns": np.asarray(sm, dtype=np.int64),
        "grid_path": np.asarray(gp),
        "grid_size": np.asarray(gs, dtype=np.int64),
        "grid_mtime_ns": np.asarray(gm, dtype=np.int64),
        "obc_nodes": np.asarray(obc_ids, dtype=np.int32),
        "obc_lat": np.asarray(obc_lat, dtype=np.float64),
        "obc_lon": np.asarray(obc_lon, dtype=np.float64),
        "cell_j": np.asarray(cell_j, dtype=np.int64),
        "cell_i": np.asarray(cell_i, dtype=np.int64),
        "time_mjd": np.asarray(time_mjd, dtype=np.float64),
    }


def _load_profile_cache(cache_dir, year, fingerprint):
    """Return the cached raw profiles dict if a valid cache exists, else None.

    Valid means: every fingerprint entry (source path/size/mtime, grid file identity, node ids and
    coordinates, selected cells, time axis) is identical and each profile has the shape
    ``(n_time, 46, n_obc)``. Anything else -- including an unreadable file -- means rebuild.
    """
    # ---> jsasaki 2026-11-05: review round 2 #4: full fingerprint; the whole file is read and
    # validated inside the try block
    if not cache_dir:
        return None
    p = _profile_cache_path(cache_dir, year)
    if not p.exists():
        return None
    n_time = fingerprint["time_mjd"].size
    n_obc = fingerprint["obc_nodes"].size
    try:
        with np.load(p, allow_pickle=False) as d:
            for k, v in fingerprint.items():
                if k not in d.files or not np.array_equal(d[k], v):
                    return None
            raw = {k[4:]: np.array(d[k]) for k in d.files if k.startswith("var_")}
        if not raw or any(a.shape != (n_time, 46, n_obc) for a in raw.values()):
            return None
        return raw
    except Exception:
        return None
    # <--- jsasaki 2026-11-05


def _save_profile_cache(cache_dir, year, fingerprint, raw):
    if not cache_dir:
        return
    p = _profile_cache_path(cache_dir, year)
    p.parent.mkdir(parents=True, exist_ok=True)
    # jsasaki 2026-11-05: review round 2 #4 #16: fingerprint stored; unique temporary, removed on failure
    fd, name = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as fh:  # a handle: np.savez does NOT append '.npz'
            np.savez(fh, **fingerprint, **{f"var_{k}": v for k, v in raw.items()})
        tmp.replace(p)  # atomic swap (each year writes its own file)
    finally:
        if tmp.exists():
            tmp.unlink()


# jsasaki 2026-11-05: review round 7 #1: new helper -- an external per-node
# coordinate must be tied to the requested OBC nodes by id and by depth
def load_siglev_npz(
    path: Path,
    obc_ids: np.ndarray,
    obc_h: np.ndarray,
    ignore_depth: bool = False,
    depth_tol: float = 1e-3,
) -> np.ndarray:
    """Load an explicit per-node OBC coordinate, ordered like ``obc_ids``.

    The npz holds ``siglev`` (n_obc, KB) and, for identity checks,
    ``obc_node`` (1-based node ids, row order of ``siglev``) and ``h``
    (depth in m that each row was built for). With ``obc_node`` the rows are
    reordered by id to the OBC list order; the id sets must be identical. With
    ``h`` the recorded depths must equal ``obc_h`` (the depths the runner
    samples at) within ``depth_tol`` m unless ``ignore_depth``. Missing keys
    only produce a warning (identity unverified). Returns (n_obc, KB) float64.
    Raises ValueError on any mismatch.
    """
    obc_ids = np.asarray(obc_ids, dtype=np.int64)
    with np.load(path) as d:
        if "siglev" not in d.files:
            raise ValueError(f"--siglev-npz {path} has no array 'siglev'")
        # ---> jsasaki 2026-11-05: review round 18 #8: real numeric dtypes required BEFORE the cast
        # (a complex -> float64 cast drops a non-finite imaginary part before the finiteness check)
        z = _plain_float(d["siglev"], "--siglev-npz 'siglev'")
        ids = np.asarray(d["obc_node"]) if "obc_node" in d.files else None
        h_rec = _plain_float(d["h"], "--siglev-npz 'h'") if "h" in d.files else None
        # <--- jsasaki 2026-11-05
    if z.ndim != 2 or z.shape[1] < 2:
        raise ValueError(f"--siglev-npz 'siglev' must be (n_obc, KB), got {z.shape}")
    if not np.all(np.isfinite(z)):
        raise ValueError("--siglev-npz 'siglev' holds non-finite values")
    if z.shape[0] != obc_ids.shape[0]:
        raise ValueError(
            f"--siglev-npz has {z.shape[0]} nodes != OBC list {obc_ids.shape[0]}"
        )
    if h_rec is not None and h_rec.shape != (z.shape[0],):
        raise ValueError(
            f"--siglev-npz 'h' has shape {h_rec.shape}, expected ({z.shape[0]},)"
        )
    if ids is None:
        print(
            f"WARNING: --siglev-npz {path} has no 'obc_node' array: node identity "
            "could NOT be verified; rows are assumed to follow the OBC list order"
        )
    else:
        if ids.shape != (z.shape[0],):
            raise ValueError(
                f"--siglev-npz 'obc_node' has shape {ids.shape}, expected ({z.shape[0]},)"
            )
        if not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(
                f"--siglev-npz 'obc_node' must be integer, got {ids.dtype}"
            )
        ids = ids.astype(np.int64)
        if np.unique(ids).size != ids.size:
            raise ValueError("--siglev-npz 'obc_node' holds duplicate node ids")
        if set(ids.tolist()) != set(obc_ids.tolist()):
            only_npz = sorted(set(ids.tolist()) - set(obc_ids.tolist()))
            only_obc = sorted(set(obc_ids.tolist()) - set(ids.tolist()))
            raise ValueError(
                "--siglev-npz 'obc_node' differs from the OBC list: "
                f"only in npz {only_npz}, only in OBC list {only_obc}"
            )
        pos = {int(n): i for i, n in enumerate(ids)}
        order = np.array([pos[int(n)] for n in obc_ids], dtype=np.int64)
        if not np.array_equal(order, np.arange(order.size)):
            print("[siglev-npz] rows reordered by node id to the OBC list order")
        z = z[order]
        if h_rec is not None:
            h_rec = h_rec[order]
    if h_rec is None:
        print(
            f"WARNING: --siglev-npz {path} has no 'h' array: the depths the "
            "coordinate was built for could NOT be verified"
        )
    else:
        obc_h = np.asarray(obc_h, dtype=np.float64)
        diff = np.abs(h_rec - obc_h)
        dmax = float(np.max(diff)) if diff.size else 0.0
        if not np.all(np.isfinite(diff)) or dmax > depth_tol:
            k = int(np.nanargmax(diff)) if np.any(np.isfinite(diff)) else 0
            msg = (
                f"--siglev-npz recorded depths 'h' differ from the selected OBC "
                f"depths by up to {dmax:.6g} m (node {int(obc_ids[k])}: npz "
                f"{h_rec[k]:.6g} m vs depth file {obc_h[k]:.6g} m; tolerance "
                f"{depth_tol:g} m) -- wrong depth file or wrong npz?"
            )
            if not ignore_depth:
                raise ValueError(msg)
            print("WARNING (--siglev-npz-ignore-depth): " + msg)
    return z


def _process_year(
    year,
    region_nc,
    basic_nc,
    obc_ids,
    obc_lat,
    obc_lon,
    obc_h_fvcom,
    n_siglay,
    siglay_per_node,
    siglev_per_node,
    no_tsobc,
    no_elevation,
    profile_cache_dir=None,
):
    """Interpolate one year's OBC — the per-year loop body, extracted so it can
    run as a joblib worker (``--jobs>1``) and share the profile-cache path. Each
    worker opens its own JcopeGrid + region archive, so years are independent;
    all args are picklable (numpy arrays / Path / scalars). Returns a plain dict
    keyed for reassembly in the parent. With ``profile_cache_dir`` the raw JCOPE
    profiles are loaded from / saved to the cache (skipping the 27 GB read on a
    hit); the output is identical either way.
    """
    # jsasaki 2026-11-04: region_nc may be a (picklable) FullDomainSource
    grid = JcopeGrid(basic_nc)
    gen = JcopeObcGenerator(
        grid=grid,
        region_nc=region_nc,
        obc_nodes=obc_ids,
        obc_lat=obc_lat,
        obc_lon=obc_lon,
        obc_h_fvcom=obc_h_fvcom,
        n_siglay=n_siglay,
        siglay_per_node=siglay_per_node,
        siglev_per_node=siglev_per_node,
    )
    tsobc = None
    if not no_tsobc:
        if profile_cache_dir:
            # jsasaki 2026-11-05: review round 2 #4: cache bound to a full fingerprint
            fp = _profile_fingerprint(
                region_nc,
                basic_nc,
                obc_ids,
                obc_lat,
                obc_lon,
                gen._jr,
                gen._ir,
                gen.mjd,
            )
            raw = _load_profile_cache(profile_cache_dir, year, fp)
            hit = raw is not None
            if raw is None:
                raw = gen.read_raw_profiles()
                _save_profile_cache(profile_cache_dir, year, fp, raw)
            print(
                f"[cache] year {year}: profile cache {'HIT' if hit else 'MISS (built)'}",
                flush=True,
            )
            tsobc = gen.build_tsobc_arrays(raw_profiles=raw)
        else:
            tsobc = gen.build_tsobc_arrays()
    # jsasaki 2026-11-05: review round 2 #18: provenance from the source this worker actually read
    prov_ts = gen.source_lines(["TT", "ST"])
    elevation = None if no_elevation else gen.build_elevation_array()
    out = {
        "year": year,
        "mjd": gen.mjd.copy(),
        "siglev": gen.siglev,
        "siglay": gen.siglay,
        "obc_h_jcope": gen.obc_h_jcope.copy(),
        "tsobc": tsobc,
        "elevation": elevation,
        "prov_ts": prov_ts,
        "prov_el": gen.source_lines(["EGT"]),
    }
    gen.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate FVCOM OBC NetCDFs from a JCOPE-T DA archive via "
            "xfvcom.io.JcopeObcGenerator."
        ),
    )
    parser.add_argument(
        "--basic-nc",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", "/var/empty"))
        / "jcopetda/grid/basic.nc",
        help="path to jcopetda basic.nc (default: $DATA_DIR/jcopetda/grid/basic.nc)",
    )
    parser.add_argument(
        "--region-nc",
        type=Path,
        default=None,
        help="path to jcopetda_region_YYYY.nc time-series archive "
        "(single-year mode; mutually exclusive with --region-pattern)",
    )
    parser.add_argument(
        "--allow-year-gaps",
        action="store_true",
        help="accept a --years list that skips years (the output time axis then has a jump)",
    )
    # jsasaki 2026-11-04: read the daily full-domain archive instead of region files
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=None,
        help="root of the full-domain archive (YYYY/jcopetda_YYYYMMDD.nc), e.g. "
        "$DATA_DIR/jcopetda_provisional; replaces --region-nc/--region-pattern. Each year of "
        "--year/--years is read whole (Jan 1 - Dec 31); the profile cache is not used.",
    )
    parser.add_argument(
        "--missing",
        choices=("error", "interpolate"),
        default="error",
        help="archive mode: what to do with hours the archive declares missing "
        "(default error; the provisional tree has none)",
    )
    parser.add_argument(
        "--region-pattern",
        type=str,
        default=None,
        help="path template containing {year}, e.g. "
        "'/data/jcopetda/region/.../jcopetda_region_{year}.nc' "
        "(required when --years selects more than one year)",
    )
    parser.add_argument(
        "--fvcom-grid", type=Path, required=True, help="path to FVCOM *_grd.dat"
    )
    parser.add_argument(
        "--fvcom-dep", type=Path, required=True, help="path to FVCOM *_dep.dat"
    )
    parser.add_argument(
        "--fvcom-obc",
        type=Path,
        required=True,
        help="path to FVCOM *_obc.dat (node ID list)",
    )
    parser.add_argument(
        "--utm-zone",
        type=int,
        default=54,
        help="UTM zone for the grd file (default 54 for Tokyo Bay)",
    )
    parser.add_argument(
        "--n-siglay",
        type=int,
        default=30,
        help="number of FVCOM sigma layers (uniform, default 30)",
    )
    parser.add_argument(
        "--sigma-dat",
        type=Path,
        default=None,
        help="SADAPT / sigma-z sigma.dat. When given, build a PER-NODE OBC "
        "vertical coordinate from it (overrides --n-siglay) so the OBC lands on "
        "the model's actual sigma-z layer depths. Requires --fvcom-grid + "
        "--fvcom-dep to match the grid the sigma.dat was built for.",
    )
    # jsasaki 2026-11-05: review round 7 #1: help names the identity arrays
    parser.add_argument(
        "--siglev-npz",
        type=Path,
        default=None,
        help="npz with array 'siglev' of shape (n_obc, KB): an EXPLICIT per-node "
        "OBC vertical coordinate (sigma levels, 0..-1) built externally -- e.g. "
        "the LSC2 surface-aligned per-column grid, which has no sigma.dat. "
        "Overrides --n-siglay; mutually exclusive with --sigma-dat. "
        "Optional arrays 'obc_node' (node ids) and 'h' (depths) are checked "
        "against the OBC list and the depth file (rows reordered by id).",
    )
    # jsasaki 2026-11-05: review round 7 #1: explicit override of the depth check
    parser.add_argument(
        "--siglev-npz-ignore-depth",
        action="store_true",
        help="warn instead of stopping when the --siglev-npz 'h' depths differ "
        "from the selected OBC depths (default: stop)",
    )
    parser.add_argument(
        "--obc-depth-control",
        action="store_true",
        help="sample at the depths FVCOM's OBC_DEPTH_CONTROL_ON=T actually "
        "imposes the file at: substitute each OBC node's depth with its "
        "NEXT_OBC inward neighbour's (a-priori port of mod_obcs.F SETUP_OBC; "
        "boundary-string endpoints keep their own depth per observed runtime "
        "behaviour). Uniform-sigma path only. ALWAYS contract-check the "
        "assumed depths against a run's output h before production use "
        "(see xfvcom.io.obc_depth_control).",
    )
    # jsasaki 2026-11-05: review round 13 #1: explicit endpoint rule for --obc-depth-control
    parser.add_argument(
        "--obc-depth-control-endpoints",
        choices=("observed", "source"),
        default="observed",
        help="boundary-string endpoint rule for --obc-depth-control: 'observed' "
        "(default) keeps each endpoint's own depth, as the runtime h of the "
        "TB-FVCOM b9 60-rank production run shows; 'source' follows mod_obcs.F "
        "SETUP_OBC / mod_startup.F SET_WATER_DEPTH literally (endpoints also "
        "take their NEXT_OBC neighbour's depth)",
    )
    parser.add_argument("--tag", default="jcope", help="filename tag (default: jcope)")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="parallel workers over years (joblib; default 1 = serial, "
        "byte-identical to the original path). Each worker reads one year's "
        "region NC independently; the deterministic time concatenation is "
        "unchanged, so N>1 only speeds up a multi-year build.",
    )
    parser.add_argument(
        "--profile-cache-dir",
        type=Path,
        default=None,
        help="directory for the raw-profile cache (obc_raw_profiles_<year>.npz). "
        "The JCOPE T/S profiles at the OBC cells are bathymetry-INDEPENDENT "
        "(fixed OBC node lat/lon), so caching them once lets every later "
        "depth-variant OBC rebuild skip the ~27 GB/year region read and redo "
        # jsasaki 2026-11-05: review round 2 #4: the help names the full fingerprint
        "only the cheap vertical interpolation. Cache is invalidated if the region "
        "file (path, size, mtime), the grid file, the OBC nodes or their coordinates, "
        "the selected cells or the time axis change. Omit to disable.",
    )
    year_group = parser.add_mutually_exclusive_group(required=True)
    year_group.add_argument(
        "--year", type=int, help="single year; use --region-nc for the archive"
    )
    year_group.add_argument(
        "--years",
        type=str,
        help="year list ('2020,2022') or inclusive range ('2020-2023'); "
        "requires --region-pattern",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory to write tsobc / elevation NetCDFs to",
    )
    parser.add_argument(
        "--no-tsobc", action="store_true", help="skip writing the T/S OBC file"
    )
    parser.add_argument(
        "--no-elevation", action="store_true", help="skip writing the SSH OBC file"
    )
    args = parser.parse_args()

    # ---- resolve year list and region-NC source(s) ----
    if args.year is not None:
        years = [args.year]
    else:
        years = parse_year_spec(args.years)
    if not years:
        sys.exit("no years to process")
    check_year_list(years, args.allow_year_gaps)

    if args.archive_root is not None:
        # jsasaki 2026-11-04: one FullDomainSource per calendar year
        region_paths: list = [
            FullDomainSource(
                args.archive_root, f"{y}-01-01", f"{y}-12-31", missing=args.missing
            )
            for y in years
        ]
        args.profile_cache_dir = None
    elif len(years) == 1 and args.region_nc is not None:
        region_paths = [args.region_nc]
    elif args.region_pattern is not None:
        # jsasaki 2026-11-05: review round 2 #2: one distinct file per year
        if len(years) > 1 and "{year}" not in args.region_pattern:
            sys.exit(
                "--region-pattern must contain {year} when several years are built"
            )
        region_paths = [Path(args.region_pattern.format(year=y)) for y in years]
        if len({p.expanduser().resolve() for p in region_paths}) != len(region_paths):
            sys.exit("--region-pattern gives the same file for different years")
    elif len(years) == 1:
        sys.exit("single-year run needs either --region-nc or --region-pattern")
    else:
        sys.exit("multi-year run requires --region-pattern")

    missing = [p for p in region_paths if not p.exists()]
    if missing:
        sys.exit("missing region NCs: " + ", ".join(str(p) for p in missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load FVCOM-side metadata once ----
    print(f"[grid] reading FVCOM mesh from {args.fvcom_grid}")
    # jsasaki 2026-11-05: review round 9 #2: FVCOM's row-order semantics (ids 1..n in file order)
    mesh = FvcomGrid.from_dat(
        args.fvcom_grid, utm_zone=args.utm_zone, allow_unordered=False
    )
    if mesh.lat is None or mesh.lon is None:
        sys.exit("FvcomGrid lacks lon/lat — check utm_zone and the *_grd.dat file")
    n_nodes = mesh.node
    print(f"  nodes: {n_nodes}, elements: {mesh.nele}")

    print(f"[grid] reading FVCOM depth from {args.fvcom_dep}")
    # jsasaki 2026-11-05: review round 2 #12: the depth rows must sit on the mesh nodes, in order
    try:
        h_all = parse_dep_dat(
            args.fvcom_dep, np.column_stack([np.asarray(mesh.x), np.asarray(mesh.y)])
        )
    except ValueError as e:
        sys.exit(str(e))
    if h_all.size != n_nodes:
        sys.exit(
            f"depth file has {h_all.size} rows; grid has {n_nodes} nodes — mismatch"
        )

    print(f"[grid] reading OBC node list from {args.fvcom_obc}")
    obc_ids = parse_obc_dat(args.fvcom_obc)
    print(f"  OBC nodes ({obc_ids.size}): {obc_ids.tolist()}")
    if int(obc_ids.max()) > n_nodes:
        sys.exit(f"OBC node {int(obc_ids.max())} exceeds the mesh's {n_nodes} nodes")

    idx0: NDArray[np.int64] = (obc_ids - 1).astype(np.int64)
    obc_lat = np.asarray(mesh.lat)[idx0]
    obc_lon = np.asarray(mesh.lon)[idx0]
    obc_h_fvcom = h_all[idx0]

    # jsasaki 2026-11-05: review round 13 #1: OBC-depth provenance for the T/S history
    tsobc_history_note: str | None = None
    # jsasaki 2026-11-05: review round 13 #1: the endpoint rule is meaningless without depth control
    if args.obc_depth_control_endpoints != "observed" and not args.obc_depth_control:
        sys.exit("--obc-depth-control-endpoints requires --obc-depth-control")
    if args.obc_depth_control:
        if args.sigma_dat is not None or args.siglev_npz is not None:
            sys.exit(
                "--obc-depth-control supports the uniform-sigma path only "
                "(per-node coordinates run with OBC_DEPTH_CONTROL_ON=F instead)"
            )
        from xfvcom.io.obc_depth_control import substituted_obc_depths

        # ---> jsasaki 2026-11-05: review round 13 #1: endpoint rule chosen explicitly,
        # printed and recorded in the T/S file's history
        rule = args.obc_depth_control_endpoints
        h_sub, next_ids = substituted_obc_depths(
            mesh.x, mesh.y, mesh.nv, obc_ids, h_all, endpoint_rule=rule
        )
        rule_text = (
            "endpoints keep own depth (observed runtime rule)"
            if rule == "observed"
            else "endpoints take their NEXT_OBC depth (mod_obcs.F source rule)"
        )
        tsobc_history_note = (
            "OBC depths: OBC_DEPTH_CONTROL_ON=T substitution (a-priori NEXT_OBC), "
            f"endpoint rule '{rule}': {rule_text}"
        )
        print(
            "[obc-depth-control] sampling at OBC_DEPTH_CONTROL_ON=T substituted "
            f"depths (a-priori NEXT_OBC; endpoint rule '{rule}': {rule_text}):"
        )
        # <--- jsasaki 2026-11-05
        for i, n in enumerate(obc_ids):
            mark = (
                ""
                if next_ids[i] == n
                else f" <- NEXT_OBC {next_ids[i]}"
                + (
                    f" (dH {h_sub[i] - obc_h_fvcom[i]:+.2f} m)"
                    if abs(h_sub[i] - obc_h_fvcom[i]) > 0.005
                    else " (equal depth)"
                )
            )
            print(f"  node {n:6d}: h {obc_h_fvcom[i]:8.2f} -> {h_sub[i]:8.2f}{mark}")
        print(
            "  NOTE: contract-check these depths against a run's output h "
            "(build_dep_obcsub.py harvest) before production use."
        )
        obc_h_fvcom = h_sub.astype(obc_h_fvcom.dtype)

    # ---- optional SADAPT / sigma-z per-node vertical coordinate ----
    # Without --sigma-dat the OBC uses the uniform --n-siglay grid (default).
    # With it, build the FVCOM sigma-z coordinate from the SADAPT sigma.dat and
    # take each OBC node's actual layer depths (coord.Z) so the OBC values land on
    # the model's per-node sigma-z column (FVCOM does NO vertical re-interpolation
    # of the OBC file). The grid/dep MUST be the ones the sigma.dat was built for.
    siglay_per_node = None
    siglev_per_node = None
    if args.siglev_npz is not None:
        if args.sigma_dat is not None:
            sys.exit("--siglev-npz and --sigma-dat are mutually exclusive")
        # jsasaki 2026-11-05: review round 7 #1: verify node ids / depths, reorder by id
        try:
            z_obc = load_siglev_npz(
                args.siglev_npz,
                obc_ids,
                obc_h_fvcom,
                ignore_depth=args.siglev_npz_ignore_depth,
            )  # (n_obc, KB), OBC list order
        except ValueError as e:
            sys.exit(str(e))
        siglev_per_node = z_obc.T  # (KB, n_obc)
        siglay_per_node = 0.5 * (siglev_per_node[:-1] + siglev_per_node[1:])
        print(
            f"[siglev-npz] explicit per-node OBC coordinate from {args.siglev_npz.name}: "
            f"KB={siglev_per_node.shape[0]} KBM1={siglay_per_node.shape[0]} "
            f"(uniform --n-siglay={args.n_siglay} IGNORED)"
        )
    if args.sigma_dat is not None:
        from xfvcom.grid.gtsz_builder import build_gtsz, load_mesh
        from xfvcom.io import read_sigma_dat

        sf = read_sigma_dat(args.sigma_dat)
        if sf.gtsz is None:
            sys.exit(
                f"--sigma-dat {args.sigma_dat} is not a SIGMAZ file "
                f"(SIGMA COORDINATE TYPE = {sf.stype}); use --n-siglay for uniform σ"
            )
        gmesh = load_mesh(args.fvcom_grid, args.fvcom_dep)
        if gmesh.h.shape[0] != n_nodes:
            sys.exit(f"--sigma-dat mesh has {gmesh.h.shape[0]} nodes != grid {n_nodes}")
        coord = build_gtsz(gmesh, sf.gtsz)
        z_obc = np.asarray(coord.Z, dtype=np.float64)[idx0]  # (n_obc, KB) siglev
        siglev_per_node = z_obc.T  # (KB, n_obc)
        siglay_per_node = 0.5 * (
            siglev_per_node[:-1] + siglev_per_node[1:]
        )  # (KBM1, n_obc)
        print(
            f"[sigma-z] SADAPT per-node OBC coordinate from {args.sigma_dat.name}: "
            f"KB={sf.gtsz.kb} KBM1={siglay_per_node.shape[0]} "
            f"(uniform --n-siglay={args.n_siglay} IGNORED); "
            f"KBP range {int(coord.kbp[idx0].min())}-{int(coord.kbp[idx0].max())}"
        )

    print(f"[jcope] opening basic.nc at {args.basic_nc}")
    grid = JcopeGrid(args.basic_nc)
    print(
        f"  grid: {grid.im} x {grid.jm} x {grid.km}, "
        f"ocean fraction {grid.mask.mean()*100:.1f}%"
    )

    # ---- loop over years, accumulating arrays ----
    tsobc_fields_by_year: dict[int, dict[str, np.ndarray]] = {}
    elevation_by_year: dict[int, np.ndarray] = {}
    mjd_by_year: dict[int, np.ndarray] = {}
    siglev: NDArray[np.float64] | None = None
    siglay: NDArray[np.float64] | None = None
    obc_h_jcope_first: NDArray[np.float32] | None = None

    # Years are independent — process each via _process_year (raw-profile read /
    # cache + vertical interp). With --jobs>1 they run concurrently (joblib),
    # otherwise serially; the deterministic per-year concatenation below is
    # order-independent, so the output is identical either way.
    def _args_for(year, region_nc):
        return (
            year,
            region_nc,
            args.basic_nc,
            obc_ids,
            obc_lat,
            obc_lon,
            obc_h_fvcom,
            args.n_siglay,
            siglay_per_node,
            siglev_per_node,
            args.no_tsobc,
            args.no_elevation,
            args.profile_cache_dir,
        )

    if args.jobs and args.jobs > 1 and len(years) > 1:
        from joblib import Parallel, delayed

        print(
            f"[jcope] parallel over {len(years)} years with n_jobs={args.jobs} (joblib) ...",
            flush=True,
        )
        results = Parallel(n_jobs=args.jobs)(
            delayed(_process_year)(*_args_for(year, region_nc))
            for year, region_nc in zip(years, region_paths)
        )
    else:
        results = [
            _process_year(*_args_for(year, region_nc))
            for year, region_nc in zip(years, region_paths)
        ]
    for r in results:
        y = r["year"]
        mjd_by_year[y] = r["mjd"]
        if not args.no_tsobc:
            tsobc_fields_by_year[y] = r["tsobc"]
        if not args.no_elevation:
            elevation_by_year[y] = r["elevation"]
    # jsasaki 2026-11-05: review round 2 #2: validate the actual periods and the joined axis
    check_time_axis(years, mjd_by_year, args.allow_year_gaps)
    siglev = results[0]["siglev"]
    siglay = results[0]["siglay"]
    obc_h_jcope_first = results[0]["obc_h_jcope"]

    # All three must be set after the loop runs at least once (years is
    # guaranteed non-empty above).
    assert siglev is not None
    assert siglay is not None
    assert obc_h_jcope_first is not None

    # ---- summary table (FVCOM h vs JCOPE h, computed once) ----
    print()
    print("OBC summary (node → FVCOM h vs JCOPE h):")
    print("  node  lat       lon        h_fvcom  h_jcope")
    for n, nid in enumerate(obc_ids):
        print(
            f"  {int(nid):4d}  {obc_lat[n]:8.4f}  {obc_lon[n]:8.4f}  "
            f"{float(obc_h_fvcom[n]):7.2f}  {float(obc_h_jcope_first[n]):7.2f}"
        )
    print()

    label = format_year_label(years)
    # jsasaki 2026-11-04: a FullDomainSource carries its own label
    # jsasaki 2026-11-05: review round 2 #18: collected from the workers' sources (what they read)
    source_files_ts = [line for r in results for line in r["prov_ts"]]
    source_files_el = [line for r in results for line in r["prov_el"]]

    if not args.no_tsobc:
        time_mjd = np.concatenate([mjd_by_year[y] for y in years])
        fields_concat = {
            v: np.concatenate([tsobc_fields_by_year[y][v] for y in years], axis=0)
            for v in tsobc_fields_by_year[years[0]]
        }
        out = args.output_dir / f"tb_tsobc_{args.tag}_{label}.nc"
        print(f"[write] {out}  (n_time={time_mjd.size})")
        write_tsobc_nc(
            out,
            time_mjd=time_mjd,
            obc_nodes=obc_ids,
            obc_h=obc_h_fvcom,
            siglev=siglev,
            siglay=siglay,
            fields=fields_concat,
            source_files=source_files_ts,  # jsasaki 2026-11-05: review round 2 #18: T/S provenance
            # jsasaki 2026-11-05: review round 13 #1: endpoint rule in the history
            history_note=tsobc_history_note,
        )

    if not args.no_elevation:
        time_mjd = np.concatenate([mjd_by_year[y] for y in years])
        elevation_concat = np.concatenate([elevation_by_year[y] for y in years], axis=0)
        out = args.output_dir / f"tb_julian_obc_{args.tag}_{label}.nc"
        print(f"[write] {out}  (n_time={time_mjd.size})")
        write_elevation_nc(
            out,
            time_mjd=time_mjd,
            obc_nodes=obc_ids,
            elevation=elevation_concat,
            source_files=source_files_el,  # jsasaki 2026-11-05: review round 2 #18: EGT provenance
        )

    grid.close()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

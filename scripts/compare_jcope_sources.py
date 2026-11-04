#!/usr/bin/env python3
# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Check that the full-domain archive and a jcopetda_region_YYYY.nc give the same OBC columns.

    python scripts/compare_jcope_sources.py --basic-nc B --region-nc R --archive-root A \
        --fvcom-grid G --fvcom-obc O --start 2020-01-01 --end 2020-01-03

The OBC cells are located exactly as run_jcope_obc.py does; the raw columns (TT, ST, UT, VT, EGT) of the
first hours of the region file and of the archive are compared. The archive is int16 packed with the
steps 0.01 (TT, ST) and 0.001 (UT, VT, EGT), so differences up to half a step are expected.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# jsasaki 2026-11-05: review round 2 #19: time-index comparison
import pandas as pd
import xarray as xr

# ---> jsasaki 2026-11-05: review round 10 #5: the script directory must be on
# sys.path BEFORE run_jcope_obc is imported (python -m scripts.compare_jcope_sources)
sys.path.insert(0, str(Path(__file__).resolve().parent))
# isort: off
from run_jcope_obc import parse_obc_dat  # noqa: E402

# isort: on
# <--- jsasaki 2026-11-05

from xfvcom.grid.grid_obj import FvcomGrid  # noqa: E402
from xfvcom.io import FullDomainSource, JcopeGrid  # noqa: E402

# jsasaki 2026-11-05: review round 5 #11: share the generator's regional-schema validation
from xfvcom.io.jcope_obc_generator import (  # noqa: E402
    _region_time_index,
    _validate_region_schema,
)

STEP = {"TT": 0.01, "ST": 0.01, "UT": 0.001, "VT": 0.001, "EGT": 0.001}
# jsasaki 2026-11-05: review round 3 #16: one explicit absolute axis tolerance (degree) for both
# sources, no relative part (rtol scaled by ~140 degrees would admit shifts of ~1e-3 degree)
AXIS_ATOL = 1e-6


# ---> jsasaki 2026-11-05: review round 15 #1: each compared variable must lie on the canonical
# dimensions (the checked coordinates), not merely have a matching shape
REGION_DIMS = {
    "TT": ("time", "level", "lat", "lon"),
    "ST": ("time", "level", "lat", "lon"),
    "UT": ("time", "level", "lat", "lon"),
    "VT": ("time", "level", "lat", "lon"),
    "EGT": ("time", "lat", "lon"),
}


def region_columns(reg, name: str, pos, rj, ri) -> np.ndarray:
    """The region file's ``name`` at time positions ``pos`` and cells (rj, ri): (time, [level,] n).

    The variable must use exactly the canonical dimension names (in any order); it is transposed
    to the canonical order explicitly, so a field on another vertical (or horizontal) dimension is
    refused instead of being compared by array position.
    """
    want = REGION_DIMS[name]
    da = reg[name]
    if len(da.dims) != len(want) or set(da.dims) != set(want):
        sys.exit(
            f"region variable {name} has dimensions {da.dims}; expected {want} (in any order)"
        )
    da = da.transpose(*want)
    sub = da.isel(time=pos).isel(
        lat=xr.DataArray(np.asarray(rj), dims="n"),
        lon=xr.DataArray(np.asarray(ri), dims="n"),
    )
    return sub.transpose("time", ..., "n").values


# <--- jsasaki 2026-11-05


def check_region_block(grid, reg) -> tuple[int, int]:
    """The region file's axes must be a contiguous block of the grid's; return its (j0, i0) offset."""
    # ---> jsasaki 2026-11-05: review round 3 #16: rtol=0 and an explicit absolute tolerance
    r_lat = np.asarray(reg["lat"].values, dtype=np.float64)
    r_lon = np.asarray(reg["lon"].values, dtype=np.float64)
    j0, i0 = grid.index_of(r_lat[0], r_lon[0])
    j0, i0 = int(j0), int(i0)
    b_lat = grid.lat[j0 : j0 + r_lat.size]
    b_lon = grid.lon[i0 : i0 + r_lon.size]
    if not (
        b_lat.shape == r_lat.shape
        and b_lon.shape == r_lon.shape
        and np.allclose(r_lat, b_lat, rtol=0, atol=AXIS_ATOL)
        and np.allclose(r_lon, b_lon, rtol=0, atol=AXIS_ATOL)
    ):
        sys.exit("the region file's axes are not a block of the grid's")
    return j0, i0
    # <--- jsasaki 2026-11-05


def check_alignment(src, grid, reg) -> np.ndarray:
    """Refuse misaligned sources; return the region-file positions of the archive's hours."""
    # ---> jsasaki 2026-11-05: review round 2 #19: the archive's own axes against basic.nc, the
    # vertical index of the region file, and the exact hours selected from both sources
    assert src.lat is not None and src.lon is not None
    # jsasaki 2026-11-05: review round 3 #16: the shared explicit tolerance
    if not (
        src.lat.shape == grid.lat.shape
        and src.lon.shape == grid.lon.shape
        and np.allclose(src.lat, grid.lat, rtol=0, atol=AXIS_ATOL)
        and np.allclose(src.lon, grid.lon, rtol=0, atol=AXIS_ATOL)
    ):
        sys.exit("the archive's lat/lon axes are not the grid file's")
    # ---> jsasaki 2026-11-05: review round 5 #11: the generator's regional schema (time/level/lat/lon
    # on their own dimensions, level exactly 0..45) and decoded, strictly increasing time are required
    label = Path(str(reg.encoding.get("source", "the region file")))
    try:
        _validate_region_schema(reg, label)
        r_time = _region_time_index(reg, label)
    except ValueError as exc:
        sys.exit(str(exc))
    # <--- jsasaki 2026-11-05
    want = pd.DatetimeIndex(src.time_index)
    pos = r_time.get_indexer(want)
    if (pos < 0).any():
        sys.exit(
            f"{int((pos < 0).sum())} hour(s) of the period are absent from the region file"
        )
    if not r_time[pos].equals(want):
        sys.exit("the region file's selected times differ from the archive's")
    return pos
    # <--- jsasaki 2026-11-05


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--basic-nc", type=Path, required=True)
    ap.add_argument("--region-nc", type=Path, required=True)
    ap.add_argument("--archive-root", type=Path, required=True)
    ap.add_argument("--fvcom-grid", type=Path, required=True)
    ap.add_argument("--fvcom-obc", type=Path, required=True)
    ap.add_argument("--utm-zone", type=int, default=54)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    a = ap.parse_args()

    mesh = FvcomGrid.from_dat(a.fvcom_grid, utm_zone=a.utm_zone)
    ids = parse_obc_dat(a.fvcom_obc)
    lat, lon = np.asarray(mesh.lat)[ids - 1], np.asarray(mesh.lon)[ids - 1]
    grid = JcopeGrid(a.basic_nc)
    jj, ii = grid.find_nearest_ocean(lat, lon)
    src = FullDomainSource(a.archive_root, a.start, a.end)
    reg = xr.open_dataset(a.region_nc)
    # jsasaki 2026-11-05: review round 3 #16: the axis-block check first, with rtol=0
    j0, i0 = check_region_block(grid, reg)
    rj, ri = jj - int(j0), ii - int(i0)
    if (
        (rj < 0).any()
        or (ri < 0).any()
        or (rj >= reg.sizes["lat"]).any()
        or (ri >= reg.sizes["lon"]).any()
    ):
        sys.exit("an OBC cell lies outside the region file")
    # jsasaki 2026-11-05: review round 2 #19: alignment checked before any value
    pos = check_alignment(src, grid, reg)
    bad = 0
    for name in ("TT", "ST", "UT", "VT", "EGT"):
        got = (
            src.read_profile(name, jj, ii)
            if name != "EGT"
            else src.read_2d(name, jj, ii)
        )
        # jsasaki 2026-11-05: review round 15 #1: canonical dimensions checked, explicit transpose
        ref = region_columns(reg, name, pos, rj, ri)
        # jsasaki 2026-11-05: review round 15 #1: an explicit error, not an assertion
        if got.shape != ref.shape:
            sys.exit(
                f"{name}: archive columns {got.shape} and region columns {ref.shape} differ in shape"
            )
        both = np.isfinite(got) & np.isfinite(ref)
        if not both.any():
            print(
                f"{name:3s}: no jointly finite values (archive finite {int(np.isfinite(got).sum())}, "
                f"region finite {int(np.isfinite(ref).sum())})  INSUFFICIENT"
            )
            bad += 1
            continue
        d = np.abs(got - ref)[both]
        nan_mismatch = int(np.count_nonzero(np.isfinite(got) != np.isfinite(ref)))
        ok = d.max() <= STEP[name] * 0.5 * (1 + 1e-3) + 1e-6 and nan_mismatch == 0
        bad += not ok
        print(
            f"{name:3s}: {both.sum()} values, max |archive - region| {d.max():.3g} "
            f"(half step {STEP[name]/2:g}), NaN mismatch {nan_mismatch}  {'OK' if ok else 'DIFFERS'}"
        )
    print("n_obc", ids.size, "n_time", src.n_time)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

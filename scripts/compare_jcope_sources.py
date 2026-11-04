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
import xarray as xr
from run_jcope_obc import parse_obc_dat  # noqa: E402

from xfvcom.grid.grid_obj import FvcomGrid  # noqa: E402
from xfvcom.io import FullDomainSource, JcopeGrid  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

STEP = {"TT": 0.01, "ST": 0.01, "UT": 0.001, "VT": 0.001, "EGT": 0.001}


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
    j0, i0 = grid.index_of(reg["lat"].values[0], reg["lon"].values[0])
    pts = dict(
        lat=xr.DataArray(jj - int(j0), dims="n"),
        lon=xr.DataArray(ii - int(i0), dims="n"),
    )
    tsel = dict(time=slice(src.time_index[0], src.time_index[-1]))
    bad = 0
    for name in ("TT", "ST", "UT", "VT", "EGT"):
        got = (
            src.read_profile(name, jj, ii)
            if name != "EGT"
            else src.read_2d(name, jj, ii)
        )
        sub = reg[name].sel(**tsel).isel(**pts)
        ref = sub.transpose("time", ..., "n").values
        assert got.shape == ref.shape, (name, got.shape, ref.shape)
        both = np.isfinite(got) & np.isfinite(ref)
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

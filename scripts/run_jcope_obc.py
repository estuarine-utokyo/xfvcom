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
import sys
from pathlib import Path

import numpy as np

from xfvcom.grid.grid_obj import FvcomGrid
from xfvcom.io import (
    JcopeGrid,
    JcopeObcGenerator,
    write_elevation_nc,
    write_tsobc_nc,
)


def parse_year_spec(spec: str) -> list[int]:
    """Accept a single year, a comma list, or a 'YYYY-YYYY' inclusive range."""
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",") if s.strip()]


def format_year_label(years: list[int]) -> str:
    if len(years) == 1:
        return str(years[0])
    return f"{min(years)}-{max(years)}"


def parse_obc_dat(path: Path) -> np.ndarray:
    """Return the 1-based OBC node IDs from a FVCOM ``*_obc.dat`` file."""
    nodes: list[int] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("OBC Node Number"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                nodes.append(int(parts[1]))
            except ValueError:
                continue
    if not nodes:
        raise ValueError(f"no OBC nodes parsed from {path}")
    return np.asarray(nodes, dtype=np.int32)


def parse_dep_dat(path: Path) -> np.ndarray:
    """Return the bathymetry column from a FVCOM ``*_dep.dat`` file."""
    depths: list[float] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Node Number"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                depths.append(float(parts[2]))
            except ValueError:
                continue
    if not depths:
        raise ValueError(f"no depths parsed from {path}")
    return np.asarray(depths, dtype=np.float32)


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
    parser.add_argument("--tag", default="jcope", help="filename tag (default: jcope)")
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

    if len(years) == 1 and args.region_nc is not None:
        region_paths = [args.region_nc]
    elif args.region_pattern is not None:
        region_paths = [Path(args.region_pattern.format(year=y)) for y in years]
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
    mesh = FvcomGrid.from_dat(args.fvcom_grid, utm_zone=args.utm_zone)
    if mesh.lat is None or mesh.lon is None:
        sys.exit("FvcomGrid lacks lon/lat — check utm_zone and the *_grd.dat file")
    n_nodes = mesh.node
    print(f"  nodes: {n_nodes}, elements: {mesh.nele}")

    print(f"[grid] reading FVCOM depth from {args.fvcom_dep}")
    h_all = parse_dep_dat(args.fvcom_dep)
    if h_all.size != n_nodes:
        sys.exit(
            f"depth file has {h_all.size} rows; grid has {n_nodes} nodes — mismatch"
        )

    print(f"[grid] reading OBC node list from {args.fvcom_obc}")
    obc_ids = parse_obc_dat(args.fvcom_obc)
    print(f"  OBC nodes ({obc_ids.size}): {obc_ids.tolist()}")

    idx0 = obc_ids - 1
    obc_lat = np.asarray(mesh.lat)[idx0]
    obc_lon = np.asarray(mesh.lon)[idx0]
    obc_h_fvcom = h_all[idx0]

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
    siglev = None
    siglay = None
    obc_h_jcope_first = None

    for year, region_nc in zip(years, region_paths):
        print(f"[jcope] opening region archive for {year}: {region_nc}")
        gen = JcopeObcGenerator(
            grid=grid,
            region_nc=region_nc,
            obc_nodes=obc_ids,
            obc_lat=obc_lat,
            obc_lon=obc_lon,
            obc_h_fvcom=obc_h_fvcom,
            n_siglay=args.n_siglay,
        )
        if siglev is None:
            siglev = gen.siglev
            siglay = gen.siglay
            obc_h_jcope_first = gen.obc_h_jcope.copy()

        mjd_by_year[year] = gen.mjd.copy()
        if not args.no_tsobc:
            tsobc_fields_by_year[year] = gen.build_tsobc_arrays()
        if not args.no_elevation:
            elevation_by_year[year] = gen.build_elevation_array()
        gen.close()

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
    source_files = [Path(p).name for p in region_paths]

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
            source_files=source_files,
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
            source_files=source_files,
        )

    grid.close()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

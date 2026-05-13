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
from xfvcom.io import JcopeGrid, JcopeObcGenerator


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
        required=True,
        help="path to jcopetda_region_YYYY.nc time-series archive",
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
    parser.add_argument(
        "--year", type=int, required=True, help="year stamp for output filename"
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

    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    # Translate 1-based IDs to 0-based array indices.
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

    print(f"[jcope] opening region archive at {args.region_nc}")
    gen = JcopeObcGenerator(
        grid=grid,
        region_nc=args.region_nc,
        obc_nodes=obc_ids,
        obc_lat=obc_lat,
        obc_lon=obc_lon,
        obc_h_fvcom=obc_h_fvcom,
        n_siglay=args.n_siglay,
    )

    print()
    print("OBC summary (node → FVCOM h vs JCOPE h):")
    print("  node  lat       lon        h_fvcom  h_jcope")
    for n, nid in enumerate(obc_ids):
        print(
            f"  {int(nid):4d}  {obc_lat[n]:8.4f}  {obc_lon[n]:8.4f}  "
            f"{float(obc_h_fvcom[n]):7.2f}  {float(gen.obc_h_jcope[n]):7.2f}"
        )
    print()

    if not args.no_tsobc:
        out = args.output_dir / f"tb_tsobc_{args.tag}_{args.year}.nc"
        print(f"[write] {out}")
        gen.write_tsobc(out)
    if not args.no_elevation:
        out = args.output_dir / f"tb_julian_obc_{args.tag}_{args.year}.nc"
        print(f"[write] {out}")
        gen.write_elevation(out)

    gen.close()
    grid.close()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

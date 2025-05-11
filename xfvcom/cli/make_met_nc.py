# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from xfvcom.io.met_nc_generator import MetNetCDFGenerator


def main() -> None:
    p = argparse.ArgumentParser(description="Generate constant meteorology NetCDF-4.")
    p.add_argument("grid", type=Path, help="FVCOM grid file (.dat or .nc)")

    p.add_argument("--start", required=True, help="UTC ISO time")
    p.add_argument("--end", required=True, help="UTC ISO time")
    p.add_argument("--dt", type=int, default=3600, help="Δt [s]")

    # constant parameters (any omitted key falls back to default)
    for key in MetNetCDFGenerator._DEFAULTS:
        p.add_argument(f"--{key}", type=float)

    p.add_argument("-o", "--output", type=Path, help="Output NetCDF")
    args = p.parse_args()

    gen = MetNetCDFGenerator(
        grid_nc=args.grid,
        start=args.start,
        end=args.end,
        dt_seconds=args.dt,
        **{k: getattr(args, k) for k in MetNetCDFGenerator._DEFAULTS},
    )

    out = args.output if args.output else args.grid.with_name("met.nc")
    gen.write(out)
    print(f"Written: {out}")


if __name__ == "__main__":
    main()

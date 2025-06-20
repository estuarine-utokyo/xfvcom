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
    p.add_argument(
        "--start-tz",
        default="UTC",
        help="Timezone for naive start/end (default: UTC)",
    )
    p.add_argument("--dt", type=int, default=3600, help="Δt [s]")

    p.add_argument(
        "--ts",
        action="append",
        metavar="SPEC",
        help="CSV/TSV time-series path[:var1,var2,…]",
    )
    p.add_argument(
        "--data-tz",
        default="Asia/Tokyo",
        help="Timezone of CSV/TSV data (default: Asia/Tokyo)",
    )

    # constant parameters (any omitted key falls back to default)
    for key in MetNetCDFGenerator._DEFAULTS:
        p.add_argument(f"--{key}", type=float)

    p.add_argument("-o", "--output", type=Path, help="Output NetCDF")
    p.add_argument(
        "--utm-zone",
        type=int,
        required=True,
        help="UTM zone number (e.g. 54 for Tokyo Bay)",
    )
    p.add_argument(
        "--southern",
        action="store_true",
        help="Use southern hemisphere UTM (default: northern)",
    )
    args = p.parse_args()

    gen = MetNetCDFGenerator(
        grid_nc=args.grid,
        start=args.start,
        end=args.end,
        dt_seconds=args.dt,
        utm_zone=args.utm_zone,
        northern=not args.southern,
        start_tz=args.start_tz,
        ts_specs=args.ts,
        data_tz=args.data_tz,
        **{k: getattr(args, k) for k in MetNetCDFGenerator._DEFAULTS},
    )

    out = args.output if args.output else args.grid.with_name("met.nc")
    gen.write(out)
    print(f"Written: {out}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from xfvcom.io.river_nc_generator import RiverNetCDFGenerator


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate river forcing NetCDF-4 from NML and time-series"
    )
    p.add_argument("nml", type=Path, help="rivers.nml file")
    p.add_argument("--start", required=True, help="ISO time, UTC")
    p.add_argument("--end", required=True, help="ISO time, UTC")
    p.add_argument("--dt", type=int, default=3600, help="time step [s]")

    # Default constants (CLI fallback)
    p.add_argument("--flux", type=float, default=0.0, help="default discharge")
    p.add_argument("--temp", type=float, default=20.0, help="default temperature")
    p.add_argument("--salt", type=float, default=0.0, help="default salinity")

    # Extended options
    p.add_argument(
        "--ts",
        action="append",
        metavar="SPEC",
        help="add time-series (RIVER=path:var or path:var)",
    )
    p.add_argument(
        "--const",
        action="append",
        metavar="SPEC",
        help="set constant value (RIVER.var=value or var=value)",
    )
    p.add_argument(
        "--config",
        type=Path,
        metavar="PATH",
        help="YAML config with defaults and river definitions",
    )

    p.add_argument("-o", "--output", type=Path, help="Output NetCDF file")

    args = p.parse_args()

    gen = RiverNetCDFGenerator(
        args.nml,
        args.start,
        args.end,
        args.dt,
        args.flux,
        args.temp,
        args.salt,
        ts_specs=args.ts,
        const_specs=args.const,
        config=args.config,
    )

    out_path = args.output if args.output else args.nml.with_suffix(".nc")
    gen.write(out_path)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()

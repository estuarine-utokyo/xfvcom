from __future__ import annotations

import argparse
from pathlib import Path

from xfvcom.io.river_nc_generator import RiverNetCDFGenerator


def main() -> None:
    p = argparse.ArgumentParser(description="Generate constant river NetCDF-4.")
    p.add_argument("nml", type=Path, help="rivers.nml file")
    p.add_argument("--start", required=True, help="ISO time, UTC")
    p.add_argument("--end", required=True, help="ISO time, UTC")
    p.add_argument("--dt", type=int, default=3600, help="time step [s]")
    p.add_argument("--flux", type=float, default=0.0)
    p.add_argument("--temp", type=float, default=20.0)
    p.add_argument("--salt", type=float, default=0.0)
    args = p.parse_args()

    gen = RiverNetCDFGenerator(
        args.nml, args.start, args.end, args.dt, args.flux, args.temp, args.salt
    )
    out_path = args.output if args.output else args.nml.with_suffix(".nc")
    gen.write(out_path)
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()

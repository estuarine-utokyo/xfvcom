# -*- coding: utf-8 -*-
"""Generate FVCOM atmospheric forcing NetCDF from a metforce OI archive.

This CLI is a focused wrapper around :class:`xfvcom.io.met_nc_generator.MetNetCDFGenerator`
in its metforce mode. It takes the unified ``fvcom_forcing_<YEAR>.nc`` produced
by the ``metforce`` toolkit (regular MSM-S / ERA5 native grid, eight variables
on ``(time, lat, lon)``) and bilinearly interpolates it onto an FVCOM mesh,
writing a FVCOM-compliant atmospheric forcing file
(``uwind_speed``, ``vwind_speed``, ``air_temperature``, ``relative_humidity``,
``air_pressure``, ``short_wave``, ``long_wave``, ``Precipitation`` and a
constant ``cloud_cover``).

See ``TB-FVCOM/hydro/docs/bc_construction_protocol.md`` §5.1 for the spec.

Example
-------
::

    xfvcom-make-met-nc-from-metforce path/to/TokyoBay18_grd.dat \
        --metforce-file ~/Github/metforce/analysis/fvcom_forcing_2020.nc \
        --start 2020 --utm-zone 54 \
        -o tb18_wnd_metforce_2020.nc
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xfvcom.io.gwo_reader import parse_period
from xfvcom.io.met_nc_generator import MetNetCDFGenerator


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Generate FVCOM atmospheric forcing NetCDF-4 by bilinearly "
            "interpolating a metforce OI archive onto an unstructured mesh."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:

  xfvcom-make-met-nc-from-metforce TokyoBay18_grd.dat \\
      --metforce-file ~/Github/metforce/analysis/fvcom_forcing_2020.nc \\
      --start 2020 --utm-zone 54 \\
      -o tb18_wnd_metforce_2020.nc
""",
    )
    p.add_argument("grid", type=Path, help="FVCOM grid file (.dat or .nc)")

    p.add_argument(
        "--metforce-file",
        type=Path,
        required=True,
        help=(
            "metforce unified analysis NetCDF "
            "(e.g. ~/Github/metforce/analysis/fvcom_forcing_<YEAR>.nc)"
        ),
    )

    # Time specification
    p.add_argument(
        "--start",
        required=True,
        help="Start time: year (2020), date (2020-01-01), or ISO datetime",
    )
    p.add_argument(
        "--end",
        help="End time (optional; defaults to year/date-end of --start)",
    )
    p.add_argument(
        "--start-tz",
        default="UTC",
        help="Timezone for naive start/end (default: UTC; metforce times are UTC)",
    )
    p.add_argument(
        "--dt",
        type=int,
        default=3600,
        help="Δt [s] (default: 3600; metforce native cadence is hourly)",
    )

    # Spatial interpolation behaviour
    p.add_argument(
        "--metforce-fallback",
        choices=["nearest", "none"],
        default="nearest",
        help=(
            "Interpolation for mesh points outside the metforce bbox "
            "(default: nearest). 'none' lets NaN propagate to expose "
            "coverage gaps."
        ),
    )

    # Trailing endpoint behaviour
    p.add_argument(
        "--pad-trailing-bookend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Extend the metforce source by one extra record at "
            "t_last + native_step, copying the last record's field values. "
            "Use when --end is exactly one source-hour past the metforce "
            "end (e.g. --end 2021-01-01T00:00:00 on a 2020 file ending at "
            "2020-12-31T23:00:00), so FVCOM's inclusive END_DATE finds a "
            "forcing sample at the closing timestep. Default: off."
        ),
    )

    # Output and grid options
    p.add_argument("-o", "--output", type=Path, help="Output NetCDF file path")
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

    # Constant values for variables NOT provided by metforce (currently only
    # cloud_cover). Other constants are accepted but normally redundant.
    const_group = p.add_argument_group(
        "Constant fallbacks (cloud_cover is the only var metforce does not provide)"
    )
    for key in MetNetCDFGenerator._DEFAULTS:
        const_group.add_argument(f"--{key}", type=float)

    args = p.parse_args()

    # Parse period
    start, end = parse_period(args.start, args.end)
    start_str = start.isoformat()
    end_str = end.isoformat()

    # Constants (only non-None pass through)
    const_vals = {
        k: getattr(args, k)
        for k in MetNetCDFGenerator._DEFAULTS
        if getattr(args, k) is not None
    }

    fallback = None if args.metforce_fallback == "none" else args.metforce_fallback

    gen = MetNetCDFGenerator(
        grid_nc=args.grid,
        start=start_str,
        end=end_str,
        dt_seconds=args.dt,
        utm_zone=args.utm_zone,
        northern=not args.southern,
        start_tz=args.start_tz,
        metforce_file=args.metforce_file,
        metforce_fallback=fallback,
        metforce_pad_trailing_bookend=args.pad_trailing_bookend,
        **const_vals,
    )

    out = args.output if args.output else args.grid.with_name("met.nc")
    gen.write(out)
    print(f"Written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

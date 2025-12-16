# -*- coding: utf-8 -*-
"""
Generate FVCOM meteorological forcing NetCDF files.

Supports multiple data sources:
- Constant values (default)
- CSV/TSV time series
- GWO-AMD meteorological data (with comprehensive gap filling)

Examples
--------
# Using constant values (default)
xfvcom-make-met-nc grid.dat --start 2020-01-01 --end 2020-12-31 --utm-zone 54

# Using GWO-AMD data
xfvcom-make-met-nc grid.dat --start 2020-01-01 --end 2021-01-01 \\
    --gwo-dir /path/to/GWO/Hourly \\
    --station-map "slht:Tokyo,kous:Tokyo,*:Chiba" \\
    --wind-factor 1.8 --utm-zone 54 -o output.nc

# With gap filling enabled
xfvcom-make-met-nc grid.dat --start 2020 --gwo-dir /path/to/GWO/Hourly \\
    --fill-gaps --fallback-stations "Chiba:Tokyo,Yokohama,Tateyama" \\
    --solar-model empirical --utm-zone 54 -o output.nc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xfvcom.io.gwo_reader import parse_period, parse_station_map
from xfvcom.io.met_nc_generator import MetNetCDFGenerator


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate FVCOM meteorological forcing NetCDF-4.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with constant values
  xfvcom-make-met-nc grid.dat --start 2020-01-01 --end 2020-12-31 --utm-zone 54

  # Generate from GWO-AMD data for year 2020
  xfvcom-make-met-nc grid.dat --start 2020 --gwo-dir /path/to/GWO/Hourly \\
      --station-map "slht:Tokyo,kous:Tokyo,*:Chiba" --utm-zone 54

  # With comprehensive gap filling
  xfvcom-make-met-nc grid.dat --start 2020 --gwo-dir /path/to/GWO/Hourly \\
      --fill-gaps --fallback-stations "Chiba:Tokyo,Yokohama,Tateyama" \\
      --solar-model empirical --utm-zone 54

  # Short-wave and precipitation from Tokyo, others from Chiba
  xfvcom-make-met-nc grid.dat --start 2020 --gwo-dir /path/to/GWO/Hourly \\
      --station-map "slht:Tokyo,kous:Tokyo,*:Chiba" --wind-factor 1.8 --utm-zone 54
""",
    )
    p.add_argument("grid", type=Path, help="FVCOM grid file (.dat or .nc)")

    # Time specification
    p.add_argument(
        "--start",
        required=True,
        help="Start time: year (2020), date (2020-01-01), or ISO datetime",
    )
    p.add_argument(
        "--end",
        help="End time (optional for year/date-only start)",
    )
    p.add_argument(
        "--start-tz",
        default="UTC",
        help="Timezone for naive start/end (default: UTC)",
    )
    p.add_argument("--dt", type=int, default=3600, help="Δt [s] (default: 3600)")

    # Time series source (legacy)
    p.add_argument(
        "--ts",
        action="append",
        metavar="SPEC",
        help="CSV/TSV time-series path[:var1,var2,…]",
    )
    p.add_argument(
        "--data-tz",
        default="Asia/Tokyo",
        help="Timezone of input data (default: Asia/Tokyo)",
    )

    # GWO-AMD source
    gwo_group = p.add_argument_group("GWO-AMD options")
    gwo_group.add_argument(
        "--gwo-dir",
        type=Path,
        metavar="PATH",
        help="Base directory for GWO hourly data (enables GWO mode)",
    )
    gwo_group.add_argument(
        "--station-map",
        metavar="SPEC",
        default="slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba",
        help='Variable-to-station mapping (default: "slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba")',
    )
    gwo_group.add_argument(
        "--wind-factor",
        type=float,
        default=1.8,
        help="Wind speed multiplier (default: 1.8)",
    )
    gwo_group.add_argument(
        "--max-gap-hours",
        type=int,
        default=6,
        help="Maximum hours to interpolate for missing data (default: 6)",
    )

    # Gap filling options
    gap_group = p.add_argument_group("Gap filling options (requires --gwo-dir)")
    gap_group.add_argument(
        "--fill-gaps",
        action="store_true",
        help="Enable comprehensive gap filling (4 steps: temporal, boundary, fallback, solar)",
    )
    gap_group.add_argument(
        "--fallback-stations",
        type=str,
        metavar="SPEC",
        help='Fallback station mapping, format: "Primary:FB1,FB2;Primary2:FB1,FB2"',
    )
    gap_group.add_argument(
        "--correlation-file",
        type=Path,
        metavar="PATH",
        help="YAML file with pre-computed station correlations",
    )
    gap_group.add_argument(
        "--solar-model",
        choices=["empirical", "pvlib-kasten", "pvlib-larson"],
        default="empirical",
        help="Solar radiation estimation model (default: empirical)",
    )
    gap_group.add_argument(
        "--no-extend-boundary",
        action="store_true",
        help="Disable using prev/next year data for boundary gaps",
    )
    gap_group.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if any gaps remain unfilled",
    )

    # Constant parameters (any omitted key falls back to default)
    const_group = p.add_argument_group("Constant values (fallback when not using GWO)")
    for key in MetNetCDFGenerator._DEFAULTS:
        const_group.add_argument(f"--{key}", type=float)

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

    args = p.parse_args()

    # Parse period specification
    start, end = parse_period(args.start, args.end)
    start_str = start.isoformat()
    end_str = end.isoformat()

    # Parse station map if GWO mode
    station_map = None
    if args.gwo_dir:
        station_map = parse_station_map(args.station_map)

    # Parse fallback stations if provided
    fallback_stations = None
    if args.fallback_stations:
        from xfvcom.io.gwo_correlations import parse_fallback_stations

        fallback_stations = parse_fallback_stations(args.fallback_stations)

    # Load correlations if provided
    correlations = None
    if args.correlation_file:
        from xfvcom.io.gwo_correlations import StationCorrelations

        correlations = StationCorrelations.from_yaml(args.correlation_file)

    # Build constant values dict (only non-None values)
    const_vals = {
        k: getattr(args, k)
        for k in MetNetCDFGenerator._DEFAULTS
        if getattr(args, k) is not None
    }

    gen = MetNetCDFGenerator(
        grid_nc=args.grid,
        start=start_str,
        end=end_str,
        dt_seconds=args.dt,
        utm_zone=args.utm_zone,
        northern=not args.southern,
        start_tz=args.start_tz,
        ts_specs=args.ts,
        data_tz=args.data_tz,
        gwo_dir=args.gwo_dir,
        station_map=station_map,
        wind_factor=args.wind_factor,
        max_gap_hours=args.max_gap_hours,
        fill_gaps=args.fill_gaps,
        fallback_stations=fallback_stations,
        correlations=correlations,
        solar_model=args.solar_model,
        extend_boundary=not args.no_extend_boundary,
        **const_vals,
    )

    out = args.output if args.output else args.grid.with_name("met.nc")

    # Run gap analysis/filling if in GWO mode
    if args.gwo_dir:
        if args.fill_gaps:
            # Run gap filling and get report
            gap_results = gen.run_gap_filling()
            if gap_results:
                from xfvcom.io.gwo_gap_filler import print_gap_fill_report

                print_gap_fill_report(gap_results)

                # Check for remaining gaps in strict mode
                remaining = sum(r.remaining for r in gap_results)
                if args.strict and remaining > 0:
                    print(
                        f"\n[ERROR] --strict mode: {remaining} gaps remain unfilled.",
                        file=sys.stderr,
                    )
                    return 1
        else:
            # Just report missing values
            missing_report = gen.analyze_missing_values()
            if missing_report:
                from xfvcom.io.gwo_gap_filler import print_missing_value_report

                print_missing_value_report(missing_report)

    gen.write(out)
    print(f"Written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""Generate FVCOM meteorological forcing NetCDF files.

Supports: constant values (default), CSV/TSV time series, GWO-AMD data with gap filling.
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
Example bash script (edit variables as needed):

  #!/bin/bash
  GRID=grid.dat
  YEAR=2020
  UTM_ZONE=54
  OUTPUT=met_forcing.nc
  # GWO-AMD options
  GWO_DIR=/path/to/GWO/Hourly
  STATION_MAP="slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba"
  WIND_FACTOR=1.8
  MAX_GAP_HOURS=6
  # Gap filling options
  FILL_GAPS=true
  FALLBACK_STATIONS="Chiba:Tokyo,Yokohama,Tateyama"
  SOLAR_MODEL=empirical

  xfvcom-make-met-nc "$GRID" --start "$YEAR" --utm-zone "$UTM_ZONE" \\
      --gwo-dir "$GWO_DIR" --station-map "$STATION_MAP" \\
      --wind-factor "$WIND_FACTOR" --max-gap-hours "$MAX_GAP_HOURS" \\
      $($FILL_GAPS && echo "--fill-gaps") \\
      --fallback-stations "$FALLBACK_STATIONS" \\
      --solar-model "$SOLAR_MODEL" \\
      -o "$OUTPUT"
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

    # MPOS source (Tokyo Bay Monitoring Post wind / air-temperature)
    mpos_group = p.add_argument_group("MPOS options (spatial wind from Tokyo Bay)")
    mpos_group.add_argument(
        "--mpos-dir",
        type=Path,
        metavar="PATH",
        help=(
            "Directory containing Mpos_K_{stn}_{year}.nc (typically "
            "$DATA_DIR/MPOS/nc_meteo). Enables MPOS mode: spatial wind / "
            "air-temperature from the 3-station Tokyo Bay network "
            "(02kawasaki excluded by default) interpolated onto the FVCOM "
            "mesh via inverse-distance weighting."
        ),
    )
    mpos_group.add_argument(
        "--mpos-stations",
        type=str,
        metavar="LIST",
        default=None,
        help=(
            "Comma-separated station identifiers, overriding the default "
            "exclusion list. Example: '01kemigawa,03urayasu,04chiba1gou'."
        ),
    )
    mpos_group.add_argument(
        "--mpos-idw-power",
        type=float,
        default=2.0,
        help="IDW power for spatial interpolation (default: 2.0)",
    )
    mpos_group.add_argument(
        "--mpos-keep-jst",
        action="store_true",
        help=(
            "Skip the JST->UTC conversion of MPOS timestamps. Use only "
            "when matching a non-standard FVCOM forcing convention."
        ),
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
    gap_group.add_argument(
        "--recompute-correlations",
        action="store_true",
        help="Force recompute correlations even if cache exists",
    )
    gap_group.add_argument(
        "--no-correlation-cache",
        action="store_true",
        help="Disable auto-caching of correlations (recompute every time)",
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

    # Load correlations
    correlations = None
    if args.correlation_file:
        # Explicit file specified - use it directly
        from xfvcom.io.gwo_correlations import StationCorrelations

        correlations = StationCorrelations.from_yaml(args.correlation_file)
    elif (
        args.fill_gaps
        and args.fallback_stations
        and not args.no_correlation_cache
        and args.gwo_dir
    ):
        # Auto-cache mode: use cache or compute correlations
        from xfvcom.io.gwo_correlations import get_cached_correlations

        # Extract all stations from fallback specification
        all_stations: set[str] = set()
        for primary, fallbacks in fallback_stations.items():
            all_stations.add(primary)
            all_stations.update(fallbacks)

        # Also add stations from station_map
        if station_map:
            for station in station_map.values():
                if station != "*":
                    all_stations.add(station)

        correlations = get_cached_correlations(
            gwo_dir=args.gwo_dir,
            stations=sorted(all_stations),
            force_recompute=args.recompute_correlations,
        )

    # Build constant values dict (only non-None values)
    const_vals = {
        k: getattr(args, k)
        for k in MetNetCDFGenerator._DEFAULTS
        if getattr(args, k) is not None
    }

    mpos_stations: tuple[str, ...] | None = None
    if args.mpos_stations:
        mpos_stations = tuple(s.strip() for s in args.mpos_stations.split(","))

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
        mpos_dir=args.mpos_dir,
        mpos_stations=mpos_stations,
        mpos_idw_power=args.mpos_idw_power,
        mpos_convert_to_utc=not args.mpos_keep_jst,
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

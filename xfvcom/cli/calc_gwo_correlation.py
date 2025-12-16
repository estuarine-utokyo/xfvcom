# -*- coding: utf-8 -*-
"""
Calculate correlations between GWO weather stations.

This CLI tool computes correlation parameters between weather stations
for use in gap filling. It can also build and compare solar radiation
estimation models.

Examples
--------
# Compute correlations between stations
xfvcom-calc-gwo-corr --gwo-dir /path/to/GWO/Hourly \\
    --stations Tokyo,Chiba,Yokohama,Tateyama \\
    --years 2015-2020 \\
    -o gwo_correlations.yaml

# Include solar model calibration
xfvcom-calc-gwo-corr --gwo-dir /path/to/GWO/Hourly \\
    --stations Tokyo \\
    --years 2015-2020 \\
    --include-solar-model \\
    -o gwo_correlations.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_years(spec: str) -> list[int]:
    """Parse year specification (e.g., '2015-2020' or '2015,2016,2017')."""
    if "-" in spec:
        parts = spec.split("-")
        if len(parts) == 2:
            return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(y.strip()) for y in spec.split(",")]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Calculate correlations between GWO weather stations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute correlations between stations
  xfvcom-calc-gwo-corr --gwo-dir /path/to/GWO/Hourly \\
      --stations Tokyo,Chiba,Yokohama,Tateyama \\
      --years 2015-2020 \\
      -o gwo_correlations.yaml

  # Include solar model calibration
  xfvcom-calc-gwo-corr --gwo-dir /path/to/GWO/Hourly \\
      --stations Tokyo \\
      --years 2015-2020 \\
      --include-solar-model \\
      -o gwo_correlations.yaml

  # Compare solar models
  xfvcom-calc-gwo-corr --gwo-dir /path/to/GWO/Hourly \\
      --compare-solar-models \\
      --train-years 2015-2019 \\
      --test-year 2020 \\
      --stations Tokyo
""",
    )

    p.add_argument(
        "--gwo-dir",
        type=Path,
        help="Base directory for GWO hourly data",
    )
    p.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the correlation cache and exit",
    )
    p.add_argument(
        "--stations",
        type=str,
        help="Comma-separated list of stations (e.g., 'Tokyo,Chiba,Yokohama')",
    )
    p.add_argument(
        "--years",
        type=str,
        default="2015-2020",
        help="Year range for correlation computation (e.g., '2015-2020')",
    )
    p.add_argument(
        "--variables",
        type=str,
        default="kion,rhum,shpa,sped,clod",
        help="Comma-separated list of variables to compute correlations for",
    )
    p.add_argument(
        "--include-solar-model",
        action="store_true",
        help="Also compute solar model parameters",
    )
    p.add_argument(
        "--compare-solar-models",
        action="store_true",
        help="Compare different solar estimation models",
    )
    p.add_argument(
        "--train-years",
        type=str,
        help="Years for solar model training (default: same as --years)",
    )
    p.add_argument(
        "--test-year",
        type=int,
        default=2020,
        help="Year for solar model testing (default: 2020)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output YAML file path",
    )

    args = p.parse_args()

    # Handle --clear-cache
    if args.clear_cache:
        from xfvcom.io.gwo_correlations import clear_correlation_cache, get_cache_path

        cache_path = get_cache_path()
        if clear_correlation_cache():
            print(f"[OK] Cleared cache: {cache_path}")
        else:
            print(f"[INFO] No cache file found at: {cache_path}")
        return 0

    # Validate required arguments for normal operation
    if not args.gwo_dir:
        p.error("--gwo-dir is required when not using --clear-cache")
    if not args.stations:
        p.error("--stations is required when not using --clear-cache")
    if not args.output:
        p.error("-o/--output is required when not using --clear-cache")

    # Parse arguments
    stations = [s.strip() for s in args.stations.split(",")]
    years = parse_years(args.years)
    variables = [v.strip() for v in args.variables.split(",")]
    train_years = parse_years(args.train_years) if args.train_years else years

    # Import modules
    from xfvcom.io.gwo_correlations import StationCorrelations
    from xfvcom.io.gwo_reader import GWOReader

    reader = GWOReader(args.gwo_dir)

    print(f"Computing correlations for stations: {', '.join(stations)}")
    print(f"Years: {years[0]}-{years[-1]}")
    print(f"Variables: {', '.join(variables)}")

    # Generate all station pairs
    station_pairs: list[tuple[str, str]] = []
    for i, s1 in enumerate(stations):
        for s2 in stations[i + 1 :]:
            station_pairs.append((s1, s2))
            station_pairs.append((s2, s1))  # Both directions

    if not station_pairs:
        print("[WARNING] Only one station specified, no pairs to compute.")
        station_pairs = []

    # Compute correlations
    print("\nComputing station correlations...")
    correlations = StationCorrelations.compute(
        gwo_reader=reader,
        station_pairs=station_pairs,
        years=years,
        variables=variables,
    )

    # Print summary
    summary = correlations.summary()
    if not summary.empty:
        print("\nCorrelation Summary:")
        print("-" * 80)
        print(
            f"{'Primary':<12} {'Secondary':<12} {'Variable':<8} "
            f"{'R²':>8} {'Slope':>8} {'Intercept':>10} {'RMSE':>8}"
        )
        print("-" * 80)
        for _, row in summary.iterrows():
            print(
                f"{row['primary']:<12} {row['secondary']:<12} {row['variable']:<8} "
                f"{row['r2']:>8.4f} {row['slope']:>8.4f} {row['intercept']:>10.4f} "
                f"{row['rmse']:>8.4f}"
            )

    # Add solar model parameters if requested
    if args.include_solar_model:
        print("\nComputing solar model parameters...")
        try:
            from xfvcom.io.solar_estimation import SolarEstimator

            # Use first station for solar model
            solar_station = stations[0] if stations else "Tokyo"
            params = SolarEstimator.build_empirical_model(
                gwo_reader=reader,
                station=solar_station,
                years=train_years,
            )

            print(f"\nSolar Model Parameters (station: {solar_station}):")
            print(f"  a = {params.a:.4f}")
            print(f"  b = {params.b:.4f}")
            print(f"  R² = {params.r2:.4f}" if params.r2 else "  R² = N/A")
            print(f"  RMSE = {params.rmse:.2f} W/m²" if params.rmse else "  RMSE = N/A")
            print(
                f"  Samples = {params.n_samples}"
                if params.n_samples
                else "  Samples = N/A"
            )

            # Add to output data
            correlations.data["solar_model"] = {
                "station": solar_station,
                "a": params.a,
                "b": params.b,
                "r2": params.r2,
                "rmse": params.rmse,
                "n_samples": params.n_samples,
                "train_years": train_years,
            }

        except ImportError:
            print("[WARNING] pvlib not installed, skipping solar model computation.")
        except Exception as e:
            print(f"[WARNING] Solar model computation failed: {e}")

    # Compare solar models if requested
    if args.compare_solar_models:
        print("\nComparing solar estimation models...")
        try:
            from xfvcom.io.solar_estimation import SolarEstimator

            solar_station = stations[0] if stations else "Tokyo"
            comparison = SolarEstimator.compare_models(
                gwo_reader=reader,
                station=solar_station,
                train_years=train_years,
                test_year=args.test_year,
            )

            print(f"\nSolar Model Comparison (test year: {args.test_year}):")
            print("-" * 60)
            print(f"{'Model':<20} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'Bias':>10}")
            print("-" * 60)
            for _, row in comparison.iterrows():
                print(
                    f"{row['model']:<20} {row['r2']:>8.4f} {row['rmse']:>10.2f} "
                    f"{row['mae']:>10.2f} {row['bias']:>10.2f}"
                )

            # Add to output data
            correlations.data["solar_model_comparison"] = comparison.to_dict(
                orient="records"
            )

        except ImportError:
            print("[WARNING] pvlib not installed, skipping model comparison.")
        except Exception as e:
            print(f"[WARNING] Model comparison failed: {e}")

    # Save to YAML
    correlations.to_yaml(args.output)
    print(f"\nWritten: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

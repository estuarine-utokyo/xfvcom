# -*- coding: utf-8 -*-
"""CLI for checking meteorological forcing NetCDF files for abnormal values."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xfvcom.validation import MetValidator, PhysicalBounds


def _parse_bound_spec(spec: str) -> tuple[str, PhysicalBounds]:
    """Parse bound specification string.

    Format: VAR:MIN:MAX or VAR:MIN: or VAR::MAX

    Parameters
    ----------
    spec : str
        Bound specification string.

    Returns
    -------
    tuple[str, PhysicalBounds]
        Variable name and bounds.
    """
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid bound spec '{spec}'. Expected format: VAR:MIN:MAX")

    var_name = parts[0].strip()
    min_str = parts[1].strip()
    max_str = parts[2].strip()

    min_val = float(min_str) if min_str else None
    max_val = float(max_str) if max_str else None

    return var_name, PhysicalBounds(min_val, max_val, "user-defined")


def main(args: list[str] | None = None) -> int:
    """Main entry point for xfvcom-check-met CLI.

    Parameters
    ----------
    args : list[str] | None
        Command-line arguments (None = sys.argv).

    Returns
    -------
    int
        Exit code (0 = clean, 1 = anomalies found, 2 = error).
    """
    parser = argparse.ArgumentParser(
        prog="xf-check-met",
        description="Check meteorological forcing NetCDF for abnormal values.",
        epilog=(
            "Examples:\n"
            "  xf-check-met met_forcing.nc\n"
            "  xf-check-met met_forcing.nc --no-bounds\n"
            "  xf-check-met met_forcing.nc -o report.csv\n"
            "  xf-check-met met_forcing.nc --bound 'air_temperature:-40:50'\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input meteorological forcing NetCDF file",
    )
    parser.add_argument(
        "--no-bounds",
        action="store_true",
        help="Skip physical bounds checking (only check NaN/Inf)",
    )
    parser.add_argument(
        "--no-nan",
        action="store_true",
        help="Skip NaN checking",
    )
    parser.add_argument(
        "--no-inf",
        action="store_true",
        help="Skip Inf checking",
    )
    parser.add_argument(
        "--max-per-var",
        type=int,
        default=100,
        metavar="N",
        help="Max anomalies to report per variable (default: 100, 0 = unlimited)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="CSV",
        help="Output CSV file for detailed report",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print summary, not individual anomalies",
    )
    parser.add_argument(
        "--bound",
        action="append",
        metavar="SPEC",
        help="Override bounds for variable (format: VAR:MIN:MAX)",
    )
    parser.add_argument(
        "--var",
        action="append",
        metavar="NAME",
        help="Only check specific variable(s)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )

    parsed = parser.parse_args(args)

    # Validate input file
    if not parsed.input.exists():
        print(f"Error: File not found: {parsed.input}", file=sys.stderr)
        return 2

    # Parse custom bounds
    custom_bounds: dict[str, PhysicalBounds] | None = None
    if parsed.bound:
        custom_bounds = {}
        for spec in parsed.bound:
            try:
                var_name, bounds = _parse_bound_spec(spec)
                custom_bounds[var_name] = bounds
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 2

    # Run validation
    try:
        validator = MetValidator(parsed.input)
        reports = validator.validate(
            check_nan=not parsed.no_nan,
            check_inf=not parsed.no_inf,
            check_bounds=not parsed.no_bounds,
            custom_bounds=custom_bounds,
            max_reports_per_var=parsed.max_per_var,
            variables=parsed.var,
            show_progress=not parsed.no_progress,
        )
    except Exception as e:
        print(f"Error during validation: {e}", file=sys.stderr)
        return 2

    # Generate summary
    if parsed.quiet:
        # Quiet mode: only summary stats
        total_nan = sum(r.total_nan for r in reports)
        total_inf = sum(r.total_inf for r in reports)
        total_bounds = sum(r.total_below_min + r.total_above_max for r in reports)
        total = total_nan + total_inf + total_bounds

        print(f"File: {parsed.input.name}")
        print(
            f"Total anomalies: {total} (NaN: {total_nan}, Inf: {total_inf}, "
            f"Bounds: {total_bounds})"
        )
        status = "PASSED" if total == 0 else "FAILED"
        print(f"Status: {status}")
    else:
        # Full summary
        print(validator.summary(reports))

    # Export to CSV if requested
    if parsed.output:
        df = validator.to_dataframe(reports)
        df.to_csv(parsed.output, index=False)
        print(f"\nDetailed report written to: {parsed.output}")

    # Return appropriate exit code
    total_anomalies = sum(r.total_anomalies for r in reports)
    return 0 if total_anomalies == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

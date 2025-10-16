#!/usr/bin/env python3
"""CLI for creating stacked DYE concentration plots."""

from __future__ import annotations

import argparse
import sys

import xarray as xr

from xfvcom.plot import plot_dye_timeseries_stacked


def main() -> None:
    """CLI entry point for xfvcom-dye-ts."""
    parser = argparse.ArgumentParser(
        description="Create stacked area plot of DYE concentration time series",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple stacked plot with all members
  xfvcom-dye-ts --input data.nc --var dye --output stacked.png

  # Select specific members
  xfvcom-dye-ts --input data.nc --var dye --member-ids 1 2 3 4 5 \\
    --output stacked_selected.png

  # Focus on a time window
  xfvcom-dye-ts --input data.nc --var dye --member-ids 1 2 3 \\
    --start 2021-01-01 --end 2021-01-31 --output jan2021.png
        """,
    )

    # Input/output
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input NetCDF file containing DYE data",
    )
    parser.add_argument(
        "--var",
        "-v",
        type=str,
        default="dye",
        help="Variable name in the NetCDF file (default: dye)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path for the plot (PNG recommended)",
    )

    # Member selection
    parser.add_argument(
        "--member-ids",
        type=int,
        nargs="+",
        help="List of member IDs to plot (e.g., 1 2 3 4 5). If not specified, plots all members.",
    )

    # Time window
    parser.add_argument(
        "--start",
        type=str,
        help="Start time for the plot window (e.g., 2021-01-01)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End time for the plot window (e.g., 2021-01-31)",
    )

    # Plot styling
    parser.add_argument(
        "--title",
        type=str,
        help="Plot title. If not specified, uses default title.",
    )
    parser.add_argument(
        "--ylabel",
        type=str,
        default="Dye Concentration",
        help="Y-axis label (default: 'Dye Concentration')",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=[14, 6],
        help="Figure size in inches (default: 14 6)",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading {args.input}...", file=sys.stderr)
    try:
        ds = xr.open_dataset(args.input)
        if args.var not in ds:
            print(
                f"Error: Variable '{args.var}' not found in dataset.",
                file=sys.stderr,
            )
            print(f"Available variables: {list(ds.data_vars)}", file=sys.stderr)
            sys.exit(1)
        data = ds[args.var]
    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        sys.exit(1)

    # Create plot
    print("Creating stacked area plot...", file=sys.stderr)
    try:
        result = plot_dye_timeseries_stacked(
            data=data,
            member_ids=args.member_ids,
            start=args.start,
            end=args.end,
            figsize=tuple(args.figsize),
            title=args.title,
            ylabel=args.ylabel,
            output=args.output,
        )
        print(f"✓ Saved plot to: {args.output}", file=sys.stderr)
        print(
            f"  Members plotted: {len(result['legend_labels'])}",
            file=sys.stderr,
        )
        print(
            f"  Timesteps: {len(result['data_used'])}",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"Error creating plot: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

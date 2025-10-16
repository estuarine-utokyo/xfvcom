#!/usr/bin/env python3
"""Example script for extracting multi-year dye time series from FVCOM outputs.

This is a plain example script demonstrating programmatic usage of
xfvcom.dye_timeseries module. This is NOT the official CLI entry point.

For the official CLI, use: xfvcom-dye-ts (see xfvcom/cli/dye_timeseries.py)

Examples:
    # Basic extraction
    python plot_dye_timeseries.py --years 2021 --members 0 1 2 --nodes 123,456 --sigmas 0,1

    # With linearity check and output
    python plot_dye_timeseries.py --years 2021 2022 --members 0 1 18 \\
        --nodes 123 --sigmas 0 --verify-linearity --ref-member 0 --parts 1 18 \\
        --save output/dye_series.nc

    # Climatology mode
    python plot_dye_timeseries.py --years 2020 2021 2022 --members 0 1 2 \\
        --nodes 100,200,300 --sigmas 0,1,2 --align climatology \\
        --save output/dye_climatology.zarr
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import zarr

# Add parent directory to path for xfvcom import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xfvcom.dye_timeseries import (
    AlignPolicy,
    DyeCase,
    NegPolicy,
    Paths,
    Selection,
    aggregate,
    collect_member_files,
    negative_stats,
    verify_linearity,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_int_list(value: str) -> list[int]:
    """Parse comma-separated or space-separated integer list."""
    if "," in value:
        return [int(x.strip()) for x in value.split(",")]
    else:
        return [int(value)]


def prepare_for_zarr(ds, time_chunk_size: int = 100):
    """Prepare dataset for Zarr export with proper chunking and encoding.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to prepare
    time_chunk_size : int
        Chunk size for time dimension (default: 100)

    Returns
    -------
    tuple
        Chunked dataset and encoding dictionary
    """
    # Create chunk dictionary
    chunk_dict = {}
    for dim in ds.dims:
        if dim == "time":
            chunk_dict[dim] = min(time_chunk_size, len(ds[dim]))
        else:
            # Keep other dimensions unchunked for efficient access
            chunk_dict[dim] = len(ds[dim])

    # Chunk the dataset
    ds_chunked = ds.chunk(chunk_dict)

    # Create encoding with explicit chunks for all variables
    encoding = {}
    for var in ds_chunked.data_vars:
        var_chunks = []
        for dim in ds_chunked[var].dims:
            var_chunks.append(chunk_dict.get(dim, 1))
        encoding[var] = {"chunks": tuple(var_chunks)}

    # Add encoding for coordinates with dimensions (excluding scalar coords)
    for coord in ds_chunked.coords:
        if ds_chunked[coord].dims:  # Only if coordinate has dimensions
            coord_chunks = []
            for dim in ds_chunked[coord].dims:
                coord_chunks.append(chunk_dict.get(dim, 1))
            encoding[coord] = {"chunks": tuple(coord_chunks)}

    return ds_chunked, encoding


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract multi-year dye time series from FVCOM outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        required=True,
        help="Years to process (e.g., --years 2020 2021 2022)",
    )
    parser.add_argument(
        "--members",
        type=int,
        nargs="+",
        required=True,
        help="Member IDs to process (e.g., --members 0 1 2 18)",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        required=True,
        help="Node indices (comma-separated or space-separated, e.g., '123,456' or '123 456')",
    )
    parser.add_argument(
        "--sigmas",
        type=str,
        required=True,
        help="Sigma layer indices (0-based, e.g., '0,1' or '0 1 2')",
    )

    # Optional arguments
    parser.add_argument(
        "--basename",
        type=str,
        default="tb_w18_r16",
        help="Base name of output files (default: tb_w18_r16)",
    )
    parser.add_argument(
        "--neg-policy",
        type=str,
        choices=["keep", "clip_zero"],
        default="keep",
        help="Policy for negative values (default: keep)",
    )
    parser.add_argument(
        "--align",
        type=str,
        choices=["native_intersection", "same_calendar", "climatology"],
        default="native_intersection",
        help="Time alignment mode (default: native_intersection)",
    )
    parser.add_argument(
        "--tb-fvcom-dir",
        type=str,
        default=None,
        help="Path to TB-FVCOM directory (default: ../../TB-FVCOM from script location, or $TB_FVCOM_DIR)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output file path (.nc for NetCDF, .zarr for Zarr)",
    )

    # Analysis options
    parser.add_argument(
        "--verify-linearity",
        action="store_true",
        help="Verify linearity assumption (ref = sum(parts))",
    )
    parser.add_argument(
        "--ref-member",
        type=int,
        default=0,
        help="Reference member for linearity check (default: 0)",
    )
    parser.add_argument(
        "--parts",
        type=int,
        nargs="*",
        default=None,
        help="Part members for linearity check (default: all except ref)",
    )
    parser.add_argument(
        "--print-neg-stats",
        action="store_true",
        help="Print negative value statistics",
    )

    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (DEBUG level)",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("xfvcom").setLevel(logging.DEBUG)

    # Resolve TB-FVCOM directory
    if args.tb_fvcom_dir:
        tb_fvcom_dir = Path(args.tb_fvcom_dir)
    elif os.environ.get("TB_FVCOM_DIR"):
        tb_fvcom_dir = Path(os.environ["TB_FVCOM_DIR"])
    else:
        # Default: ~/Github/TB-FVCOM (absolute path from home directory)
        tb_fvcom_dir = Path("~/Github/TB-FVCOM").expanduser()

    # Validate TB-FVCOM directory
    output_dir = tb_fvcom_dir / "goto2023" / "dye_run" / "output"
    if not output_dir.exists():
        logger.error(
            f"TB-FVCOM output directory not found: {output_dir}\n"
            f"  Searched in: {tb_fvcom_dir}\n"
            f"  Please specify correct path via --tb-fvcom-dir or $TB_FVCOM_DIR"
        )
        sys.exit(1)

    # Parse node and sigma lists
    nodes = (
        parse_int_list(args.nodes)
        if "," in args.nodes
        else [int(x) for x in args.nodes.split()]
    )
    sigmas = (
        parse_int_list(args.sigmas)
        if "," in args.sigmas
        else [int(x) for x in args.sigmas.split()]
    )

    # Build configuration objects
    paths = Paths(tb_fvcom_dir=tb_fvcom_dir)
    case = DyeCase(basename=args.basename, years=args.years, members=args.members)
    sel = Selection(nodes=nodes, sigmas=sigmas)
    neg = NegPolicy(mode=args.neg_policy)  # type: ignore[arg-type]
    align_policy = AlignPolicy(mode=args.align)  # type: ignore[arg-type]

    # Print run summary
    print("\n" + "=" * 70)
    print("DYE TIME SERIES EXTRACTION")
    print("=" * 70)
    print(f"Basename:       {case.basename}")
    print(f"Years:          {case.years}")
    print(f"Members:        {case.members}")
    print(f"Nodes:          {sel.nodes}")
    print(f"Sigma layers:   {sel.sigmas}")
    print(f"Negative policy: {neg.mode}")
    print(f"Alignment:      {align_policy.mode}")
    print(f"TB-FVCOM dir:   {tb_fvcom_dir}")
    print("=" * 70 + "\n")

    # Collect files
    logger.info("Collecting member files...")
    member_map = collect_member_files(paths, case)

    # Aggregate time series
    logger.info("Aggregating time series...")
    ds = aggregate(member_map, case, sel, neg, align_policy)

    # Compute negative statistics
    if args.print_neg_stats:
        logger.info("Computing negative value statistics...")
        # Load raw series for accurate stats (before clipping)
        from xfvcom.dye_timeseries import load_member_series

        series_dict = {}
        for (year, member), files in member_map.items():
            series = load_member_series(
                files, case, sel, NegPolicy(mode="keep"), year, member
            )
            series_dict[(year, member)] = series

        stats = negative_stats(ds, series_dict)

        print("\n" + "=" * 70)
        print("NEGATIVE VALUE STATISTICS")
        print("=" * 70)
        print(json.dumps(stats, indent=2))
        print("=" * 70 + "\n")

    # Linearity verification
    if args.verify_linearity:
        logger.info("Verifying linearity...")
        linearity_metrics = verify_linearity(
            ds, ref_member=args.ref_member, parts=args.parts
        )

        print("\n" + "=" * 70)
        print("LINEARITY VERIFICATION")
        print("=" * 70)
        print(json.dumps(linearity_metrics, indent=2))
        print("=" * 70 + "\n")

    # Save output
    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving to {save_path}...")

        # NetCDF doesn't support MultiIndex - convert to regular coordinates
        ds_to_save = (
            ds.reset_index("ensemble")
            if "ensemble" in ds.indexes
            and isinstance(ds.indexes["ensemble"], pd.MultiIndex)
            else ds
        )

        if save_path.suffix == ".nc":
            ds_to_save.to_netcdf(save_path)
            logger.info(f"Saved NetCDF: {save_path}")
            if ds_to_save is not ds:
                logger.info(
                    "Note: MultiIndex 'ensemble' was converted to separate 'year' and 'member' coordinates"
                )
        elif save_path.suffix == ".zarr":
            # Prepare dataset with proper encoding for Zarr compatibility
            ds_to_save_chunked, encoding = prepare_for_zarr(
                ds_to_save, time_chunk_size=100
            )
            ds_to_save_chunked.to_zarr(
                save_path,
                mode="w",
                encoding=encoding,
                zarr_version=2,
                consolidated=False,
            )
            logger.info(f"Saved Zarr: {save_path}")
            if ds_to_save is not ds:
                logger.info(
                    "Note: MultiIndex 'ensemble' was converted to separate 'year' and 'member' coordinates"
                )
        else:
            logger.warning(f"Unknown format: {save_path.suffix}. Saving as NetCDF...")
            ds_to_save.to_netcdf(save_path.with_suffix(".nc"))
            save_path = save_path.with_suffix(".nc")

        print(f"\n✓ Output saved to: {save_path}\n")

    # Final summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Time steps:     {ds.sizes.get('time', 'N/A')}")
    print(f"Ensemble size:  {ds.sizes.get('ensemble', 'N/A')}")
    print(f"Variables:      {list(ds.data_vars)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

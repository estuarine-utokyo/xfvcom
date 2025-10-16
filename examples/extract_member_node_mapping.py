#!/usr/bin/env python3
"""Extract member-node mappings from FVCOM namelist files.

This script parses FVCOM namelist files to extract which nodes are active
in each ensemble member and exports the results to various formats.

Examples:
    # Extract all members and save to CSV
    python extract_member_node_mapping.py --year 2021 --members 0 1 2 3 4 5

    # Extract with custom paths
    python extract_member_node_mapping.py \\
        --nml-dir ~/Github/TB-FVCOM/goto2023/dye_run \\
        --year 2021 \\
        --members $(seq 0 18) \\
        --output member_mapping.csv

    # Export to multiple formats
    python extract_member_node_mapping.py \\
        --year 2021 \\
        --members 0 1 2 3 \\
        --output mapping.csv \\
        --summary summary.csv \\
        --markdown mapping.md

    # Extract coordinates from NetCDF
    python extract_member_node_mapping.py \\
        --year 2021 \\
        --members 0 1 2 \\
        --output mapping.csv \\
        --nc-file ~/Github/TB-FVCOM/goto2023/dye_run/output/2021/0/tb_w18_r16_2021_0_0001.nc \\
        --coords coords.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for xfvcom import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xfvcom.ensemble_analysis.member_info import (
    export_member_mapping,
    extract_member_node_mapping,
    get_member_summary,
    get_node_coordinates,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract member-node mappings from FVCOM namelist files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input arguments
    parser.add_argument(
        "--nml-dir",
        type=str,
        default="~/Github/TB-FVCOM/goto2023/dye_run",
        help="Directory containing namelist files (default: ~/Github/TB-FVCOM/goto2023/dye_run)",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="tb_w18_r16",
        help="Case basename (default: tb_w18_r16)",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year of the run",
    )
    parser.add_argument(
        "--members",
        type=int,
        nargs="+",
        required=True,
        help="List of member IDs to process (space-separated)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="member_node_mapping.csv",
        help="Output file for full mapping (default: member_node_mapping.csv)",
    )
    parser.add_argument(
        "--summary",
        type=str,
        help="Output file for member summary (optional)",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        help="Output file for markdown format (optional)",
    )

    # Coordinate extraction
    parser.add_argument(
        "--nc-file",
        type=str,
        help="NetCDF file to extract node coordinates from (optional)",
    )
    parser.add_argument(
        "--coords",
        type=str,
        help="Output file for node coordinates (requires --nc-file)",
    )

    # Display options
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress messages",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Expand paths
    nml_dir = Path(args.nml_dir).expanduser()
    output_path = Path(args.output)

    if not nml_dir.exists():
        print(f"Error: Namelist directory not found: {nml_dir}", file=sys.stderr)
        sys.exit(1)

    # Extract member-node mapping
    if not args.quiet:
        print("=" * 80)
        print("MEMBER-NODE MAPPING EXTRACTION")
        print("=" * 80)
        print(f"Namelist directory: {nml_dir}")
        print(f"Case: {args.basename}")
        print(f"Year: {args.year}")
        print(f"Members: {args.members}")
        print()

    mapping_df = extract_member_node_mapping(
        nml_dir=nml_dir,
        basename=args.basename,
        year=args.year,
        members=args.members,
    )

    if mapping_df.empty:
        print("Error: No data extracted. Check namelist files exist.", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"✓ Extracted {len(mapping_df)} source-member combinations")
        print(f"  Unique nodes: {len(mapping_df['node_id'].unique())}")
        print(f"  Unique sources: {len(mapping_df['source_name'].unique())}")
        print()

    # Export full mapping
    if not args.quiet:
        print(f"Exporting full mapping to: {output_path}")

    export_member_mapping(
        mapping_df,
        output_path,
        format='csv',
    )

    # Export summary if requested
    if args.summary:
        if not args.quiet:
            print(f"Generating member summary...")

        summary_df = get_member_summary(
            nml_dir=nml_dir,
            basename=args.basename,
            year=args.year,
            members=args.members,
        )

        summary_path = Path(args.summary)
        summary_df.to_csv(summary_path, index=False)

        if not args.quiet:
            print(f"✓ Exported summary to: {summary_path}")
            print()

    # Export markdown if requested
    if args.markdown:
        if not args.quiet:
            print(f"Exporting markdown to: {args.markdown}")

        export_member_mapping(
            mapping_df,
            args.markdown,
            format='markdown',
        )

    # Extract coordinates if NetCDF file provided
    if args.nc_file:
        nc_path = Path(args.nc_file).expanduser()

        if not nc_path.exists():
            print(f"Warning: NetCDF file not found: {nc_path}", file=sys.stderr)
        else:
            if not args.quiet:
                print(f"Extracting node coordinates from: {nc_path}")

            unique_nodes = mapping_df['node_id'].unique().tolist()
            coords_df = get_node_coordinates(nc_path, unique_nodes)

            if args.coords:
                coords_path = Path(args.coords)
                coords_df.to_csv(coords_path, index=False)

                if not args.quiet:
                    print(f"✓ Exported coordinates to: {coords_path}")
            else:
                if not args.quiet:
                    print(f"✓ Extracted coordinates for {len(coords_df)} nodes")
                    print("  (Use --coords to save to file)")

            print()

    # Summary
    if not args.quiet:
        print("=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"Files created:")
        print(f"  - {output_path}")
        if args.summary:
            print(f"  - {args.summary}")
        if args.markdown:
            print(f"  - {args.markdown}")
        if args.coords and args.nc_file:
            print(f"  - {args.coords}")
        print()


if __name__ == "__main__":
    main()

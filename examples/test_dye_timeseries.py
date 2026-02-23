#!/usr/bin/env python3
"""Quick test of dye_timeseries functionality with actual data."""

import sys
from pathlib import Path

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


def main():
    print("=" * 70)
    print("DYE TIMESERIES TEST")
    print("=" * 70)

    # Configuration - use absolute path from home directory
    # xfvcom is at ~/Github/xfvcom, TB-FVCOM is at ~/Github/TB-FVCOM
    tb_fvcom_dir = Path("~/Github/TB-FVCOM").expanduser()

    print(f"\nTB-FVCOM dir: {tb_fvcom_dir}")
    print(f"Exists: {tb_fvcom_dir.exists()}")

    # Test with minimal data: 2021, members 0-2, single node and sigma
    paths = Paths(tb_fvcom_dir=tb_fvcom_dir)
    case = DyeCase(basename="tb_w18_r16", years=[2021], members=[0, 1, 2])
    sel = Selection(nodes=100, sigmas=0)  # Single node, surface layer
    neg = NegPolicy(mode="keep")
    align = AlignPolicy(mode="native_intersection")

    print("\nConfiguration:")
    print(f"  Years: {case.years}")
    print(f"  Members: {case.members}")
    print(f"  Nodes: {sel.nodes}")
    print(f"  Sigmas: {sel.sigmas}")

    # Collect files
    print("\nCollecting files...")
    try:
        member_map = collect_member_files(paths, case)
        print(f"✓ Found files for {len(member_map)} (year, member) pairs\n")

        # Show what we found
        print("Member files:")
        for (year, member), files in sorted(member_map.items()):
            print(f"  (year={year}, member={member}):")
            for f in files:
                print(f"    - {f.name}")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return 1

    # Aggregate (this is the main test)
    print("\nAggregating time series...")
    try:
        ds = aggregate(member_map, case, sel, neg, align)
        print("✓ Aggregation successful")
        print(f"  Dimensions: {dict(ds.sizes)}")
        print(f"  Data variables: {list(ds.data_vars)}")
        print(f"  Time steps: {ds.sizes.get('time', 'N/A')}")
        print(f"  Ensemble size: {ds.sizes.get('ensemble', 'N/A')}")

        # Basic statistics
        if "dye" in ds:
            dye = ds["dye"]
            print("\n  DYE statistics:")
            print(f"    Min: {float(dye.min()):.6e}")
            print(f"    Max: {float(dye.max()):.6e}")
            print(f"    Mean: {float(dye.mean()):.6e}")
            print(f"    Std: {float(dye.std()):.6e}")

        # Test negative stats
        print("\nComputing negative stats...")
        from xfvcom.dye_timeseries import load_member_series

        series_dict = {}
        for (year, member), files in member_map.items():
            series = load_member_series(files, case, sel, neg, year, member)
            series_dict[(year, member)] = series

        stats = negative_stats(ds, series_dict)

        if "global" in stats:
            print("✓ Global negative stats:")
            print(f"    Min value: {stats['global']['min_value']:.6e}")
            print(f"    Count neg: {stats['global']['count_neg']}")
            print(f"    Share neg: {stats['global']['share_neg']*100:.2f}%")

        # Test linearity
        if len(case.members) > 1:
            print("\nVerifying linearity...")
            linearity = verify_linearity(ds, ref_member=0, parts=[1, 2])
            print("✓ Linearity metrics:")
            print(f"    RMSE: {linearity['rmse']:.6e}")
            print(f"    MAE: {linearity['mae']:.6e}")
            print(f"    NSE: {linearity['nse']:.6f}")

        print("\n" + "=" * 70)
        print("TEST PASSED ✓")
        print("=" * 70)

        return 0

    except Exception as e:
        print(f"✗ Error during aggregation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

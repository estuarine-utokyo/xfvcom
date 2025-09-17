#!/usr/bin/env python
"""
Demonstration of the enhanced text clipping feature for Cartopy geographic coordinates.

This script shows how the improved text clipping in xfvcom handles the known Cartopy
issue where clip_on=True doesn't work properly with geographic coordinates.

Before the fix: Text labels would appear outside the map extent even with clip_on=True
After the fix: Text labels are properly clipped to the map extent when using Cartopy
"""

from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cartopy.io.img_tiles import GoogleTiles

from xfvcom import (
    FvcomInputLoader,
    FvcomPlotConfig,
    FvcomPlotOptions,
    FvcomPlotter,
    make_node_marker_post,
)


def main():
    # Load the FVCOM grid
    grid_file = Path("~/Github/TB-FVCOM/goto2023/input/TokyoBay18_grd.dat").expanduser()
    utm_zone = 54

    if not grid_file.exists():
        print(f"Grid file not found: {grid_file}")
        return

    print("Loading FVCOM grid...")
    loader = FvcomInputLoader(
        grid_path=grid_file,
        utm_zone=utm_zone,
        add_dummy_time=False,
        add_dummy_siglay=False,
    )

    grid_ds = loader.ds
    grid_obj = loader.grid
    print(f"Grid loaded: {grid_obj.node} nodes, {grid_obj.nele} elements")

    # Create plotter
    cfg = FvcomPlotConfig()
    plotter = FvcomPlotter(grid_ds, cfg)

    # Select nodes to display - every 20th node for demonstration
    display_nodes = np.arange(1, min(grid_obj.node + 1, 500), 20)

    # Define the map extent - a zoomed area where text clipping will be tested
    xlim = (139.85, 139.95)
    ylim = (35.36, 35.45)

    print(f"\nMap extent: lon={xlim}, lat={ylim}")
    print(f"Displaying {len(display_nodes)} nodes with markers and labels")

    # Create two plots for comparison
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(20, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Plot 1: Without respect_bounds (all text shown regardless of extent)
    print("\nCreating Plot 1: Without text clipping (respect_bounds=False)...")
    pp_no_clip = make_node_marker_post(
        display_nodes,
        plotter,
        marker_kwargs={"marker": "o", "color": "red", "markersize": 4, "zorder": 4},
        text_kwargs={
            "fontsize": 8,
            "color": "yellow",
            "ha": "center",
            "va": "bottom",
            "zorder": 5,
            "clip_on": False,
        },
        index_base=1,
        respect_bounds=False,  # Show all markers and text
    )

    ax1.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=ccrs.PlateCarree())
    ax1.add_image(GoogleTiles(style="satellite"), 10)
    ax1.gridlines(draw_labels=True, alpha=0.5)
    pp_no_clip(ax1, opts=FvcomPlotOptions(xlim=xlim, ylim=ylim, use_latlon=True))
    ax1.set_title(
        "WITHOUT Enhanced Clipping\n(Text appears outside map extent)", fontsize=14
    )

    # Plot 2: With enhanced text clipping (only text within extent shown)
    print("Creating Plot 2: With enhanced text clipping (clip_on=True)...")
    pp_with_clip = make_node_marker_post(
        display_nodes,
        plotter,
        marker_kwargs={"marker": "o", "color": "red", "markersize": 4, "zorder": 4},
        text_kwargs={
            "fontsize": 8,
            "color": "yellow",
            "ha": "center",
            "va": "bottom",
            "zorder": 5,
            "clip_on": True,
        },  # Enable clipping
        index_base=1,
        respect_bounds=False,  # Show all markers (but text will be clipped)
    )

    ax2.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=ccrs.PlateCarree())
    ax2.add_image(GoogleTiles(style="satellite"), 10)
    ax2.gridlines(draw_labels=True, alpha=0.5)
    pp_with_clip(ax2, opts=FvcomPlotOptions(xlim=xlim, ylim=ylim, use_latlon=True))
    ax2.set_title(
        "WITH Enhanced Clipping\n(Text properly clipped to map extent)", fontsize=14
    )

    plt.suptitle(
        "Enhanced Text Clipping for Cartopy Geographic Coordinates",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save the comparison
    output_dir = Path("PNG")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "text_clipping_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {output_file}")

    plt.show()

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey improvements:")
    print("1. Text labels now respect map extent when clip_on=True")
    print("2. Works around known Cartopy issue with text clipping")
    print("3. Performance optimized with vectorized bounds checking")
    print("4. Optional text_clip_buffer parameter for fine-tuning")
    print("\nThe enhanced clipping is automatically activated when:")
    print("- Using Cartopy GeoAxes with geographic coordinates")
    print("- clip_on=True is set in text_kwargs")
    print("- Map extent (xlim/ylim) is specified")


if __name__ == "__main__":
    main()

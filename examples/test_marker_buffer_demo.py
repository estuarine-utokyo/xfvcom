#!/usr/bin/env python
"""
Demonstration of the marker_clip_buffer and text_clip_buffer parameters.

This script shows how independent buffer control for markers and text
allows fine-tuning of node visibility near map boundaries.
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

    # Select all nodes for demonstration
    all_nodes = np.arange(1, grid_obj.node + 1)

    # Define the map extent
    xlim = (139.85, 139.95)
    ylim = (35.36, 35.45)

    print(f"\nMap extent: lon={xlim}, lat={ylim}")

    # Create figure with 4 subplots for different buffer combinations
    fig = plt.figure(figsize=(20, 20))

    # Common plot options
    base_opts = FvcomPlotOptions(
        xlim=xlim,
        ylim=ylim,
        use_latlon=True,
        add_tiles=False,  # Disable for faster rendering
        mesh_color="lightgray",
        mesh_linewidth=0.1,
    )

    # Subplot 1: No buffers (default behavior)
    ax1 = fig.add_subplot(221, projection=ccrs.PlateCarree())
    pp1 = make_node_marker_post(
        all_nodes,
        plotter,
        marker_kwargs={"marker": "o", "color": "red", "markersize": 2},
        text_kwargs={"fontsize": 6, "color": "blue", "clip_on": True},
        index_base=1,
        respect_bounds=True,
        marker_clip_buffer=0.0,  # No buffer
        text_clip_buffer=0.0,  # No buffer
    )
    ax1.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=ccrs.PlateCarree())
    ax1.add_image(GoogleTiles(style="satellite"), 10)
    ax1.gridlines(draw_labels=True, alpha=0.5)
    pp1(ax1, opts=base_opts)
    ax1.set_title("No Buffers\n(Strict boundary clipping)", fontsize=14)

    # Subplot 2: Positive marker buffer
    ax2 = fig.add_subplot(222, projection=ccrs.PlateCarree())
    pp2 = make_node_marker_post(
        all_nodes,
        plotter,
        marker_kwargs={"marker": "o", "color": "red", "markersize": 2},
        text_kwargs={"fontsize": 6, "color": "blue", "clip_on": True},
        index_base=1,
        respect_bounds=True,
        marker_clip_buffer=0.002,  # Include markers slightly outside
        text_clip_buffer=0.0,  # No text buffer
    )
    ax2.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=ccrs.PlateCarree())
    ax2.add_image(GoogleTiles(style="satellite"), 10)
    ax2.gridlines(draw_labels=True, alpha=0.5)
    pp2(ax2, opts=base_opts)
    ax2.set_title("Marker Buffer +0.002°\n(Shows edge markers)", fontsize=14)

    # Subplot 3: Negative text buffer
    ax3 = fig.add_subplot(223, projection=ccrs.PlateCarree())
    pp3 = make_node_marker_post(
        all_nodes,
        plotter,
        marker_kwargs={"marker": "o", "color": "red", "markersize": 2},
        text_kwargs={"fontsize": 6, "color": "blue", "clip_on": True},
        index_base=1,
        respect_bounds=True,
        marker_clip_buffer=0.0,  # No marker buffer
        text_clip_buffer=-0.002,  # Exclude text near edges
    )
    ax3.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=ccrs.PlateCarree())
    ax3.add_image(GoogleTiles(style="satellite"), 10)
    ax3.gridlines(draw_labels=True, alpha=0.5)
    pp3(ax3, opts=base_opts)
    ax3.set_title("Text Buffer -0.002°\n(Hides edge text)", fontsize=14)

    # Subplot 4: Combined buffers
    ax4 = fig.add_subplot(224, projection=ccrs.PlateCarree())
    pp4 = make_node_marker_post(
        all_nodes,
        plotter,
        marker_kwargs={"marker": "o", "color": "red", "markersize": 2},
        text_kwargs={"fontsize": 6, "color": "blue", "clip_on": True},
        index_base=1,
        respect_bounds=True,
        marker_clip_buffer=0.001,  # Small positive buffer for markers
        text_clip_buffer=-0.001,  # Small negative buffer for text
    )
    ax4.set_extent([xlim[0], xlim[1], ylim[0], ylim[1]], crs=ccrs.PlateCarree())
    ax4.add_image(GoogleTiles(style="satellite"), 10)
    ax4.gridlines(draw_labels=True, alpha=0.5)
    pp4(ax4, opts=base_opts)
    ax4.set_title("Combined Buffers\n(Marker +0.001°, Text -0.001°)", fontsize=14)

    plt.suptitle(
        "Independent Marker and Text Buffer Control", fontsize=18, fontweight="bold"
    )
    plt.tight_layout()

    # Save the comparison
    output_dir = Path("PNG")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "marker_buffer_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {output_file}")

    plt.show()

    print("\n" + "=" * 60)
    print("MARKER BUFFER DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nKey features:")
    print("1. marker_clip_buffer: Controls marker visibility at boundaries")
    print("   - Positive: Include markers outside bounds")
    print("   - Negative: Exclude markers near boundaries")
    print("2. text_clip_buffer: Controls text visibility (with Cartopy)")
    print("   - Positive: Show more text")
    print("   - Negative: Hide text near edges")
    print("3. Independent control allows optimal visualization")
    print("\nUse cases:")
    print("- Dense grids: Use negative text buffer to reduce edge clutter")
    print("- Edge analysis: Use positive marker buffer to see boundary nodes")
    print("- Clean plots: Use negative buffers for crisp boundaries")


if __name__ == "__main__":
    main()

"""
Node Checker for FVCOM Grid Files

This module provides functionality to visualize FVCOM grid nodes with various display options.
Supports displaying mesh, node markers, node numbers (one-based), and highlighting specific nodes.

Author: Jun Sasaki
Created: 2025-01-14
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sequence

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from cartopy.io.img_tiles import OSM, GoogleTiles
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from xfvcom import FvcomInputLoader, FvcomPlotConfig, FvcomPlotOptions, FvcomPlotter
from xfvcom.plot.markers import make_node_marker_post


def plot_node_checker(
    grid_file: Path | str,
    utm_zone: int | None = None,
    *,
    background: Literal["tiles", "white", None] = "tiles",
    tile_provider: Any = None,
    show_mesh: bool = True,
    show_all_nodes: bool = False,
    show_node_numbers: bool = False,
    specific_nodes: Sequence[int] | None = None,
    specific_node_color: str = "red",
    all_node_color: str = "blue",
    mesh_color: str = "#808080",
    mesh_linewidth: float = 0.3,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (12, 10),
    title: str | None = None,
    save_path: Path | str | None = None,
    dpi: int = 150,
) -> tuple[Figure, Axes]:
    """
    Plot FVCOM grid nodes with various display options.

    Parameters
    ----------
    grid_file : Path or str
        Path to FVCOM grid file (.grd or .dat)
    utm_zone : int, optional
        UTM zone for coordinate conversion (required for geographic grids)
    background : {"tiles", "white", None}, default "tiles"
        Background type: "tiles" for map tiles, "white" for white background, None for transparent
    tile_provider : cartopy tile provider, optional
        Map tile provider (e.g., GoogleTiles(style="satellite")). If None, uses OSM.
    show_mesh : bool, default True
        Whether to display the mesh triangulation
    show_all_nodes : bool, default False
        Whether to show markers at all nodes
    show_node_numbers : bool, default False
        Whether to show node number text labels (one-based)
    specific_nodes : sequence of int, optional
        List of specific node numbers to highlight (one-based)
    specific_node_color : str, default "red"
        Color for specific node markers
    all_node_color : str, default "blue"
        Color for all node markers
    mesh_color : str, default "#808080"
        Color for mesh lines
    mesh_linewidth : float, default 0.3
        Line width for mesh
    xlim : tuple of float, optional
        Longitude/x limits for the plot
    ylim : tuple of float, optional
        Latitude/y limits for the plot
    figsize : tuple of float, default (12, 10)
        Figure size in inches
    title : str, optional
        Plot title. If None, generates automatic title.
    save_path : Path or str, optional
        Path to save the figure
    dpi : int, default 150
        DPI for saved figure

    Returns
    -------
    fig : Figure
        The matplotlib figure
    ax : Axes
        The matplotlib axes

    Examples
    --------
    >>> # Basic mesh display
    >>> fig, ax = plot_node_checker("grid.dat", utm_zone=54)

    >>> # Show all nodes with markers
    >>> fig, ax = plot_node_checker("grid.dat", utm_zone=54, show_all_nodes=True)

    >>> # Highlight specific nodes with numbers
    >>> fig, ax = plot_node_checker(
    ...     "grid.dat", utm_zone=54,
    ...     specific_nodes=[100, 200, 300],
    ...     show_node_numbers=True
    ... )
    """
    # Convert to Path
    grid_file = Path(grid_file).expanduser()
    if not grid_file.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_file}")

    # Load grid using FvcomInputLoader
    loader = FvcomInputLoader(
        grid_path=grid_file,
        utm_zone=utm_zone,
        add_dummy_time=False,
        add_dummy_siglay=False,
    )
    grid_ds = loader.ds
    n_nodes = len(grid_ds.lon)

    # Create plotter
    cfg = FvcomPlotConfig()
    plotter = FvcomPlotter(grid_ds, cfg)

    # Determine marker sizes based on node count and display options
    all_marker_size = _auto_marker_size(n_nodes, show_all_nodes, show_node_numbers)
    specific_marker_size = max(all_marker_size * 1.5, 5)  # Specific nodes larger

    # Determine font size for labels
    font_size = (
        _auto_font_size(n_nodes, xlim, ylim, grid_ds) if show_node_numbers else 8
    )

    # Build post-processing functions
    post_process_funcs = []

    # Add all nodes if requested
    if show_all_nodes:
        all_nodes = list(range(1, n_nodes + 1))  # One-based node numbers

        mkw_all = {
            "marker": "o",
            "color": all_node_color,
            "markersize": all_marker_size,
            "zorder": 3,
            "alpha": 0.6,
        }

        # Only add text for all nodes if requested
        tkw_all = None
        if show_node_numbers:  # Remove limit - user can zoom in for overlapping text
            tkw_all = {
                "fontsize": font_size,
                "color": "black",  # Simple black text
                "ha": "center",
                "va": "bottom",  # Position text above the node marker
                "zorder": 4,
                # No bbox or other decorations - just simple text
            }

        pp_all = make_node_marker_post(
            all_nodes,
            plotter,
            marker_kwargs=mkw_all,
            text_kwargs=tkw_all,
            index_base=1,
            respect_bounds=True,
        )
        post_process_funcs.append(pp_all)

    # Add specific nodes if provided
    if specific_nodes:
        mkw_specific = {
            "marker": "o",
            "color": specific_node_color,
            "markersize": specific_marker_size,
            "zorder": 5,
        }

        tkw_specific = None
        if show_node_numbers:
            tkw_specific = {
                "fontsize": font_size * 1.2,  # Slightly larger for specific nodes
                "color": "red",  # Red text for specific nodes to stand out
                "ha": "center",
                "va": "bottom",  # Position text above the node marker
                "zorder": 6,
                # No bbox for simplicity unless specifically needed
            }

        pp_specific = make_node_marker_post(
            specific_nodes,
            plotter,
            marker_kwargs=mkw_specific,
            text_kwargs=tkw_specific,
            index_base=1,
            respect_bounds=True,
        )
        post_process_funcs.append(pp_specific)

    # Combine post-processing functions
    def combined_post_process(ax, **kwargs):
        for pp in post_process_funcs:
            pp(ax, **kwargs)

    # Set up plot options
    if background == "tiles":
        add_tiles = True
        if tile_provider is None:
            tile_provider = OSM()
    else:
        add_tiles = False
        tile_provider = None

    # Generate title if not provided
    if title is None:
        title_parts = ["FVCOM Grid Node Checker"]
        if show_all_nodes:
            title_parts.append(f"All {n_nodes} nodes")
        if specific_nodes:
            title_parts.append(f"{len(specific_nodes)} highlighted")
        title = " - ".join(title_parts)

    # Create plot options
    opts = FvcomPlotOptions(
        figsize=figsize,
        add_tiles=add_tiles,
        tile_provider=tile_provider,
        with_mesh=show_mesh,
        mesh_color=mesh_color,
        mesh_linewidth=mesh_linewidth,
        title=title,
        xlim=xlim,
        ylim=ylim,
    )

    # Create the plot
    if background == "white":
        # For white background, create figure manually
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        ax.set_facecolor("white")
        ax = plotter.plot_2d(
            da=None,
            post_process_func=combined_post_process if post_process_funcs else None,
            opts=opts,
            ax=ax,
        )
    else:
        # Use default plotting
        ax = plotter.plot_2d(
            da=None,
            post_process_func=combined_post_process if post_process_funcs else None,
            opts=opts,
        )

    fig = ax.figure

    # Add info text
    info_text = _generate_info_text(n_nodes, show_all_nodes, specific_nodes)
    if info_text:
        fig.text(
            0.02,
            0.02,
            info_text,
            fontsize=9,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig, ax


def _auto_marker_size(n_nodes: int, show_all: bool, show_numbers: bool) -> float:
    """Calculate appropriate marker size based on node count."""
    if not show_all:
        return 5  # Default for specific nodes only

    # Scale marker size inversely with node count
    if n_nodes < 100:
        base_size = 8
    elif n_nodes < 500:
        base_size = 5
    elif n_nodes < 1000:
        base_size = 3
    elif n_nodes < 5000:
        base_size = 2
    else:
        base_size = 1

    # Reduce size if showing numbers to avoid overlap
    if show_numbers:
        base_size = int(base_size * 0.7)

    return base_size


def _auto_font_size(
    n_nodes: int,
    xlim: tuple[float, float] | None,
    ylim: tuple[float, float] | None,
    grid_ds: Any,
) -> float:
    """Calculate appropriate font size for node labels."""
    # Base font size on node count
    if n_nodes < 50:
        base_font = 10
    elif n_nodes < 100:
        base_font = 8
    elif n_nodes < 500:
        base_font = 6
    else:
        base_font = 4

    # Adjust based on zoom level if bounds are provided
    if xlim and ylim:
        full_x_range = float(grid_ds.lon.max() - grid_ds.lon.min())
        full_y_range = float(grid_ds.lat.max() - grid_ds.lat.min())
        view_x_range = xlim[1] - xlim[0]
        view_y_range = ylim[1] - ylim[0]

        # Calculate zoom factor (>1 means zoomed in)
        zoom_factor = min(full_x_range / view_x_range, full_y_range / view_y_range)

        # Scale font size with zoom
        base_font = int(base_font * min(2.0, max(0.5, zoom_factor)))

    return base_font


def _generate_info_text(
    n_nodes: int,
    show_all: bool,
    specific_nodes: Sequence[int] | None,
) -> str:
    """Generate information text for the plot."""
    info_parts = [f"Total nodes: {n_nodes}"]

    if show_all:
        info_parts.append("Showing: all nodes")

    if specific_nodes:
        info_parts.append(f"Highlighted: {len(specific_nodes)} nodes")
        if len(specific_nodes) <= 10:
            info_parts.append(f"Node IDs: {', '.join(map(str, specific_nodes))}")

    return " | ".join(info_parts) if len(info_parts) > 1 else ""


def check_nodes_in_bounds(
    grid_file: Path | str,
    node_list: Sequence[int],
    utm_zone: int | None = None,
) -> dict[str, Any]:
    """
    Check which nodes from a list are within the grid bounds.

    Parameters
    ----------
    grid_file : Path or str
        Path to FVCOM grid file
    node_list : sequence of int
        List of node numbers to check (one-based)
    utm_zone : int, optional
        UTM zone for coordinate conversion

    Returns
    -------
    dict
        Dictionary with node information including coordinates and validity
    """
    grid_file = Path(grid_file).expanduser()

    # Load grid
    loader = FvcomInputLoader(
        grid_path=grid_file,
        utm_zone=utm_zone,
        add_dummy_time=False,
        add_dummy_siglay=False,
    )
    grid_ds = loader.ds
    n_nodes = len(grid_ds.lon)

    results: dict[str, Any] = {
        "total_nodes": n_nodes,
        "valid_nodes": [],
        "invalid_nodes": [],
        "node_info": {},
    }

    for node_id in node_list:
        if 1 <= node_id <= n_nodes:
            idx = node_id - 1  # Convert to 0-based
            results["valid_nodes"].append(node_id)
            results["node_info"][node_id] = {
                "lon": float(grid_ds.lon.values[idx]),
                "lat": float(grid_ds.lat.values[idx]),
                "x": float(grid_ds.x.values[idx]) if "x" in grid_ds.data_vars else None,
                "y": float(grid_ds.y.values[idx]) if "y" in grid_ds.data_vars else None,
            }
        else:
            results["invalid_nodes"].append(node_id)

    return results


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python node_checker.py <grid_file> [utm_zone]")
        sys.exit(1)

    grid_path = Path(sys.argv[1])
    utm = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # Create output directory
    output_dir = Path("node_checker_output")
    output_dir.mkdir(exist_ok=True)

    # Example 1: Basic mesh
    fig, ax = plot_node_checker(
        grid_path,
        utm_zone=utm,
        save_path=output_dir / "mesh_only.png",
    )
    plt.close(fig)

    # Example 2: All nodes
    fig, ax = plot_node_checker(
        grid_path,
        utm_zone=utm,
        show_all_nodes=True,
        save_path=output_dir / "all_nodes.png",
    )
    plt.close(fig)

    # Example 3: Specific nodes with numbers
    fig, ax = plot_node_checker(
        grid_path,
        utm_zone=utm,
        specific_nodes=[10, 50, 100, 200, 500],
        show_node_numbers=True,
        save_path=output_dir / "specific_nodes.png",
    )
    plt.close(fig)

    print(f"Examples saved to {output_dir}/")

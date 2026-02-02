"""Map plot showing source nodes and target nodes.

This module provides tools for creating map plots showing the locations
of river/sewer source nodes and target analysis nodes in Tokyo Bay.

Source configurations can be loaded from external YAML files via
xfvcom.config.load_source_config(). When external config is loaded,
this module will use those values instead of the hardcoded defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from ..ensemble_analysis import MEMBER_SOURCE_TYPES

if TYPE_CHECKING:
    pass


# =============================================================================
# Default hardcoded values (for backward compatibility)
# =============================================================================

# Source nodes mapping (1-based node IDs)
# Rivers (22 nodes in 12 groups) + Sewers (7 nodes in 5 groups)
_DEFAULT_SOURCE_NODES: dict[str, list[int]] = {
    # Rivers
    "Arakawa": [310, 241, 312, 448],
    "Sumida": [4, 2, 12],
    "Edo": [506, 560, 616],
    "Tama": [1117, 1058, 1239],
    "Tsurumi": [1267, 1325],
    "Mama": [145],
    "Ebi": [77],
    "Yoro": [412],
    "Obitsu": [1505],
    "Koito": [1734],
    "Murata": [182],
    "Hanami": [221],
    # Sewers
    "Shibaura": [18],
    "Sunamachi": [240],
    "Ariake": [17],
    "Kasai": [242],
    "Morigasaki": [687, 813, 822],
}

# Source types
_DEFAULT_SOURCE_TYPES: dict[str, str] = {
    "Arakawa": "River",
    "Sumida": "River",
    "Edo": "River",
    "Tama": "River",
    "Tsurumi": "River",
    "Mama": "River",
    "Ebi": "River",
    "Yoro": "River",
    "Obitsu": "River",
    "Koito": "River",
    "Murata": "River",
    "Hanami": "River",
    "Shibaura": "Sewer",
    "Sunamachi": "Sewer",
    "Ariake": "Sewer",
    "Kasai": "Sewer",
    "Morigasaki": "Sewer",
}


# =============================================================================
# Dynamic accessors
# =============================================================================


def _get_source_nodes_from_config() -> dict[str, list[int]] | None:
    """Get source nodes from external config if loaded."""
    try:
        from xfvcom.config import source_config

        if source_config.is_config_loaded():
            return source_config.get_source_nodes()
    except ImportError:
        pass
    return None


def _get_source_types_from_config() -> dict[str, str] | None:
    """Get source types from external config if loaded."""
    try:
        from xfvcom.config import source_config

        if source_config.is_config_loaded():
            config = source_config.get_source_config()
            types = {}
            for name in config.get("rivers", {}):
                types[name] = "River"
            for name in config.get("sewers", {}):
                types[name] = "Sewer"
            return types if types else None
    except ImportError:
        pass
    return None


def get_source_nodes() -> dict[str, list[int]]:
    """Get source nodes mapping (from config if loaded, else defaults).

    Returns
    -------
    dict[str, list[int]]
        Mapping from source name to list of node IDs.
    """
    nodes = _get_source_nodes_from_config()
    return nodes if nodes else _DEFAULT_SOURCE_NODES


def get_source_types() -> dict[str, str]:
    """Get source types mapping (from config if loaded, else defaults).

    Returns
    -------
    dict[str, str]
        Mapping from source name to type ('River' or 'Sewer').
    """
    types = _get_source_types_from_config()
    return types if types else _DEFAULT_SOURCE_TYPES


# For backward compatibility, expose as module-level variables
# Note: These are evaluated at import time and won't update dynamically.
# Use get_source_nodes() and get_source_types() for dynamic access.
SOURCE_NODES = _DEFAULT_SOURCE_NODES
SOURCE_TYPES = _DEFAULT_SOURCE_TYPES


def load_grid_coordinates(nc_file: str | Path) -> dict:
    """Load grid coordinates from NetCDF file.

    Parameters
    ----------
    nc_file : str or Path
        Path to FVCOM NetCDF output file

    Returns
    -------
    dict
        Dictionary with 'lon', 'lat', 'x', 'y', 'nv' (triangulation)
    """
    import netCDF4 as nc

    ds = nc.Dataset(nc_file)
    coords = {
        "lon": ds.variables["lon"][:],
        "lat": ds.variables["lat"][:],
        "x": ds.variables["x"][:],
        "y": ds.variables["y"][:],
        "lonc": ds.variables["lonc"][:],
        "latc": ds.variables["latc"][:],
        "nv": ds.variables["nv"][:] - 1,  # Convert to 0-based
        "h": ds.variables["h"][:],
    }
    ds.close()
    return coords


def extract_boundary_edges(nv: np.ndarray) -> list[tuple[int, int]]:
    """Extract boundary edges from triangulation.

    Boundary edges are edges that belong to only one triangle.

    Parameters
    ----------
    nv : np.ndarray
        Triangulation array (3, nele) with 0-based node indices

    Returns
    -------
    list of tuple
        List of boundary edge pairs (node1, node2)
    """
    from collections import Counter

    # Count edge occurrences
    edge_counts: Counter = Counter()

    # nv is (3, nele) - each column is a triangle
    nele = nv.shape[1]

    for i in range(nele):
        tri = nv[:, i]
        # Add edges (sorted to ensure consistent ordering)
        for j in range(3):
            edge = tuple(sorted([tri[j], tri[(j + 1) % 3]]))
            edge_counts[edge] += 1

    # Boundary edges appear only once
    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]

    return boundary_edges


def order_boundary_nodes(
    boundary_edges: list[tuple[int, int]],
) -> list[list[int]]:
    """Order boundary edges into continuous paths.

    Parameters
    ----------
    boundary_edges : list of tuple
        List of boundary edge pairs

    Returns
    -------
    list of list
        List of ordered node sequences (one per closed boundary)
    """
    if not boundary_edges:
        return []

    # Build adjacency dict
    adj: dict[int, list[int]] = {}
    for n1, n2 in boundary_edges:
        if n1 not in adj:
            adj[n1] = []
        if n2 not in adj:
            adj[n2] = []
        adj[n1].append(n2)
        adj[n2].append(n1)

    # Find all closed paths
    paths: list[list[int]] = []
    visited_edges: set[tuple[int, int]] = set()

    for start_edge in boundary_edges:
        if start_edge in visited_edges:
            continue

        # Start a new path
        path = [start_edge[0], start_edge[1]]
        visited_edges.add(start_edge)
        visited_edges.add((start_edge[1], start_edge[0]))

        current = start_edge[1]
        prev = start_edge[0]

        while True:
            # Find next node
            neighbors = adj[current]
            next_node = None
            for n in neighbors:
                edge = (min(current, n), max(current, n))
                if edge not in visited_edges or n == path[0]:
                    if n != prev:
                        next_node = n
                        break

            if next_node is None or next_node == path[0]:
                # Path complete
                break

            path.append(next_node)
            edge = (min(current, next_node), max(current, next_node))
            visited_edges.add(edge)
            visited_edges.add((edge[1], edge[0]))
            prev = current
            current = next_node

        if len(path) > 2:
            paths.append(path)

    return paths


def get_source_coordinates(
    coords: dict,
    source_nodes: dict[str, list[int]] | None = None,
) -> dict[str, dict]:
    """Get coordinates for each source.

    Parameters
    ----------
    coords : dict
        Grid coordinates from load_grid_coordinates()
    source_nodes : dict, optional
        Source-to-node mapping. If None, uses get_source_nodes().

    Returns
    -------
    dict
        Dictionary mapping source name to {'lon', 'lat', 'nodes', 'type'}
    """
    if source_nodes is None:
        source_nodes = get_source_nodes()

    lon = coords["lon"]
    lat = coords["lat"]

    # Get source types (dynamic)
    source_types = get_source_types()

    source_coords = {}
    for name, nodes in source_nodes.items():
        # Convert to 0-based
        node_idx = [n - 1 for n in nodes]
        source_coords[name] = {
            "lon": np.mean([lon[i] for i in node_idx]),
            "lat": np.mean([lat[i] for i in node_idx]),
            "nodes": nodes,
            "type": source_types.get(name, "Unknown"),
        }

    return source_coords


def plot_source_map(
    nc_file: str | Path,
    target_nodes: list[int] | None = None,
    output: str | Path | None = None,
    figsize: tuple[float, float] = (12, 10),
    fontsize_title: float = 16,
    fontsize_label: float = 12,
    fontsize_source: float = 9,
    fontsize_target: float = 10,
    title: str | None = None,
    show_mesh: bool = False,
    mesh_linewidth: float = 0.1,
    mesh_color: str = "gray",
    mesh_alpha: float = 0.3,
    show_depth: bool = True,
    show_coastline: bool = True,
    coastline_color: str = "black",
    coastline_linewidth: float = 1.0,
    land_color: str = "lightgray",
    ocean_color: str = "white",
    river_color: str = "blue",
    sewer_color: str = "red",
    target_color: str = "green",
    river_marker: str = "^",
    sewer_marker: str = "s",
    target_marker: str = "*",
    marker_size_source: float = 80,
    marker_size_target: float = 200,
    dpi: int = 300,
    label_offset: tuple[float, float] = (0.005, 0.005),
    label_configs: dict[str, tuple[float, float, bool]] | None = None,
    target_label_offset: tuple[float, float] = (0.025, -0.025),
) -> dict:
    """Create map plot showing source nodes and target nodes.

    Parameters
    ----------
    nc_file : str or Path
        Path to FVCOM NetCDF output file (for grid coordinates)
    target_nodes : list of int, optional
        Target node indices (1-based, Fortran convention) to highlight
    output : str or Path, optional
        Output file path. If provided, saves figure.
    figsize : tuple
        Figure size (width, height) in inches
    fontsize_title : float
        Title font size
    fontsize_label : float
        Axis label font size
    fontsize_source : float
        Source label font size
    fontsize_target : float
        Target label font size
    title : str, optional
        Plot title. If None, auto-generated.
    show_mesh : bool
        Whether to show mesh overlay (default: False)
    mesh_linewidth : float
        Line width for mesh overlay (default: 0.1)
    mesh_color : str
        Color for mesh lines (default: 'gray')
    mesh_alpha : float
        Transparency for mesh lines (default: 0.3)
    show_depth : bool
        Whether to show bathymetry contours
    show_coastline : bool
        Whether to show coastline (default: True)
    coastline_color : str
        Color for coastline (default: 'black')
    coastline_linewidth : float
        Line width for coastline (default: 1.0)
    land_color : str
        Color for land fill (default: 'lightgray')
    ocean_color : str
        Color for ocean background (default: 'white')
    river_color : str
        Color for river markers
    sewer_color : str
        Color for sewer markers
    target_color : str
        Color for target node markers
    river_marker : str
        Marker style for rivers
    sewer_marker : str
        Marker style for sewers
    target_marker : str
        Marker style for target nodes
    marker_size_source : float
        Marker size for source nodes
    marker_size_target : float
        Marker size for target nodes
    dpi : int
        Output resolution
    label_offset : tuple
        Default (lon_offset, lat_offset) for text labels when no custom config
    label_configs : dict, optional
        Custom label configurations for each source. Keys are source names,
        values are tuples of (lon_offset, lat_offset, use_leader_line).
        If None, uses default label_offset for all sources.
        Example: {"Arakawa": (0.01, 0.02, True), "Sumida": (-0.02, 0.01, False)}
    target_label_offset : tuple
        (lon_offset, lat_offset) for target node labels (default: (0.025, -0.025))

    Returns
    -------
    dict
        Dictionary with 'fig', 'ax', 'source_coords', 'target_coords'
    """
    print("=" * 70, file=sys.stdout)
    print("SOURCE AND TARGET NODE MAP", file=sys.stdout)
    print("=" * 70, file=sys.stdout)

    # Load grid coordinates
    coords = load_grid_coordinates(nc_file)
    print(f"Grid loaded: {len(coords['lon'])} nodes", file=sys.stdout)

    # Get source coordinates
    source_coords = get_source_coordinates(coords)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set ocean background color
    ax.set_facecolor(ocean_color)

    # Extract and plot coastline with land fill
    if show_coastline:
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon as MplPolygon

        # Extract boundary edges
        boundary_edges = extract_boundary_edges(coords["nv"])
        print(f"Boundary edges: {len(boundary_edges)}", file=sys.stdout)

        # Order boundary nodes into paths
        boundary_paths = order_boundary_nodes(boundary_edges)
        print(f"Boundary paths: {len(boundary_paths)}", file=sys.stdout)

        # Create land polygon (exterior boundary)
        # The exterior boundary is typically the longest path
        if boundary_paths:
            # Sort by path length (longest first)
            boundary_paths.sort(key=len, reverse=True)

            # Fill land area outside the mesh
            # First, create a large rectangle covering the plot area
            lon_min, lon_max = coords["lon"].min(), coords["lon"].max()
            lat_min, lat_max = coords["lat"].min(), coords["lat"].max()
            margin = 0.02  # Add margin

            # Create outer rectangle vertices
            outer_rect = [
                (lon_min - margin, lat_min - margin),
                (lon_max + margin, lat_min - margin),
                (lon_max + margin, lat_max + margin),
                (lon_min - margin, lat_max + margin),
            ]

            # Fill the entire plot with land color first
            ax.fill(
                [p[0] for p in outer_rect],
                [p[1] for p in outer_rect],
                color=land_color,
                zorder=0,
            )

            # Fill ocean (inside the mesh boundary) with ocean color
            for path in boundary_paths:
                path_coords = [(coords["lon"][n], coords["lat"][n]) for n in path]
                # Close the path
                path_coords.append(path_coords[0])
                ax.fill(
                    [p[0] for p in path_coords],
                    [p[1] for p in path_coords],
                    color=ocean_color,
                    zorder=1,
                )

            # Draw coastline on top
            for path in boundary_paths:
                path_lons = [coords["lon"][n] for n in path]
                path_lats = [coords["lat"][n] for n in path]
                # Close the path
                path_lons.append(path_lons[0])
                path_lats.append(path_lats[0])
                ax.plot(
                    path_lons,
                    path_lats,
                    color=coastline_color,
                    linewidth=coastline_linewidth,
                    zorder=5,
                )

    # Plot mesh overlay if requested
    if show_mesh:
        from matplotlib.tri import Triangulation

        tri = Triangulation(coords["lon"], coords["lat"], coords["nv"].T)
        ax.triplot(
            tri,
            color=mesh_color,
            linewidth=mesh_linewidth,
            alpha=mesh_alpha,
            zorder=2,
        )
        print(f"Mesh overlay: linewidth={mesh_linewidth}", file=sys.stdout)

    # Plot bathymetry contours if requested
    if show_depth:
        from matplotlib.tri import Triangulation

        tri = Triangulation(coords["lon"], coords["lat"], coords["nv"].T)
        levels = [5, 10, 20, 30, 40, 50]
        cs = ax.tricontour(
            tri,
            coords["h"],
            levels=levels,
            colors="gray",
            alpha=0.5,
            linewidths=0.5,
            zorder=3,
        )
        ax.clabel(cs, inline=True, fontsize=8, fmt="%d m")

    # Plot river sources
    river_lons = []
    river_lats = []
    river_names = []
    for name, info in source_coords.items():
        if info["type"] == "River":
            river_lons.append(info["lon"])
            river_lats.append(info["lat"])
            river_names.append(name)

    ax.scatter(
        river_lons,
        river_lats,
        c=river_color,
        marker=river_marker,
        s=marker_size_source,
        label="River",
        zorder=10,
        edgecolors="white",
        linewidths=0.5,
    )

    # Add river labels with custom offsets and optional leader lines
    # Use label_configs if provided, otherwise fall back to default label_offset
    _label_configs = label_configs if label_configs is not None else {}
    for lon, lat, name in zip(river_lons, river_lats, river_names):
        config = _label_configs.get(name, (label_offset[0], label_offset[1], False))
        offset_lon, offset_lat, use_leader = config

        if use_leader:
            # Use annotate with arrow for leader line
            ax.annotate(
                name,
                xy=(lon, lat),  # marker position
                xytext=(lon + offset_lon, lat + offset_lat),  # label position
                fontsize=fontsize_source,
                color=river_color,
                fontweight="bold",
                zorder=11,
                arrowprops=dict(
                    arrowstyle="-",
                    color=river_color,
                    alpha=0.6,
                    linewidth=0.8,
                    shrinkA=5,  # gap from label
                    shrinkB=3,  # gap from marker
                ),
            )
        else:
            # Simple text without leader line
            ax.annotate(
                name,
                (lon + offset_lon, lat + offset_lat),
                fontsize=fontsize_source,
                color=river_color,
                fontweight="bold",
                zorder=11,
            )

    # Plot sewer sources
    sewer_lons = []
    sewer_lats = []
    sewer_names = []
    for name, info in source_coords.items():
        if info["type"] == "Sewer":
            sewer_lons.append(info["lon"])
            sewer_lats.append(info["lat"])
            sewer_names.append(name)

    ax.scatter(
        sewer_lons,
        sewer_lats,
        c=sewer_color,
        marker=sewer_marker,
        s=marker_size_source,
        label="Sewer",
        zorder=10,
        edgecolors="white",
        linewidths=0.5,
    )

    # Add sewer labels with custom offsets and optional leader lines
    for lon, lat, name in zip(sewer_lons, sewer_lats, sewer_names):
        config = _label_configs.get(name, (label_offset[0], label_offset[1], False))
        offset_lon, offset_lat, use_leader = config

        if use_leader:
            # Use annotate with arrow for leader line
            ax.annotate(
                name,
                xy=(lon, lat),  # marker position
                xytext=(lon + offset_lon, lat + offset_lat),  # label position
                fontsize=fontsize_source,
                color=sewer_color,
                fontweight="bold",
                zorder=11,
                arrowprops=dict(
                    arrowstyle="-",
                    color=sewer_color,
                    alpha=0.6,
                    linewidth=0.8,
                    shrinkA=5,  # gap from label
                    shrinkB=3,  # gap from marker
                ),
            )
        else:
            # Simple text without leader line
            ax.annotate(
                name,
                (lon + offset_lon, lat + offset_lat),
                fontsize=fontsize_source,
                color=sewer_color,
                fontweight="bold",
                zorder=11,
            )

    # Plot target nodes if provided
    target_coords_list = []
    if target_nodes is not None and len(target_nodes) > 0:
        # Convert 1-based (Fortran) to 0-based (Python) indices
        target_idx = [n - 1 for n in target_nodes]
        target_lons = [coords["lon"][i] for i in target_idx]
        target_lats = [coords["lat"][i] for i in target_idx]
        target_coords_list = list(zip(target_lons, target_lats))

        ax.scatter(
            target_lons,
            target_lats,
            c=target_color,
            marker=target_marker,
            s=marker_size_target,
            label="Target",
            zorder=12,
            edgecolors="black",
            linewidths=1,
        )

        # Add target labels with leader lines to avoid overlap
        for i, (lon, lat, node) in enumerate(
            zip(target_lons, target_lats, target_nodes)
        ):
            # Use leader line for target labels
            ax.annotate(
                f"Node {node}",
                xy=(lon, lat),  # marker position
                xytext=(lon + target_label_offset[0], lat + target_label_offset[1]),
                fontsize=fontsize_target,
                color=target_color,
                fontweight="bold",
                zorder=13,
                arrowprops=dict(
                    arrowstyle="-",
                    color=target_color,
                    alpha=0.6,
                    linewidth=0.8,
                    shrinkA=5,  # gap from label
                    shrinkB=8,  # gap from marker (larger for star marker)
                ),
            )

        print(f"Target nodes: {target_nodes}", file=sys.stdout)

    # Labels and title
    ax.set_xlabel("Longitude", fontsize=fontsize_label)
    ax.set_ylabel("Latitude", fontsize=fontsize_label)

    if title is None:
        title = "Tokyo Bay - Source and Target Nodes"
    ax.set_title(title, fontsize=fontsize_title)

    # Legend
    ax.legend(loc="lower right", fontsize=fontsize_source)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--", zorder=4)

    # Set aspect ratio to equal for geographic plot
    ax.set_aspect("equal")

    # Set axis limits with small margin
    lon_min, lon_max = coords["lon"].min(), coords["lon"].max()
    lat_min, lat_max = coords["lat"].min(), coords["lat"].max()
    margin = 0.01
    ax.set_xlim(lon_min - margin, lon_max + margin)
    ax.set_ylim(lat_min - margin, lat_max + margin)

    # Tight layout
    plt.tight_layout()

    # Save if output path provided
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output}", file=sys.stdout)

    print(f"Rivers: {len(river_names)}", file=sys.stdout)
    print(f"Sewers: {len(sewer_names)}", file=sys.stdout)
    print("=" * 70, file=sys.stdout)

    return {
        "fig": fig,
        "ax": ax,
        "source_coords": source_coords,
        "target_coords": target_coords_list,
        "grid_coords": coords,
    }

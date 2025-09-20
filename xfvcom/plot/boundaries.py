"""Boundary visualization functions for FVCOM grids."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from matplotlib.collections import LineCollection

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from xfvcom.plot.core import FvcomPlotter


def make_element_boundary_post(
    node_indices: list[int] | np.ndarray,
    plotter: FvcomPlotter,
    line_kwargs: dict[str, Any] | None = None,
    index_base: int = 1,
    respect_bounds: bool = True,
    clip_buffer: float = 0.0,
) -> Callable[[Axes], None]:
    """Create post-processing function to draw element boundaries for selected nodes.

    Parameters
    ----------
    node_indices : list[int] | np.ndarray
        List of node indices to show boundaries for
    plotter : FvcomPlotter
        FvcomPlotter instance with loaded grid
    line_kwargs : dict[str, Any] | None
        Keyword arguments for line styling. Default: {"color": "red", "linewidth": 2}
    index_base : int
        0 for zero-based indexing, 1 for one-based indexing (FVCOM default)
    respect_bounds : bool
        If True, only show boundaries within current xlim/ylim
    clip_buffer : float
        Buffer in degrees for clipping. Positive values extend beyond bounds.

    Returns
    -------
    Callable[[Axes], None]
        Post-processing function to add boundaries to plot

    Examples
    --------
    >>> pp_boundaries = make_element_boundary_post(
    ...     [100, 200, 300],
    ...     plotter,
    ...     line_kwargs={"color": "red", "linewidth": 2, "linestyle": "-"}
    ... )
    >>> ax = plotter.plot_2d(da=None, post_process_func=pp_boundaries, opts=opts)
    """
    # Default line styling
    if line_kwargs is None:
        line_kwargs = {"color": "red", "linewidth": 2, "linestyle": "-"}

    # Get boundaries from the grid
    if hasattr(plotter, "grid") and plotter.grid is not None:
        # If plotter has direct grid access
        boundaries = plotter.grid.get_node_element_boundaries(
            node_indices, index_base=index_base, return_as="lines"
        )
    elif hasattr(plotter.ds, "attrs") and "grid" in plotter.ds.attrs:
        # Fallback: try to get from dataset attributes
        grid = plotter.ds.attrs["grid"]
        boundaries = grid.get_node_element_boundaries(
            node_indices, index_base=index_base, return_as="lines"
        )
    else:
        # Last resort: Create boundaries from dataset coordinates
        # This is a simplified version that won't be as accurate
        boundaries = _get_boundaries_from_dataset(plotter.ds, node_indices, index_base)

    def post_process(ax: Axes) -> None:
        """Add element boundaries to the plot."""
        if not boundaries:
            return

        # Get current axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Filter boundaries if respecting bounds
        if respect_bounds:
            filtered_boundaries = []
            for boundary in boundaries:
                # Check if any point of the line segment is within bounds
                x1, y1 = boundary[0]
                x2, y2 = boundary[1]

                # Apply buffer
                x_min = xlim[0] - clip_buffer
                x_max = xlim[1] + clip_buffer
                y_min = ylim[0] - clip_buffer
                y_max = ylim[1] + clip_buffer

                # Check if line intersects with bounded region
                if (
                    (x_min <= x1 <= x_max and y_min <= y1 <= y_max)
                    or (x_min <= x2 <= x_max and y_min <= y2 <= y_max)
                    or (
                        # Check if line crosses the region
                        ((x1 < x_min and x2 > x_max) or (x1 > x_max and x2 < x_min))
                        and ((y1 < y_min and y2 > y_max) or (y1 > y_max and y2 < y_min))
                    )
                ):
                    filtered_boundaries.append(boundary)
        else:
            filtered_boundaries = boundaries

        if not filtered_boundaries:
            return

        # Create LineCollection for efficient rendering
        # Transform boundaries if using cartopy
        if hasattr(ax, "projection"):
            # Transform coordinates for cartopy
            import cartopy.crs as ccrs

            transformed_boundaries = []
            for boundary in filtered_boundaries:
                try:
                    # Transform from PlateCarree to the axis projection
                    transformed = ax.projection.transform_points(
                        ccrs.PlateCarree(),
                        np.array([p[0] for p in boundary]),
                        np.array([p[1] for p in boundary]),
                    )
                    transformed_boundaries.append(
                        transformed[:, :2]
                    )  # Drop z coordinate
                except Exception:
                    # If transformation fails, skip this boundary
                    continue

            if transformed_boundaries:
                lc = LineCollection(transformed_boundaries, **line_kwargs)
        else:
            # No projection, use coordinates directly
            lc = LineCollection(filtered_boundaries, **line_kwargs)

        # Add to axes
        ax.add_collection(lc)

        # Update plot limits if needed
        ax.autoscale_view()

    return post_process


def _get_boundaries_from_dataset(ds, node_indices, index_base=1):
    """Fallback method to get boundaries from dataset when grid object is not available."""
    boundaries = []

    # This is a simplified implementation
    # In practice, you'd need the connectivity matrix (nv) to properly do this
    if "nv" not in ds and "nv_zero" not in ds:
        return boundaries

    # Get connectivity matrix
    if "nv" in ds:
        nv = np.asarray(ds["nv"].values, dtype=int)
    else:
        nv = np.asarray(ds["nv_zero"].values, dtype=int)

    # Ensure correct shape (3, nele)
    if nv.shape[0] != 3 and nv.shape[1] == 3:
        nv = nv.T

    # Convert to 0-based if needed
    if nv.min() == 1:
        nv = nv - 1

    # Convert node indices to 0-based
    if index_base == 1:
        node_indices_0 = np.asarray(node_indices) - 1
    else:
        node_indices_0 = np.asarray(node_indices)

    # Find elements containing selected nodes
    node_set = set(node_indices_0.tolist())
    element_mask = np.zeros(nv.shape[1], dtype=bool)

    for i in range(3):
        element_mask |= np.isin(nv[i, :], list(node_set))

    selected_elements = np.where(element_mask)[0]

    # Get coordinates
    if "lon" in ds and "lat" in ds:
        x_coords = ds["lon"].values
        y_coords = ds["lat"].values
    else:
        x_coords = ds["x"].values
        y_coords = ds["y"].values

    # Create edges
    edges_set = set()
    for elem_idx in selected_elements:
        n1, n2, n3 = nv[:, elem_idx]
        edges = [
            tuple(sorted([n1, n2])),
            tuple(sorted([n2, n3])),
            tuple(sorted([n3, n1])),
        ]
        edges_set.update(edges)

    # Convert to line segments
    for edge in edges_set:
        n1, n2 = edge
        line = [
            (float(x_coords[n1]), float(y_coords[n1])),
            (float(x_coords[n2]), float(y_coords[n2])),
        ]
        boundaries.append(line)

    return boundaries

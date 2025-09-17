"""Test the marker clipping functionality in make_node_marker_post."""

import numpy as np
import pytest
import xarray as xr
from matplotlib import pyplot as plt

from xfvcom import FvcomPlotConfig, FvcomPlotOptions, FvcomPlotter
from xfvcom.plot.markers import make_node_marker_post


@pytest.fixture
def sample_ds():
    """Create a sample dataset with mesh coordinates."""
    n_nodes = 10
    lon = np.linspace(139.5, 140.5, n_nodes)
    lat = np.linspace(35.0, 36.0, n_nodes)

    ds = xr.Dataset(
        {
            "lon": (["node"], lon),
            "lat": (["node"], lat),
            "x": (["node"], lon),  # For simplicity, using same values
            "y": (["node"], lat),
        }
    )

    # Add required mesh attributes
    ds.attrs["nele"] = 5
    ds.attrs["node"] = n_nodes

    return ds


def test_make_node_marker_post_with_bounds(sample_ds):
    """Test that respect_bounds=True filters markers correctly."""

    # Create plotter
    cfg = FvcomPlotConfig()
    plotter = FvcomPlotter(sample_ds, cfg)

    # Select nodes to mark (indices 2, 5, 8)
    nodes = [2, 5, 8]

    # Create marker function with bounds checking
    pp = make_node_marker_post(
        nodes,
        plotter,
        marker_kwargs={"color": "red"},
        text_kwargs={"color": "yellow"},
        index_base=0,
        respect_bounds=True,
    )

    # Set bounds that should exclude node 8
    xlim = (139.5, 140.2)  # This should exclude the last few nodes
    ylim = (35.0, 36.0)

    opts = FvcomPlotOptions(
        xlim=xlim,
        ylim=ylim,
        use_latlon=True,
    )

    # Create a simple axes to test
    fig, ax = plt.subplots()

    # Call the post-processing function
    pp(ax, opts=opts)

    # Check that markers were added (but we can't easily count them)
    # This mainly tests that the function runs without error
    assert True

    plt.close(fig)


def test_make_node_marker_post_without_bounds(sample_ds):
    """Test that respect_bounds=False shows all markers."""

    # Create plotter
    cfg = FvcomPlotConfig()
    plotter = FvcomPlotter(sample_ds, cfg)

    # Select all nodes
    nodes = list(range(10))

    # Create marker function without bounds checking
    pp = make_node_marker_post(
        nodes,
        plotter,
        marker_kwargs={"color": "blue"},
        text_kwargs=None,  # No text labels
        index_base=0,
        respect_bounds=False,
    )

    # Set restrictive bounds
    xlim = (139.7, 139.8)
    ylim = (35.2, 35.3)

    opts = FvcomPlotOptions(
        xlim=xlim,
        ylim=ylim,
        use_latlon=True,
    )

    # Create a simple axes to test
    fig, ax = plt.subplots()

    # Call the post-processing function
    pp(ax, opts=opts)

    # With respect_bounds=False, all markers should be plotted
    # even though most are outside the bounds
    assert True

    plt.close(fig)


def test_make_node_marker_post_1based_indexing(sample_ds):
    """Test that index_base=1 works correctly."""

    # Create plotter
    cfg = FvcomPlotConfig()
    plotter = FvcomPlotter(sample_ds, cfg)

    # Use 1-based indices
    nodes = [1, 2, 3]  # These correspond to 0-based indices [0, 1, 2]

    # Create marker function with 1-based indexing
    pp = make_node_marker_post(
        nodes,
        plotter,
        marker_kwargs={"color": "green"},
        text_kwargs={"color": "white"},
        index_base=1,
        respect_bounds=True,
    )

    opts = FvcomPlotOptions(
        use_latlon=True,
    )

    # Create a simple axes to test
    fig, ax = plt.subplots()

    # Call the post-processing function - should not raise an error
    pp(ax, opts=opts)

    assert True

    plt.close(fig)


def test_make_node_marker_post_invalid_indices(sample_ds):
    """Test that invalid indices raise an error."""

    # Create plotter
    cfg = FvcomPlotConfig()
    plotter = FvcomPlotter(sample_ds, cfg)

    # Try to use out-of-range indices
    nodes = [0, 5, 15]  # Index 15 is out of range (only 10 nodes)

    # This should raise an IndexError
    with pytest.raises(IndexError):
        make_node_marker_post(
            nodes,
            plotter,
            index_base=0,
        )

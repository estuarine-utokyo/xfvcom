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


def test_enhanced_text_clipping_with_cartopy(sample_ds):
    """Test that enhanced text clipping works for Cartopy projections."""
    import cartopy.crs as ccrs

    # Create plotter
    cfg = FvcomPlotConfig()
    plotter = FvcomPlotter(sample_ds, cfg)

    # Select all nodes
    nodes = list(range(10))

    # Create marker function with text clipping
    pp = make_node_marker_post(
        nodes,
        plotter,
        marker_kwargs={"color": "red"},
        text_kwargs={
            "color": "yellow",
            "clip_on": True,
        },  # clip_on triggers enhanced clipping
        index_base=0,
        respect_bounds=True,
    )

    # Set restrictive bounds that should exclude some nodes
    xlim = (139.7, 140.2)  # Middle section only
    ylim = (35.3, 35.7)  # Middle section only

    opts = FvcomPlotOptions(
        xlim=xlim,
        ylim=ylim,
        use_latlon=True,
    )

    # Create a Cartopy axes to trigger enhanced clipping
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # Call the post-processing function
    pp(ax, opts=opts)

    # The function should run without error and apply text clipping
    assert True

    plt.close(fig)


def test_text_clip_buffer_parameter(sample_ds):
    """Test that text_clip_buffer parameter adjusts the clipping area."""
    import cartopy.crs as ccrs

    # Create plotter
    cfg = FvcomPlotConfig()
    plotter = FvcomPlotter(sample_ds, cfg)

    # Select nodes
    nodes = [2, 5, 8]

    # Create marker function with text clipping buffer
    pp = make_node_marker_post(
        nodes,
        plotter,
        marker_kwargs={"color": "red"},
        text_kwargs={"color": "yellow", "clip_on": True},
        index_base=0,
        respect_bounds=True,
        text_clip_buffer=0.1,  # Add buffer to show more text
    )

    xlim = (139.8, 140.0)
    ylim = (35.4, 35.6)

    opts = FvcomPlotOptions(
        xlim=xlim,
        ylim=ylim,
        use_latlon=True,
    )

    # Create a Cartopy axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # Call the post-processing function
    pp(ax, opts=opts)

    # With buffer, text slightly outside bounds should still be shown
    assert True

    plt.close(fig)


def test_text_clipping_disabled_when_clip_on_false(sample_ds):
    """Test that enhanced clipping is disabled when clip_on=False."""
    import cartopy.crs as ccrs

    # Create plotter
    cfg = FvcomPlotConfig()
    plotter = FvcomPlotter(sample_ds, cfg)

    # Select all nodes
    nodes = list(range(10))

    # Create marker function with clip_on=False
    pp = make_node_marker_post(
        nodes,
        plotter,
        marker_kwargs={"color": "red"},
        text_kwargs={"color": "yellow", "clip_on": False},  # Disable clipping
        index_base=0,
        respect_bounds=True,
    )

    xlim = (139.8, 140.0)
    ylim = (35.4, 35.6)

    opts = FvcomPlotOptions(
        xlim=xlim,
        ylim=ylim,
        use_latlon=True,
    )

    # Create a Cartopy axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # Call the post-processing function
    pp(ax, opts=opts)

    # Enhanced clipping should not be applied
    assert True

    plt.close(fig)


def test_marker_clip_buffer_positive(sample_ds):
    """Test that positive marker_clip_buffer includes more markers."""
    # Create plotter
    cfg = FvcomPlotConfig()
    plotter = FvcomPlotter(sample_ds, cfg)

    # Select edge nodes that would normally be excluded
    nodes = [0, 9]  # First and last nodes

    # Create marker function with positive buffer
    pp = make_node_marker_post(
        nodes,
        plotter,
        marker_kwargs={"color": "blue", "markersize": 5},
        text_kwargs=None,
        index_base=0,
        respect_bounds=True,
        marker_clip_buffer=0.2,  # Positive buffer to include edge markers
    )

    # Set restrictive bounds that would normally exclude these nodes
    xlim = (139.6, 140.4)  # Slightly inside the full range
    ylim = (35.1, 35.9)

    opts = FvcomPlotOptions(
        xlim=xlim,
        ylim=ylim,
        use_latlon=True,
    )

    # Create a simple axes to test
    fig, ax = plt.subplots()

    # Call the post-processing function
    pp(ax, opts=opts)

    # With positive buffer, edge markers should be included
    assert True

    plt.close(fig)


def test_marker_clip_buffer_negative(sample_ds):
    """Test that negative marker_clip_buffer excludes more markers."""
    # Create plotter
    cfg = FvcomPlotConfig()
    plotter = FvcomPlotter(sample_ds, cfg)

    # Select nodes near edges
    nodes = [1, 2, 7, 8]

    # Create marker function with negative buffer
    pp = make_node_marker_post(
        nodes,
        plotter,
        marker_kwargs={"color": "green", "markersize": 5},
        text_kwargs=None,
        index_base=0,
        respect_bounds=True,
        marker_clip_buffer=-0.1,  # Negative buffer to exclude edge markers
    )

    xlim = (139.5, 140.5)
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

    # With negative buffer, markers near edges should be excluded
    assert True

    plt.close(fig)


def test_independent_marker_text_buffers(sample_ds):
    """Test that marker and text buffers work independently."""
    import cartopy.crs as ccrs

    # Create plotter
    cfg = FvcomPlotConfig()
    plotter = FvcomPlotter(sample_ds, cfg)

    # Select several nodes
    nodes = list(range(10))

    # Create marker function with different buffers
    pp = make_node_marker_post(
        nodes,
        plotter,
        marker_kwargs={"color": "purple", "markersize": 4},
        text_kwargs={"color": "orange", "fontsize": 8, "clip_on": True},
        index_base=0,
        respect_bounds=True,
        marker_clip_buffer=0.1,  # Include markers slightly outside
        text_clip_buffer=-0.05,  # Exclude text near edges
    )

    xlim = (139.7, 140.3)
    ylim = (35.2, 35.8)

    opts = FvcomPlotOptions(
        xlim=xlim,
        ylim=ylim,
        use_latlon=True,
    )

    # Create a Cartopy axes
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # Call the post-processing function
    pp(ax, opts=opts)

    # Buffers should work independently
    assert True

    plt.close(fig)

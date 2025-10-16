"""Tests for node element boundary functions."""

import numpy as np
import pytest

from xfvcom.grid import FvcomGrid


def test_get_boundaries_as_lines():
    """Test getting element boundaries as line segments."""
    # Create a simple triangular mesh with one triangle
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    lon = x  # Use same values for geographic coords
    lat = y
    nv = np.array([[0, 1, 2]]).T  # One triangle, 0-based

    grid = FvcomGrid(x=x, y=y, nv=nv, lon=lon, lat=lat)

    # Test with all nodes (1-based indexing)
    boundaries = grid.get_node_element_boundaries(
        [1, 2, 3], index_base=1, return_as="lines"
    )

    # Should have 3 edges for a single triangle
    assert len(boundaries) == 3, f"Expected 3 edges, got {len(boundaries)}"

    # Each edge should have 2 points
    for boundary in boundaries:
        assert len(boundary) == 2, "Each edge should have 2 points"
        assert len(boundary[0]) == 2, "Each point should have (x, y) coordinates"


def test_get_boundaries_as_polygons():
    """Test getting element boundaries as closed polygons."""
    # Create a simple triangular mesh with one triangle
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    lon = x
    lat = y
    nv = np.array([[0, 1, 2]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv, lon=lon, lat=lat)

    # Test with all nodes
    polygons = grid.get_node_element_boundaries(
        [1, 2, 3], index_base=1, return_as="polygons"
    )

    # Should have 1 triangle
    assert len(polygons) == 1, f"Expected 1 polygon, got {len(polygons)}"

    # Triangle should be closed (4 points, first == last)
    assert len(polygons[0]) == 4, "Triangle should have 4 points (closed)"
    assert polygons[0][0] == polygons[0][-1], "First and last points should be same"


def test_boundaries_with_multiple_triangles():
    """Test boundaries for multiple triangles."""
    # Create a mesh with two triangles forming a square
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    lon = x
    lat = y
    # Two triangles: (0,1,2) and (0,2,3)
    nv = np.array([[0, 1, 2], [0, 2, 3]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv, lon=lon, lat=lat)

    # Get boundaries for all nodes
    lines = grid.get_node_element_boundaries(
        [1, 2, 3, 4], index_base=1, return_as="lines"
    )

    # Should have 5 unique edges (shared edge counted once)
    # Edges: (0,1), (1,2), (2,0), (2,3), (3,0)
    assert len(lines) == 5, f"Expected 5 unique edges, got {len(lines)}"

    # Get as polygons
    polygons = grid.get_node_element_boundaries(
        [1, 2, 3, 4], index_base=1, return_as="polygons"
    )
    assert len(polygons) == 2, f"Expected 2 triangles, got {len(polygons)}"


def test_boundaries_single_node():
    """Test boundaries for a single node."""
    # Create a mesh with multiple triangles
    x = np.array([0.0, 1.0, 0.5, 1.5])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    lon = x
    lat = y
    # Two triangles sharing node 0: (0,1,2) and (0,1,3)
    nv = np.array([[0, 1, 2], [0, 1, 3]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv, lon=lon, lat=lat)

    # Get boundaries for node 1 only (0-based: node 0)
    lines = grid.get_node_element_boundaries([1], index_base=1, return_as="lines")

    # Both triangles contain node 1, so we get all edges from both
    # Unique edges: (0,1), (1,2), (0,2), (1,3), (0,3)
    assert (
        len(lines) == 5
    ), f"Expected 5 edges for triangles containing node 1, got {len(lines)}"


def test_boundaries_empty_nodes():
    """Test boundaries with empty node list."""
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    lon = x
    lat = y
    nv = np.array([[0, 1, 2]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv, lon=lon, lat=lat)

    boundaries = grid.get_node_element_boundaries([], index_base=1, return_as="lines")
    assert boundaries == [], "Expected empty list for no nodes"


def test_boundaries_invalid_return_type():
    """Test that invalid return_as parameter raises error."""
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    lon = x
    lat = y
    nv = np.array([[0, 1, 2]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv, lon=lon, lat=lat)

    with pytest.raises(ValueError, match="return_as must be"):
        grid.get_node_element_boundaries([1], index_base=1, return_as="invalid")


def test_boundaries_zero_based_indexing():
    """Test boundaries with zero-based node indexing."""
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    lon = x
    lat = y
    nv = np.array([[0, 1, 2]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv, lon=lon, lat=lat)

    # Test with 0-based indexing
    boundaries = grid.get_node_element_boundaries(
        [0, 1, 2], index_base=0, return_as="lines"
    )
    assert len(boundaries) == 3, "Expected 3 edges with 0-based indexing"


def test_boundaries_all_nodes():
    """Test getting boundaries for all nodes (None parameter)."""
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    lon = x
    lat = y
    nv = np.array([[0, 1, 2]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv, lon=lon, lat=lat)

    boundaries = grid.get_node_element_boundaries(None, index_base=1, return_as="lines")
    assert len(boundaries) == 3, "Expected 3 edges for all nodes"

"""Tests for node area calculation functionality."""

import numpy as np
import pytest

from xfvcom.grid import FvcomGrid


def test_calculate_node_area_simple_triangle():
    """Test area calculation for a single triangle."""
    # Create a simple triangular mesh with one triangle
    # Triangle vertices at (0,0), (1,0), (0,1) - area should be 0.5
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    nv = np.array([[0, 1, 2]]).T  # One triangle, 0-based

    grid = FvcomGrid(x=x, y=y, nv=nv)

    # Test with 1-based indexing (FVCOM default)
    area = grid.calculate_node_area([1, 2, 3], index_base=1)
    assert np.isclose(area, 0.5), f"Expected area 0.5, got {area}"

    # Test with 0-based indexing
    area = grid.calculate_node_area([0, 1, 2], index_base=0)
    assert np.isclose(area, 0.5), f"Expected area 0.5, got {area}"

    # Test with single node
    area = grid.calculate_node_area([1], index_base=1)
    assert np.isclose(area, 0.5), f"Expected area 0.5, got {area}"


def test_calculate_node_area_multiple_triangles():
    """Test area calculation for multiple triangles."""
    # Create a mesh with two triangles forming a square
    # Square vertices at (0,0), (1,0), (1,1), (0,1)
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    # Two triangles: (0,1,2) and (0,2,3)
    nv = np.array([[0, 1, 2], [0, 2, 3]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv)

    # Test all nodes - should give total area of 1.0
    area = grid.calculate_node_area([1, 2, 3, 4], index_base=1)
    assert np.isclose(area, 1.0), f"Expected area 1.0, got {area}"

    # Test nodes of first triangle only
    area = grid.calculate_node_area([1, 2, 3], index_base=1)
    assert np.isclose(
        area, 1.0
    ), f"Expected area 1.0 (both triangles share these nodes), got {area}"

    # Test single node that's in both triangles
    area = grid.calculate_node_area([1], index_base=1)  # Node 1 (0-based: 0)
    assert np.isclose(area, 1.0), f"Expected area 1.0, got {area}"


def test_calculate_node_area_no_nodes():
    """Test area calculation with empty node list."""
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    nv = np.array([[0, 1, 2]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv)

    area = grid.calculate_node_area([], index_base=1)
    assert area == 0.0, f"Expected area 0.0 for empty node list, got {area}"


def test_calculate_node_area_invalid_nodes():
    """Test area calculation with invalid node indices."""
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    nv = np.array([[0, 1, 2]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv)

    # Test with out-of-range node index (1-based)
    with pytest.raises(ValueError, match="Invalid node indices"):
        grid.calculate_node_area([5], index_base=1)

    # Test with negative node index (0-based)
    with pytest.raises(ValueError, match="Invalid node indices"):
        grid.calculate_node_area([-1], index_base=0)


def test_calculate_node_area_all_nodes():
    """Test area calculation with None (all nodes)."""
    x = np.array([0.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0])
    nv = np.array([[0, 1, 2]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv)

    area = grid.calculate_node_area(None, index_base=1)
    assert np.isclose(area, 0.5), f"Expected area 0.5 for all nodes, got {area}"

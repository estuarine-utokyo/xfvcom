"""Tests for median-dual control volume area calculations."""

import numpy as np
import pytest

from xfvcom.grid import FvcomGrid


def test_simple_triangle_control_volumes():
    """Test control volumes for a single triangle."""
    # Create an equilateral triangle
    x = np.array([0.0, 1.0, 0.5])
    y = np.array([0.0, 0.0, np.sqrt(3) / 2])
    lon = x
    lat = y
    nv = np.array([[0, 1, 2]]).T  # One triangle, 0-based

    grid = FvcomGrid(x=x, y=y, nv=nv, lon=lon, lat=lat)

    # Get control volume for interior point (if all nodes were interior)
    # Each node should get 1/3 of the triangle area
    triangle_area = 0.5 * 1.0 * np.sqrt(3) / 2  # Base * Height / 2

    # For a single triangle, each node's control volume is approximately 1/3
    area1 = grid.calculate_node_area_median_dual([1], index_base=1)
    area2 = grid.calculate_node_area_median_dual([2], index_base=1)
    area3 = grid.calculate_node_area_median_dual([3], index_base=1)

    # Due to boundary effects in single triangle, areas won't be exactly equal
    # but total should equal triangle area
    total_area = area1 + area2 + area3
    assert abs(total_area - triangle_area) < 1e-10, "Total area should equal triangle area"


def test_regular_hexagon_control_volume():
    """Test control volume for a node surrounded by 6 triangles in a regular pattern."""
    # Create a regular hexagon with center node
    # Center at origin, 6 nodes in a circle
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 points
    radius = 1.0

    # Nodes: center + 6 surrounding
    x = np.concatenate([[0.0], radius * np.cos(angles)])
    y = np.concatenate([[0.0], radius * np.sin(angles)])
    lon = x
    lat = y

    # Create 6 triangles all sharing the center node
    # Triangle i connects center, node i, node (i+1)%6
    triangles = []
    for i in range(6):
        triangles.append([0, i + 1, (i % 6) + 1])  # 0-based

    nv = np.array(triangles).T

    grid = FvcomGrid(x=x, y=y, nv=nv, lon=lon, lat=lat)

    # The control volume for the center node should be a regular hexagon
    # formed by connecting the centroids and edge midpoints
    center_area = grid.calculate_node_area_median_dual([1], index_base=1)

    # For a regular hexagon pattern, the control volume is smaller than
    # the area of the 6 triangles
    six_triangle_area = 6 * 0.5 * radius * radius * np.sin(np.pi / 3)

    # The median-dual control volume is approximately 0.5 of the triangles
    assert center_area < six_triangle_area
    assert center_area > 0.4 * six_triangle_area  # Reasonable bounds


def test_boundary_node_control_volume():
    """Test control volume for boundary/coastal nodes."""
    # Create a simple coastal mesh: 4 nodes forming 2 triangles
    # Bottom two nodes are on boundary
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    lon = x
    lat = y

    # Two triangles: (0,1,2) and (1,3,2)
    nv = np.array([[0, 1, 2], [1, 3, 2]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv, lon=lon, lat=lat)

    # Nodes 0 and 1 are boundary nodes (only in 1 and 2 triangles respectively)
    # Their control volumes should include boundary closure
    boundary_area1 = grid.calculate_node_area_median_dual([1], index_base=1)  # Node 0
    boundary_area2 = grid.calculate_node_area_median_dual([2], index_base=1)  # Node 1

    # Interior nodes
    interior_area1 = grid.calculate_node_area_median_dual([3], index_base=1)  # Node 2
    interior_area2 = grid.calculate_node_area_median_dual([4], index_base=1)  # Node 3

    # Total area should equal the area of the two triangles
    total_mesh_area = 2 * 0.5  # Two triangles, each with area 0.5
    total_cv_area = boundary_area1 + boundary_area2 + interior_area1 + interior_area2

    assert abs(total_cv_area - total_mesh_area) < 1e-10, "Total CV area should equal mesh area"


def test_control_volume_polygon_structure():
    """Test that control volume polygons are properly closed."""
    x = np.array([0.0, 1.0, 0.5])
    y = np.array([0.0, 0.0, 1.0])
    lon = x
    lat = y
    nv = np.array([[0, 1, 2]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv, lon=lon, lat=lat)

    # Get control volume polygon for node 1
    polygon = grid.get_node_control_volume(1, index_base=1)

    # Check it's closed
    assert len(polygon) >= 4, "Polygon should have at least 4 points (triangle)"
    assert polygon[0] == polygon[-1], "Polygon should be closed"

    # Check coordinates are tuples of floats
    for point in polygon:
        assert len(point) == 2, "Each point should have 2 coordinates"
        assert isinstance(point[0], (float, np.floating))
        assert isinstance(point[1], (float, np.floating))


def test_get_multiple_control_volumes():
    """Test getting control volumes for multiple nodes."""
    # Create a mesh with 4 nodes and 2 triangles
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    lon = x
    lat = y
    nv = np.array([[0, 1, 2], [0, 2, 3]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv, lon=lon, lat=lat)

    # Get control volumes for nodes 1 and 2
    volumes = grid.get_node_control_volumes([1, 2], index_base=1)

    assert len(volumes) == 2, "Should get 2 control volumes"

    for vol in volumes:
        assert len(vol) >= 4, "Each volume should have at least 4 points"
        assert vol[0] == vol[-1], "Each volume should be closed"


def test_median_dual_vs_triangle_area():
    """Test that median-dual gives different (correct) results than triangle-based."""
    # Create a simple mesh
    x = np.array([0.0, 2.0, 1.0])
    y = np.array([0.0, 0.0, 2.0])
    lon = x
    lat = y
    nv = np.array([[0, 1, 2]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv, lon=lon, lat=lat)

    # Calculate using both methods
    triangle_area = grid.calculate_node_area([1, 2, 3], index_base=1)
    median_dual_area = grid.calculate_node_area_median_dual([1, 2, 3], index_base=1)

    # The triangle area is simply the area of the triangle (2.0)
    assert abs(triangle_area - 2.0) < 1e-10

    # The median-dual area should also be 2.0 (sum of all control volumes)
    # but calculated differently
    assert abs(median_dual_area - 2.0) < 1e-10

    # Individual node areas should differ between methods
    tri_area_1 = grid.calculate_node_area([1], index_base=1)
    md_area_1 = grid.calculate_node_area_median_dual([1], index_base=1)

    # For a single triangle, triangle method gives full triangle area
    assert abs(tri_area_1 - 2.0) < 1e-10
    # But median-dual gives only the control volume portion
    assert md_area_1 < 2.0
    assert md_area_1 > 0


def test_all_nodes_median_dual():
    """Test calculating median-dual area for all nodes (None parameter)."""
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    lon = x
    lat = y
    nv = np.array([[0, 1, 2], [1, 3, 2]]).T

    grid = FvcomGrid(x=x, y=y, nv=nv, lon=lon, lat=lat)

    # Get area for all nodes
    total_area = grid.calculate_node_area_median_dual(None, index_base=1)

    # Should equal total mesh area (2 triangles, each area 0.5)
    assert abs(total_area - 1.0) < 1e-10
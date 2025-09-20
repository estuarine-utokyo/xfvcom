"""Tests for triangular element area calculations."""

import numpy as np
import pytest

from xfvcom.grid import FvcomGrid


def _build_simple_grid() -> FvcomGrid:
    # Square split into two triangles
    x = np.array([0.0, 1.0, 1.0, 0.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    nv = np.array([[0, 1], [1, 2], [3, 3]])  # columns correspond to elements
    return FvcomGrid(x=x, y=y, nv=nv)


def test_single_triangle_area():
    grid = _build_simple_grid()
    areas = grid.calculate_element_area([1], index_base=1)
    assert np.allclose(areas, [0.5])


def test_all_elements_default():
    grid = _build_simple_grid()
    areas = grid.calculate_element_area()
    assert areas.shape == (2,)
    assert np.isclose(areas.sum(), 1.0)


def test_zero_based_indices():
    grid = _build_simple_grid()
    areas = grid.calculate_element_area([0, 1], index_base=0)
    assert np.allclose(areas, [0.5, 0.5])


def test_invalid_indices_raise():
    grid = _build_simple_grid()
    with pytest.raises(ValueError):
        grid.calculate_element_area([0, 3], index_base=0)

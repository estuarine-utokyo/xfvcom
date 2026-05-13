# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Tests for :mod:`xfvcom.grid.sigma`."""

from __future__ import annotations

import numpy as np
import pytest

from xfvcom.grid.sigma import sigma_from_z, vertical_interp, z_from_sigma


class TestSigmaZRoundTrip:
    def test_scalar(self):
        h = 100.0
        sigma = -0.3
        z = z_from_sigma(sigma, h)
        assert z == pytest.approx(-30.0)
        assert sigma_from_z(z, h) == pytest.approx(sigma)

    def test_with_eta(self):
        h = 100.0
        eta = 1.5
        sigma = -0.5
        z = z_from_sigma(sigma, h, eta)
        assert z == pytest.approx(-0.5 * (100.0 + 1.5))
        assert sigma_from_z(z, h, eta) == pytest.approx(sigma)

    def test_broadcasting(self):
        sigma = np.array([0.0, -0.5, -1.0])
        h = np.array([100.0, 50.0])
        # sigma column (3,) × h row (2,) → (3, 2)
        z = z_from_sigma(sigma[:, None], h[None, :])
        assert z.shape == (3, 2)
        np.testing.assert_allclose(z[0], 0.0)
        np.testing.assert_allclose(z[-1], -h)


class TestVerticalInterp:
    def test_identity(self):
        src_z = np.array([0.0, -25.0, -50.0, -75.0, -100.0])
        profile = np.array([20.0, 18.0, 15.0, 10.0, 5.0])
        out = vertical_interp(profile, src_z, src_z)
        np.testing.assert_allclose(out, profile)

    def test_subsample(self):
        src_z = np.linspace(0.0, -100.0, 5)
        profile = np.linspace(20.0, 5.0, 5)
        dst_z = np.array([-25.0, -75.0])
        out = vertical_interp(profile, src_z, dst_z)
        # Linear: f(-25) ≈ 16.25, f(-75) ≈ 8.75 (matches the linear profile)
        np.testing.assert_allclose(out, [16.25, 8.75], rtol=1e-12)

    def test_unsorted_src(self):
        src_z = np.array([-100.0, 0.0, -50.0])
        profile = np.array([5.0, 20.0, 12.5])
        out = vertical_interp(profile, src_z, [-25.0, -75.0])
        np.testing.assert_allclose(out, [16.25, 8.75], rtol=1e-12)

    def test_extrapolate_constant(self):
        src_z = np.array([-10.0, -50.0])
        profile = np.array([18.0, 10.0])
        out = vertical_interp(profile, src_z, np.array([0.0, -100.0]))
        # Above src top → take surface value; below src bottom → take bottom value
        assert out[0] == pytest.approx(18.0)
        assert out[1] == pytest.approx(10.0)

    def test_nan_handling(self):
        src_z = np.array([0.0, -10.0, -20.0, -30.0])
        profile = np.array([20.0, np.nan, 15.0, 10.0])
        out = vertical_interp(profile, src_z, np.array([-10.0, -25.0]))
        # NaN at -10 is dropped, interpolating between 0 (20) and -20 (15) for
        # the -10 target gives 17.5; -25 between -20 (15) and -30 (10) gives 12.5
        np.testing.assert_allclose(out, [17.5, 12.5], rtol=1e-12)

    def test_all_nan(self):
        out = vertical_interp(
            np.array([np.nan, np.nan]),
            np.array([0.0, -10.0]),
            np.array([-5.0]),
        )
        assert np.isnan(out).all()

    def test_single_valid(self):
        out = vertical_interp(
            np.array([np.nan, 12.0]),
            np.array([0.0, -10.0]),
            np.array([-5.0, -50.0]),
        )
        np.testing.assert_allclose(out, [12.0, 12.0])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            vertical_interp(np.array([1.0, 2.0]), np.array([0.0]), np.array([0.0]))

    def test_masked_array_input(self):
        ma = np.ma.array([20.0, 15.0, 10.0], mask=[False, True, False])
        out = vertical_interp(
            ma,
            np.array([0.0, -10.0, -20.0]),
            np.array([-10.0]),
        )
        # The masked middle value should be treated as NaN; the -10 target
        # falls between the two valid endpoints (0, -20) → linear avg = 15
        np.testing.assert_allclose(out, [15.0])

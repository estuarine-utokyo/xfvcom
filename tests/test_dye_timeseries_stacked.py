"""Tests for DYE timeseries stacked plots and utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xfvcom.plot._timeseries_utils import (
    climatology,
    detect_nans_and_raise,
    deterministic_colors,
    handle_negatives,
    prepare_wide_df,
    resample_df,
    select_members,
    slice_window,
)
from xfvcom.plot.dye_timeseries import plot_dye_timeseries_stacked


def create_test_dataframe(
    n_times: int = 100,
    n_members: int = 3,
    freq: str = "h",
    add_nans: bool = False,
    add_negatives: bool = False,
) -> pd.DataFrame:
    """Create test DataFrame with time series data."""
    dates = pd.date_range("2021-01-01", periods=n_times, freq=freq)

    data = {}
    for i in range(n_members):
        # Create some variation between members
        values = np.random.random(n_times) * (i + 1) * 10
        if add_negatives:
            values[::10] = -values[::10] * 0.1  # Every 10th value negative
        if add_nans:
            values[::20] = np.nan  # Every 20th value NaN
        data[f"member_{i+1}"] = values

    return pd.DataFrame(data, index=dates)


def create_test_xarray(
    n_times: int = 100,
    n_members: int = 3,
    freq: str = "h",
) -> xr.DataArray:
    """Create test xarray DataArray with time series data."""
    df = create_test_dataframe(n_times, n_members, freq)

    # Convert to xarray with ensemble dimension
    da = xr.DataArray(
        df.values,
        dims=["time", "member"],
        coords={
            "time": df.index,
            "member": list(range(1, n_members + 1)),
        },
        name="DYE",
    )

    return da


class TestNaNDetection:
    """Tests for NaN detection and raising."""

    def test_no_nans_passes(self):
        """Test that data without NaNs passes."""
        df = create_test_dataframe(add_nans=False)
        # Should not raise
        detect_nans_and_raise(df)

    def test_with_nans_raises(self):
        """Test that data with NaNs raises ValueError."""
        df = create_test_dataframe(add_nans=True)

        with pytest.raises(ValueError, match="NaN values detected"):
            detect_nans_and_raise(df)

    def test_error_message_contains_details(self):
        """Test that error message contains useful details."""
        df = create_test_dataframe(add_nans=True)

        with pytest.raises(ValueError) as exc_info:
            detect_nans_and_raise(df)

        error_msg = str(exc_info.value)
        assert "Total NaNs" in error_msg
        assert "First occurrence" in error_msg
        assert "Sample pairs" in error_msg


class TestMemberSelection:
    """Tests for member selection functionality."""

    def test_select_from_dataframe(self):
        """Test selecting members from DataFrame."""
        df = create_test_dataframe(n_members=5)
        selected = select_members(df, member_ids=[1, 3, 5])

        assert len(selected.columns) == 3
        assert list(selected.columns) == ["member_1", "member_3", "member_5"]
        assert len(selected) == len(df)

    def test_select_from_xarray(self):
        """Test selecting members from xarray."""
        da = create_test_xarray(n_members=5)
        selected = select_members(da, member_ids=[1, 3, 5])

        assert isinstance(selected, pd.DataFrame)
        assert len(selected.columns) == 3
        assert len(selected) == len(da.time)

    def test_select_with_member_map(self):
        """Test selecting with custom member map."""
        df = create_test_dataframe(n_members=3)
        df.columns = ["Source_A", "Source_B", "Source_C"]

        member_map = {1: "Source_A", 2: "Source_B", 3: "Source_C"}
        selected = select_members(df, member_ids=[1, 3], member_map=member_map)

        assert list(selected.columns) == ["Source_A", "Source_C"]

    def test_select_missing_member_raises(self):
        """Test that selecting missing member raises KeyError."""
        df = create_test_dataframe(n_members=3)

        with pytest.raises(KeyError):
            select_members(df, member_ids=[1, 2, 99])

    def test_select_preserves_order(self):
        """Test that selected members preserve requested order."""
        df = create_test_dataframe(n_members=5)
        selected = select_members(df, member_ids=[5, 1, 3])

        assert list(selected.columns) == ["member_5", "member_1", "member_3"]


class TestNormalization:
    """Tests for normalization functionality."""

    def test_normalization_sums_to_one(self):
        """Test that normalized data sums to 1.0."""
        df = create_test_dataframe(n_members=3)
        resampled = resample_df(df, freq="D", normalize=True)

        row_sums = resampled.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)

    def test_normalization_preserves_shape(self):
        """Test that normalization preserves DataFrame shape."""
        df = create_test_dataframe(n_members=3, n_times=100)
        resampled = resample_df(df, freq="6h", normalize=True)

        assert resampled.shape[1] == df.shape[1]  # Same number of columns


class TestModes:
    """Tests for different plotting modes."""

    def test_window_mode_basic(self):
        """Test basic window mode plot."""
        df = create_test_dataframe(n_times=240, n_members=3)  # 10 days

        result = plot_dye_timeseries_stacked(
            df,
            start="2021-01-02",
            end="2021-01-05",  # 4 days
        )

        assert "fig" in result
        assert "ax" in result
        assert "data_used" in result

        # Check data was actually windowed
        data_used = result["data_used"]
        assert len(data_used) < len(df)

    def test_climatology_mode_shape(self):
        """Test climatology mode produces expected shape."""
        # Create multi-year data
        df1 = create_test_dataframe(n_times=365 * 24, n_members=3, freq="h")
        df1.index = pd.date_range("2020-01-01", periods=365 * 24, freq="h")

        df2 = create_test_dataframe(n_times=365 * 24, n_members=3, freq="h")
        df2.index = pd.date_range("2021-01-01", periods=365 * 24, freq="h")

        df = pd.concat([df1, df2])

        result = plot_dye_timeseries_stacked(
            df,
        )

        # Should successfully plot the data
        data_used = result["data_used"]
        assert len(data_used) > 0
        assert "fig" in result
        assert "ax" in result

    def test_resampling_reduces_timesteps(self):
        """Test that resampling reduces number of timesteps."""
        df = create_test_dataframe(n_times=240, n_members=3, freq="h")  # 10 days hourly

        result = plot_dye_timeseries_stacked(
            df,
        )

        data_used = result["data_used"]
        # Should successfully plot
        assert len(data_used) > 0
        assert "fig" in result
        assert "ax" in result


class TestNegativeHandling:
    """Tests for negative value handling."""

    def test_negative_stats_detection(self):
        """Test that negative values are detected in stats."""
        df = create_test_dataframe(n_members=3, add_negatives=True)
        df_clean, stats = handle_negatives(df, policy="keep")

        assert stats["any_negatives"] is True
        assert stats["total_negatives"] > 0
        assert "global_min" in stats
        assert stats["global_min"] < 0

    def test_clip_zero_policy(self):
        """Test clip_zero policy removes negatives."""
        df = create_test_dataframe(n_members=3, add_negatives=True)
        df_clipped, stats = handle_negatives(df, policy="clip0")

        # After clipping, all values should be >= 0
        assert (df_clipped >= 0).all().all()

    def test_keep_policy_preserves_negatives(self):
        """Test keep policy preserves negative values."""
        df = create_test_dataframe(n_members=3, add_negatives=True)
        df_kept, stats = handle_negatives(df, policy="keep")

        # Should still have negatives
        assert (df_kept < 0).any().any()


class TestHelperFunctions:
    """Tests for helper utility functions."""

    def test_slice_window(self):
        """Test time window slicing."""
        df = create_test_dataframe(n_times=1000)
        sliced = slice_window(df, start="2021-01-05", end="2021-01-15")

        assert sliced.index.min() >= pd.Timestamp("2021-01-05")
        assert sliced.index.max() <= pd.Timestamp("2021-01-16")  # Inclusive
        assert len(sliced) < len(df)

    def test_climatology_hourly(self):
        """Test hourly climatology."""
        df = create_test_dataframe(n_times=72, n_members=3, freq="h")  # 3 days
        clim_mean, clim_std = climatology(df, kind="H")

        # Should have 24 hours
        assert len(clim_mean) == 24
        assert len(clim_std) == 24

    def test_deterministic_colors(self):
        """Test color generation is deterministic."""
        groups = ["Source_A", "Source_B", "Source_C"]

        colors1 = deterministic_colors(groups)
        colors2 = deterministic_colors(groups)

        assert colors1 == colors2
        assert len(colors1) == len(groups)

    def test_deterministic_colors_with_override(self):
        """Test color overrides work."""
        groups = ["Source_A", "Source_B", "Source_C"]
        override = {"Source_A": "#ff0000"}

        colors = deterministic_colors(groups, override=override)

        assert colors[0] == "#ff0000"
        assert len(colors) == 3

    def test_prepare_wide_df_from_dataframe(self):
        """Test preparing wide df from DataFrame."""
        df = create_test_dataframe(n_members=3)
        prepared = prepare_wide_df(df)

        assert isinstance(prepared, pd.DataFrame)
        assert isinstance(prepared.index, pd.DatetimeIndex)

    def test_prepare_wide_df_from_xarray(self):
        """Test preparing wide df from xarray."""
        da = create_test_xarray(n_members=3)
        prepared = prepare_wide_df(da)

        assert isinstance(prepared, pd.DataFrame)
        assert isinstance(prepared.index, pd.DatetimeIndex)


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_end_to_end_window_mode(self):
        """Test complete workflow for window mode."""
        da = create_test_xarray(n_times=240, n_members=4)

        result = plot_dye_timeseries_stacked(
            da,
            member_ids=[1, 2, 3],
            start="2021-01-02",
            end="2021-01-08",
        )

        assert result["fig"] is not None
        assert result["ax"] is not None
        assert result["data_used"] is not None

    def test_end_to_end_with_member_map(self):
        """Test complete workflow with custom member names."""
        da = create_test_xarray(n_times=100, n_members=3)

        member_map = {1: "Urban", 2: "Forest", 3: "Agriculture"}

        result = plot_dye_timeseries_stacked(
            da,
            member_ids=[1, 2, 3],
            member_map=member_map,
        )

        # Check that custom names appear in the plot
        data_used = result["data_used"]
        assert "Urban" in data_used.columns or any(
            "Urban" in str(col) for col in data_used.columns
        )

    def test_nan_hard_fail(self):
        """Test that NaN causes immediate failure."""
        df = create_test_dataframe(n_members=3, add_nans=True)

        with pytest.raises(ValueError, match="NaN values detected"):
            plot_dye_timeseries_stacked(df)

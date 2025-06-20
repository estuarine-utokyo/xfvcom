from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xfvcom.io.sources.timeseries import TimeSeriesSource

TEST_FILE = Path(__file__).parent / "data/sample_sjis.tsv"  # test data


def _timeline():
    return pd.date_range("2024-12-31 15:00", periods=5, freq="6h", tz="UTC")


def test_linear_interp():
    src = TimeSeriesSource(TEST_FILE, interp_method="linear")
    out = src.get_series("flux", _timeline())
    assert np.isclose(out[0], 120.0)
    assert np.isclose(out[-1], 130.0)
    # Ensure no NaN values in the output
    assert not np.isnan(out).any()


def test_nearest_interp():

    with pytest.raises(ValueError, match="interp_method must be 'linear'"):
        TimeSeriesSource(TEST_FILE, interp_method="nearest")
    # nearest は禁止なので以降の out 解析は不要


def test_input_timezone():
    src_tokyo = TimeSeriesSource(TEST_FILE)
    src_utc = TimeSeriesSource(TEST_FILE, input_tz="UTC")
    assert src_tokyo._df.index[0] == pd.Timestamp("2024-12-31 15:00")
    assert src_utc._df.index[0] == pd.Timestamp("2025-01-01 00:00")

# -*- coding: utf-8 -*-
"""
Tests for GWO RMK (remark) code handling.

Verifies that variable-specific RMK rules are correctly applied:
- Precipitation (kous): RMK 2,6 → 0.0 (no rain), RMK 0,1 → NaN
- Solar (slht, lght): RMK 2 → 0.0 (nighttime), RMK 0,1 → NaN
- Default variables: RMK 0,1,2 → NaN
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xfvcom.io.gwo_reader import (
    GWO_COLUMNS,
    RMK_RULES,
    VAR_RMK_TYPE,
    GWOReader,
)


def make_test_df(var: str, values: list[float], rmk_codes: list[int]) -> pd.DataFrame:
    """Create a minimal test DataFrame with GWO columns."""
    n = len(values)
    data: dict[str, list] = {col: [0.0] * n for col in GWO_COLUMNS}  # type: ignore[type-arg]

    # Set time columns
    data["YYYY"] = [2020] * n
    data["MM"] = [1] * n
    data["DD"] = [1] * n
    data["HH"] = list(range(1, n + 1))

    # Set the variable and its RMK
    data[var] = values
    data[f"{var}RMK"] = rmk_codes

    df = pd.DataFrame(data)

    # Create datetime index
    df["datetime"] = pd.to_datetime(
        df["YYYY"].astype(str)
        + "-"
        + df["MM"].astype(str).str.zfill(2)
        + "-"
        + df["DD"].astype(str).str.zfill(2)
        + " "
        + (df["HH"] % 24).astype(str).str.zfill(2)
        + ":00:00"
    )
    df = df.set_index("datetime")
    df.index.name = "time"

    return df


class TestRMKRulesDefinition:
    """Test that RMK rules are correctly defined."""

    def test_precip_rules_have_zero_rmk_codes(self):
        """Precipitation should have RMK 2,6 as zero codes."""
        assert RMK_RULES["precip"]["zero"] == {2, 6}
        assert RMK_RULES["precip"]["missing"] == {0, 1}

    def test_solar_rules_have_zero_rmk_codes(self):
        """Solar variables should have RMK 2 as zero code."""
        assert RMK_RULES["solar"]["zero"] == {2}
        assert RMK_RULES["solar"]["missing"] == {0, 1}

    def test_default_rules_no_zero_codes(self):
        """Default variables should have no zero codes."""
        assert RMK_RULES["default"]["zero"] == set()
        assert RMK_RULES["default"]["missing"] == {0, 1, 2}

    def test_kous_uses_precip_rules(self):
        """kous should use precip rules."""
        assert VAR_RMK_TYPE["kous"] == "precip"

    def test_slht_uses_solar_rules(self):
        """slht should use solar rules."""
        assert VAR_RMK_TYPE["slht"] == "solar"

    def test_kion_uses_default_rules(self):
        """kion should use default rules."""
        assert VAR_RMK_TYPE["kion"] == "default"


class TestApplyRMKMaskPrecipitation:
    """Test RMK masking for precipitation (kous)."""

    @pytest.fixture
    def reader(self, tmp_path):
        """Create a GWOReader with a dummy directory."""
        dummy_dir = tmp_path / "gwo"
        dummy_dir.mkdir()
        return GWOReader(dummy_dir)

    def test_kous_rmk6_becomes_zero(self, reader):
        """RMK 6 (no phenomenon) should become 0.0, not NaN."""
        df = make_test_df("kous", [100, 200, 300], [8, 6, 8])
        result = reader.apply_rmk_mask(df, "kous")

        assert result.iloc[0] == 100.0  # RMK 8: normal value
        assert result.iloc[1] == 0.0  # RMK 6: no precipitation = 0
        assert result.iloc[2] == 300.0  # RMK 8: normal value

    def test_kous_rmk2_becomes_zero(self, reader):
        """RMK 2 (not observed) should become 0.0 for precipitation."""
        df = make_test_df("kous", [100, 200, 300], [8, 2, 8])
        result = reader.apply_rmk_mask(df, "kous")

        assert result.iloc[0] == 100.0  # RMK 8: normal value
        assert result.iloc[1] == 0.0  # RMK 2: not observed = 0 for precip
        assert result.iloc[2] == 300.0  # RMK 8: normal value

    def test_kous_rmk0_becomes_nan(self, reader):
        """RMK 0 (observation not created) should become NaN."""
        df = make_test_df("kous", [100, 200, 300], [8, 0, 8])
        result = reader.apply_rmk_mask(df, "kous")

        assert result.iloc[0] == 100.0
        assert np.isnan(result.iloc[1])  # RMK 0: missing
        assert result.iloc[2] == 300.0

    def test_kous_rmk1_becomes_nan(self, reader):
        """RMK 1 (missing) should become NaN."""
        df = make_test_df("kous", [100, 200, 300], [8, 1, 8])
        result = reader.apply_rmk_mask(df, "kous")

        assert result.iloc[0] == 100.0
        assert np.isnan(result.iloc[1])  # RMK 1: missing
        assert result.iloc[2] == 300.0

    def test_kous_mixed_rmk_codes(self, reader):
        """Test mixed RMK codes for precipitation."""
        df = make_test_df(
            "kous",
            [100, 200, 300, 400, 500],
            [8, 6, 2, 0, 1],
        )
        result = reader.apply_rmk_mask(df, "kous")

        assert result.iloc[0] == 100.0  # RMK 8: normal
        assert result.iloc[1] == 0.0  # RMK 6: no phenomenon
        assert result.iloc[2] == 0.0  # RMK 2: not observed
        assert np.isnan(result.iloc[3])  # RMK 0: not created
        assert np.isnan(result.iloc[4])  # RMK 1: missing


class TestApplyRMKMaskSolar:
    """Test RMK masking for solar radiation (slht)."""

    @pytest.fixture
    def reader(self, tmp_path):
        """Create a GWOReader with a dummy directory."""
        dummy_dir = tmp_path / "gwo"
        dummy_dir.mkdir()
        return GWOReader(dummy_dir)

    def test_slht_rmk2_becomes_zero(self, reader):
        """RMK 2 (nighttime) should become 0.0 for solar radiation."""
        df = make_test_df("slht", [100, 200, 300], [8, 2, 8])
        result = reader.apply_rmk_mask(df, "slht")

        assert result.iloc[0] == 100.0  # RMK 8: normal value
        assert result.iloc[1] == 0.0  # RMK 2: nighttime = 0
        assert result.iloc[2] == 300.0  # RMK 8: normal value

    def test_slht_rmk0_becomes_nan(self, reader):
        """RMK 0 should become NaN for solar radiation."""
        df = make_test_df("slht", [100, 200, 300], [8, 0, 8])
        result = reader.apply_rmk_mask(df, "slht")

        assert result.iloc[0] == 100.0
        assert np.isnan(result.iloc[1])  # RMK 0: missing
        assert result.iloc[2] == 300.0


class TestApplyRMKMaskDefault:
    """Test RMK masking for default variables (e.g., kion)."""

    @pytest.fixture
    def reader(self, tmp_path):
        """Create a GWOReader with a dummy directory."""
        dummy_dir = tmp_path / "gwo"
        dummy_dir.mkdir()
        return GWOReader(dummy_dir)

    def test_kion_rmk2_becomes_nan(self, reader):
        """RMK 2 should become NaN for temperature (default rules)."""
        df = make_test_df("kion", [100, 200, 300], [8, 2, 8])
        result = reader.apply_rmk_mask(df, "kion")

        assert result.iloc[0] == 100.0  # RMK 8: normal value
        assert np.isnan(result.iloc[1])  # RMK 2: missing for default
        assert result.iloc[2] == 300.0  # RMK 8: normal value

    def test_kion_rmk6_kept_as_value(self, reader):
        """RMK 6 should keep the value for temperature (not a zero code)."""
        df = make_test_df("kion", [100, 200, 300], [8, 6, 8])
        result = reader.apply_rmk_mask(df, "kion")

        assert result.iloc[0] == 100.0
        assert result.iloc[1] == 200.0  # RMK 6: valid for default vars
        assert result.iloc[2] == 300.0


class TestDeprecationWarning:
    """Test that using is_solar parameter raises deprecation warning."""

    @pytest.fixture
    def reader(self, tmp_path):
        """Create a GWOReader with a dummy directory."""
        dummy_dir = tmp_path / "gwo"
        dummy_dir.mkdir()
        return GWOReader(dummy_dir)

    def test_is_solar_param_raises_warning(self, reader):
        """Using is_solar parameter should raise DeprecationWarning."""
        df = make_test_df("slht", [100, 200], [8, 8])

        with pytest.warns(DeprecationWarning, match="is_solar parameter is deprecated"):
            reader.apply_rmk_mask(df, "slht", is_solar=True)

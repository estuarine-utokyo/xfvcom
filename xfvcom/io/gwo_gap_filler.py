# -*- coding: utf-8 -*-
"""
Gap filling module for GWO meteorological data.

This module provides comprehensive gap filling for GWO-AMD data using
a 4-step process:
1. Temporal interpolation (short-term gaps)
2. Boundary interpolation (start/end gaps using adjacent year data)
3. Fallback station interpolation (long-term gaps)
4. Solar radiation estimation (slht only, using pvlib + cloud cover)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from .gwo_correlations import (
    STATION_COORDS,
    StationCorrelations,
    get_station_coordinates,
)
from .gwo_reader import RMK_RULES, VAR_RMK_TYPE

if TYPE_CHECKING:
    from .gwo_reader import GWOReader

# Variables that should be gap-filled
FILLABLE_VARIABLES = ["kion", "rhum", "shpa", "muki", "sped", "clod", "slht", "kous"]

# Annual mean cloud cover by station (fallback when all else fails)
# Values are 0-1 fraction
ANNUAL_MEAN_CLOUD: dict[str, float] = {
    "Tokyo": 0.55,
    "Chiba": 0.53,
    "Yokohama": 0.54,
    "Tateyama": 0.52,
}


@dataclass
class GapInfo:
    """Information about a single gap in the data."""

    start: datetime
    end: datetime
    duration_hours: int


@dataclass
class GapFillResult:
    """Result of gap filling for a single variable."""

    variable: str
    station: str
    total_gaps: int
    total_gap_hours: int
    step1_filled: int  # Temporal interpolation
    step2_filled: int  # Boundary interpolation
    step3_filled: int  # Fallback station
    step3_station: str | None  # Which fallback station was used
    step4_filled: int  # Solar estimation (slht only)
    annual_mean_filled: int  # Last resort for cloud cover
    remaining: int
    gap_details: list[GapInfo] = field(default_factory=list)

    @property
    def fill_rate(self) -> float:
        """Calculate fill rate as percentage."""
        if self.total_gaps == 0:
            return 100.0
        filled = (
            self.step1_filled
            + self.step2_filled
            + self.step3_filled
            + self.step4_filled
            + self.annual_mean_filled
        )
        return 100.0 * filled / self.total_gaps


@dataclass
class MissingValueInfo:
    """Information about missing values for a variable (when gap filling is disabled)."""

    variable: str
    station: str
    missing: int
    total: int
    rate: float  # percentage
    max_gap_hours: int
    first_gap_start: datetime | None
    gap_details: list[GapInfo] = field(default_factory=list)


class GapFiller:
    """
    Comprehensive gap filling for GWO meteorological data.

    Implements a 4-step gap filling process:
    1. Temporal interpolation for gaps ≤ max_gap_hours
    2. Boundary interpolation using adjacent year data
    3. Fallback station interpolation with correlation conversion
    4. Solar radiation estimation using pvlib + cloud cover model

    Parameters
    ----------
    gwo_reader : GWOReader
        GWO data reader instance
    fallback_stations : dict[str, list[str]] | None
        Mapping of primary station to fallback station list
        Example: {"Chiba": ["Tokyo", "Yokohama", "Tateyama"]}
    correlations : StationCorrelations | None
        Pre-computed station correlations
    max_gap_hours : int
        Maximum gap for temporal interpolation (default: 6)
    solar_model : str
        Model for solar radiation estimation: "empirical", "pvlib-kasten", "pvlib-larson"
    extend_boundary : bool
        Whether to use prev/next year data for boundary gaps (default: True)
    """

    def __init__(
        self,
        gwo_reader: "GWOReader",
        fallback_stations: dict[str, list[str]] | None = None,
        correlations: StationCorrelations | None = None,
        max_gap_hours: int = 6,
        solar_model: Literal["empirical", "pvlib-kasten", "pvlib-larson"] = "empirical",
        extend_boundary: bool = True,
    ) -> None:
        self.reader = gwo_reader
        self.fallback_stations = fallback_stations or {}
        self.correlations = correlations
        self.max_gap_hours = max_gap_hours
        self.solar_model = solar_model
        self.extend_boundary = extend_boundary
        self._solar_estimator: object | None = None

    def fill_all(
        self,
        df: pd.DataFrame,
        station: str,
        start: datetime,
        end: datetime,
    ) -> tuple[pd.DataFrame, list[GapFillResult]]:
        """
        Apply all gap filling steps to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with GWO columns (raw values, not converted)
        station : str
            Primary station name
        start : datetime
            Start of requested period
        end : datetime
            End of requested period

        Returns
        -------
        tuple[pd.DataFrame, list[GapFillResult]]
            Filled DataFrame and list of results per variable
        """
        df = df.copy()
        results: list[GapFillResult] = []

        for var in FILLABLE_VARIABLES:
            if var not in df.columns:
                continue

            result = self._fill_variable(df, var, station, start, end)
            results.append(result)

        return df, results

    def analyze_missing(
        self,
        df: pd.DataFrame,
        station: str,
    ) -> list[MissingValueInfo]:
        """
        Analyze missing values without filling (for reporting).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with GWO columns
        station : str
            Station name

        Returns
        -------
        list[MissingValueInfo]
            Information about missing values for each variable
        """
        results: list[MissingValueInfo] = []

        for var in FILLABLE_VARIABLES:
            if var not in df.columns:
                continue

            rmk_col = f"{var}RMK"
            if rmk_col not in df.columns:
                continue

            # Count missing (RMK 0, 1, 2 except for solar where 2 is valid)
            if var in ("slht", "lght"):
                missing_rmk = {0, 1}
            else:
                missing_rmk = {0, 1, 2}

            is_missing = df[rmk_col].isin(missing_rmk)
            missing_count = int(is_missing.sum().item())  # type: ignore[union-attr]
            total_count = len(df)

            # Find gaps
            gap_details = self._find_gaps(is_missing)

            max_gap = max((g.duration_hours for g in gap_details), default=0)
            first_gap = gap_details[0].start if gap_details else None

            results.append(
                MissingValueInfo(
                    variable=var,
                    station=station,
                    missing=missing_count,
                    total=total_count,
                    rate=100.0 * missing_count / total_count if total_count > 0 else 0,
                    max_gap_hours=max_gap,
                    first_gap_start=first_gap,
                    gap_details=gap_details,
                )
            )

        return results

    def _find_gaps(self, is_missing: pd.Series) -> list[GapInfo]:
        """Find contiguous gaps in the data."""
        gaps: list[GapInfo] = []

        if not is_missing.any():
            return gaps

        # Find gap boundaries
        missing_diff = is_missing.astype(int).diff()
        gap_starts = is_missing.index[
            (missing_diff == 1) | (is_missing & (missing_diff.isna()))
        ]
        gap_ends = is_missing.index[missing_diff == -1]

        # Handle edge cases
        if is_missing.iloc[-1]:
            gap_ends = gap_ends.append(pd.Index([is_missing.index[-1]]))

        for i, gap_start in enumerate(gap_starts):
            if i < len(gap_ends):
                gap_end = gap_ends[i]
            else:
                gap_end = is_missing.index[-1]

            # Calculate duration
            duration = int((gap_end - gap_start).total_seconds() / 3600) + 1

            gaps.append(
                GapInfo(
                    start=gap_start.to_pydatetime(),
                    end=gap_end.to_pydatetime(),
                    duration_hours=duration,
                )
            )

        return gaps

    def _fill_variable(
        self,
        df: pd.DataFrame,
        var: str,
        station: str,
        start: datetime,
        end: datetime,
    ) -> GapFillResult:
        """Fill gaps for a single variable using all applicable steps."""
        rmk_col = f"{var}RMK"

        # Handle non-numeric strings like "静穏" (calm) which means zero wind
        # Replace "静穏" with 0 before converting to numeric
        if df[var].dtype == object:
            df[var] = df[var].replace("静穏", 0)
        df[var] = pd.to_numeric(df[var], errors="coerce")

        # Apply RMK masking using variable-specific rules
        rule_type = VAR_RMK_TYPE.get(var, "default")
        rules = RMK_RULES[rule_type]

        # First, set "zero" RMK codes to 0.0 (e.g., RMK 6 = no precipitation)
        if rules["zero"] and rmk_col in df.columns:
            zero_mask = df[rmk_col].isin(rules["zero"])
            df.loc[zero_mask, var] = 0.0

        # Then, set "missing" RMK codes to NaN
        if rmk_col in df.columns:
            is_missing = df[rmk_col].isin(rules["missing"])
            df.loc[is_missing, var] = np.nan

        # Track which rows were originally missing (for updating RMK later)
        originally_missing = df[var].isna().copy()

        result = GapFillResult(
            variable=var,
            station=station,
            total_gaps=int(df[var].isna().values.sum()),
            total_gap_hours=int(df[var].isna().values.sum()),  # Each row is 1 hour
            step1_filled=0,
            step2_filled=0,
            step3_filled=0,
            step3_station=None,
            step4_filled=0,
            annual_mean_filled=0,
            remaining=0,
            gap_details=self._find_gaps(df[var].isna()),
        )

        if result.total_gaps == 0:
            return result

        # Step 1: Temporal interpolation
        before = int(df[var].isna().values.sum())
        df[var] = df[var].interpolate(method="time", limit=self.max_gap_hours)
        result.step1_filled = before - int(df[var].isna().values.sum())

        # Step 2: Boundary interpolation
        if self.extend_boundary and df[var].isna().any():
            result.step2_filled = self._fill_boundary(df, var, station, start, end)

        # Step 3: Fallback station interpolation
        if df[var].isna().any() and station in self.fallback_stations:
            filled, used_station = self._fill_from_fallback(
                df, var, station, start, end
            )
            result.step3_filled = filled
            result.step3_station = used_station

        # Step 4: Solar estimation (slht only)
        if var == "slht" and df[var].isna().any():
            result.step4_filled = self._fill_solar(df, station, start, end)

        # Last resort for cloud cover: annual mean
        if var == "clod" and df[var].isna().any():
            annual_mean = self._get_annual_mean_cloud(station)
            before = int(df[var].isna().values.sum())
            df[var] = df[var].fillna(annual_mean * 10.0)  # Convert back to 0-10
            result.annual_mean_filled = before - int(df[var].isna().values.sum())

        result.remaining = int(df[var].isna().values.sum())

        # Update RMK codes for filled values to mark as valid (8 = normal value)
        # This prevents convert_units from masking these values again
        if rmk_col in df.columns:
            filled_mask = originally_missing & ~df[var].isna()
            df.loc[filled_mask, rmk_col] = 8

        return result

    def _fill_boundary(
        self,
        df: pd.DataFrame,
        var: str,
        station: str,
        start: datetime,
        end: datetime,
    ) -> int:
        """Step 2: Fill gaps at start/end using prev/next year data."""
        filled = 0

        # Check for gaps at start
        first_valid = df[var].first_valid_index()
        if first_valid is not None and first_valid > df.index[0]:
            # Try loading previous year
            prev_year = start.year - 1
            try:
                prev_df = self.reader.load_station_year(station, prev_year)
                # Get end of previous year data that overlaps
                overlap_start = df.index[0] - timedelta(hours=self.max_gap_hours)
                overlap_end = first_valid
                prev_slice = prev_df.loc[
                    (prev_df.index >= overlap_start) & (prev_df.index < overlap_end)
                ]

                if len(prev_slice) > 0:
                    # Merge and interpolate
                    # Handle "静穏" (calm) as 0, then convert to numeric
                    prev_var = prev_slice[var].replace("静穏", 0)
                    prev_var = pd.to_numeric(prev_var, errors="coerce")
                    combined = pd.concat([prev_var.to_frame(var), df[[var]]])
                    combined = combined.sort_index()
                    combined = combined[~combined.index.duplicated(keep="first")]
                    combined[var] = combined[var].interpolate(
                        method="time", limit=self.max_gap_hours
                    )

                    # Copy back to df
                    before = int(df[var].isna().values.sum())
                    for idx in df.index:
                        if pd.isna(df.loc[idx, var]) and idx in combined.index:
                            val = combined.loc[idx, var]
                            if not pd.isna(val):
                                df.loc[idx, var] = val
                    filled += before - int(df[var].isna().values.sum())

            except FileNotFoundError:
                pass

        # Check for gaps at end
        last_valid = df[var].last_valid_index()
        if last_valid is not None and last_valid < df.index[-1]:
            # Try loading next year
            next_year = end.year + 1
            try:
                next_df = self.reader.load_station_year(station, next_year)
                # Get start of next year data that overlaps
                next_slice = next_df.loc[
                    (next_df.index > last_valid)
                    & (
                        next_df.index
                        <= df.index[-1] + timedelta(hours=self.max_gap_hours)
                    )
                ]

                if len(next_slice) > 0:
                    # Merge and interpolate
                    # Handle "静穏" (calm) as 0, then convert to numeric
                    next_var = next_slice[var].replace("静穏", 0)
                    next_var = pd.to_numeric(next_var, errors="coerce")
                    combined = pd.concat([df[[var]], next_var.to_frame(var)])
                    combined = combined.sort_index()
                    combined = combined[~combined.index.duplicated(keep="first")]
                    combined[var] = combined[var].interpolate(
                        method="time", limit=self.max_gap_hours
                    )

                    # Copy back to df
                    before = int(df[var].isna().values.sum())
                    for idx in df.index:
                        if pd.isna(df.loc[idx, var]) and idx in combined.index:
                            val = combined.loc[idx, var]
                            if not pd.isna(val):
                                df.loc[idx, var] = val
                    filled += before - int(df[var].isna().values.sum())

            except FileNotFoundError:
                pass

        return filled

    def _fill_from_fallback(
        self,
        df: pd.DataFrame,
        var: str,
        primary_station: str,
        start: datetime,
        end: datetime,
    ) -> tuple[int, str | None]:
        """Step 3: Fill from fallback stations with correlation conversion."""
        filled = 0
        used_station: str | None = None

        for fallback in self.fallback_stations.get(primary_station, []):
            if not df[var].isna().any():
                break

            try:
                fallback_df = self.reader.load_range(fallback, start, end)

                if var not in fallback_df.columns:
                    continue

                # Get RMK-filtered values from fallback using variable-specific rules
                rmk_col = f"{var}RMK"
                # Handle "静穏" (calm) as 0, then convert to numeric
                fallback_values = fallback_df[var].replace("静穏", 0)
                fallback_values = pd.to_numeric(fallback_values, errors="coerce").copy()
                if rmk_col in fallback_df.columns:
                    rule_type = VAR_RMK_TYPE.get(var, "default")
                    rules = RMK_RULES[rule_type]
                    # Apply zero RMK codes (e.g., RMK 6 = no precipitation = 0)
                    if rules["zero"]:
                        zero_mask = fallback_df[rmk_col].isin(rules["zero"])
                        fallback_values[zero_mask] = 0.0
                    # Apply missing RMK codes
                    missing_mask = fallback_df[rmk_col].isin(rules["missing"])
                    fallback_values[missing_mask] = np.nan

                # Align indices
                fallback_values = fallback_values.reindex(df.index)

                # Apply correlation-based conversion
                if self.correlations:
                    converted = self.correlations.convert(
                        primary_station,
                        fallback,
                        var,
                        fallback_values,
                        use_converted_units=False,  # Raw GWO units
                    )
                else:
                    converted = fallback_values  # No conversion if no correlation data

                # Fill where primary is NaN and fallback has data
                mask = df[var].isna() & ~converted.isna()
                before = int(df[var].isna().values.sum())
                df.loc[mask, var] = converted[mask]
                this_filled = before - int(df[var].isna().values.sum())
                filled += this_filled

                if this_filled > 0 and used_station is None:
                    used_station = fallback

            except FileNotFoundError:
                continue

        return filled, used_station

    def _fill_solar(
        self,
        df: pd.DataFrame,
        station: str,
        start: datetime,
        end: datetime,
    ) -> int:
        """Step 4: Estimate solar radiation using pvlib + cloud model."""
        try:
            from .solar_estimation import SolarEstimator
        except ImportError:
            # pvlib not available
            return 0

        # Get station coordinates
        try:
            coords = get_station_coordinates(station)
        except ValueError:
            # Unknown station, use Tokyo defaults
            coords = STATION_COORDS.get(
                "Tokyo", {"latitude": 35.41, "longitude": 139.4548, "altitude": 6.5}
            )

        if self._solar_estimator is None:
            try:
                self._solar_estimator = SolarEstimator(
                    model=self.solar_model,
                    latitude=coords["latitude"],
                    longitude=coords["longitude"],
                    altitude=coords["altitude"],
                )
            except ImportError:
                return 0

        # Get cloud cover (try fallback stations if primary missing)
        cloud = self._get_cloud_cover(df, station, start, end)

        # Convert cloud to 0-1 fraction
        cloud_frac: pd.Series = cloud / 10.0

        before = int(df["slht"].isna().values.sum())

        # Estimate GHI for missing values
        missing_mask = df["slht"].isna()
        if not missing_mask.any():
            return 0

        missing_times = df.index[missing_mask]
        try:
            estimated_ghi = self._solar_estimator.estimate(missing_times, cloud_frac)  # type: ignore[union-attr]

            # Convert W/m² back to GWO units (0.01 MJ/m²/h)
            # W/m² * 3600 s / 1e6 * 100 = W/m² * 0.36
            estimated_raw = estimated_ghi * 3600.0 / 1e6 * 100.0

            df.loc[missing_mask, "slht"] = estimated_raw.values
        except Exception:
            # If estimation fails, leave as NaN
            pass

        return before - int(df["slht"].isna().values.sum())

    def _get_cloud_cover(
        self,
        df: pd.DataFrame,
        station: str,
        start: datetime,
        end: datetime,
    ) -> pd.Series:
        """Get cloud cover, trying fallback stations if primary is missing."""
        cloud = df["clod"].copy() if "clod" in df.columns else pd.Series(index=df.index)

        # If cloud has missing values, try fallback stations
        if cloud.isna().any() and station in self.fallback_stations:
            for fallback in self.fallback_stations.get(station, []):
                if not cloud.isna().any():
                    break
                try:
                    fallback_df = self.reader.load_range(fallback, start, end)
                    if "clod" in fallback_df.columns:
                        fallback_cloud = fallback_df["clod"].reindex(df.index)
                        cloud = cloud.fillna(fallback_cloud)
                except FileNotFoundError:
                    continue

        # Last resort: use annual mean
        if cloud.isna().any():
            annual_mean = self._get_annual_mean_cloud(station) * 10.0  # 0-10 scale
            cloud = cloud.fillna(annual_mean)

        return cloud

    def _get_annual_mean_cloud(self, station: str) -> float:
        """Get annual mean cloud cover for a station (0-1 scale)."""
        return ANNUAL_MEAN_CLOUD.get(station, 0.55)


def print_missing_value_report(results: list[MissingValueInfo]) -> None:
    """Print missing value report (when --fill-gaps is not used)."""
    print("=" * 80)
    print("Missing Value Report")
    print("=" * 80)
    print(
        f"{'Variable':<12} {'Station':<10} {'Total':>8} {'Rate':>8} "
        f"{'Max Gap':>10} {'First Gap Start':<20}"
    )
    print("-" * 80)

    total_missing = 0
    for r in results:
        if r.missing > 0:
            first_gap_str = (
                r.first_gap_start.strftime("%Y-%m-%d %H:%M")
                if r.first_gap_start
                else "N/A"
            )
            print(
                f"{r.variable:<12} {r.station:<10} {r.missing:>8} "
                f"{r.rate:>7.2f}% {r.max_gap_hours:>9}h {first_gap_str:<20}"
            )
            total_missing += r.missing

    if total_missing > 0:
        print()
        print(f"[WARNING] {total_missing} missing values detected.")
        print("          Use --fill-gaps to enable gap filling.")
        print("          Output NetCDF will contain NaN values.")
    else:
        print()
        print("[OK] No missing values detected.")


def print_gap_fill_report(results: list[GapFillResult]) -> None:
    """Print detailed gap filling report."""
    print("=" * 80)
    print("Gap Filling Report")
    print("=" * 80)

    for r in results:
        if r.total_gaps == 0:
            continue

        print(f"\n[{r.variable}] {r.station} - {r.total_gaps} gaps detected")

        if r.step1_filled > 0:
            print(f"  +-- Step 1 (Temporal): {r.step1_filled} filled")

        if r.step2_filled > 0:
            print(f"  +-- Step 2 (Boundary): {r.step2_filled} filled")

        if r.step3_filled > 0:
            print(f"  +-- Step 3 (Fallback {r.step3_station}): {r.step3_filled} filled")

        if r.step4_filled > 0:
            print(f"  +-- Step 4 (Solar model): {r.step4_filled} filled")

        if r.annual_mean_filled > 0:
            print(f"  +-- Annual mean: {r.annual_mean_filled} filled")

        print(f"  `-- Remaining: {r.remaining}")

    # Summary table
    print("\n" + "=" * 80)
    print("Fill Summary")
    print("=" * 80)
    print(
        f"{'Variable':<10} {'Total':>8} {'Step1':>8} {'Step2':>8} "
        f"{'Step3':>8} {'Step4':>8} {'Remain':>8} {'Rate':>8}"
    )
    print("-" * 80)

    total = sum(r.total_gaps for r in results)
    s1 = sum(r.step1_filled for r in results)
    s2 = sum(r.step2_filled for r in results)
    s3 = sum(r.step3_filled for r in results)
    s4 = sum(r.step4_filled for r in results)
    remain = sum(r.remaining for r in results)

    for r in results:
        if r.total_gaps > 0:
            print(
                f"{r.variable:<10} {r.total_gaps:>8} {r.step1_filled:>8} "
                f"{r.step2_filled:>8} {r.step3_filled:>8} {r.step4_filled:>8} "
                f"{r.remaining:>8} {r.fill_rate:>7.1f}%"
            )

    print("-" * 80)
    fill_rate = 100.0 * (total - remain) / total if total > 0 else 100.0
    print(
        f"{'TOTAL':<10} {total:>8} {s1:>8} {s2:>8} {s3:>8} {s4:>8} "
        f"{remain:>8} {fill_rate:>7.1f}%"
    )

    if remain == 0:
        print("\n[OK] All gaps filled successfully.")
    else:
        print(f"\n[WARNING] {remain} gaps could not be filled.")

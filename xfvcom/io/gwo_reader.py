# -*- coding: utf-8 -*-
"""
GWO-AMD meteorological data reader for FVCOM forcing generation.

This module reads JMA Ground Weather Observation (GWO) hourly CSV files
and converts them to FVCOM-compatible meteorological variables.

Column Format (33 columns, no header):
    KanID, Kname, KanID_1, YYYY, MM, DD, HH,
    lhpa, lhpaRMK, shpa, shpaRMK, kion, kionRMK,
    stem, stemRMK, rhum, rhumRMK, muki, mukiRMK, sped, spedRMK,
    clod, clodRMK, tnki, tnkiRMK, humd, humdRMK,
    lght, lghtRMK, slht, slhtRMK, kous, kousRMK

RMK Codes:
    0 - Observation value not created
    1 - Missing
    2 - Not observed
    3-9 - Various quality flags (generally usable)
    8 - Normal observation value
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Column names for GWO hourly CSV (33 columns)
GWO_COLUMNS = [
    "KanID",
    "Kname",
    "KanID_1",
    "YYYY",
    "MM",
    "DD",
    "HH",
    "lhpa",
    "lhpaRMK",
    "shpa",
    "shpaRMK",
    "kion",
    "kionRMK",
    "stem",
    "stemRMK",
    "rhum",
    "rhumRMK",
    "muki",
    "mukiRMK",
    "sped",
    "spedRMK",
    "clod",
    "clodRMK",
    "tnki",
    "tnkiRMK",
    "humd",
    "humdRMK",
    "lght",
    "lghtRMK",
    "slht",
    "slhtRMK",
    "kous",
    "kousRMK",
]

# =============================================================================
# RMK (Remark) Code Handling Rules
# =============================================================================
# GWO-AMD RMK codes indicate data quality:
#   0 - Observation value not created → NaN
#   1 - Missing observation → NaN
#   2 - Not observed (e.g., nighttime for solar) → depends on variable
#   6 - No phenomenon (e.g., no precipitation) → 0.0
#   8 - Normal observation value → use raw value
#
# Reference: https://github.com/jsasaki-utokyo/GWO-AMD
# =============================================================================

# Variable-specific RMK handling rules
# Each rule defines:
#   - "missing": RMK codes that indicate truly missing data → NaN
#   - "zero": RMK codes that indicate valid zero values → 0.0
RMK_RULES: dict[str, dict[str, set[int]]] = {
    # Default: most variables (temperature, humidity, pressure, wind)
    # RMK 0,1,2 are all missing
    "default": {
        "missing": {0, 1, 2},
        "zero": set(),
    },
    # Solar radiation and sunshine duration
    # RMK 2 (nighttime/not observed) is valid 0, not missing
    "solar": {
        "missing": {0, 1},
        "zero": {2},
    },
    # Precipitation
    # RMK 2 (not observed) and RMK 6 (no phenomenon) mean "no rain" = 0
    "precip": {
        "missing": {0, 1},
        "zero": {2, 6},
    },
    # Cloud cover (3-hourly observation)
    # RMK 2 should be interpolated, so treat as missing
    "cloud": {
        "missing": {0, 1, 2},
        "zero": set(),
    },
}

# Mapping from variable names to RMK rule types
VAR_RMK_TYPE: dict[str, str] = {
    "kion": "default",  # Air temperature
    "rhum": "default",  # Relative humidity
    "shpa": "default",  # Sea-level pressure
    "lhpa": "default",  # Station pressure
    "muki": "default",  # Wind direction
    "sped": "default",  # Wind speed
    "stem": "default",  # Dew point temperature
    "humd": "default",  # Vapor pressure
    "tnki": "default",  # Weather code
    "slht": "solar",  # Solar radiation
    "lght": "solar",  # Sunshine duration
    "kous": "precip",  # Precipitation
    "clod": "cloud",  # Cloud cover
}

# Legacy constants for backward compatibility (deprecated)
MISSING_RMK_DEFAULT = {0, 1, 2}
MISSING_RMK_SOLAR = {0, 1}


class GWOReader:
    """
    Read GWO-AMD format CSV files and convert to FVCOM variables.

    Parameters
    ----------
    base_dir : Path or str
        Base directory containing station subdirectories.
        Expected structure: base_dir/{Station}/{Station}{Year}.csv

    Examples
    --------
    >>> reader = GWOReader("/path/to/GWO/Hourly")
    >>> df = reader.load_range("Chiba", datetime(2020, 1, 1), datetime(2020, 12, 31, 23))
    >>> u, v = reader.convert_wind(df, wind_factor=1.8)
    """

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"GWO base directory not found: {self.base_dir}")

    def load_station_year(self, station: str, year: int) -> pd.DataFrame:
        """
        Load one year of data for a station.

        Parameters
        ----------
        station : str
            Station name (e.g., "Tokyo", "Chiba")
        year : int
            Year to load

        Returns
        -------
        pd.DataFrame
            DataFrame with datetime index (JST) and GWO columns
        """
        path = self.base_dir / station / f"{station}{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"GWO data file not found: {path}")

        df = pd.read_csv(path, header=None, names=GWO_COLUMNS)

        # Create datetime index from YYYY, MM, DD, HH columns
        # GWO uses hours 1-24, where 24 means 00:00 of the next day
        # But pandas expects 0-23, so we handle this
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
        # Adjust for hour 24 (becomes 00:00 of next day)
        mask_hour24 = df["HH"] == 24
        df.loc[mask_hour24, "datetime"] = df.loc[
            mask_hour24, "datetime"
        ] + pd.Timedelta(days=1)

        df = df.set_index("datetime")
        df.index.name = "time"

        return df

    def load_range(
        self,
        station: str,
        start: datetime,
        end: datetime,
        *,
        input_tz: str = "Asia/Tokyo",
    ) -> pd.DataFrame:
        """
        Load data spanning a date range (possibly multiple years).

        Parameters
        ----------
        station : str
            Station name (e.g., "Tokyo", "Chiba")
        start : datetime
            Start datetime (inclusive)
        end : datetime
            End datetime (inclusive)
        input_tz : str
            Timezone of input data (default: "Asia/Tokyo")

        Returns
        -------
        pd.DataFrame
            DataFrame with datetime index (UTC) and GWO columns
        """
        start_year = start.year
        end_year = end.year

        dfs = []
        for year in range(start_year, end_year + 1):
            try:
                df_year = self.load_station_year(station, year)
                dfs.append(df_year)
            except FileNotFoundError:
                # Try adjacent years for boundary cases
                continue

        if not dfs:
            raise FileNotFoundError(
                f"No GWO data found for {station} in range {start_year}-{end_year}"
            )

        df = pd.concat(dfs)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # Keep JST values as-is but make timezone-naive
        # FVCOM convention: use JST values but label as UTC
        # (This matches how the reference files are created)
        df.index = pd.DatetimeIndex(df.index)
        # Index is already naive from load_station_year, no conversion needed

        # Slice to requested range
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        # Strip timezone info if present (keep the clock time, just make naive)
        if start_ts.tzinfo is not None:
            start_ts = start_ts.tz_localize(None)
        if end_ts.tzinfo is not None:
            end_ts = end_ts.tz_localize(None)

        df = df.loc[start_ts:end_ts]

        return df

    def apply_rmk_mask(
        self, df: pd.DataFrame, var: str, *, is_solar: bool | None = None
    ) -> pd.Series:
        """
        Apply RMK code masking based on variable-specific rules.

        RMK codes are processed as follows:
        - "missing" RMK codes → NaN (truly missing data)
        - "zero" RMK codes → 0.0 (valid zero measurement)
        - Other RMK codes → use raw value

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with GWO columns
        var : str
            Variable name (e.g., "kion", "shpa", "slht", "kous")
        is_solar : bool | None
            Deprecated. RMK rules are now determined automatically by variable name.
            This parameter is ignored but kept for backward compatibility.

        Returns
        -------
        pd.Series
            Series with masked values (NaN for missing, 0.0 for zero RMK codes)
        """
        if is_solar is not None:
            import warnings

            warnings.warn(
                "is_solar parameter is deprecated. "
                "RMK rules are now determined automatically by variable name.",
                DeprecationWarning,
                stacklevel=2,
            )

        rmk_col = f"{var}RMK"
        values = df[var].copy().astype(float)

        if rmk_col not in df.columns:
            return values

        # Get RMK rules for this variable
        rule_type = VAR_RMK_TYPE.get(var, "default")
        rules = RMK_RULES[rule_type]

        # Apply zero RMK codes first (set to 0.0)
        if rules["zero"]:
            zero_mask = df[rmk_col].isin(rules["zero"])
            values[zero_mask] = 0.0

        # Apply missing RMK codes (set to NaN)
        missing_mask = df[rmk_col].isin(rules["missing"])
        values[missing_mask] = np.nan

        return values

    def interpolate_gaps(self, series: pd.Series, max_gap_hours: int = 6) -> pd.Series:
        """
        Interpolate NaN values with maximum gap limit.

        Parameters
        ----------
        series : pd.Series
            Time series with potential NaN values
        max_gap_hours : int
            Maximum consecutive hours to interpolate

        Returns
        -------
        pd.Series
            Interpolated series
        """
        return series.interpolate(method="time", limit=max_gap_hours)

    def convert_units(
        self,
        df: pd.DataFrame,
        *,
        max_gap_hours: int = 6,
        fill_gaps: bool = True,
    ) -> dict[str, pd.Series]:
        """
        Convert GWO raw values to physical units.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with GWO columns
        max_gap_hours : int
            Maximum gap to interpolate
        fill_gaps : bool
            If True, apply temporal interpolation to fill short gaps.
            If False, leave NaN values for gap filler to handle.

        Returns
        -------
        dict[str, pd.Series]
            Dictionary with converted variables:
            - shpa: Sea-level pressure [hPa]
            - kion: Air temperature [°C]
            - rhum: Relative humidity [%]
            - muki: Wind direction angle [degrees, meteorological convention]
            - sped: Wind speed [m/s]
            - clod: Cloud cover [0-1]
            - slht: Short-wave radiation [W/m²]
            - kous: Precipitation rate [m/s]
        """
        result = {}

        def maybe_interp(series: pd.Series) -> pd.Series:
            """Apply interpolation if fill_gaps is True."""
            if fill_gaps:
                return self.interpolate_gaps(series, max_gap_hours)
            return series

        # Sea-level pressure: 0.1 hPa -> hPa
        shpa = self.apply_rmk_mask(df, "shpa")
        result["shpa"] = maybe_interp(shpa * 0.1)

        # Air temperature: 0.1°C -> °C
        kion = self.apply_rmk_mask(df, "kion")
        result["kion"] = maybe_interp(kion * 0.1)

        # Relative humidity: keep as % (FVCOM expects %)
        rhum = self.apply_rmk_mask(df, "rhum")
        result["rhum"] = maybe_interp(rhum)

        # Wind direction: code 0-16 -> angle
        # 0=calm/undefined, 1=NNE, 2=NE, ..., 8=S, ..., 16=N
        # Formula: angle = (-90.0 - muki * 22.5) % 360.0
        muki = self.apply_rmk_mask(df, "muki")
        muki_angle = (-90.0 - muki * 22.5) % 360.0
        result["muki"] = maybe_interp(muki_angle)

        # Wind speed: 0.1 m/s -> m/s
        sped = self.apply_rmk_mask(df, "sped")
        result["sped"] = maybe_interp(sped * 0.1)

        # Cloud cover: 0-10 -> 0-1 (3-hourly in GWO, needs interpolation)
        clod = self.apply_rmk_mask(df, "clod")
        result["clod"] = maybe_interp(clod / 10.0)

        # Short-wave radiation: 0.01 MJ/m²/h -> W/m²
        # 0.01 MJ/m²/h = 0.01 * 1e6 J / 3600 s = 2.7778 W/m²
        slht = self.apply_rmk_mask(df, "slht")
        slht_wm2: pd.Series = slht * 0.01 * 1e6 / 3600.0  # type: ignore[assignment]
        # DO NOT use fillna(0.0) - let gap filler handle daytime gaps
        # Only apply interpolation if fill_gaps is True
        if fill_gaps:
            slht_wm2 = self.interpolate_gaps(slht_wm2, max_gap_hours)
        result["slht"] = slht_wm2.clip(lower=0.0)

        # Precipitation: 0.1 mm/h -> m/s
        # 0.1 mm/h = 0.0001 m / 3600 s = 2.778e-8 m/s
        kous = self.apply_rmk_mask(df, "kous")
        kous_ms: pd.Series = kous * 0.1 * 0.001 / 3600.0  # type: ignore[assignment]
        # Use interpolation instead of fillna(0.0)
        if fill_gaps:
            kous_ms = self.interpolate_gaps(kous_ms, max_gap_hours)
        result["kous"] = kous_ms.clip(lower=0.0)

        return result

    def convert_wind(
        self,
        df: pd.DataFrame,
        *,
        wind_factor: float = 1.0,
        max_gap_hours: int = 6,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Convert wind direction and speed to u, v components.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with GWO columns
        wind_factor : float
            Multiplier for wind speed (default: 1.0)
        max_gap_hours : int
            Maximum gap to interpolate

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (u, v) wind components in m/s
            u: eastward component (positive = from west)
            v: northward component (positive = from south)
        """
        # Get raw values with RMK masking
        muki_raw = self.apply_rmk_mask(df, "muki")
        sped_raw = self.apply_rmk_mask(df, "sped")

        # Convert units: sped is 0.1 m/s -> m/s, then apply factor
        sped_ms = sped_raw * 0.1 * wind_factor

        # Direction conversion: code -> angle
        # Formula: angle = (-90.0 - muki * 22.5) % 360.0
        angle_deg = (-90.0 - muki_raw * 22.5) % 360.0
        angle_rad = np.deg2rad(angle_deg)

        # Calculate u, v components
        u = sped_ms * np.cos(angle_rad)
        v = sped_ms * np.sin(angle_rad)

        # Handle calm wind (muki=0 or speed near zero)
        calm_mask = (muki_raw == 0) | (sped_ms <= 0.3)
        u[calm_mask] = 0.0
        v[calm_mask] = 0.0

        # Interpolate gaps
        u = self.interpolate_gaps(u, max_gap_hours)
        v = self.interpolate_gaps(v, max_gap_hours)

        return u.values, v.values

    @staticmethod
    def estimate_longwave(
        temp_c: NDArray[np.float64],
        rh_pct: NDArray[np.float64],
        cloud_frac: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Estimate downward long-wave radiation using Brutsaert formula.

        Parameters
        ----------
        temp_c : np.ndarray
            Air temperature in °C
        rh_pct : np.ndarray
            Relative humidity in %
        cloud_frac : np.ndarray
            Cloud cover fraction (0-1)

        Returns
        -------
        np.ndarray
            Downward long-wave radiation in W/m²
        """
        # Convert to Kelvin
        T_K = np.asarray(temp_c, dtype=np.float64) + 273.15

        # Saturation vapor pressure (Tetens formula) [hPa]
        e_sat = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))

        # Actual vapor pressure [hPa]
        rh_frac = np.asarray(rh_pct, dtype=np.float64) / 100.0
        e = rh_frac * e_sat

        # Stefan-Boltzmann constant [W/m²/K⁴]
        sigma = 5.67e-8

        # Clear-sky emissivity (Brutsaert, 1975)
        # Avoid division by zero
        T_K_safe = np.maximum(T_K, 1.0)
        epsilon_clear = 1.24 * np.power(e / T_K_safe, 1.0 / 7.0)

        # Clear-sky downward long-wave radiation
        LW_clear = epsilon_clear * sigma * np.power(T_K, 4)

        # Cloud effect (Idso, 1981)
        cloud = np.asarray(cloud_frac, dtype=np.float64)
        LW_down = LW_clear * (1.0 + 0.17 * np.power(cloud, 2))

        # Ensure positive values
        return np.maximum(LW_down, 0.0)


class GWOForcingSource:
    """
    Forcing source that reads from GWO-AMD data.

    This class integrates with the MetNetCDFGenerator to provide
    meteorological forcing data from GWO stations.

    Parameters
    ----------
    gwo_dir : Path or str
        Base directory for GWO hourly data
    station_map : dict[str, str]
        Mapping of variable names to station names.
        Use "*" as a key for default station.
        Example: {"slht": "Tokyo", "kous": "Tokyo", "*": "Chiba"}
    wind_factor : float
        Wind speed multiplier (default: 1.8)
    max_gap_hours : int
        Maximum hours to interpolate for missing data (default: 6)
    input_tz : str
        Timezone of input data (default: "Asia/Tokyo")
    """

    # Map FVCOM variable names to GWO variable names
    _VAR_MAP = {
        "uwind": "uwind",  # computed from muki + sped
        "vwind": "vwind",  # computed from muki + sped
        "air_temp": "kion",
        "rh": "rhum",
        "prmsl": "shpa",
        "swrad": "slht",
        "lwrad": "lwrad",  # estimated
        "precip": "kous",
        "cloud": "clod",
    }

    def __init__(
        self,
        gwo_dir: Path | str,
        station_map: dict[str, str],
        *,
        wind_factor: float = 1.8,
        max_gap_hours: int = 6,
        input_tz: str = "Asia/Tokyo",
    ) -> None:
        self.reader = GWOReader(gwo_dir)
        self.station_map = station_map
        self.wind_factor = wind_factor
        self.max_gap_hours = max_gap_hours
        self.input_tz = input_tz
        self._cache: dict[str, pd.DataFrame] = {}
        self._converted: dict[str, dict[str, pd.Series]] = {}
        self._wind_cache: dict[str, tuple[NDArray, NDArray]] = {}
        self._prefilled: dict[str, pd.DataFrame] = {}  # Pre-filled data by station

    def set_prefilled_data(self, station: str, df: pd.DataFrame) -> None:
        """
        Set pre-filled data for a station.

        This allows gap-filled data to be injected instead of loading fresh
        data from files. The pre-filled data should contain raw GWO columns
        (kion, rhum, etc.) with gaps already filled.

        Parameters
        ----------
        station : str
            Station name
        df : pd.DataFrame
            Pre-filled DataFrame with raw GWO columns
        """
        self._prefilled[station] = df
        # Clear any cached data for this station to force reload from prefilled
        keys_to_remove = [k for k in self._cache if k.startswith(f"{station}_")]
        for key in keys_to_remove:
            del self._cache[key]
        keys_to_remove = [k for k in self._converted if k.startswith(f"{station}_")]
        for key in keys_to_remove:
            del self._converted[key]
        keys_to_remove = [k for k in self._wind_cache if k.startswith(f"{station}_")]
        for key in keys_to_remove:
            del self._wind_cache[key]

    def _get_station(self, var: str) -> str:
        """Get station for a variable."""
        if var in self.station_map:
            return self.station_map[var]
        if "*" in self.station_map:
            return self.station_map["*"]
        raise ValueError(f"No station mapping for variable: {var}")

    def _load_station_data(
        self, station: str, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Load and cache station data, using pre-filled data if available."""
        cache_key = f"{station}_{start}_{end}"
        if cache_key not in self._cache:
            if station in self._prefilled:
                # Use pre-filled data (already gap-filled)
                df = self._prefilled[station]
                # Slice to requested range
                start_ts = pd.Timestamp(start)
                end_ts = pd.Timestamp(end)
                if start_ts.tzinfo is not None:
                    start_ts = start_ts.tz_localize(None)
                if end_ts.tzinfo is not None:
                    end_ts = end_ts.tz_localize(None)
                self._cache[cache_key] = df.loc[start_ts:end_ts]
            else:
                # Load fresh from files
                self._cache[cache_key] = self.reader.load_range(
                    station, start, end, input_tz=self.input_tz
                )
        return self._cache[cache_key]

    def _get_converted(
        self, station: str, start: datetime, end: datetime
    ) -> dict[str, pd.Series]:
        """Get converted variables for a station."""
        cache_key = f"{station}_{start}_{end}"
        if cache_key not in self._converted:
            df = self._load_station_data(station, start, end)
            self._converted[cache_key] = self.reader.convert_units(
                df, max_gap_hours=self.max_gap_hours
            )
        return self._converted[cache_key]

    def _get_wind(
        self, station: str, start: datetime, end: datetime
    ) -> tuple[NDArray, NDArray]:
        """Get wind u, v components."""
        cache_key = f"{station}_{start}_{end}"
        if cache_key not in self._wind_cache:
            df = self._load_station_data(station, start, end)
            self._wind_cache[cache_key] = self.reader.convert_wind(
                df, wind_factor=self.wind_factor, max_gap_hours=self.max_gap_hours
            )
        return self._wind_cache[cache_key]

    def get_series(self, var_name: str, times: pd.DatetimeIndex) -> NDArray[np.float64]:
        """
        Return 1-D array of meteorological values aligned to times.

        Parameters
        ----------
        var_name : str
            Variable name (uwind, vwind, air_temp, rh, prmsl, swrad, lwrad, precip, cloud)
        times : pd.DatetimeIndex
            Target timestamps

        Returns
        -------
        np.ndarray
            Values aligned to times
        """
        # Ensure times are naive (strip timezone, keep clock time)
        # GWO data uses JST values labeled as UTC (FVCOM convention)
        if hasattr(times, "tz") and times.tz is not None:
            times = times.tz_localize(None)

        # Determine time range - convert to naive datetime
        start = times.min().to_pydatetime()
        end = times.max().to_pydatetime()
        # Ensure start/end are naive (keep clock time)
        if hasattr(start, "tzinfo") and start.tzinfo is not None:
            start = start.replace(tzinfo=None)
        if hasattr(end, "tzinfo") and end.tzinfo is not None:
            end = end.replace(tzinfo=None)

        # Determine which GWO variable and station to use
        gwo_var = self._VAR_MAP.get(var_name, var_name)

        if var_name in ("uwind", "vwind"):
            # Wind components
            station = self._get_station("sped")  # Use wind speed station
            u, v = self._get_wind(station, start, end)
            df = self._load_station_data(station, start, end)

            if var_name == "uwind":
                series = pd.Series(u, index=df.index)
            else:
                series = pd.Series(v, index=df.index)

        elif var_name == "lwrad":
            # Long-wave radiation (estimated)
            # Need temperature, humidity, and cloud cover
            station_temp = self._get_station("kion")
            station_cloud = self._get_station("clod")

            converted_temp = self._get_converted(station_temp, start, end)
            converted_cloud = self._get_converted(station_cloud, start, end)

            # Use temperature station for humidity too (usually same station)
            temp_c = converted_temp["kion"].values
            rh_pct = converted_temp["rhum"].values
            cloud_frac = converted_cloud["clod"].values

            lw = GWOReader.estimate_longwave(temp_c, rh_pct, cloud_frac)
            series = pd.Series(lw, index=converted_temp["kion"].index)

        else:
            # Other variables
            station = self._get_station(gwo_var)
            converted = self._get_converted(station, start, end)
            series = converted[gwo_var]

        # Interpolate to target times
        if not times.equals(series.index):
            # Ensure times are timezone-naive
            if hasattr(times, "tz") and times.tz is not None:
                times = times.tz_localize(None)
            if hasattr(series.index, "tz") and series.index.tz is not None:
                series.index = series.index.tz_localize(None)

            # Union and interpolate
            union_idx = series.index.union(times)
            series = series.reindex(union_idx).interpolate(method="time")
            series = series.reindex(times)

        return series.values.astype(np.float64)


def parse_station_map(spec: str) -> dict[str, str]:
    """
    Parse station map specification string.

    Format: "var1:station1,var2:station2,*:default_station"

    Parameters
    ----------
    spec : str
        Station map specification

    Returns
    -------
    dict[str, str]
        Mapping of variable names to station names

    Examples
    --------
    >>> parse_station_map("slht:Tokyo,kous:Tokyo,*:Chiba")
    {'slht': 'Tokyo', 'kous': 'Tokyo', '*': 'Chiba'}
    """
    result = {}
    for item in spec.split(","):
        item = item.strip()
        if ":" in item:
            var, station = item.split(":", 1)
            result[var.strip()] = station.strip()
    return result


def parse_period(
    start_str: str, end_str: str | None = None
) -> tuple[datetime, datetime]:
    """
    Parse period specification.

    Parameters
    ----------
    start_str : str
        Start specification:
        - Year only ("2020"): END optional, defaults to full year
        - Date only ("2020-06-15"): END required
        - Full datetime: END required
    end_str : str, optional
        End specification (same format as start_str):
        - Year only: end_year+1-01-01T00:00
        - Date only: end_date+1 day T00:00
        - Full datetime: Exact as specified

    Returns
    -------
    tuple[datetime, datetime]
        (start, end) datetimes
    """
    from datetime import timedelta

    if len(start_str) == 4:  # Year only
        start = datetime(int(start_str), 1, 1, 0, 0)
        if end_str and len(end_str) == 4:
            end = datetime(int(end_str) + 1, 1, 1, 0, 0)
        else:
            end = datetime(int(start_str) + 1, 1, 1, 0, 0)
    elif len(start_str) == 10:  # Date only (YYYY-MM-DD)
        start = datetime.strptime(start_str, "%Y-%m-%d")
        if end_str and len(end_str) == 10:
            end = datetime.strptime(end_str, "%Y-%m-%d") + timedelta(days=1)
        else:
            raise ValueError(
                "End date required for date-only start (e.g., --end 2020-01-07)"
            )
    else:  # Full datetime
        start = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        if end_str:
            end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
        else:
            raise ValueError("End datetime required for full datetime specification")
    return start, end

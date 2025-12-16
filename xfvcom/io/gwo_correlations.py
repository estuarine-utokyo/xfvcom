# -*- coding: utf-8 -*-
"""
Station correlation management for GWO-AMD gap filling.

This module computes and manages correlations between weather stations
for use in gap filling when primary station data is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    from .gwo_reader import GWOReader


@dataclass
class CorrelationParams:
    """Correlation parameters between two stations for a variable."""

    r2: float
    slope: float
    intercept: float
    rmse: float


# Physical constraints for each variable (raw GWO units)
PHYSICAL_CONSTRAINTS: dict[str, dict[str, float | None]] = {
    "kion": {"min": -400, "max": 500},  # 0.1°C → -40 to 50°C
    "rhum": {"min": 0, "max": 100},  # Humidity %
    "shpa": {"min": 8500, "max": 11000},  # 0.1 hPa → 850-1100 hPa
    "sped": {"min": 0, "max": None},  # 0.1 m/s, no upper limit
    "muki": {"min": 0, "max": 16},  # Direction code 0-16
    "clod": {"min": 0, "max": 10},  # Cloud cover 0-10
    "slht": {"min": 0, "max": None},  # Solar radiation, no upper limit
    "kous": {"min": 0, "max": None},  # Precipitation, no upper limit
}

# Physical constraints in converted units
PHYSICAL_CONSTRAINTS_CONVERTED: dict[str, dict[str, float | None]] = {
    "kion": {"min": -40.0, "max": 50.0},  # °C
    "rhum": {"min": 0.0, "max": 100.0},  # %
    "shpa": {"min": 850.0, "max": 1100.0},  # hPa
    "sped": {"min": 0.0, "max": None},  # m/s
    "muki": {"min": 0.0, "max": 360.0},  # degrees
    "clod": {"min": 0.0, "max": 1.0},  # fraction
    "slht": {"min": 0.0, "max": None},  # W/m²
    "kous": {"min": 0.0, "max": None},  # m/s
}

# Default station coordinates (from gwo_stn.csv)
STATION_COORDS: dict[str, dict[str, float]] = {
    "Tokyo": {"latitude": 35.41, "longitude": 139.4548, "altitude": 6.5},
    "Chiba": {"latitude": 35.3554, "longitude": 140.0624, "altitude": 3.5},
    "Yokohama": {"latitude": 35.2612, "longitude": 139.3918, "altitude": 39.1},
    "Tateyama": {"latitude": 34.59, "longitude": 139.5206, "altitude": 5.8},
}


class StationCorrelations:
    """
    Manage correlations between weather stations for gap filling.

    Parameters
    ----------
    data : dict | None
        Correlation data loaded from YAML file or computed
    """

    def __init__(self, data: dict[str, object] | None = None) -> None:
        self.data: dict[str, object] = data or {}
        self.correlations: dict[str, dict[str, CorrelationParams]] = {}
        self._parse_data()

    def _parse_data(self) -> None:
        """Parse correlation data into CorrelationParams objects."""
        station_correlations = self.data.get("station_correlations", {})
        if not isinstance(station_correlations, dict):
            return

        for pair_key, variables in station_correlations.items():
            if not isinstance(variables, dict):
                continue
            self.correlations[pair_key] = {}
            for var, params in variables.items():
                if isinstance(params, dict):
                    self.correlations[pair_key][var] = CorrelationParams(
                        r2=float(params.get("r2", 0.0)),
                        slope=float(params.get("slope", 1.0)),
                        intercept=float(params.get("intercept", 0.0)),
                        rmse=float(params.get("rmse", 0.0)),
                    )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "StationCorrelations":
        """Load correlations from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(data)

    @classmethod
    def compute(
        cls,
        gwo_reader: "GWOReader",
        station_pairs: list[tuple[str, str]],
        years: list[int],
        variables: list[str] | None = None,
    ) -> "StationCorrelations":
        """
        Compute correlations from GWO data.

        Parameters
        ----------
        gwo_reader : GWOReader
            GWO data reader instance
        station_pairs : list[tuple[str, str]]
            List of (primary, secondary) station pairs to compute
        years : list[int]
            Years to use for correlation computation
        variables : list[str] | None
            Variables to compute correlations for (default: common variables)

        Returns
        -------
        StationCorrelations
            Instance with computed correlations
        """
        if variables is None:
            variables = ["kion", "rhum", "shpa", "sped", "clod"]

        data: dict[str, object] = {"station_correlations": {}}
        station_corr: dict[str, dict[str, dict[str, float]]] = {}

        for primary, secondary in station_pairs:
            pair_key = f"{primary}_{secondary}"
            station_corr[pair_key] = {}

            for var in variables:
                params = cls._compute_correlation(
                    gwo_reader, primary, secondary, var, years
                )
                station_corr[pair_key][var] = {
                    "r2": params.r2,
                    "slope": params.slope,
                    "intercept": params.intercept,
                    "rmse": params.rmse,
                }

        data["station_correlations"] = station_corr
        return cls(data)

    @staticmethod
    def _compute_correlation(
        gwo_reader: "GWOReader",
        station1: str,
        station2: str,
        var: str,
        years: list[int],
    ) -> CorrelationParams:
        """
        Compute correlation between two stations for a variable.

        Parameters
        ----------
        gwo_reader : GWOReader
            GWO data reader
        station1 : str
            Primary station name
        station2 : str
            Secondary station name
        var : str
            Variable name (e.g., "kion", "rhum")
        years : list[int]
            Years to use for computation

        Returns
        -------
        CorrelationParams
            Computed correlation parameters
        """
        from scipy import stats

        all_s1: list[float] = []
        all_s2: list[float] = []

        for year in years:
            try:
                df1 = gwo_reader.load_station_year(station1, year)
                df2 = gwo_reader.load_station_year(station2, year)
            except FileNotFoundError:
                continue

            # Align by time index
            common_idx = df1.index.intersection(df2.index)

            v1 = df1.loc[common_idx, var]
            v2 = df2.loc[common_idx, var]

            # Filter valid observations (RMK == 8 is normal)
            rmk1 = df1.loc[common_idx, f"{var}RMK"]
            rmk2 = df2.loc[common_idx, f"{var}RMK"]

            # For most variables, RMK >= 3 is usable; 8 is normal
            valid = (rmk1 >= 3) & (rmk2 >= 3)

            all_s1.extend(v1[valid].dropna().values)
            all_s2.extend(v2[valid].dropna().values)

        if len(all_s1) < 10:
            # Not enough data, return identity transform
            return CorrelationParams(
                r2=0.0, slope=1.0, intercept=0.0, rmse=float("nan")
            )

        s1 = np.array(all_s1, dtype=np.float64)
        s2 = np.array(all_s2, dtype=np.float64)

        # Linear regression: s1 = slope * s2 + intercept
        slope, intercept, r_value, _, _ = stats.linregress(s2, s1)
        r2 = r_value**2

        predicted = slope * s2 + intercept
        rmse = float(np.sqrt(np.mean((s1 - predicted) ** 2)))

        return CorrelationParams(
            r2=float(r2),
            slope=float(slope),
            intercept=float(intercept),
            rmse=rmse,
        )

    def convert(
        self,
        primary: str,
        secondary: str,
        var: str,
        values: pd.Series,
        *,
        use_converted_units: bool = True,
    ) -> pd.Series:
        """
        Convert values from secondary station to primary station scale.

        Applies: primary = slope * secondary + intercept
        Then enforces physical constraints.

        Parameters
        ----------
        primary : str
            Primary station name
        secondary : str
            Secondary station name
        var : str
            Variable name
        values : pd.Series
            Values from secondary station
        use_converted_units : bool
            If True, use constraints for converted units (default)

        Returns
        -------
        pd.Series
            Converted values with physical constraints applied
        """
        pair_key = f"{primary}_{secondary}"

        if pair_key not in self.correlations:
            # No conversion available, return as-is
            return values.copy()

        if var not in self.correlations[pair_key]:
            return values.copy()

        params = self.correlations[pair_key][var]
        converted = params.slope * values + params.intercept

        # Apply physical constraints
        constraints = (
            PHYSICAL_CONSTRAINTS_CONVERTED
            if use_converted_units
            else PHYSICAL_CONSTRAINTS
        )
        var_constraints = constraints.get(var, {})
        if var_constraints.get("min") is not None:
            converted = converted.clip(lower=var_constraints["min"])
        if var_constraints.get("max") is not None:
            converted = converted.clip(upper=var_constraints["max"])

        return converted

    def get_params(
        self, primary: str, secondary: str, var: str
    ) -> CorrelationParams | None:
        """Get correlation parameters for a station pair and variable."""
        pair_key = f"{primary}_{secondary}"
        if pair_key not in self.correlations:
            return None
        return self.correlations[pair_key].get(var)

    def to_yaml(self, path: Path | str) -> None:
        """Save correlations to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.data, f, default_flow_style=False, allow_unicode=True)

    def summary(self) -> pd.DataFrame:
        """
        Generate summary DataFrame of all correlations.

        Returns
        -------
        pd.DataFrame
            Summary with columns: primary, secondary, variable, r2, slope, intercept, rmse
        """
        rows: list[dict[str, object]] = []
        for pair_key, variables in self.correlations.items():
            parts = pair_key.split("_", 1)
            if len(parts) != 2:
                continue
            primary, secondary = parts
            for var, params in variables.items():
                rows.append(
                    {
                        "primary": primary,
                        "secondary": secondary,
                        "variable": var,
                        "r2": params.r2,
                        "slope": params.slope,
                        "intercept": params.intercept,
                        "rmse": params.rmse,
                    }
                )
        return pd.DataFrame(rows)


def parse_fallback_stations(spec: str) -> dict[str, list[str]]:
    """
    Parse fallback station specification string.

    Format: "Primary:Fallback1,Fallback2;Primary2:Fallback1,Fallback2"

    Parameters
    ----------
    spec : str
        Fallback station specification

    Returns
    -------
    dict[str, list[str]]
        Mapping of primary station to list of fallback stations

    Examples
    --------
    >>> parse_fallback_stations("Chiba:Tokyo,Yokohama,Tateyama")
    {'Chiba': ['Tokyo', 'Yokohama', 'Tateyama']}
    """
    result: dict[str, list[str]] = {}
    for item in spec.split(";"):
        item = item.strip()
        if ":" not in item:
            continue
        primary, fallbacks = item.split(":", 1)
        fallback_list = [fb.strip() for fb in fallbacks.split(",") if fb.strip()]
        if fallback_list:
            result[primary.strip()] = fallback_list
    return result


def get_station_coordinates(station: str) -> dict[str, float]:
    """
    Get coordinates for a station.

    Parameters
    ----------
    station : str
        Station name

    Returns
    -------
    dict[str, float]
        Dictionary with latitude, longitude, altitude

    Raises
    ------
    ValueError
        If station not found in known stations
    """
    if station not in STATION_COORDS:
        raise ValueError(
            f"Unknown station: {station}. "
            f"Known stations: {', '.join(STATION_COORDS.keys())}"
        )
    return STATION_COORDS[station].copy()


def get_cache_path() -> Path:
    """
    Get the path to the correlation cache file.

    Returns
    -------
    Path
        Path to xfvcom/data/gwo_correlations_cache.yaml
    """
    from xfvcom.data import DATA_DIR

    return DATA_DIR / "gwo_correlations_cache.yaml"


def get_cached_correlations(
    gwo_dir: Path,
    stations: list[str],
    years: list[int] | None = None,
    force_recompute: bool = False,
) -> StationCorrelations:
    """
    Get correlations from cache or compute if needed.

    Parameters
    ----------
    gwo_dir : Path
        GWO data directory (used for computing if cache miss)
    stations : list[str]
        List of stations to include
    years : list[int] | None
        Years to use for computation (default: 2015-2020)
    force_recompute : bool
        Force recomputation even if cache exists

    Returns
    -------
    StationCorrelations
        Cached or newly computed correlations
    """
    from .gwo_reader import GWOReader

    cache_path = get_cache_path()

    if years is None:
        years = [2015, 2016, 2017, 2018, 2019, 2020]

    # Check if cache exists and is valid
    if not force_recompute and cache_path.exists():
        try:
            cached = StationCorrelations.from_yaml(cache_path)

            # Validate cache has required station pairs
            required_pairs: set[str] = set()
            for s1 in stations:
                for s2 in stations:
                    if s1 != s2:
                        required_pairs.add(f"{s1}_{s2}")

            cached_pairs = set(cached.correlations.keys())

            if required_pairs.issubset(cached_pairs):
                print(f"[INFO] Using cached correlations from: {cache_path}")
                return cached
            else:
                missing = required_pairs - cached_pairs
                print(f"[INFO] Cache missing pairs: {missing}. Recomputing...")
        except Exception as e:
            print(f"[WARNING] Failed to load cache: {e}. Recomputing...")

    # Compute correlations
    print(f"[INFO] Computing correlations for stations: {', '.join(stations)}")
    print("[INFO] This may take a few minutes...")

    reader = GWOReader(gwo_dir)

    # Generate all station pairs (both directions)
    station_pairs: list[tuple[str, str]] = []
    for s1 in stations:
        for s2 in stations:
            if s1 != s2:
                station_pairs.append((s1, s2))

    correlations = StationCorrelations.compute(
        gwo_reader=reader,
        station_pairs=station_pairs,
        years=years,
        variables=["kion", "rhum", "shpa", "sped", "clod"],
    )

    # Try to add solar model parameters
    try:
        from .solar_estimation import SolarEstimator

        solar_station = "Tokyo" if "Tokyo" in stations else stations[0]
        params = SolarEstimator.build_empirical_model(
            gwo_reader=reader,
            station=solar_station,
            years=years,
        )
        correlations.data["solar_model"] = {
            "station": solar_station,
            "a": params.a,
            "b": params.b,
            "r2": params.r2,
            "rmse": params.rmse,
            "n_samples": params.n_samples,
        }
        print(
            f"[INFO] Solar model calibrated: "
            f"a={params.a:.4f}, b={params.b:.4f}, R²={params.r2:.4f}"
        )
    except ImportError:
        print("[WARNING] pvlib not installed, skipping solar model calibration")
    except Exception as e:
        print(f"[WARNING] Solar model calibration failed: {e}")

    # Save to cache
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        correlations.to_yaml(cache_path)
        print(f"[INFO] Correlations cached to: {cache_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save cache: {e}")

    return correlations


def clear_correlation_cache() -> bool:
    """
    Clear the correlation cache.

    Returns
    -------
    bool
        True if cache was deleted, False if it didn't exist
    """
    cache_path = get_cache_path()
    if cache_path.exists():
        cache_path.unlink()
        return True
    return False

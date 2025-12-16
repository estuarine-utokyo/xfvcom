# -*- coding: utf-8 -*-
"""
Solar radiation estimation using pvlib and cloud cover.

This module estimates global horizontal irradiance (GHI) from clear-sky
models and cloud cover observations. It supports multiple estimation
models for gap filling of solar radiation data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

try:
    import pvlib
    from pvlib.location import Location

    PVLIB_AVAILABLE = True
except ImportError:
    PVLIB_AVAILABLE = False
    Location = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from .gwo_reader import GWOReader


@dataclass
class EmpiricalModelParams:
    """Parameters for empirical solar radiation model.

    Model: GHI = G_cs × (1 - a × C^b)
    where G_cs is clear-sky GHI and C is cloud cover fraction.
    """

    a: float = 0.72
    b: float = 3.2
    r2: float | None = None
    rmse: float | None = None
    n_samples: int | None = None


# Default empirical parameters (calibrated from Tokyo data)
DEFAULT_EMPIRICAL_PARAMS = EmpiricalModelParams(a=0.72, b=3.2)


class SolarEstimator:
    """
    Estimate global horizontal irradiance (GHI) from clear-sky model and cloud cover.

    Parameters
    ----------
    model : str
        Model type: "empirical", "pvlib-kasten", "pvlib-larson"
    latitude : float
        Station latitude (default: Tokyo = 35.41)
    longitude : float
        Station longitude (default: Tokyo = 139.4548)
    altitude : float
        Station altitude in meters (default: 6.5)
    empirical_params : EmpiricalModelParams | None
        Parameters for empirical model (used only when model="empirical")
    timezone : str
        Timezone for location (default: "Asia/Tokyo")
    """

    def __init__(
        self,
        model: Literal["empirical", "pvlib-kasten", "pvlib-larson"] = "empirical",
        latitude: float = 35.41,
        longitude: float = 139.4548,
        altitude: float = 6.5,
        empirical_params: EmpiricalModelParams | None = None,
        timezone: str = "Asia/Tokyo",
    ) -> None:
        if not PVLIB_AVAILABLE:
            raise ImportError(
                "pvlib is required for solar estimation. "
                "Install with: pip install pvlib"
            )

        self.model = model
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.timezone = timezone
        self.location = Location(latitude, longitude, timezone, altitude)
        self.empirical_params = empirical_params or DEFAULT_EMPIRICAL_PARAMS

    def estimate(
        self,
        times: pd.DatetimeIndex,
        cloud_cover: pd.Series,
    ) -> pd.Series:
        """
        Estimate GHI for given times and cloud cover.

        Parameters
        ----------
        times : pd.DatetimeIndex
            Timestamps to estimate (should be timezone-naive or in local time)
        cloud_cover : pd.Series
            Cloud cover fraction (0-1), indexed by time

        Returns
        -------
        pd.Series
            Estimated GHI in W/m²
        """
        # Ensure times are localized for pvlib
        if times.tz is None:
            times_local = times.tz_localize(self.timezone)
        else:
            times_local = times.tz_convert(self.timezone)

        # Calculate clear-sky GHI
        clearsky = self.location.get_clearsky(times_local, model="ineichen")
        ghi_clearsky = clearsky["ghi"]

        # Get solar position for nighttime detection
        solar_pos = self.location.get_solarposition(times_local)
        is_night = solar_pos["elevation"] < 0

        # Align cloud cover to times
        cloud = cloud_cover.reindex(times).fillna(method="ffill").fillna(method="bfill")
        cloud = cloud.clip(0, 1)

        # Apply cloud correction based on model
        if self.model == "empirical":
            a = self.empirical_params.a
            b = self.empirical_params.b
            cloud_factor = 1 - a * np.power(cloud.values, b)
        elif self.model == "pvlib-kasten":
            # Kasten-Czeplak (1980) model
            cloud_factor = 1 - 0.75 * np.power(cloud.values, 3.4)
        elif self.model == "pvlib-larson":
            # Larson et al. (2016) simplified model
            cloud_factor = 1 - 0.87 * cloud.values
        else:
            raise ValueError(f"Unknown solar model: {self.model}")

        cloud_factor = np.clip(cloud_factor, 0, 1)

        # Calculate GHI
        ghi_values = ghi_clearsky.values * cloud_factor

        # Set nighttime to 0
        ghi_values[is_night.values] = 0.0

        # Ensure non-negative
        ghi_values = np.maximum(ghi_values, 0.0)

        # Return as Series with original index
        return pd.Series(ghi_values, index=times, name="ghi")

    def get_clearsky_ghi(self, times: pd.DatetimeIndex) -> pd.Series:
        """
        Get clear-sky GHI for given times.

        Parameters
        ----------
        times : pd.DatetimeIndex
            Timestamps

        Returns
        -------
        pd.Series
            Clear-sky GHI in W/m²
        """
        if times.tz is None:
            times_local = times.tz_localize(self.timezone)
        else:
            times_local = times.tz_convert(self.timezone)

        clearsky = self.location.get_clearsky(times_local, model="ineichen")
        return pd.Series(clearsky["ghi"].values, index=times, name="ghi_clearsky")

    def is_nighttime(self, times: pd.DatetimeIndex) -> pd.Series:
        """
        Determine nighttime (solar elevation < 0).

        Parameters
        ----------
        times : pd.DatetimeIndex
            Timestamps

        Returns
        -------
        pd.Series
            Boolean series, True for nighttime
        """
        if times.tz is None:
            times_local = times.tz_localize(self.timezone)
        else:
            times_local = times.tz_convert(self.timezone)

        solar_pos = self.location.get_solarposition(times_local)
        return pd.Series(
            solar_pos["elevation"].values < 0, index=times, name="is_night"
        )

    @classmethod
    def build_empirical_model(
        cls,
        gwo_reader: "GWOReader",
        station: str = "Tokyo",
        years: list[int] | None = None,
    ) -> EmpiricalModelParams:
        """
        Build empirical model parameters from observed data.

        Fits: GHI = G_cs × (1 - a × C^b)

        Parameters
        ----------
        gwo_reader : GWOReader
            GWO data reader
        station : str
            Station to use for calibration (default: Tokyo)
        years : list[int] | None
            Years to use for calibration (default: 2015-2020)

        Returns
        -------
        EmpiricalModelParams
            Fitted parameters with statistics
        """
        from scipy.optimize import curve_fit

        from .gwo_correlations import get_station_coordinates

        if not PVLIB_AVAILABLE:
            raise ImportError("pvlib is required for model calibration")

        if years is None:
            years = [2015, 2016, 2017, 2018, 2019, 2020]

        coords = get_station_coordinates(station)
        location = Location(
            coords["latitude"],
            coords["longitude"],
            "Asia/Tokyo",
            coords["altitude"],
        )

        all_obs: list[float] = []
        all_cs: list[float] = []
        all_cloud: list[float] = []

        for year in years:
            try:
                df = gwo_reader.load_station_year(station, year)
            except FileNotFoundError:
                continue

            # Convert units
            # Solar radiation: 0.01 MJ/m²/h -> W/m²
            obs_ghi: pd.Series = df["slht"] * 0.01 * 1e6 / 3600.0
            # Cloud cover: 0-10 -> 0-1
            cloud: pd.Series = df["clod"] / 10.0

            # Filter valid observations
            # slht: RMK >= 3 is usable (2 = nighttime is valid 0)
            # clod: RMK >= 3 is usable (3-hourly observations)
            valid_slht = df["slhtRMK"].isin([3, 4, 5, 6, 7, 8])
            valid_clod = df["clodRMK"].isin([3, 4, 5, 6, 7, 8])
            valid = valid_slht & valid_clod

            # Localize times for pvlib
            times_local = df.index.tz_localize("Asia/Tokyo")

            # Calculate clear-sky
            clearsky = location.get_clearsky(times_local, model="ineichen")
            cs_ghi = clearsky["ghi"]

            # Daytime only (clear-sky > 10 W/m²)
            daytime = cs_ghi > 10

            mask = valid & daytime
            all_obs.extend(obs_ghi[mask].values)
            all_cs.extend(cs_ghi[mask].values)
            all_cloud.extend(cloud[mask].values)

        if len(all_obs) < 100:
            raise ValueError(
                f"Insufficient data for model calibration: {len(all_obs)} samples"
            )

        obs = np.array(all_obs, dtype=np.float64)
        cs = np.array(all_cs, dtype=np.float64)
        cld = np.array(all_cloud, dtype=np.float64)

        # Remove any NaN values
        valid_mask = ~(np.isnan(obs) | np.isnan(cs) | np.isnan(cld))
        obs = obs[valid_mask]
        cs = cs[valid_mask]
        cld = cld[valid_mask]

        # Fit Kasten-Czeplak style model
        def model_func(
            X: tuple[np.ndarray, np.ndarray], a: float, b: float
        ) -> np.ndarray:
            g_cs, cloud = X
            return g_cs * (1 - a * np.power(np.clip(cloud, 0, 1), b))

        try:
            popt, _ = curve_fit(
                model_func,
                (cs, cld),
                obs,
                p0=[0.75, 3.4],
                bounds=([0.1, 1.0], [1.5, 5.0]),
                maxfev=10000,
            )
        except RuntimeError:
            # If fitting fails, use default parameters
            return DEFAULT_EMPIRICAL_PARAMS

        # Evaluate
        predicted = model_func((cs, cld), *popt)
        predicted = np.maximum(predicted, 0)

        ss_res: float = float(np.sum((obs - predicted) ** 2))
        ss_tot: float = float(np.sum((obs - np.mean(obs)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean((obs - predicted) ** 2)))

        return EmpiricalModelParams(
            a=float(popt[0]),
            b=float(popt[1]),
            r2=float(r2),
            rmse=rmse,
            n_samples=len(obs),
        )

    @classmethod
    def compare_models(
        cls,
        gwo_reader: "GWOReader",
        station: str = "Tokyo",
        train_years: list[int] | None = None,
        test_year: int = 2020,
    ) -> pd.DataFrame:
        """
        Compare different solar estimation models.

        Parameters
        ----------
        gwo_reader : GWOReader
            GWO data reader
        station : str
            Station to use
        train_years : list[int] | None
            Years for calibration (default: 2015-2019)
        test_year : int
            Year for testing (default: 2020)

        Returns
        -------
        pd.DataFrame
            Comparison results with columns: model, r2, rmse, mae, bias
        """
        from .gwo_correlations import get_station_coordinates

        if not PVLIB_AVAILABLE:
            raise ImportError("pvlib is required for model comparison")

        if train_years is None:
            train_years = [2015, 2016, 2017, 2018, 2019]

        coords = get_station_coordinates(station)

        # Build empirical model
        empirical_params = cls.build_empirical_model(gwo_reader, station, train_years)

        # Load test data
        try:
            df_test = gwo_reader.load_station_year(station, test_year)
        except FileNotFoundError:
            raise ValueError(f"Test data not found for {station} {test_year}")

        # Convert units
        obs_ghi: pd.Series = df_test["slht"] * 0.01 * 1e6 / 3600.0
        cloud: pd.Series = df_test["clod"] / 10.0

        # Filter valid daytime observations
        valid_slht = df_test["slhtRMK"].isin([3, 4, 5, 6, 7, 8])
        valid_clod = df_test["clodRMK"].isin([3, 4, 5, 6, 7, 8])

        location = Location(
            coords["latitude"],
            coords["longitude"],
            "Asia/Tokyo",
            coords["altitude"],
        )
        times_local = df_test.index.tz_localize("Asia/Tokyo")
        clearsky = location.get_clearsky(times_local, model="ineichen")
        cs_ghi = clearsky["ghi"]

        daytime = cs_ghi > 10
        mask = valid_slht & valid_clod & daytime

        obs: np.ndarray = obs_ghi[mask].values
        cs = cs_ghi[mask].values
        cld: np.ndarray = cloud[mask].values

        # Calculate predictions for each model
        models = ["empirical", "pvlib-kasten", "pvlib-larson"]
        results: list[dict[str, object]] = []

        for model_name in models:
            if model_name == "empirical":
                pred = cs * (1 - empirical_params.a * np.power(cld, empirical_params.b))
            elif model_name == "pvlib-kasten":
                pred = cs * (1 - 0.75 * np.power(cld, 3.4))
            else:  # pvlib-larson
                pred = cs * (1 - 0.87 * cld)

            pred = np.maximum(pred, 0)

            ss_res: float = float(np.sum((obs - pred) ** 2))
            ss_tot: float = float(np.sum((obs - np.mean(obs)) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
            mae = float(np.mean(np.abs(obs - pred)))
            bias = float(np.mean(pred - obs))

            results.append(
                {
                    "model": model_name,
                    "r2": float(r2),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "bias": float(bias),
                    "n_samples": len(obs),
                }
            )

        return pd.DataFrame(results)


def estimate_solar_radiation(
    times: pd.DatetimeIndex,
    cloud_cover: pd.Series,
    latitude: float = 35.41,
    longitude: float = 139.4548,
    altitude: float = 6.5,
    model: str = "empirical",
) -> pd.Series:
    """
    Convenience function to estimate solar radiation.

    Parameters
    ----------
    times : pd.DatetimeIndex
        Timestamps
    cloud_cover : pd.Series
        Cloud cover fraction (0-1)
    latitude : float
        Station latitude
    longitude : float
        Station longitude
    altitude : float
        Station altitude in meters
    model : str
        Estimation model: "empirical", "pvlib-kasten", or "pvlib-larson"

    Returns
    -------
    pd.Series
        Estimated GHI in W/m²
    """
    estimator = SolarEstimator(
        model=model,  # type: ignore[arg-type]
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
    )
    return estimator.estimate(times, cloud_cover)

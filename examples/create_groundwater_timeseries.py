#!/usr/bin/env python3
"""
Create time series CSV files for groundwater forcing with realistic variations.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def create_flux_timeseries(
    start_date: str,
    end_date: str,
    active_nodes: list,
    output_file: str = "flux_timeseries.csv",
):
    """Create flux time series with tidal variation."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Create hourly timestamps
    times = pd.date_range(start, end, freq="1H")

    # Create DataFrame
    df = pd.DataFrame(index=times)
    df.index.name = "datetime"

    # Generate flux data for each node with tidal variation
    for i, node in enumerate(active_nodes):
        # Base flux
        base_flux = 1e-6 * (1 + i * 0.2)  # Different base for each node

        # Tidal variation (M2 tide ~12.42 hours)
        tidal_period = 12.42
        phase = i * np.pi / 4  # Different phase for each node

        # Calculate flux with tidal variation
        hours = np.arange(len(times))
        flux = base_flux * (1 + 0.3 * np.sin(2 * np.pi * hours / tidal_period + phase))

        # Add some random noise
        flux += np.random.normal(0, base_flux * 0.05, len(times))

        df[str(node)] = flux

    # Save to CSV
    df.to_csv(output_file, float_format="%.3e")
    print(f"Created flux time series: {output_file}")
    return df


def create_temperature_timeseries(
    start_date: str,
    end_date: str,
    active_nodes: list,
    output_file: str = "temperature_timeseries.csv",
):
    """Create temperature time series with seasonal and diurnal variation."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Create hourly timestamps
    times = pd.date_range(start, end, freq="1H")

    # Create DataFrame
    df = pd.DataFrame(index=times)
    df.index.name = "datetime"

    # Generate temperature data for each node
    for i, node in enumerate(active_nodes):
        # Base temperature (groundwater is typically stable)
        base_temp = 10.0 + i * 0.5  # Slightly different for each node

        # Small seasonal variation (groundwater has minimal seasonal change)
        day_of_year = times.dayofyear
        seasonal = 2.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak in summer

        # Very small diurnal variation (groundwater is insulated)
        hour_of_day = times.hour
        diurnal = 0.2 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # Peak at noon

        # Combine variations
        temp = base_temp + seasonal + diurnal

        # Add small random fluctuations
        temp += np.random.normal(0, 0.1, len(times))

        df[str(node)] = temp

    # Save to CSV
    df.to_csv(output_file, float_format="%.2f")
    print(f"Created temperature time series: {output_file}")
    return df


def create_salinity_timeseries(
    start_date: str,
    end_date: str,
    active_nodes: list,
    output_file: str = "salinity_timeseries.csv",
):
    """Create salinity time series with variations based on location."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    # Create hourly timestamps
    times = pd.date_range(start, end, freq="1H")

    # Create DataFrame
    df = pd.DataFrame(index=times)
    df.index.name = "datetime"

    # Generate salinity data for each node
    for i, node in enumerate(active_nodes):
        # Base salinity (varies by proximity to coast)
        if i == 0:
            base_salinity = 0.0  # Fresh groundwater
        elif i == 1:
            base_salinity = 0.5  # Slightly brackish
        elif i == 2:
            base_salinity = 1.0  # Brackish
        else:
            base_salinity = 2.0  # More saline

        # Add slow variation (e.g., due to rainfall events)
        days = np.arange(len(times)) / 24
        variation = 0.2 * base_salinity * np.sin(2 * np.pi * days / 30)  # 30-day cycle

        # Combine
        salinity = base_salinity + variation

        # Ensure non-negative
        salinity = np.maximum(salinity, 0)

        # Add small random fluctuations
        salinity += np.random.normal(0, 0.05, len(times))
        salinity = np.maximum(salinity, 0)

        df[str(node)] = salinity

    # Save to CSV
    df.to_csv(output_file, float_format="%.3f")
    print(f"Created salinity time series: {output_file}")
    return df


def main():
    """Create all time series files."""
    # Configuration
    start_date = "2024-01-01"
    end_date = "2024-01-10"
    active_nodes = [637, 638, 639, 662]

    # Create time series
    flux_df = create_flux_timeseries(start_date, end_date, active_nodes)
    temp_df = create_temperature_timeseries(start_date, end_date, active_nodes)
    salt_df = create_salinity_timeseries(start_date, end_date, active_nodes)

    # Print summary statistics
    print("\nFlux velocity statistics (m/s):")
    print(flux_df.describe())

    print("\nTemperature statistics (°C):")
    print(temp_df.describe())

    print("\nSalinity statistics (PSU):")
    print(salt_df.describe())

    # Create a config file that uses these time series
    config = f"""# Groundwater configuration using time series data
grid_nc: "../../FVCOM/Tests/GroundwaterDye/input/chn_grd.dat"
utm_zone: 33
northern: true

start: "{start_date}T00:00:00Z"
end: "{end_date}T00:00:00Z"
dt_seconds: 3600

output: "groundwater_with_timeseries.nc"

active_nodes: {active_nodes}

# Time series data files
flux: "flux_timeseries.csv"
temperature: "temperature_timeseries.csv"
salinity: "salinity_timeseries.csv"

# Optional dye
dye: 100.0

timezone: "UTC"
"""

    with open("groundwater_timeseries_auto.yaml", "w") as f:
        f.write(config)

    print("\nCreated configuration file: groundwater_timeseries_auto.yaml")
    print("\nTo generate the NetCDF file, run:")
    print("  xfvcom-make-groundwater-nc groundwater_timeseries_auto.yaml")


if __name__ == "__main__":
    main()

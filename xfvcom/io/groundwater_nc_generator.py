# -*- coding: utf-8 -*-
"""Generate groundwater forcing NetCDF files for FVCOM."""

from __future__ import annotations

import tempfile
from datetime import timezone
from pathlib import Path
from typing import Any, Sequence

import netCDF4 as nc
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..grid.grid_obj import FvcomGrid
from .base_generator import BaseGenerator
from .sources.base import BaseForcingSource
from .sources.timeseries import TimeSeriesSource


class GroundwaterNetCDFGenerator(BaseGenerator):
    """
    Generate a groundwater forcing NetCDF file for FVCOM.

    Based on the MATLAB function write_groundwater.m, this generator creates
    NetCDF files containing groundwater flux, temperature, and salinity data
    at model nodes.

    Parameters
    ----------
    grid_nc : Path
        NetCDF file that contains static mesh variables (`x`, `y`, `lon`, `lat`, `nv`).
        Can also be a .dat file.
    start, end : str
        ISO-8601 UTC datetimes (e.g. '2025-01-01T00:00:00Z').
    dt_seconds : int
        Time step in seconds.
    utm_zone : int, optional
        UTM zone number if using .dat file input.
    northern : bool
        Whether in northern hemisphere (for UTM conversion).
    start_tz : str
        Timezone for start/end if not specified.
    flux : array-like or float
        Groundwater flux velocity (m/s). Can be:
        - Single float: constant value for all nodes/times
        - 1D array (node,): constant in time, varies by node
        - 2D array (node, time): varies by node and time
        Note: This is a velocity (m/s), not volumetric flux. FVCOM multiplies
        this by the node's bottom area internally.
    temperature : array-like or float
        Groundwater temperature (°C). Same format options as flux.
    salinity : array-like or float
        Groundwater salinity (PSU). Same format options as flux.
    dye : array-like or float, optional
        Groundwater dye concentration (tracer units). Same format options as flux.
    ideal : bool
        If True, use ideal time format (days since 0.0).
        If False, use modified Julian day format.
    """

    def __init__(
        self,
        grid_nc: Path | str,
        start: str,
        end: str,
        dt_seconds: int = 3600,
        *,
        utm_zone: int | None = None,
        northern: bool = True,
        start_tz: str = "UTC",
        flux: float | NDArray[np.float64] = 0.0,
        temperature: float | NDArray[np.float64] = 0.0,
        salinity: float | NDArray[np.float64] = 0.0,
        dye: float | NDArray[np.float64] | None = None,
        ideal: bool = False,
    ) -> None:
        # Parse timestamps
        t0 = pd.Timestamp(start)
        t1 = pd.Timestamp(end)
        if t0.tzinfo is None:
            t0 = t0.tz_localize(start_tz)
        if t1.tzinfo is None:
            t1 = t1.tz_localize(start_tz)
        self.start = t0.tz_convert("UTC")
        self.end = t1.tz_convert("UTC")

        source = Path(grid_nc)
        super().__init__(source)

        self.source: Path = source
        self.dt = dt_seconds
        self.utm_zone = utm_zone
        self.northern = northern
        self.ideal = ideal

        # Store forcing data
        self.flux_data = flux
        self.temp_data = temperature
        self.salt_data = salinity
        self.dye_data = dye

        # Build timeline in UTC
        self.timeline = pd.date_range(
            self.start,
            self.end,
            freq=f"{self.dt}s",
            inclusive="both",
            tz="UTC",
        )

    # ------------------------------------------------------------------
    # Time conversion helpers (from met_nc_generator.py)
    # ------------------------------------------------------------------
    @staticmethod
    def _to_mjd(t: pd.DatetimeIndex) -> NDArray[np.float32]:
        """Convert to Modified Julian Day."""
        origin = pd.Timestamp("1858-11-17T00:00:00Z")
        return ((t - origin) / pd.Timedelta("1D")).to_numpy("f4")

    @staticmethod
    def _to_ideal_days(t: pd.DatetimeIndex) -> NDArray[np.float32]:
        """Convert to days since 0.0 (ideal format)."""
        # FVCOM ideal format: days since year 0
        origin = pd.Timestamp("0000-01-01T00:00:00Z")
        # pandas doesn't support year 0, so we use an offset
        # Days from year 1 to 1858-11-17 is approximately 678576 days
        mjd = GroundwaterNetCDFGenerator._to_mjd(t)
        return mjd + 678576.0

    @staticmethod
    def _itime_pair(days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split days into integer days and milliseconds."""
        itime: NDArray[np.int32] = days.astype("int32")
        itime2: NDArray[np.int32] = ((days - itime) * 86400000).astype("int32")
        return itime, itime2

    @staticmethod
    def _times_char(t: pd.DatetimeIndex) -> np.ndarray:
        """Format times as character array."""
        s = t.strftime("%Y-%m-%dT%H:%M:%S.000000")
        return np.asarray([list(i.ljust(26)) for i in s], dtype="S1")

    def _prepare_forcing_data(
        self, data: float | NDArray[np.float64], node: int, nt: int
    ) -> NDArray[np.float64]:
        """
        Prepare forcing data array with shape (time, node) for FVCOM.

        Parameters
        ----------
        data : float or array
            Input data
        node : int
            Number of nodes
        nt : int
            Number of time steps

        Returns
        -------
        NDArray
            Array with shape (nt, node) following Fortran convention
        """
        if isinstance(data, (float, int)):
            # Constant value for all nodes and times
            return np.full((nt, node), float(data), dtype=np.float64)

        data_arr = np.asarray(data, dtype=np.float64)

        if data_arr.ndim == 1:
            # 1D array: constant in time, varies by node
            if len(data_arr) != node:
                raise ValueError(
                    f"1D array must have length {node} (number of nodes), "
                    f"got {len(data_arr)}"
                )
            # Broadcast to all time steps - shape (nt, node)
            return np.tile(data_arr[np.newaxis, :], (nt, 1))

        elif data_arr.ndim == 2:
            # 2D array: varies by node and time
            # Accept either (node, nt) or (nt, node) and transpose if needed
            if data_arr.shape == (node, nt):
                # Input is (node, time), transpose to (time, node)
                return data_arr.T
            elif data_arr.shape == (nt, node):
                # Already in correct shape
                return data_arr
            else:
                raise ValueError(
                    f"2D array must have shape ({node}, {nt}) or ({nt}, {node}), "
                    f"got {data_arr.shape}"
                )

        else:
            raise ValueError(
                f"Data must be scalar, 1D, or 2D array, got {data_arr.ndim}D"
            )

    # ------------------------------------------------------------------
    # BaseGenerator interface
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load grid data from file."""
        # Accept .dat or .nc
        if self.source.suffix.lower() == ".dat":
            grid = FvcomGrid.from_dat(
                self.source,
                utm_zone=self.utm_zone,
                northern=self.northern,
            )
            ds = grid.to_xarray()
            self.mesh_ds = ds
        else:  # assume NetCDF grid
            import xarray as xr

            self.mesh_ds = xr.open_dataset(self.source)

        self.timeline = pd.date_range(
            self.start, self.end, freq=f"{self.dt}s", inclusive="both", tz="UTC"
        )

    def validate(self) -> None:
        """Validate that required grid variables exist."""
        # Must have either x,y or lon,lat coordinates
        has_xy = "x" in self.mesh_ds and "y" in self.mesh_ds
        has_geo = "lon" in self.mesh_ds and "lat" in self.mesh_ds

        if not (has_xy or has_geo):
            raise ValueError(
                "Grid file must have either (x,y) or (lon,lat) coordinates"
            )

        # Must have connectivity matrix
        if "nv" not in self.mesh_ds:
            raise ValueError("Grid file missing connectivity matrix 'nv'")

    def render(self) -> bytes:
        """Generate the groundwater NetCDF file content."""
        if not hasattr(self, "mesh_ds"):
            self.load()

        nt = self.timeline.size
        nele: int = int(self.mesh_ds.sizes["nele"])
        node: int = int(self.mesh_ds.sizes["node"])

        # Prepare time variables
        if self.ideal:
            time_days = self._to_ideal_days(self.timeline)
        else:
            time_days = self._to_mjd(self.timeline)

        itime, itime2 = self._itime_pair(time_days)
        times_char = self._times_char(self.timeline)

        # Prepare forcing data arrays
        flux_array = self._prepare_forcing_data(self.flux_data, node, nt)
        temp_array = self._prepare_forcing_data(self.temp_data, node, nt)
        salt_array = self._prepare_forcing_data(self.salt_data, node, nt)
        dye_array = None
        if self.dye_data is not None:
            dye_array = self._prepare_forcing_data(self.dye_data, node, nt)

        # Determine coordinate system based on original MATLAB logic
        # MATLAB code checks coordinate type, but for FVCOM compatibility
        # we should write the coordinate system that matches the input grid
        has_geo = "lon" in self.mesh_ds and "lat" in self.mesh_ds
        has_xy = "x" in self.mesh_ds and "y" in self.mesh_ds

        # Write to temporary file then return bytes
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        with nc.Dataset(tmp_path, "w", format="NETCDF4_CLASSIC") as ds_out:
            # ---------------------------------------------------------
            # Dimensions
            # ---------------------------------------------------------
            ds_out.createDimension("node", node)
            ds_out.createDimension("nele", nele)
            ds_out.createDimension("time", None)  # unlimited
            ds_out.createDimension("DateStrLen", 26)

            # ---------------------------------------------------------
            # Coordinate variables - matching MATLAB write_groundwater.m
            # ---------------------------------------------------------
            # Check coordinate system type from grid attributes or data
            coord_system = self.mesh_ds.attrs.get("CoordinateSystem", "Cartesian")

            if coord_system == "Geographic" or (has_geo and not has_xy):
                # Write geographic coordinates
                v_lon = ds_out.createVariable("lon", "f4", ("node",))
                v_lon.long_name = "nodal longitude"
                v_lon.units = "degrees_east"
                v_lon[:] = self.mesh_ds["lon"].values

                v_lat = ds_out.createVariable("lat", "f4", ("node",))
                v_lat.long_name = "nodal latitude"
                v_lat.units = "degrees_north"
                v_lat[:] = self.mesh_ds["lat"].values
            else:
                # Write Cartesian coordinates
                v_x = ds_out.createVariable("x", "f4", ("node",))
                v_x.long_name = "nodal x"
                v_x.units = "meter"
                v_x[:] = self.mesh_ds["x"].values

                v_y = ds_out.createVariable("y", "f4", ("node",))
                v_y.long_name = "nodal y"
                v_y.units = "meter"
                v_y[:] = self.mesh_ds["y"].values

            # ---------------------------------------------------------
            # Time variables
            # ---------------------------------------------------------
            v_time = ds_out.createVariable("time", "f4", ("time",))
            v_time.long_name = "time"
            v_time.time_zone = "UTC"

            v_itime = ds_out.createVariable("Itime", "i4", ("time",))
            v_itime.time_zone = "UTC"

            v_itime2 = ds_out.createVariable("Itime2", "i4", ("time",))
            v_itime2.units = "msec since 00:00:00"
            v_itime2.time_zone = "UTC"

            if self.ideal:
                v_time.units = "days since 0.0"
                v_itime.units = "days since 0.0"
            else:
                v_time.units = "days since 1858-11-17 00:00:00"
                v_time.format = "modified julian day (MJD)"
                v_itime.units = "days since 1858-11-17 00:00:00"
                v_itime.format = "modified julian day (MJD)"

                # Times character array (only for non-ideal)
                v_times = ds_out.createVariable("Times", "S1", ("time", "DateStrLen"))
                v_times.time_zone = "UTC"

            # ---------------------------------------------------------
            # Groundwater forcing variables
            # ---------------------------------------------------------
            v_flux = ds_out.createVariable("groundwater_flux", "f4", ("time", "node"))
            v_flux.long_name = "Ground Water Flux Velocity"
            v_flux.units = "m s-1"

            v_temp = ds_out.createVariable("groundwater_temp", "f4", ("time", "node"))
            v_temp.long_name = "Ground Water Temperature"
            v_temp.units = "degree C"

            v_salt = ds_out.createVariable("groundwater_salt", "f4", ("time", "node"))
            v_salt.long_name = "Ground Water Salinity"
            v_salt.units = "psu"

            # Optional dye variable
            if dye_array is not None:
                v_dye = ds_out.createVariable("groundwater_dye", "f4", ("time", "node"))
                v_dye.long_name = "Ground Water Dye Concentration"
                v_dye.units = "tracer units"

            # ---------------------------------------------------------
            # Global attributes
            # ---------------------------------------------------------
            ds_out.source = "fvcom grid (unstructured) surface forcing"

            # ---------------------------------------------------------
            # Write data
            # ---------------------------------------------------------
            # Time variables
            v_time[:] = time_days
            v_itime[:] = itime
            v_itime2[:] = itime2
            if not self.ideal:
                v_times[:] = times_char

            # Forcing variables
            v_flux[:] = flux_array
            v_temp[:] = temp_array
            v_salt[:] = salt_array
            if dye_array is not None:
                v_dye[:] = dye_array

        # Read file contents and cleanup
        content = tmp_path.read_bytes()
        tmp_path.unlink()

        return content

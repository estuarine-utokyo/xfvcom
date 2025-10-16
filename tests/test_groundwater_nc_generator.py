"""Tests for groundwater NetCDF generator."""

from __future__ import annotations


import netCDF4 as nc
import numpy as np
import pytest
import xarray as xr

from xfvcom.io.groundwater_nc_generator import GroundwaterNetCDFGenerator


class TestGroundwaterNetCDFGenerator:
    """Test groundwater NetCDF file generation."""

    @pytest.fixture
    def sample_grid_file(self, tmp_path):
        """Create a sample grid NetCDF file."""
        grid_file = tmp_path / "test_grid.nc"

        # Create simple triangular mesh
        node = 4
        nele = 2

        with nc.Dataset(grid_file, "w") as ds:
            # Dimensions
            ds.createDimension("node", node)
            ds.createDimension("nele", nele)
            ds.createDimension("three", 3)

            # Variables
            x = ds.createVariable("x", "f8", ("node",))
            y = ds.createVariable("y", "f8", ("node",))
            lon = ds.createVariable("lon", "f8", ("node",))
            lat = ds.createVariable("lat", "f8", ("node",))
            nv = ds.createVariable("nv", "i4", ("nele", "three"))

            # Simple square grid
            x[:] = [0.0, 1.0, 1.0, 0.0]
            y[:] = [0.0, 0.0, 1.0, 1.0]
            lon[:] = [139.0, 140.0, 140.0, 139.0]
            lat[:] = [35.0, 35.0, 36.0, 36.0]
            nv[:] = [[1, 2, 3], [1, 3, 4]]  # 1-indexed

            # Attributes
            ds.CoordinateSystem = "Cartesian"

        return grid_file

    def test_constant_values(self, sample_grid_file, tmp_path):
        """Test generation with constant values for all nodes and times."""
        output_file = tmp_path / "groundwater_const.nc"

        gen = GroundwaterNetCDFGenerator(
            grid_nc=sample_grid_file,
            start="2025-01-01T00:00:00Z",
            end="2025-01-01T03:00:00Z",
            dt_seconds=3600,
            flux=0.001,
            temperature=15.0,
            salinity=0.5,
        )

        gen.write(output_file)

        # Verify output (decode_times=False to avoid issues with FVCOM's time format)
        with xr.open_dataset(output_file, decode_times=False) as ds:
            # Check dimensions
            assert ds.sizes["node"] == 4
            assert ds.sizes["time"] == 4  # 0, 1, 2, 3 hours

            # Check variables exist
            assert "groundwater_flux" in ds
            assert "groundwater_temp" in ds
            assert "groundwater_salt" in ds

            # Check values
            assert np.allclose(ds["groundwater_flux"].values, 0.001)
            assert np.allclose(ds["groundwater_temp"].values, 15.0)
            assert np.allclose(ds["groundwater_salt"].values, 0.5)

            # Check time variables
            assert "time" in ds
            assert "Itime" in ds
            assert "Itime2" in ds
            assert "Times" in ds

    def test_node_varying_values(self, sample_grid_file, tmp_path):
        """Test generation with node-specific values."""
        output_file = tmp_path / "groundwater_nodes.nc"

        # Different values for each node
        flux_by_node = np.array([0.001, 0.002, 0.003, 0.004])
        temp_by_node = np.array([10.0, 12.0, 14.0, 16.0])

        gen = GroundwaterNetCDFGenerator(
            grid_nc=sample_grid_file,
            start="2025-01-01T00:00:00Z",
            end="2025-01-01T01:00:00Z",
            dt_seconds=3600,
            flux=flux_by_node,
            temperature=temp_by_node,
            salinity=0.0,  # constant salinity
        )

        gen.write(output_file)

        # Verify output (decode_times=False to avoid issues with FVCOM's time format)
        with xr.open_dataset(output_file, decode_times=False) as ds:
            # Check flux varies by node (dimensions are time, node)
            for i in range(4):
                assert np.allclose(ds["groundwater_flux"][:, i].values, flux_by_node[i])

            # Check temperature varies by node (dimensions are time, node)
            for i in range(4):
                assert np.allclose(ds["groundwater_temp"][:, i].values, temp_by_node[i])

            # Check salinity is constant
            assert np.allclose(ds["groundwater_salt"].values, 0.0)

    def test_time_varying_values(self, sample_grid_file, tmp_path):
        """Test generation with time-varying values."""
        output_file = tmp_path / "groundwater_time.nc"

        # Create time-varying data (time x node)
        nt = 5  # 5 time steps
        node = 4
        flux_data = np.random.rand(nt, node) * 0.01

        gen = GroundwaterNetCDFGenerator(
            grid_nc=sample_grid_file,
            start="2025-01-01T00:00:00Z",
            end="2025-01-01T04:00:00Z",
            dt_seconds=3600,
            flux=flux_data,
            temperature=20.0,
            salinity=35.0,
        )

        gen.write(output_file)

        # Verify output (decode_times=False to avoid issues with FVCOM's time format)
        with xr.open_dataset(output_file, decode_times=False) as ds:
            # Check dimensions
            assert ds.sizes["time"] == nt

            # Check flux varies correctly
            assert np.allclose(ds["groundwater_flux"].values, flux_data)

    def test_ideal_time_format(self, sample_grid_file, tmp_path):
        """Test generation with ideal time format."""
        output_file = tmp_path / "groundwater_ideal.nc"

        gen = GroundwaterNetCDFGenerator(
            grid_nc=sample_grid_file,
            start="2025-01-01T00:00:00Z",
            end="2025-01-01T06:00:00Z",
            dt_seconds=3600,
            flux=0.0,
            temperature=15.0,
            salinity=0.0,
            ideal=True,
        )

        gen.write(output_file)

        # Verify output (decode_times=False to avoid issues with FVCOM's time format)
        with xr.open_dataset(output_file, decode_times=False) as ds:
            # Check time units for ideal format
            assert ds["time"].attrs["units"] == "days since 0.0"
            assert ds["Itime"].attrs["units"] == "days since 0.0"

            # Should not have Times variable in ideal format
            assert "Times" not in ds

    def test_geographic_coordinates(self, tmp_path):
        """Test with geographic coordinate system."""
        # Create grid with geographic coordinates
        grid_file = tmp_path / "geo_grid.nc"

        with nc.Dataset(grid_file, "w") as ds:
            ds.createDimension("node", 3)
            ds.createDimension("nele", 1)
            ds.createDimension("three", 3)

            lon = ds.createVariable("lon", "f8", ("node",))
            lat = ds.createVariable("lat", "f8", ("node",))
            nv = ds.createVariable("nv", "i4", ("nele", "three"))

            lon[:] = [139.0, 140.0, 139.5]
            lat[:] = [35.0, 35.0, 35.5]
            nv[:] = [[1, 2, 3]]

            ds.CoordinateSystem = "Geographic"

        output_file = tmp_path / "groundwater_geo.nc"

        gen = GroundwaterNetCDFGenerator(
            grid_nc=grid_file,
            start="2025-01-01T00:00:00Z",
            end="2025-01-01T01:00:00Z",
            flux=0.001,
            temperature=10.0,
            salinity=35.0,
        )

        gen.write(output_file)

        # Verify output has geographic coordinates
        with xr.open_dataset(output_file, decode_times=False) as ds:
            assert "lon" in ds
            assert "lat" in ds
            assert "x" not in ds
            assert "y" not in ds

    def test_cli_integration(self, sample_grid_file, tmp_path):
        """Test CLI interface."""
        from xfvcom.cli.make_groundwater_nc import main

        output_file = tmp_path / "groundwater_cli.nc"

        # Test basic CLI usage
        args = [
            str(sample_grid_file),
            "--start",
            "2025-01-01T00:00:00Z",
            "--end",
            "2025-01-01T12:00:00Z",
            "--dt",
            "3600",
            "--flux",
            "0.005",
            "--temperature",
            "18.5",
            "--salinity",
            "32.0",
            "-o",
            str(output_file),
        ]

        result = main(args)
        assert result == 0
        assert output_file.exists()

        # Verify the generated file
        with xr.open_dataset(output_file, decode_times=False) as ds:
            assert np.allclose(ds["groundwater_flux"].values, 0.005)
            assert np.allclose(ds["groundwater_temp"].values, 18.5)
            assert np.allclose(ds["groundwater_salt"].values, 32.0)

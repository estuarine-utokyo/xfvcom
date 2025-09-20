"""
FVCOM input file loader that creates xarray Datasets compatible with FvcomPlotter.

This module provides functionality to load FVCOM input files (grid, OBC, forcing)
and create xarray Datasets with the same structure as FVCOM output files,
allowing the use of existing plotting tools for both input and output data.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from xfvcom.grid import FvcomGrid


class FvcomInputLoader:
    """
    Load FVCOM input files and create xarray Dataset compatible with FvcomPlotter.

    This class reads FVCOM input files (grid, OBC, forcing) and creates an xarray
    Dataset with the same structure as FVCOM output NetCDF files. This allows
    using the same plotting tools (FvcomPlotter) for both input and output data.

    Parameters
    ----------
    grid_path : str or Path, optional
        Path to FVCOM grid file (.dat or .grd format)
    obc_path : str or Path, optional
        Path to open boundary condition file
    forcing_path : str or Path, optional
        Path to forcing file (river, meteorological, etc.)
    utm_zone : int, optional
        UTM zone for coordinate transformation. Required for geographic grids.
        For idealized grids without real geographic meaning, any valid zone works.
    add_dummy_time : bool, default=True
        Add a dummy time dimension for compatibility with FvcomPlotter
    add_dummy_siglay : bool, default=True
        Add a dummy siglay (vertical) dimension for compatibility
    time_ref : str or pd.Timestamp, default="2000-01-01"
        Reference time for dummy time dimension

    Attributes
    ----------
    ds : xr.Dataset
        xarray Dataset compatible with FvcomPlotter
    grid : FvcomGrid or None
        Grid object if grid_path was provided

    Examples
    --------
    >>> # Load grid only
    >>> loader = FvcomInputLoader(grid_path="grid.dat", utm_zone=54)
    >>> plotter = FvcomPlotter(loader.ds, config)
    >>> plotter.plot_2d(da=None, opts=opts)  # Plot mesh

    >>> # Load grid and OBC
    >>> loader = FvcomInputLoader(
    ...     grid_path="grid.dat",
    ...     obc_path="obc.dat",
    ...     utm_zone=54
    ... )
    """

    def __init__(
        self,
        grid_path: Optional[Union[str, Path]] = None,
        obc_path: Optional[Union[str, Path]] = None,
        forcing_path: Optional[Union[str, Path]] = None,
        utm_zone: Optional[int] = None,
        add_dummy_time: bool = True,
        add_dummy_siglay: bool = True,
        time_ref: Union[str, pd.Timestamp] = "2000-01-01",
    ):
        self.grid_path = Path(grid_path) if grid_path else None
        self.obc_path = Path(obc_path) if obc_path else None
        self.forcing_path = Path(forcing_path) if forcing_path else None
        self.utm_zone = utm_zone
        self.add_dummy_time = add_dummy_time
        self.add_dummy_siglay = add_dummy_siglay
        self.time_ref = pd.Timestamp(time_ref)

        self.grid: Optional[FvcomGrid] = None
        self.ds: Optional[xr.Dataset] = None

        # Load data and create dataset
        self._load_data()

    def _load_data(self) -> None:
        """Load input files and create compatible dataset."""
        # Start with empty dataset
        data_vars: Dict[str, Any] = {}
        coords: Dict[str, Any] = {}
        attrs = {
            "source": "FVCOM Input Files",
            "data_type": "input",
            "created_by": "FvcomInputLoader",
        }

        # Load grid if provided
        if self.grid_path:
            self._load_grid(data_vars, coords, attrs)

        # Load OBC if provided
        if self.obc_path:
            self._load_obc(data_vars, attrs)

        # Load forcing if provided
        if self.forcing_path:
            self._load_forcing(data_vars, attrs)

        # Add dummy dimensions if requested
        if self.add_dummy_time:
            self._add_dummy_time(data_vars, coords)

        if self.add_dummy_siglay:
            self._add_dummy_siglay(data_vars, coords)

        # Create dataset
        self.ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)

        # Add coordinate attributes for CF compliance
        self._add_coordinate_attributes()

        # Create nv_zero and nv_ccw for matplotlib compatibility
        self._setup_nv_ccw()

    def _load_grid(
        self, data_vars: Dict[str, Any], coords: Dict[str, Any], attrs: Dict[str, Any]
    ) -> None:
        """Load grid file and add mesh variables to dataset."""
        if self.grid_path is None:
            raise ValueError("grid_path is None")

        if not self.grid_path.exists():
            raise FileNotFoundError(f"Grid file not found: {self.grid_path}")

        # Load grid based on file extension
        if self.grid_path.suffix in [".dat", ".grd"]:
            if self.utm_zone is None:
                raise ValueError(
                    "utm_zone is required for .dat/.grd grid files. "
                    "For idealized grids, any valid zone (1-60) can be used."
                )
            self.grid = FvcomGrid.from_dat(self.grid_path, utm_zone=self.utm_zone)
        else:
            raise ValueError(f"Unsupported grid file format: {self.grid_path.suffix}")

        # Convert grid to xarray dataset
        grid_ds = self.grid.to_xarray()

        # Add mesh variables
        # Node coordinates
        data_vars["x"] = xr.DataArray(
            grid_ds["x"].values,
            dims=["node"],
            attrs={"long_name": "x-coordinate", "units": "meters"},
        )
        data_vars["y"] = xr.DataArray(
            grid_ds["y"].values,
            dims=["node"],
            attrs={"long_name": "y-coordinate", "units": "meters"},
        )
        data_vars["lon"] = xr.DataArray(
            grid_ds["lon"].values,
            dims=["node"],
            attrs={"long_name": "longitude", "units": "degrees_east"},
        )
        data_vars["lat"] = xr.DataArray(
            grid_ds["lat"].values,
            dims=["node"],
            attrs={"long_name": "latitude", "units": "degrees_north"},
        )

        # Element connectivity (ensure 1-based for FVCOM compatibility)
        nv = grid_ds["nv"].values
        if nv.min() == 0:
            nv = nv + 1  # Convert 0-based to 1-based

        data_vars["nv"] = xr.DataArray(
            nv,
            dims=["three", "nele"],
            attrs={
                "long_name": "nodes surrounding element",
                "note": "1-based indexing for FVCOM compatibility",
            },
        )

        # Bathymetry (if available)
        if "h" in grid_ds:
            data_vars["h"] = xr.DataArray(
                grid_ds["h"].values,
                dims=["node"],
                attrs={
                    "long_name": "bathymetry",
                    "units": "meters",
                    "positive": "down",
                },
            )
        else:
            # Create dummy bathymetry if not provided
            data_vars["h"] = xr.DataArray(
                np.ones(self.grid.node) * 10.0,  # 10m depth everywhere
                dims=["node"],
                attrs={
                    "long_name": "bathymetry",
                    "units": "meters",
                    "positive": "down",
                    "note": "dummy bathymetry for input data",
                },
            )

        # Element center coordinates
        if "xc" in grid_ds:
            data_vars["xc"] = xr.DataArray(
                grid_ds["xc"].values,
                dims=["nele"],
                attrs={"long_name": "x-coordinate element center", "units": "meters"},
            )
            data_vars["yc"] = xr.DataArray(
                grid_ds["yc"].values,
                dims=["nele"],
                attrs={"long_name": "y-coordinate element center", "units": "meters"},
            )
            data_vars["lonc"] = xr.DataArray(
                grid_ds["lonc"].values,
                dims=["nele"],
                attrs={
                    "long_name": "longitude element center",
                    "units": "degrees_east",
                },
            )
            data_vars["latc"] = xr.DataArray(
                grid_ds["latc"].values,
                dims=["nele"],
                attrs={
                    "long_name": "latitude element center",
                    "units": "degrees_north",
                },
            )

        # Add grid info to attributes
        attrs["grid_file"] = str(self.grid_path)
        attrs["node"] = self.grid.node
        attrs["nele"] = self.grid.nele
        attrs["utm_zone"] = self.utm_zone

    def _load_obc(self, data_vars: Dict[str, Any], attrs: Dict[str, Any]) -> None:
        """Load open boundary condition file."""
        if not self.obc_path.exists():
            raise FileNotFoundError(f"OBC file not found: {self.obc_path}")

        # Read OBC file (implement based on OBC file format)
        # For now, just add to attributes
        attrs["obc_file"] = str(self.obc_path)

        # TODO: Implement OBC loading based on file format
        # This would add variables like:
        # - obc_nodes: nodes on open boundary
        # - obc_types: type of boundary condition
        # - obc_values: prescribed values if any

    def _load_forcing(self, data_vars: Dict[str, Any], attrs: Dict[str, Any]) -> None:
        """Load forcing file."""
        if not self.forcing_path.exists():
            raise FileNotFoundError(f"Forcing file not found: {self.forcing_path}")

        attrs["forcing_file"] = str(self.forcing_path)

        # TODO: Implement forcing loading based on file format
        # This would add variables based on forcing type:
        # - River forcing: river_flux, river_temp, river_salt
        # - Met forcing: wind_x, wind_y, air_pressure, etc.

    def _add_dummy_time(
        self, data_vars: Dict[str, Any], coords: Dict[str, Any]
    ) -> None:
        """Add dummy time dimension for compatibility."""
        coords["time"] = xr.DataArray(
            [self.time_ref],
            dims=["time"],
            attrs={
                "long_name": "time",
                "units": f"days since {self.time_ref.strftime('%Y-%m-%d %H:%M:%S')}",
                "calendar": "proleptic_gregorian",
                "note": "dummy dimension for input data",
            },
        )

        # Also create Itime and Itime2 for full FVCOM compatibility
        data_vars["Itime"] = xr.DataArray(
            [self.time_ref.toordinal()],
            dims=["time"],
            attrs={"long_name": "modified julian day", "units": "days"},
        )

        milliseconds = (
            self.time_ref.hour * 3600 + self.time_ref.minute * 60 + self.time_ref.second
        ) * 1000
        data_vars["Itime2"] = xr.DataArray(
            [milliseconds],
            dims=["time"],
            attrs={"long_name": "milliseconds since midnight", "units": "milliseconds"},
        )

    def _add_dummy_siglay(
        self, data_vars: Dict[str, Any], coords: Dict[str, Any]
    ) -> None:
        """Add dummy siglay (vertical) dimension for compatibility."""
        # Add single sigma layer
        coords["siglay"] = xr.DataArray(
            [-0.5],  # Mid-depth
            dims=["siglay"],
            attrs={
                "long_name": "sigma layer coordinate",
                "units": "dimensionless",
                "positive": "up",
                "note": "dummy dimension for input data",
            },
        )

        coords["siglev"] = xr.DataArray(
            [0.0, -1.0],  # Surface and bottom
            dims=["siglev"],
            attrs={
                "long_name": "sigma level coordinate",
                "units": "dimensionless",
                "positive": "up",
                "note": "dummy dimension for input data",
            },
        )

    def _add_coordinate_attributes(self) -> None:
        """Add coordinate attributes for CF compliance and FvcomPlotter compatibility."""
        # Mark coordinate variables
        if self.ds is not None and "x" in self.ds:
            self.ds["x"].attrs["standard_name"] = "projection_x_coordinate"
        if self.ds is not None and "y" in self.ds:
            self.ds["y"].attrs["standard_name"] = "projection_y_coordinate"
        if self.ds is not None and "lon" in self.ds:
            self.ds["lon"].attrs["standard_name"] = "longitude"
        if self.ds is not None and "lat" in self.ds:
            self.ds["lat"].attrs["standard_name"] = "latitude"
        if self.ds is not None and "time" in self.ds:
            self.ds["time"].attrs["standard_name"] = "time"

        # Add global attributes expected by FvcomPlotter
        if self.ds is not None:
            self.ds.attrs["Conventions"] = "CF-1.6"
            self.ds.attrs["CoordinateSystem"] = "Cartesian"
            if self.utm_zone:
                self.ds.attrs["CoordinateProjection"] = f"UTM Zone {self.utm_zone}"

    def _setup_nv_ccw(self) -> None:
        """
        Setup nv_zero and nv_ccw for matplotlib compatibility.

        This method creates:
        - nv_zero: 0-based node connectivity (matplotlib expects 0-based)
        - nv_ccw: counter-clockwise ordering of nodes
        """
        if self.ds is None or "nv" not in self.ds:
            return

        # Create 0-based version (FVCOM uses 1-based indexing)
        self.ds["nv_zero"] = xr.DataArray(
            self.ds["nv"].values.T - 1, dims=("nele", "three")
        )
        self.ds["nv_zero"].attrs[
            "long_name"
        ] = "nodes surrounding element in zero-based for matplotlib"

        # Create counter-clockwise version
        nv_ccw = self.ds["nv_zero"].to_numpy()
        nv_ccw = nv_ccw[:, ::-1]  # Reverse node order for CCW
        self.ds["nv_ccw"] = xr.DataArray(nv_ccw, dims=("nele", "three"))
        self.ds["nv_ccw"].attrs[
            "long_name"
        ] = "nodes surrounding element in counter-clockwise direction for matplotlib"

    def compare_with_output(self, output_ds: xr.Dataset) -> Dict[str, Any]:
        """
        Compare input data with FVCOM output dataset.

        Parameters
        ----------
        output_ds : xr.Dataset
            FVCOM output dataset to compare against

        Returns
        -------
        dict
            Comparison results including differences in grid, dimensions, etc.
        """
        results: Dict[str, Any] = {"grid": {}, "dimensions": {}, "variables": {}}

        # Compare grid coordinates
        if self.ds is not None:
            for coord in ["x", "y", "lon", "lat"]:
                if coord in self.ds and coord in output_ds:
                    diff = np.abs(self.ds[coord].values - output_ds[coord].values)
                    results["grid"][coord] = {
                        "max_diff": float(np.max(diff)),
                        "mean_diff": float(np.mean(diff)),
                        "matching": np.allclose(
                            self.ds[coord].values, output_ds[coord].values, rtol=1e-5
                        ),
                    }

            # Compare dimensions
            for dim in ["node", "nele"]:
                if dim in self.ds.dims and dim in output_ds.dims:
                    results["dimensions"][dim] = {
                        "input": self.ds.dims[dim],
                        "output": output_ds.dims[dim],
                        "matching": self.ds.dims[dim] == output_ds.dims[dim],
                    }

        return results

    def calculate_node_area(
        self,
        node_indices: list[int] | np.ndarray | None = None,
        index_base: int = 1,
    ) -> float:
        """Calculate total area of triangular elements containing specified nodes.

        Parameters
        ----------
        node_indices : list[int] | np.ndarray | None
            List of node indices. If None, calculates area for all nodes.
        index_base : int
            0 for zero-based indexing, 1 for one-based indexing (FVCOM default)

        Returns
        -------
        float
            Total area in square meters (assuming x, y are in meters/UTM)

        Examples
        --------
        >>> loader = FvcomInputLoader("grid.dat", utm_zone=54)
        >>> area = loader.calculate_node_area([100, 200, 300])
        >>> print(f"Total area: {area:.0f} m²")
        """
        if self.grid is None:
            raise ValueError("Grid not loaded. Initialize with a valid grid file.")

        return self.grid.calculate_node_area(node_indices, index_base)

import os
import numpy as np
import xarray as xr
import pyproj
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import DateFormatter
from .helpers import PlotHelperMixin

class FvcomDataLoader:
    """
    Responsible for loading FVCOM output NetCDF files into an xarray.Dataset.
    """
    def __init__(self, dirpath=None, ncfile=None, utm2geo=True, zone=54, north=True,
                 inverse=False, time_tolerance=None, **kwargs):
        """
        Initialize the FvcomDataLoader instance.
        
        Parameters:
        - dirpath: Directory path where the NetCDF file is located.
        - ncfile: Name of the NetCDF file to load.
        - utm2geo: Convert UTM coordinates to geographic (lon, lat).
        - zone: UTM zone number.
        - north: True if the UTM zone is in the northern hemisphere.
        - inverse: True to convert geographic coordinates to UTM.
        - time_tolerence: Tolerence in minutes in integer to snap time to the nearest hour.
        - **kwargs: Additional keyword arguments for xarray.open_dataset.
        """
        dirpath = os.path.expanduser(dirpath) if dirpath else None
        dirpath = self._add_trailing_slash(dirpath) if dirpath else None
        self.ncfilepath = f"{dirpath}{ncfile}" if dirpath else ncfile
        self.engine = kwargs.get("engine", "netcdf4")
        self.chunks = kwargs.get("chunks", None)
        self.decode_times = kwargs.get("decode_times", False)
        self.utm2geo = utm2geo
        self.zone = zone
        self.north = north
        self.inverse = inverse
        #self.time_tolerence = time_tolerence
        self.ds = self._load_dataset()
        if 'time' in self.ds.dims:
            if time_tolerance:
                self._time_tolerence(time_tolerance)
        if all(var in self.ds for var in ['x', 'y', 'xc', 'yc']):
            if self.utm2geo:
                self._convert_utm_to_geo()
        if all(var in self.ds for var in ['zeta', 'siglay', 'h']):
            self._add_depth_variables()

    def slice_by_time(self, start, end, copy=False, reset_time=False):
        """
        Slice the dataset by a time range.

        Parameters:
        - start: Start time as a string (e.g., "2020-01-01 00:00:00") or datetime.
        - end: End time as a string (e.g., "2020-01-07 00:00:00") or datetime.
        - copy: If True, return a copied dataset to avoid modifying the original.
        - reset_time: If True, reset the time dimension to start from zero after slicing.

        Returns:
        - Sliced xarray.Dataset or None if the operation fails.
        """
        if "time" not in self.ds.dims:
            print("Error: The dataset does not have a 'time' dimension.")
            return None

        try:
            # Convert start and end to np.datetime64 if they are strings
            start = np.datetime64(start) if isinstance(start, str) else start
            end = np.datetime64(end) if isinstance(end, str) else end

            # Check if start and end are valid
            if start > end:
                print("Warning: Start time is later than end time. Swapping the values.")
                start, end = end, start

            # Check if time range is within dataset bounds
            dataset_time_min = self.ds["time"].min().values
            dataset_time_max = self.ds["time"].max().values
            if start < dataset_time_min or end > dataset_time_max:
                print(f"Warning: Specified time range ({start} to {end}) is out of dataset bounds "
                      f"({dataset_time_min} to {dataset_time_max}).")

            # Perform slicing
            sliced_ds = self.ds.sel(time=slice(start, end))
            if copy:
                sliced_ds = sliced_ds.copy(deep=True)

            # Reset time dimension if requested
            if reset_time:
                sliced_ds["time"] = (sliced_ds["time"] - sliced_ds["time"].min()) / np.timedelta64(1, "s")
                sliced_ds["time"].attrs["units"] = "seconds since start of slice"

            print(f"Slicing successful: Dataset sliced from {start} to {end}.")
            return sliced_ds

        except Exception as e:
            print(f"Error slicing dataset: {e}")
            return None

    def _load_dataset(self):
        try:
            ds = xr.open_dataset(self.ncfilepath, engine=self.engine, chunks=self.chunks, decode_times=self.decode_times)
            ds = ds.drop_vars('Itime2') if 'Itime2' in ds.variables else ds
            return xr.decode_cf(ds)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.ncfilepath}")

    def _convert_utm_to_geo(self):
        """Convert UTM coordinates (x, y) and (xc, yc) to geographic (lon, lat)."""
        lon, lat = self._xy_to_lonlat(self.ds.x.values, self.ds.y.values)
        lonc, latc = self._xy_to_lonlat(self.ds.xc.values, self.ds.yc.values)

        self.ds["lon"] = xr.DataArray(lon, dims="node")
        self.ds["lat"] = xr.DataArray(lat, dims="node")
        self.ds["lonc"] = xr.DataArray(lonc, dims="nele")
        self.ds["latc"] = xr.DataArray(latc, dims="nele")

    def _add_depth_variables(self):
        """Add 'z' and 'z_dfs' depth variables to the dataset."""
        z = xr.apply_ufunc(
            lambda zeta, siglay, h: zeta + siglay * (h + zeta),
            self.ds.zeta, self.ds.siglay, self.ds.h,
            input_core_dims=[["time", "node"], ["siglay"], ["node"]],
            output_core_dims=[["time", "siglay", "node"]],
            vectorize=True, dask="parallelized"
        )
        z_dfs = self.ds.zeta - z

        self.ds['z'] = z
        self.ds['z_dfs'] = z_dfs

        self.ds['z'].attrs['long_name'] = 'Depth'
        self.ds['z'].attrs['standard_name'] = 'Depth at siglay'
        self.ds['z'].attrs['units'] = 'm'
        self.ds['z'].attrs['positive'] = 'up'
        self.ds['z'].attrs['origin'] = 'still water level'

        self.ds['z_dfs'].attrs['long_name'] = 'Depth from surface'
        self.ds['z_dfs'].attrs['standard_name'] = 'Depth at siglay from the surface'
        self.ds['z_dfs'].attrs['units'] = 'm'
        self.ds['z_dfs'].attrs['positive'] = 'down'
        self.ds['z_dfs'].attrs['origin'] = 'surface'

    def _xy_to_lonlat(self, x, y):
        """Convert UTM (x, y) to geographic coordinates (lon, lat)."""
        crs_from = f"+proj=utm +zone={self.zone} {'+north' if self.north else ''} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        crs_to = f"+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        transformer = pyproj.Transformer.from_crs(crs_from, crs_to)
        return transformer.transform(x, y)

    def _add_trailing_slash(self, directory_path):
        '''
        Add slash ("/") if directory path does not end with slash.
        '''

        if not directory_path.endswith('/'):
            directory_path += '/'
        return directory_path
    
    def _time_tolerence(self, time_tolerance):
        is_positive_integer = isinstance(time_tolerance, int) and time_tolerance > 0
        if not is_positive_integer:
            print("Error: time_tolerance must be a positive integer in minutes. Continuing without correction.")
            return None
        tolerance = np.timedelta64(time_tolerance, 'm')

        # 時間を datetime64 に変換
        time = self.ds['time'].values.astype('datetime64[s]')
        #time = self.ds['time'].values
        # 誤差範囲内で正時にスナップ
        corrected_time = (time + tolerance // 2).astype('datetime64[h]').astype('datetime64[ns]')
        self.ds['time'] = corrected_time

class FvcomAnalyzer:
    """
    Provides analysis capabilities for FVCOM datasets.
    """
    def __init__(self, dataset, zone=54, north=True):
        self.ds = dataset
        self.zone = zone
        self.north = north

    def get_variables_by_dims(self, dims):
        """
        Returns a list of variable names that have the specified dimensions.

        Parameters:
        - dims: A tuple of dimensions to filter by (e.g., ('time', 'node')).

        Returns:
        - A list of variable names matching the specified dimensions.
        """
        variables = [
            var_name for var_name, var in self.ds.variables.items() if var.dims == dims
        ]
        return variables
        
    def nearest_neighbor(self, lon, lat, node=True, distances=False):
        """
        Find the nearest node or element to the specified coordinates.
        """
        A = self._lonlat_to_xy(lon, lat, inverse=True)
        if node:
            points = np.column_stack((self.ds.x.values, self.ds.y.values))
        else:
            points = np.column_stack((self.ds.xc.values, self.ds.yc.values))

        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(points)
        if distances:
            return nn.kneighbors([A])
        return nn.kneighbors([A])[1].item()

    def _lonlat_to_xy(self, lon, lat, inverse=False):
        """
        Convert geographic coordinates to UTM or vice versa.
        """
        crs_from = f"+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        crs_to = f"+proj=utm +zone={self.zone} {'+north' if self.north else ''} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        transformer = pyproj.Transformer.from_crs(crs_to, crs_from) if inverse else pyproj.Transformer.from_crs(crs_from, crs_to)
        return transformer.transform(lon, lat)

class FvcomPlotConfig:
    """
    Stores plot configuration settings.
    """
    def __init__(self, figsize=(8,2), width=800, height=200, fontsize=None, dpi=300, **kwargs):
        self.figsize = figsize
        self.width = width
        self.height = height
        self.fontsize = fontsize or {'xticks': 11, 'yticks': 11, 'xlabel': 12, 'ylabel': 12}
        self.dpi = dpi
        self.plot_color = kwargs.get("plot_color", "red")
        self.date_format = kwargs.get('date_format', '%Y-%m-%d')
        # Arrow configuration with defaults provided dynamically
        defaults = {
            "arrow_scale": 1.0,
            "arrow_scale": 1.0,
            "arrow_width": 0.002,
            "arrow_color": "blue",
            "arrow_alpha": 0.7,
            "arrow_angles": "uv",
            "arrow_headlength": 5,
            "arrow_headwidth": 3,
            "arrow_headaxislength": 4.5
        }
        for key, value in defaults.items():
            setattr(self, key, kwargs.get(key, value))

class FvcomPlotter(PlotHelperMixin):
    """
    Creates plots from FVCOM datasets.
    """
    def __init__(self, dataset, plot_config):
        self.ds = dataset
        self.cfg = plot_config

    def plot_time_series(self, var_name, index, start=None, end=None, rolling_window=None,
                         ax=None, save_path=None, **kwargs):
        """
        Plot a time series for a specified variable at a given node or element index.

        Parameters:
        - var_name: Name of the variable to plot.
        - index: Index of the `node` or `nele` to plot.
        - dim: Dimension to use ('node' or 'nele' or nobc).
        - start: Start time for the plot (datetime or string).
        - end: End time for the plot (datetime or string).
        - rolling_window: Size of the rolling window for moving average (optional).
        - ax: matplotlib axis object. If None, a new axis will be created.
        - save_path: Path to save the plot as an image (optional).
        - **kwargs: Additional arguments for customization (e.g., dpi, figsize).
        """
        if var_name not in self.ds:
            print(f"Error: the variable '{var_name}' is not found in the dataset.")
            return None
    
        # Validate the dimension
        variable_dims = self.ds[var_name].dims
        if "node" in variable_dims:
            dim = "node"
        elif "nele" in variable_dims:
            dim = "nele"
        elif "nobc" in variable_dims:
            dim = "nobc"
        else:
            raise ValueError(f"Variable {var_name} does not have 'node' or 'nele' as a dimension.")
        
        # Select the data
        data = self.ds[var_name].isel({dim: index})
        # Apply rolling mean if specified
        if rolling_window:
            data = data.rolling(time=rolling_window, center=True).mean()

        # Time range filtering
        time = self.ds["time"]
        if start:
            start = np.datetime64(start)
        if end:
            end = np.datetime64(end)
        time_mask = (time >= start) & (time <= end) if start and end else slice(None)
        data = data.isel(time=time_mask)
        time = time[time_mask]

        # If no axis is provided, create a new one
        if ax is None:
            figsize = kwargs.get("figsize", self.cfg.figsize)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure  # Get the figure from the provided axis
        # Plotting
        color = kwargs.get('color', self.cfg.plot_color)
        ax.plot(time, data, label=f"{var_name} ({dim}={index})", color=color, **kwargs)

        # Formatting
        rolling_text = f" with {rolling_window}-hour Rolling Mean" if rolling_window else ""
        title = f"Time Series of {var_name} ({dim}={index}){rolling_text}"
        ax.set_title(title, fontsize=self.cfg.fontsize['xlabel'])
        ax.set_xlabel("Time", fontsize=self.cfg.fontsize['xlabel'])
        ax.set_ylabel(var_name, fontsize=self.cfg.fontsize['ylabel'])
        date_format = kwargs.get('date_format', self.cfg.date_format)
        ax.xaxis.set_major_formatter(DateFormatter(date_format))
        fig.autofmt_xdate()
        ax.grid(True)
        ax.legend()

        # Save or show the plot
        if save_path:
            dpi = kwargs.get("dpi", self.cfg.dpi)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        
        return ax


    def hvplot_time_series(self, var, siglay=None, node=None, **kwargs):
        """
        Plot a time series for the specified variable.
        """
        da = self.ds[var]
        return da[:, siglay, node].hvplot(
            x="time",
            width=self.cfg.width,
            height=self.cfg.height,
            fontsize=self.cfg.fontsize,
            **kwargs
        )

    def plot_time_series_for_river(self, var_name, river_index, start=None, end=None, rolling_window=None,
                                   ax=None, save_path=None, **kwargs):
        """
        Plot a time series for a specified variable at a given river index.

        Parameters:
        - var_name: Name of the variable to plot.
        - river_index: Index of the `rivers` to plot.
        - start: Start time for the plot (datetime or string).
        - end: End time for the plot (datetime or string).
        - rolling_window: Size of the rolling window for moving average (optional).
        - ax: matplotlib axis object. If None, a new axis will be created.
        - save_path: Path to save the plot as an image (optional).
        - **kwargs: Additional arguments for customization (e.g., dpi, figsize).
        """
        if var_name not in self.ds:
            print(f"Error: the variable '{var_name}' is not found in the dataset.")
            return None

        # Validate the dimensions of the variable
        variable_dims = self.ds[var_name].dims
        if "rivers" not in variable_dims or "time" not in variable_dims:
            raise ValueError(f"Variable {var_name} does not have 'rivers' and 'time' as dimensions.")

        # Retrieve and clean river name
        if "river_names" not in self.ds:
            raise ValueError("Dataset does not contain 'river_names' variable for labeling rivers.")
        river_name = self.ds["river_names"].isel(rivers=river_index).values
        if isinstance(river_name, np.ndarray):
            river_name = river_name.item()  # 単一値を取得
        river_name = river_name.decode('utf-8').strip() 


        # Select the data
        data = self.ds[var_name].isel(rivers=river_index)
        # Apply rolling mean if specified
        if rolling_window:
            data = data.rolling(time=rolling_window, center=True).mean()

        # Time range filtering
        time = self.ds["time"]
        if start:
            start = np.datetime64(start)
        if end:
            end = np.datetime64(end)
        time_mask = (time >= start) & (time <= end) if start and end else slice(None)
        data = data.isel(time=time_mask)
        time = time[time_mask]

        # If no axis is provided, create a new one
        if ax is None:
            figsize = kwargs.get("figsize", self.cfg.figsize)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure  # Get the figure from the provided axis

        # Plotting
        color = kwargs.get('color', self.cfg.plot_color)
        ax.plot(time, data, label=f"{var_name} (river={river_index})", color=color, **kwargs)

        # Formatting
        rolling_text = f" with {rolling_window}-hour Rolling Mean" if rolling_window else ""
        title = f"Time Series of {var_name} for {river_name} (river={river_index}){rolling_text}"
        ax.set_title(title, fontsize=self.cfg.fontsize['xlabel'])
        ax.set_xlabel("Time", fontsize=self.cfg.fontsize['xlabel'])
        ax.set_ylabel(var_name, fontsize=self.cfg.fontsize['ylabel'])
        date_format = kwargs.get('date_format', self.cfg.date_format)
        ax.xaxis.set_major_formatter(DateFormatter(date_format))
        fig.autofmt_xdate()
        ax.grid(True)
        ax.legend()

        # Save or show the plot
        if save_path:
            dpi = kwargs.get("dpi", self.cfg.dpi)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return ax

    def plot_wind_vector_timeseries(self, u_var="uwind_speed", v_var="vwind_speed", nele=None, start=None, end=None,
                                    rolling_window=None, save_path=None, plot_wind_speed=True, **kwargs):
        """
        Plot wind vector time series for a specific element.

        Parameters:
        - u_var: Name of the variable representing the u-component of the wind.
        - v_var: Name of the variable representing the v-component of the wind.
        - nele: Element index to plot the data for.
        - start: Start time for the period to plot (e.g., "2020-01-01 00:00:00").
        - end: End time for the period to plot (e.g., "2020-12-31 23:59:59").
        - rolling_window: Size of the rolling window for moving average (optional).        - save_path: Path to save the plot as an image. If None, the plot is displayed.
        - **kwargs: Additional keyword arguments for customization (e.g., dpi).
        """
        if u_var not in self.ds or v_var not in self.ds:
            print(f"Error: One or both of the variables '{u_var}' and '{v_var}' are not found in the dataset.")
            return None
        
        # Select the data
        if nele is not None:
            if "nele" not in self.ds[u_var].dims:
                print(f"Error: The variable '{u_var}' does not have 'nele' as a dimension. Try with 'nele=None'.")
                return None
            else:
                u_data = self.ds[u_var].isel(nele=nele)
                v_data = self.ds[v_var].isel(nele=nele)
        else:
            u_data = self.ds[u_var]
            v_data = self.ds[v_var]

        # Apply rolling mean if specified
        if rolling_window:
            u_data = u_data.rolling(time=rolling_window, center=True).mean().dropna(dim="time")
            v_data = v_data.rolling(time=rolling_window, center=True).mean().dropna(dim="time")
            # Ensure time alignment after rolling and dropna
            time = u_data["time"]
        else:
            time = self.ds["time"]

        if start:
            start = np.datetime64(start)
        if end:
            end = np.datetime64(end)
        time_mask = (time >= start) & (time <= end) if start and end else slice(None)
        u = u_data.isel(time=time_mask).values
        v = v_data.isel(time=time_mask).values
        time = time[time_mask]

        # Compute wind speed magnitude
        speed = np.sqrt(u**2 + v**2)

        # Adjust scale for quiver plot
        max_speed = np.max(speed)
        #scale_factor = max_speed / 10  # Adjust to fit arrows within the plot

        figsize = kwargs.get('figsize', self.cfg.figsize)
        fig, ax = plt.subplots(figsize=figsize)

        # Plot wind speed magnitude
        if plot_wind_speed:
            ax.plot(time, speed, label="Wind Speed (m/s)", color=self.cfg.plot_color, alpha=0.5)

        # Quiver plot for wind vectors
        quiver_kwargs = {key: value for key, value in kwargs.items() if key not in ["rolling_window", "figsize", "dpi"]}
        angles = kwargs.get('angles', self.cfg.arrow_angles)
        headlength = kwargs.get('headlength', self.cfg.arrow_headlength)
        headwidth = kwargs.get('headwidth', self.cfg.arrow_headwidth)
        headaxislength = kwargs.get('headaxislength', self.cfg.arrow_headaxislength)
        width = kwargs.get('width', self.cfg.arrow_width)
        scale = kwargs.get('scale', self.cfg.arrow_scale)
        color = kwargs.get('color', self.cfg.arrow_color)
        alpha = kwargs.get('alpha', self.cfg.arrow_alpha)
        date_format = kwargs.get('date_format', self.cfg.date_format)

        ax.quiver(
            time, [0] * len(time), u, v, angles=angles, headlength=headlength, headwidth=headwidth,
            headaxislength=headaxislength, width=width, scale_units="y", scale=scale, 
            color=color, alpha=alpha, **quiver_kwargs
        )

        # Format y-axis to accommodate negative and positive values
        max_v = np.max(np.abs(max_speed))
        ax.set_ylim(-max_v, max_v)

        # Format axes
        rolling_text = f" with {rolling_window}-hour Rolling Mean" if rolling_window else ""
        nele_text = f" (nele={nele})" if nele is not None else ""
        title = f"Wind Vector and Speed Time Series {nele_text}{rolling_text}"
        ax.set_title(title, fontsize=self.cfg.fontsize['xlabel'])
        ax.set_xlabel("Time", fontsize=self.cfg.fontsize['xlabel'])
        ax.set_ylabel("Wind Speed (m/s)", fontsize=self.cfg.fontsize['ylabel'])
        ax.xaxis.set_major_formatter(DateFormatter(date_format))
        fig.autofmt_xdate()
        ax.grid(True)
        ax.legend()

        if save_path:
            dpi = kwargs.get('dpi', self.cfg.dpi)  # Use provided dpi or default to config dpi
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        
        return ax

# Example usage
if __name__ == "__main__":
    # Load data
    loader = FvcomDataLoader(dirpath="/path/to/data", ncfile="sample.nc")
    
    # Analyze
    analyzer = FvcomAnalyzer(loader.ds)
    nearest_node = analyzer.nearest_neighbor(lon=140.0, lat=35.0)

    # Plot
    plot_config = FvcomPlotConfig(width=1000, height=400)
    plotter = FvcomPlotter(loader.ds, plot_config)
    plot = plotter.plot_time_series("z", siglay=0, node=nearest_node)
    print(plot)

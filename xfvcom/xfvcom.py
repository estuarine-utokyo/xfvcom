# xfvcom.py: A Python module for loading, analyzing, and plotting FVCOM model output data in xfvcom package.
# Author: Jun Sasaki
import os
import numpy as np
import xarray as xr
import pandas as pd
import pyproj
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from datetime import datetime
from matplotlib.dates import DateFormatter
import matplotlib.cm as cm
from matplotlib.colors import Normalize, BoundaryNorm
import matplotlib.colors as mcolors
import matplotlib.tri as tri
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogLocator, LogFormatter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#from cartopy.io.img_tiles import StadiaMapsTiles
#from cartopy.io.img_tiles import Stamen
import cartopy.io.img_tiles as cimgt
from cartopy.io.img_tiles import GoogleTiles

from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import inspect
import matplotlib.tri as mtri
from scipy.spatial import KDTree
import pyproj
from .helpers import PlotHelperMixin

class FvcomDataLoader:
    """
    Responsible for loading FVCOM output NetCDF files into an xarray.Dataset.
    """
    # def __init__(self, base_path=None, ncfile=None, obcfile_path=None,
    def __init__(self, ncfile_path=None, obcfile_path=None,
                 engine="netcdf4", chunks=None,
                 utm2geo=True, zone=54, north=True,
                 inverse=False, time_tolerance=None, verbose=False, **kwargs):
        """
        Initialize the FvcomDataLoader instance.
        
        Parameters:
        #- base_path: Directory path where the NetCDF file is located.
        #- ncfile: Name of the NetCDF file to load.
        - ncfile_path: Netcdf file path
        - obcfile_path: Path to the open boundary node file.
        - engine: {"netcdf4", "scipy", "pydap", "h5netcdf", "zarr", None}, installed backend.
        - chunks: Chunk size for dask array. Default is "auto".  
        - utm2geo: Convert UTM coordinates to geographic (lon, lat).
        - zone: UTM zone number.
        - north: True if the UTM zone is in the northern hemisphere.
        - inverse: True to convert geographic coordinates to UTM.
        - time_tolerence: Tolerence in minutes in integer to snap time to the nearest hour.
        - **kwargs: Additional keyword arguments for xarray.open_dataset.
        """
        # base_path = os.path.expanduser(base_path) if base_path else None
        # base_path = self._add_trailing_slash(base_path) if base_path else None
        # self.ncfilepath = f"{base_path}{ncfile}" if base_path else ncfile
        self.ncfile_path = ncfile_path
        self.engine = engine
        self.chunks = chunks
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
        if 'nv' in self.ds:
            #self.ds['nv_zero'] = xr.DataArray(self.ds['nv'].values.T - 1)
            #self.ds['nv_zero'].attrs['long_name'] = 'nodes surrounding element in zero-based for matplotlib'
            self._setup_nv_ccw()
        
        # ERSEM O2 concentration conversion to mg/L
        if "O2_o" in self.ds.data_vars:
            # Keep the attributes before conversion, or the attributes will be lost.
            attrs = self.ds["O2_o"].attrs.copy()
            self.ds["O2_o"] = self.ds["O2_o"] * 0.032
            # Restore the attributes after conversion.
            attrs["units"] = "mg/L"
            attrs["long_name"] = r"O$_2$"
            self.ds["O2_o"].attrs = attrs
        
        # Load FVCOM open boundary node if provided
        if obcfile_path:
            if os.path.isfile(obcfile_path):
                #print(f"Loading open boundary nodes from {obcfile_path}")
                df = pd.read_csv(obcfile_path, header=None, skiprows=1, delim_whitespace=True)
                node_bc = df.iloc[:,1].values - 1
                if verbose:
                    print(f"{node_bc}")
                self.ds['node_bc'] = xr.DataArray(node_bc, dims=("obc_node"))
                self.ds['node_bc'].attrs['long_name'] = 'open boundary nodes'
                print(f"Open boundary nodes loaded successfully from {obcfile_path}")
                #print(self.ds.node_bc.values)

    def _setup_nv_ccw(self):
        """
        Set up the counterclockwise nv_ccw.
        """
        self.ds['nv_zero'] = xr.DataArray(self.ds['nv'].values.T - 1)
        self.ds['nv_zero'].attrs['long_name'] = 'nodes surrounding element in zero-based for matplotlib'
        # Extract triangle connectivity
        nv_ccw = self.ds["nv"].values.T - 1
        nv_ccw = self.ds['nv_zero'].values
        # Reverse node order for counter-clockwise triangles that matplotlib expects.
        nv_ccw = nv_ccw[:, ::-1]
        # print(nv_ccw.shape)
        self.ds['nv_ccw'] = xr.DataArray(nv_ccw, dims=("nele", "three"))
        self.ds['nv_ccw'].attrs['long_name'] = 'nodes surrounding element in unti-clockwise direction for matplotlib'

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
            ds = xr.open_dataset(self.ncfile_path, engine=self.engine, chunks=self.chunks, decode_times=self.decode_times)
            ds = ds.drop_vars('Itime2') if 'Itime2' in ds.variables else ds
            print(f"Dataset loaded successfully from {self.ncfile_path}")
            return xr.decode_cf(ds)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.ncfile_path}")

    def _convert_utm_to_geo(self):
        """Convert UTM coordinates (x, y) and (xc, yc) to geographic (lon, lat)."""
        lon, lat = self._xy_to_lonlat(self.ds.x.values, self.ds.y.values)
        lonc, latc = self._xy_to_lonlat(self.ds.xc.values, self.ds.yc.values)

        self.ds["lon"] = xr.DataArray(lon, dims="node")
        self.ds["lat"] = xr.DataArray(lat, dims="node")
        self.ds["lonc"] = xr.DataArray(lonc, dims="nele")
        self.ds["latc"] = xr.DataArray(latc, dims="nele")

    def _add_depth_variables(self):
        """Add 'z' and 'z_dfs' depth variables to the dataset. `z_dfs` is depth from the surface in positive."""
        try:
            ## The following is too slow and revised using numpy's broadcasting as follows.
            # self.ds['z'] = (("siglay", "time", "node" ), np.array([self.ds.zeta + siglay * (self.ds.h + self.# ds.zeta) for siglay in self.ds.siglay]))
            # self.ds['z_dfs'] = self.ds.zeta - self.ds.z
            # self.ds['z'] = self.ds['z'].transpose("time", "siglay", "node")
            # self.ds['z_dfs'] = self.ds['z_dfs'].transpose("time", "siglay", "node")
            ## Revised for speeding up using numpy's broadcasing
            time_size = self.ds.time.shape[0]
            siglay_size = self.ds.siglay.shape[0]
            node_size = self.ds.node.shape[0]
            # h を numpy 配列として取得し、形状を (1, 1, node) にブロードキャスト
            # Obtain h as a numpy array and broadcast the shape (1, 1, node).
            h_np = self.ds.h.values[np.newaxis, np.newaxis, :]
            h_broadcasted = np.broadcast_to(h_np, (time_size, siglay_size, node_size))
            # siglay を numpy 配列として取得し、形状を (siglay, 1, node) にブロードキャストしてから、(time, siglay, node) にリシェイプ
            # Obtain siglay as a numpy array and broadcast the shape (siglay, 1, node) and reshape to (time, siglay, node).
            siglay_np = self.ds.siglay.values[:, np.newaxis, :]
            siglay_broadcasted = np.broadcast_to(siglay_np, (siglay_size, time_size, node_size)).transpose(1, 0, 2)
            # zeta はすでに (time, node) の形状なので、そのまま使用
            # zeta is already in the shape of (time, node), so use it as is.
            zeta_np = self.ds.zeta.values
            # 計算実行: zeta + siglay * (h + zeta)、ここで適切にブロードキャストを使用
            # Execute the calculation: zeta + siglay * (h + zeta) using broadcasting appropriately.
            result = zeta_np[:, np.newaxis, :] + siglay_broadcasted * (h_broadcasted + zeta_np[:, np.newaxis, :])
            # 新しい DataArray を作成
            # Create a new DataArray.
            # new_da_corrected = xr.DataArray(result, dims=("time", "siglay", "node"))
            self.ds['z'] = xr.DataArray(result, dims=("time", "siglay", "node"))
            self.ds['z_dfs'] = (self.ds.zeta - self.ds.z).transpose("time", "siglay", "node")
            ## Add attributes
            self.ds['z'].attrs['long_name'] = 'Depth'
            self.ds['z'].attrs['standard_name'] = 'Depth at siglay'
            self.ds['z'].attrs['units'] = 'm'
            self.ds['z'].attrs['positive'] = 'up'
            self.ds['z'].attrs['origin'] = 'still water lavel'
            self.ds['z_dfs'].attrs['long_name'] = 'Depth'
            self.ds['z_dfs'].attrs['standard_name'] = 'Depth at siglay from the surface'
            self.ds['z_dfs'].attrs['units'] = 'm'
            self.ds['z_dfs'].attrs['positive'] = 'down'
            self.ds['z_dfs'].attrs['origin'] = 'surface'
        except Exception as e:
            raise ValueError(f"Error in adding depth variables: {e}")

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
        # Convert time to datetime64
        time = self.ds['time'].values.astype('datetime64[s]')
        #time = self.ds['time'].values
        # 誤差範囲内で正時にスナップ
        # Snap to the nearest hour within the tolerance range
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

        Parameters:
        - lon: Longitude of the target point.
        - lat: Latitude of the target point.
        - node: If True, search among nodes. If False, search among elements.
        - distances: If True, return both distance and index of the nearest neighbor.

        Returns:
        - Index of the nearest neighbor (and optionally the distance).
        """
        # Convert geographic (lon, lat) to UTM (x, y)
        target_coords = np.array(self._lonlat_to_xy(lon, lat, inverse=False)).reshape(1, -1)
        
        # Get search points in UTM (x, y)
        if node:
            points = np.column_stack((self.ds.x.values, self.ds.y.values))
        else:
            points = np.column_stack((self.ds.xc.values, self.ds.yc.values))
        
        # Ensure there are no NaN values in the search points
        valid_mask = ~np.isnan(points).any(axis=1)
        points = points[valid_mask]
        
        # Perform nearest-neighbor search
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(points)
        distances_array, indices_array = nn.kneighbors(target_coords)
        
        # Map the result back to the original indices
        nearest_index = np.where(valid_mask)[0][indices_array[0, 0]]
        
        if distances:
            return distances_array[0, 0], nearest_index
        return nearest_index

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
    def __init__(self, figsize=(8,2), width=800, height=200, dpi=100, 
                 cbar_size="1%", cbar_pad=0.1, cmap="jet", levels=21,
                 title_fontsize=14, label_fontsize=12, tick_fontsize=11, fontsize=None, **kwargs):
        """
        Initialize the FvcomPlotConfig instance.
        Parameters:
        - figsize: Figure size in inches (width, height).
        - width: Width of the plot in pixels.
        - height: Height of the plot in pixels.
        - dpi: Dots per inch for the plot.
        - cbar_size: Size of the colorbar.
        - cbar_pad: Padding between the colorbar and the plot.
        - cmap: Colormap to use for the plot.
        - levels: Number of levels for the colormap.
        - fontsize: Font size settings for various plot elements.
        - **kwargs: Additional keyword arguments for customization.
        """
        self.figsize = figsize
        self.width = width
        self.height = height
        self.dpi = dpi
        self.cbar_size = cbar_size
        self.cbar_pad = cbar_pad
        self.cmap = cmap
        self.levels = levels
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        # Default font sizes for various plot elements
        default_fontsize = {
            'xticks': 11, 'yticks': 11, 'xlabel': 12, 'ylabel': 12, 'title': 14,
            'legend': 12, 'annotation': 12, 'colorbar': 11, 'tick_params': 11,
            'text': 12, 'legend_title': 12, 'cbar_title': 14, 'cbar_label': 14
        }
        self.fontsize = {**default_fontsize, **(fontsize or {})}
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
        """
        Initialize the FvcomPlotter instance.

        Parameters:
        - dataset: An xarray.Dataset object containing FVCOM model output or input.
        - plot_config: An instance of FvcomPlotConfig with plot configuration settings.
        """
        self.ds = dataset
        self.cfg = plot_config

    def plot_timeseries(self, var_name, index, log=False, k=None, start=None, end=None, rolling_window=None,
                         ax=None, save_path=None, **kwargs):
        """
        Plot a time series for a specified variable at a given node or element index.

        Parameters:
        - var_name: Name of the variable to plot.
        - index: Index of the `node` or `nele` to plot.
        - k: Vertical layer index for 3D variables (optional).
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
        
        variable_dims = self.ds[var_name].dims
        if "siglay" in variable_dims:
            dimk = "siglay"
        elif "siglev" in variable_dims:
            dimk = "siglev"
        else:
            if k is not None:
                raise ValueError(f"Variable {var_name} does not have 'siglay' or 'siglev' as a dimension.")
        
        # Select the data
        if k is not None:
            data = self.ds[var_name].isel({dim: index, dimk: k})
        else:
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

        if log: # Check if log scale is requested
            if data.min() <= 0:
                print("Warning: Logarithmic scale cannot be used with non-positive values.")
                print("Switching to linear scale.")
                log = False

        # If no axis is provided, create a new one
        if ax is None:
            figsize = kwargs.get("figsize", self.cfg.figsize)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure  # Get the figure from the provided axis
        # Plotting
        color = kwargs.pop('color', self.cfg.plot_color)
        if k is not None:
            label = f"{var_name} ({dim}={index}, {dimk}={k})"
        else:
            label = f"{var_name} ({dim}={index})"
        ax.plot(time, data, label=label, color=color, **kwargs)
        if log: # Set log scale if specified
            ax.set_yscale('log')

        # Formatting
        rolling_text = f" with {rolling_window}-hour Rolling Mean" if rolling_window else ""
        if k is not None:
            title = f"Time Series of {var_name} ({dim}={index}, {dimk}={k}){rolling_text}"
        else:
            title = f"Time Series of {var_name} ({dim}={index}){rolling_text}"
        ax.set_title(title, fontsize=self.cfg.fontsize['title'])
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

    def plot_timeseries_for_river(self, var_name, river_index, start=None, end=None, rolling_window=None,
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
        #color = kwargs.get('color', self.cfg.plot_color)
        color = kwargs.pop('color', self.cfg.plot_color)
        ax.plot(time, data, label=f"{var_name} (river={river_index})", color=color, **kwargs)

        # Formatting
        rolling_text = f" with {rolling_window}-hour Rolling Mean" if rolling_window else ""
        title = f"Time Series of {var_name} for {river_name} (river={river_index}){rolling_text}"
        ax.set_title(title, fontsize=self.cfg.fontsize['title'])
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
        ax.set_title(title, fontsize=self.cfg.fontsize['title'])
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

    def ts_contourf_z(self, da: xr.DataArray, index: int = None, 
                      xlim: tuple = None, ylim: tuple = None,
                      xlabel: str = "Time", ylabel: str = "Depth (m)", title: str = None,
                      rolling_window: int = None, ax=None, cmap=None, label=None,
                      contourf_kwargs: dict = None, colorbar_kwargs: dict = None,
                      plot_surface: bool = False,
                      surface_kwargs: dict | None = None,
                      **kwargs) -> tuple[plt.Figure, plt.Axes, Colorbar]:
        """
        Plot a 2D time-series contour (time vs depth) for the specified variable.
        This method is specialized for z-coordinate (depth) data and does not support sigma coordinates.
        Parameters:
        - da (xarray.DataArray): The DataArray to plot.
        - index (int): Index of the node/nele for spatial dimension (required if data has 'node' or 'nele' dim).
        - xlim (tuple): (start_time, end_time) for the x-axis time range.
        - ylim (tuple): Depth range for the y-axis (e.g., (0, 100)) in meters.
        - xlabel, ylabel (str): Axis labels for time and depth.
        - title (str): Plot title. If None, a default title is generated.
        - rolling_window (int): Window size (in time steps) for centered rolling mean smoothing.
        - ax (matplotlib.axes.Axes): Axis to plot on. If None, a new figure and axis are created.
        - cmap: Colormap for the contour. Uses default from config if None.
        - label (str): Label for the colorbar (overrides variable long_name/units if provided).
        - contourf_kwargs (dict): Additional keyword arguments for contourf.
        - colorbar_kwargs (dict): Additional keyword arguments for colorbar.
        - plot_surface (bool): If True, plot the surface elevation on top of the contour.
        - surface_kwargs (dict): Additional keyword arguments for surface plot.
        - **kwargs: Additional contourf keyword arguments (levels, etc.).
        Returns:
        - (fig, ax, cbar): Figure, Axes, and Colorbar objects for the plot.
        """
        # 1. Verify da has the required dimensions
        if "time" not in da.dims:
            raise ValueError(f"DataArray must have 'time' dimension, got {da.dims}")

        # 2. Handle spatial dimension (node/nele) – require index if present
        spatial_dim = next((d for d in ("node", "nele") if d in da.dims), None)
        if spatial_dim:
            if index is None:
                raise ValueError(f"Index must be provided for spatial dimension '{spatial_dim}'.")
            da = da.isel({spatial_dim: index})
        elif index is not None:
            raise ValueError(f"No spatial dimension in '{var_name}', but index was provided.")

        # 3. Verify vertical (sigma) dimension is present (required for depth plot)
        vertical_dim = "siglay" if "siglay" in da.dims else ("siglev" if "siglev" in da.dims else None)
        if vertical_dim is None:
            raise ValueError(f"Variable '{var_name}' has no sigma layer dimension ('siglay' or 'siglev'), cannot plot depth profile.")

        # 4. Apply rolling mean on time axis if specified
        da = self._apply_rolling(da, rolling_window)  # uses centered rolling mean&#8203;:contentReference[oaicite:0]{index=0}

        # 5. Filter data by time range (xlim) if provided
        da = self._apply_time_filter(da, xlim)        # uses start/end from xlim to slice time&#8203;:contentReference[oaicite:1]{index=1}

        # 6. Prepare depth (z) values for the selected location and times
        #    We assume the dataset contains 'z' (depth) with same dims (time, vertical, spatial)
        if spatial_dim:
            z_da = self.ds["z"].isel({spatial_dim: index})
        else:
            z_da = self.ds["z"]
        z_da = self._apply_time_filter(z_da, xlim)
        # Align dimensions ordering for consistent (time, vertical) shape
        z_da = z_da.transpose("time", vertical_dim)
        da   = da.transpose("time", vertical_dim)
        # Assign depth values as a 2D coordinate for the DataArray (for plotting)
        da.coords["Depth"] = (("time", vertical_dim), z_da.values)
        
        # 7. Create figure and axis if not provided
        if ax is None:
            fig = plt.figure(figsize=self.cfg.figsize, dpi=self.cfg.dpi)
            ax = fig.add_subplot(1, 1, 1)
        else:
            fig = ax.figure

        # 8. Determine contourf parameters (levels, cmap, etc.), merging **kwargs 
        #    and using defaults from config when not specified
        if cmap is not None:
            kwargs["cmap"] = cmap
        merged_cf_kwargs, levels, cmap_used, vmin, vmax, extend = \
            self._prepare_contourf_args(da, None, kwargs)  # unify contour args&#8203;:contentReference[oaicite:2]{index=2}

        # 9. Plot the filled contour using time vs depth
        contour = da.plot.contourf(x="time", y="Depth", levels=levels, cmap=cmap_used,
                                vmin=vmin, vmax=vmax, extend=extend, ax=ax,
                                add_colorbar=False, **merged_cf_kwargs)

        # Optionally add contour lines on top (if desired, similar to original add_contour logic)
        # Example:
        # if kwargs.get("add_contour"):
        #     cs = ax.contour(da["time"].values, da.coords["Depth"].values, da.values,
        #                     levels=levels, colors="k", linewidths=0.5)
        #     if kwargs.get("label_contours"):
        #         ax.clabel(cs, inline=True, fontsize=8)

        # 10. Optional: plot water surface elevation line
        #       surface_kwargs: dict passed to ax.plot (e.g. color, linewidth)
        if plot_surface:
            skw = surface_kwargs or {}
            # zeta を取得し、同じ spatial_dim,index で抽出
            surf = self.ds["zeta"]
            if spatial_dim:
                surf = surf.isel({spatial_dim: index})
            # 時間フィルタ
            surf = self._apply_time_filter(surf, xlim)
            # プロット
            ax.plot(surf["time"], surf.values, **skw)

        # 11. Invert y-axis so that depth=0 is at the top
        # ax.invert_yaxis()

        # 12. Set axis labels, title, and format the time axis
        # Construct default title if none provided
        if title is None:
            long_name = da.attrs.get("long_name", da.name)
            rolling_text = f" with {rolling_window}-step Rolling Mean" if rolling_window else ""
            title = (f"Time Series of {long_name}" +
                    (f" ({spatial_dim}={index})" if spatial_dim else "") +
                    f"{rolling_text}")
        self._format_time_axis(ax, title, xlabel, ylabel, self.cfg.date_format)

        # Apply depth limits if provided
        if ylim is not None:
            ax.set_ylim(ylim)

        # 13. Create and attach colorbar
        # Determine colorbar label from variable metadata or provided `label`
        units = da.attrs.get("units", "")
        cbar_label = label if label is not None else (
            f"{long_name} ({units})" if units else long_name
        )
        cbar = self._make_colorbar(ax, contour, cbar_label, colorbar_kwargs or {})
        return fig, ax, cbar


    def plot_timeseries_2d(self, var_name, index=None, start=None, end=None, depth=False, rolling_window=None, ax=None, 
                                   ylim=None, levels=20, vmin=None, vmax=None, cmap=None, save_path=None, method='contourf',
                                   add_contour=False, label_contours=False, **kwargs):
        """
        Obsolete. Remove this method in future versions.
        Plot a 2D time series for a specified variable as a contour map with time on the x-axis and a vertical coordinate (siglay/siglev) on the y-axis.

        Parameters:
        - var_name: Name of the variable to plot.
        - index: Index of the `node` or `nele` to plot (default: None).
        - start: Start time for the plot (datetime or string).
        - end: End time for the plot (datetime or string).
        - depth: If True, plot depth instead of vertical coordinate.
        - rolling_window: Size of the rolling window for moving average (optional).
        - ax: matplotlib axis object. If None, a new axis will be created.
        - ylim: Y-axis limits (optional).
        - levels: Number of contour levels or specific levels (optional).
        - vmin: Minimum value for color scale (optional).
        - vmax: Maximum value for color scale (optional).
        - cmap: Color map for the plot (optional).
        - save_path: Path to save the plot as an image (optional).
        - method: Plotting method ('contourf' or 'pcolormesh').
        - add_contour: If True, add contour lines on top of the filled contour.
        - label_contours: If True, label contour lines.
        - **kwargs: Additional arguments for customization (e.g., levels).
        """
        if var_name not in self.ds:
            print(f"Error: The variable '{var_name}' is not found in the dataset.")
            return None

        # Auto-detect the vertical coordinate
        if "siglay" in self.ds[var_name].dims:
            y_coord = "siglay"
        elif "siglev" in self.ds[var_name].dims:
            y_coord = "siglev"
        else:
            raise ValueError(f"Variable {var_name} does not have 'siglay' or 'siglev' as a vertical coordinate.")

        # Validate the variable's dimensions
        if "time" not in self.ds[var_name].dims:
            raise ValueError(f"Variable {var_name} does not have 'time' as a dimension.")

        # Select the data for the specified index
        # Validate the dimension
        variable_dims = self.ds[var_name].dims
        if "node" in variable_dims:
            dim = "node"
        elif "nele" in variable_dims:
            dim = "nele"
        else:
            raise ValueError(f"Variable {var_name} does not have 'node' or 'nele' as a dimension.")

        data = self.ds[var_name]
        if index is not None:
            data = data.isel({dim: index})
        else:
            raise ValueError("Index for 'node' or 'nele' must be provided.")

        # Apply rolling mean if specified
        if rolling_window:
            data = data.rolling(time=rolling_window, center=True).mean()
        
        # Time range filtering
        time = data["time"]
        if start:
            start = np.datetime64(start)
        if end:
            end = np.datetime64(end)
        time_mask = (time >= start) & (time <= end) if start and end else slice(None)
        data = data.isel(time=time_mask)
        # Transpose data to ensure (Ny, Nx) shape
        data = data.transpose(y_coord, "time")

        # Extract time, vertical coordinate, and data values
        time_vals = data["time"].values  # (Nx,)
        y_vals = data[y_coord].values  # (Ny,)
        values = data.values  # (Ny, Nx)

        # Ensure data dimensions are correct
        if values.shape != (len(y_vals), len(time_vals)):
            raise ValueError(
                f"Shape mismatch: data={values.shape}, time={len(time_vals)}, vertical={len(y_vals)}"
            )

        # Create 2D grid for time and vertical coordinate
        time_grid, y_grid = np.meshgrid(time_vals, y_vals, indexing="xy")
        if depth:
            z = self.ds.z.isel(time=time_mask)[:,:,index].T.values
            if z.shape != (len(y_vals), len(time_vals)):
                raise ValueError(
                    f"Shape mismatch: depth={z.shape}, time={len(time_vals)}, vertical={len(y_vals)}"
                )
            else:
                y_grid = z
                y_coord = "Depth (m)"

        # Create a new axis if not provided
        if ax is None:
            figsize = kwargs.get("figsize", self.cfg.figsize)
            fig, ax = plt.subplots(figsize=figsize)

        # Plot using contourf or pcolormesh
        cmap = cmap or kwargs.pop("cmap", "viridis") 
        auto_levels = True
        if vmin is not None or vmax is not None:
            auto_levels = False
        vmin = vmin or kwargs.pop("vmin", values.min())
        vmax = vmax or kwargs.pop("vmax", values.max())
        levels = levels or kwargs.get("levels", 20)  # Number of contour levels
        if isinstance(levels, int):
            levels = np.linspace(vmin, vmax, levels)
        elif isinstance(levels, (list, np.ndarray)):
            levels = np.array(levels)
            auto_levels = False
        if auto_levels:
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = BoundaryNorm(levels, ncolors=256, clip=False)
        if method == "contourf":
            cf = ax.contourf(time_grid, y_grid, values, levels=levels, cmap=cmap, norm=norm, extend='both', **kwargs)
            cbar = plt.colorbar(cf, ax=ax, extend='both')
            if add_contour:
                cs = ax.contour(time_grid, y_grid, values, levels=levels, colors='k', linewidths=0.5)
                if label_contours:
                    plt.clabel(cs, inline=True, fontsize=8)
        elif method == "pcolormesh":
            mesh = ax.pcolormesh(time_grid, y_grid, values, cmap=cmap, **kwargs)
            cbar = plt.colorbar(mesh, ax=ax)
        else:
            raise ValueError(f"Invalid method '{method}' for plotting 2D time series.")
        
        # Format colorbar
        cbar.set_label(var_name, fontsize=self.cfg.fontsize['ylabel'])
        # Add contour labels if requested
        
        # Format axes
        if ylim is not None:
            ax.set_ylim(ylim)
        rolling_text = f" with {rolling_window}-hour Rolling Mean" if rolling_window else ""
        title = f"Time Series of {var_name} ({dim}={index}){rolling_text}"

        ax.set_title(title, fontsize=self.cfg.fontsize['title'])
        ax.set_xlabel("Time", fontsize=self.cfg.fontsize['xlabel'])
        ax.set_ylabel(y_coord, fontsize=self.cfg.fontsize['ylabel'])
        date_format = kwargs.get('date_format', self.cfg.date_format)
        ax.xaxis.set_major_formatter(DateFormatter(date_format))
        fig.autofmt_xdate()


        # Save or show the plot
        if save_path:
            dpi = kwargs.get("dpi", self.cfg.dpi)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return ax

    def plot_2d(self, da=None, with_mesh=False, coastlines=False, obclines=False,
                vmin=None, vmax=None, levels=20, ax=None, save_path=None,
                use_latlon=True, projection=ccrs.Mercator(), plot_grid=False,
                add_tiles=False, tile_provider=GoogleTiles(style="satellite"), tile_zoom=12,
                verbose=False, post_process_func=None, xlim=None, ylim=None, **kwargs):
        """
        Plot the triangular mesh of the FVCOM grid.

        Parameters:
        - da: DataArray to plot. Default is None (wireframe).
        - with_mesh: If True, plot the mesh lines. Default is True.
        - coastlines: If True, plot coastlines. Default is False.
        - obclines: If True, plot open boundary lines. Default is False.
        - vmin: Minimum value for color scaling.
        - vmax: Maximum value for color scaling.
        - levels: Number of contour levels or a list of levels.
        - ax: matplotlib axis object. If None, a new axis will be created.
        - save_path: Path to save the plot as an image (optional).
        - use_latlon: If True, use (lon, lat) for nodes. If False, use (x, y).
        - projection: Cartopy projection for geographic plotting. Default is PlateCarree.
        - add_tiles: If True, add a tile map (for lat/lon plotting only).
        - tile_provider: Tile provider for the background.
        - tile_zoom (int): Zoom level for the tile map.
        - post_process_func: Function to apply custom plots or decorations to the Axes.
        - **kwargs: Additional arguments for customization.
        
        Note:
        projection in the following can be ccrs.Mercator(), which is the best in mid-latitudes.
            plt.subplots(figsize=figsize, subplot_kw={'projection': projection})
        The other parts, transform=ccrs.PlateCarree() must be set to inform lon/lat coords are used.
        Mercator is not lon/lat coords, so transform=ccrs.PlateCarree() is necessary.
        
        """

        self.use_latlon = use_latlon       
        transform = ccrs.PlateCarree() if self.use_latlon else None
        # Extract coordinates
        if self.use_latlon:
            x = self.ds["lon"].values
            y = self.ds["lat"].values
        else:
            x = self.ds["x"].values
            y = self.ds["y"].values

        if da is not None:
            values = da.values
            default_cbar_label = f"{da.long_name} ({da.units})"
            cbar_label = kwargs.get("cbar_label", default_cbar_label)
        else:
            with_mesh=True
        # Extract triangle connectivity
        #nv = self.ds["nv"].values.T - 1  # Convert to 0-based indexing
        nv = self.ds["nv_zero"]
        # Output ranges and connectivity
        if xlim is None:
            xmin, xmax = x.min(), x.max()
        else:
            xmin, xmax = xlim
        if ylim is None:
            ymin, ymax = y.min(), y.max()
        else:
            ymin, ymax = ylim
        if verbose:
            print(f"x range: {xmin} to {xmax}")
            print(f"y range: {ymin} to {ymax}")
            print(f"nv_ccw shape: {self.ds.nv_ccw.shape}, nv_ccw min: {self.ds.nv_ccw.min()}, nv_ccw max: {self.ds.nv_ccw.max()}")

        # Validate nv_ccw and coordinates
        if verbose:
            if np.isnan(x).any() or np.isnan(y).any():
                raise ValueError("NaN values found in node coordinates.")
            if np.isinf(x).any() or np.isinf(y).any():
                raise ValueError("Infinite values found in node coordinates.")
            if (self.ds.nv_ccw < 0).any() or (self.ds.nv_ccw >= len(x)).any():
                raise ValueError("Invalid indices in nv_ccw. Check if nv_ccw points to valid nodes.")

        # Reverse node order for counter-clockwise triangles that matplotlib expects.
        #nv = nv[:, ::-1]

        # Create Triangulation
        try:
            triang = tri.Triangulation(x, y, triangles=nv)
            if verbose:
                print(f"Number of triangles: {len(triang.triangles)}")
        except ValueError as e:
            print(f"Error creating Triangulation: {e}")
            return None

        # Set up axis
        if ax is None:
            figsize = kwargs.get("figsize", self.cfg.figsize)
            if self.use_latlon:
                fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': projection})
            else:
                fig, ax = plt.subplots(figsize=figsize)  # No projection for Cartesian
        else:
            fig = ax.figure
        
        # Add map tiles if requested
        if add_tiles and self.use_latlon:
            #tile_provider = cimgt.OSM()
            #tile_provider = cimgt.Stamen('terrain')
            #tile_provider = GoogleTiles(style="satellite")
            if tile_provider is None:
                raise ValueError("Tile provider is not set. Please provide a valid tile provider, \
                                 e.g., GoogleTiles(style='satellite')")
            else:
                ax.add_image(tile_provider, tile_zoom)
            #ax.add_image(tile_provider, 8)  # Zoom level 8 is suitable for regional plots

        # Argument treatment to avoid conflicts with **kwargs
        with_mesh = kwargs.pop('with_mesh', with_mesh)  # Remove with_mesh from kwargs
        lw = kwargs.pop('lw', 0.5)  # Line width for mesh plot
        
        #if not with_mesh:
        #    lw = 0
        color = kwargs.pop('color', "#36454F")  # Line color for mesh plot
        coastline_color = kwargs.pop('coastline_color', 'gray')  # coastline color
        obcline_color = kwargs.pop('obcline_color', "blue")  # Open boundary line color
        linestyle = kwargs.pop('linestyle', '--')  # Line style for lat/lon gridlines
        #linewidth = kwargs.pop('linewidth', 0.5)  # Line width for lat/lon gridlines
        
        # Filter tricontourf-specific kwargs to avoid conflicts
        valid_tricontourf_args = inspect.signature(ax.tricontourf).parameters.keys()
        tricontourf_kwargs = {key: kwargs[key] for key in valid_tricontourf_args if key in kwargs}

        # Prepare color plot
        if da is not None:
            cmap = kwargs.pop("cmap", "viridis") 
            auto_levels = True
            if vmin is not None or vmax is not None:
                auto_levels = False
            vmin = vmin or kwargs.pop("vmin", values.min().item())
            vmax = vmax or kwargs.pop("vmax", values.max().item())
            if np.all(values == values[0]):  
                vmax += 1e-6  # Sligtly increase vmax to avoid errors
            if vmin > vmax:
                raise ValueError(f"Invalid range: vmin ({vmin}) must be less than vmax ({vmax}).")

            if verbose:
                print(f"Color range: {vmin} to {vmax}")

            levels = levels or kwargs.pop("levels", 20)  # Number of contour levels
            if isinstance(levels, int):
                levels = np.linspace(vmin, vmax, levels)
                auto_levels = False
            elif isinstance(levels, (list, np.ndarray)):
                levels = np.array(levels)
                levels = levels[(levels >= vmin) & (levels <= vmax)]
                if len(levels) == 0:
                    raise ValueError("Filtered levels are empty after applying vmin and vmax.")
                auto_levels = False
            else:
                raise ValueError("Invalid levels argument. Must be an integer or a list of levels.")

            if auto_levels:
                norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = BoundaryNorm(levels, ncolors=256, clip=False)

        # Handle Cartesian coordinates
        if not self.use_latlon:
            title = kwargs.pop("title", "FVCOM Mesh (Cartesian)")
            if da is not None:
                cf = ax.tricontourf(triang, values, levels=levels, cmap=cmap, norm=norm, extend='both', **tricontourf_kwargs)
                cbar = plt.colorbar(cf, ax=ax, extend='both', orientation='vertical', shrink=1.0)
                cbar.set_label(cbar_label, fontsize=self.cfg.fontsize['cbar_label'], labelpad=10)
            if with_mesh:
                ax.triplot(triang, color=color, lw=lw)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_title(title, fontsize=self.cfg.fontsize["title"])
            xlabel = kwargs.get("xlabel", "X (m)")
            ylabel = kwargs.get("ylabel", "Y (m)")
            ax.set_xlabel(xlabel, fontsize=self.cfg.fontsize["xlabel"])
            ax.set_ylabel(ylabel, fontsize=self.cfg.fontsize["ylabel"])
            ax.set_aspect("equal")
        # Handle lat/lon coordinates
        else:
            #if add_tiles:
            #    ax.add_image(tile_provider, 8)
            #if da is not None:
            #    ax.add_patch(plt.Rectangle(
            #    (xmin, ymin),   # 左下の座標
            #     xmax - xmin,    # 横幅
            #     ymax - ymin,    # 縦幅
            #    color="lightgray",         # 塗りつぶしの色
            #    transform=ccrs.PlateCarree(),  # 緯度経度座標系での指定
            #    zorder=0  # 他のプロットの下に描画
            #    ))
            title = kwargs.pop("title", "")
            if da is not None:
                cf = ax.tricontourf(triang, values, levels=levels, cmap=cmap, norm=norm, extend='both',
                                    transform=ccrs.PlateCarree(), **tricontourf_kwargs)
                cbar = plt.colorbar(cf, ax=ax, extend='both', orientation='vertical', shrink=1.0)
                cbar.set_label(cbar_label, fontsize=self.cfg.fontsize["cbar_label"], labelpad=10)
            if with_mesh:
                # Always use PlateCarree here.
                ax.triplot(triang, color=color, lw=lw, transform=ccrs.PlateCarree())
            # Use set_extent for lat/lon ranges
            ax.set_extent([xmin, xmax, ymin, ymax], crs=ccrs.PlateCarree())
            #ax.set_extent([xmin, xmax, ymin, ymax], crs=projection)
            ax.set_title(title, fontsize=self.cfg.fontsize["title"])
            xlabel = kwargs.get("xlabel", "Longitude")
            ylabel = kwargs.get("ylabel", "Latitude")
            ax.set_xlabel(xlabel, fontsize=self.cfg.fontsize["xlabel"])
            ax.set_ylabel(ylabel, fontsize=self.cfg.fontsize["ylabel"])
            ax.set_aspect('equal')

            # Add gridlines for lat/lon. Always use PlateCarree here.
            if plot_grid:
                gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), linestyle=linestyle, lw=lw)
                gl.top_labels = False
                gl.right_labels = False
                gl.xlabel_style = {'size': 11}
                gl.ylabel_style = {'size': 11}
            else:
                gl = ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), linestyle=linestyle, lw=0)
                lon_ticks = gl.xlocator.tick_values(xmin, xmax)
                lat_ticks = gl.ylocator.tick_values(ymin, ymax)
                ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
                ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
                ax.xaxis.set_major_formatter(LongitudeFormatter())
                ax.yaxis.set_major_formatter(LatitudeFormatter())
                x_min_proj, y_min_proj = projection.transform_point(xmin, ymin, src_crs=ccrs.PlateCarree())
                x_max_proj, y_max_proj = projection.transform_point(xmax, ymax, src_crs=ccrs.PlateCarree())
                ax.set_xlim(x_min_proj, x_max_proj)
                ax.set_ylim(y_min_proj, y_max_proj)
                ax.tick_params(labelsize=11, labelcolor='black')

        if coastlines:
            print("Plotting coastlines...")
            nv = self.ds.nv_ccw.values
            nbe = np.array([[nv[n, j], nv[n, (j+2)%3]] for n in range(len(triang.neighbors))
                           for j in range(3) if triang.neighbors[n,j] == -1])
            for m in range(len(nbe)):
                #ax.plot(x[nbe[m,:]], y[nbe[m,:]], color='gray', linewidth=1, transform=transform)
                ax.plot(x[nbe[m,:]], y[nbe[m,:]], color=coastline_color, linewidth=1, transform=transform)


        if obclines:
            # Plot open boundary lines
            print("Plotting open boundary lines...")
            if "node_bc" not in self.ds:
                raise ValueError("Dataset does not contain 'node_bc' variable for open boundary lines."
                                 " obcfile must be read in FvcomDataLoader.")    
            node_bc = self.ds.node_bc.values
            ax.plot(x[node_bc[:]], y[node_bc[:]], color=obcline_color, linewidth=1, transform=transform)

        if post_process_func:
            # post_process_func の引数を解析
            func_signature = inspect.signature(post_process_func)
            valid_args = func_signature.parameters.keys()

            # 有効な引数に対応する値を取得または生成
            dynamic_kwargs = {}
            for arg in valid_args:
                if arg == "ax":
                    dynamic_kwargs[arg] = ax
                elif arg == "da":
                    dynamic_kwargs[arg] = da
                elif arg == "time":
                    dynamic_kwargs[arg] = time
                elif arg in locals():  # ローカルスコープで値を取得
                    dynamic_kwargs[arg] = locals()[arg]
                elif arg in globals():  # グローバルスコープで値を取得
                    dynamic_kwargs[arg] = globals()[arg]
                else:
                    print(f"Warning: Unable to resolve argument '{arg}'.")

            # post_process_funcを呼び出し
            post_process_func(**dynamic_kwargs)

        #if post_process_func:
        #    post_process_func(ax, **kwargs)

        # Save the plot if requested
        if save_path:
            dpi = kwargs.get("dpi", self.cfg.dpi)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Plot saved to: {save_path}")

        return ax

    def add_marker(self, ax=None, x=None, y=None, marker="o", color="red", size=20, **kwargs):
        """
        Add a marker to the existing plot (e.g., a mesh plot). Must be used with plot_2d.

        Parameters:
        - ax: matplotlib axis object. If None, raise an error because this method must follow plot_2d().
        - x: X coordinate (Cartesian) or longitude (geographic) of the marker.
        - y: Y coordinate (Cartesian) or latitude (geographic) of the marker.
        - marker: Marker style (default: "o").
        - color: Marker color (default: "red").
        - size: Marker size (default: 20).
        - **kwargs: Additional arguments passed to scatter.

        Returns:
        - ax: The axis object with the added marker.
        """
        # Ensure ax is provided
        if ax is None:
            raise ValueError("An axis object (ax) must be provided. Use plot_2d() first.")

        # Ensure use_latlon is determined by plot_2d
        if not hasattr(self, "use_latlon"):
            raise AttributeError("The 'use_latlon' attribute must be set by plot_2d().")

        # Ensure x and y are provided
        if x is None or y is None:
            raise ValueError("Both x and y must be specified.")

        transform = ccrs.PlateCarree() if self.use_latlon else None
        ax.scatter(x, y, transform=transform, marker=marker, color=color, s=size, **kwargs)

        return ax

    def ts_contourf(self, da: xr.DataArray, index: int = None, x='time', y='siglay', xlim=None, ylim=None,
                    xlabel='Time', ylabel='Sigma', title=None,
                    rolling_window=None, min_periods=None, ax=None, date_format=None,
                    contourf_kwargs: dict = None, colorbar_kwargs: dict = None, **kwargs
                   ) -> tuple[plt.Figure, plt.Axes, Colorbar]: 
        """
        Plot a contour map of vertical time-series DataArray.
        contourf_kwargs and **kwargs are combined to flexibly pass any contourf parameters; colorbar_kwargs is for colorbar settings.

        Parameters:
        ----------
        da (DataArray): DataArray for specified var_name with the dimension of (time, siglay/siglev [, node/nele]).
        index (int): Index of the node or element to plot (optional).
        x (str): Name of the x-axis coordinate. Default is 'time'.
        y (str): Name of the y-axis coordinate. Default is 'siglay'.
        xlim (tuple): Tuple of start and end times (e.g., ('2010-01-01', '2022-12-31')).
        ylim (tuple): Vertical range for the y-axis (e.g., (0, 1)).
        xlabel (str): Label for the x-axis. Default is 'Time'.
        ylabel (str): Label for the y-axis. Default is 'Depth (m)'.
        title (str): Title for the plot. Default is None.
        rolling_window (int): Size of the rolling window for moving average in hours (Default: None).
            24*30+1 (monthly mean)
        min_periods (int): Minimum number of data points required in the rolling window.
                        If None, defaults to window // 2 + 1.
        ax (matplotlib.axes.Axes): An existing axis to plot on. If None, a new axis will be created.
        date_format (str): Date format for the x-axis. Default is None.
        contourf_kwargs (dict): Arguments for contourf.
        colorbar_kwargs (dict): Arguments for colorbar.
        **kwargs: Arguments for contourf. Not supporting additional kwargs for colorbar.

        Returns:
        ----------
        tuple: (fig, ax, cbar)
        """
        contourf_kwargs = contourf_kwargs or {}
        colorbar_kwargs = colorbar_kwargs or {}

        # Automatically detect the spatial dimension ('node' or 'nele') and apply the given index
        spatial_dim = next((d for d in ('node', 'nele') if d in da.dims), None)
        if spatial_dim:
            if index is None:
                raise ValueError(f"Index must be provided for '{spatial_dim}' dimension")
            da = da.isel({spatial_dim: index})
        elif index is not None:
            raise ValueError(f"No 'node' or 'nele' dimension found in DataArray dims {da.dims}")

        # Apply rolling mean if specified
        if rolling_window:
            if min_periods is None:
                min_periods = rolling_window // 2 + 1
            da = da.rolling(time=rolling_window, center=True, min_periods=min_periods).mean()

        # Extract metadata for labels
        long_name = da.attrs.get('long_name', da.name)
        units = da.attrs.get('units', '-')
        cbar_label = f"{long_name} ({units})"
        if title is None:
            rolling_text = f" with {rolling_window}-hour Rolling Mean" if rolling_window else ""  
            if spatial_dim is not None:
                title = f"Time Series of {long_name} ({spatial_dim}={index}){rolling_text}"
            else:
                title = f"Time Series of {long_name}{rolling_text}"

        date_format = date_format or self.cfg.date_format

        # Time range filtering via xlim tuple
        da = self._apply_time_filter(da, xlim)

        # Create figure and axis if not provided
        if ax is None:
            fig = plt.figure(figsize=self.cfg.figsize, dpi=self.cfg.dpi)
            gs = GridSpec(1, 1, left=0.1, right=0.9, top=0.9, bottom=0.1)
            ax = fig.add_subplot(gs[0])
        else:
            fig = ax.figure

        # Merge kwargs with contourf_kwargs
        merged_contourf_kwargs, levels, cmap, vmin, vmax, extend = \
            self._prepare_contourf_args(da, contourf_kwargs, kwargs)

        # Plot the contour map
        #plot = da.plot.contourf(x=x, y=y, ylim=ylim, levels=levels, cmap=cmap,
        #                        vmin=vmin, vmax=vmax, extend=extend, ax=ax,
        #                        add_colorbar=False, **merged_contourf_kwargs)
        #if ylim is None:
        #    ylim = (da[y].min().item(), da[y].max().item())
        #    ax.set_ylim(ylim)
        # Build kwargs for contourf
        plot_kwargs = {
            "x": x, "y": y,
            "levels": levels, "cmap": cmap,
            "vmin": vmin, "vmax": vmax, "extend": extend,
            "ax": ax, "add_colorbar": False
        }
        # Only include ylim if explicitly set
        if ylim is not None:
            plot_kwargs["ylim"] = ylim

        # Call contourf
        plot = da.plot.contourf(**plot_kwargs, **merged_contourf_kwargs)        

        # Set axis formatting
        self._format_time_axis(ax, title, xlabel, ylabel, date_format)

        # ax.invert_yaxis()

        # Add colorbar
        cb_copy      = dict(colorbar_kwargs or {})
        label_to_use = cb_copy.pop("label", cbar_label)
        cbar         = self._make_colorbar(ax, plot, label_to_use, cb_copy)

        return fig, ax, cbar

    def ts_plot(self, da: xr.DataArray, index: int = None, k: int = None, ax=None,
                xlabel: str = None, ylabel: str = None, title: str = None,
                color: str = None, linestyle: str = None, date_format: str = None,
                xlim: tuple = None, ylim: tuple = None, rolling_window=None, log=False,
                **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """
        1-D time series plot.
        
        Parameters:
        ----------
        da : xr.DataArray
            DataArray with a 'time' dimension.
        index : int, optional
            Index for spatial dimension (node/nele). Not needed for pure time series.
        k : int, optional
            Layer index for vertical dimension (siglay/siglev). Not needed for 2D or 1D series.
        ax : matplotlib.axes.Axes, optional
            Existing axis. Creates new one if None.
        xlabel : str, optional
            X-axis label. Default: 'Time'.
        ylabel : str, optional
            Y-axis label. Default: da.long_name or da.name.
        title : str, optional
            Plot title. Default: '<ylabel> Time Series'.
        color : str, optional
            Line color. Default: self.cfg.plot_color.
        linestyle : str, optional
            Line style. Default: '-'.
        date_format : str, optional
            Date formatter. Default: self.cfg.date_format.
        xlim : tuple, optional
            X-axis limits (start, end). Default: None.
        ylim : tuple, optional
            Y-axis limits (ymin, ymax). Default: None.
        rolling_window : int, optional
            Size of the rolling window for moving average. Default: None.
        log : bool, optional
            If True, use logarithmic scale. Default: False.
        **kwargs : dict
            Extra keyword args for ax.plot().
        
        Returns:
        ----------
        tuple: (fig, ax)
        """
        # 1) Slice da based on its dimensions
        #dims = da.dims
        data, spatial_dim, layer_dim = self._slice_time_series(da, index, k)

        # 2) Apply rolling mean before time filtering
        data = self._apply_rolling(data, rolling_window)

        # 3) Prepare labels and title
        xlabel, ylabel, title = self._prepare_ts_labels(
            data, spatial_dim, layer_dim,
            index, k, rolling_window, xlabel, ylabel, title)
        
        # 4) Time filtering and formatting
        date_format = date_format or self.cfg.date_format
        data        = self._apply_time_filter(data, xlim)
        times       = data["time"].values
        values      = data.values

        # 5) Prepare figure/axis
        if ax is None:
            fig, ax = plt.subplots(figsize=self.cfg.figsize, dpi=self.cfg.dpi)
        else:
            fig = ax.figure

        # 6) Plot
        color     = color or self.cfg.plot_color
        linestyle = linestyle or "-"
        ax.plot(times, values, color=color, linestyle=linestyle, **kwargs)
        #if log:
        #    ax.set_yscale("log")
        # Apply log scale via helper (handles warnings for non-positive data)
        self._apply_log_scale(ax, data, log)

        # 7) Y‑axis limits
        if ylim is not None:
            ymin, ymax    = ylim
            curr_min, curr_max = ax.get_ylim()
            ymin = curr_min if ymin is None else ymin
            ymax = curr_max if ymax is None else ymax
            ax.set_ylim(ymin, ymax)

        # 8) Final formatting
        self._format_time_axis(ax, title, xlabel, ylabel, date_format)

        return fig, ax

    def section_contourf_z(self, da: xr.DataArray, lat: float = None, lon: float = None,
        line: list[tuple[float, float]] = None, spacing: float = 200.0, xlim: tuple = None, ylim: tuple = None,
        xlabel: str = "Distance (m)", ylabel: str = "Depth (m)", title: str = None,
        ax=None, land_color: str = "#A0522D",  # Default seabed/land color (sienna)
        contourf_kwargs: dict = None, colorbar_kwargs: dict = None, **kwargs):
        """
        Plot a vertical section of a 3D variable (da) on FVCOM mesh.

        Parameters:
          da: DataArray with dims (siglay/siglev, node) at single time
          lat, lon: constant latitude or longitude for section
          line: list of (lon, lat) pairs defining arbitrary transect
          spacing: sampling interval (m)
          xlim, ylim: axis limits
          xlabel, ylabel, title: plot labels
          ax: existing Matplotlib Axes
          land_color: seabed/land color (default: "#A0522D" # Sienna)
          contourf_kwargs: dict of base contourf args
          colorbar_kwargs: dict for colorbar
          **kwargs: extra contourf keywords (override contourf_kwargs)

        Returns: fig, ax, cbar
        """

        # 0 Validate that requested lat/lon lie within the dataset domain
        lon_vals = self.ds['lon'].values if 'lon' in self.ds else None
        lat_vals = self.ds['lat'].values if 'lat' in self.ds else None
        if lat is not None and lat_vals is not None:
            lat_min, lat_max = float(lat_vals.min()), float(lat_vals.max())
            if not (lat_min <= lat <= lat_max):
                raise ValueError(f"Latitude {lat} is outside domain bounds [{lat_min}, {lat_max}].")
        if lon is not None and lon_vals is not None:
            lon_min, lon_max = float(lon_vals.min()), float(lon_vals.max())
            if not (lon_min <= lon <= lon_max):
                raise ValueError(f"Longitude {lon} is outside domain bounds [{lon_min}, {lon_max}].")

        # Determine vertical dimension
        vert_dim = 'siglay' if 'siglay' in da.dims else 'siglev'

        # Get depth array at same time
        z_all = self.ds['z']
        if 'time' in z_all.dims:
            if 'time' in da.coords:
                z_slice = z_all.sel(time=da['time'], method='nearest')
            else:
                z_slice = z_all.isel(time=0)
            z2d = z_slice.values  # (vertical, node)
        else:
            z2d = z_all.values

        # Prepare mesh triangulation for domain test
        lon_n = self.ds['lon'].values; lat_n = self.ds['lat'].values
        tris = self.ds['nv'].values.T - 1
        triang = mtri.Triangulation(lon_n, lat_n, triangles=tris)
        trifinder = triang.get_trifinder()

        # Build KDTree on projected nodes for nearest-node lookup
        mean_lon, mean_lat = lon_n.mean(), lat_n.mean()
        zone = int((mean_lon + 180)//6) + 1
        hemi = 'north' if mean_lat >= 0 else 'south'
        proj = pyproj.Proj(f"+proj=utm +zone={zone} +{hemi} +datum=WGS84")
        x_n, y_n = proj(lon_n, lat_n)
        tree = KDTree(np.column_stack((x_n, y_n)))

        # Define transect endpoints
        if line:
            pts = line
        elif lat is not None:
            pts = [(float(lon_n.min()), lat), (float(lon_n.max()), lat)]
        elif lon is not None:
            pts = [(lon, float(lat_n.min())), (lon, float(lat_n.max()))]
        else:
            raise ValueError("Specify lat, lon, or line for section.")

        lons, lats, distances = self._sample_transect(lat=lat, lon=lon, line=line, spacing=spacing)
        # Domain mask
        tri_idx = trifinder(lons, lats)
        inside = tri_idx != -1

        # Nearest-node for each sample
        x_s, y_s = proj(lons, lats)
        _, idx_n = tree.query(np.column_stack((x_s, y_s)))

        # Extract variable and depth
        X, Y, V = self._extract_section_data(da, lons, lats, distances)
        # Plot
        fig = ax.figure if ax else plt.figure(figsize=self.cfg.figsize, dpi=self.cfg.dpi)
        ax = ax or fig.add_subplot(1,1,1)

        # Build DataArray for section: 
        # - dims: (siglay, distance)
        # - coords:
        #    * distance: 1D array of sampled distances
        #    * depth:    2D array of shape (siglay, distance)
        sec_da = xr.DataArray(V, dims=("siglay", "distance"),
            coords={"distance": distances, "depth": (("siglay","distance"), Y)})
        
        # 1 Prepare contourf args on section-DataArray
        merged_cf_kwargs, levels, cmap_used, vmin, vmax, extend = \
            self._prepare_contourf_args(sec_da, contourf_kwargs, kwargs)

        # 2 use xarray's contourf wrapper (same as ts_contourf)
        cs = sec_da.plot.contourf(x="distance", y="depth", levels=levels, cmap=cmap_used,
            vmin=vmin, vmax=vmax, extend=extend, ax=ax, add_colorbar=False, **merged_cf_kwargs)

        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if title: ax.set_title(title)
        ax.set_xlim(distances.min(), distances.max())
        # Set y-axis so shallow (near-zero) is at top and deep (large negative) at bottom
        ax.set_ylim(np.nanmin(Y), np.nanmax(Y))

        # 3 Now that y-limits are fixed, fill seabed and mesh-missing regions
        bottom_depth = np.nanmin(Y, axis=0)      # seabed profile (deepest mesh)
        ymin_axis, ymax_axis = ax.get_ylim()     # current axis limits
        fill_base = min(ymin_axis, ymax_axis)    # lower boundary in data coords
        # a) fill below seabed line (land patch under ocean)
        ax.fill_between(distances, bottom_depth, fill_base, where=~np.isnan(bottom_depth),
            facecolor=land_color, edgecolor=None, zorder=cs.zorder - 0.5,  # between axes background and contourf
            clip_on=False)
        # b) fill entire vertical for columns completely outside mesh domain
        mask_nan = np.all(np.isnan(V), axis=0)
        if mask_nan.any():
            ax.fill_between(distances[mask_nan], fill_base,
                ymax_axis,               # fill up to top of axis (air remains white elsewhere)
                facecolor=land_color, edgecolor=None, zorder=cs.zorder - 0.5, clip_on=False)

        # Plot the seabed line on top (use true bottom = deepest depth)
        ax.plot(distances, bottom_depth, color='k', linestyle='-', linewidth=1, zorder=cs.zorder + 1)

        cbar = self._make_colorbar(ax, cs, da.attrs.get('long_name', da.name) + (f" ({da.attrs.get('units','')})" if 'units' in da.attrs else ''), colorbar_kwargs or {})

        return fig, ax, cbar

    # --------------------------------
    # Private helper methods
    # --------------------------------

    def _sample_transect(self, lat: float = None, lon: float = None, line: list[tuple[float, float]] = None,
        spacing: float = 200.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate evenly spaced sample points along a transect.
        Returns: lons, lats, cumulative distances (m).
        """
        import numpy as np
        import pyproj

        # Determine transect line endpoints
        if line:
            pts = line
        elif lat is not None:
            pts = [(float(self.ds['lon'].min()), lat), (float(self.ds['lon'].max()), lat)]
        elif lon is not None:
            pts = [(lon, float(self.ds['lat'].min())), (lon, float(self.ds['lat'].max()))]
        else:
            raise ValueError("Specify lat, lon, or line for section.")

        geod = pyproj.Geod(ellps='WGS84')
        samples = [pts[0]]
        for p0, p1 in zip(pts[:-1], pts[1:]):
            lon0, lat0 = p0; lon1, lat1 = p1
            _, _, dist = geod.inv(lon0, lat0, lon1, lat1)
            steps = int(dist // spacing)
            for i in range(1, steps+1):
                lon_i, lat_i, _ = geod.fwd(lon0, lat0, geod.inv(lon0, lat0, lon1, lat1)[0], i*spacing)
                samples.append((lon_i, lat_i))
            samples.append((lon1, lat1))

        lons = np.array([p[0] for p in samples])
        lats = np.array([p[1] for p in samples])
        dists = np.zeros(len(samples))
        for i in range(1, len(samples)):
            _, _, seg = geod.inv(lons[i-1], lats[i-1], lons[i], lats[i])
            dists[i] = dists[i-1] + seg
        return lons, lats, dists

    def _extract_section_data(self, da: xr.DataArray, lons: np.ndarray, lats: np.ndarray,
        dists: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build 2D grids X (distance), Y (depth), and V (values).
        """
        import numpy as np
        import matplotlib.tri as mtri
        from scipy.spatial import KDTree
        import pyproj

        # Triangulate mesh for domain test
        lon_n = self.ds['lon'].values; lat_n = self.ds['lat'].values
        triangles = self.ds['nv'].values.T - 1
        triang = mtri.Triangulation(lon_n, lat_n, triangles=triangles)
        trifinder = triang.get_trifinder()

        # Project nodes to UTM for KDTree
        mean_lon, mean_lat = lon_n.mean(), lat_n.mean()
        zone = int((mean_lon+180)//6)+1; hemi = 'north' if mean_lat>=0 else 'south'
        proj = pyproj.Proj(f"+proj=utm +zone={zone} +{hemi} +datum=WGS84")
        x_n, y_n = proj(lon_n, lat_n)
        tree = KDTree(np.column_stack((x_n, y_n)))

        # Determine in-domain samples
        tri_idx = trifinder(lons, lats)
        inside = tri_idx != -1

        # Nearest-node lookup
        x_s, y_s = proj(lons, lats)
        _, idx_n = tree.query(np.column_stack((x_s, y_s)))

        # Extract variable values and mask
        V = da.values[:, idx_n]
        V[:, ~inside] = np.nan

        # Extract depth z and mask
        z_all = self.ds['z']
        if 'time' in z_all.dims and 'time' in da.coords:
            z2d = z_all.sel(time=da['time'], method='nearest').values
        else:
            z2d = z_all.values
        Y = z2d[:, idx_n]
        Y[:, ~inside] = np.nan

        # Build distance grid
        X = np.broadcast_to(dists[np.newaxis, :], Y.shape)
        return X, Y, V


    def _prepare_contourf_args(self, da, contourf_kwargs, extra_kwargs):
        """
        Merge contourf_kwargs and extra_kwargs, and extract levels, cmap, vmin, vmax, extend.
        Returns:
            merged_kwargs, levels, cmap, vmin, vmax, extend
        """
        contourf_kwargs = contourf_kwargs or {}
        merged = {**contourf_kwargs, **extra_kwargs}

        # Extract and handle contourf parameters with appropriate defaults
        levels   = merged.pop("levels", getattr(self.cfg, "levels", None))
        cmap     = merged.pop("cmap",   getattr(self.cfg, "cmap", None))
        raw_vmin = merged.pop("vmin",   None)
        raw_vmax = merged.pop("vmax",   None)

        # Determine vmin and vmax from data if not explicitly provided
        vmin = raw_vmin if raw_vmin is not None else da.min().item()
        vmax = raw_vmax if raw_vmax is not None else da.max().item()
        print(f"vmin: {vmin}, vmax: {vmax}")

        # Convert levels if necessary:
        if levels is None:
            # Default to config levels if available, otherwise fallback to 21 levels
            levels = self.cfg.levels if hasattr(self.cfg, "levels") else None
            if levels is None:
                levels = 21
        if isinstance(levels, (int, np.integer)):
            # If levels is an integer, create that many linearly spaced levels
            n_levels = int(levels)
            if vmin == vmax:
                # Handle degenerate case: all data equal
                levels = np.array([vmin, vmax])
            else:
                levels = np.linspace(vmin, vmax, n_levels)
        elif isinstance(levels, (list, np.ndarray)):
            # Use the list/array of levels directly
            levels = np.asarray(levels)

        # If no explicit vmin/vmax given, align vmin/vmax with the levels range
        if raw_vmin is None and raw_vmax is None and isinstance(levels, np.ndarray):
            if levels.size > 0:
                vmin = levels.min()
                vmax = levels.max()

        # Determine the "extend" for contourf (how to handle values outside levels range)
        if "extend" in merged:
            extend = merged.pop("extend")
        else:
            data_min = da.min().item()
            data_max = da.max().item()
            print(f"data_min: {data_min}, data_max: {data_max}")
            print(f"vmin: {vmin}, vmax: {vmax}")
            if vmin <= data_min and vmax >= data_max:
                extend = "neither"
            elif vmin > data_min and vmax >= data_max:
                extend = "min"
            elif vmax < data_max and vmin <= data_min:
                extend = "max"
                print("Here")
            else:
                extend = "both"
        print(f"extend: {extend}")

        return merged, levels, cmap, vmin, vmax, extend

    def _make_colorbar(self, ax, mappable, label, colorbar_kwargs):
        """
        Create and attach a colorbar to `ax` for the given mappable (QuadContourSet).
        """
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",
                                  size=self.cfg.cbar_size,
                                  pad=self.cfg.cbar_pad)
        cbar = ax.figure.colorbar(mappable,
                                  cax=cax,
                                  extend='both',
                                  label=label,
                                  **(colorbar_kwargs or {}))
        cbar.ax.yaxis.label.set_size(self.cfg.label_fontsize)
        return cbar

    def _format_time_axis(self, ax: plt.Axes, title: str, xlabel: str, ylabel: str,
                          date_format: str) -> None:
        """
        Helper function to format the time axis for time series plots.
        """
        ax.set_title(title,      fontsize=self.cfg.fontsize['title'])
        ax.set_xlabel(xlabel,    fontsize=self.cfg.fontsize['xlabel'])
        ax.set_ylabel(ylabel,    fontsize=self.cfg.fontsize['ylabel'])
        ax.xaxis.set_major_formatter(DateFormatter(date_format))
        ax.figure.autofmt_xdate()
    
    def _apply_time_filter(self, da: xr.DataArray, xlim: tuple | None) -> xr.DataArray:
        """
        Apply time filtering to da based on xlim=(start, end).
        If xlim is None, return da unchanged.
        """
        if xlim is None:
            return da

        start, end = xlim
        start_sel = np.datetime64(start) if start is not None else None
        end_sel   = np.datetime64(end)   if end   is not None else None
        return da.sel(time=slice(start_sel, end_sel))

    def _slice_time_series(self, da: xr.DataArray, index: int = None,
                           k: int = None) -> tuple[xr.DataArray, str | None, str | None]:
        """
        Slice a DataArray for 1D or vertical time series.

        Returns:
          sliced DataArray, spatial dimension name, layer dimension name
        """
        dims = da.dims

        # 3D time series (time, layer, space)
        if "time" in dims and ("siglay" in dims or "siglev" in dims):
            layer_dim   = "siglay" if "siglay" in dims else "siglev"
            spatial_dim = next(d for d in dims if d not in ("time", layer_dim))
            if index is None or k is None:
                raise ValueError(f"Both index and k are required for dims {dims}")
            sliced = da.isel({spatial_dim: index, layer_dim: k})

        # 2D time series (time, space)
        elif "time" in dims and ("node" in dims or "nele" in dims):
            layer_dim   = None
            spatial_dim = "node" if "node" in dims else "nele"
            if index is None:
                raise ValueError(f"Index is required for dims {dims}")
            sliced = da.isel({spatial_dim: index})

        # Pure 1D time series (time,)
        elif dims == ("time",):
            sliced      = da
            spatial_dim = layer_dim = None

        else:
            raise ValueError(f"Unsupported DataArray dims: {dims}")

        return sliced, spatial_dim, layer_dim

    def _prepare_ts_labels(self, data: xr.DataArray, spatial_dim: str | None, layer_dim: str | None,
        index: int | None, k: int | None, rolling_window: int | None,
        xlabel: str | None, ylabel: str | None, title: str | None) -> tuple[str, str, str]:
        """
        Prepare and return xlabel, ylabel, title for ts_plot.

        Returns: (xlabel, ylabel, title)
        """
        # Use long_name or variable name for label base
        long_name = data.attrs.get("long_name", data.name)
        units     = data.attrs.get("units", "")

        # Set default xlabel only when None
        if xlabel is None:
            xlabel = "Time"

        # Set default ylabel only when None
        if ylabel is None:
            ylabel = f"{long_name} ({units})"

        # Build title only when None
        if title is None:
            # add rolling text if requested
            roll_txt = f" with {rolling_window}-hour Rolling Mean" if rolling_window else ""
            if spatial_dim:
                if layer_dim:
                    title = f"Time Series of {long_name} ({spatial_dim}={index}, {layer_dim}={k}){roll_txt}"
                else:
                    title = f"Time Series of {long_name} ({spatial_dim}={index}){roll_txt}"
            else:
                title = f"Time Series of {long_name}{roll_txt}"

        return xlabel, ylabel, title

    def _apply_log_scale(self, ax: plt.Axes, data: xr.DataArray, log_flag: bool) -> None:
        """
        Apply logarithmic scale to the y-axis if requested, using proper locator and formatter.
        """
        if not log_flag:
            return

        # Only positive values can be plotted on a log scale
        if data.min().item() <= 0:
            import warnings
            warnings.warn(
                "Log scale requested but data contains non-positive values; skipping log scale."
            )
            return

        # Set y-axis to log scale with base-10 locator/formatter
        ax.set_yscale("log", base=10)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(LogFormatter())

    def _apply_rolling(self, da: xr.DataArray, window: int, min_periods: int | None=None) -> xr.DataArray:
        """
        Apply centered rolling mean on time axis with optional min_periods.
        """
        if window is None:
            return da
        mp = min_periods if min_periods is not None else window//2 + 1
        return da.rolling(time=window, center=True, min_periods=mp).mean()


# Example usage
if __name__ == "__main__":
    # Load data
    loader = FvcomDataLoader(base_path="/path/to/data", ncfile="sample.nc")
    
    # Analyze
    analyzer = FvcomAnalyzer(loader.ds)
    nearest_node = analyzer.nearest_neighbor(lon=140.0, lat=35.0)

    # Plot
    plot_config = FvcomPlotConfig(width=1000, height=400)
    plotter = FvcomPlotter(loader.ds, plot_config)
    plot = plotter.plot_time_series("z", siglay=0, node=nearest_node)
    print(plot)

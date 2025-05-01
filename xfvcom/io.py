# --- standard library -------------------------------------------------
import os
from pathlib import Path

# --- third-party ------------------------------------------------------
import numpy as np
import pandas as pd
import xarray as xr
import pyproj

# --- package internal -------------------------------------------------
from .helpers_utils import ensure_time_index

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

        if "siglay" in self.ds and "siglay_width" not in self.ds:
            if "siglev" in self.ds:
                # use first node; all nodes are identical in sigma space
                d_sigma = -self.ds["siglev"].isel(node=0).diff("siglev")
                d_sigma = d_sigma.rename({"siglev": "siglay"})    # dims: ('siglay')
            else:
                nl = self.ds.dims["siglay"]
                d_sigma = xr.DataArray(np.ones(nl) / nl, dims="siglay")

            d_sigma.attrs.update({
                "long_name": "sigma layer thickness (positive)",
                "units": "1"
            })
            self.ds["siglay_width"] = d_sigma      # only ('siglay'), no 'node'

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

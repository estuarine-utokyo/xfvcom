from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pyproj
import xarray as xr
from sklearn.neighbors import NearestNeighbors


class FvcomAnalyzer:
    """
    Provides analysis capabilities for FVCOM datasets.
    """

    def __init__(self, dataset: xr.Dataset, zone: int = 54, north: bool = True) -> None:
        """
        Initializes the FvcomAnalyzer with a dataset and projection parameters.

        Parameters:
        - dataset: An xarray Dataset containing FVCOM data.
        - zone: UTM zone for coordinate transformation (default is 54).
        - north: Boolean indicating if the UTM zone is in the northern hemisphere (default is True).
        """

        self.ds = dataset
        self.zone = zone
        self.north = north

    def get_variables_by_dims(self, dims: Sequence[str]) -> list[str]:
        """
        Returns a list of variable names that have the specified dimensions.

        Parameters:
        - dims: A tuple of dimensions to filter by (e.g., ('time', 'node')).

        Returns:
        - A list of variable names matching the specified dimensions.
        """
        # variables = [
        #     var_name for var_name, var in self.ds.variables.items() if var.dims == dims
        # ]
        dims_tup = tuple(dims)
        variables: list[str] = [
            var_name
            for var_name, var in self.ds.variables.items()
            if var.dims == dims_tup
        ]

        return variables

    def nearest_neighbor(
        self,
        lon: float | np.ndarray,
        lat: float | np.ndarray,
        node: bool = True,
        distances: bool = False,
    ) -> int | tuple[float, int]:
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
        target_coords = np.array(self._lonlat_to_xy(lon, lat, inverse=False)).reshape(
            1, -1
        )

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

    def _lonlat_to_xy(
        self, lon: float | np.ndarray, lat: float | np.ndarray, inverse: bool = False
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """
        Convert geographic coordinates to UTM or vice versa.
        """
        crs_from = f"+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
        crs_to = f"+proj=utm +zone={self.zone} {'+north' if self.north else ''} +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
        transformer = (
            pyproj.Transformer.from_crs(crs_to, crs_from)
            if inverse
            else pyproj.Transformer.from_crs(crs_from, crs_to)
        )
        return transformer.transform(lon, lat)

import numpy as np
from sklearn.neighbors import NearestNeighbors
import pyproj

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

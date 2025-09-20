from __future__ import annotations

"""Core mesh (grid) object used throughout xfvcom.

This module defines :class:`FvcomGrid`, a lightweight dataclass that
encapsulates the horizontal (2‑D) unstructured mesh used by FVCOM.

Key features
------------
* **Multiple constructors**
  * ``from_dataset`` – create from an *output* NetCDF that already contains the
    grid variables.
  * ``from_dat`` – parse the ASCII ``*_grd.dat`` file (UTM) and automatically
    compute geographic lon/lat.
* **NumPy‑based attributes** for fast numerics, plus :py:meth:`to_xarray` for
  high‑level analysis/visualisation.
* **Zero‑based connectivity** is enforced inside the class.

All in‑code comments remain in **English** as requested.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .dat_reader import read_dat
from .geo_utils import utm_to_lonlat

# -----------------------------------------------------------------------------
# Helper protocol – minimal Dataset interface
# -----------------------------------------------------------------------------


@runtime_checkable
class _DatasetLike(Protocol):
    def __getitem__(self, key: str) -> Any: ...
    def __contains__(self, key: str) -> bool: ...


# -----------------------------------------------------------------------------
# Main dataclass
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class FvcomGrid:
    """FVCOM horizontal grid (unstructured triangular mesh)."""

    # core arrays ---------------------------------------------------------
    x: NDArray[np.float64]  # node x (UTM metres)
    y: NDArray[np.float64]  # node y (UTM metres)
    nv: NDArray[np.int_]  # connectivity (3, nele) – **zero-based**

    # projection meta -----------------------------------------------------
    zone: int | None = None  # UTM zone number (1‑60)
    northern: bool = True  # hemisphere flag

    # optional geographic -----------------------------------------------
    lon: NDArray[np.float64] | None = field(default=None, repr=False)
    lat: NDArray[np.float64] | None = field(default=None, repr=False)
    lonc: NDArray[np.float64] | None = field(default=None, repr=False)
    latc: NDArray[np.float64] | None = field(default=None, repr=False)
    xc: NDArray[np.float64] | None = field(default=None, repr=False)
    yc: NDArray[np.float64] | None = field(default=None, repr=False)

    # cached topology helpers -----------------------------------------
    _boundary_edge_cache: set[tuple[int, int]] | None = field(
        default=None, init=False, repr=False
    )
    _boundary_node_cache: set[int] | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dataset(cls, ds: _DatasetLike, *, validate: bool = True) -> "FvcomGrid":
        """Build from an xarray.Dataset that already contains grid variables."""
        req = ("x", "y", "nv")
        miss = [n for n in req if n not in ds]
        if miss and validate:
            raise KeyError("Dataset missing grid vars: " + ", ".join(miss))

        # ---- fallback for loose validation --------------------------------
        if not validate:
            # nv_zero → nv
            if "nv" not in ds and "nv_zero" in ds:
                ds = ds.assign(nv=ds["nv_zero"])
            # if planar x/y missing but lon/lat present, copy for plotting
            if "x" not in ds and "lon" in ds:
                ds = ds.assign(x=ds["lon"])
            if "y" not in ds and "lat" in ds:
                ds = ds.assign(y=ds["lat"])
            miss = [n for n in req if n not in ds]
            if miss:
                raise KeyError("Dataset still missing vars: " + ", ".join(miss))

        kw: dict[str, Any] = {
            "x": np.asarray(ds["x"].values, dtype=float),
            "y": np.asarray(ds["y"].values, dtype=float),
        }
        # ------------------------------------------------------------------
        # nv: ensure (3, nele) shape & 0-based
        nv_raw = np.asarray(ds["nv"].values, dtype=int)
        if nv_raw.shape[0] != 3 and nv_raw.shape[1] == 3:
            nv_raw = nv_raw.T  # (nele,3) → (3,nele)
        # 1-based → 0-based
        if nv_raw.min() == 1:
            nv_raw = nv_raw - 1
        kw["nv"] = nv_raw

        for name in ("lon", "lat", "lonc", "latc"):
            if name in ds:
                kw[name] = np.asarray(ds[name].values, dtype=float)
        # compute element centres
        nv = kw["nv"]
        kw["xc"] = kw["x"][nv].mean(axis=0)
        kw["yc"] = kw["y"][nv].mean(axis=0)

        return cls(**kw)  # type: ignore[arg-type]

    @classmethod
    def from_dat(
        cls,
        path: str | Path,
        *,
        utm_zone: int | None = None,
        northern: bool | None = None,
    ) -> "FvcomGrid":
        """Parse ``*_grd.dat`` and return a fully populated grid object."""
        data = read_dat(path)

        # UTM zone & hemisphere – file hint < user override ----------------
        zone = utm_zone or data["zone"]
        if zone is None:
            raise ValueError(
                "UTM zone could not be determined – pass utm_zone explicitly."
            )
        hemi = data["northern"] if northern is None else northern

        # geographic conversion ------------------------------------------
        lon, lat = utm_to_lonlat(data["x"], data["y"], zone=zone, northern=hemi)

        # sanity-check nv shape vs node count
        if data["nv"].max() >= data["x"].size:
            raise ValueError(
                "Connectivity (nv) references node index beyond range — "
                "DAT file may have been mis-parsed. "
                "Check nNode/nElem detection."
            )

        xc = data["x"][data["nv"]].mean(axis=0)
        yc = data["y"][data["nv"]].mean(axis=0)
        lonc, latc = utm_to_lonlat(xc, yc, zone=zone, northern=hemi)

        return cls(
            x=data["x"],
            y=data["y"],
            nv=data["nv"],
            zone=zone,
            northern=hemi,
            lon=lon,
            lat=lat,
            lonc=lonc,
            latc=latc,
            xc=xc,
            yc=yc,
        )

    # ------------------------------------------------------------------
    # Quick properties
    # ------------------------------------------------------------------
    @property
    def nele(self) -> int:  # number of elements
        return self.nv.shape[1]

    @property
    def node(self) -> int:  # number of nodes
        return self.x.size

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def to_xarray(self) -> xr.Dataset:
        """Return a minimal Dataset containing the grid variables."""
        ds = xr.Dataset(
            {
                "x": ("node", self.x),
                "y": ("node", self.y),
                "nv": (("three", "nele"), self.nv),
            },
            coords={
                "node": ("node", np.arange(self.node)),
                "nele": ("nele", np.arange(self.nele)),
                "three": ("three", np.arange(3)),
            },
            attrs={"cf_role": "mesh_topology"},
        )
        if self.lon is not None:
            ds["lon"] = ("node", self.lon)
        if self.lat is not None:
            ds["lat"] = ("node", self.lat)
        if self.lonc is not None:
            ds["lonc"] = ("nele", self.lonc)
        if self.latc is not None:
            ds["latc"] = ("nele", self.latc)
        if self.xc is not None:
            ds["xc"] = ("nele", self.xc)
        if self.yc is not None:
            ds["yc"] = ("nele", self.yc)

        return ds

    # ------------------------------------------------------------------
    # Area calculation methods
    # ------------------------------------------------------------------
    def calculate_node_area(
        self,
        node_indices: list[int] | NDArray[np.int_] | None = None,
        index_base: int = 1,
    ) -> float:
        """Calculate total area of triangular elements containing specified nodes.

        Parameters
        ----------
        node_indices : list[int] | NDArray[np.int_] | None
            List of node indices. If None, calculates area for all nodes.
        index_base : int
            0 for zero-based indexing, 1 for one-based indexing (FVCOM default)

        Returns
        -------
        float
            Total area in square meters (assuming x, y are in meters/UTM)
        """
        import numpy as np

        # Handle None case - all nodes
        if node_indices is None:
            node_indices = np.arange(
                1 if index_base == 1 else 0, self.node + (1 if index_base == 1 else 0)
            )

        # Convert to numpy array
        node_indices_arr: NDArray[np.int_] = np.asarray(node_indices, dtype=int)

        # Convert to zero-based if needed
        if index_base == 1:
            node_indices_0 = node_indices_arr - 1
        else:
            node_indices_0 = node_indices_arr

        # Validate indices
        if np.any(node_indices_0 < 0) or np.any(node_indices_0 >= self.node):
            invalid = node_indices_arr[
                (node_indices_0 < 0) | (node_indices_0 >= self.node)
            ]
            raise ValueError(
                f"Invalid node indices (base-{index_base}): {invalid.tolist()}. "
                f"Valid range: {1 if index_base == 1 else 0} to "
                f"{self.node if index_base == 1 else self.node - 1}"
            )

        # Find all elements containing any of the selected nodes
        # nv is already 0-based internally
        node_set = set(node_indices_0.tolist())  # Convert to list for set creation
        element_mask: NDArray[np.bool_] = np.zeros(self.nele, dtype=bool)

        for i in range(3):  # Check each vertex of triangles
            element_mask |= np.isin(self.nv[i, :], list(node_set))

        # Get unique elements containing selected nodes
        selected_elements = np.where(element_mask)[0]

        if len(selected_elements) == 0:
            return 0.0

        # Calculate area of each selected triangle using shoelace formula
        total_area = 0.0

        for elem_idx in selected_elements:
            # Get the three nodes of this triangle (0-based)
            n1, n2, n3 = self.nv[:, elem_idx]

            # Get coordinates
            x1, y1 = self.x[n1], self.y[n1]
            x2, y2 = self.x[n2], self.y[n2]
            x3, y3 = self.x[n3], self.y[n3]

            # Shoelace formula for triangle area
            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            total_area += area

        return total_area

    def calculate_element_area(
        self,
        element_indices: list[int] | NDArray[np.int_] | None = None,
        index_base: int = 1,
    ) -> NDArray[np.float64]:
        """Calculate areas of triangular elements for selected indices.

        Parameters
        ----------
        element_indices : list[int] | NDArray[np.int_] | None
            Element numbers to evaluate. ``None`` returns areas for every element.
        index_base : int
            0 for zero-based indexing, 1 for one-based indexing (FVCOM default).

        Returns
        -------
        numpy.ndarray
            Array of areas in square metres, in the same order as ``element_indices``
            (or element order if ``element_indices`` is ``None``).
        """
        import numpy as np

        if element_indices is None:
            element_indices = np.arange(
                1 if index_base == 1 else 0, self.nele + (1 if index_base == 1 else 0)
            )

        elem_arr: NDArray[np.int_] = np.asarray(element_indices, dtype=int)

        if index_base == 1:
            elem_idx0 = elem_arr - 1
        else:
            elem_idx0 = elem_arr

        if np.any(elem_idx0 < 0) or np.any(elem_idx0 >= self.nele):
            invalid = elem_arr[(elem_idx0 < 0) | (elem_idx0 >= self.nele)]
            raise ValueError(
                f"Invalid element indices (base-{index_base}): {invalid.tolist()}. "
                f"Valid range: {1 if index_base == 1 else 0} to "
                f"{self.nele if index_base == 1 else self.nele - 1}"
            )

        tri_nodes = self.nv[:, elem_idx0]
        x_coords = self.x[tri_nodes]
        y_coords = self.y[tri_nodes]

        v1x = x_coords[1] - x_coords[0]
        v1y = y_coords[1] - y_coords[0]
        v2x = x_coords[2] - x_coords[0]
        v2y = y_coords[2] - y_coords[0]

        cross = v1x * v2y - v1y * v2x
        areas = 0.5 * np.abs(cross)
        return np.asarray(areas, dtype=float)

    def get_node_element_boundaries(
        self,
        node_indices: list[int] | NDArray[np.int_] | None = None,
        index_base: int = 1,
        return_as: str = "lines",  # "lines" or "polygons"
    ) -> list[list[tuple[float, float]]]:
        """Get boundaries of triangular elements containing specified nodes.

        Parameters
        ----------
        node_indices : list[int] | NDArray[np.int_] | None
            List of node indices. If None, gets boundaries for all elements.
        index_base : int
            0 for zero-based indexing, 1 for one-based indexing (FVCOM default)
        return_as : str
            "lines" returns list of line segments (edges)
            "polygons" returns list of closed triangles

        Returns
        -------
        list[list[tuple[float, float]]]
            List of polylines/polygons as coordinate pairs (lon, lat)
            For "lines": each item is a line segment with 2 points
            For "polygons": each item is a closed triangle with 4 points (first=last)
        """
        import numpy as np

        # Handle None case - all nodes
        if node_indices is None:
            node_indices = np.arange(
                1 if index_base == 1 else 0, self.node + (1 if index_base == 1 else 0)
            )

        # Convert to numpy array
        node_indices_arr: NDArray[np.int_] = np.asarray(node_indices, dtype=int)

        # Convert to zero-based if needed
        if index_base == 1:
            node_indices_0 = node_indices_arr - 1
        else:
            node_indices_0 = node_indices_arr

        # Validate indices
        if np.any(node_indices_0 < 0) or np.any(node_indices_0 >= self.node):
            invalid = node_indices_arr[
                (node_indices_0 < 0) | (node_indices_0 >= self.node)
            ]
            raise ValueError(
                f"Invalid node indices (base-{index_base}): {invalid.tolist()}. "
                f"Valid range: {1 if index_base == 1 else 0} to "
                f"{self.node if index_base == 1 else self.node - 1}"
            )

        # Find all elements containing any of the selected nodes
        node_set = set(node_indices_0.tolist())
        element_mask: NDArray[np.bool_] = np.zeros(self.nele, dtype=bool)

        for i in range(3):  # Check each vertex of triangles
            element_mask |= np.isin(self.nv[i, :], list(node_set))

        # Get unique elements containing selected nodes
        selected_elements = np.where(element_mask)[0]

        if len(selected_elements) == 0:
            return []

        # Use geographic coordinates if available, otherwise use UTM
        if self.lon is not None and self.lat is not None:
            x_coords = self.lon
            y_coords = self.lat
        else:
            x_coords = self.x
            y_coords = self.y

        boundaries = []

        if return_as == "polygons":
            # Return closed triangles
            for elem_idx in selected_elements:
                # Get the three nodes of this triangle (0-based)
                n1, n2, n3 = self.nv[:, elem_idx]

                # Create closed polygon (first point repeated at end)
                polygon = [
                    (float(x_coords[n1]), float(y_coords[n1])),
                    (float(x_coords[n2]), float(y_coords[n2])),
                    (float(x_coords[n3]), float(y_coords[n3])),
                    (float(x_coords[n1]), float(y_coords[n1])),  # Close the polygon
                ]
                boundaries.append(polygon)

        elif return_as == "lines":
            # Return unique edges as line segments
            edges_set = set()

            for elem_idx in selected_elements:
                # Get the three nodes of this triangle (0-based)
                n1, n2, n3 = self.nv[:, elem_idx]

                # Add three edges (sorted to ensure uniqueness)
                edges = [
                    tuple(sorted([n1, n2])),
                    tuple(sorted([n2, n3])),
                    tuple(sorted([n3, n1])),
                ]
                edges_set.update(edges)

            # Convert edges to line segments
            for edge in edges_set:
                n1, n2 = edge
                line = [
                    (float(x_coords[n1]), float(y_coords[n1])),
                    (float(x_coords[n2]), float(y_coords[n2])),
                ]
                boundaries.append(line)
        else:
            raise ValueError(
                f"return_as must be 'lines' or 'polygons', got '{return_as}'"
            )

        return boundaries

    def get_node_control_volume(
        self, node_idx: int, index_base: int = 1
    ) -> list[tuple[float, float]]:
        """Get FVCOM median-dual control volume polygon for a node.

        Parameters
        ----------
        node_idx : int
            Node index
        index_base : int
            0 for zero-based indexing, 1 for one-based indexing (FVCOM default)

        Returns
        -------
        list[tuple[float, float]]
            List of (x, y) coordinates forming the control volume polygon
        """
        import numpy as np

        # Convert to zero-based
        if index_base == 1:
            node_idx = node_idx - 1

        # Validate index
        if node_idx < 0 or node_idx >= self.node:
            raise ValueError(f"Invalid node index: {node_idx}")

        # Find all triangles containing this node
        incident_triangles = self._get_incident_triangles(node_idx)

        if len(incident_triangles) == 0:
            return []

        # Special case: single triangle
        if len(incident_triangles) == 1:
            tri_idx = incident_triangles[0]
            tri_nodes = self.nv[:, tri_idx]

            # Find the other two nodes
            other_nodes = [n for n in tri_nodes if n != node_idx]

            # Calculate triangle centroid
            cx = np.mean(self.x[tri_nodes])
            cy = np.mean(self.y[tri_nodes])

            # Get edge midpoints
            mx1 = (self.x[node_idx] + self.x[other_nodes[0]]) / 2
            my1 = (self.y[node_idx] + self.y[other_nodes[0]]) / 2
            mx2 = (self.x[node_idx] + self.x[other_nodes[1]]) / 2
            my2 = (self.y[node_idx] + self.y[other_nodes[1]]) / 2

            # Create control volume: node -> midpoint1 -> centroid -> midpoint2 -> node
            polygon = [
                (float(self.x[node_idx]), float(self.y[node_idx])),
                (float(mx1), float(my1)),
                (float(cx), float(cy)),
                (float(mx2), float(my2)),
                (float(self.x[node_idx]), float(self.y[node_idx]))
            ]
            return polygon

        # Order triangles counterclockwise around the node
        ordered_triangles = self._order_triangles_ccw(node_idx, incident_triangles)

        # Build control volume polygon
        ordered_triangles = [int(idx) for idx in ordered_triangles]

        def _norm_edge(u: int, v: int) -> tuple[int, int]:
            return (u, v) if u < v else (v, u)

        def _edge_midpoint(edge: tuple[int, int]) -> tuple[float, float]:
            u, v = edge
            return (
                float((self.x[u] + self.x[v]) / 2.0),
                float((self.y[u] + self.y[v]) / 2.0),
            )

        def _edge_angle(edge: tuple[int, int]) -> float:
            u, v = edge
            other = v if u == node_idx else u
            mx, my = _edge_midpoint(edge)
            return float(np.arctan2(my - self.y[node_idx], mx - self.x[node_idx]))

        boundary_edges = self._get_boundary_edges()
        node_boundary_edges = [edge for edge in boundary_edges if node_idx in edge]

        # Build edge → triangles map for adjacency walking
        edge_to_triangles: dict[tuple[int, int], list[int]] = {}
        for tri_idx in incident_triangles:
            tri_idx_int = int(tri_idx)
            tri_nodes = self.nv[:, tri_idx_int]
            for raw_node in tri_nodes:
                other_node = int(raw_node)
                if other_node == node_idx:
                    continue
                edge = _norm_edge(node_idx, other_node)
                edge_to_triangles.setdefault(edge, []).append(tri_idx_int)

        # Select starting edge: prefer boundary edge so the polygon opens along coastline
        if node_boundary_edges:
            node_boundary_edges.sort(key=_edge_angle)
            start_edge = node_boundary_edges[0]
        else:
            first_tri = ordered_triangles[0]
            tri_nodes = self.nv[:, first_tri]
            candidate_edges = [
                _norm_edge(node_idx, int(n)) for n in tri_nodes if int(n) != node_idx
            ]
            start_edge = min(candidate_edges, key=_edge_angle)

        # Identify starting triangle that contains the starting edge
        candidate_tris = edge_to_triangles.get(start_edge, [])
        if not candidate_tris:
            return []

        def _triangle_order_key(tri_idx: int) -> int:
            try:
                return ordered_triangles.index(tri_idx)
            except ValueError:
                return 0

        start_triangle = min(candidate_tris, key=_triangle_order_key)

        polygon: list[tuple[float, float]] = []
        polygon.append(_edge_midpoint(start_edge))

        visited_triangles: set[int] = set()
        current_triangle = start_triangle
        current_edge = start_edge

        while True:
            visited_triangles.add(current_triangle)
            tri_nodes = self.nv[:, current_triangle]
            cx = float(np.mean(self.x[tri_nodes]))
            cy = float(np.mean(self.y[tri_nodes]))
            polygon.append((cx, cy))

            edges_current = [
                _norm_edge(node_idx, int(n)) for n in tri_nodes if int(n) != node_idx
            ]
            if not edges_current:
                break

            if len(edges_current) == 1:
                outgoing_edge = edges_current[0]
            elif edges_current[0] == current_edge:
                outgoing_edge = edges_current[1]
            elif edges_current[1] == current_edge:
                outgoing_edge = edges_current[0]
            else:
                # Choose the edge whose midpoint is closest to the last polygon vertex
                last_point = polygon[-2]
                outgoing_edge = min(
                    edges_current,
                    key=lambda edge: ( _edge_midpoint(edge)[0] - last_point[0]) ** 2
                    + ( _edge_midpoint(edge)[1] - last_point[1]) ** 2,
                )

            polygon.append(_edge_midpoint(outgoing_edge))

            neighbors = [
                tri for tri in edge_to_triangles.get(outgoing_edge, []) if tri != current_triangle
            ]

            if not neighbors:
                current_edge = outgoing_edge
                break

            next_triangle = None
            for candidate in neighbors:
                if candidate not in visited_triangles:
                    next_triangle = candidate
                    break

            if next_triangle is None:
                current_edge = outgoing_edge
                break

            current_edge = outgoing_edge
            current_triangle = next_triangle

        # Remove duplicated final midpoint for interior nodes
        if polygon and np.allclose(polygon[0], polygon[-1]):
            polygon.pop()

        # Boundary nodes include the node itself to trace along the coastline/open boundary
        if node_boundary_edges:
            polygon.append((float(self.x[node_idx]), float(self.y[node_idx])))

        if polygon and polygon[0] != polygon[-1]:
            polygon.append(polygon[0])

        return polygon

    def _get_incident_triangles(self, node_idx: int) -> np.ndarray:
        """Get indices of all triangles containing the specified node (0-based)."""
        import numpy as np

        # Find triangles where node appears in any vertex
        mask = (self.nv[0, :] == node_idx) | (self.nv[1, :] == node_idx) | (self.nv[2, :] == node_idx)
        return np.where(mask)[0]

    def _order_triangles_ccw(self, node_idx: int, triangle_indices: np.ndarray) -> list[int]:
        """Order triangles counterclockwise around a node."""
        import numpy as np

        if len(triangle_indices) == 0:
            return []

        # Calculate centroid of each triangle and its angle relative to the node
        angles = []
        for tri_idx in triangle_indices:
            tri_nodes = self.nv[:, tri_idx]
            cx = np.mean(self.x[tri_nodes])
            cy = np.mean(self.y[tri_nodes])

            # Calculate angle from node to centroid
            angle = np.arctan2(cy - self.y[node_idx], cx - self.x[node_idx])
            angles.append(angle)

        # Sort triangles by angle
        sorted_indices = np.argsort(angles)
        return [triangle_indices[i] for i in sorted_indices]

    def _get_boundary_edges(self) -> set[tuple[int, int]]:
        """Compute (or return cached) set of boundary edges as node index pairs."""
        if self._boundary_edge_cache is not None:
            return self._boundary_edge_cache

        edge_counts: dict[tuple[int, int], int] = {}
        for tri_nodes in self.nv.T:
            n1, n2, n3 = int(tri_nodes[0]), int(tri_nodes[1]), int(tri_nodes[2])
            edges = (
                (n1, n2),
                (n2, n3),
                (n3, n1),
            )
            for u, v in edges:
                key = (u, v) if u < v else (v, u)
                edge_counts[key] = edge_counts.get(key, 0) + 1

        boundary_edges = {edge for edge, count in edge_counts.items() if count == 1}
        self._boundary_edge_cache = boundary_edges
        return boundary_edges

    def _get_boundary_nodes(self) -> set[int]:
        """Return set of node indices that lie on any boundary edge."""
        if self._boundary_node_cache is not None:
            return self._boundary_node_cache

        boundary_edges = self._get_boundary_edges()
        boundary_nodes: set[int] = set()
        for u, v in boundary_edges:
            boundary_nodes.add(u)
            boundary_nodes.add(v)

        self._boundary_node_cache = boundary_nodes
        return boundary_nodes

    def _is_boundary_node(self, node_idx: int, incident_triangles: np.ndarray) -> bool:
        """Check if a node is on the boundary (coastal node)."""
        return node_idx in self._get_boundary_nodes()

    def calculate_node_area_median_dual(
        self,
        node_indices: list[int] | NDArray[np.int_] | None = None,
        index_base: int = 1,
    ) -> float:
        """Calculate total area of median-dual control volumes for specified nodes.

        This uses the FVCOM median-dual method where each node's control volume
        is formed by connecting triangle centroids and edge midpoints.

        Parameters
        ----------
        node_indices : list[int] | NDArray[np.int_] | None
            List of node indices. If None, calculates area for all nodes.
        index_base : int
            0 for zero-based indexing, 1 for one-based indexing (FVCOM default)

        Returns
        -------
        float
            Total area in square meters (assuming x, y are in meters/UTM)
        """
        import numpy as np

        # Handle None case - all nodes
        if node_indices is None:
            node_indices = np.arange(
                1 if index_base == 1 else 0, self.node + (1 if index_base == 1 else 0)
            )

        # Convert to numpy array
        node_indices_arr: NDArray[np.int_] = np.asarray(node_indices, dtype=int)

        # Convert to zero-based if needed
        if index_base == 1:
            node_indices_0 = node_indices_arr - 1
        else:
            node_indices_0 = node_indices_arr

        # Validate indices
        if np.any(node_indices_0 < 0) or np.any(node_indices_0 >= self.node):
            invalid = node_indices_arr[
                (node_indices_0 < 0) | (node_indices_0 >= self.node)
            ]
            raise ValueError(
                f"Invalid node indices (base-{index_base}): {invalid.tolist()}. "
                f"Valid range: {1 if index_base == 1 else 0} to "
                f"{self.node if index_base == 1 else self.node - 1}"
            )

        total_area = 0.0

        # Calculate area for each node's control volume
        for node_idx in node_indices_0:
            # Get control volume polygon
            polygon = self.get_node_control_volume(node_idx, index_base=0)

            if len(polygon) < 3:
                continue

            # Calculate area using shoelace formula
            area = 0.0
            n = len(polygon)
            for i in range(n):
                j = (i + 1) % n
                area += polygon[i][0] * polygon[j][1]
                area -= polygon[j][0] * polygon[i][1]

            area = abs(area) / 2.0
            total_area += area

        return total_area

    def get_node_control_volumes(
        self,
        node_indices: list[int] | NDArray[np.int_] | None = None,
        index_base: int = 1,
    ) -> list[list[tuple[float, float]]]:
        """Get median-dual control volume polygons for specified nodes.

        Parameters
        ----------
        node_indices : list[int] | NDArray[np.int_] | None
            List of node indices. If None, gets volumes for all nodes.
        index_base : int
            0 for zero-based indexing, 1 for one-based indexing (FVCOM default)

        Returns
        -------
        list[list[tuple[float, float]]]
            List of control volume polygons, each as a list of (lon, lat) coordinates
        """
        import numpy as np

        # Handle None case - all nodes
        if node_indices is None:
            node_indices = np.arange(
                1 if index_base == 1 else 0, self.node + (1 if index_base == 1 else 0)
            )

        # Convert to numpy array
        node_indices_arr: NDArray[np.int_] = np.asarray(node_indices, dtype=int)

        volumes = []
        for node_idx in node_indices_arr:
            polygon = self.get_node_control_volume(node_idx, index_base)

            # Convert to geographic coordinates if available
            if self.lon is not None and self.lat is not None and len(polygon) > 0:
                # Need to interpolate since polygon points are at centroids/midpoints
                geo_polygon = []
                for x, y in polygon:
                    # Find nearest node for approximate lon/lat
                    # This is simplified - better would be proper interpolation
                    distances = np.sqrt((self.x - x)**2 + (self.y - y)**2)
                    nearest = np.argmin(distances)

                    # Use offset from nearest node to estimate lon/lat
                    dx = x - self.x[nearest]
                    dy = y - self.y[nearest]

                    # Rough approximation (assuming small distances)
                    # Better would use proper UTM to geographic conversion
                    lon_approx = self.lon[nearest] + dx * 1e-5  # Rough scale
                    lat_approx = self.lat[nearest] + dy * 1e-5

                    geo_polygon.append((lon_approx, lat_approx))

                volumes.append(geo_polygon)
            else:
                volumes.append(polygon)

        return volumes

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"FvcomGrid(nodes={self.node}, elements={self.nele}, "
            f"lonlat={'yes' if self.lon is not None else 'no'})"
        )


# -----------------------------------------------------------------------------
# Convenience wrapper – accept various inputs and always return FvcomGrid
# -----------------------------------------------------------------------------


def get_grid(obj: "FvcomGrid | xr.Dataset | str | Path") -> FvcomGrid:  # type: ignore[name-defined]
    if isinstance(obj, FvcomGrid):
        return obj
    if isinstance(obj, (str, Path)):
        return FvcomGrid.from_dat(obj)
    if isinstance(obj, xr.Dataset):
        # try strict first; if fails, fallback to loose
        try:
            return FvcomGrid.from_dataset(obj)
        except KeyError:
            return FvcomGrid.from_dataset(obj, validate=False)
    raise TypeError(
        f"Unsupported object type for grid extraction: {type(obj).__name__}"
    )

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

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .connectivity import connectivity_three_by_nele, zero_based_connectivity
from .dat_reader import read_dat
from .geo_utils import utm_to_lonlat

# -----------------------------------------------------------------------------
# Helper protocol – minimal Dataset interface
# -----------------------------------------------------------------------------


@runtime_checkable
class _DatasetLike(Protocol):
    def __getitem__(self, key: str) -> Any: ...
    def __contains__(self, key: str) -> bool: ...


# jsasaki 2026-11-05: review round 6 #1: the connectivity helper moved to .connectivity (shared with io)
_connectivity_three_by_nele = connectivity_three_by_nele


# jsasaki 2026-11-05: review round 5 #5: projection helper
def _projection_from_attrs(attrs: Any) -> dict[str, Any]:
    """Read ``utm_zone`` / ``utm_hemisphere`` written by :meth:`FvcomGrid.to_xarray`.

    Datasets without them (older exports, model output) keep the defaults.
    """
    out: dict[str, Any] = {}
    zone = attrs.get("utm_zone") if hasattr(attrs, "get") else None
    if zone is not None:
        z = np.asarray(zone)
        if (
            z.size != 1
            or z.dtype.kind not in "iu"
            or not 1 <= int(z.reshape(-1)[0]) <= 60
        ):
            raise ValueError(
                f"utm_zone attribute must be an integer 1..60 (got {zone!r})"
            )
        out["zone"] = int(z.reshape(-1)[0])
    hemi = attrs.get("utm_hemisphere") if hasattr(attrs, "get") else None
    if hemi is not None:
        if str(hemi) not in ("north", "south"):
            raise ValueError(
                f"utm_hemisphere attribute must be 'north' or 'south' (got {hemi!r})"
            )
        out["northern"] = str(hemi) == "north"
    return out


# ---> jsasaki 2026-11-05: review round 6 #8: one validation for every node/element selector.
# The old np.asarray(..., dtype=int) truncated 1.9 to 1 and read through masked entries.
def _select_indices(
    indices: Any, count: int, index_base: Any, kind: str
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Validate a selector and return ``(as_given, zero_based)`` integer arrays.

    ``None`` selects all ``count`` items. ``index_base`` must be 0 or 1. Masked
    entries, non-numeric or boolean values, and non-integral or non-finite numbers
    are refused before conversion; every index must lie in the valid range.
    """
    if isinstance(index_base, (bool, np.bool_)) or index_base not in (0, 1):
        raise ValueError(f"index_base must be 0 or 1 (got {index_base!r})")
    base = int(index_base)
    if indices is None:
        idx = np.arange(base, count + base)
        return idx, idx - base
    if np.ma.isMaskedArray(indices) and np.ma.getmaskarray(indices).any():
        raise ValueError(f"{kind} indices contain masked entries")
    raw = np.asarray(np.ma.getdata(indices))
    if raw.size == 0:
        arr = np.zeros(raw.shape, dtype=int)
    elif raw.dtype.kind in "iu":
        arr = raw.astype(int)
    elif raw.dtype.kind == "f":
        if not np.isfinite(raw).all():
            raise ValueError(f"{kind} indices contain non-finite values")
        if not np.array_equal(raw, np.round(raw)):
            raise ValueError(
                f"{kind} indices contain non-integral values: "
                f"{raw[raw != np.round(raw)].tolist()}"
            )
        arr = raw.astype(int)
    else:
        raise ValueError(f"{kind} indices must be integers (got dtype {raw.dtype})")
    idx0 = arr - base
    bad = (idx0 < 0) | (idx0 >= count)
    if np.any(bad):
        raise ValueError(
            f"Invalid {kind} indices (base-{base}): {np.atleast_1d(arr[bad]).tolist()}. "
            f"Valid range: {base} to {count - 1 + base}"
        )
    return arr, idx0


# <--- jsasaki 2026-11-05


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

    # jsasaki 2026-11-05: review round 6 #2: x/y hold lon/lat (degrees), not projected metres
    geographic: bool = False

    # cached topology helpers -----------------------------------------
    _boundary_edge_cache: set[tuple[int, int]] | None = field(
        default=None, init=False, repr=False
    )
    _boundary_node_cache: set[int] | None = field(default=None, init=False, repr=False)
    # jsasaki 2026-11-05: review round 6 #7: fingerprint of the nv the caches were built from
    _topology_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)
    # jsasaki 2026-11-05: review round 6 #7: set while one public call loops over nodes (nv fixed)
    _topology_pinned: bool = field(default=False, init=False, repr=False)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dataset(
        cls,
        ds: _DatasetLike,
        *,
        validate: bool = True,
        # jsasaki 2026-11-05: review round 4 #4: explicit indexing base for ambiguous inputs
        start_index: int | None = None,
    ) -> "FvcomGrid":
        """Build from an xarray.Dataset that already contains grid variables.

        The indexing base of ``nv`` is taken, in order, from the ``start_index``
        argument (0 or 1), from ``nv.attrs["start_index"]`` (0 or 1), and
        otherwise inferred: ``nv`` whose minimum is 1 is taken as one-based
        (FVCOM output), anything else as zero-based. The inference is ambiguous
        for zero-based connectivity that does not reference node 0 (e.g. a subset
        keeping unused nodes); pass ``start_index`` or set the attribute then.
        :meth:`to_xarray` writes ``start_index=0``.
        """
        req = ("x", "y", "nv")
        miss = [n for n in req if n not in ds]
        if miss and validate:
            raise KeyError("Dataset missing grid vars: " + ", ".join(miss))

        # ---- fallback for loose validation --------------------------------
        # jsasaki 2026-11-05: review round 5 #3: nv_zero is zero-based by definition
        nv_zero_adopted = False
        # jsasaki 2026-11-05: review round 6 #2: coordinate type survives a to_xarray round trip
        geographic = getattr(ds, "attrs", {}).get("xy_coordinates") == "geographic"
        if not validate:
            # nv_zero → nv
            if "nv" not in ds and "nv_zero" in ds:
                ds = ds.assign(nv=ds["nv_zero"])
                # jsasaki 2026-11-05: review round 5 #3: keep its base unless the caller overrides it
                nv_zero_adopted = True
            # if planar x/y missing but lon/lat present, copy for plotting
            if "x" not in ds and "lon" in ds:
                ds = ds.assign(x=ds["lon"])
                # jsasaki 2026-11-05: review round 6 #2: degrees, so metric areas must refuse
                geographic = True
            if "y" not in ds and "lat" in ds:
                ds = ds.assign(y=ds["lat"])
                # jsasaki 2026-11-05: review round 6 #2: degrees, so metric areas must refuse
                geographic = True
            miss = [n for n in req if n not in ds]
            if miss:
                raise KeyError("Dataset still missing vars: " + ", ".join(miss))

        kw: dict[str, Any] = {
            "x": np.asarray(ds["x"].values, dtype=float),
            "y": np.asarray(ds["y"].values, dtype=float),
        }
        # ------------------------------------------------------------------
        # nv: ensure (3, nele) shape & 0-based
        # ---> jsasaki 2026-11-05: review round 6 #1: one shared normalisation (argument, then the
        # start_index attribute, then the inference: an adopted nv_zero is zero-based, otherwise
        # min()==1 means one-based), validated and range-checked against the node count
        nv_raw = zero_based_connectivity(
            ds["nv"],
            start_index=start_index,
            fallback=0 if nv_zero_adopted else "min_is_one",
            n_nodes=kw["x"].size,
        )
        # <--- jsasaki 2026-11-05
        kw["nv"] = nv_raw

        for name in ("lon", "lat", "lonc", "latc"):
            if name in ds:
                kw[name] = np.asarray(ds[name].values, dtype=float)
        # compute element centres
        nv = kw["nv"]
        kw["xc"] = kw["x"][nv].mean(axis=0)
        kw["yc"] = kw["y"][nv].mean(axis=0)
        # jsasaki 2026-11-05: review round 5 #5: restore the projection written by to_xarray
        kw.update(_projection_from_attrs(getattr(ds, "attrs", {})))
        # jsasaki 2026-11-05: review round 6 #2: record that x/y are lon/lat
        kw["geographic"] = geographic

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
        # jsasaki 2026-11-05: review round 3 #19: both bounds (a negative index wraps silently)
        if data["nv"].min() < 0 or data["nv"].max() >= data["x"].size:
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
                # jsasaki 2026-11-05: review round 4 #4: the exported base is stated
                "nv": (("three", "nele"), self.nv, {"start_index": 0}),
            },
            coords={
                "node": ("node", np.arange(self.node)),
                "nele": ("nele", np.arange(self.nele)),
                "three": ("three", np.arange(3)),
            },
            attrs={"cf_role": "mesh_topology"},
        )
        # ---> jsasaki 2026-11-05: review round 5 #5: the projection survives a round trip
        if self.zone is not None:
            ds.attrs["utm_zone"] = int(self.zone)
        ds.attrs["utm_hemisphere"] = "north" if self.northern else "south"
        # <--- jsasaki 2026-11-05
        # jsasaki 2026-11-05: review round 6 #2: a geographic grid stays geographic after a round trip
        if self.geographic:
            ds.attrs["xy_coordinates"] = "geographic"
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
    # ---> jsasaki 2026-11-05: review round 6 #2: areas are in m^2 only when x/y are metres; a
    # geographic grid (x/y copied from lon/lat by the loose fallback) would return degrees^2
    def _require_metric(self, what: str) -> None:
        if self.geographic:
            raise ValueError(
                f"{what}: this grid's x/y are geographic lon/lat (degrees), so the "
                "area would be in degrees^2, not m^2; build the grid from projected "
                "x/y (e.g. FvcomGrid.from_dat, or a dataset with x/y in metres)"
            )

    # <--- jsasaki 2026-11-05
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

        # jsasaki 2026-11-05: review round 6 #2: metric areas need projected x/y
        self._require_metric("calculate_node_area")
        # jsasaki 2026-11-05: review round 6 #8: validated selector (masks, fractions, base, range)
        node_indices_arr, node_indices_0 = _select_indices(
            node_indices, self.node, index_base, "node"
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

        # jsasaki 2026-11-05: review round 6 #2: metric areas need projected x/y
        self._require_metric("calculate_element_area")
        # jsasaki 2026-11-05: review round 6 #8: validated selector (masks, fractions, base, range)
        elem_arr, elem_idx0 = _select_indices(
            element_indices, self.nele, index_base, "element"
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

        # jsasaki 2026-11-05: review round 6 #8: validated selector (masks, fractions, base, range)
        node_indices_arr, node_indices_0 = _select_indices(
            node_indices, self.node, index_base, "node"
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

        # ---> jsasaki 2026-11-05: review round 6 #8: one integral, unmasked scalar in range
        if np.ndim(node_idx) != 0:
            raise ValueError(f"node_idx must be a single index (got {node_idx!r})")
        try:
            _, idx0 = _select_indices(
                np.atleast_1d(node_idx), self.node, index_base, "node"
            )
        except ValueError as exc:
            raise ValueError(f"Invalid node index: {node_idx!r} ({exc})") from None
        node_idx = int(idx0[0])
        # <--- jsasaki 2026-11-05

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
            cx = float(np.mean(self.x[tri_nodes]))
            cy = float(np.mean(self.y[tri_nodes]))

            # Get edge midpoints
            mx1 = (self.x[node_idx] + self.x[other_nodes[0]]) / 2
            my1 = (self.y[node_idx] + self.y[other_nodes[0]]) / 2
            mx2 = (self.x[node_idx] + self.x[other_nodes[1]]) / 2
            my2 = (self.y[node_idx] + self.y[other_nodes[1]]) / 2

            # Create control volume: node -> midpoint1 -> centroid -> midpoint2 -> node
            simple_polygon = [
                (float(self.x[node_idx]), float(self.y[node_idx])),
                (float(mx1), float(my1)),
                (float(cx), float(cy)),
                (float(mx2), float(my2)),
                (float(self.x[node_idx]), float(self.y[node_idx])),
            ]
            return simple_polygon

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
        # jsasaki 2026-11-05: review round 5 #1: last edge whose midpoint was appended
        last_edge: tuple[int, int] | None = None

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
                    key=lambda edge: (_edge_midpoint(edge)[0] - last_point[0]) ** 2
                    + (_edge_midpoint(edge)[1] - last_point[1]) ** 2,
                )

            polygon.append(_edge_midpoint(outgoing_edge))
            # jsasaki 2026-11-05: review round 5 #1: remember the edge for the topological closure test
            last_edge = outgoing_edge

            neighbors = [
                tri
                for tri in edge_to_triangles.get(outgoing_edge, [])
                if tri != current_triangle
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

        # ---> jsasaki 2026-11-05: review round 5 #1: decide the closure by topology. np.allclose
        # (rtol=1e-5) treated distinct boundary midpoints as equal at UTM magnitudes and dropped a vertex.
        # Remove duplicated final midpoint for interior nodes: the walk returned to the start edge
        if len(polygon) > 1 and last_edge == start_edge:
            polygon.pop()
        # <--- jsasaki 2026-11-05

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
        mask = (
            (self.nv[0, :] == node_idx)
            | (self.nv[1, :] == node_idx)
            | (self.nv[2, :] == node_idx)
        )
        return np.where(mask)[0]

    def _order_triangles_ccw(
        self, node_idx: int, triangle_indices: np.ndarray
    ) -> list[int]:
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

    # ---> jsasaki 2026-11-05: review round 6 #7: the caches follow nv. Replacing nv or editing it in
    # place changes the fingerprint (shape, dtype, content digest), which drops both caches.
    def _sync_topology_cache(self) -> None:
        if self._topology_pinned:
            return
        nv = np.ascontiguousarray(self.nv)
        key = (
            nv.shape,
            nv.dtype.str,
            hashlib.blake2b(nv.tobytes(), digest_size=16).digest(),
        )
        if key != self._topology_key:
            self._boundary_edge_cache = None
            self._boundary_node_cache = None
            self._topology_key = key

    @contextmanager
    def _pinned_topology(self) -> Iterator[None]:
        """Check nv once, then skip the per-node fingerprint inside a loop."""
        if self._topology_pinned:
            yield
            return
        self._sync_topology_cache()
        self._topology_pinned = True
        try:
            yield
        finally:
            self._topology_pinned = False

    # <--- jsasaki 2026-11-05

    def _get_boundary_edges(self) -> set[tuple[int, int]]:
        """Compute (or return cached) set of boundary edges as node index pairs."""
        # jsasaki 2026-11-05: review round 6 #7: a cache built from another nv is discarded
        self._sync_topology_cache()
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
        # jsasaki 2026-11-05: review round 6 #7: a cache built from another nv is discarded
        self._sync_topology_cache()
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

        # jsasaki 2026-11-05: review round 6 #2: metric areas need projected x/y
        self._require_metric("calculate_node_area_median_dual")
        # jsasaki 2026-11-05: review round 6 #8: validated selector (masks, fractions, base, range)
        node_indices_arr, node_indices_0 = _select_indices(
            node_indices, self.node, index_base, "node"
        )

        total_area = 0.0

        # jsasaki 2026-11-05: review round 6 #7: nv is fingerprinted once for the whole loop
        with self._pinned_topology():
            # Calculate area for each node's control volume
            # jsasaki 2026-11-05: review round 6 #8: a node counts once, as in calculate_node_area
            for node_idx in np.unique(node_indices_0):
                # Get control volume polygon
                polygon = self.get_node_control_volume(node_idx, index_base=0)

                if len(polygon) < 3:
                    continue

                # ---> jsasaki 2026-11-05: review round 5 #1: shoelace on coordinates relative to the
                # first vertex, so the area does not lose precision (or change) with the mesh origin
                pts = np.asarray(polygon, dtype=float)
                px = pts[:, 0] - pts[0, 0]
                py = pts[:, 1] - pts[0, 1]
                area = float(np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py))
                # <--- jsasaki 2026-11-05

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
        # ---> jsasaki 2026-11-05: review round 6 #8: validated selector (one polygon per entry)
        _, node_indices_0 = _select_indices(node_indices, self.node, index_base, "node")
        # <--- jsasaki 2026-11-05

        # ---> jsasaki 2026-11-05: review round 6 #6: polygon vertices (centroids, edge midpoints)
        # are transformed with the grid's own projection; the old nearest-node offset with a fixed
        # 1e-5 deg/m displaced them by tens of metres. x/y that already are lon/lat are returned as
        # they are; projected x/y without a UTM zone cannot be transformed and are refused.
        to_geo = self.lon is not None and self.lat is not None
        xy_is_lonlat = self.geographic or (
            to_geo
            and np.array_equal(self.lon, self.x)
            and np.array_equal(self.lat, self.y)
        )
        if to_geo and not xy_is_lonlat and self.zone is None:
            raise ValueError(
                "get_node_control_volumes: the grid has lon/lat but no UTM zone, so the "
                "polygon vertices cannot be converted; set FvcomGrid.zone (and northern)"
            )

        volumes: list[list[tuple[float, float]]] = []
        # jsasaki 2026-11-05: review round 6 #7: nv is fingerprinted once for the whole loop
        with self._pinned_topology():
            for node_idx in np.atleast_1d(node_indices_0):
                polygon = self.get_node_control_volume(int(node_idx), index_base=0)
                if to_geo and not xy_is_lonlat and len(polygon) > 0:
                    pts = np.asarray(polygon, dtype=float)
                    assert self.zone is not None  # checked above
                    plon, plat = utm_to_lonlat(
                        pts[:, 0], pts[:, 1], zone=self.zone, northern=self.northern
                    )
                    volumes.append(
                        [(float(a), float(b)) for a, b in zip(plon, plat, strict=True)]
                    )
                else:
                    volumes.append(polygon)
        # <--- jsasaki 2026-11-05

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

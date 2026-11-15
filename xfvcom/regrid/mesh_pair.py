"""Strict mesh-bundle primitives; array indices and maps are always zero-based.

Mapped nodes are frozen, including the patch rim. Only matched vertex triples
are frozen elements: retaining three nodes alone does not retain an element.
Depth/Coriolis tables are read with xfvcom after checking their complete shape.
"""

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import polygonize, unary_union

from .. import fvcom_input as _fvcom_input
from ..fvcom_input import (
    INT,
    REAL,
    list_read,
    locate_records,
    physical_lines,
    scan_count_headers,
)
from ..grid.dat_reader import read_dat
from ..io.dep_reader import read_dep as _xf_read_dep


@dataclass(frozen=True)
class Mesh:
    """Serialized mesh geometry, CCW triangles, and positive-down depths."""

    xy: np.ndarray
    triangles: np.ndarray
    depths: np.ndarray


@dataclass(frozen=True)
class Classification:
    """Boolean masks on the base and refined node/element axes."""

    base_frozen_nodes: np.ndarray
    frozen_nodes: np.ndarray
    new_nodes: np.ndarray
    base_frozen_elements: np.ndarray
    frozen_elements: np.ndarray
    new_elements: np.ndarray


def _table(path, header, columns):
    lines = Path(path).read_text().splitlines()
    match = (
        re.fullmatch(re.escape(header) + r"\s*=\s*(\d+)\s*", lines[0])
        if lines
        else None
    )
    if not match:
        raise ValueError(f"{path}: expected '{header} = N' header")
    count = int(match[1])
    rows = [line.split() for line in lines[1:]]
    if len(rows) != count or any(len(row) != columns for row in rows):
        raise ValueError(f"{path}: expected exactly {count} rows of {columns} fields")
    return rows


def _boundary_table(path, header, types):
    lines = physical_lines(Path(path))
    # The checkout's scanner requires positive counts. For boundary headers only,
    # validate zero using an equal-width positive surrogate, preserving every
    # syntax, duplicate, position and 400-byte-buffer check. Grid scanning stays positive.
    scanned, zero = [], False
    for lineno, raw in lines:
        full, seen = _fvcom_input._views(raw)
        if (
            full == seen
            and full
            and full[0] == header
            and re.fullmatch(r"[+-]?0+", full[1])
        ):
            before, sep, value = raw.partition("=")
            body, comment, tail = value.partition("!")
            pos = body.rfind("0")
            body = body[:pos].replace("-", "+") + "1" + body[pos + 1 :]
            raw = before + sep + body + comment + tail
            zero = True
        scanned.append((lineno, raw))
    counts = scan_count_headers(
        Path(path), scanned, {header: "count"}, 1, block="boundary records"
    )
    if zero:
        counts["count"] = 0
    if "count" not in counts:
        raise ValueError(f"{path}: expected '{header} = N' header")
    count = counts["count"]
    rows, pos = [], 1
    while pos < len(lines):
        record = list_read(lines, pos, types)
        if record.status != "ok":
            raise ValueError(
                f"{path}: invalid Fortran boundary record: {record.message or record.status}"
            )
        rows.append(record.values)
        pos = record.end + 1
    if len(rows) != count:
        raise ValueError(
            f"{path}: expected exactly {count} rows of {len(types)} fields"
        )
    return rows


def _integers(values, label):
    values = np.asarray(values)
    if values.dtype.kind not in "iu":
        raise ValueError(f"{label} must have integer dtype")
    return values


def read_grd(path):
    """Return (xy[N,2], triangles[E,3]) using xfvcom's strict FVCOM reader."""
    # Use the reader's FVCOM scanner, including multiline/blank-record semantics.
    # Every successful data READ consumes at least one physical record. Counts
    # exceeding all available records cannot possibly describe this file.
    checked_path = Path(path)
    lines = physical_lines(checked_path)
    start = locate_records(
        checked_path, lines, (INT, INT, INT, INT), block="connectivity block"
    )
    counts = scan_count_headers(
        checked_path,
        lines,
        {"Node Number": "node", "Cell Number": "nele"},
        start,
        block="connectivity block",
    )
    available = len(lines) - start
    if sum(counts.values()) > available:
        raise ValueError(
            f"{path}: declared grid counts exceed available records ({available})"
        )
    data = read_dat(path)
    xy = np.column_stack((data["x"], data["y"]))
    triangles = data["nv"].T.copy()
    _validate_geometry(xy, triangles)
    return xy, triangles


def read_dep(path):
    """Return (xy, depth); reject extra/missing rows and nonfinite values."""
    _table(path, "Node Number", 3)
    data = _xf_read_dep(path)
    xy = np.column_stack((data.x, data.y))
    if not np.isfinite(xy).all() or not np.isfinite(data.h).all():
        raise ValueError(f"{path}: nonfinite coordinate or value")
    return xy, data.h


def read_cor(path):
    """Return (xy, latitude in degrees), the Cartesian FVCOM cor format."""
    xy, latitude = read_dep(path)
    if (np.abs(latitude) > 90).any():
        raise ValueError(f"{path}: Coriolis latitude outside -90..90 degrees")
    return xy, latitude


def read_obc(path):
    """Return (zero-based node ids, types), preserving boundary row order."""
    rows = _boundary_table(path, "OBC Node Number", (INT, INT, INT))
    data = np.array(rows, dtype=np.int64).reshape(-1, 3)
    if not np.array_equal(data[:, 0], np.arange(1, len(data) + 1)):
        raise ValueError(f"{path}: OBC counters must run 1..N")
    ids, types = data[:, 1] - 1, data[:, 2]
    if (ids < 0).any() or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path}: OBC node ids must be positive and unique")
    if ((types < 1) | (types > 10)).any():
        raise ValueError(f"{path}: OBC types must be 1..10")
    return ids, types


def read_spg(path):
    """Return (zero-based node ids, radius/damping[N,2]), in file order."""
    rows = _boundary_table(path, "Sponge Node Number", (INT, REAL, REAL))
    ids = np.array([row[0] - 1 for row in rows], dtype=np.int64)
    values = np.array([row[1:] for row in rows], dtype=float).reshape(-1, 2)
    if (ids < 0).any() or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path}: sponge node ids must be positive and unique")
    if (
        not np.isfinite(values).all()
        or (values[:, 0] <= 0).any()
        or (values[:, 1] < 0).any()
    ):
        raise ValueError(f"{path}: invalid sponge radius/damping")
    return ids, values


def _validate_geometry(xy, triangles):
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError("Mesh coordinates must be a finite nonempty (N, 2) array")
    _integers(triangles, "triangles")
    if (
        triangles.ndim != 2
        or triangles.shape[1] != 3
        or not len(triangles)
        or (triangles < 0).any()
        or (triangles >= len(xy)).any()
    ):
        raise ValueError(
            "Triangles must be a nonempty (E, 3) array of valid node indices"
        )
    p, q, r = (xy[triangles[:, k]] for k in range(3))
    area2 = (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (
        r[:, 0] - p[:, 0]
    )
    if not np.isfinite(area2).all() or (area2 <= 0).any():
        raise ValueError("Element orientation must be CCW with positive finite area")
    if len(np.unique(np.sort(triangles, axis=1), axis=0)) != len(triangles):
        raise ValueError(
            "Duplicate element vertex triple; correspondence is not unique"
        )
    if len(np.unique(triangles)) != len(xy):
        raise ValueError("Mesh contains unreferenced nodes")


def read_mesh(grd, dep):
    """Read serialized geometry/depths, allowing six-decimal XY serialization only.

    The finishing writer rounds new XY to six decimals. Mapped XY equality is
    checked separately against the base grd, without this tolerance.
    """
    xy, triangles = read_grd(grd)
    dep_xy, depths = read_dep(dep)
    # Compare the writer's decimal round trip itself. A half-micrometre
    # absolute tolerance fails at large northings because binary subtraction
    # can exceed it by one ULP even for correctly rounded decimal text.
    serialized_xy = np.array([[float(f"{value:.6f}") for value in row] for row in xy])
    if dep_xy.shape != xy.shape or not np.all(
        (dep_xy == xy) | (dep_xy == serialized_xy)
    ):
        raise ValueError(
            f"{dep}: depth coordinates do not match grd at write precision"
        )
    return Mesh(xy, triangles, depths)


def validate_node_map(node_map, base, refined):
    """Validate old->new integer map (shape Nbase, -1 removed), exactly on mapped data."""
    nm = _integers(node_map, "node_map")
    if nm.shape != (len(base.xy),):
        raise ValueError("node_map shape must be (number of base nodes,)")
    if (nm < -1).any() or (nm >= len(refined.xy)).any():
        raise ValueError(
            "node_map bounds must be -1 or a zero-based refined node index"
        )
    kept = nm >= 0
    target = nm[kept]
    if len(np.unique(target)) != len(target):
        raise ValueError("Duplicate target in node_map; mapped nodes must be unique")
    if not np.array_equal(base.xy[kept], refined.xy[target]):
        raise ValueError("Mapped frozen coordinates differ from the base")
    if not np.array_equal(base.depths[kept], refined.depths[target]):
        raise ValueError("Mapped frozen depths differ from the base")
    return nm


def build_element_map(base, refined, node_map):
    """Return old->new element map, -1 unmatched; cyclic permutations preserve orientation."""
    nm = validate_node_map(node_map, base, refined)
    _validate_geometry(base.xy, base.triangles)
    _validate_geometry(refined.xy, refined.triangles)
    lookup = {tuple(sorted(row)): i for i, row in enumerate(refined.triangles.tolist())}
    result = np.full(len(base.triangles), -1, dtype=np.int64)
    for i, row in enumerate(nm[base.triangles]):
        if (row < 0).any():
            continue
        target = lookup.get(tuple(sorted(row.tolist())))
        if target is None:
            continue
        actual = refined.triangles[target]
        if not any(np.array_equal(np.roll(row, shift), actual) for shift in range(3)):
            raise ValueError(f"Element {i + 1}: mapped orientation changed")
        result[i] = target
    if len(np.unique(result[result >= 0])) != int((result >= 0).sum()):
        raise ValueError("Element correspondence is not unique")
    return result


def classify_mesh(node_map, elem_map, n_nodes, n_elements):
    """Classify mapped nodes and matched elements on both axes."""
    nm = _integers(node_map, "node_map")
    em = _integers(elem_map, "elem_map")
    if nm.ndim != 1 or em.ndim != 1:
        raise ValueError("Maps must be one-dimensional")
    for values, size in ((nm, n_nodes), (em, n_elements)):
        if (values < -1).any() or (values >= size).any():
            raise ValueError("Map bounds invalid for classification")
        if len(np.unique(values[values >= 0])) != int((values >= 0).sum()):
            raise ValueError("Map correspondence is not unique")
    fn = np.zeros(n_nodes, bool)
    fe = np.zeros(n_elements, bool)
    fn[nm[nm >= 0]] = True
    fe[em[em >= 0]] = True
    return Classification(nm >= 0, fn, ~fn, em >= 0, fe, ~fe)


def footprint_polygon(base, elem_map):
    """Union unmatched base triangles in mesh coordinates; empty MultiPolygon for identity."""
    em = _integers(elem_map, "elem_map")
    if em.shape != (len(base.triangles),):
        raise ValueError("elem_map shape must be (number of base elements,)")
    if (em < -1).any():
        raise ValueError(
            "elem_map: entries must be -1 (unmatched) or a refined element index"
        )
    polygons = [Polygon(base.xy[row]) for row in base.triangles[em < 0]]
    result = unary_union(polygons) if polygons else MultiPolygon([])
    if not result.is_valid or result.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("Footprint union is not a valid polygon")
    return result


def mesh_edges(triangles):
    """Return unique undirected zero-based edges in sorted order."""
    tri = _integers(triangles, "triangles")
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("triangles: expected an (E, 3) integer array")
    if (tri < 0).any():
        raise ValueError("triangles: negative node index")
    return np.unique(
        np.sort(np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1),
        axis=0,
    )


def r_factors(depths, edges):
    """Recompute |hi-hj|/(hi+hj), rejecting nonpositive/nonfinite depths."""
    h = np.asarray(depths)
    if h.dtype.kind not in "fiu":
        raise ValueError("r-factor requires real numeric depths")
    # float64 working type: integer depths would wrap on subtraction or
    # overflow on addition; a wider float must fit before the cast
    if (
        h.dtype.kind == "f"
        and h.size
        and bool((np.abs(h) > np.finfo(np.float64).max).any())
    ):
        raise ValueError("r-factor requires depths representable in float64")
    h = h.astype(np.float64)
    if h.ndim != 1 or not np.isfinite(h).all() or (h <= 0).any():
        raise ValueError("r-factor requires finite positive depths")
    e = _integers(edges, "edges")
    if e.ndim != 2 or e.shape[1] != 2:
        raise ValueError("edges: expected a (K, 2) integer array")
    if (e < 0).any() or (e >= len(h)).any():
        raise ValueError("edges: node index outside the depth vector")
    pair = h[e]
    return np.abs(pair[:, 0] - pair[:, 1]) / (pair[:, 0] + pair[:, 1])


def to_one_based(indices):
    """Convert valid zero-based indices to FVCOM ids; never turn -1 into zero."""
    ids = _integers(indices, "indices")
    if (ids < 0).any():
        raise ValueError("Cannot convert a removed/negative index to one-based ids")
    # int64 before adding one: a narrow unsigned dtype would wrap 255 to 0
    if ids.size and int(ids.max()) >= np.iinfo(np.int64).max:
        raise ValueError("index too large for a one-based id")
    return ids.astype(np.int64) + 1


def reindex_one_based(ids, node_map):
    """Re-index FVCOM ids through node_map, refusing removed or invalid source nodes."""
    ids = _integers(ids, "FVCOM ids")
    nm = _integers(node_map, "node_map")
    if nm.ndim != 1:
        raise ValueError("node_map must be one-dimensional")
    if (nm < -1).any():
        raise ValueError(
            "node_map: entries must be -1 (removed) or a refined node index"
        )
    if (ids < 1).any() or (ids > len(nm)).any():
        raise ValueError("FVCOM ids outside the base node range")
    return to_one_based(nm[ids - 1])


def validate_boundary_nodes(ids, node_map, base, refined, label):
    """Require every boundary node to map to a frozen node with its original depth."""
    ids = _integers(ids, label)
    node_map = _integers(node_map, "node_map")
    if ids.ndim != 1 or node_map.ndim != 1:
        raise ValueError(f"{label}: ids and node_map must be one-dimensional")
    if len(node_map) != len(base.xy):
        raise ValueError(f"{label}: node_map length differs from the base node count")
    if (ids < 0).any() or (ids >= len(node_map)).any():
        raise ValueError(f"{label}: node outside base range")
    mapped = node_map[ids]
    if (mapped >= len(refined.xy)).any():
        raise ValueError(f"{label}: node_map points outside the refined mesh")
    if (mapped < 0).any():
        raise ValueError(f"{label}: every node must be frozen; a node was removed")
    if not np.array_equal(
        refined.depths[mapped], base.depths[ids]
    ) or not np.array_equal(refined.xy[mapped], base.xy[ids]):
        raise ValueError(f"{label}: frozen coordinates/depths changed")
    return mapped


def validate_g0(base, refined, node_map, finishing_report, obc_ids, sponge_ids):
    """Run mesh-file G0; return JSON-ready recomputed statistics or raise ValueError.

    Only edges inherited from the base can use the frozen-edge exception.
    A new diagonal between retained nodes still has to meet the new-edge limit.
    """
    validate_node_map(node_map, base, refined)
    em = build_element_map(base, refined, node_map)
    cls = classify_mesh(node_map, em, len(refined.xy), len(refined.triangles))
    validate_finishing_schema(finishing_report)
    report = finishing_report
    if (
        report.get("converged_at_write_precision") is not True
        or report.get("rfactor_report", {}).get("converged") is not True
    ):
        raise ValueError("Finishing report is not converged")
    for key, value in (("hmin_m", 3.0), ("hmax_m", 300.0), ("rfactor", 0.2)):
        if report.get(key) != value:
            raise ValueError(f"Finishing report {key} must be {value}")
    tol = report.get("write_precision_tolerance")
    expected_tol = 0.2 * 1e-6 + 1e-9
    if (
        not isinstance(tol, (int, float))
        or isinstance(tol, bool)
        or not finite_number(tol)
        or not 0 <= tol <= expected_tol
    ):
        raise ValueError("Invalid finishing serialization tolerance (maximum 2.01e-7)")
    if not np.isfinite(refined.depths).all() or (refined.depths < 3).any():
        raise ValueError("Finished depth is nonfinite or below the 3 m floor")
    if (refined.depths > 300).any():
        raise ValueError("Finished depth exceeds the 300 m cap")
    validate_boundary_nodes(obc_ids, node_map, base, refined, "OBC")
    validate_boundary_nodes(sponge_ids, node_map, base, refined, "Sponge")
    old_edges = mesh_edges(base.triangles)
    old_r = r_factors(base.depths, old_edges)
    inherited = {}
    for edge, value in zip(old_edges, old_r):
        mapped = node_map[edge]
        if (mapped >= 0).all():
            inherited[tuple(sorted(mapped.tolist()))] = value
    edges = mesh_edges(refined.triangles)
    values = r_factors(refined.depths, edges)
    frozen_pair = cls.frozen_nodes[edges].all(axis=1)
    inherited_mask = np.zeros(len(edges), bool)
    for i in np.flatnonzero(frozen_pair):
        key = tuple(edges[i])
        if key in inherited:
            if values[i] != inherited[key]:
                raise ValueError(
                    "Frozen-frozen edge r-factor differs exactly from base"
                )
            inherited_mask[i] = True
    new_edges = ~inherited_mask
    bad = new_edges & (values > 0.2 + tol)
    if bad.any():
        raise ValueError(
            f"r-factor violation on {bad.sum()} new/moved-depth edge(s): "
            f"max {values[bad].max():.12g} > 0.2 + {tol:.12g}"
        )
    maximum = lambda mask: float(values[mask].max()) if mask.any() else 0.0
    return {
        "verdict": "PASS",
        "nodes": len(refined.xy),
        "elements": len(refined.triangles),
        "frozen_nodes": int(cls.frozen_nodes.sum()),
        "new_nodes": int(cls.new_nodes.sum()),
        "removed_nodes": int((node_map < 0).sum()),
        "frozen_elements": int(cls.frozen_elements.sum()),
        "new_elements": int(cls.new_elements.sum()),
        "unmatched_base_elements": int((em < 0).sum()),
        "depth_min_m": float(refined.depths.min()),
        "depth_max_m": float(refined.depths.max()),
        "frozen_depths_exact": True,
        "mapped_coordinates_exact": True,
        "finishing_converged": True,
        "serialization_tolerance": float(tol),
        "max_r_new_or_moved_depth": maximum(new_edges),
        "max_r_frozen_pair": maximum(frozen_pair),
        "inherited_frozen_edges": int(inherited_mask.sum()),
        "inherited_frozen_edges_above_0p2": int(
            (inherited_mask & (values > 0.2)).sum()
        ),
        "frozen_edge_r_exact": True,
        "obc_frozen_nodes": len(obc_ids),
        "sponge_frozen_nodes": len(sponge_ids),
        "new_edge_violations": 0,
    }


def finite_number(value):
    # Compare Python numbers before any float/NumPy conversion of unbounded JSON integers.
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and -sys.float_info.max <= value <= sys.float_info.max
        and math.isfinite(value)
    )


def validate_finishing_schema(report):
    """Reject malformed containers/fields before accessing finishing evidence."""
    if not isinstance(report, dict):
        raise ValueError("Finishing report: expected an object")
    for key, kind in (
        ("rfactor_report", dict),
        ("variant", str),
        ("scope", str),
        ("n_movable", int),
    ):
        if not isinstance(report.get(key), kind) or (
            kind is int and isinstance(report.get(key), bool)
        ):
            raise ValueError(f"Finishing report.{key}: expected {kind.__name__}")
    for key in ("hmin_m", "hmax_m", "rfactor"):
        value = report.get(key)
        if not finite_number(value):
            raise ValueError(f"Finishing report.{key}: expected a finite number")


def validate_topology(mesh):
    """Independently verify the actual triangulation, including holes and overlap.

    Directed boundary edges are traced by node identity (slit-wall copies remain
    distinct). Polygonized boundary faces with nonzero oriented winding give
    outer minus holes, including disconnected islands. Triangle union additionally detects
    overlaps even if boundary area errors would cancel. Area tolerances allow
    only floating-point round-off, using local coordinates for area arithmetic.
    """
    xy, tri = mesh.xy, mesh.triangles
    _validate_geometry(xy, tri)
    directed = np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    edges, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if (counts > 2).any():
        raise ValueError(
            "Nonmanifold edge: every edge must belong to one or two elements"
        )
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    interior = starts[counts == 2]
    if not np.array_equal(
        directed[order[interior]], directed[order[interior + 1]][:, ::-1]
    ):
        raise ValueError("Overlapping elements: shared edge has equal directions")
    boundary = directed[counts[inverse] == 1]
    outgoing = {}
    incoming = {}
    for a, b in boundary.tolist():
        if a in outgoing or b in incoming:
            raise ValueError("Boundary does not form disjoint loops by node identity")
        outgoing[a] = b
        incoming[b] = a
    if not outgoing or set(outgoing) != set(incoming):
        raise ValueError("Boundary loops are missing or open")
    local = xy - xy[0]
    loops = 0
    while outgoing:
        start = next(iter(outgoing))
        node = start
        while True:
            node = outgoing.pop(node)
            if node == start:
                break
        loops += 1
    # Slit walls legitimately retrace coordinates with distinct node identities.
    # Polygonize their geometric linework, then use oriented winding to exclude
    # hole interiors. This handles retraced slits without buffering or tolerance.
    segments = local[boundary]
    faces = polygonize(unary_union([LineString(segment) for segment in segments]))
    enclosed = []
    a, b = segments[:, 0], segments[:, 1]
    for face in faces:
        point = face.representative_point()
        cross = (b[:, 0] - a[:, 0]) * (point.y - a[:, 1]) - (point.x - a[:, 0]) * (
            b[:, 1] - a[:, 1]
        )
        winding = np.count_nonzero(
            (a[:, 1] <= point.y) & (b[:, 1] > point.y) & (cross > 0)
        )
        winding -= np.count_nonzero(
            (a[:, 1] > point.y) & (b[:, 1] <= point.y) & (cross < 0)
        )
        if winding:
            enclosed.append(face)
    domain = unary_union(enclosed)
    p, q, r = (local[tri[:, k]] for k in range(3))
    areas = (
        (q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1])
        - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0])
    ) / 2
    total = float(areas.sum())
    # Union is a second geometric check, independent of boundary traversal.
    union = unary_union([Polygon(local[row]) for row in tri])
    tolerance = 128 * np.finfo(float).eps * max(total, domain.area, 1.0)
    if abs(total - domain.area) > tolerance or abs(total - union.area) > tolerance:
        raise ValueError(
            "Overlapping elements: summed area differs from boundary/union area"
        )
    parent = np.arange(len(xy))

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in edges:
        parent[root(a)] = root(b)
    components = len({root(i) for i in range(len(xy))})
    return {
        "topology": "PASS",
        "components": components,
        "boundary_loops": loops,
        "edges": len(edges),
        "boundary_edges": len(boundary),
        "element_area_m2": total,
        "boundary_area_m2": float(domain.area),
        "area_roundoff_tolerance_m2": tolerance,
    }


def crosscheck_reports(report, finish, checks, topology):
    """Reports cross-check independent facts; they never certify the geometry.

    Stitch counts precede slit-wall duplication and are not final output counts.
    Only achieved and on-disk verification describe the serialized final mesh.
    """
    expected = {
        "n_frozen_nodes": checks["frozen_nodes"],
        "n_frozen_moved": 0,
        "n_inverted_elements": 0,
        "n_orphan_nodes": 0,
        "n_nonmanifold_edges": 0,
        "n_components": topology["components"],
        "n_retained_faces_missing": 0,
        "n_extra_faces": 0,
        "n_unexpected_boundary_edges": 0,
        "n_missing_boundary_edges": 0,
        "n_interface_segments_split": 0,
    }
    for section in ("verify", "verify_on_disk"):
        obj = report.get(section, {})
        if not isinstance(obj, dict):
            raise ValueError(f"Refinement report.{section}: expected an object")
        for key, value in expected.items():
            if key in obj:
                if type(obj[key]) is not int:
                    raise ValueError(f"Refinement report.{section}.{key}: expected int")
                if obj[key] != value:
                    raise ValueError(
                        f"Refinement report count disagreement: {section}.{key}"
                    )
    achieved = report.get("achieved", {})
    if not isinstance(achieved, dict):
        raise ValueError("Refinement report.achieved: expected an object")
    for key, value in (
        ("n_nodes", checks["nodes"]),
        ("n_elements", checks["elements"]),
    ):
        if key in achieved and (
            type(achieved[key]) is not int or achieved[key] != value
        ):
            raise ValueError(f"Refinement report count disagreement: achieved.{key}")
    if finish["n_movable"] != checks["new_nodes"]:
        raise ValueError(
            "Finishing report n_movable differs from the validated node map"
        )


def crosscheck_depth_counts(base, raw, refined, node_map, finish):
    """Cross-check finishing counts that describe the actual input/output tables."""
    movable = np.ones(len(refined.xy), dtype=bool)
    movable[node_map[node_map >= 0]] = False
    edges = mesh_edges(refined.triangles)
    over = r_factors(refined.depths, edges) > 0.2 + finish["write_precision_tolerance"]
    pair_frozen = (~movable[edges]).all(axis=1)
    inherited = {
        tuple(sorted(node_map[edge]))
        for edge in mesh_edges(base.triangles)
        if (node_map[edge] >= 0).all()
    }
    inherited_mask = np.array([tuple(edge) in inherited for edge in edges])
    expected = {
        "n_floored": int((raw.depths[movable] < 3).sum()),
        "n_capped": int((raw.depths[movable] > 300).sum()),
        "n_below_zero_before": int((raw.depths[movable] < 0).sum()),
        "n_frozen_below_hmin": int((raw.depths[~movable] < 3).sum()),
        "n_floored_after_smoothing": int((refined.depths[movable] == 3).sum()),
        "n_capped_after_smoothing": int((refined.depths[movable] == 300).sum()),
        "n_over_movable_at_tolerance": int((over & ~pair_frozen).sum()),
        "n_over_frozen_pair_at_tolerance": int((over & pair_frozen).sum()),
        "n_over_frozen_pair_inherited": int(
            (over & pair_frozen & inherited_mask).sum()
        ),
        "n_over_frozen_pair_new": int((over & pair_frozen & ~inherited_mask).sum()),
    }
    for key, value in expected.items():
        if key in finish and (type(finish[key]) is not int or finish[key] != value):
            raise ValueError(f"Finishing report count disagreement: {key}")

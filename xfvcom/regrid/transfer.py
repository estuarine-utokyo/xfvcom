"""Bounded base-to-refined transfer, without output, processes or on-disk caches.

Maps are zero-based BASE -> TARGET, with -1 for removed/unmatched entries.
Fields are finite float32/float64, spatial axis first: (N,) for horizontal
scalars, (N, K) for horizontal components, (N, L[, K]) for layered fields.
Sigma is a common, strictly decreasing interface vector from 0 to -1. Frozen
nodes/matched elements copy even in 3-D (including signed zeros and dtype).
Geometry uses metres; ties use the lowest zero-based node/triangle ID.
A frozen anchor uses the lowest triangle incident to its mapped base vertex.

The input meshes are already topology-validated by the caller. This module checks
array contracts and map consistency; it does not requalify a mesh bundle.
"""

import heapq
import math
import warnings
from dataclasses import dataclass
from fractions import Fraction
from numbers import Real

import numpy as np
from scipy.sparse import csr_matrix

from . import mesh_pair
from .transfer_reference import independent_evaluate


def _mesh_module():
    return mesh_pair


def _real(array, shape, label, field=False):
    a = np.asarray(array)
    kinds = "f" if field else "fiu"
    if a.dtype.kind not in kinds or (field and a.dtype.itemsize not in (4, 8)):
        raise ValueError(
            f"{label}: expected {'float32/float64' if field else 'real numeric'} dtype"
        )
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"{label}: expected finite shape {shape}")
    return a


# The physical range of the inputs (refused outside it, never computed): a
# coastal-ocean mesh in projected metres. These bounds keep every derived
# quantity (areas, lengths, volumes, means, layer centres) far from overflow
# and underflow, so no scaled arithmetic is needed anywhere.
COORDINATE_LIMIT_M = 1e8  # |x|, |y| of projected coordinates
MIN_EDGE_M = 1e-6  # shortest triangle edge
MIN_AREA_M2 = 1e-12  # smallest triangle area (an edge bound alone does not bound it)
MIN_SIGMA_THICKNESS = 1e-6  # thinnest sigma layer (fraction of the column)
DEPTH_RANGE_M = (1e-3, 1e5)  # positive-down depth
ELEVATION_LIMIT_M = 1e3  # |zeta|


def _physical_mesh(label, mesh):
    xy = np.asarray(mesh.xy, dtype=np.float64)
    if np.abs(xy).max(initial=0.0) > COORDINATE_LIMIT_M:
        raise ValueError(
            f"{label}: coordinates outside the physical range |x| <= {COORDINATE_LIMIT_M:g} m"
        )
    tri = np.asarray(mesh.triangles)
    for k in range(3):
        a, b = xy[tri[:, k]], xy[tri[:, (k + 1) % 3]]
        if (
            np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1]).min(initial=np.inf)
            < MIN_EDGE_M
        ):
            raise ValueError(
                f"{label}: an edge shorter than {MIN_EDGE_M:g} m is outside the physical range"
            )
    p = xy[tri]
    twice = np.abs(
        (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
        - (p[:, 1, 1] - p[:, 0, 1]) * (p[:, 2, 0] - p[:, 0, 0])
    )
    if (twice / 2).min(initial=np.inf) < MIN_AREA_M2:
        raise ValueError(
            f"{label}: a triangle smaller than {MIN_AREA_M2:g} m2 is outside the physical range"
        )
    # compared in the input dtype, before any narrowing cast can overflow
    h = np.asarray(mesh.depths)
    if bool(((h < DEPTH_RANGE_M[0]) | (h > DEPTH_RANGE_M[1])).any()):
        raise ValueError(
            f"{label}: depths outside the physical range {DEPTH_RANGE_M} m"
        )


def _validate_pair(base, target, nm, em=None):
    rm = _mesh_module()
    for label, mesh in (("base", base), ("target", target)):
        xy = np.asarray(mesh.xy)
        if xy.ndim != 2:
            raise ValueError(f"{label}.xy: expected (N, 2)")
        _real(xy, (len(xy), 2), f"{label}.xy")
        # Step 1 delivers float64 coordinates. Any other dtype is refused: a
        # narrower one rounds centroids and lengths, a wider or integer one
        # would lose digits in the float64 working geometry unnoticed.
        if xy.dtype != np.float64:
            raise ValueError(
                f"{label}.xy: expected float64 coordinates, got {xy.dtype}"
            )
        # the coordinate range first, before any geometry arithmetic can overflow
        if float(np.abs(xy).max(initial=0.0)) > COORDINATE_LIMIT_M:
            raise ValueError(
                f"{label}: coordinates outside the physical range |x| <= {COORDINATE_LIMIT_M:g} m"
            )
        rm._validate_geometry(xy, np.asarray(mesh.triangles))
        h = _real(mesh.depths, (len(xy),), f"{label}.depths")
        if (h <= 0).any():
            raise ValueError(f"{label}.depths must be positive")
        _physical_mesh(label, mesh)
    nm = rm.validate_node_map(nm, base, target)
    expected = rm.build_element_map(base, target, nm)
    if em is not None:
        em = np.asarray(em)
        if em.dtype.kind not in "iu" or em.shape != expected.shape:
            raise ValueError("elem_map: expected integer base-element vector")
        if not np.array_equal(em, expected):
            raise ValueError("elem_map inconsistent with grids and node_map")
    return expected


def _finite(label, *values):
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError(f"{label}: non-finite derived value or statistic")


def _cross(u, v):
    """Compensated local determinant, retaining the rounded product's residue."""
    product = float(u[1]) * float(v[0])
    error = math.fma(-float(u[1]), float(v[0]), product)
    return math.fma(float(u[0]), float(v[1]), -product) + error


def _exact_area(a, b, c):
    """Twice-free exact area of one triangle from its input floats."""
    (ax, ay), (bx, by), (cx, cy) = ([Fraction(float(x)) for x in p] for p in (a, b, c))
    return ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2


def _areas(mesh):
    """Triangle areas, each certified to 1e-12 relative against exact arithmetic.

    The coordinate differences are rounded before the compensated determinant,
    so near-degenerate triangles far from the origin can lose every digit; such
    geometry is refused rather than returned with a wrong area.
    """
    p = np.asarray(mesh.xy, dtype=float)[mesh.triangles]
    d, e = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    area = np.array([_cross(u, v) / 2 for u, v in zip(d, e)])
    _finite("triangle areas", area)
    if (area <= 0).any():
        raise ValueError("Triangle areas must be positive")
    tolerance = Fraction(1, 10**12)
    for k, (tri, value) in enumerate(zip(p, area)):
        exact = _exact_area(*tri)
        if exact <= 0 or abs(Fraction(float(value)) - exact) > tolerance * exact:
            raise ValueError(
                f"Triangle {k}: area cannot be established to 1e-12 relative"
            )
    return area


def _certify_weights(vertices, point, weights):
    """Certify floating weights against exact input-float Cramer determinants.

    This is a guard, not a fallback: geometry with >1e-12 absolute weight error
    is refused. The evaluator independently uses rational elimination instead.
    """
    a, b, c, q = [[Fraction(float(x)) for x in p] for p in (*vertices, point)]
    u, v, d = ([p[k] - a[k] for k in (0, 1)] for p in (b, c, q))
    determinant = u[0] * v[1] - u[1] * v[0]
    if determinant <= 0:
        raise ValueError(
            "Triangle weights cannot be established to 1e-12: degenerate triangle"
        )
    w1 = (d[0] * v[1] - d[1] * v[0]) / determinant
    w2 = (u[0] * d[1] - u[1] * d[0]) / determinant
    exact = (1 - w1 - w2, w1, w2)
    if not np.isfinite(weights).all() or any(
        abs(Fraction(float(w)) - r) > Fraction(1, 10**12)
        for w, r in zip(weights, exact)
    ):
        raise ValueError(
            "Triangle weights cannot be established to 1e-12 against exact values"
        )


def _locate(mesh, points):
    """Search actual mesh triangles (never a Delaunay fill across land)."""
    p = np.asarray(mesh.xy, dtype=float)[mesh.triangles]
    lo, hi = p.min(axis=1), p.max(axis=1)
    u, v = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    determinant = np.array([_cross(a, b) for a, b in zip(u, v)])
    ids = np.full(len(points), -1, dtype=int)
    weights = np.zeros((len(points), 3))
    for i, q in enumerate(points):
        candidates = np.flatnonzero(((q >= lo) & (q <= hi)).all(axis=1))
        for e in candidates:
            d = q - p[e, 0]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                b = _cross(d, v[e]) / determinant[e]
                c = _cross(u[e], d) / determinant[e]
                w = np.array([1 - b - c, b, c])
            _certify_weights(p[e], q, w)
            if (w >= -2e-12).all():
                clipped = np.maximum(w, 0)
                clipped /= clipped.sum()
                try:
                    _certify_weights(p[e], q, clipped)
                except ValueError:
                    # a neighbour within the boundary tolerance whose weights,
                    # once clipped, are not those of the point: the point lies
                    # in another candidate; keep searching
                    continue
                ids[i] = e
                weights[i] = clipped
                break
    return ids, weights


def _paths(mesh, inside):
    """Multi-source Dijkstra; label is (distance, donor ID)."""
    adjacency = [[] for _ in mesh.xy]
    for a, b in _mesh_module().mesh_edges(mesh.triangles):
        length = math.hypot(
            float(mesh.xy[a, 0]) - float(mesh.xy[b, 0]),
            float(mesh.xy[a, 1]) - float(mesh.xy[b, 1]),
        )
        adjacency[a].append((b, length))
        adjacency[b].append((a, length))
    distance = np.full(len(mesh.xy), np.inf)
    donor = np.full(len(mesh.xy), -1, dtype=int)
    queue = []
    for i in np.flatnonzero(inside):
        distance[i], donor[i] = 0.0, i
        heapq.heappush(queue, (0.0, int(i), int(i)))
    while queue:
        d, source, a = heapq.heappop(queue)
        if (d, source) != (distance[a], donor[a]):
            continue
        for b, length in adjacency[a]:
            candidate = (d + length, source)
            if candidate < (distance[b], donor[b]):
                distance[b], donor[b] = candidate
                heapq.heappush(queue, (*candidate, int(b)))
    return donor, distance


def _matrix(rows, nbase):
    indices, data, indptr = [], [], [0]
    for row in rows:
        assert all(np.isfinite(v) and v >= 0 for v in row.values()), "Invalid weights"
        for col, value in sorted(row.items()):
            if value > 0:
                indices.append(col)
                data.append(value)
        indptr.append(len(data))
    matrix = csr_matrix((data, indices, indptr), shape=(len(rows), nbase))
    assert (
        np.isfinite(matrix.data).all() and (matrix.data >= 0).all()
    ), "Invalid weights"
    assert np.allclose(
        np.asarray(matrix.sum(axis=1)).ravel(), 1.0, rtol=0, atol=2e-14
    ), "Row sum differs from one"
    return matrix


@dataclass(frozen=True)
class Transfer:
    """In-memory geometry, CSR operators, target->base copy indices and reports.

    Reports contain branch counts, outside IDs, target anchor/donor IDs and
    distances in metres (zero for inside/copy rows). Arrays must not be mutated.
    """

    base: object
    target: object
    node_map: np.ndarray
    elem_map: np.ndarray
    node_matrix: csr_matrix
    element_matrix: csr_matrix
    node_copy: np.ndarray
    element_copy: np.ndarray
    node_report: dict
    element_report: dict
    max_distance: float


def _distance_limit(max_distance):
    if (
        isinstance(max_distance, (bool, np.bool_))
        or not isinstance(max_distance, Real)
        or max_distance < 0
    ):
        raise ValueError(
            "max_distance must be a finite nonnegative real, non-boolean scalar"
        )
    try:
        limit = float(max_distance)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "max_distance must be representable as a finite real scalar"
        ) from exc
    if not np.isfinite(limit) or limit < 0:
        raise ValueError(
            "max_distance must be representable as a finite nonnegative real scalar"
        )
    return limit


def _horizontal_geometry(base, target, node_map, max_distance):
    nm = np.asarray(node_map)
    nc = np.full(len(target.xy), -1, dtype=int)
    nc[nm[nm >= 0]] = np.flatnonzero(nm >= 0)
    triangle, bary = _locate(base, target.xy)
    # A frozen node belongs to its mapped vertex, even where separate banks
    # have coincident coordinates. Its anchor triangle must use that vertex.
    first_incident = {}
    for e, vertices in enumerate(base.triangles):
        for k, v in enumerate(vertices):
            first_incident.setdefault(int(v), (e, k))
    for v in np.flatnonzero(nm >= 0):
        e, k = first_incident[int(v)]
        triangle[nm[v]], bary[nm[v]] = e, np.eye(3)[k]
    donor, distance = _paths(target, triangle >= 0)
    return nc, triangle, bary, donor, distance


def _nodes(base, target, geometry, max_distance):
    nc, triangle, bary, donor, distance = geometry
    outside = np.flatnonzero(triangle < 0)
    if (distance[outside] > max_distance).any():
        raise ValueError(
            "Outside-water node donor exceeds max_distance or is disconnected"
        )
    rows = []
    for i in range(len(target.xy)):
        if nc[i] >= 0:
            row = {int(nc[i]): 1.0}
        else:
            a = i if triangle[i] >= 0 else donor[i]
            row = (
                {int(nc[a]): 1.0}
                if nc[a] >= 0
                else dict(zip(base.triangles[triangle[a]], bary[a]))
            )
        rows.append(row)
    nodes = _matrix(rows, len(base.xy))
    node_area = np.zeros(len(target.xy))
    target_area = _areas(target)
    for k in range(3):
        np.add.at(node_area, target.triangles[:, k], target_area / 3)
    with np.errstate(over="ignore", invalid="ignore"):
        outside_volume = float(
            np.multiply(
                node_area[outside], np.asarray(target.depths, dtype=np.float64)[outside]
            ).sum()
        )
    report = {
        "frozen": int((nc >= 0).sum()),
        "inside": int(((nc < 0) & (triangle >= 0)).sum()),
        "outside": len(outside),
        "outside_nodes": outside,
        "donor_nodes": donor,
        "donor_distances_m": distance,
        "outside_area_m2": float(node_area[outside].sum()),
        "outside_volume_at_level_surface_m3": outside_volume,
        "max_donor_distance_m": float(distance.max(initial=0)),
    }
    with np.errstate(over="ignore", invalid="ignore"):
        _finite(
            "horizontal report",
            node_area,
            report["outside_area_m2"],
            report["outside_volume_at_level_surface_m3"],
            report["max_donor_distance_m"],
        )
    return nodes, report


def _elements(base, target, em, geometry, max_distance):
    _, triangle, bary, donor, distance = geometry
    ec = np.full(len(target.triangles), -1, dtype=int)
    ec[em[em >= 0]] = np.flatnonzero(em >= 0)
    # Area share at a base vertex = incident triangle area / all incident areas.
    area = _areas(base)
    incident = [[] for _ in base.xy]
    for e, vertices in enumerate(base.triangles):
        for v in vertices:
            incident[v].append(e)
    vertex_area = np.array([area[es].sum() for es in incident])
    centroids = target.xy[target.triangles].mean(axis=1)
    ct, cb = _locate(base, centroids)
    anchors = np.full(len(centroids), -1, dtype=int)
    ad = np.zeros(len(centroids))
    anchored = np.flatnonzero((ct < 0) & (ec < 0))
    for i in anchored:
        choices = [
            (
                math.hypot(
                    float(centroids[i, 0]) - float(target.xy[v, 0]),
                    float(centroids[i, 1]) - float(target.xy[v, 1]),
                )
                + distance[v],
                int(donor[v]),
            )
            for v in target.triangles[i]
        ]
        ad[i], anchors[i] = min(choices)
        if not np.isfinite(ad[i]) or ad[i] > max_distance:
            raise ValueError(
                "Outside centroid anchor exceeds max_distance or is disconnected"
            )
        ct[i], cb[i] = triangle[anchors[i]], bary[anchors[i]]
    rows = []
    for i in range(len(centroids)):
        if ec[i] >= 0:
            row = {int(ec[i]): 1.0}
        else:
            row = {}
            for v, w in zip(base.triangles[ct[i]], cb[i]):
                for e in incident[v]:
                    row[e] = row.get(e, 0.0) + w * area[e] / vertex_area[v]
        rows.append(row)
    elements = _matrix(rows, len(base.triangles))
    report = {
        "matched": int((ec >= 0).sum()),
        "stencil": int(((ec < 0) & (anchors < 0)).sum()),
        "anchor": len(anchored),
        "outside_elements": anchored,
        "anchor_nodes": anchors,
        "anchor_distances_m": ad,
        "max_anchor_distance_m": float(ad.max(initial=0)),
    }
    return elements, report, ec


def build_transfer(base, target, node_map, elem_map=None, *, max_distance=500.0):
    """Build both convex operators; refuse inconsistent arrays or remote donors.

    ``max_distance`` is a finite nonnegative edge-path limit in metres, applied
    to both outside node donors and centroid anchors. Matrices are target x base.
    Optional elem_map must exactly match the vertex-triple correspondence.
    """
    max_distance = _distance_limit(max_distance)
    em = _validate_pair(base, target, node_map, elem_map)
    nm = np.asarray(node_map).copy()
    geometry = _horizontal_geometry(base, target, nm, max_distance)
    nodes, nr = _nodes(base, target, geometry, max_distance)
    elements, er, ec = _elements(base, target, em, geometry, max_distance)
    return Transfer(
        base,
        target,
        nm,
        em,
        nodes,
        elements,
        geometry[0],
        ec,
        nr,
        er,
        float(max_distance),
    )


def build_node_operator(base, target, node_map, *, max_distance=500.0):
    """Return (node CSR, report), applying only the outside-node distance limit."""
    max_distance = _distance_limit(max_distance)
    _validate_pair(base, target, node_map)
    geometry = _horizontal_geometry(base, target, node_map, max_distance)
    return _nodes(base, target, geometry, max_distance)


def build_element_operator(
    base, target, node_map, elem_map=None, *, max_distance=500.0
):
    """Return (element CSR, report), applying only the centroid-anchor limit."""
    max_distance = _distance_limit(max_distance)
    em = _validate_pair(base, target, node_map, elem_map)
    geometry = _horizontal_geometry(base, target, node_map, max_distance)
    matrix, report, _ = _elements(base, target, em, geometry, max_distance)
    return matrix, report


def _operator(op, location):
    if location == "node":
        return op.node_matrix, op.node_copy
    if location == "element":
        return op.element_matrix, op.element_copy
    raise ValueError("location must be 'node' or 'element'")


FIELD_MAGNITUDE_LIMIT = 1e100


def _field(field, n, layered=False):
    """A model field: float32/float64, finite, |value| <= 1e100.

    The bound is far beyond any state FVCOM carries (T, S, zeta, velocity,
    tracer); it keeps the convex combinations, whose weights sum to one only
    to round-off, from overflowing.
    """
    a = np.asarray(field)
    if a.ndim not in ((2, 3) if layered else (1, 2)) or a.shape[0] != n:
        raise ValueError("field: wrong shape (spatial axis first)")
    a = _real(a, a.shape, "field", field=True)
    if a.size and float(np.abs(a).max()) > FIELD_MAGNITUDE_LIMIT:
        raise ValueError(
            f"field: magnitude above {FIELD_MAGNITUDE_LIMIT:g} is not a model state"
        )
    return a


def apply_2d(op, field, *, location="node"):
    """Apply horizontal weights, preserving input dtype and copy-row bytes."""
    matrix, copies = _operator(op, location)
    field = _field(field, matrix.shape[1])
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(matrix @ field, dtype=field.dtype)
    keep = copies >= 0
    result[keep] = field[copies[keep]]
    _finite("horizontal transfer", result)
    return result


def _columns(op, location, sigma, base_zeta, target_zeta):
    sigma = np.asarray(sigma)
    _real(sigma, sigma.shape, "sigma")
    # interfaces lie in [-1, 0]: checked in the input dtype, before the cast
    # and the differences, which could otherwise overflow
    if sigma.size and bool(((sigma < -1) | (sigma > 0)).any()):
        raise ValueError("sigma: interfaces must lie in [-1, 0]")
    sigma = sigma.astype(np.float64)
    if (
        sigma.ndim != 1
        or len(sigma) < 3
        or sigma[0] != 0
        or sigma[-1] != -1
        or not (np.diff(sigma) < 0).all()
    ):
        raise ValueError(
            "sigma: need decreasing interfaces 0 to -1, at least two layers"
        )
    if (-np.diff(np.asarray(sigma, dtype=np.float64))).min() < MIN_SIGMA_THICKNESS:
        raise ValueError(
            f"sigma: a layer thinner than {MIN_SIGMA_THICKNESS:g} of the column is outside the physical range"
        )
    # float64 working dtype: integer or unsigned depths would wrap on negation
    # and overflow on addition.
    bh = np.asarray(op.base.depths, dtype=np.float64)
    th = np.asarray(op.target.depths, dtype=np.float64)
    raw_bz = _real(base_zeta, bh.shape, "base_zeta")
    raw_tz = _real(target_zeta, th.shape, "target_zeta")
    # the range is tested in the input dtype, before a narrowing cast can overflow
    for label, z in (("base_zeta", raw_bz), ("target_zeta", raw_tz)):
        # direct comparisons: abs() of the most negative integer is itself
        if z.size and bool(((z < -ELEVATION_LIMIT_M) | (z > ELEVATION_LIMIT_M)).any()):
            raise ValueError(
                f"{label}: outside the physical range |zeta| <= {ELEVATION_LIMIT_M:g} m"
            )
    bz, tz = raw_bz.astype(np.float64), raw_tz.astype(np.float64)
    if location == "element":
        bh, bz = bh[op.base.triangles].mean(axis=1), bz[op.base.triangles].mean(axis=1)
        th, tz = th[op.target.triangles].mean(axis=1), tz[op.target.triangles].mean(
            axis=1
        )
        area = _areas(op.target)
    else:
        area, third = np.zeros(len(th)), _areas(op.target) / 3
        for k in range(3):
            np.add.at(area, op.target.triangles[:, k], third)
    with np.errstate(over="ignore", invalid="ignore"):
        base_depth, target_depth = bh + bz, th + tz
    _finite("column depths", base_depth, target_depth)
    if (base_depth <= 0).any() or (target_depth <= 0).any():
        raise ValueError("Depth plus elevation must be positive in every column")
    centres = (sigma[:-1] + sigma[1:]) / 2
    with np.errstate(over="ignore", invalid="ignore"):
        base_centres = bz[:, None] + centres * (bh + bz)[:, None]
        target_centres = tz[:, None] + centres * (th + tz)[:, None]
        volumes = area[:, None] * (th + tz)[:, None] * -np.diff(sigma)
    _finite("column geometry", base_centres, target_centres, volumes)
    # Distinct sigma layers must stay distinct in physical z, or interpolation
    # would silently merge them (in the evaluator too).
    for label, c in (("base", base_centres), ("target", target_centres)):
        if not (np.diff(c, axis=1) < 0).all():
            raise ValueError(
                f"{label} layer centres are not strictly decreasing in physical z"
            )
    return base_centres, target_centres, bh, bz, volumes


def _interpolate(z, values, targets):
    """Return constant-end interpolation and its unbounded linear alternative."""
    order = np.searchsorted(-z, -targets, side="right") - 1
    j = np.clip(order, 0, len(z) - 2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        t = (targets - z[j]) / (z[j + 1] - z[j])
    _finite("interpolation fractions", t)
    shape = (len(t),) + (1,) * (values.ndim - 1)
    a, b = values[j], values[j + 1]
    fraction = np.broadcast_to(np.clip(t, 0, 1).reshape(shape), a.shape)
    same_sign = np.signbit(a) == np.signbit(b)
    clipped = np.empty_like(a)
    # Same-sign differences cannot overflow. Opposite-sign weighted sums cannot
    # overflow. Clip rounding at the last ulp to the closed donor interval.
    clipped[same_sign] = a[same_sign] + fraction[same_sign] * (
        b[same_sign] - a[same_sign]
    )
    clipped[~same_sign] = (1 - fraction[~same_sign]) * a[~same_sign] + fraction[
        ~same_sign
    ] * b[~same_sign]
    clipped = np.clip(clipped, np.minimum(a, b), np.maximum(a, b))
    above, below = targets >= z[0], targets <= z[-1]
    clipped[above] = values[0]
    clipped[below] = values[-1]
    linear = clipped.copy()
    outside = (targets > z[0]) | (targets < z[-1])
    # Extrapolation is only a statistic. Wider intermediates avoid a spurious
    # overflow of b-a; an unrepresentable float64 statistic is still refused.
    with np.errstate(over="ignore", invalid="ignore"):
        aa, bb = a[outside].astype(np.longdouble), b[outside].astype(np.longdouble)
        linear[outside] = aa + t[outside].astype(np.longdouble).reshape(
            (outside.sum(),) + shape[1:]
        ) * (bb - aa)
    _finite("vertical interpolation/extrapolation", clipped, linear)
    return clipped, linear


def _statistic(events, denominator, delta):
    distance, weight = (
        np.concatenate([e[k] for e in events]) if events else np.array([])
        for k in (0, 1)
    )
    _finite("end statistic inputs", distance, weight, denominator, delta)
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(weight.sum())
    _finite("end statistic volume", total)
    quantiles = np.zeros(3)
    if total:
        order = np.argsort(distance)
        with np.errstate(over="ignore", invalid="ignore"):
            cumulative = np.cumsum(weight[order])
        _finite("end statistic cumulative volume", cumulative)
        quantiles = distance[order][
            np.minimum(
                np.searchsorted(cumulative, np.array([0.5, 0.95, 0.99]) * total),
                len(order) - 1,
            )
        ]
    report = {
        "weighted_volume_m3": total,
        "fraction": total / denominator if denominator else 0.0,
        "donor_layer_events": len(distance),
        "distances_m": distance,
        "volume_weights_m3": weight,
        "distance_quantiles_m": quantiles,
        "max_distance_m": float(distance.max(initial=0)),
        "linear_extrapolation_delta": delta,
        "max_abs_delta": float(np.abs(delta).max(initial=0)),
    }
    _finite("end statistics", *report.values())
    return report


def _warn_extension(statistics, location):
    """Draw attention to a large difference from the base depths (no limit: the
    deepest base layer continued downward is the only sensible fill)."""
    ext = statistics["extension"]
    if ext["max_distance_m"] > EXTENSION_WARNING_M:
        q50, q95, q99 = (float(v) for v in ext["distance_quantiles_m"])
        warnings.warn(
            f"{location} transfer: {100 * ext['fraction']:.2f} % of the new volume is filled by continuing "
            f"the base's deepest (or shallowest) layer beyond its water column; distance beyond it "
            f"p50/p95/p99/max = {q50:.2f}/{q95:.2f}/{q99:.2f}/{ext['max_distance_m']:.2f} m. "
            f"The refined depths differ from the base depths there; the spin-up replaces this initial "
            f"state.",
            ExtensionWarning,
            stacklevel=3,
        )


class ExtensionWarning(UserWarning):
    """The refined columns reach below (or above) the base columns they are filled from."""


# Above this distance beyond the donor's water column, apply_3d warns.
EXTENSION_WARNING_M = 1.0


def apply_3d(op, field, sigma, base_zeta, target_zeta, *, location="node"):
    """Return (values, end statistics), remapping EACH donor before combining.

    Statistics cover new nodes (area/3 control volumes) or unmatched elements
    (triangle volumes). Each event weighs horizontal donor weight times target
    layer volume. Extension is beyond physical surface/bottom; clamping is only
    beyond layer centres while still in the water column. Each category reports
    the aggregate target-field change if linear extrapolation replaces constant
    extension for that category alone; distances are to the water boundary for
    extension, to the nearest layer centre for clamping. Copy rows are excluded.
    """
    matrix, copies = _operator(op, location)
    field = _field(field, matrix.shape[1], layered=True)
    zb, zt, bh, bz, volume = _columns(op, location, sigma, base_zeta, target_zeta)
    if field.shape[1] != zb.shape[1]:
        raise ValueError("field layer count differs from sigma interfaces")
    shape = (matrix.shape[0],) + field.shape[1:]
    result = np.zeros(shape, dtype=np.float64)
    deltas = {key: np.zeros(shape) for key in ("clamping", "extension")}
    events = {key: [] for key in deltas}
    for i in range(matrix.shape[0]):
        if copies[i] >= 0:
            result[i] = field[copies[i]]
            continue
        for k in range(matrix.indptr[i], matrix.indptr[i + 1]):
            j, w = matrix.indices[k], matrix.data[k]
            values, linear = _interpolate(zb[j], field[j].astype(float), zt[i])
            with np.errstate(over="ignore", invalid="ignore"):
                result[i] += w * values
            extended = (zt[i] > bz[j]) | (zt[i] < -bh[j])
            beyond_centres = (zt[i] > zb[j, 0]) | (zt[i] < zb[j, -1])
            distances = {
                "extension": np.maximum(zt[i] - bz[j], -bh[j] - zt[i]),
                "clamping": np.maximum(zt[i] - zb[j, 0], zb[j, -1] - zt[i]),
            }
            for key, mask in (
                ("extension", extended),
                ("clamping", beyond_centres & ~extended),
            ):
                if mask.any():
                    with np.errstate(over="ignore", invalid="ignore"):
                        events[key].append((distances[key][mask], w * volume[i, mask]))
                        deltas[key][i, mask] += w * (linear[mask] - values[mask])
    with np.errstate(over="ignore", invalid="ignore"):
        result = result.astype(field.dtype)
    keep = copies >= 0
    result[keep] = field[copies[keep]]
    _finite("layered transfer", result)
    with np.errstate(over="ignore", invalid="ignore"):
        denominator = float(volume[~keep].sum())
    statistics = {
        key: _statistic(events[key], denominator, deltas[key]) for key in deltas
    }
    statistics["target_volume_m3"] = denominator
    _warn_extension(statistics, location)
    return result, statistics


def check(
    op,
    field,
    *,
    location="node",
    sigma=None,
    base_zeta=None,
    target_zeta=None,
    actual=None,
    rtol=None,
    atol=None,
):
    """Compare every row/layer with the independent evaluator; refuse mismatch.

    With ``actual`` omitted, evaluate the production operator as well. Passing
    an existing result permits checks against perturbed fields/surfaces. Returns
    max_abs_error and number of scalar values checked, or raises ValueError.
    """
    matrix, _ = _operator(op, location)
    source = _field(field, matrix.shape[1], layered=sigma is not None)
    if sigma is not None:
        _columns(op, location, sigma, base_zeta, target_zeta)
        if source.shape[1] != len(sigma) - 1:
            raise ValueError("field layer count differs from sigma interfaces")
    if actual is None:
        actual = (
            apply_2d(op, source, location=location)
            if sigma is None
            else apply_3d(op, source, sigma, base_zeta, target_zeta, location=location)[
                0
            ]
        )
    actual = _real(
        actual, (matrix.shape[0],) + source.shape[1:], "check actual", field=True
    )
    expected, scale = independent_evaluate(
        op.base,
        op.target,
        op.node_map,
        op.elem_map,
        source,
        location=location,
        sigma=sigma,
        base_zeta=base_zeta,
        target_zeta=target_zeta,
        max_distance=op.max_distance,
        return_scale=True,
    )
    # Allowed difference per value: forward round-off of the combination in the
    # field's dtype plus the 1e-12 to which the geometry (weights) is certified,
    # both times the magnitude of the values that enter it, plus the smallest
    # subnormal of the dtype (a correctly rounded result may underflow to zero).
    # Computed in Python floats, independently of both implementations.
    for label, value in (("rtol", rtol), ("atol", atol)):
        if value is None:
            continue
        bad = isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        )
        if not bad:
            try:
                bad = bool(value < 0) or not math.isfinite(float(value))
            except (OverflowError, ValueError):
                bad = True
        if bad:
            raise ValueError(
                f"check: {label} must be a finite non-negative real scalar"
            )
    info = np.finfo(source.dtype)
    relative = 64.0 * float(info.eps) + 3.0e-12
    floor = 4.0 * float(info.smallest_subnormal)
    allowance = (
        relative * scale + floor if atol is None else np.full(scale.shape, float(atol))
    )
    if rtol is not None:
        allowance = allowance + float(rtol) * np.abs(expected)
    difference = np.abs(
        np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    )
    if (difference > allowance).any():
        raise ValueError(
            f"check: independent evaluator differs by {difference.max():.9g}"
        )
    return {
        "max_abs_error": float(difference.max(initial=0)),
        "values_checked": expected.size,
    }

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
FRACTION_TOLERANCE = 1e-12  # certified error of a vertical interpolation fraction
_UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2


def _outside(values, low, high):
    """True if any value lies outside [low, high], compared without narrowing.

    The bounds are typed with ``np.result_type(values.dtype, float64)``: a
    narrow input (float16) is promoted instead of the bound being narrowed
    (which overflows), and a wider one (longdouble) keeps its own range, so no
    cast happens before the range is known.
    """
    values = np.asarray(values)
    if not values.size:
        return False
    bounds = np.array([low, high], dtype=np.result_type(values.dtype, np.float64))
    return bool(((values < bounds[0]) | (values > bounds[1])).any())


def _exact_in_float64(values, label):
    """Refuse values that float64, the working type, cannot hold exactly.

    A wider float (longdouble) or a large integer would be narrowed silently,
    and the certified bounds (and the evaluator) would then describe the
    narrowed problem, not the given one (round 12). Called after the range
    check, so the cast cannot overflow.
    """
    values = np.asarray(values)
    if values.dtype.kind in "iu" or values.dtype.itemsize > 8:
        narrowed = values.astype(np.float64)
        if not (narrowed.astype(values.dtype) == values).all():
            raise ValueError(f"{label}: values not exactly representable in float64")


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
    h = np.asarray(mesh.depths)
    if _outside(h, *DEPTH_RANGE_M):
        raise ValueError(
            f"{label}: depths outside the physical range {DEPTH_RANGE_M} m"
        )
    _exact_in_float64(h, f"{label}.depths")


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


def _rational(x):
    return x if isinstance(x, Fraction) else Fraction(float(x))


def _exact_weights(vertices, point):
    """Exact barycentric weights (Cramer determinants) of exact coordinates.

    Floats are taken as the exact values they hold; a point may also be given
    as Fractions (an exact centroid).
    """
    a, b, c, q = [[_rational(x) for x in p] for p in (*vertices, point)]
    u, v, d = ([p[k] - a[k] for k in (0, 1)] for p in (b, c, q))
    determinant = u[0] * v[1] - u[1] * v[0]
    if determinant <= 0:
        raise ValueError("Triangle weights cannot be established: degenerate triangle")
    w1 = (d[0] * v[1] - d[1] * v[0]) / determinant
    w2 = (u[0] * d[1] - u[1] * d[0]) / determinant
    return (1 - w1 - w2, w1, w2)


def _exact_centroids(mesh):
    """Exact centroids of the triangles (sums of the vertex floats over 3)."""
    xy = np.asarray(mesh.xy, dtype=np.float64)
    return [
        tuple(sum(Fraction(float(xy[v, k])) for v in tri) / 3 for k in (0, 1))
        for tri in mesh.triangles
    ]


def _locate(mesh, points, exact_points=None):
    """Search actual mesh triangles (never a Delaunay fill across land).

    Containment and weights are EXACT (rational) for the exact query point --
    the given float, or ``exact_points`` (e.g. exact centroids, which a float
    cannot hold) -- and are rounded once at the end (round 12). The bounding
    boxes only preselect candidates, widened so that rounding a centroid to
    ``points`` cannot drop its container.
    """
    p = np.asarray(mesh.xy, dtype=float)[mesh.triangles]
    lo, hi = p.min(axis=1), p.max(axis=1)
    # A point accepted by the clipping rule lies within 1e-12 (in weights) of
    # its triangle, i.e. within 1e-12 of the triangle's extent outside it:
    # boxes padded by 1e-11 of the extent are a superset of every accepted
    # point (round 13); the query's own rounding is added per point below.
    slack = 1e-11 * (hi - lo).max(axis=1)[:, None]
    ids = np.full(len(points), -1, dtype=int)
    weights = np.zeros((len(points), 3))
    tolerance = Fraction(1, 10**12)
    for i, q in enumerate(points):
        exact_q = exact_points[i] if exact_points is not None else q
        # a rounded query (a centroid) is within one spacing of the exact one
        pad = slack + np.spacing(np.abs(q))
        candidates = np.flatnonzero(((q >= lo - pad) & (q <= hi + pad)).all(axis=1))
        for e in candidates:
            # A point is in a candidate when clipping its exact weights onto
            # the triangle moves them by at most 1e-12 (a boundary point is
            # clipped onto it); a candidate whose box merely covers the point
            # moves them far more and is passed over.
            exact = _exact_weights(p[e], exact_q)
            clipped = [max(w, Fraction(0)) for w in exact]
            total = sum(clipped)
            clipped = [w / total for w in clipped]
            if max(abs(c - w) for c, w in zip(clipped, exact)) > tolerance:
                continue
            ids[i] = e
            weights[i] = [float(w) for w in clipped]
            break
    return ids, weights


def _tied(a, b, offset_error):
    """Whether two computed path lengths (d, node, hops) cannot be ordered.

    Each edge length (a rounded coordinate difference and hypot) and each
    addition along a path rounds by at most about 2u of the path length, so a
    path of h edges is within 4u (h + 1) d of its exact length; ``offset_error``
    adds the rounding of a query point that is itself rounded (a centroid).
    """
    u = _UNIT_ROUNDOFF
    bound = 4 * u * (a[2] + b[2] + 2) * max(a[0], b[0]) + 2 * offset_error
    return abs(a[0] - b[0]) <= bound


def _choose(found, label, offset_error=0.0):
    """The donor among (distance, node, hops) candidates, or refuse an ambiguous one.

    Candidates whose lengths differ by less than their rounding bound are
    ties. Bit-equal ties go to the lowest ID; other ties cannot be ordered by
    the float arithmetic and are refused (round 12), never decided by rounding.
    """
    # an unreachable vertex (no inside node within the limit) is no candidate
    found = [c for c in found if c[1] >= 0 and math.isfinite(c[0])]
    if not found:
        return -1, math.inf, 0
    best = min(found)
    close = [c for c in found if _tied(c, best, offset_error)]
    if any(d != best[0] for d, v, _ in close if v != best[1]):
        raise ValueError(
            f"{label}: donors {sorted({int(v) for _, v, _ in close})} have water-path "
            "lengths closer than their rounding bound; the choice is numerically "
            "ambiguous and is refused"
        )
    return min(v for d, v, _ in close if d == best[0]), best[0], best[2]


def _beyond_ties(d, first, min_edge, margin):
    """Whether every path of length >= d is outside the tie band of ``first``.

    A path of length d has at most d / min_edge edges, so its rounding bound
    is at most 4u (h0 + d / min_edge + 2) d; ``margin`` adds the rounding of a
    rounded query point (an element centroid) for the anchors built on these
    candidates (round 13: a fixed relative cutoff could drop a tied path).
    """
    d0, _, h0 = first
    bound = 4 * _UNIT_ROUNDOFF * (h0 + d / min_edge + 2) * d + 2 * margin
    return d - d0 > bound


def _nearest(graph, starts, max_distance, margin=0.0):
    """Reverse Dijkstra from ``starts`` [(distance, hops, node)]: the tie band.

    Before any inside node is found the search stops only beyond
    ``max_distance`` plus the rounding a path of that length can carry; once
    one is found it continues -- past the limit if need be (round 14) --
    until no longer path can enter the first one's tie band, also through
    inside nodes. Whether the chosen donor is within ``max_distance`` is the
    caller's decision, made after the choice is certified.
    """
    adjacency, inside, min_edge = graph
    queue = [(float(d), int(h), int(v)) for d, h, v in starts]
    heapq.heapify(queue)
    seen, found = set(), []
    while queue:
        d, hops, a = heapq.heappop(queue)
        if found:
            if _beyond_ties(d, found[0], min_edge, margin):
                break
        elif (
            d - max_distance > 4 * _UNIT_ROUNDOFF * (d / min_edge + 2) * d + 2 * margin
        ):
            break
        if a in seen:
            continue
        seen.add(a)
        if inside[a]:
            found.append((d, a, hops))
        for b, length in adjacency[a]:
            if b not in seen:
                heapq.heappush(queue, (d + length, hops + 1, b))
    return found


def _graph(mesh, inside):
    """Refined-edge adjacency with float lengths, the inside mask, the shortest edge."""
    adjacency = [[] for _ in mesh.xy]
    min_edge = math.inf
    for a, b in _mesh_module().mesh_edges(mesh.triangles):
        length = math.hypot(
            float(mesh.xy[a, 0]) - float(mesh.xy[b, 0]),
            float(mesh.xy[a, 1]) - float(mesh.xy[b, 1]),
        )
        min_edge = min(min_edge, length)
        adjacency[a].append((b, length))
        adjacency[b].append((a, length))
    return adjacency, np.asarray(inside, dtype=bool), min_edge


def _paths(mesh, inside, max_distance=math.inf):
    """Nearest inside node of every node along refined edges (ties: lowest ID).

    Also returns the search graph, from which the element anchors are searched
    directly from their centroids.
    """
    graph = _graph(mesh, inside)
    inside = graph[1]
    distance = np.where(inside, 0.0, np.inf)
    donor = np.where(inside, np.arange(len(mesh.xy)), -1)
    for i in np.flatnonzero(~inside):
        found = _nearest(graph, [(0.0, 0, i)], max_distance)
        donor[i], distance[i], _ = _choose(found, f"node {i}")
    return donor, distance, graph


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
    # Node donors are chosen only for the node operator (round 15): an
    # ambiguous donor of an outside node must not stop the element operator,
    # whose anchors are searched from the centroids.
    return nc, triangle, bary, _graph(target, triangle >= 0)


def _nodes(base, target, geometry, max_distance):
    nc, triangle, bary, _ = geometry
    donor, distance, graph = _paths(target, triangle >= 0, max_distance)
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
    _, triangle, bary, graph = geometry
    ec = np.full(len(target.triangles), -1, dtype=int)
    ec[em[em >= 0]] = np.flatnonzero(em >= 0)
    # Area share at a base vertex = incident triangle area / all incident areas.
    area = _areas(base)
    incident = [[] for _ in base.xy]
    for e, vertices in enumerate(base.triangles):
        for v in vertices:
            incident[v].append(e)
    vertex_area = np.array([area[es].sum() for es in incident])
    exact_centroids = _exact_centroids(target)
    # correctly rounded exact centroids (a float mean can be far off when its
    # sum cancels, round 13)
    centroids = np.array([[float(c) for c in q] for q in exact_centroids])
    ct, cb = _locate(base, centroids, exact_centroids)
    anchors = np.full(len(centroids), -1, dtype=int)
    ad = np.zeros(len(centroids))
    anchored = np.flatnonzero((ct < 0) & (ec < 0))
    for i in anchored:
        # searched directly from the centroid through its three vertices
        # (round 14: vertex-level pruning missed the centroid's wider band);
        # the centroid is rounded to float64: its position errs by about 2u
        # of its largest coordinate
        offset = 2 * _UNIT_ROUNDOFF * float(np.abs(centroids[i]).max())
        starts = [
            (
                math.hypot(
                    float(centroids[i, 0]) - float(target.xy[v, 0]),
                    float(centroids[i, 1]) - float(target.xy[v, 1]),
                ),
                1,
                int(v),
            )
            for v in target.triangles[i]
        ]
        choices = _nearest(graph, starts, max_distance, offset)
        anchors[i], ad[i], _ = _choose(choices, f"element {i}", offset)
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
    _exact_in_float64(sigma, "sigma")
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
        if _outside(z, -ELEVATION_LIMIT_M, ELEVATION_LIMIT_M):
            raise ValueError(
                f"{label}: outside the physical range |zeta| <= {ELEVATION_LIMIT_M:g} m"
            )
    _exact_in_float64(raw_bz, "base_zeta")
    _exact_in_float64(raw_tz, "target_zeta")
    bz, tz = raw_bz.astype(np.float64), raw_tz.astype(np.float64)
    nodal = (bh, bz, th, tz)
    u = _UNIT_ROUNDOFF
    # Every NODAL water column must be positive, also when the columns used
    # are element means: a mean can hide a dry or inverted node (round 13).
    with np.errstate(over="ignore", invalid="ignore"):
        nodal_depths = (bh + bz, th + tz)
    _finite("column depths", *nodal_depths)
    if any((d <= 0).any() for d in nodal_depths):
        raise ValueError("Depth plus elevation must be positive in every column")
    if location == "element":
        # Element columns are node means of the elevation and of the water
        # column, each an exactly rounded sum over three (no cancellation
        # between separately rounded means, round 13): relative error <= 2u.
        def mean(values, tri):
            return np.array([math.fsum(values[v] for v in t) / 3 for t in tri])

        def column(h, z, tri):
            return np.array(
                [math.fsum([*(h[v] for v in t), *(z[v] for v in t)]) / 3 for t in tri]
            )

        tb, tt = op.base.triangles, op.target.triangles
        bz_e, tz_e = mean(bz, tb), mean(tz, tt)
        base_depth, target_depth = column(bh, bz, tb), column(th, tz, tt)
        bz, tz = bz_e, tz_e
        ebz, etz = 2 * u * np.abs(bz), 2 * u * np.abs(tz)
        base_error, target_error = 2 * u * base_depth, 2 * u * target_depth
        area = _areas(op.target)
    else:
        ebz, etz = np.zeros(len(bz)), np.zeros(len(tz))
        base_depth, target_depth = nodal_depths
        base_error, target_error = 2 * u * base_depth, 2 * u * target_depth
        area, third = np.zeros(len(th)), _areas(op.target) / 3
        for k in range(3):
            np.add.at(area, op.target.triangles[:, k], third)
    if (base_depth <= 0).any() or (target_depth <= 0).any():
        raise ValueError("Depth plus elevation must be positive in every column")
    # Layer centres are kept RELATIVE TO THEIR OWN SURFACE (sigma times the
    # water column). Absolute z (surface plus that) is never formed: rounding
    # it first loses the differences that matter in a thin column (round 11).
    centres = (sigma[:-1] + sigma[1:]) / 2
    with np.errstate(over="ignore", invalid="ignore"):
        base_centres = centres * base_depth[:, None]
        target_centres = centres * target_depth[:, None]
        volumes = area[:, None] * target_depth[:, None] * -np.diff(sigma)
    _finite("column geometry", base_centres, target_centres, volumes)
    # Distinct sigma layers must stay distinct, or interpolation would
    # silently merge them (in the evaluator too).
    for label, c in (("base", base_centres), ("target", target_centres)):
        if not (np.diff(c, axis=1) < 0).all():
            raise ValueError(
                f"{label} layer centres are not strictly decreasing in physical z"
            )
    # absolute error bounds of the relative centres (2u per rounding: slack for
    # the second-order terms); the centres themselves are rounded once
    centre_error = 2 * u * np.abs(centres)
    base_centre_error = (
        np.abs(centres) * base_error[:, None]
        + base_depth[:, None] * centre_error
        + 2 * u * np.abs(base_centres)
    )
    target_centre_error = (
        np.abs(centres) * target_error[:, None]
        + target_depth[:, None] * centre_error
        + 2 * u * np.abs(target_centres)
    )
    return _Columns(
        base_centres,
        base_centre_error,
        target_centres,
        target_centre_error,
        bz,
        ebz,
        tz,
        etz,
        base_depth,
        base_error,
        volumes,
        nodal,
        (op.base.triangles, op.target.triangles) if location == "element" else None,
        tuple(float(x) for x in sigma),
        {},
    )


@dataclass(frozen=True)
class _Columns:
    """Columns of one location: centres relative to their own surface."""

    base_centres: np.ndarray  # (base columns, layers), sigma * (h + zeta)
    base_centre_error: np.ndarray
    target_centres: np.ndarray  # (target columns, layers)
    target_centre_error: np.ndarray
    base_zeta: np.ndarray
    base_zeta_error: np.ndarray
    target_zeta: np.ndarray
    target_zeta_error: np.ndarray
    base_depth: np.ndarray  # h + zeta of the base columns
    base_depth_error: np.ndarray
    volumes: np.ndarray  # (target columns, layers)
    nodal: tuple  # float64 node depths/elevations: base h, base zeta, target h, target zeta
    triangles: object  # (base, target) triangles for element columns, else None
    sigma: tuple
    cache: dict  # exact donor columns

    def _exact(self, values, triangles, i):
        """Exact column value: the node's, or the exact mean of an element's nodes."""
        if triangles is None:
            return Fraction(float(values[i]))
        return sum(Fraction(float(values[v])) for v in triangles[i]) / 3

    def exact(self, i, j, layers):
        """Exact (rational) geometry of target i against donor j.

        Returns the donor's water column and layer centres (relative to its
        surface) and target i's centres of ``layers`` relative to the same
        surface, from the float64 inputs without rounding (round 12).
        """
        bh, bz, th, tz = self.nodal
        tb, tt = self.triangles or (None, None)
        sigma = [Fraction(x) for x in self.sigma]
        middles = [(a + b) / 2 for a, b in zip(sigma, sigma[1:])]
        if j not in self.cache:
            column = self._exact(bh, tb, j) + self._exact(bz, tb, j)
            self.cache[j] = (
                self._exact(bz, tb, j),
                column,
                [m * column for m in middles],
            )
        surface, column, centres = self.cache[j]
        target_surface = self._exact(tz, tt, i)
        target_column = self._exact(th, tt, i) + target_surface
        targets = [
            target_surface - surface + middles[k] * target_column for k in layers
        ]
        return column, centres, targets

    def relative(self, i, j):
        """Target i's layer centres relative to donor j's surface, with error bounds.

        The difference of the two surfaces is formed first (one rounding of
        two exact inputs), then target i's own relative centres are added.
        """
        u = _UNIT_ROUNDOFF
        with np.errstate(over="ignore", invalid="ignore"):
            offset = self.target_zeta[i] - self.base_zeta[j]
            z = offset + self.target_centres[i]
        error = (
            self.target_zeta_error[i]
            + self.base_zeta_error[j]
            + 2 * u * abs(offset)
            + self.target_centre_error[i]
            + 2 * u * np.abs(z)
        )
        return z, error


def _fractions(z, targets):
    """Bracket index, fraction and end classification of targets in donor centres z."""
    order = np.searchsorted(-z, -targets, side="right") - 1
    j = np.clip(order, 0, len(z) - 2)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        t = (targets - z[j]) / (z[j + 1] - z[j])
    _finite("interpolation fractions", t)
    return j, t, targets >= z[0], targets <= z[-1]


def _uncertain(z, z_error, targets, target_error, j, t):
    """Layers whose float classification or fraction cannot be certified.

    A target within its error of an end centre may be classified on the wrong
    side (and an end value is not continuous with a far bracket when the error
    exceeds a layer); inside the column, t = N / D with N = z[j] - target and
    D = z[j] - z[j+1] > 0 has the first-order bound (eN + |t| eD) / (D - eD)
    plus the quotient's rounding, certified to FRACTION_TOLERANCE.
    """
    u = _UNIT_ROUNDOFF
    near_end = (np.abs(targets - z[0]) <= target_error + z_error[0]) | (
        np.abs(targets - z[-1]) <= target_error + z_error[-1]
    )
    # A target within its error of an INTERNAL centre may lie in the other
    # bracket; that is harmless only if the error is negligible against the
    # narrower of the two brackets around that centre (round 13).
    spacing = z[:-1] - z[1:]
    for c in (j, j + 1):
        error = target_error + z_error[c]
        internal = (c > 0) & (c < len(z) - 1)
        narrow = np.minimum(
            spacing[np.clip(c - 1, 0, len(spacing) - 1)],
            spacing[np.clip(c, 0, len(spacing) - 1)],
        )
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            risky = (narrow <= 2 * error) | (
                2 * error / (narrow - 2 * error) > FRACTION_TOLERANCE
            )
        near_end |= internal & (np.abs(targets - z[c]) <= error) & risky
    numerator = z[j] - targets
    denominator = z[j] - z[j + 1]
    e_numerator = z_error[j] + target_error + 2 * u * np.abs(numerator)
    e_denominator = z_error[j] + z_error[j + 1] + 2 * u * denominator
    fraction = np.abs(t)
    margin = denominator - e_denominator
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        bound = np.where(
            margin > 0,
            (e_numerator + fraction * e_denominator) / margin + 2 * u * fraction,
            np.inf,
        )
    inside = (targets < z[0]) & (targets > z[-1])
    return near_end | (inside & ~(bound <= FRACTION_TOLERANCE))


def _exact_fractions(column, centres, targets):
    """Exact bracket, fraction, end and extension classes of rational targets."""
    n = len(targets)
    j, t = np.zeros(n, dtype=int), np.zeros(n)
    above, below = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    extended, beyond = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    position, extension, clamping = np.zeros(n), np.zeros(n), np.zeros(n)
    for k, r in enumerate(targets):
        above[k], below[k] = r >= centres[0], r <= centres[-1]
        beyond[k] = r > centres[0] or r < centres[-1]
        extended[k] = r > 0 or r < -column
        b = 0
        while b < len(centres) - 2 and r < centres[b + 1]:
            b += 1
        j[k] = b
        t[k] = float((r - centres[b]) / (centres[b + 1] - centres[b]))
        position[k] = float(r)
        extension[k] = float(max(r, -column - r))
        clamping[k] = float(max(r - centres[0], centres[-1] - r))
    return j, t, above, below, extended, beyond, position, extension, clamping


def _interpolate(values, j, t, above, below):
    """Return constant-end interpolation and its unbounded linear alternative."""
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
    clipped[above] = values[0]
    clipped[below] = values[-1]
    linear = clipped.copy()
    outside = above | below
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
    columns = _columns(op, location, sigma, base_zeta, target_zeta)
    zb, volume, depth = columns.base_centres, columns.volumes, columns.base_depth
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
            # every coordinate is relative to donor j's own surface
            zt, zt_error = columns.relative(i, j)
            ze = columns.base_centre_error[j]
            bracket, t, above, below = _fractions(zb[j], zt)
            extended = (zt > 0) | (zt < -depth[j])
            beyond_centres = (zt > zb[j, 0]) | (zt < zb[j, -1])
            distances = {
                "extension": np.maximum(zt, -depth[j] - zt),
                "clamping": np.maximum(zt - zb[j, 0], zb[j, -1] - zt),
            }
            # Layers the float64 frame cannot certify are computed exactly
            # (rational arithmetic on the inputs), never refused (round 12).
            uncertain = _uncertain(zb[j], ze, zt, zt_error, bracket, t) | (
                (np.abs(zt) <= zt_error)
                | (np.abs(zt + depth[j]) <= zt_error + columns.base_depth_error[j])
            )
            if uncertain.any():
                layers = np.flatnonzero(uncertain)
                exact = _exact_fractions(*columns.exact(i, j, layers))
                for array, value in zip(
                    (bracket, t, above, below, extended, beyond_centres),
                    exact[:6],
                ):
                    array[layers] = value
                distances["extension"][layers] = exact[7]
                distances["clamping"][layers] = exact[8]
            values, linear = _interpolate(
                field[j].astype(float), bracket, t, above, below
            )
            with np.errstate(over="ignore", invalid="ignore"):
                result[i] += w * values
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

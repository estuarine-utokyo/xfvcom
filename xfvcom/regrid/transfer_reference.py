"""Independent rational evaluator; shares only input validation with transfer."""

import heapq
import math
from fractions import Fraction

import numpy as np


def _finite(label, *values):
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError(f"{label}: non-finite derived value or statistic")


def _product(xs, ys):
    """Exact product of two expansions, as an expansion (error-free products)."""
    out = []
    for a in xs:
        for b in ys:
            p = a * b
            out += [p, math.fma(a, b, -p)]
    return out


def _difference(xs, ys):
    """Correctly rounded value of sum(xs) - sum(ys) (exact sign, exact zero)."""
    return math.fsum(xs + [-y for y in ys])


def _sign(xs, ys):
    difference = _difference(xs, ys)
    return (difference > 0) - (difference < 0)


def independent_evaluate(
    base,
    target,
    node_map,
    elem_map,
    field,
    *,
    location="node",
    sigma=None,
    base_zeta=None,
    target_zeta=None,
    max_distance=500.0,
    return_scale=False,
):
    """Slow reference for already validated inputs; no operator or shared weights.

    Own triangle search, graph walk, element area average and vertical loops.
    All searched barycentrics and incident areas use exact Fraction arithmetic
    on the input floats. Vertical coordinates are exact expansions relative to
    each donor's own surface (error-free products and fsum, unlike
    production's certified float64 frame); each interpolation fraction is a
    quotient of two correctly rounded exact differences. Only field
    combinations use plain floating point. Rational
    elimination deliberately differs from production's compensated determinants.
    Evaluates ALL rows, including copies, with no cached geometry from Transfer.
    Use build_transfer/apply_* first to validate raw application inputs.
    """
    from .transfer import _distance_limit

    if location not in ("node", "element"):
        raise ValueError("location must be 'node' or 'element'")
    max_distance = _distance_limit(max_distance)
    field = np.asarray(field)
    # Copies keep the input bytes; all arithmetic is in float64, so that a
    # float32 field is not rounded product by product (NumPy keeps
    # Python float * np.float32 in float32).
    original, field = field, field.astype(np.float64)
    layered = sigma is not None
    if layered:
        sigma = [float(x) for x in sigma]  # float64 working type throughout
    polygons = np.asarray(base.xy, dtype=float)[base.triangles]
    lower, upper = polygons.min(axis=1), polygons.max(axis=1)
    size = (upper - lower).max(axis=1)
    rational = []
    for vertices in polygons:
        a, b, c = [[Fraction(float(x)) for x in p] for p in vertices]
        rational.append(
            (a, [b[k] - a[k] for k in (0, 1)], [c[k] - a[k] for k in (0, 1)])
        )

    def find(point):
        # Bounding boxes only accelerate the search. Solve the translated 2x2
        # system by exact rational elimination, without production determinants.
        # A point may be exact (Fractions: an element centroid). The boxes only
        # preselect: a point the clipping rule accepts has every weight above
        # about -1e-12, so it lies within 1e-12 of the triangle's size outside
        # it; padding by 1e-11 of the box size, plus one spacing of the rounded
        # query, keeps every such triangle (round 14).
        x, y = (float(c) for c in point)
        pad = 1e-11 * size + np.spacing(max(abs(x), abs(y)))
        possible = np.flatnonzero(
            (lower[:, 0] <= x + pad)
            & (upper[:, 0] >= x - pad)
            & (lower[:, 1] <= y + pad)
            & (upper[:, 1] >= y - pad)
        )
        for e in possible:
            a, u, v = rational[e]
            dx, dy = (
                (c if isinstance(c, Fraction) else Fraction(float(c))) - a[k]
                for k, c in enumerate(point)
            )
            if u[0]:
                ratio = u[1] / u[0]
                w2 = (dy - ratio * dx) / (v[1] - ratio * v[0])
                w1 = (dx - v[0] * w2) / u[0]
            else:
                w2 = dx / v[0]
                w1 = (dy - v[1] * w2) / u[1]
            weights = [1 - w1 - w2, w1, w2]
            if min(weights) >= -Fraction(2, 10**12):
                clipped = [max(w, 0) for w in weights]
                s = sum(clipped)
                clipped = [w / s for w in clipped]
                # Same acceptance criterion as production, reached with exact
                # arithmetic: a boundary candidate whose clipped weights are not
                # within 1e-12 of the point's own weights does not contain it.
                if max(abs(c - w) for c, w in zip(clipped, weights)) > Fraction(
                    1, 10**12
                ):
                    continue
                return int(e), [float(w) for w in clipped]
        return -1, None

    # Like production, every outside query starts its own reverse search (its
    # node, or a centroid's three vertices). Separately coded; the shared
    # ASSUMPTIONS are the edge-path metric, the rounding tie band and the
    # search cutoff derived from the shortest edge.
    membership = [find(p) for p in target.xy]
    vertex_triangle = {}
    for e, tri in enumerate(base.triangles):
        for v in tri:
            if int(v) not in vertex_triangle:
                vertex_triangle[int(v)] = (e, [float(k == v) for k in tri])
    for old, new in enumerate(node_map):
        if new >= 0:
            membership[int(new)] = vertex_triangle[old]
    neighbours = [dict() for _ in target.xy]
    for tri in target.triangles:
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            dx = float(target.xy[a, 0]) - float(target.xy[b, 0])
            dy = float(target.xy[a, 1]) - float(target.xy[b, 1])
            length = math.hypot(dx, dy)
            neighbours[a][b] = length
            neighbours[b][a] = length

    # Two candidate paths are tied when their computed lengths differ by no
    # more than their rounding bound: 4u per edge (and one) of the longer path,
    # plus twice the rounding of a rounded query point. Bit-equal ties go to
    # the lowest ID; other ties are refused, as in production.
    unit = 2.0**-53
    shortest_edge = min(min(n.values()) for n in neighbours if n)

    def anchor(starts, point_error=0.0):
        queue = [(d, hop, int(v)) for d, v, hop in starts]
        heapq.heapify(queue)
        visited, found = set(), []
        while queue:
            distance, hop, v = heapq.heappop(queue)
            if v in visited:
                continue
            visited.add(v)
            # Until an anchor is found, stop only beyond the limit plus the
            # rounding such a path can carry; once one is found, complete its
            # tie band even past the limit (round 14), then apply the limit.
            reach = 4 * unit * (distance / shortest_edge + 2) * distance
            if not found and distance - max_distance > reach + 2 * point_error:
                break
            if found:
                # a path this long has at most distance / shortest-edge hops:
                # beyond this nothing can enter the tie band of the first
                d0, _, h0 = found[0]
                band = 4 * unit * (h0 + distance / shortest_edge + 2) * distance
                if distance - d0 > band + 2 * point_error:
                    break
            if membership[v][0] >= 0:
                found.append((distance, v, hop))
            for neighbour, length in neighbours[v].items():
                if neighbour not in visited:
                    heapq.heappush(queue, (distance + length, hop + 1, neighbour))
        if not found:
            raise ValueError("Independent evaluator: no anchor within max_distance")
        best = min(found)
        if best[0] > max_distance:
            raise ValueError("Independent evaluator: no anchor within max_distance")
        close = [
            c
            for c in found
            if abs(c[0] - best[0])
            # every route that could be shortest: at most length / shortest edge
            # edges each (round 16), not the hop count of the route kept
            <= 4 * unit * ((c[0] + best[0]) / shortest_edge + 2) * max(c[0], best[0])
            + 2 * point_error
        ]
        if any(c[0] != best[0] for c in close if c[1] != best[1]):
            raise ValueError("Independent evaluator: ambiguous anchor (rounding band)")
        return membership[min(c[1] for c in close if c[0] == best[0])]

    inverse = {}
    mapping = node_map if location == "node" else elem_map
    for old, new in enumerate(mapping):
        if new >= 0:
            inverse[int(new)] = old
    # Vertical columns as EXPANSIONS: lists of floats whose exact sum is the
    # quantity. A node column is its own depth and elevation; an element column
    # is the node mean, kept exact by scaling every element column (base and
    # target alike) by 3, which leaves interpolation fractions unchanged.
    if location == "node":
        points = target.xy
        old_cells = [[v] for v in range(len(base.xy))]
        new_cells = [[v] for v in range(len(target.xy))]
    else:
        # exact centroids: the sum of the vertex floats over three
        points = [
            tuple(
                sum(Fraction(float(target.xy[v, k])) for v in tri) / 3 for k in (0, 1)
            )
            for tri in target.triangles
        ]
        old_cells = [[int(v) for v in tri] for tri in base.triangles]
        new_cells = [[int(v) for v in tri] for tri in target.triangles]
    if layered:
        old_depth = [[float(base.depths[v]) for v in c] for c in old_cells]
        old_surface = [[float(base_zeta[v]) for v in c] for c in old_cells]
        new_depth = [[float(target.depths[v]) for v in c] for c in new_cells]
        new_surface = [[float(target_zeta[v]) for v in c] for c in new_cells]
        # sigma-layer centres, exactly: halving a float is exact
        middles = [[sigma[k] / 2, sigma[k + 1] / 2] for k in range(len(sigma) - 1)]
        donor_cache = {}
        target_cache = {}

        def target_centres(i):
            # target surface plus centre k relative to it (cached per target)
            if i not in target_cache:
                column = new_depth[i] + new_surface[i]
                target_cache.clear()
                target_cache[i] = [
                    new_surface[i] + _product(m, column) for m in middles
                ]
            return target_cache[i]

        def donor_centres(d):
            # centre k relative to donor d's own surface: middle * (h + zeta)
            if d not in donor_cache:
                column = old_depth[d] + old_surface[d]
                centres = [_product(m, column) for m in middles]
                gaps = [_difference(a, b) for a, b in zip(centres, centres[1:])]
                donor_cache[d] = centres, gaps
            return donor_cache[d]

    incident = [[] for _ in base.xy]
    reference_area = []
    for e, tri in enumerate(base.triangles):
        _, u, v = rational[e]
        reference_area.append(float(abs(u[0] * v[1] - v[0] * u[1]) / 2))
        for v in tri:
            incident[v].append(e)
    output = np.zeros((len(points),) + field.shape[1:], dtype=field.dtype)
    # Per output value: the sum of |donor value| over the values that enter it.
    # It bounds the round-off of the combination and is built from this
    # evaluator's own donors, per position, layer and component.
    scale = np.zeros(output.shape)
    for i, point in enumerate(points):
        if i in inverse:
            output[i] = original[inverse[i]]
            scale[i] = np.abs(field[inverse[i]])
            continue
        e, weights = membership[i] if location == "node" else find(point)
        if e < 0:
            if location == "node":
                e, weights = anchor([(0.0, i, 0)])
            else:
                starts = []
                for v in target.triangles[i]:
                    dx = float(point[0]) - float(target.xy[v, 0])
                    dy = float(point[1]) - float(target.xy[v, 1])
                    starts.append((math.hypot(dx, dy), v, 1))
                # the anchor search starts from the centroid rounded to float64
                e, weights = anchor(
                    starts, 2 * unit * max(abs(float(c)) for c in point)
                )
        donors = {}
        for v, w in zip(base.triangles[e], weights):
            if location == "node":
                donors[int(v)] = w
            else:
                total_area = sum(reference_area[k] for k in incident[v])
                for k in incident[v]:
                    donors[k] = donors.get(k, 0.0) + w * reference_area[k] / total_area
        value = np.zeros(field.shape[1:], dtype=float)
        for d, weight in donors.items():
            # A donor with zero weight still bounds the round-off of a weight
            # that is zero only exactly (e.g. an edge midpoint), so it enters
            # the scale; its contribution to the value is zero.
            if not layered:
                value += weight * field[d]
                scale[i] += np.abs(field[d])
                continue
            centres, gaps = donor_centres(d)
            below_surface = [-z for z in old_surface[d]]
            layers = len(sigma) - 1
            bracket = np.zeros(layers, dtype=int)
            fraction = np.zeros(layers)
            above = np.zeros(layers, dtype=bool)
            below = np.zeros(layers, dtype=bool)
            # Target centres decrease with k, exactly, and so do the donor's:
            # the state goes above -> inside -> below once, and the bracket
            # index only moves down (a merge walk).
            state, j = "above", 0
            for k in range(layers):
                # target centre k relative to donor d's surface:
                # (target surface - donor surface) + middle * target column
                z = target_centres(i)[k] + below_surface
                if state == "above" and _sign(z, centres[0]) < 0:
                    state = "inside"
                if state == "inside" and _sign(z, centres[-1]) <= 0:
                    state = "below"
                if state == "above":
                    above[k] = True
                elif state == "below":
                    below[k] = True
                else:
                    while _sign(z, centres[j + 1]) < 0:
                        j += 1
                    # a correctly rounded exact difference over another
                    bracket[k], fraction[k] = j, _difference(centres[j], z) / gaps[j]
            f = fraction.reshape((layers,) + (1,) * (field.ndim - 2))
            lower_value, upper_value = field[d, bracket], field[d, bracket + 1]
            interpolated = (1 - f) * field[d, bracket].astype(
                np.longdouble
            ) + f * field[d, bracket + 1].astype(np.longdouble)
            interpolated = np.clip(
                interpolated,
                np.minimum(lower_value, upper_value),
                np.maximum(lower_value, upper_value),
            )
            interpolated[above] = field[d, 0]
            interpolated[below] = field[d, -1]
            inside = ~(above | below)
            scale[i, inside] += np.abs(lower_value[inside]) + np.abs(
                upper_value[inside]
            )
            scale[i, above] += np.abs(field[d, 0])
            scale[i, below] += np.abs(field[d, -1])
            value += weight * interpolated
        with np.errstate(over="ignore", invalid="ignore"):
            output[i] = value
    _finite("independent evaluation", output)
    return (output, scale) if return_scale else output

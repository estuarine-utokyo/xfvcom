"""Independent rational evaluator; shares only input validation with transfer."""

import heapq
import math
from fractions import Fraction

import numpy as np


def _finite(label, *values):
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError(f"{label}: non-finite derived value or statistic")


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
    on the input floats; only field combinations use floating point. Rational
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
    rational = []
    for vertices in polygons:
        a, b, c = [[Fraction(float(x)) for x in p] for p in vertices]
        rational.append(
            (a, [b[k] - a[k] for k in (0, 1)], [c[k] - a[k] for k in (0, 1)])
        )

    def find(point):
        # Bounding boxes only accelerate the search. Solve the translated 2x2
        # system by exact rational elimination, without production determinants.
        possible = np.flatnonzero(
            (lower[:, 0] <= point[0])
            & (upper[:, 0] >= point[0])
            & (lower[:, 1] <= point[1])
            & (upper[:, 1] >= point[1])
        )
        for e in possible:
            a, u, v = rational[e]
            dx, dy = (Fraction(float(point[k])) - a[k] for k in (0, 1))
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

    # This is intentionally not the production multi-source shortest-path walk.
    # Every outside query starts a fresh search from its own vertex/vertices.
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

    def anchor(starts):
        queue = [(d, int(v)) for d, v in starts]
        heapq.heapify(queue)
        visited = set()
        while queue:
            distance, v = heapq.heappop(queue)
            if v in visited:
                continue
            visited.add(v)
            if distance > max_distance:
                break
            if membership[v][0] >= 0:
                return membership[v]
            for neighbour, length in neighbours[v].items():
                if neighbour not in visited:
                    heapq.heappush(queue, (distance + length, neighbour))
        raise ValueError("Independent evaluator: no anchor within max_distance")

    inverse = {}
    mapping = node_map if location == "node" else elem_map
    for old, new in enumerate(mapping):
        if new >= 0:
            inverse[int(new)] = old
    if location == "node":
        points = target.xy
        old_h = [float(h) for h in base.depths]
        new_h = [float(h) for h in target.depths]
        if layered:
            old_surface = [float(z) for z in base_zeta]
            new_surface = [float(z) for z in target_zeta]
    else:
        points = [
            sum((target.xy[v] for v in tri), np.zeros(2)) / 3
            for tri in target.triangles
        ]
        old_h = [sum(float(base.depths[v]) for v in tri) / 3 for tri in base.triangles]
        new_h = [
            sum(float(target.depths[v]) for v in tri) / 3 for tri in target.triangles
        ]
        if layered:
            old_surface = [
                sum(float(base_zeta[v]) for v in tri) / 3 for tri in base.triangles
            ]
            new_surface = [
                sum(float(target_zeta[v]) for v in tri) / 3 for tri in target.triangles
            ]
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
                e, weights = anchor([(0.0, i)])
            else:
                starts = []
                for v in target.triangles[i]:
                    dx, dy = point[0] - target.xy[v, 0], point[1] - target.xy[v, 1]
                    starts.append((math.hypot(dx, dy), v))
                e, weights = anchor(starts)
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
            donor_centres = [
                old_surface[d]
                + (sigma[k] + sigma[k + 1]) / 2 * (old_h[d] + old_surface[d])
                for k in range(len(sigma) - 1)
            ]
            for k in range(len(sigma) - 1):
                z = new_surface[i] + (sigma[k] + sigma[k + 1]) / 2 * (
                    new_h[i] + new_surface[i]
                )
                if z >= donor_centres[0]:
                    interpolated = field[d, 0]
                    scale[i, k] += np.abs(field[d, 0])
                elif z <= donor_centres[-1]:
                    interpolated = field[d, -1]
                    scale[i, k] += np.abs(field[d, -1])
                else:
                    for j in range(len(donor_centres) - 1):
                        if donor_centres[j] >= z >= donor_centres[j + 1]:
                            fraction = (donor_centres[j] - z) / (
                                donor_centres[j] - donor_centres[j + 1]
                            )
                            interpolated = (1 - fraction) * field[d, j].astype(
                                np.longdouble
                            ) + fraction * field[d, j + 1].astype(np.longdouble)
                            interpolated = np.clip(
                                interpolated,
                                np.minimum(field[d, j], field[d, j + 1]),
                                np.maximum(field[d, j], field[d, j + 1]),
                            )
                            scale[i, k] += np.abs(field[d, j]) + np.abs(field[d, j + 1])
                            break
                value[k] += weight * interpolated
        with np.errstate(over="ignore", invalid="ignore"):
            output[i] = value
    _finite("independent evaluation", output)
    return (output, scale) if return_scale else output

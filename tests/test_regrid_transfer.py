"""Synthetic transfer, analytic statistics, independent checks and in-memory faults."""

import io
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import load_npz, save_npz

from xfvcom.regrid import mesh_pair as rm
from xfvcom.regrid import transfer as rt
from xfvcom.regrid import transfer_reference

SOURCE = Path(rt.__file__).read_text()
REFERENCE_SOURCE = Path(transfer_reference.__file__).read_text()
SIGMA = np.linspace(0.0, -1.0, 31)


@pytest.fixture
def pair():
    return synthetic_pair()


def mesh(xy, tri, h):
    return rm.Mesh(
        np.asarray(xy, dtype=float),
        np.asarray(tri, dtype=np.int64),
        np.broadcast_to(np.asarray(h, dtype=float), (len(xy),)).copy(),
    )


def synthetic_pair():
    # Unequal triangle areas, unequal depths; one triangle split, one matched.
    base = mesh(
        [[0, 0], [3, 0], [2, 2], [0, 1]], [[0, 1, 2], [0, 2, 3]], [8, 12, 16, 10]
    )
    target = mesh(
        np.vstack((base.xy, [0.6, 0.4])),
        [[0, 1, 4], [1, 2, 4], [2, 0, 4], [0, 2, 3]],
        [8, 12, 16, 10, 11],
    )
    return base, target, np.arange(4), np.array([-1, 3])


def build(pair):
    return rt.build_transfer(*pair, max_distance=500.0)


def centres(mesh, sigma, zeta, location):
    h = mesh.depths
    if location == "element":
        h = h[mesh.triangles].mean(axis=1)
        zeta = zeta[mesh.triangles].mean(axis=1)
    return zeta[:, None] + ((sigma[:-1] + sigma[1:]) / 2) * (h + zeta)[:, None]


def bitwise(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def test_g1_identity_and_permutation(pair):
    base = pair[0]
    for permuted in (False, True):
        permutation = (
            np.random.default_rng(19).permutation(len(base.xy))
            if permuted
            else np.arange(len(base.xy))
        )
        element_order = np.arange(len(base.triangles))
        if permuted:
            element_order = element_order[::-1]
        target = mesh(
            base.xy[permutation],
            np.argsort(permutation)[base.triangles[element_order]],
            base.depths[permutation],
        )
        nm = np.argsort(permutation)
        em = np.argsort(element_order)
        op = build((base, target, nm, em))
        for location, mapping in (("node", nm), ("element", em)):
            for dtype in (np.float32, np.float64):
                n = len(mapping)
                field = np.arange(n * 30 * 2, dtype=dtype).reshape(n, 30, 2)
                field[0, 0] = [-0.0, np.nextafter(dtype(0), dtype(1))]
                expected = np.empty_like(field)
                expected[mapping] = field
                zeta = np.full(len(base.xy), 0.7)
                new_zeta = zeta[permutation]
                got, stats = rt.apply_3d(
                    op, field, SIGMA, zeta, new_zeta, location=location
                )
                bitwise(got, expected)
                bitwise(rt.apply_2d(op, field[:, 0], location=location), expected[:, 0])
                assert stats["target_volume_m3"] == 0


def test_g1_frozen_and_matched(pair):
    op = build(pair)
    rng = np.random.default_rng(193)
    for location, copies in (("node", op.node_copy), ("element", op.element_copy)):
        n = len(pair[0].xy) if location == "node" else len(pair[0].triangles)
        values = rng.normal(size=(n, 30, 2)).astype(np.float32)
        values[:, 0, 0] = -0.0
        keep = copies >= 0
        assert keep.any()
        zeta = np.full(len(pair[0].xy), 0.4)
        tzeta = rt.apply_2d(op, zeta)
        result, _ = rt.apply_3d(op, values, SIGMA, zeta, tzeta, location=location)
        bitwise(result[keep], values[copies[keep]])
        bitwise(
            rt.apply_2d(op, values[:, 0], location=location)[keep],
            values[copies[keep], 0],
        )


def test_g2_constants_linear_nodes_and_weights(pair):
    op = build(pair)
    base, target = pair[:2]
    inside = np.ones(len(target.xy), bool)
    inside[op.node_report["outside_nodes"]] = False
    # Offset the map projection origin so round-off tolerance is meaningful.
    origin = base.xy.mean(axis=0)
    analytic = (
        lambda xy: 0.37 * (xy[:, 0] - origin[0]) - 0.81 * (xy[:, 1] - origin[1]) + 3.0
    )
    for matrix in (op.node_matrix, op.element_matrix):
        assert (matrix.data >= 0).all()
        np.testing.assert_allclose(
            np.asarray(matrix.sum(axis=1)).ravel(), 1.0, rtol=0, atol=2e-14
        )
        # Exercise G2 saved/reloaded weights in memory; no production cache here.
        buffer = io.BytesIO()
        save_npz(buffer, matrix)
        buffer.seek(0)
        reloaded = load_npz(buffer)
        assert (reloaded != matrix).nnz == 0
    np.testing.assert_allclose(
        rt.apply_2d(op, analytic(base.xy))[inside],
        analytic(target.xy)[inside],
        rtol=2e-13,
        atol=2e-9,
    )
    for location, matrix in (("node", op.node_matrix), ("element", op.element_matrix)):
        np.testing.assert_allclose(
            rt.apply_2d(op, np.full(matrix.shape[1], 7.3), location=location),
            7.3,
            rtol=0,
            atol=8e-15,
        )


def element_errors(op):
    base, target = op.base, op.target
    bxy, txy = base.xy[base.triangles], target.xy[target.triangles]
    bc, tc = bxy.mean(axis=1), txy.mean(axis=1)
    origin = base.xy.mean(axis=0)
    size = np.linalg.norm(bxy - np.roll(bxy, 1, axis=1), axis=2).max(axis=1)
    # Smooth field with a physically resolved 10-km length scale.
    fields = {
        "x": lambda p: p[..., 0],
        "smooth": lambda p: np.sin((p[..., 0] - origin[0]) / 10000.0)
        + 0.4 * np.cos((p[..., 1] - origin[1]) / 10000.0),
    }
    metrics = {}
    for name, f in fields.items():
        error = np.abs(rt.apply_2d(op, f(bc), location="element") - f(tc))
        element_variation = np.ptp(f(bxy), axis=1)
        variation = element_variation.max()
        local_variation = np.array(
            [
                element_variation[
                    op.element_matrix.indices[
                        op.element_matrix.indptr[i] : op.element_matrix.indptr[i + 1]
                    ]
                ].max()
                for i in range(len(tc))
            ]
        )
        unmatched = op.element_copy < 0
        assert unmatched.any()
        assert error.max() <= variation + 2e-12
        # Section 6q does not say which element sets the variation scale. The
        # domain maximum is the common bound, including outside centroids.
        # A local bound also holds inside; outside, anchor displacement adds
        # error and must be included (seven real x-errors exceed the raw local
        # variation). Report the unadjusted ratio so this is never hidden.
        inside = op.element_report["anchor_nodes"] < 0
        assert (error[inside] <= local_variation[inside] + 2e-12).all()
        lipschitz = 1.0 if name == "x" else 0.00014
        anchor_distance = op.element_report["anchor_distances_m"]
        assert (error <= local_variation + lipschitz * anchor_distance + 2e-12).all()
        # Also a local size/Lipschitz bound, using every triangle in the donor
        # stencil and the target's containing triangle/anchor displacement.
        assert error.max() <= (1.0 if name == "x" else 0.00014) * size.max() + 2e-12
        metrics[name] = {
            "max_abs_error": float(error[unmatched].max()),
            "rms_error": float(np.sqrt(np.mean(error[unmatched] ** 2))),
            "base_element_variation_bound": float(variation),
            "max_error_per_local_element_variation": float(
                np.max(error / np.maximum(local_variation, 1e-30))
            ),
            "elements_above_unadjusted_local_variation": int(
                (error > local_variation + 2e-12).sum()
            ),
            "max_error_per_max_base_edge": float(error.max() / size.max()),
        }
    metrics["base_edge_median_m"] = float(np.median(size))
    metrics["base_edge_max_m"] = float(size.max())
    return metrics


def test_g2_element_accuracy_and_area_share(pair):
    op = build(pair)
    metrics = element_errors(op)
    assert metrics["x"]["max_abs_error"] > 0  # no affine-exactness claim
    if len(pair[0].xy) == 4:
        # At new centroid (.6,.13333), containing base triangle (0,1,2),
        # barycentric weights (.77777,.15555,.06666). Vertices 0 and 2
        # see base areas 3:1, vertex 1 only triangle 0.
        centroid = pair[1].xy[pair[1].triangles[0]].mean(axis=0)
        c = centroid[1] / 2
        b = (centroid[0] - 2 * c) / 3
        expected = (1 - b) * 0.25
        assert op.element_matrix[0, 1] == pytest.approx(expected, abs=2e-16)
    print("ELEMENT_ERRORS", metrics)


def test_g2_vertical_physical_z_and_saturation(pair):
    op = build(pair)
    base, target = pair[:2]
    bz = 0.7 + 0.05 * np.sin(np.arange(len(base.xy)))
    tz = rt.apply_2d(op, bz)
    for location, matrix, copies in (
        ("node", op.node_matrix, op.node_copy),
        ("element", op.element_matrix, op.element_copy),
    ):
        zb = centres(base, SIGMA, bz, location)
        zt = centres(target, SIGMA, tz, location)
        field = 11.0 + 0.37 * zb
        got, stats = rt.apply_3d(op, field, SIGMA, bz, tz, location=location)
        interior = np.zeros(zt.shape, bool)
        expected_saturated = np.zeros_like(zt)
        for i in np.flatnonzero(copies < 0):
            js = matrix.indices[matrix.indptr[i] : matrix.indptr[i + 1]]
            ws = matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]]
            interior[i] = (zt[i] <= zb[js, 0].min()) & (zt[i] >= zb[js, -1].max())
            for j, w in zip(js, ws):
                expected_saturated[i] += w * (
                    11 + 0.37 * np.clip(zt[i], zb[j, -1], zb[j, 0])
                )
        assert interior.any()
        np.testing.assert_allclose(
            got[interior], (11 + 0.37 * zt)[interior], rtol=3e-14, atol=3e-14
        )
        keep = copies < 0
        np.testing.assert_allclose(
            got[keep], expected_saturated[keep], rtol=3e-14, atol=3e-14
        )
        assert stats["clamping"]["weighted_volume_m3"] > 0
        assert stats["extension"]["weighted_volume_m3"] > 0


def test_g2_bounded_discontinuities(pair):
    op = build(pair)
    bz, tz = np.zeros(len(pair[0].xy)), np.zeros(len(pair[1].xy))
    for location, matrix in (("node", op.node_matrix), ("element", op.element_matrix)):
        field = np.where(np.arange(matrix.shape[1]) % 2, -4.0, 13.0)
        horizontal = rt.apply_2d(op, field, location=location)
        layered = np.where(
            (np.arange(matrix.shape[1])[:, None] + np.arange(30)) % 2, -4.0, 13.0
        )
        result, _ = rt.apply_3d(op, layered, SIGMA, bz, tz, location=location)
        assert horizontal.min() >= -4.0 - 1e-14 and horizontal.max() <= 13.0 + 1e-14
        assert result.min() >= -4.0 - 1e-14 and result.max() <= 13.0 + 1e-14


def outside_pair():
    # Two pieces of old coastline. Euclidean-nearest donor 3 is across land;
    # target 6 connects only to the left triangle and must choose node 1.
    base = mesh(
        [[0, 0], [1, 0], [0, 1], [1.12, -0.01], [2, -0.01], [2, 1]],
        [[0, 1, 2], [3, 4, 5]],
        3.0,
    )
    target = mesh(
        np.vstack((base.xy, [1.1, -0.1])), [[0, 1, 2], [3, 4, 5], [0, 6, 1]], 3.0
    )
    return base, target, np.arange(6), np.array([0, 1])


def test_outside_path_anchor_limit_and_ties():
    pair = outside_pair()
    op = build(pair)
    assert op.node_report["outside_nodes"].tolist() == [6]
    assert np.argmin(np.linalg.norm(pair[0].xy - pair[1].xy[6], axis=1)) == 3
    assert op.node_report["donor_nodes"][6] == 1
    assert op.node_report["donor_distances_m"][6] == pytest.approx(np.sqrt(0.02))
    assert op.node_matrix[6, 1] == 1 and op.node_matrix[6, 3] == 0
    assert op.element_report["outside_elements"].tolist() == [2]
    assert op.element_report["anchor_nodes"][2] == 1
    centroid = pair[1].xy[pair[1].triangles[2]].mean(axis=0)
    assert op.element_report["anchor_distances_m"][2] == pytest.approx(
        np.linalg.norm(centroid - [1, 0])
    )
    assert op.element_matrix[2, 0] == 1 and op.element_matrix[2, 1] == 0
    for limit in (0.1, 0.2):  # .1 rejects node; .2 accepts node but rejects centroid
        with pytest.raises(ValueError, match="max_distance"):
            rt.build_transfer(*pair, max_distance=limit)
    # Exact edge-path tie: equidistant endpoints; centroid tie as well.
    b = mesh([[0, 0], [2, 0], [0, 2]], [[0, 1, 2]], 3.0)
    t = mesh(np.vstack((b.xy, [1, -1])), [[0, 1, 2], [0, 3, 1]], 3.0)
    tie = build((b, t, np.arange(3), np.array([0])))
    assert tie.node_report["donor_nodes"][3] == 0
    assert tie.element_report["anchor_nodes"][1] == 0
    field = np.arange(6, dtype=float)
    rt.check(op, field)
    rt.check(op, np.array([4.0, 90.0]), location="element")


def test_outside_donor_can_be_new_inside_node():
    b = mesh([[0, 0], [2, 0], [0, 2]], [[0, 1, 2]], 3.0)
    t = mesh(
        np.vstack((b.xy, [1, 0], [1, -0.2])),
        [[0, 3, 2], [3, 1, 2], [0, 4, 3], [3, 4, 1]],
        3.0,
    )
    op = build((b, t, np.arange(3), np.array([-1])))
    assert op.node_report["donor_nodes"][4] == 3
    np.testing.assert_array_equal(op.node_matrix.toarray()[4], [0.5, 0.5, 0])
    np.testing.assert_array_equal(
        rt.apply_2d(op, np.array([2.0, 6.0, 20.0]))[3:], [4.0, 4.0]
    )


def test_anchor_reaches_inside_via_outside_vertices():
    b = mesh(
        [[0, 0], [0, 1], [-1, 0], [2.05, 0.3], [2.1, 0.3], [2.1, 0.35]],
        [[0, 1, 2], [3, 4, 5]],
        3.0,
    )
    t = mesh(
        np.vstack((b.xy, [[1, 0], [1, 1], [2, 0], [2, 1]])),
        [[0, 1, 2], [3, 4, 5], [0, 6, 7], [0, 7, 1], [6, 8, 9], [6, 9, 7]],
        3.0,
    )
    pair = b, t, np.arange(6), np.array([0, 1])
    op = build(pair)
    centroid = t.xy[t.triangles[4]].mean(axis=0)
    assert np.argmin(np.linalg.norm(b.xy - centroid, axis=1)) == 3  # across land
    assert (op.node_report["donor_distances_m"][t.triangles[4]] > 0).all()
    assert op.element_report["anchor_nodes"][4] == 0
    expected_distance = np.linalg.norm(centroid - t.xy[6]) + 1.0
    assert op.element_report["anchor_distances_m"][4] == pytest.approx(
        expected_distance
    )
    assert op.element_matrix[4, 0] == 1 and op.element_matrix[4, 1] == 0
    rt.check(op, np.array([4.0, 90.0]), location="element")
    # Disconnected new water has no possible inside donor, even with a huge cap.
    isolated = mesh(
        np.vstack((b.xy, [[100, 100], [101, 100], [100, 101]])),
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        3.0,
    )
    with pytest.raises(ValueError, match="disconnected"):
        rt.build_node_operator(b, isolated, np.arange(6), max_distance=1e9)


def test_separate_operator_limits():
    pair = outside_pair()
    # Centroid needs .3018 m; outside node needs .1414 m.
    nodes, report = rt.build_node_operator(*pair[:3], max_distance=0.2)
    assert nodes.shape == (7, 6) and report["outside"] == 1
    with pytest.raises(ValueError, match="anchor"):
        rt.build_element_operator(*pair, max_distance=0.2)
    b, t, nm, em = pair
    xy = t.xy.copy()
    xy[-1] = [2.0, -0.1]
    t = mesh(xy, t.triangles, t.depths)
    # Now the node is distant, but the centroid is just .0333 m from node 1.
    elements, report = rt.build_element_operator(b, t, nm, em, max_distance=0.2)
    assert elements.shape == (3, 2) and report["anchor"] == 1
    with pytest.raises(ValueError, match="node donor"):
        rt.build_node_operator(b, t, nm, max_distance=0.2)


def test_coincident_frozen_donor_uses_its_own_bank():
    # Vertices 1 and 3 coincide geometrically but have distinct water paths and
    # distinct base values. The frozen donor/anchor is vertex 3 on bank 2.
    b = mesh(
        [[0, 0], [1, 0], [0, 1], [1, 0], [2, 0], [2, 1]], [[0, 1, 2], [3, 4, 5]], 3.0
    )
    t = mesh(np.vstack((b.xy, [1.1, -0.1])), [[0, 1, 2], [3, 4, 5], [3, 6, 4]], 3.0)
    op = build((b, t, np.arange(6), np.array([0, 1])))
    assert op.node_report["donor_nodes"][6] == 3
    assert op.node_matrix[6, 3] == 1 and op.node_matrix[6, 1] == 0
    assert op.element_report["anchor_nodes"][2] == 3
    assert op.element_matrix[2, 1] == 1 and op.element_matrix[2, 0] == 0
    fields = np.array([0.0, 2.0, 4.0, 90.0, 100.0, 110.0])
    assert rt.apply_2d(op, fields)[6] == 90
    rt.check(op, fields)
    rt.check(op, np.array([4.0, 90.0]), location="element")
    zeta = np.zeros(len(b.xy))
    new_zeta = np.zeros(len(t.xy))
    result, _ = rt.apply_3d(
        op, np.repeat(fields[:, None], 30, axis=1), SIGMA, zeta, new_zeta
    )
    np.testing.assert_array_equal(result[6], 90.0)


def test_exact_clamping_extension_weights_and_deltas():
    # New target depth 12, old depth 4, sigma centres -.25,-.75:
    # target z=-3,-9; donor z=-1,-3. Only -9 extends, by 5 m.
    b = mesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]], 4.0)
    t = mesh(
        np.vstack((b.xy, [1 / 3, 1 / 3])),
        [[0, 1, 3], [1, 2, 3], [2, 0, 3]],
        [4, 4, 4, 12],
    )
    op = build((b, t, np.arange(3), np.array([-1])))
    sigma = np.array([0.0, -0.5, -1.0])
    bz, tz = np.zeros(3), np.zeros(4)
    for location, expected_volume, extension_fraction in (
        ("node", 2.0, 0.5),
        ("element", 10 / 3, 0.5),
    ):
        z = centres(b, sigma, bz, location)
        got, stats = rt.apply_3d(op, z, sigma, bz, tz, location=location)
        assert stats["target_volume_m3"] == pytest.approx(expected_volume)
        assert stats["extension"]["fraction"] == pytest.approx(extension_fraction)
        if location == "node":
            np.testing.assert_allclose(got[3], [-3, -3])
            assert stats["extension"]["max_distance_m"] == pytest.approx(5.0)
            assert stats["extension"]["weighted_volume_m3"] == pytest.approx(1.0)
            assert stats["extension"]["max_abs_delta"] == pytest.approx(6.0)
            assert stats["clamping"]["fraction"] == 0
        else:
            assert stats["extension"]["max_distance_m"] == pytest.approx(1.0)
            assert stats["extension"]["max_abs_delta"] == pytest.approx(2.0)
    # Shallower target: lower centre inside old column, beyond old last centre.
    t2 = mesh(t.xy, t.triangles, [4, 4, 4, 5])
    op2 = build((b, t2, np.arange(3), np.array([-1])))
    _, stats = rt.apply_3d(op2, centres(b, sigma, bz, "node"), sigma, bz, tz)
    assert stats["clamping"]["fraction"] == pytest.approx(0.5)
    assert stats["clamping"]["max_distance_m"] == pytest.approx(0.75)
    assert stats["clamping"]["max_abs_delta"] == pytest.approx(0.75)
    assert stats["extension"]["fraction"] == 0


@pytest.mark.parametrize("offset", [[0.0, 0.0], [400000.0, 3900000.0]])
def test_nearly_degenerate_quarter_weights(offset):
    p = (
        np.array([[0.0, 0.0], [100000018.0, 100000036.0], [100000005.0, 100000023.0]])
        / 65536
    )
    q = (p[1] + p[2]) / 4
    p, q = p + offset, q + offset
    b = mesh(p, [[0, 1, 2]], 4.0)
    t = mesh(np.vstack((p, q)), [[0, 1, 3], [1, 2, 3], [2, 0, 3]], 4.0)
    rm.validate_topology(b)
    rm.validate_topology(t)
    op = rt.build_transfer(b, t, np.arange(3))
    expected = np.array([0.5, 0.25, 0.25])
    np.testing.assert_array_equal(op.node_matrix.toarray()[3], expected)
    # Every basis field exposes a separate weight. In particular [0,0,1] must
    # yield .25 even if production and reference could otherwise agree wrongly.
    field = np.eye(3)
    np.testing.assert_array_equal(rt.apply_2d(op, field)[3], expected)
    reference = rt.independent_evaluate(b, t, op.node_map, op.elem_map, field)
    np.testing.assert_array_equal(reference[3], expected)
    assert rt.check(op, field)["max_abs_error"] == 0


@pytest.mark.parametrize("offset", [[0.0, 0.0], [400000.0, 3900000.0]])
def test_nearly_degenerate_exact_areas(offset):
    p = (
        np.array([[0.0, 0.0], [100000018.0, 100000036.0], [100000005.0, 100000023.0]])
        / 65536
    )
    q = (p[1] + p[2]) / 4
    t = mesh(np.vstack((p, q)) + offset, [[0, 1, 3], [1, 2, 3], [2, 0, 3]], 4.0)
    # These dyadic areas are exactly representable; an approximate comparison
    # would conceal the reviewer's [29.5,58.5,29.] cancellation error.
    np.testing.assert_array_equal(rt._areas(t) * 65536**2, [29.25, 58.5, 29.25])


def test_uncertifiable_triangle_refused():
    # Subtracting the local origin rounds the long edges, and the triangle is
    # thin, so the floating weights of a point INSIDE it (exact weights
    # [.5, .4516.., .0484..]) are wrong: a true container is refused, never
    # passed over. (An exterior point is passed over: round 11 #2.)
    b = mesh([[-1.0, -1.0], [1e16 + 2, 1e16 + 4], [1e16, 1e16 + 64]], [[0, 1, 2]], 4.0)
    q = np.array([5e15, 5e15 + 16])
    assert min(rt._exact_weights(b.xy[[0, 1, 2]], q)) > 0
    with pytest.raises(ValueError, match="weights cannot be established to 1e-12"):
        rt._locate(b, q[None])


def test_rational_evaluator_uses_no_production_geometry(monkeypatch):
    b, t, nm, em = synthetic_pair()
    expected = rt.independent_evaluate(b, t, nm, em, np.arange(2.0), location="element")

    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluator shared production geometry")

    for name in ("_locate", "_areas", "_cross", "_certify_weights"):
        monkeypatch.setattr(rt, name, forbidden)
    np.testing.assert_array_equal(
        rt.independent_evaluate(b, t, nm, em, np.arange(2.0), location="element"),
        expected,
    )


def deep_two_layer_case():
    b = mesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]], 4.0)
    t = mesh(
        np.vstack((b.xy, [1 / 3, 1 / 3])),
        [[0, 1, 3], [1, 2, 3], [2, 0, 3]],
        [4, 4, 4, 12],
    )
    return rt.build_transfer(b, t, np.arange(3)), np.array([0.0, -0.5, -1.0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_large_constant_ends(dtype):
    op, sigma = deep_two_layer_case()
    field = np.tile(np.array([1e16, 1.0], dtype=dtype), (3, 1))
    got, stats = rt.apply_3d(op, field, sigma, np.zeros(3), np.zeros(4))
    np.testing.assert_array_equal(got[3], [1.0, 1.0])
    assert got.dtype == dtype
    assert np.isfinite(stats["extension"]["max_abs_delta"])
    rt.check(
        op,
        field,
        sigma=sigma,
        base_zeta=np.zeros(3),
        target_zeta=np.zeros(4),
        actual=got,
    )
    # Both ends must assign directly, including the top endpoint when t=0.
    clipped, _ = rt._interpolate(
        np.array([-1.0, -3.0]), field[0].astype(float), np.array([1.0, -9.0])
    )
    np.testing.assert_array_equal(clipped, field[0])


def test_extreme_extension_refused():
    op, sigma = deep_two_layer_case()
    field = np.tile([1e308, -1e308], (3, 1))
    # Refused at the input now (round 4: field magnitude bound), before any
    # arithmetic could produce a non-finite value.
    with pytest.raises(ValueError, match="magnitude"):
        rt.apply_3d(op, field, sigma, np.zeros(3), np.zeros(4))


@pytest.mark.parametrize(
    "values",
    [
        [1e308, -1e308],
        [-1e308, 1e308],
        [1e308, 1.7e308],
        [-1.7e308, -1e308],
        [1e16, 1.0],
    ],
)
def test_extreme_interior_interpolation_bounded(values):
    targets = np.array([0.0, -0.125, -0.5, -0.875, -1.0])
    for field in (np.array(values), np.array([values, values]).T):
        got, linear = rt._interpolate(np.array([0.0, -1.0]), field, targets)
        assert np.isfinite(got).all() and np.isfinite(linear).all()
        assert (got >= min(values)).all() and (got <= max(values)).all()
        np.testing.assert_array_equal(got[[0, -1]], field)
        np.testing.assert_array_equal(got, linear)
        if values[0] == -values[1]:
            assert np.all(np.abs(got[2]) < 1e294)


def test_nonfinite_derived_statistics_refused():
    for events, denominator, delta in (
        ([(np.ones(2), np.full(2, 1e308))], 1.0, np.zeros(1)),
        ([], np.inf, np.zeros(1)),
        ([], 1.0, np.array([np.nan])),
        ([(np.ones(1), np.ones(1))], 1e-309, np.zeros(1)),
    ):
        with pytest.raises(ValueError, match="non-finite"):
            rt._statistic(events, denominator, delta)


@pytest.mark.parametrize("layered", [False, True])
@pytest.mark.parametrize(
    "fault",
    ["int", "complex", "str", "bool", "float16", "longdouble", "shape", "nan", "inf"],
)
def test_check_actual_contract(layered, fault):
    op = build(synthetic_pair())
    field = np.ones((4, 30)) if layered else np.ones(4)
    actual = np.ones((5, 30)) if layered else np.ones(5)
    if fault in ("int", "complex", "str", "bool", "float16", "longdouble"):
        actual = actual.astype(fault)
    if fault == "shape":
        actual = actual[:-1]
    if fault == "nan":
        actual.flat[0] = np.nan
    if fault == "inf":
        actual.flat[0] = np.inf
    kwargs = (
        dict(sigma=SIGMA, base_zeta=np.zeros(4), target_zeta=np.zeros(5))
        if layered
        else {}
    )
    with pytest.raises(ValueError, match="check actual"):
        rt.check(op, field, actual=actual, **kwargs)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_check_actual_float_contract_accepted(dtype):
    op = build(synthetic_pair())
    assert (
        rt.check(op, np.ones(4), actual=np.ones(5, dtype=dtype))["max_abs_error"]
        < 1e-14
    )


@pytest.mark.parametrize(
    "limit",
    [
        500 + 20j,
        np.complex128(500 + 20j),
        np.complex64(500),
        True,
        np.bool_(True),
        "500",
        b"500",
        [500.0],
        np.array(500.0),
        np.array([500.0]),
        None,
        -1.0,
        np.nan,
        np.inf,
        10**1000,
        np.longdouble("-1e-400"),
    ],
)
def test_distance_limit_scalar_contract(limit):
    b, t, nm, em = synthetic_pair()
    for builder, args in (
        (rt.build_transfer, (b, t, nm, em)),
        (rt.build_node_operator, (b, t, nm)),
        (rt.build_element_operator, (b, t, nm, em)),
        (rt.independent_evaluate, (b, t, nm, em, np.ones(4))),
    ):
        with pytest.raises(ValueError, match="max_distance"):
            builder(*args, max_distance=limit)


@pytest.mark.parametrize(
    "limit",
    [
        500,
        500.0,
        np.int64(500),
        np.uint64(500),
        np.float32(500),
        np.float64(500),
        np.longdouble(500),
    ],
)
def test_real_distance_scalars_accepted(limit):
    op = rt.build_transfer(*synthetic_pair(), max_distance=limit)
    assert type(op.max_distance) is float and op.max_distance == 500.0


def test_independent_every_row_layer_and_perturbations(pair):
    op = build(pair)
    base, target = pair[:2]
    bz = 0.6 + 0.2 * np.sin(np.arange(len(base.xy)))
    tz = rt.apply_2d(op, bz)
    rng = np.random.default_rng(129)
    for location, matrix in (("node", op.node_matrix), ("element", op.element_matrix)):
        # Random discontinuities in every layer/component make wrong stencils,
        # donor ordering and vertical interpolation observable everywhere.
        field = rng.normal(size=(matrix.shape[1], 30, 2))
        got, _ = rt.apply_3d(op, field, SIGMA, bz, tz, location=location)
        result = rt.check(
            op,
            field,
            location=location,
            sigma=SIGMA,
            base_zeta=bz,
            target_zeta=tz,
            actual=got,
            rtol=4e-13,
            atol=4e-13,
        )
        assert result["values_checked"] == matrix.shape[0] * 30 * 2
        print("INDEPENDENT_COMPARISON", location, result)
        rt.check(op, field[:, 0], location=location)
        # A constant perturbation must be visible at exactly that magnitude.
        shifted = rt.independent_evaluate(
            base,
            target,
            pair[2],
            pair[3],
            field + 0.125,
            location=location,
            sigma=SIGMA,
            base_zeta=bz,
            target_zeta=tz,
        )
        np.testing.assert_allclose(shifted - got, 0.125, atol=2e-13, rtol=0)
        with pytest.raises(ValueError, match="independent evaluator differs"):
            rt.check(
                op,
                field + 0.125,
                location=location,
                sigma=SIGMA,
                base_zeta=bz,
                target_zeta=tz,
                actual=got,
            )
    # For a z-linear field, a target-surface perturbation produces dz*(1+sigma)
    # where donors bracket; frozen rows still copy.
    field = centres(base, SIGMA, bz, "node")
    got, _ = rt.apply_3d(op, field, SIGMA, bz, tz)
    shift = 0.01
    changed = rt.independent_evaluate(
        base,
        target,
        pair[2],
        pair[3],
        field,
        sigma=SIGMA,
        base_zeta=bz,
        target_zeta=tz + shift,
    )
    mask = op.node_copy < 0
    # First interior mid-column level, away from all ends on the synthetic pair.
    i = np.flatnonzero(mask)[0]
    level = 10
    js = op.node_matrix.indices[op.node_matrix.indptr[i] : op.node_matrix.indptr[i + 1]]
    ztarget = centres(target, SIGMA, tz + shift, "node")[i, level]
    assert field[js, -1].max() < ztarget < field[js, 0].min()
    expected = shift * (1 + (SIGMA[level] + SIGMA[level + 1]) / 2)
    assert changed[i, level] - got[i, level] == pytest.approx(expected, abs=3e-13)
    with pytest.raises(ValueError, match="independent evaluator differs"):
        rt.check(
            op, field, sigma=SIGMA, base_zeta=bz, target_zeta=tz + shift, actual=got
        )


@pytest.mark.parametrize(
    "fault",
    [
        "xy_shape",
        "xy_dtype",
        "tri_dtype",
        "depth_shape",
        "nan",
        "node_dtype",
        "node_shape",
        "node_bounds",
        "node_duplicate",
        "node_inconsistent",
        "element_dtype",
        "element_shape",
        "element_inconsistent",
        "negative_limit",
        "nan_limit",
    ],
)
def test_malformed_pair_refused(fault):
    b, t, nm, em = synthetic_pair()
    b = SimpleNamespace(
        xy=b.xy.copy(), triangles=b.triangles.copy(), depths=b.depths.copy()
    )
    limit = 500.0
    if fault == "xy_shape":
        b.xy = b.xy[:, 0]
    if fault == "xy_dtype":
        b.xy = b.xy.astype(str)
    if fault == "tri_dtype":
        b.triangles = b.triangles.astype(float)
    if fault == "depth_shape":
        b.depths = b.depths[:, None]
    if fault == "nan":
        b.depths[0] = np.nan
    if fault == "node_dtype":
        nm = nm.astype(float)
    if fault == "node_shape":
        nm = nm[:, None]
    if fault == "node_bounds":
        nm[0] = 99
    if fault == "node_duplicate":
        nm[0] = nm[1]
    if fault == "node_inconsistent":
        nm[[0, 1]] = nm[[1, 0]]
    if fault == "element_dtype":
        em = em.astype(float)
    if fault == "element_shape":
        em = em[:, None]
    if fault == "element_inconsistent":
        em[1] = -1
    if fault == "negative_limit":
        limit = -1.0
    if fault == "nan_limit":
        limit = np.nan
    with pytest.raises(ValueError):
        rt.build_transfer(b, t, nm, em, max_distance=limit)


@pytest.mark.parametrize(
    "fault",
    [
        "field_shape",
        "field_dtype",
        "field_nan",
        "sigma_shape",
        "sigma_order",
        "sigma_ends",
        "layer_count",
        "zeta_shape",
        "zeta_nan",
        "dry",
        "location",
    ],
)
def test_malformed_application_refused(fault):
    pair = synthetic_pair()
    op = build(pair)
    field = np.ones((4, 30))
    sigma, bz, tz, loc = SIGMA.copy(), np.zeros(4), np.zeros(5), "node"
    if fault == "field_shape":
        field = field.T
    if fault == "field_dtype":
        field = field.astype(int)
    if fault == "field_nan":
        field[0, 0] = np.nan
    if fault == "sigma_shape":
        sigma = sigma[:, None]
    if fault == "sigma_order":
        sigma[10] = sigma[9]
    if fault == "sigma_ends":
        sigma[-1] = -0.99
    if fault == "layer_count":
        field = field[:, :-1]
    if fault == "zeta_shape":
        bz = bz[:, None]
    if fault == "zeta_nan":
        tz[0] = np.nan
    if fault == "dry":
        bz[0] = -100.0
    if fault == "location":
        loc = "edge"
    with pytest.raises(ValueError):
        rt.apply_3d(op, field, sigma, bz, tz, location=loc)


def test_matrix_asserts_invalid_weights():
    for row in ({0: -0.1, 1: 1.1}, {0: 0.9}, {0: np.nan}):
        with pytest.raises(AssertionError):
            rt._matrix([row], 2)


def test_upper_extension_and_donor_weight_accounting():
    # Unequal h and zeta, with a target whose top is above every old surface.
    # Independently count donor/layer events and volumes without operator helpers.
    pair = synthetic_pair()
    op = build(pair)
    bz = np.array([0.1, 0.3, -0.2, 0.4])
    tz = rt.apply_2d(op, bz)
    tz[-1] = 4.0
    for location, matrix, copies in (
        ("node", op.node_matrix, op.node_copy),
        ("element", op.element_matrix, op.element_copy),
    ):
        target, base = pair[1], pair[0]
        field = centres(base, SIGMA, bz, location)
        _, stats = rt.apply_3d(op, field, SIGMA, bz, tz, location=location)
        triangle_areas = []
        for tri in target.triangles:
            a, b, c = target.xy[tri]
            triangle_areas.append(abs(np.linalg.det(np.array([b - a, c - a]))) / 2)
        if location == "node":
            areas = np.zeros(len(target.xy))
            for tri, area in zip(target.triangles, triangle_areas):
                for v in tri:
                    areas[v] += area / 3
            th, surf, dh, dsurf = target.depths, tz, base.depths, bz
        else:
            areas = np.array(triangle_areas)
            th = np.array([np.mean(target.depths[t]) for t in target.triangles])
            surf = np.array([np.mean(tz[t]) for t in target.triangles])
            dh = np.array([np.mean(base.depths[t]) for t in base.triangles])
            dsurf = np.array([np.mean(bz[t]) for t in base.triangles])
        counts = {"clamping": 0, "extension": 0}
        volumes = {"clamping": 0.0, "extension": 0.0}
        deltas = {
            "clamping": np.zeros((len(areas), 30)),
            "extension": np.zeros((len(areas), 30)),
        }
        total = 0.0
        upper_events = 0
        for i in np.flatnonzero(copies < 0):
            for l in range(30):
                s = (SIGMA[l] + SIGMA[l + 1]) / 2
                z = surf[i] + s * (th[i] + surf[i])
                vol = areas[i] * (th[i] + surf[i]) * (SIGMA[l] - SIGMA[l + 1])
                total += vol
                for j, w in enumerate(matrix.toarray()[i]):
                    if not w:
                        continue
                    upper = dsurf[j] + (SIGMA[0] + SIGMA[1]) / 2 * (dh[j] + dsurf[j])
                    lower = dsurf[j] + (SIGMA[-2] + SIGMA[-1]) / 2 * (dh[j] + dsurf[j])
                    category = None
                    if z > dsurf[j] or z < -dh[j]:
                        category = "extension"
                        upper_events += z > dsurf[j]
                    elif z > upper or z < lower:
                        category = "clamping"
                    if category:
                        counts[category] += 1
                        volumes[category] += w * vol
                        deltas[category][i, l] += w * (z - np.clip(z, lower, upper))
        assert upper_events > 0
        assert stats["target_volume_m3"] == pytest.approx(total)
        for key in counts:
            assert stats[key]["donor_layer_events"] == counts[key]
            assert stats[key]["weighted_volume_m3"] == pytest.approx(volumes[key])
            assert stats[key]["fraction"] == pytest.approx(volumes[key] / total)
            np.testing.assert_allclose(
                stats[key]["linear_extrapolation_delta"], deltas[key], atol=3e-14
            )


# Each entry runs the NAMED, unchanged analytic test against a fresh in-memory
# implementation with one small fault. Nothing is rewritten on disk. These are
# specific regression checks, not a mutation framework or a source transformer.
MUTATIONS = [
    (
        "copy-bytes",
        "result[keep] = field[copies[keep]]",
        "result[keep] = field[copies[keep]] + 1",
        "test_g1_identity_and_permutation",
        True,
    ),
    (
        "refined-copy",
        "result[keep] = field[copies[keep]]",
        "result[keep] = field[copies[keep]] + 1",
        "test_g1_frozen_and_matched",
        True,
    ),
    (
        "node-barycentrics",
        "weights[i] = clipped",
        "weights[i] = np.array([1., 0., 0.])",
        "test_g2_constants_linear_nodes_and_weights",
        True,
    ),
    (
        "element-area-share",
        "w * area[e] / vertex_area[v]",
        "w / len(incident[v])",
        "test_g2_element_accuracy_and_area_share",
        True,
    ),
    (
        "vertical-donor-depth",
        "zt, zt_error = columns.relative(i, j)",
        "zt, zt_error = columns.relative(i, j)\n            zt = zb[j].copy()",
        "test_g2_vertical_physical_z_and_saturation",
        True,
    ),
    (
        "vertical-zeta",
        "offset = self.target_zeta[i] - self.base_zeta[j]",
        "offset = self.target_zeta[i]",
        "test_g2_vertical_physical_z_and_saturation",
        True,
    ),
    (
        "end-saturation",
        "return clipped, linear",
        "return linear, linear",
        "test_g2_vertical_physical_z_and_saturation",
        True,
    ),
    (
        "bounded-discontinuity",
        "return clipped, linear",
        "return linear, linear",
        "test_g2_bounded_discontinuities",
        True,
    ),
    (
        "across-land",
        "donor, distance = _paths(target, triangle >= 0)",
        "donor, distance = _paths(target, triangle >= 0)\n    donor[triangle < 0] = [int(np.argmin(np.linalg.norm(target.xy - p, axis=1) + np.where(triangle >= 0, 0., np.inf))) for p in target.xy[triangle < 0]]",
        "test_outside_path_anchor_limit_and_ties",
        False,
    ),
    (
        "anchor-tie",
        "ad[i], anchors[i] = min(choices)",
        "ad[i], anchors[i] = min(choices, key=lambda item: (item[0], -item[1]))",
        "test_outside_path_anchor_limit_and_ties",
        False,
    ),
    (
        "node-tie",
        "candidate < (distance[b], donor[b])",
        "(candidate[0] < distance[b] or (candidate[0] == distance[b] and source > donor[b]))",
        "test_outside_path_anchor_limit_and_ties",
        False,
    ),
    (
        "anchor-across-land",
        "ad[i], anchors[i] = min(choices)",
        "ad[i], anchors[i] = min(choices)\n        anchors[i] = int(np.argmin(np.linalg.norm(target.xy - centroids[i], axis=1) + np.where(triangle >= 0, 0., np.inf)))",
        "test_anchor_reaches_inside_via_outside_vertices",
        False,
    ),
    (
        "distance-limit",
        "ad[i] > max_distance",
        "False",
        "test_outside_path_anchor_limit_and_ties",
        False,
    ),
    (
        "new-inside-donor",
        "donor, distance = _paths(target, triangle >= 0)",
        "donor, distance = _paths(target, nc >= 0)",
        "test_outside_donor_can_be_new_inside_node",
        False,
    ),
    (
        "extension-volume",
        "w * volume[i, mask]",
        "volume[i, mask]",
        "test_exact_clamping_extension_weights_and_deltas",
        False,
    ),
    (
        "extension-category",
        "beyond_centres & ~extended",
        "beyond_centres",
        "test_exact_clamping_extension_weights_and_deltas",
        False,
    ),
    (
        "extension-delta",
        "w * (linear[mask] - values[mask])",
        "0. * (linear[mask] - values[mask])",
        "test_exact_clamping_extension_weights_and_deltas",
        False,
    ),
    (
        "upper-extension",
        "extended = (zt > 0) | (zt < -depth[j])",
        "extended = (zt < -depth[j])",
        "test_upper_extension_and_donor_weight_accounting",
        False,
    ),
    (
        "clamping-volume",
        "w * volume[i, mask]",
        "volume[i, mask]",
        "test_upper_extension_and_donor_weight_accounting",
        False,
    ),
    (
        "independent-vertical",
        "field[d, bracket].astype(\n                np.longdouble\n            )",
        "(field[d, bracket].astype(\n                np.longdouble\n            ) + .01)",
        "test_independent_every_row_layer_and_perturbations",
        True,
    ),
    (
        "independent-field-sensitivity",
        "value += weight * field[d]",
        "value += weight * (field[d] + .01)",
        "test_independent_every_row_layer_and_perturbations",
        True,
    ),
    (
        "check-perturbation-sensitivity",
        "if (difference > allowance).any():",
        "if False and (difference > allowance).any():",
        "test_independent_every_row_layer_and_perturbations",
        True,
    ),
    (
        "check-allowance-per-value",
        "relative * scale + floor",
        "relative * scale.max(initial=0) + floor",
        "test_r6_large_component_does_not_hide_another",
        False,
    ),
    (
        "coincident-anchor-bank",
        "triangle[nm[v]], bary[nm[v]] = e, np.eye(3)[k]",
        "pass",
        "test_coincident_frozen_donor_uses_its_own_bank",
        False,
    ),
    (
        "thin-column-relative-frame",
        "base_centres = centres * base_depth[:, None]",
        "base_centres = (bz[:, None] + centres * base_depth[:, None]) - bz[:, None]",
        "test_r11_thin_column_interpolates_in_donor_relative_frame",
        False,
    ),
    (
        "fraction-certificate",
        "if not (bound <= FRACTION_TOLERANCE).all():",
        "if False:",
        "test_r11_unresolved_fraction_refused",
        False,
    ),
    (
        "exact-containment",
        "if min(exact) < -BOUNDARY_TOLERANCE:\n                continue",
        "pass",
        "test_r11_thin_candidate_does_not_abort_search",
        False,
    ),
    (
        "independent-donor-surface",
        "below_surface = [-z for z in old_surface[d]]",
        "below_surface = []",
        "test_r11_thin_column_interpolates_in_donor_relative_frame",
        False,
    ),
    (
        "weight-sum-assertion",
        'assert np.allclose(\n        np.asarray(matrix.sum(axis=1)).ravel(), 1.0, rtol=0, atol=2e-14\n    ), "Row sum differs from one"',
        "pass",
        "test_matrix_asserts_invalid_weights",
        False,
    ),
]


@pytest.mark.parametrize(
    "name,old,new,test_name,needs_pair", MUTATIONS, ids=[m[0] for m in MUTATIONS]
)
def test_named_gate_detects_mutation(
    monkeypatch, name, old, new, test_name, needs_pair
):
    reference = name.startswith("independent-")
    source = REFERENCE_SOURCE if reference else SOURCE
    assert old in source, f"Missing mutation site: {name}"
    owner = transfer_reference if reference else rt
    mutated = ModuleType(owner.__package__ + ".transfer_mutant")
    mutated.__file__ = owner.__file__
    mutated.__package__ = owner.__package__
    monkeypatch.setitem(sys.modules, mutated.__name__, mutated)
    exec(compile(source.replace(old, new), owner.__file__, "exec"), mutated.__dict__)
    if reference:
        monkeypatch.setattr(rt, "independent_evaluate", mutated.independent_evaluate)
    else:
        monkeypatch.setitem(globals(), "rt", mutated)
    with pytest.raises((AssertionError, ValueError, pytest.fail.Exception)):
        globals()[test_name](*([synthetic_pair()] if needs_pair else []))


def test_r2_subtraction_sensitive_area_refused():
    """A triangle whose area the rounded coordinate differences cannot give is refused."""
    p = (
        np.array(
            [[-1.0, -1.0], [1e16 + 2, 1e16 + 4], [1e16 + 2, 1e16 + 2], [1e16, 1e16 + 2]]
        )
        * 2.0**-54
    )
    b = mesh(p[:3], [[2, 1, 0]], 4.0)
    t = mesh(p, [[2, 1, 0], [0, 1, 3]], [4, 4, 4, 5])
    # refused by the physical range (round 7: edges of 1e-16 m) before the
    # area certification, which still guards thin triangles within the range
    with pytest.raises(ValueError, match="physical range|area cannot be established"):
        op = rt.build_transfer(b, t, np.arange(3))
        rt.apply_3d(
            op,
            np.tile([1.0, 2.0], (3, 1)),
            np.array([0.0, -0.5, -1.0]),
            np.zeros(3),
            np.zeros(4),
        )


def test_r2_areas_match_exact_values_far_from_origin():
    offset = np.array([4.0e5, 3.9e6])
    xy = np.array([[0.0, 0.0], [3.0, 0.0], [1.0, 2.0]]) + offset
    area = rt._areas(mesh(xy, [[0, 1, 2]], 4.0))
    assert area[0] == 3.0


@pytest.mark.parametrize("dtype", [np.uint8, np.uint64, np.int16])
def test_r2_integer_depths_give_float_statistics(dtype):
    """Integer or unsigned depths must not wrap: identical columns extend nothing."""
    base, target, nm, _ = synthetic_pair()
    base = rm.Mesh(base.xy, base.triangles, np.full(len(base.xy), 4, dtype=dtype))
    target = rm.Mesh(
        target.xy, target.triangles, np.full(len(target.xy), 4, dtype=dtype)
    )
    op = rt.build_transfer(base, target, nm)
    _, stats = rt.apply_3d(op, np.ones((4, 30)), SIGMA, np.zeros(4), np.zeros(5))
    for kind in ("extension", "clamping"):
        assert stats[kind]["fraction"] == 0, (kind, stats[kind]["fraction"])
        assert stats[kind]["max_distance_m"] == 0, (kind, stats[kind]["max_distance_m"])
    assert stats["target_volume_m3"] > 0


def test_r3_float32_elevations_evaluator_float64():
    """The evaluator computes in float64 whatever the elevation dtype (round-3 #1)."""
    b, t, nm, em = synthetic_pair()
    op = rt.build_transfer(b, t, nm, em)
    field = np.arange(120.0, dtype=float).reshape(4, 30)
    bz = np.array([0.1, 0.3, -0.2, 0.4], dtype=np.float32)
    tz = np.full(5, 0.21, dtype=np.float32)
    rt.check(op, field, sigma=SIGMA, base_zeta=bz, target_zeta=tz)
    rt.check(
        op,
        np.arange(60.0, dtype=float).reshape(2, 30),
        location="element",
        sigma=SIGMA,
        base_zeta=bz,
        target_zeta=tz,
    )
    rt.check(op, field, sigma=SIGMA.astype(np.float32), base_zeta=bz, target_zeta=tz)


def test_r3_integer_elevations_do_not_overflow():
    b, t, nm, em = synthetic_pair()
    op = rt.build_transfer(b, t, nm, em)
    bz, tz = np.full(4, 100, dtype=np.uint8), np.full(5, 100, dtype=np.uint8)
    rt.check(
        op,
        np.arange(60.0, dtype=float).reshape(2, 30),
        location="element",
        sigma=SIGMA,
        base_zeta=bz,
        target_zeta=tz,
    )


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.longdouble])
def test_r3_non_float64_coordinates_refused(dtype):
    b, t, nm, _ = synthetic_pair()
    b = rm.Mesh(b.xy.astype(dtype), b.triangles, b.depths)
    t = rm.Mesh(t.xy.astype(dtype), t.triangles, t.depths)
    with pytest.raises(ValueError, match="float64 coordinates"):
        rt.build_transfer(b, t, nm)


def _strip(L, H):
    base = mesh([[0, 0], [2 * L, 0], [0, H]], [[0, 1, 2]], 3.0)
    target = mesh([[0, 0], [2 * L, 0], [0, H], [3 * L, 0]], [[0, 1, 2], [1, 3, 2]], 3.0)
    return base, target


def test_r3_extreme_scale_geometry_refused():
    """Rounds 3-4 found squared lengths underflowing (1e-200 m) and overflowing (1e155 m);
    since round 7 such geometry is outside the physical range and refused before any
    distance is computed. hypot is still used, so in-range lengths never square-overflow.
    """
    for L, H in ((1e-200, 1e-100), (1e155, 1e-155)):
        base, target = _strip(L, H)
        with pytest.raises(ValueError, match="physical range"):
            rt.build_node_operator(base, target, np.arange(3), max_distance=4e155)
        with pytest.raises(ValueError, match="physical range"):
            rt.build_transfer(base, target, np.arange(3), max_distance=4e155)


def _split(point, depth=4.0):
    base = mesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]], depth)
    target = mesh(
        [[0, 0], [1, 0], [0, 1], point], [[0, 1, 3], [1, 2, 3], [2, 0, 3]], depth
    )
    return rt.build_transfer(base, target, np.arange(3))


def test_r4_float32_field_cancellation():
    """The evaluator multiplies float32 fields in float64 (2-D and 3-D)."""
    op = _split([0.2, 0.7])
    field = np.array([33625.0859375, -35361.7265625, 5299.7666015625], dtype=np.float32)
    rt.check(op, field)
    rt.check(
        op,
        np.repeat(field[:, None], 30, axis=1),
        sigma=SIGMA,
        base_zeta=np.zeros(3),
        target_zeta=np.zeros(4),
    )


def test_r4_collapsed_layer_centres_refused():
    op = _split([0.25, 0.25])
    sigma = np.array([0.0, -(2.0**-60), -(2.0**-59), -1.0])
    with pytest.raises(ValueError, match="strictly decreasing|physical range"):
        rt.apply_3d(op, np.tile([0.0, 1.0, 2.0], (3, 1)), sigma, np.ones(3), np.ones(4))


@pytest.mark.parametrize("value", [np.finfo(np.float64).max, 1e101, -1e101])
def test_r4_field_magnitude_bound(value):
    op = rt.build_transfer(*synthetic_pair())
    with pytest.raises(ValueError, match="magnitude"):
        rt.apply_2d(op, np.full(2, value), location="element")


@pytest.mark.parametrize("value", [1e100, -1e100])
def test_r4_largest_allowed_constant_is_preserved(value):
    op = rt.build_transfer(*synthetic_pair())
    out = rt.apply_2d(op, np.full(2, value), location="element")
    assert np.allclose(out, value, rtol=4e-16, atol=0)
    out3, _ = rt.apply_3d(
        op, np.full((2, 30), value), SIGMA, np.zeros(4), np.zeros(5), location="element"
    )
    assert np.allclose(out3, value, rtol=4e-16, atol=0)


def test_r5_float32_magnitude_check_strict_overflow():
    op = rt.build_transfer(*synthetic_pair())
    with np.errstate(over="raise"):
        rt.apply_2d(op, np.ones(4, dtype=np.float32))
        rt.apply_3d(
            op, np.ones((4, 30), dtype=np.float32), SIGMA, np.zeros(4), np.zeros(5)
        )
        rt.check(op, np.ones(4, dtype=np.float32))


def test_r5_check_allows_cancellation_round_off():
    op = _split([0.2, 0.7])
    field = np.array([-7000.0, -7000.0, 3000.0])
    rt.check(op, field)
    rt.check(
        op,
        np.repeat(field[:, None], 30, axis=1),
        sigma=SIGMA,
        base_zeta=np.zeros(3),
        target_zeta=np.zeros(4),
    )
    # a real difference of the field's own size must still be refused
    good = rt.apply_2d(op, field)
    bad = good.copy()
    bad[3] += 1e-9 * 7000
    with pytest.raises(ValueError, match="differs"):
        rt.check(op, field, actual=bad)


def test_r6_float32_subnormal_rounding_accepted():
    op = _split([0.25, 0.25])
    tiny = np.nextafter(np.float32(0), np.float32(1))
    field = np.array([0, 0, tiny], dtype=np.float32)
    rt.check(op, field)
    rt.check(
        op,
        np.repeat(field[:, None], 30, axis=1),
        sigma=SIGMA,
        base_zeta=np.zeros(3),
        target_zeta=np.zeros(4),
    )


def test_r6_large_component_does_not_hide_another():
    op = rt.build_transfer(*synthetic_pair())
    field = np.column_stack((np.full(4, 1e20), np.arange(4.0)))
    actual = rt.apply_2d(op, field)
    actual[-1, 1] += 1
    with pytest.raises(ValueError, match="differs"):
        rt.check(op, field, actual=actual)
    rt.check(op, field)


def test_r6_non_dyadic_thin_triangle_self_check():
    p = np.array([[-0.1, -0.2], [100.1, 100.3], [50.0, 50.08]])
    q = p.T @ np.array([0.1, 0.3, 0.6])
    b = mesh(p, [[0, 1, 2]], 4.0)
    t = mesh(np.vstack((p, q)), [[0, 1, 3], [1, 2, 3], [2, 0, 3]], 4.0)
    op = rt.build_transfer(b, t, np.arange(3))
    rt.check(op, np.array([0.0, 1.0, 0.0]))


def test_r6_horizontal_report_overflow_refused():
    b, t, nm, em = outside_pair()
    b = rm.Mesh(b.xy * 100, b.triangles, np.full(len(b.xy), 1e308))
    t = rm.Mesh(t.xy * 100, t.triangles, np.full(len(t.xy), 1e308))
    with (
        np.errstate(over="raise"),
        pytest.raises(ValueError, match="physical range|non-finite"),
    ):
        rt.build_transfer(b, t, nm, em)


def test_r7_edge_midpoint_zero_weight_donor_in_allowance():
    p = np.array([[-0.1, -0.2], [1.0, 0.0], [0.0, 1.0]])
    b = mesh(p, [[0, 1, 2]], 4.0)
    t = mesh(np.vstack((p, [0.5, 0.5])), [[0, 1, 3], [0, 3, 2]], 4.0)
    op = rt.build_transfer(b, t, np.arange(3))
    field = np.array([1.0, 0.0, 0.0])
    rt.check(op, field)
    rt.check(
        op,
        np.repeat(field[:, None], 2, axis=1),
        sigma=np.array([0.0, -0.5, -1.0]),
        base_zeta=np.zeros(3),
        target_zeta=np.zeros(4),
    )


@pytest.mark.parametrize("key", ["atol", "rtol"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0, True, "1e-9"])
def test_r7_check_tolerances_validated(key, value):
    op = _split([0.25, 0.25])
    field = np.array([1.0, 2.0, 3.0])
    actual = rt.apply_2d(op, field)
    actual[-1] += 100.0
    with pytest.raises(ValueError, match=key):
        rt.check(op, field, actual=actual, **{key: value})


@pytest.mark.parametrize(
    "case",
    ["tiny-coordinates", "huge-coordinates", "tiny-depth", "huge-depth", "short-edge"],
)
def test_r7_outside_physical_range_refused(case):
    b, t, nm, em = synthetic_pair()
    xy_b, xy_t, h_b, h_t = b.xy, t.xy, b.depths, t.depths
    if case == "tiny-coordinates":  # every edge below MIN_EDGE_M
        xy_b, xy_t = xy_b * 1e-50, xy_t * 1e-50
    elif case == "huge-coordinates":
        xy_b, xy_t = xy_b * 1e155, xy_t * 1e155
    elif case == "tiny-depth":
        h_b, h_t = np.full_like(h_b, 4e-300), np.full_like(h_t, 4e-300)
    elif case == "huge-depth":
        h_b, h_t = np.full_like(h_b, 1e308), np.full_like(h_t, 1e308)
    else:
        xy_b, xy_t = xy_b * 1e-7, xy_t * 1e-7
    b, t = rm.Mesh(xy_b, b.triangles, h_b), rm.Mesh(xy_t, t.triangles, h_t)
    with pytest.raises(ValueError):
        rt.build_transfer(b, t, nm, em)


def test_r7_elevation_outside_physical_range_refused():
    op = rt.build_transfer(*synthetic_pair())
    with pytest.raises(ValueError, match="physical range"):
        rt.apply_3d(op, np.ones((4, 30)), SIGMA, np.full(4, 2e3), np.zeros(5))


def test_extension_warning_names_the_depth_difference():
    """A large difference from the base depths is reported, never refused (owner, 2026-11-15)."""
    op, sigma = deep_two_layer_case()
    with pytest.warns(rt.ExtensionWarning, match=r"% of the new volume.*max = \d"):
        _, stats = rt.apply_3d(
            op, np.tile([1.0, 2.0], (3, 1)), sigma, np.zeros(3), np.zeros(4)
        )
    assert stats["extension"]["max_distance_m"] > rt.EXTENSION_WARNING_M


def test_no_extension_warning_for_equal_depths():
    import warnings as _warnings

    b, t, nm, _ = synthetic_pair()
    b = rm.Mesh(b.xy, b.triangles, np.full(len(b.xy), 4.0))
    t = rm.Mesh(t.xy, t.triangles, np.full(len(t.xy), 4.0))
    op = rt.build_transfer(b, t, nm)
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", rt.ExtensionWarning)
        rt.apply_3d(op, np.ones((4, 30)), SIGMA, np.zeros(4), np.zeros(5))


def test_r8_small_area_and_thin_layer_refused():
    s = 3 * 2.0**-1068
    b = mesh([[0, 0], [12, 0], [6, s]], [[0, 1, 2]], 0.001)
    t = mesh(
        np.vstack((b.xy, [3, s / 4])),
        [[0, 1, 3], [1, 2, 3], [2, 0, 3]],
        [0.001, 0.001, 0.001, 0.003],
    )
    with pytest.raises(ValueError, match="physical range"):
        rt.build_transfer(b, t, np.arange(3))
    op = rt.build_transfer(*synthetic_pair())
    with pytest.raises(ValueError, match="physical range"):
        rt.apply_3d(
            op, np.ones((4, 2)), np.array([0.0, -1e-7, -1.0]), np.zeros(4), np.zeros(5)
        )


def test_r8_boundary_candidate_not_certifiable_search_continues():
    L = 2000.0
    b = mesh([[0, 0], [L, 0], [0, L], [L, L]], [[0, 1, 2], [1, 3, 2]], 4.0)
    t = mesh(
        np.vstack((b.xy, [L / 2, L / 2 + 3e-9])),
        [[0, 1, 2], [1, 3, 4], [3, 2, 4], [2, 1, 4]],
        4.0,
    )
    op = rt.build_transfer(b, t, np.arange(4))
    rt.check(op, np.array([0.0, 1.0, 2.0, 3.0]))


@pytest.mark.parametrize("key", ["atol", "rtol"])
@pytest.mark.parametrize("value", [np.longdouble("-1e-400"), 10**1000])
def test_r8_check_tolerance_sign_and_overflow(key, value):
    op = _split([0.25, 0.25])
    with pytest.raises(ValueError, match=key):
        rt.check(op, np.ones(3), **{key: value})


def test_r9_reference_exterior_boundary_point():
    """The evaluator applies the same clipped-weight criterion as production."""
    b = mesh([[0, 0], [200, 0], [0, 200]], [[0, 1, 2]], 4.0)
    t = mesh(np.vstack((b.xy, [100, 100 + 3e-10])), [[0, 1, 2], [1, 3, 2]], 4.0)
    op = rt.build_transfer(b, t, np.arange(3))
    rt.check(op, np.array([0.0, 1.0, 2.0]))


def test_r9_physical_range_refusals_under_strict_errors():
    with np.errstate(over="raise", invalid="raise"):
        xy = [[0, 0], [1e200, 0], [0, 1e200]]
        b = mesh(xy, [[0, 1, 2]], 4.0)
        with pytest.raises(ValueError, match="physical range"):
            rt.build_transfer(b, b, np.arange(3))
        op = rt.build_transfer(*synthetic_pair())
        zeta = np.full(4, np.longdouble("1e400"))
        with pytest.raises(ValueError, match="physical range"):
            rt.apply_3d(op, np.ones((4, 30)), SIGMA, zeta, np.zeros(5))


@pytest.mark.parametrize("location", ["node", "element"])
@pytest.mark.parametrize("which", ["base", "target"])
def test_r10_minimum_signed_integer_elevation_refused(location, which):
    op = rt.build_transfer(*synthetic_pair())
    bz, tz = np.zeros(4), np.zeros(5)
    if which == "base":
        bz = np.full(4, -32768, dtype=np.int16)
    else:
        tz = np.full(5, -32768, dtype=np.int16)
    n = 4 if location == "node" else 2
    with pytest.raises(ValueError, match="physical range"):
        rt.apply_3d(op, np.ones((n, 30)), SIGMA, bz, tz, location=location)


def test_r10_wide_float_depths_and_sigma_refused_under_strict_errors():
    b, t, nm, _ = synthetic_pair()
    big = np.longdouble("1e400")
    with np.errstate(over="raise", invalid="raise"):
        deep = rm.Mesh(b.xy, b.triangles, np.full(len(b.xy), big))
        with pytest.raises(ValueError):
            rt.build_transfer(deep, t, nm)
        op = rt.build_transfer(b, t, nm)
        for sigma in (np.array([0, big, -1]), np.array([0.0, -1e308, 1e308, -1.0])):
            with pytest.raises(ValueError, match="sigma"):
                rt.apply_3d(
                    op, np.ones((4, len(sigma) - 1)), sigma, np.zeros(4), np.zeros(5)
                )


def _thin_column_case():
    b = mesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]], 1.0)
    t = mesh(np.vstack((b.xy, [0.25, 0.25])), [[0, 1, 3], [1, 2, 3], [2, 0, 3]], 1.0)
    return rt.build_transfer(b, t, np.arange(3))


def test_r11_thin_column_interpolates_in_donor_relative_frame():
    """Nearly dry columns: the fraction is exactly 0.3, not 1/3 (round 11 #1).

    Relative to -1 the donor centres are 3.75u and 1.25u and the target's upper
    centre 3u, so the fraction is (3.75 - 3) / (3.75 - 1.25) = 0.3; rounding
    the absolute centres first gave 1/3 in production and evaluator alike.
    """
    op = _thin_column_case()
    sigma = np.array([0.0, -0.5, -1.0])
    field = np.tile([0.0, 1.0], (3, 1))
    u = 2.0**-53
    bz = np.full(3, -1 + 5 * u)
    tz = np.r_[bz, -1 + 4 * u]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rt.ExtensionWarning)
        actual, _ = rt.apply_3d(op, field, sigma, bz, tz)
        reference = rt.independent_evaluate(
            op.base,
            op.target,
            op.node_map,
            op.elem_map,
            field,
            sigma=sigma,
            base_zeta=bz,
            target_zeta=tz,
        )
    assert actual[-1, 0] == pytest.approx(0.3, abs=1e-15)
    assert reference[-1, 0] == pytest.approx(0.3, abs=1e-15)
    assert actual[-1, 1] == 1.0
    rt.check(op, field, sigma=sigma, base_zeta=bz, target_zeta=tz, actual=actual)


def test_r11_unresolved_fraction_refused():
    """A fraction the float64 frame cannot resolve to 1e-12 is refused.

    The target surface stands 500 m above a 1 m donor column, so its centre in
    that column is a difference of ~800 m terms resolved against 1/30 m layers.
    """
    op = _thin_column_case()
    target = rm.Mesh(op.target.xy, op.target.triangles, np.r_[np.ones(3), 930.0])
    op = rt.build_transfer(op.base, target, np.arange(3))
    sigma = np.linspace(0.0, -1.0, 31)
    with pytest.raises(ValueError, match="cannot be established"):
        rt.apply_3d(op, np.ones((3, 30)), sigma, np.zeros(3), np.r_[np.zeros(3), 500.0])


def test_r11_thin_candidate_does_not_abort_search():
    """A thin triangle's bounding box covering another triangle's points (round 11 #2)."""
    p = np.array([[0.0, 0.0], [100000018.0, 100000036.0], [100000005.0, 100000023.0]])
    m = mesh(
        np.vstack((p / 65536, [[100, 300], [110, 300], [100, 310]])),
        [[0, 1, 2], [3, 4, 5]],
        4.0,
    )
    rm.validate_topology(m)
    op = rt.build_transfer(m, m, np.arange(6))
    assert (op.node_copy == np.arange(6)).all()
    rt.check(op, np.arange(6.0))


def test_r11_float16_depths_strict_errors():
    """The depth range check promotes float16 instead of narrowing its bound."""
    b = mesh([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]], 4.0)
    b16 = rm.Mesh(b.xy, b.triangles, np.full(3, 4.0, dtype=np.float16))
    with np.errstate(over="raise", invalid="raise"):
        rt.build_transfer(b16, b16, np.arange(3))
        with pytest.raises(ValueError, match="physical range"):
            deep = rm.Mesh(b.xy, b.triangles, np.full(3, 1e-4, dtype=np.float16))
            rt.build_transfer(deep, deep, np.arange(3))

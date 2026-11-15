"""Synthetic serialized mesh correspondence, topology and reader contracts."""

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from shapely.geometry import shape

from xfvcom.regrid import mesh_pair as rm

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
TRI = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
FAN = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int64)


def mesh(xy=SQUARE, tri=TRI, h=None):
    return rm.Mesh(
        np.array(xy), np.array(tri), np.full(len(xy), 3.0) if h is None else np.array(h)
    )


def refined_square(h=3.0):
    return mesh(np.vstack((SQUARE, [0.5, 0.5])), FAN, [3.0, 3.0, 3.0, 3.0, h])


def finish_report(n_new=1):
    return {
        "hmin_m": 3.0,
        "hmax_m": 300.0,
        "rfactor": 0.2,
        "converged_at_write_precision": True,
        "rfactor_report": {"converged": True},
        "write_precision_tolerance": 0.2 * 1e-6 + 1e-9,
        "variant": "min3m_rfac0p2_cap300",
        "scope": "patch nodes only",
        "n_movable": n_new,
    }


def g0(base, refined, nm, report=None, obc=(), spg=()):
    return rm.validate_g0(
        base,
        refined,
        np.array(nm),
        report or finish_report(),
        np.array(obc, dtype=np.int64),
        np.array(spg, dtype=np.int64),
    )


def write_case(directory, item, prefix="tiny", obc=(0,), types=(2,), spg=(0,)):
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        kind: directory / f"{prefix}_{kind}.dat"
        for kind in ("grd", "dep", "cor", "obc", "spg")
    }
    with paths["grd"].open("w") as stream:
        stream.write(
            f"Node Number = {len(item.xy)}\nCell Number = {len(item.triangles)}\n"
        )
        for i, row in enumerate(item.triangles + 1, 1):
            stream.write(f"{i} {row[0]} {row[1]} {row[2]}\n")
        for i, (x, y) in enumerate(item.xy, 1):
            stream.write(f"{i} {float(x)!r} {float(y)!r}\n")
    for kind, values in (("dep", item.depths), ("cor", np.full(len(item.xy), 35.0))):
        with paths[kind].open("w") as stream:
            stream.write(f"Node Number = {len(item.xy)}\n")
            for (x, y), value in zip(item.xy, values):
                stream.write(f"{float(x)!r} {float(y)!r} {float(value)!r}\n")
    paths["obc"].write_text(
        f"OBC Node Number = {len(obc)}\n"
        + "".join(
            f"{i} {node + 1} {kind}\n"
            for i, (node, kind) in enumerate(zip(obc, types), 1)
        )
    )
    paths["spg"].write_text(
        f"Sponge Node Number = {len(spg)}\n"
        + "".join(f"{node + 1} 10.125 0.001\n" for node in spg)
    )
    return paths


def test_identity_map():
    base = mesh()
    nm = np.arange(4)
    em = rm.build_element_map(base, base, nm)
    np.testing.assert_array_equal(em, [0, 1])
    cls = rm.classify_mesh(nm, em, 4, 2)
    assert cls.frozen_nodes.all() and cls.frozen_elements.all()
    assert not cls.new_nodes.any() and rm.footprint_polygon(base, em).is_empty
    assert g0(base, base, nm)["verdict"] == "PASS"


def test_refined_square_one_added_node():
    base, refined = mesh(), refined_square()
    em = rm.build_element_map(base, refined, np.arange(4))
    np.testing.assert_array_equal(em, [-1, -1])
    cls = rm.classify_mesh(np.arange(4), em, 5, 4)
    np.testing.assert_array_equal(cls.new_nodes, [False] * 4 + [True])
    assert cls.new_elements.all()
    assert rm.footprint_polygon(base, em).equals(
        shape(
            {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            }
        )
    )
    assert g0(base, refined, np.arange(4))["new_nodes"] == 1


def test_removed_node():
    base = refined_square()
    refined = mesh()
    nm = np.array([0, 1, 2, 3, -1])
    rm.validate_node_map(nm, base, refined)
    np.testing.assert_array_equal(rm.build_element_map(base, refined, nm), [-1] * 4)
    assert g0(base, refined, nm)["removed_nodes"] == 1
    with pytest.raises(ValueError, match="negative"):
        rm.reindex_one_based(np.array([5]), nm)


def test_flipped_diagonal():
    base = mesh()
    refined = mesh(tri=np.array([[0, 1, 3], [1, 2, 3]]))
    em = rm.build_element_map(base, refined, np.arange(4))
    assert (em == -1).all()
    assert rm.footprint_polygon(base, em).area == 1.0
    assert g0(base, refined, np.arange(4))["frozen_elements"] == 0


def test_cyclic_orientation_and_permuted_order():
    base = mesh()
    refined = mesh(tri=np.roll(TRI[::-1], 1, axis=1))
    np.testing.assert_array_equal(
        rm.build_element_map(base, refined, np.arange(4)), [1, 0]
    )


def test_reversed_orientation_refused():
    with pytest.raises(ValueError, match="orientation"):
        rm.build_element_map(mesh(), mesh(tri=TRI[:, ::-1]), np.arange(4))


def test_duplicate_element_refused():
    with pytest.raises(ValueError, match="Duplicate element"):
        rm.build_element_map(mesh(), mesh(tri=np.vstack((TRI, TRI[0]))), np.arange(4))


def test_obc_node_not_frozen_refused():
    with pytest.raises(ValueError, match="OBC.*frozen"):
        g0(refined_square(), mesh(), [0, 1, 2, 3, -1], obc=(4,))


def test_sponge_node_not_frozen_refused():
    with pytest.raises(ValueError, match="Sponge.*frozen"):
        g0(refined_square(), mesh(), [0, 1, 2, 3, -1], spg=(4,))


def test_depth_below_floor_refused():
    with pytest.raises(ValueError, match="3 m floor"):
        g0(mesh(), refined_square(2.999999), np.arange(4))


def test_depth_above_cap_refused():
    with pytest.raises(ValueError, match="300 m cap"):
        g0(mesh(), refined_square(300.000001), np.arange(4))


def test_rfactor_new_edge_refused():
    with pytest.raises(ValueError, match="r-factor violation"):
        g0(mesh(), refined_square(5.0), np.arange(4))


def test_rfactor_serialization_tolerance_accepted():
    assert (
        g0(mesh(), refined_square(4.500001), np.arange(4))["new_edge_violations"] == 0
    )


def test_inherited_frozen_edge_accepted():
    base = mesh(h=[3.0, 4.500001, 3.0, 3.0])
    result = g0(base, base, np.arange(4))
    assert result["max_r_frozen_pair"] > 0.2
    assert result["frozen_edge_r_exact"]
    # Even an inherited value above the serialization limit is copied exactly.
    steep = mesh(h=[3.0, 6.0, 3.0, 3.0])
    assert g0(steep, steep, np.arange(4))["max_r_frozen_pair"] == 1 / 3


def test_new_diagonal_between_frozen_nodes_has_no_inheritance_exception():
    base = mesh(h=[4.4, 3.0, 4.4, 6.0])
    new = mesh(tri=np.array([[0, 1, 3], [1, 2, 3]]), h=base.depths)
    with pytest.raises(ValueError, match="r-factor violation"):
        g0(base, new, np.arange(4))


@pytest.mark.parametrize(
    "node_map, message",
    [
        ([0, 0, 2, 3], "Duplicate target"),
        ([0, 1, 2], "shape"),
        ([0, 1, 2, 4], "bounds"),
        ([0, 1, 2, -2], "bounds"),
        ([0.0, 1.0, 2.0, 3.0], "integer dtype"),
        ([[0, 1, 2, 3]], "shape"),
    ],
)
def test_invalid_node_map_refused(node_map, message):
    with pytest.raises(ValueError, match=message):
        rm.validate_node_map(np.array(node_map), mesh(), mesh())


@pytest.mark.parametrize("field", ["coordinates", "depths"])
def test_mapped_values_must_be_exact(field):
    base, refined = mesh(), mesh()
    (refined.xy if field == "coordinates" else refined.depths).flat[0] += 1e-12
    with pytest.raises(ValueError, match=f"frozen {field}"):
        rm.validate_node_map(np.arange(4), base, refined)


@pytest.mark.parametrize("value", [False, None])
def test_unconverged_report_refused(value):
    report = finish_report()
    report["converged_at_write_precision"] = value
    with pytest.raises(ValueError, match="not converged"):
        g0(mesh(), refined_square(), np.arange(4), report)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), -1.0, 0.01])
def test_untrusted_tolerance_refused(tolerance):
    report = finish_report()
    report["write_precision_tolerance"] = tolerance
    with pytest.raises(ValueError, match="serialization tolerance"):
        g0(mesh(), refined_square(), np.arange(4), report)


def test_one_based_reindexing():
    np.testing.assert_array_equal(
        rm.reindex_one_based(np.array([1, 3]), np.array([2, -1, 0])), [3, 1]
    )
    with pytest.raises(ValueError, match="outside"):
        rm.reindex_one_based(np.array([0]), np.arange(4))


@pytest.mark.parametrize(
    "kind, header",
    [
        ("dep", "Node Number"),
        ("cor", "Node Number"),
        ("obc", "OBC Node Number"),
        ("spg", "Sponge Node Number"),
    ],
)
def test_readers_refuse_extra_rows(tmp_path, kind, header):
    path = tmp_path / f"bad_{kind}.dat"
    path.write_text(f"{header} = 1\n1 2 3\n1 2 3\n")
    with pytest.raises(ValueError, match="exactly 1 rows"):
        getattr(rm, f"read_{kind}")(path)


@pytest.mark.parametrize("perturb", [False, True])
def test_depth_coordinate_decimal_round_trip(tmp_path, perturb):
    original = mesh(SQUARE + [500000.123456789, 3909931.0998815])
    paths = write_case(tmp_path, original)
    with paths["dep"].open("w") as stream:
        stream.write("Node Number = 4\n")
        for i, (x, y) in enumerate(original.xy):
            if perturb and i == 0:
                y += 0.000002
            stream.write(f"{x:.6f} {y:.6f} 3.000000\n")
    if perturb:
        with pytest.raises(ValueError, match="coordinates do not match"):
            rm.read_mesh(paths["grd"], paths["dep"])
    else:
        result = rm.read_mesh(paths["grd"], paths["dep"])
        np.testing.assert_array_equal(result.xy, original.xy)


def test_f1_overlap_without_shared_edge():
    # Two disconnected triangles overlap geometrically, with six distinct nodes.
    xy = np.array(
        [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.2, 0.2], [2.2, 0.2], [0.2, 2.2]]
    )
    item = mesh(xy, [[0, 1, 2], [3, 4, 5]])
    with pytest.raises(ValueError, match="summed area differs"):
        rm.validate_topology(item)


def test_f1_hole_area():
    xy = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 3.0],
            [0.0, 3.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 2.0],
        ]
    )
    tri = [
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
    result = rm.validate_topology(mesh(xy, tri))
    assert result["boundary_loops"] == 2
    assert result["boundary_area_m2"] == result["element_area_m2"] == 8.0


def test_f1_nonmanifold_edge():
    xy = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(ValueError, match="Nonmanifold edge"):
        rm.validate_topology(mesh(xy, [[0, 1, 2], [0, 1, 3], [0, 1, 4]]))


@pytest.mark.parametrize("header", ["Cell", "Node"])
def test_r3_f10_counts_before_allocation(tmp_path, monkeypatch, header):
    paths = write_case(tmp_path, refined_square())
    path = paths["grd"]
    data = path.read_text()
    count = 4 if header == "Cell" else 5
    path.write_text(
        data.replace(f"{header} Number = {count}", f"{header} Number = 2147483647")
    )

    def forbidden(*args, **kwargs):
        pytest.fail("reader reached allocation before count bound")

    monkeypatch.setattr(rm, "read_dat", forbidden)
    with pytest.raises(
        ValueError, match="declared grid counts exceed available records"
    ):
        rm.read_grd(path)


@pytest.mark.parametrize(
    "kind,body,expected",
    [
        ("spg", "+1, +1.0125d1, 1.0-3", (10.125, 0.001)),
        ("spg", "1\n\t1.0125Q1, .1d-2 / ignored", (10.125, 0.001)),
        ("obc", "+1, +1, +2", (0, 2)),
        ("obc", "2*1,2 / ignored", (0, 2)),
    ],
)
def test_r4_f4_fortran_boundary_records(tmp_path, kind, body, expected):
    header = "Sponge" if kind == "spg" else "OBC"
    path = tmp_path / f"tiny_{kind}.dat"
    path.write_text(f"{header} Node Number = 1\n{body}\n")
    ids, values = getattr(rm, f"read_{kind}")(path)
    if kind == "spg":
        assert ids.tolist() == [0] and values[0].tolist() == list(expected)
    else:
        assert (ids[0], values[0]) == expected


@pytest.mark.parametrize("kind", ["obc", "spg"])
@pytest.mark.parametrize("bad", ["1_0", "0x1", "nan", "1d0", "١"])
def test_r4_f4_boundary_integer_syntax(tmp_path, kind, bad):
    header = "Sponge" if kind == "spg" else "OBC"
    path = tmp_path / f"tiny_{kind}.dat"
    path.write_text(f"{header} Node Number = 1\n{bad} 1 2\n")
    with pytest.raises(ValueError, match="Fortran boundary record"):
        getattr(rm, f"read_{kind}")(path)


@pytest.mark.parametrize("kind", ["obc", "spg"])
def test_r4_f4_boundary_nonfortran_separator(tmp_path, kind):
    header = "Sponge" if kind == "spg" else "OBC"
    path = tmp_path / f"tiny_{kind}.dat"
    path.write_text(f"{header} Node Number = 1\n1\u00a01 2\n")
    with pytest.raises(ValueError, match="Fortran boundary record"):
        getattr(rm, f"read_{kind}")(path)


@pytest.mark.parametrize(
    "kind,header", [("obc", "OBC Node Number"), ("spg", "Sponge Node Number")]
)
@pytest.mark.parametrize("zero", ["0", "+0", "-0", "000"])
def test_r6_f5_zero_boundary_readers(tmp_path, kind, header, zero):
    path = tmp_path / (kind + ".dat")
    path.write_text(f"{header} = {zero}\n")
    ids, values = getattr(rm, "read_" + kind)(path)
    assert ids.shape == (0,)
    assert values.shape == ((0,) if kind == "obc" else (0, 2))


@pytest.mark.parametrize("header", ["Node", "Cell"])
def test_r6_f5_grid_count_stays_positive(tmp_path, header):
    path = tmp_path / "grid.dat"
    path.write_text(
        f"Node Number = {0 if header == 'Node' else 4}\n"
        f"Cell Number = {0 if header == 'Cell' else 2}\n"
        "1 1 2 3\n2 1 3 4\n1 0 0\n2 1 0\n3 1 1\n4 0 1\n"
    )
    with pytest.raises(ValueError, match="positive integer"):
        rm.read_grd(path)


MUTATIONS = [
    (
        "area",
        [
            (
                "    if abs(total - domain.area) > tolerance or abs(total - union.area) > "
                "tolerance:",
                "    if False:",
            )
        ],
        ["test_f1_overlap_without_shared_edge"],
    ),
    ("hole-area", [("        if winding:", "        if True:")], ["test_f1_hole_area"]),
    (
        "edge-incidence",
        [("    if (counts > 2).any():", "    if False:")],
        ["test_f1_nonmanifold_edge"],
    ),
    (
        "r3-grid-count-bound",
        [("    if sum(counts.values()) > available:", "    if False:")],
        ["test_r3_f10_counts_before_allocation"],
    ),
    (
        "r4-boundary-tokenization",
        [
            (
                "        record = list_read(lines, pos, types)",
                "        record = list_read(\n"
                '            [(n, " ".join(text.split())) for n, text in lines], pos, types\n'
                "        )",
            )
        ],
        ["test_r4_f4_boundary_nonfortran_separator"],
    ),
    (
        "r6-zero-boundary",
        [("    if zero:", "    if False:")],
        ["test_r6_f5_zero_boundary_readers"],
    ),
]


@pytest.mark.parametrize(
    "name,replacements,tests", MUTATIONS, ids=[m[0] for m in MUTATIONS]
)
def test_named_gate_detects_mutation(tmp_path, monkeypatch, name, replacements, tests):
    source = Path(rm.__file__).read_text()
    for before, after in replacements:
        assert before in source, f"Missing mutation site: {name}"
        source = source.replace(before, after)
    mutated = ModuleType(rm.__package__ + ".mesh_pair_mutant")
    mutated.__file__ = rm.__file__
    mutated.__package__ = rm.__package__
    monkeypatch.setitem(sys.modules, mutated.__name__, mutated)
    exec(compile(source, rm.__file__, "exec"), mutated.__dict__)
    monkeypatch.setitem(globals(), "rm", mutated)
    # Parametrized reader cases need their original concrete arguments.
    cases = {
        "r3-grid-count-bound": ("test_r3_f10_counts_before_allocation", ("Cell",)),
        "r4-boundary-tokenization": (
            "test_r4_f4_boundary_nonfortran_separator",
            ("obc",),
        ),
        "r6-zero-boundary": (
            "test_r6_f5_zero_boundary_readers",
            ("obc", "OBC Node Number", "0"),
        ),
    }
    if name in cases:
        test, args = cases[name]
        with pytest.raises((AssertionError, ValueError, pytest.fail.Exception)):
            if name == "r3-grid-count-bound":
                globals()[test](tmp_path, monkeypatch, *args)
            else:
                globals()[test](tmp_path, *args)
    else:
        with pytest.raises((AssertionError, ValueError, pytest.fail.Exception)):
            for test in tests:
                globals()[test]()


def test_r9_r_factors_integer_depths():
    from xfvcom.regrid import mesh_pair as mp

    edges = np.array([[0, 1]])
    assert mp.r_factors(np.array([3, 6], dtype=np.uint8), edges)[0] == pytest.approx(
        1 / 3
    )
    assert mp.r_factors(np.array([20000, 30000], dtype=np.int16), edges)[
        0
    ] == pytest.approx(0.2)


def test_r9_one_based_does_not_wrap():
    from xfvcom.regrid import mesh_pair as mp

    assert mp.to_one_based(np.array([255], dtype=np.uint8))[0] == 256
    assert (
        mp.reindex_one_based(np.array([1]), np.array([255], dtype=np.uint8))[0] == 256
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda mp: mp.mesh_edges(np.array([[0, 1, 2, 3]])),
        lambda mp: mp.mesh_edges(np.array([0, 1, 2])),
        lambda mp: mp.r_factors(np.array([3.0, 6.0]), np.array([[-1, 0]])),
        lambda mp: mp.r_factors(np.array([3.0, 6.0]), np.array([[0.0, 1.0]])),
        lambda mp: mp.r_factors(np.array([3.0, 6.0]), np.array([[0, 2]])),
    ],
)
def test_r9_mesh_helpers_refuse_malformed_arrays(call):
    from xfvcom.regrid import mesh_pair as mp

    with pytest.raises(ValueError):
        call(mp)


def _square():
    from xfvcom.regrid import mesh_pair as mp

    return mp.Mesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        np.array([[0, 1, 2], [0, 2, 3]]),
        np.full(4, 4.0),
    )


def test_r10_r_factors_wide_float_refused():
    from xfvcom.regrid import mesh_pair as mp

    with np.errstate(over="raise", invalid="raise"), pytest.raises(ValueError):
        mp.r_factors(np.full(2, np.longdouble("1e400")), np.array([[0, 1]]))


@pytest.mark.parametrize(
    "call",
    [
        lambda mp, sq: mp.reindex_one_based(np.array([1]), np.arange(4).reshape(2, 2)),
        lambda mp, sq: mp.footprint_polygon(sq, np.array([-2, 1])),
        lambda mp, sq: mp.validate_boundary_nodes(
            np.array([0]), np.arange(4, dtype=float), sq, sq, "OBC"
        ),
        lambda mp, sq: mp.validate_boundary_nodes(
            np.array([0]), np.array([4, 1, 2, 3]), sq, sq, "OBC"
        ),
    ],
)
def test_r10_map_helpers_refuse_malformed_maps(call):
    from xfvcom.regrid import mesh_pair as mp

    with pytest.raises(ValueError):
        call(mp, _square())


def test_r11_r_factors_no_sum_overflow():
    """Finite depths whose sum overflows still give |a-b|/(a+b) (round 11 #4)."""
    with np.errstate(over="raise", invalid="raise", under="raise"):
        r = rm.r_factors(np.array([1e308, 9e307]), np.array([[0, 1]]))
    assert r[0] == pytest.approx(1 / 19, rel=1e-15)
    assert rm.r_factors(np.array([3.0, 6.0]), np.array([[0, 1]]))[0] == 1 / 3

"""Node renumbering of grid-dependent FVCOM inputs (rivers namelist, OBC files)."""

import netCDF4
import numpy as np
import pytest

from xfvcom.regrid import inputs

NODE_MAP = np.array([0, -1, 3, 1, 2])  # base node 2 (one-based) removed

NAMELIST = """ ! RIVER_GRID_LOCATION = 2 in a comment is not a key
 &NML_RIVER
  RIVER_NAME          = 'A!b',
  RIVER_GRID_LOCATION = 3,   ! keep this comment
  RIVER_VERTICAL_DISTRIBUTION = 'uniform'
 /
 &nml_river
  river_name = 'B',
  river_grid_location=5
 /
"""


def test_namelist_renumbered_and_everything_else_kept():
    new, pairs = inputs.reindex_river_namelist(NAMELIST, NODE_MAP)
    assert pairs == [(3, 4), (5, 3)]
    assert new == NAMELIST.replace(
        "RIVER_GRID_LOCATION = 3,", "RIVER_GRID_LOCATION = 4,"
    ).replace("river_grid_location=5", "river_grid_location=3")


@pytest.mark.parametrize(
    "text,match",
    [
        (NAMELIST.replace("= 3,", "= 2,"), "removed|-1|refined"),
        (NAMELIST.replace("= 3,", "= 1+2,"), "plain integer"),
        (NAMELIST.replace("= 3,", "= 2*3,"), "plain integer"),
        (
            NAMELIST.replace("  RIVER_GRID_LOCATION = 3,   ! keep this comment\n", ""),
            "blocks",
        ),
        ("no namelist here\n", "no &NML_RIVER"),
        (NAMELIST.replace("= 3,", "= 9,"), "outside"),
    ],
)
def test_namelist_refusals(text, match):
    with pytest.raises(ValueError, match=match):
        inputs.reindex_river_namelist(text, NODE_MAP)


def test_namelist_location_must_belong_to_its_block():
    text = NAMELIST.replace(
        "  RIVER_GRID_LOCATION = 3,   ! keep this comment\n", ""
    ).replace(
        "river_grid_location=5", "river_grid_location=5\n RIVER_GRID_LOCATION = 4"
    )
    with pytest.raises(ValueError, match="own RIVER_GRID_LOCATION"):
        inputs.reindex_river_namelist(text, NODE_MAP)


def _obc_file(path, ids):
    with netCDF4.Dataset(path, "w", format="NETCDF4") as d:
        d.title = "test"
        d.createDimension("nobc", len(ids))
        d.createDimension("time", None)
        d.createDimension("siglay", 2)
        v = d.createVariable("obc_nodes", "i4", ("nobc",))
        v.long_name = "Open Boundary Node Number"
        v[:] = ids
        t = d.createVariable("time", "f8", ("time",))
        t.units = "days since 1858-11-17 00:00:00"
        t[:] = np.arange(5000) / 24
        e = d.createVariable(
            "obc_temp", "f4", ("time", "siglay", "nobc"), chunksizes=(1, 2, len(ids))
        )
        e.units = "Celcius"
        e[:] = np.random.default_rng(3).normal(size=(5000, 2, len(ids)))


def test_obc_netcdf_renumbered_values_kept(tmp_path):
    src, dst = tmp_path / "a.nc", tmp_path / "b.nc"
    _obc_file(src, [5, 3, 1])
    assert inputs.reindex_obc_netcdf(src, dst, NODE_MAP) == [(5, 3), (3, 4), (1, 1)]
    with netCDF4.Dataset(src) as a, netCDF4.Dataset(dst) as b:
        assert b["obc_nodes"][:].tolist() == [3, 4, 1]
        assert b["obc_nodes"].dtype == a["obc_nodes"].dtype
        assert b["obc_temp"][:].tobytes() == a["obc_temp"][:].tobytes()
        assert b["obc_temp"].chunking() == a["obc_temp"].chunking()
        assert b.dimensions["time"].isunlimited()
        assert b.title == "test"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.nc", "b.nc"]


def test_obc_netcdf_refusals(tmp_path):
    src = tmp_path / "a.nc"
    _obc_file(src, [5, 2, 1])
    with pytest.raises(ValueError):
        inputs.reindex_obc_netcdf(src, tmp_path / "b.nc", NODE_MAP)
    assert not (tmp_path / "b.nc").exists()
    _obc_file(tmp_path / "c.nc", [5, 3, 1])
    (tmp_path / "d.nc").write_bytes(b"x")
    with pytest.raises(ValueError, match="exists"):
        inputs.reindex_obc_netcdf(tmp_path / "c.nc", tmp_path / "d.nc", NODE_MAP)
    assert (tmp_path / "d.nc").read_bytes() == b"x"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.nc", "c.nc", "d.nc"]


def _forcing_pair():
    from xfvcom.regrid import mesh_pair as rm
    from xfvcom.regrid import transfer as rt

    base = rm.Mesh(
        np.array([[0.0, 0], [3, 0], [2, 2], [0, 1]]),
        np.array([[0, 1, 2], [0, 2, 3]]),
        np.array([8.0, 12, 16, 10]),
    )
    target = rm.Mesh(
        np.vstack((base.xy, [0.6, 0.4])),
        np.array([[0, 1, 4], [1, 2, 4], [2, 0, 4], [0, 2, 3]]),
        np.array([8.0, 12, 16, 10, 11]),
    )
    return rt, rt.build_transfer(base, target, np.arange(4))


def _forcing_file(path, times):
    rng = np.random.default_rng(7)
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as d:
        d.title = "forcing"
        for name, n in (("node", 4), ("nele", 2), ("three", 3)):
            d.createDimension(name, n)
        d.createDimension("time", None)
        x = d.createVariable("x", "f8", ("node",))
        x.units = "meters"
        x[:] = [0, 3, 2, 0]
        nv = d.createVariable("nv", "i4", ("three", "nele"))
        nv[:] = [[1, 1], [2, 3], [3, 4]]
        t = d.createVariable("time", "f8", ("time",), chunksizes=(512,))
        t.units = "days since 1858-11-17 00:00:00"
        t[:] = times
        for name, dim in (("air_temperature", "node"), ("uwind_speed", "nele")):
            v = d.createVariable(
                name, "f4", ("time", dim), chunksizes=(1, 4 if dim == "node" else 2)
            )
            v.units = "x"
            v[:] = rng.normal(size=(len(times), 4 if dim == "node" else 2))
        p = d.createVariable("precip", "f4", ("time", "node"))
        p[:] = d["air_temperature"][:]


def test_forcing_transferred_per_record_and_cut(tmp_path):
    rt, op = _forcing_pair()
    src, dst = tmp_path / "f.nc", tmp_path / "g.nc"
    times = np.arange(10.0)
    _forcing_file(src, times)
    coords = {
        "x": np.arange(5.0),
        "nv": np.array([[1, 2, 3, 1], [2, 3, 1, 3], [5, 5, 5, 4]]),
    }
    report = inputs.transfer_forcing_netcdf(
        src, dst, op, coords, start=2.5, end=6.0, block=2
    )
    assert report["records"] == (2, 6)
    with netCDF4.Dataset(src) as a, netCDF4.Dataset(dst) as b:
        assert b.data_model == "NETCDF4_CLASSIC"
        assert b["time"][:].tolist() == [2, 3, 4, 5, 6]
        assert b["x"][:].tolist() == list(range(5))
        assert b["nv"][:].tolist() == coords["nv"].tolist()
        for name, loc in (("air_temperature", "node"), ("uwind_speed", "element")):
            assert b[name].dtype == np.float32
            expected = rt.apply_2d(op, np.asarray(a[name][2:7]).T, location=loc).T
            assert np.asarray(b[name][:]).tobytes() == expected.tobytes()
            # frozen nodes / the matched element keep the source bytes
        assert (
            np.asarray(b["air_temperature"][:, :4]).tobytes()
            == np.asarray(a["air_temperature"][2:7]).tobytes()
        )
        assert (
            np.asarray(b["uwind_speed"][:, 3]).tobytes()
            == np.asarray(a["uwind_speed"][2:7, 1]).tobytes()
        )
        assert (
            np.asarray(b["precip"][:]).tobytes()
            == np.asarray(b["air_temperature"][:]).tobytes()
        )
        assert b["air_temperature"].units == "x" and b.title == "forcing"


@pytest.mark.parametrize(
    "coords,kwargs,match",
    [
        ({"x": np.arange(5.0)}, {}, "nv"),
        ({"x": np.arange(4.0), "nv": np.ones((3, 4))}, {}, "wrong shape"),
        (
            {"x": np.arange(5.0), "nv": np.ones((3, 4)), "y": np.arange(5.0)},
            {},
            "not in the source",
        ),
        ({"x": np.arange(5.0), "nv": np.ones((3, 4))}, {"start": -1.0}, "bracketed"),
        ({"x": np.arange(5.0), "nv": np.ones((3, 4))}, {"end": 9.5}, "bracketed"),
        (
            {"x": np.arange(5.0), "nv": np.ones((3, 4))},
            {"start": 5.0, "end": 4.0},
            "exceed",
        ),
    ],
)
def test_forcing_refusals(tmp_path, coords, kwargs, match):
    _, op = _forcing_pair()
    src = tmp_path / "f.nc"
    _forcing_file(src, np.arange(10.0))
    window = {"start": 1.0, "end": 2.0, **kwargs}
    with pytest.raises(ValueError, match=match):
        inputs.transfer_forcing_netcdf(src, tmp_path / "g.nc", op, coords, **window)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f.nc"]


def test_forcing_unknown_layout_and_wrong_mesh_refused(tmp_path):
    _, op = _forcing_pair()
    src = tmp_path / "f.nc"
    _forcing_file(src, np.arange(10.0))
    with netCDF4.Dataset(src, "a") as d:
        d.createDimension("other", 2)
        d.createVariable("odd", "f4", ("time", "other"))
    coords = {"x": np.arange(5.0), "nv": np.ones((3, 4))}
    with pytest.raises(ValueError, match="no rule"):
        inputs.transfer_forcing_netcdf(
            src, tmp_path / "g.nc", op, coords, start=1.0, end=2.0
        )
    _, op2 = _forcing_pair()
    other = tmp_path / "h.nc"
    with netCDF4.Dataset(other, "w") as d:
        d.createDimension("node", 7)
        d.createDimension("time", None)
        d.createVariable("time", "f8", ("time",))[:] = [0.0, 1.0]
    with pytest.raises(ValueError, match="base mesh"):
        inputs.transfer_forcing_netcdf(
            other, tmp_path / "g.nc", op2, {}, start=0.0, end=1.0
        )

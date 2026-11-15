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

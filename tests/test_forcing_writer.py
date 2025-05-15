from pathlib import Path

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from xfvcom.grid import read_grid
from xfvcom.io.forcing_writer import ForcingNcWriter


def test_write(tmp_path: Path):
    dat = Path("~/Github/TB-FVCOM/goto2023/input/TokyoBay18_grd.dat").expanduser()
    g = read_grid(dat, utm_zone=54)

    t: NDArray[np.datetime64] = np.arange(
        "2025-01-01T00", "2025-01-01T06", dtype="datetime64[h]"
    ).astype("datetime64[ns]")
    wind = xr.DataArray(
        np.random.rand(len(t), g.node), dims=("time", "node"), coords={"time": t}
    )
    press = xr.DataArray(
        1013.0 + np.random.randn(len(t), g.node),
        dims=("time", "node"),
        coords={"time": t},
    )

    out = tmp_path / "forcing.nc"
    ForcingNcWriter(g, {"wind": wind, "press": press}, out_path=out).write()

    import netCDF4 as nc

    with nc.Dataset(out) as ds:
        assert ds.dimensions["node"].size == g.node
        assert ds.variables["wind"].shape == (len(t), g.node)

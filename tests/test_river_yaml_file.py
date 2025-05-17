import io
from pathlib import Path

import xarray as xr

from xfvcom.io.river_nc_generator import RiverNetCDFGenerator

DATA = Path(__file__).parent / "data"


def test_static_yaml(tmp_path):
    cfg = DATA / "river_cfg.yaml"  # ← Refer to the existing file.
    assert cfg.exists(), "river_cfg.yaml not found"

    gen = RiverNetCDFGenerator(
        nml_path=DATA / "rivers_minimal.nml",
        start="2025-01-01T00:00:00Z",
        end="2025-01-01T00:00:00Z",
        dt_seconds=3600,
        config=cfg,
    )

    nc_path = tmp_path / "out.nc"
    nc_path.write_bytes(gen.render())  # save bytes → file
    ds = xr.open_dataset(nc_path, engine="netcdf4", decode_times=False)

    # Verify.
    assert (ds["river_flux"] == 2).all()

from pathlib import Path

import xarray as xr

from xfvcom.io.river_nc_generator import RiverNetCDFGenerator

DATA = Path(__file__).parent / "data"


def test_yaml_over_cli(tmp_path):
    yaml_cfg = tmp_path / "river_cfg.yaml"
    yaml_cfg.write_text("defaults:\n  flux: 2\n", encoding="utf-8")
    gen = RiverNetCDFGenerator(
        nml_path=DATA / "data" / "rivers_minimal.nml",
        start="2025-01-01T00:00:00Z",
        end="2025-01-01T00:00:00Z",
        dt_seconds=3600,
        default_flux=8,  # CLI (lower priority)
        default_temp=10,
        default_salt=0.1,
        config=yaml_cfg,
    )

    nc_path = tmp_path / "out.nc"
    nc_path.write_bytes(gen.render())  # save bytes → file
    ds = xr.open_dataset(nc_path, engine="netcdf4", decode_times=False)
    assert (ds["river_flux"] == 2).all()

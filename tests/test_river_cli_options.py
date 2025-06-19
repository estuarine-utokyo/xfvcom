from pathlib import Path

import xarray as xr

from xfvcom.io.river_nc_generator import RiverNetCDFGenerator

DATA = Path(__file__).parent / "data"


def test_cli_ts_const(tmp_path):
    out_nc = tmp_path / "river.nc"
    gen = RiverNetCDFGenerator(
        nml_path=DATA / "rivers_minimal.nml",
        start="2024-12-31T15:00Z",
        end="2025-01-01T03:00Z",
        dt_seconds=3600,
        default_flux=5,
        default_temp=20,
        default_salt=0.1,
        ts_specs=[
            f"Arakawa={DATA/'arakawa_flux.csv'}:flux",
            f"{DATA/'global_temp.tsv'}:temp",
        ],
        const_specs=[
            "Arakawa.salt=0.05",
            "flux=30",
        ],
    )
    gen.write(out_nc)
    assert out_nc.exists()
    ds = xr.open_dataset(out_nc, engine="netcdf4", decode_times=False)
    assert ds.sizes["time"] == 13

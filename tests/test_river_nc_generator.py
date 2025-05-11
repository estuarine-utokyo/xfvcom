# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

import xarray as xr

from xfvcom.io.river_nc_generator import RiverNetCDFGenerator


def test_constant_nc(tmp_path: Path) -> None:
    """Ensure a constant NetCDF is produced and has expected dims."""
    nml = Path(__file__).parent / "data" / "rivers_minimal.nml"
    result_path = tmp_path / "river.nc"

    gen = RiverNetCDFGenerator(
        nml_path=nml,
        start="2025-01-01T00:00:00Z",
        end="2025-01-01T03:00:00Z",
        dt_seconds=3600,
        default_flux=1.0,
        default_temp=10.0,
        default_salt=0.1,
    )

    gen.write(result_path)
    assert result_path.exists()

    # basic sanity check
    ds = xr.open_dataset(result_path)
    assert ds.sizes["time"] == 4
    assert ds.sizes["rivers"] == 1
    assert "river_flux" in ds and "river_temp" in ds and "river_salinity" in ds

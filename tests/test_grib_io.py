"""GRIB I/O smoke test for the xfvcom env.

Skips automatically if no ERA5 GRIB sample is reachable on the host.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _sample_grib() -> Path | None:
    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        return None
    p = Path(data_dir) / "ERA5/lat34-36_lon139-141/grib/2020_01.grib"
    return p if p.is_file() else None


def test_cfgrib_importable() -> None:
    # cfgrib + eccodes ship via conda-forge (eccodes is a C library); the
    # pip-only CI box does not have them. importorskip lets the test still
    # catch regressions in the conda env without failing CI.
    pytest.importorskip("cfgrib")
    pytest.importorskip("eccodes")


@pytest.mark.skipif(_sample_grib() is None, reason="No ERA5 GRIB sample reachable")
def test_open_era5_sample() -> None:
    pytest.importorskip("cfgrib")
    import xarray as xr

    p = _sample_grib()
    assert p is not None  # for mypy; the skipif guarantees this
    ds = xr.open_dataset(p, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        assert {"u10", "v10"}.issubset(ds.data_vars)
        # Tokyo Bay subset: 9 lat x 9 lon
        assert ds.sizes["latitude"] == 9
        assert ds.sizes["longitude"] == 9
    finally:
        ds.close()

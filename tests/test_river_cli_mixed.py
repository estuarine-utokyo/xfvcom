from pathlib import Path

import pytest

from xfvcom.io.river_nc_generator import RiverNetCDFGenerator

DATA = Path(__file__).parent / "data"


def test_cli_mixed(tmp_path):
    tmp_path / "river.nc"
    gen = RiverNetCDFGenerator(
        nml_path=DATA / "rivers_minimal.nml",
        start="2025-01-01T00:00:00Z",
        end="2025-01-01T12:00:00Z",
        dt_seconds=6 * 3600,
        default_flux=5,  # GLOBAL fallback
        default_temp=20,
        default_salt=0.1,
        ts_specs=[
            f"Arakawa={DATA/'arakawa_flux.csv'}:flux",
            f"{DATA/'global_temp.tsv'}:temp",
        ],
        const_specs=[
            "Arakawa.salt=0.05",
            "flux=30",  # GLOBAL override (Sumidagawa)
        ],
    )
    with pytest.raises(ValueError, match="outside the available data range"):
        gen.render()

from pathlib import Path

import pytest

from xfvcom.io.river_nc_generator import RiverNetCDFGenerator


def test_missing_nml(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.nml"
    with pytest.raises(FileNotFoundError):
        RiverNetCDFGenerator(
            nml_path=missing,
            start="2025-01-01T00:00Z",
            end="2025-01-01T00:00Z",
            dt_seconds=3600,
        )

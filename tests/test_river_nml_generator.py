# ~/Github/xfvcom/tests/test_river_generator.py
from __future__ import annotations

from pathlib import Path

from xfvcom.io.river_nml_generator import RiverNmlGenerator


def test_river_nml(tmp_path: Path) -> None:
    data_dir = Path(__file__).with_suffix("").parent / "data"
    csv_path = data_dir / "rivers.csv"
    expected_nml = (data_dir / "rivers_expected.nml").read_text(encoding="utf-8")

    gen = RiverNmlGenerator(csv_path)
    result = gen.generate()

    assert result.strip() == expected_nml.strip()

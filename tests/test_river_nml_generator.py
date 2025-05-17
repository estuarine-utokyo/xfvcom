from __future__ import annotations

from pathlib import Path

from xfvcom.io.river_nml_generator import RiverNmlGenerator

DATA = Path(__file__).parent / "data"


def test_river_nml(tmp_path: Path) -> None:
    csv_path = DATA / "rivers.csv"
    expected_nml = (DATA / "rivers_expected.nml").read_text(encoding="utf-8")

    gen = RiverNmlGenerator(csv_path)
    result = gen.generate()

    assert result.strip() == expected_nml.strip()

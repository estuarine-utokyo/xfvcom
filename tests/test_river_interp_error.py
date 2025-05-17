from pathlib import Path

import pytest

from xfvcom.io.river_nc_generator import RiverNetCDFGenerator

DATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("missing_file", ["global_temp.tsv"])
def test_interp_nan_error(tmp_path, missing_file):
    # データ範囲外 → 必ず ValueError
    yaml_cfg = tmp_path / "cfg.yaml"
    yaml_cfg.write_text(
        f"""
rivers:
  - name: Arakawa
    ts: {DATA/missing_file}:temp
interp:
  method: linear
""",
        encoding="utf-8",
    )

    gen = RiverNetCDFGenerator(
        nml_path=Path(__file__).parent / "data" / "rivers_minimal.nml",
        start="2025-01-01T00:00:00Z",
        end="2025-01-02T00:00:00Z",
        dt_seconds=6 * 3600,
        config=yaml_cfg,
    )

    # Any request outside the available data span must raise
    with pytest.raises(ValueError, match="outside the available data range"):
        gen.render()

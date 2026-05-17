# -*- coding: utf-8 -*-
"""Tests for kind: constant entries in the river_dl YAML adapter.

Covers schema validation in
:func:`xfvcom.cli.make_river_nc_from_river_dl._load_river_map` and the
broadcast path inside :class:`xfvcom.io.river_nc_generator.RiverNetCDFGenerator`.
"""

from __future__ import annotations

from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xfvcom.cli.make_river_nc_from_river_dl import _load_river_map
from xfvcom.io.river_nc_generator import RiverNetCDFGenerator

NML_TXT = """\
&NML_RIVER
 RIVER_NAME = 'RealRiver',
 RIVER_FILE = 'dummy.nc',
 RIVER_GRID_LOCATION = 1,
 RIVER_VERTICAL_DISTRIBUTION = 'uniform'
/
&NML_RIVER
 RIVER_NAME = 'Kisarazu',
 RIVER_FILE = 'dummy.nc',
 RIVER_GRID_LOCATION = 2,
 RIVER_VERTICAL_DISTRIBUTION = 'uniform'
/
"""


def _make_river_dl_nc(
    path: Path,
    *,
    times: pd.DatetimeIndex,
    discharge: np.ndarray,
    river: str = "RealRiver",
    station: str = "RealStation",
) -> None:
    ds = xr.Dataset(
        {
            "discharge": (
                ("time",),
                discharge.astype(np.float32),
                {"long_name": "river discharge", "units": "m3/s"},
            ),
        },
        coords={"time": times},
        attrs={"river": river, "station": station, "Conventions": "CF-1.8"},
    )
    ds.to_netcdf(path)


@pytest.fixture
def tiny_nml(tmp_path: Path) -> Path:
    p = tmp_path / "rivers.nml"
    p.write_text(NML_TXT, encoding="utf-8")
    return p


# ------------------------------------------------------------------
# 1. YAML schema validation (_load_river_map)
# ------------------------------------------------------------------
def test_load_river_map_default_kind_is_river_dl(tmp_path: Path) -> None:
    """An entry with neither 'kind' nor explicit kind: river_dl behaves
    identically to one with kind: river_dl (backward compat)."""
    yaml_p = tmp_path / "map.yaml"
    yaml_p.write_text(
        "rivers:\n" "  - name: A\n" "    source: /tmp/a.nc\n",
        encoding="utf-8",
    )
    river_dl_map, _defaults = _load_river_map(yaml_p)
    assert river_dl_map["A"]["kind"] == "river_dl"
    assert "source" in river_dl_map["A"]


def test_load_river_map_kind_constant_basic(tmp_path: Path) -> None:
    yaml_p = tmp_path / "map.yaml"
    yaml_p.write_text(
        "defaults:\n"
        "  temp: 15.0\n"
        "  salt: 0.0\n"
        "rivers:\n"
        "  - name: K\n"
        "    kind: constant\n"
        "    flux: 0.2902\n"
        "    temp: 10.0\n",
        encoding="utf-8",
    )
    river_dl_map, _ = _load_river_map(yaml_p)
    assert river_dl_map["K"]["kind"] == "constant"
    assert river_dl_map["K"]["flux"] == 0.2902
    assert river_dl_map["K"]["temp"] == 10.0
    assert river_dl_map["K"]["salt"] == 0.0  # from defaults
    assert "source" not in river_dl_map["K"]


def test_load_river_map_kind_constant_with_source_raises(tmp_path: Path) -> None:
    yaml_p = tmp_path / "map.yaml"
    yaml_p.write_text(
        "rivers:\n"
        "  - name: K\n"
        "    kind: constant\n"
        "    flux: 1.0\n"
        "    source: /tmp/forbidden.nc\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="kind: constant"):
        _load_river_map(yaml_p)


def test_load_river_map_kind_constant_with_scale_raises(tmp_path: Path) -> None:
    yaml_p = tmp_path / "map.yaml"
    yaml_p.write_text(
        "rivers:\n"
        "  - name: K\n"
        "    kind: constant\n"
        "    flux: 1.0\n"
        "    scale: 2.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="scale"):
        _load_river_map(yaml_p)


def test_load_river_map_kind_constant_missing_flux_raises(tmp_path: Path) -> None:
    yaml_p = tmp_path / "map.yaml"
    yaml_p.write_text(
        "rivers:\n" "  - name: K\n" "    kind: constant\n" "    temp: 10.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="flux"):
        _load_river_map(yaml_p)


def test_load_river_map_unknown_kind_raises(tmp_path: Path) -> None:
    yaml_p = tmp_path / "map.yaml"
    yaml_p.write_text(
        "rivers:\n" "  - name: K\n" "    kind: streamflow\n" "    source: /tmp/x.nc\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported kind"):
        _load_river_map(yaml_p)


# ------------------------------------------------------------------
# 2. Generator integration: constant column is broadcast verbatim
# ------------------------------------------------------------------
def test_generator_kind_constant_broadcast(tmp_path: Path, tiny_nml: Path) -> None:
    times = pd.date_range("2020-01-01", periods=5, freq="1h")
    f_real = tmp_path / "real.nc"
    _make_river_dl_nc(f_real, times=times, discharge=np.full(5, 42.0, dtype=np.float32))

    out_nc = tmp_path / "out.nc"
    gen = RiverNetCDFGenerator(
        tiny_nml,
        "2020-01-01T00:00:00Z",
        "2020-01-01T03:00:00Z",  # 4 timesteps at dt=3600
        3600,
        river_dl_map={
            "RealRiver": {
                "kind": "river_dl",
                "source": f_real,
                "temp": 15.0,
                "salt": 0.0,
            },
            "Kisarazu": {
                "kind": "constant",
                "flux": 0.2902,
                "temp": 10.0,
                "salt": 0.0,
            },
        },
    )
    gen.write(out_nc)

    with nc.Dataset(out_nc, "r") as ds:
        assert ds.dimensions["rivers"].size == 2
        # The NML lists RealRiver first, Kisarazu second; this order is
        # preserved (the constant entry does NOT jump to the front).
        rname_rows = ds.variables["river_names"][:, :]
        cleaned = ["".join(c.decode() for c in row).strip() for row in rname_rows]
        assert cleaned == ["RealRiver", "Kisarazu"]

        flux = ds.variables["river_flux"][:, :]
        np.testing.assert_allclose(flux[:, 0], np.full(4, 42.0), rtol=1e-5)
        np.testing.assert_allclose(flux[:, 1], np.full(4, 0.2902), rtol=1e-7)

        temp = ds.variables["river_temp"][:, :]
        np.testing.assert_allclose(temp[:, 1], np.full(4, 10.0), rtol=1e-7)

        salt = ds.variables["river_salt"][:, :]
        np.testing.assert_array_equal(salt[:, 1], np.zeros(4))


def test_generator_kind_constant_validation_errors(
    tmp_path: Path, tiny_nml: Path
) -> None:
    """The generator must mirror the CLI-side validation: source+constant
    cannot coexist; constant + scale is forbidden; constant without flux
    is rejected."""
    common = dict(
        start="2020-01-01T00:00:00Z",
        end="2020-01-01T01:00:00Z",
        dt_seconds=3600,
    )

    with pytest.raises(ValueError, match="kind: constant.*source"):
        RiverNetCDFGenerator(
            tiny_nml,
            **common,  # type: ignore[arg-type]
            river_dl_map={
                "Kisarazu": {
                    "kind": "constant",
                    "flux": 0.1,
                    "source": tmp_path / "x.nc",
                },
            },
        )

    with pytest.raises(ValueError, match="kind: constant.*scale"):
        RiverNetCDFGenerator(
            tiny_nml,
            **common,  # type: ignore[arg-type]
            river_dl_map={
                "Kisarazu": {"kind": "constant", "flux": 0.1, "scale": 2.0},
            },
        )

    with pytest.raises(ValueError, match="flux"):
        RiverNetCDFGenerator(
            tiny_nml,
            **common,  # type: ignore[arg-type]
            river_dl_map={
                "Kisarazu": {"kind": "constant", "temp": 10.0},
            },
        )


def test_generator_does_not_perturb_river_dl_when_constant_added(
    tmp_path: Path, tiny_nml: Path
) -> None:
    """Adding a kind: constant entry must not change any river_dl entry's
    output column (regression guard for the new branch)."""
    times = pd.date_range("2020-01-01", periods=8, freq="1h")
    f_real = tmp_path / "real.nc"
    _make_river_dl_nc(
        f_real,
        times=times,
        discharge=np.arange(8, dtype=np.float32) + 100.0,  # 100..107
    )

    common = dict(
        nml_path=tiny_nml,
        start="2020-01-01T00:00:00Z",
        end="2020-01-01T03:00:00Z",
        dt_seconds=3600,
    )

    # Reference: only RealRiver in river_dl_map.  Kisarazu in the NML
    # will fall back to default flux/temp/salt (0/20/0).
    ref_nc = tmp_path / "ref.nc"
    RiverNetCDFGenerator(
        **common,  # type: ignore[arg-type]
        river_dl_map={
            "RealRiver": {
                "kind": "river_dl",
                "source": f_real,
                "temp": 15.0,
                "salt": 0.0,
            },
        },
    ).write(ref_nc)

    # With Kisarazu added as constant: RealRiver column must be unchanged.
    new_nc = tmp_path / "new.nc"
    RiverNetCDFGenerator(
        **common,  # type: ignore[arg-type]
        river_dl_map={
            "RealRiver": {
                "kind": "river_dl",
                "source": f_real,
                "temp": 15.0,
                "salt": 0.0,
            },
            "Kisarazu": {
                "kind": "constant",
                "flux": 0.2902,
                "temp": 10.0,
                "salt": 0.0,
            },
        },
    ).write(new_nc)

    with nc.Dataset(ref_nc) as a, nc.Dataset(new_nc) as b:
        # Both files have 2 rivers (Kisarazu listed in NML always);
        # what changes is only the Kisarazu column.
        np.testing.assert_array_equal(
            a.variables["river_flux"][:, 0], b.variables["river_flux"][:, 0]
        )
        np.testing.assert_array_equal(
            a.variables["river_temp"][:, 0], b.variables["river_temp"][:, 0]
        )
        np.testing.assert_array_equal(
            a.variables["river_salt"][:, 0], b.variables["river_salt"][:, 0]
        )

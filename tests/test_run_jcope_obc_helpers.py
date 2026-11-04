# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Input checks of scripts/run_jcope_obc.py (year list, OBC node file)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_jcope_obc as r  # noqa: E402  # isort: skip


@pytest.mark.parametrize("years", [[2021, 2020], [2020, 2020], [2020, 2022]])
def test_bad_year_lists_are_refused(years):
    with pytest.raises(SystemExit):
        r.check_year_list(years)


def test_year_gaps_can_be_allowed_but_order_cannot():
    r.check_year_list([2020, 2022], allow_gaps=True)
    with pytest.raises(SystemExit):
        r.check_year_list([2022, 2020], allow_gaps=True)
    r.check_year_list(r.parse_year_spec("2020-2025"))


def test_obc_file_is_parsed_strictly(tmp_path):
    ok = tmp_path / "ok.dat"
    ok.write_text("OBC Node Number = 3\n1 101 1\n2 102 1\n3 103 1\n")
    assert r.parse_obc_dat(ok).tolist() == [101, 102, 103]
    for body in (
        "OBC Node Number = 3\n1 101 1\n2 bad 1\n",
        "OBC Node Number = 2\n1 0 1\n",
        "x\n1 5 1\n2 5 1\n",
    ):
        f = tmp_path / "bad.dat"
        f.write_text(body)
        with pytest.raises(ValueError):
            r.parse_obc_dat(f)

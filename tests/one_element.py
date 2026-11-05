# Copyright Jun Sasaki
# SPDX-License-Identifier: MIT
"""Reading the one-element test meshes of the earlier review rounds (review round 21 #3).

FVCOM cannot read a one-element ``*_grd.dat``: its connectivity loop leaves the block only at a
record with ``J == 1 .AND. LM1 /= 1``, and after element 1 ``LM1`` is 1 (checked with gfortran 8.5
and ifx 2025.1 replicas of READ_COLDSTART_GRID).  The strict reader therefore refuses such a mesh,
and it does so only AFTER all its other checks.  A one-element test mesh that is otherwise valid
is read here in two steps: the strict reader must reach exactly that refusal (so every other strict
check passed), and the permissive mode returns the arrays the test inspects (the same values, since
the fixtures' ids run 1..n in order).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from xfvcom.grid.dat_reader import read_dat

ONE_ELEMENT = "the mesh has one element, whose id is 1"


def read_one_element(path: str | Path) -> Any:
    """``read_dat`` of a valid one-element mesh: strict refuses it for FVCOM's one-element limit
    only; the permissive mode's result is returned."""
    with pytest.raises(ValueError, match=ONE_ELEMENT):
        read_dat(path)
    return read_dat(path, allow_unordered=True)

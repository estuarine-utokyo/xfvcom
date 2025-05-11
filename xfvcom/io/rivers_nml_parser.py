from __future__ import annotations

import re
from pathlib import Path
from typing import List

_RIVER_RE = re.compile(r"RIVER_NAME\s*=\s*'([^']+)'")


def parse_rivers_nml(path: Path) -> List[str]:
    """Return list of river names found in a rivers.nml file."""
    txt = path.read_text(encoding="utf-8")
    return _RIVER_RE.findall(txt)

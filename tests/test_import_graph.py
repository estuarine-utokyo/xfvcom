# ~/Github/xfvcom/tests/test_import_graph.py
import importlib
import pkgutil
from pathlib import Path

import xfvcom

PKG_ROOT = Path(xfvcom.__file__).parent


def iter_submodules(pkg_name: str):
    """Yield all sub-modules of `pkg_name` (depth-first)."""
    pkg = importlib.import_module(pkg_name)
    for mod in pkgutil.walk_packages(pkg.__path__, f"{pkg_name}."):
        yield mod.name


def test_no_circular_utils_dependency():
    """
    High-level modules (plot.*, analysis, io) must NOT import from
    xfvcom.plot.core など“上位”層を逆参照しない。
    We only check that anything *outside* xfvcom.utils.*
    never imports xfvcom.plot.core inside its module code string.
    """
    bad_ref = "xfvcom.plot.core"
    for mod_name in iter_submodules("xfvcom"):
        if mod_name.startswith("xfvcom.utils."):
            continue  # utils can be imported anywhere
        source = importlib.import_module(mod_name).__dict__.get("__file__", "")
        if not source or not source.endswith(".py"):
            continue
        text = Path(source).read_text()
        assert bad_ref not in text, f"{mod_name} imports {bad_ref}"

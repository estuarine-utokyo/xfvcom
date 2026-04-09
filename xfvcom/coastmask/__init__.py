"""Deprecated: ``xfvcom.coastmask`` has moved to the standalone ``xcoast`` package.

This shim re-exports the public API from :mod:`xcoast` for backward
compatibility and emits a :class:`DeprecationWarning` on import.

New code should import from ``xcoast`` directly::

    from xcoast import load, CoastmaskConfig, CoastMask

Install ``xcoast`` with::

    pip install git+https://github.com/estuarine-utokyo/xcoast.git
    # or, for editable development:
    pip install -e /path/to/xcoast
"""

from __future__ import annotations

import warnings

try:
    from xcoast import (
        PRESETS,
        BboxPreset,
        CoastMask,
        CoastmaskConfig,
        add_land_to_axes,
        add_land_to_plain_axes,
        cleanup_land,
        fill_small_holes,
        load,
        remove_orphan_islands,
    )
except ImportError as exc:  # pragma: no cover - exercised manually
    raise ImportError(
        "xfvcom.coastmask has moved to the standalone 'xcoast' package, "
        "which is now required. Install it with:\n"
        "    pip install git+https://github.com/estuarine-utokyo/xcoast.git\n"
        "or, for editable development:\n"
        "    pip install -e /path/to/xcoast"
    ) from exc

warnings.warn(
    "xfvcom.coastmask has moved to the 'xcoast' package and the shim will be "
    "removed in a future xfvcom release. Please update imports to "
    "'from xcoast import ...'.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BboxPreset",
    "CoastMask",
    "CoastmaskConfig",
    "PRESETS",
    "add_land_to_axes",
    "add_land_to_plain_axes",
    "cleanup_land",
    "fill_small_holes",
    "load",
    "remove_orphan_islands",
]

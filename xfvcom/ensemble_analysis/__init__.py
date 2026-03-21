"""Ensemble analysis sub-package for xfvcom.

This module provides utilities for analyzing FVCOM ensemble model output,
including member-node mapping extraction, source identification, and labeling.
"""

from __future__ import annotations

from .member_info import (  # Legacy module-level variables (for backward compatibility); Functions
    DEFAULT_SOURCE_NAMES,
    MEMBER_SOURCE_NAMES,
    MEMBER_SOURCE_NAMES_FULL,
    MEMBER_SOURCE_TYPES,
    SOURCE_COLORS,
    export_member_mapping,
    extract_member_node_mapping,
    get_default_source_names,
    get_member_source_names,
    get_member_source_types,
    get_member_summary,
    get_node_coordinates,
    get_river_sewer_boundary,
    get_source_color,
    get_source_colors_dict,
    get_source_name,
)
from .source_detection import (
    SUBSOURCE_PREFIXES,
    SourceDetector,
    extract_group_name,
    simplify_source_name,
)

__all__ = [
    # Legacy hardcoded mappings (for backward compatibility)
    "DEFAULT_SOURCE_NAMES",
    "MEMBER_SOURCE_NAMES",
    "MEMBER_SOURCE_NAMES_FULL",
    "MEMBER_SOURCE_TYPES",
    "SOURCE_COLORS",
    # Functions for extracting member info
    "extract_member_node_mapping",
    "get_member_summary",
    "export_member_mapping",
    "get_node_coordinates",
    # Dynamic accessor functions (use external config if loaded)
    "get_source_name",
    "get_source_color",
    "get_source_colors_dict",
    "get_default_source_names",
    "get_member_source_names",
    "get_member_source_types",
    "get_river_sewer_boundary",
    # Dynamic source detection (recommended)
    "SourceDetector",
    "extract_group_name",
    "simplify_source_name",
    "SUBSOURCE_PREFIXES",
]

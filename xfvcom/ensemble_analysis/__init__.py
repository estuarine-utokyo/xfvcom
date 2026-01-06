"""Ensemble analysis sub-package for xfvcom.

This module provides utilities for analyzing FVCOM ensemble model output,
including member-node mapping extraction and source identification.
"""

from __future__ import annotations

from .member_info import (
    DEFAULT_SOURCE_NAMES,
    MEMBER_SOURCE_NAMES,
    MEMBER_SOURCE_NAMES_FULL,
    MEMBER_SOURCE_TYPES,
    export_member_mapping,
    extract_member_node_mapping,
    get_member_summary,
    get_node_coordinates,
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
    "extract_member_node_mapping",
    "get_member_summary",
    "export_member_mapping",
    "get_node_coordinates",
    "get_source_name",
    # Dynamic source detection (recommended)
    "SourceDetector",
    "extract_group_name",
    "simplify_source_name",
    "SUBSOURCE_PREFIXES",
]

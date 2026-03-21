"""Configuration module for xfvcom.

This module provides utilities for loading external configuration files,
allowing xfvcom to be used across different FVCOM projects without
hardcoding project-specific source definitions.

Usage:
    from xfvcom.config import load_source_config, get_source_config

    # Load project-specific configuration
    load_source_config("/path/to/sources.yaml")

    # Access configuration values
    nodes = get_source_nodes()
    colors = get_source_colors()
"""

from __future__ import annotations

from xfvcom.config.source_config import (
    build_din_array,
    clear_config,
    get_all_nodes_in_order,
    get_all_subsource_names,
    get_config_path,
    get_decay_variants,
    get_display_config,
    get_member_config,
    get_member_source_mapping,
    get_river_names,
    get_sewer_din,
    get_sewer_names,
    get_source_colors,
    get_source_config,
    get_source_nodes,
    is_config_loaded,
    load_source_config,
)

__all__ = [
    "load_source_config",
    "get_source_config",
    "get_config_path",
    "is_config_loaded",
    "clear_config",
    "get_source_nodes",
    "get_source_colors",
    "get_sewer_names",
    "get_river_names",
    "get_member_config",
    "get_member_source_mapping",
    "get_decay_variants",
    "get_display_config",
    "get_sewer_din",
    "get_all_subsource_names",
    "get_all_nodes_in_order",
    "build_din_array",
]

"""External source configuration loader for xfvcom.

This module allows xfvcom to load project-specific source definitions from
YAML files, making the package usable across different FVCOM projects without
hardcoding source names, nodes, or colors.

Usage:
    from xfvcom.config import load_source_config, get_source_config

    # Load project-specific configuration
    load_source_config("/path/to/sources.yaml")

    # Access configuration
    config = get_source_config()
    nodes = get_source_nodes()
    colors = get_source_colors()

When no configuration is loaded, the module provides empty defaults,
and other xfvcom modules fall back to their hardcoded values for
backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Optional import - yaml may not be available in minimal installations
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Default empty configuration (used when no config file is loaded)
_DEFAULT_CONFIG = {
    "rivers": {},
    "sewers": {},
    "sewer_dye_by_year": {},
    "special": {},
    "members": [],
    "display": {},
    "decay_variants": {},
    "dye_config": {},
}  # type: Dict[str, Any]

# Module-level state
_loaded_config = None  # type: Optional[Dict[str, Any]]
_config_path = None  # type: Optional[Path]


def load_source_config(config_path):
    # type: (Union[str, Path]) -> Dict[str, Any]
    """Load source configuration from a YAML file.

    Parameters
    ----------
    config_path : str or Path
        Path to the sources.yaml configuration file.

    Returns
    -------
    dict
        The loaded configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ImportError
        If PyYAML is not installed.
    ValueError
        If the YAML file cannot be parsed.

    Examples
    --------
    >>> load_source_config("/path/to/TB-FVCOM/goto2023/dye_run/config/sources.yaml")
    >>> config = get_source_config()
    >>> print(config["rivers"].keys())
    dict_keys(['Arakawa', 'Sumidagawa', ...])
    """
    global _loaded_config, _config_path

    if not YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required to load configuration files. "
            "Install it with: pip install pyyaml"
        )

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError("Configuration file not found: {}".format(config_path))

    with open(config_path, encoding="utf-8") as f:
        try:
            _loaded_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError("Failed to parse YAML configuration: {}".format(e))

    _config_path = config_path

    return _loaded_config


def get_source_config():
    # type: () -> Dict[str, Any]
    """Get the currently loaded source configuration.

    Returns
    -------
    dict
        The loaded configuration, or default empty configuration if
        no file has been loaded.

    Notes
    -----
    If no configuration has been loaded, returns a default empty
    configuration. This allows xfvcom modules to gracefully fall back
    to hardcoded values when no external config is provided.
    """
    if _loaded_config is None:
        return _DEFAULT_CONFIG.copy()
    return _loaded_config


def get_config_path():
    # type: () -> Optional[Path]
    """Get the path to the loaded configuration file.

    Returns
    -------
    Path or None
        Path to the loaded config file, or None if no file is loaded.
    """
    return _config_path


def is_config_loaded():
    # type: () -> bool
    """Check if an external configuration has been loaded.

    Returns
    -------
    bool
        True if a configuration file has been loaded.
    """
    return _loaded_config is not None


def clear_config():
    # type: () -> None
    """Clear the loaded configuration.

    This resets the module to its initial state, useful for testing
    or when switching between different configurations.
    """
    global _loaded_config, _config_path
    _loaded_config = None
    _config_path = None


# =============================================================================
# Convenience functions for accessing common configuration values
# =============================================================================


def get_source_nodes():
    # type: () -> Dict[str, List[int]]
    """Get mapping of source name to node IDs.

    Returns
    -------
    dict[str, list[int]]
        Dictionary mapping source group names to lists of node IDs.
        Empty dict if no configuration is loaded.

    Examples
    --------
    >>> nodes = get_source_nodes()
    >>> nodes.get("Arakawa")
    [310, 241, 312, 448]
    """
    config = get_source_config()
    nodes = {}  # type: Dict[str, List[int]]

    # Extract river nodes
    for name, data in config.get("rivers", {}).items():
        nodes[name] = [sub["node"] for sub in data.get("subsources", [])]

    # Extract sewer nodes
    for name, data in config.get("sewers", {}).items():
        nodes[name] = [sub["node"] for sub in data.get("subsources", [])]

    return nodes


def get_source_colors():
    # type: () -> Dict[str, str]
    """Get mapping of source name to color.

    Returns
    -------
    dict[str, str]
        Dictionary mapping source names to color hex strings.
        Empty dict if no configuration is loaded.

    Examples
    --------
    >>> colors = get_source_colors()
    >>> colors.get("Arakawa")
    '#1f77b4'
    """
    config = get_source_config()
    colors = {}  # type: Dict[str, str]

    # Extract river colors
    for name, data in config.get("rivers", {}).items():
        if "color" in data:
            colors[name] = data["color"]

    # Extract sewer colors
    for name, data in config.get("sewers", {}).items():
        if "color" in data:
            colors[name] = data["color"]

    # Extract special source colors
    for name, data in config.get("special", {}).items():
        if "color" in data:
            colors[name] = data["color"]

    return colors


def get_sewer_names():
    # type: () -> List[str]
    """Get list of sewer source names.

    Returns
    -------
    list[str]
        List of sewer source names (e.g., ['Shibaura', 'Sunamachi', ...]).
        Empty list if no configuration is loaded.
    """
    config = get_source_config()
    return list(config.get("sewers", {}).keys())


def get_river_names():
    # type: () -> List[str]
    """Get list of river source names.

    Returns
    -------
    list[str]
        List of river source names (e.g., ['Arakawa', 'Sumidagawa', ...]).
        Empty list if no configuration is loaded.
    """
    config = get_source_config()
    return list(config.get("rivers", {}).keys())


def get_member_config():
    # type: () -> List[Dict[str, Any]]
    """Get member configuration list.

    Returns
    -------
    list[dict]
        List of member configuration dictionaries, each containing:
        - id: Member ID (0, 1, 2, ...)
        - name: Short name (e.g., 'arakawa')
        - label: Display label (e.g., 'Arakawa')
        - sources: List of source names or 'all'
        - groundwater_dye: bool (optional)
        - obc_dye: bool (optional)

    Examples
    --------
    >>> members = get_member_config()
    >>> members[0]
    {'id': 0, 'name': 'all', 'label': 'All Sources', 'sources': 'all', ...}
    """
    config = get_source_config()
    return config.get("members", [])


def get_member_source_mapping():
    # type: () -> Dict[int, str]
    """Get mapping of member ID to source label.

    This is equivalent to the hardcoded MEMBER_SOURCE_NAMES in member_info.py.

    Returns
    -------
    dict[int, str]
        Dictionary mapping member IDs to source labels.
        Empty dict if no configuration is loaded.

    Examples
    --------
    >>> mapping = get_member_source_mapping()
    >>> mapping[1]
    'Arakawa'
    """
    members = get_member_config()
    return {
        m["id"]: m.get("label", m.get("name", "Member{}".format(m["id"])))
        for m in members
    }


def get_decay_variants():
    # type: () -> Dict[str, Dict[str, Any]]
    """Get decay variant configurations.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping variant names (e.g., 'r10', 'r20') to their
        configuration (decay_on, half_life).

    Examples
    --------
    >>> variants = get_decay_variants()
    >>> variants['r10']
    {'decay_on': True, 'half_life': 10.0}
    """
    config = get_source_config()
    return config.get("decay_variants", {})


def get_display_config():
    # type: () -> Dict[str, Any]
    """Get display configuration (names, colors).

    Returns
    -------
    dict
        Display configuration including short_names, full_names,
        and category_colors.
    """
    config = get_source_config()
    return config.get("display", {})


def get_sewer_din(year):
    # type: (int) -> Dict[str, float]
    """Get sewer DIN concentrations for a specific year.

    Parameters
    ----------
    year : int
        Year for which to get sewer DIN values.

    Returns
    -------
    dict[str, float]
        Dictionary mapping sewer names to DIN values (umol/L).
        Empty dict if year not found or no configuration loaded.

    Examples
    --------
    >>> din = get_sewer_din(2021)
    >>> din['Shibaura']
    913.85
    """
    config = get_source_config()
    din_by_year = config.get("sewer_dye_by_year", {})
    if not din_by_year:
        din_by_year = config.get("sewer_din_by_year", {})
    return din_by_year.get(year, {})


def get_all_subsource_names():
    # type: () -> List[str]
    """Get list of all subsource names in order.

    This returns the individual subsource names (e.g., 'EastArakawa',
    'CenterArakawa') in the order they appear in the configuration,
    which matches the order expected in DYE_SOURCE_TERM arrays.

    Returns
    -------
    list[str]
        Ordered list of all subsource names.
    """
    config = get_source_config()
    names = []  # type: List[str]

    # Rivers first
    for river_data in config.get("rivers", {}).values():
        for sub in river_data.get("subsources", []):
            names.append(sub["name"])

    # Then sewers
    for sewer_data in config.get("sewers", {}).values():
        for sub in sewer_data.get("subsources", []):
            names.append(sub["name"])

    return names


def get_all_nodes_in_order():
    # type: () -> List[int]
    """Get list of all node IDs in DYE_SOURCE_TERM order.

    Returns
    -------
    list[int]
        Ordered list of all node IDs matching M_SPECIFY order.
    """
    config = get_source_config()
    nodes = []  # type: List[int]

    # Rivers first
    for river_data in config.get("rivers", {}).values():
        for sub in river_data.get("subsources", []):
            nodes.append(sub["node"])

    # Then sewers
    for sewer_data in config.get("sewers", {}).values():
        for sub in sewer_data.get("subsources", []):
            nodes.append(sub["node"])

    return nodes


def build_din_array(sewer_year):
    # type: (int) -> List[float]
    """Build the DYE_SOURCE_TERM array for all sources.

    Parameters
    ----------
    sewer_year : int
        Year for sewer DIN values.

    Returns
    -------
    list[float]
        DIN values for all sources in M_SPECIFY order.

    Examples
    --------
    >>> din = build_din_array(2021)
    >>> len(din)
    29
    >>> din[0]  # First Arakawa subsource
    207.59
    """
    config = get_source_config()
    din_values = []  # type: List[float]
    sewer_din = get_sewer_din(sewer_year)

    # Rivers first
    for river_name, river_data in config.get("rivers", {}).items():
        river_din = river_data.get("din", 0.0)
        for _ in river_data.get("subsources", []):
            din_values.append(river_din)

    # Then sewers
    for sewer_name, sewer_data in config.get("sewers", {}).items():
        sewer_din_value = sewer_din.get(sewer_name, 0.0)
        for _ in sewer_data.get("subsources", []):
            din_values.append(sewer_din_value)

    return din_values

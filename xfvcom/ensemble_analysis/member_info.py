"""Extract and analyze member-to-node mappings from FVCOM ensemble runs.

This module provides utilities to extract which nodes are active in each
ensemble member by parsing FVCOM namelist files and configuration.

Source configurations can be loaded from external YAML files via
xfvcom.config.load_source_config(). When external config is loaded,
functions in this module will use those values instead of the hardcoded
defaults below.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

# Lazy import to avoid slow package initialization
# parse_member_namelist is imported inside extract_member_node_mapping()


# =============================================================================
# Default hardcoded values (for backward compatibility)
# These are used when no external configuration is loaded.
# =============================================================================

# Default source names for TB-FVCOM goto2023 dye runs
# 22 rivers + 8 sewers = 30 sources
_DEFAULT_SOURCE_NAMES = [
    # Rivers (22)
    "EastArakawa",
    "CenterArakawa",
    "WestArakawa",
    "SouthArakawa",
    "FirstSumidagawa",
    "SecondSumidagawa",
    "ThirdSumidagawa",
    "OneEdogawa",
    "TwoEdogawa",
    "ThreeEdogawa",
    "IchiTamagawa",
    "NiTamagawa",
    "SanTamagawa",
    "ATsurumigawa",
    "BTsurumigawa",
    "Mamagawa",
    "Ebigawa",
    "Yorogawa",
    "Obitsugawa",
    "koitogawa",
    "Muratagawa",
    "Hanamigawa",
    # Sewers (8)
    "Shibaura",
    "Sunamachi",
    "Ariake",
    "Kasai",
    "AMorigasaki",
    "BMorigasaki",
    "CMorigasaki",
    "Kisarazu",
]

# Member ID (output directory number) to grouped source name mapping
# for TB-FVCOM goto2023 dye runs (cases 1-20, excluding 0=baseline)
# Each member represents one source or a group of multi-node sources
_DEFAULT_MEMBER_SOURCE_NAMES: dict[int, str] = {
    1: "Arakawa",  # 4 nodes: East, Center, West, South
    2: "Sumida",  # 3 nodes: First, Second, Third
    3: "Edo",  # 3 nodes: One, Two, Three
    4: "Tama",  # 3 nodes: Ichi, Ni, San
    5: "Tsurumi",  # 2 nodes: A, B
    6: "Mama",  # 1 node
    7: "Ebi",  # 1 node
    8: "Yoro",  # 1 node
    9: "Obitsu",  # 1 node
    10: "Koito",  # 1 node
    11: "Murata",  # 1 node
    12: "Hanami",  # 1 node
    13: "Shibaura",  # 1 node (sewer)
    14: "Sunamachi",  # 1 node (sewer)
    15: "Ariake",  # 1 node (sewer)
    16: "Kasai",  # 1 node (sewer)
    17: "Morigasaki",  # 3 nodes: A, B, C (sewer)
    18: "OBC",  # Open boundary condition (OBC_DYE_ON=T)
    19: "Kisarazu",  # 1 node (sewer, node 1440)
    20: "GW",  # Groundwater (GROUNDWATER_DYE_ON=T)
}

# Fixed colors for each source to ensure consistency across all plots
# Rivers use warm/earth tones, sewers use cool tones, OBC/GW uses gray
_DEFAULT_SOURCE_COLORS: dict[str, str] = {
    # Rivers (warm/earth tones)
    "Arakawa": "#1f77b4",  # blue
    "Sumida": "#ff7f0e",  # orange
    "Edo": "#2ca02c",  # green
    "Tama": "#d62728",  # red
    "Tsurumi": "#9467bd",  # purple
    "Mama": "#8c564b",  # brown
    "Ebi": "#e377c2",  # pink
    "Yoro": "#7f7f7f",  # gray
    "Obitsu": "#bcbd22",  # olive
    "Koito": "#17becf",  # cyan
    "Murata": "#aec7e8",  # light blue
    "Hanami": "#ffbb78",  # light orange
    # Sewers (distinct colors)
    "Shibaura": "#98df8a",  # light green
    "Sunamachi": "#ff9896",  # light red
    "Ariake": "#c5b0d5",  # light purple
    "Kasai": "#c49c94",  # light brown
    "Morigasaki": "#f7b6d2",  # light pink
    # Open boundary, Kisarazu sewer, and groundwater
    "OBC": "#c7c7c7",  # light gray
    "Kisarazu": "#dbdb8d",  # khaki (sewer)
    "GW": "#969696",  # medium gray (groundwater)
    # Special
    "Others": "#404040",  # dark gray
    "All Sources": "#333333",  # very dark gray
}

# Full names with type suffix (alternative mapping)
_DEFAULT_MEMBER_SOURCE_NAMES_FULL: dict[int, str] = {
    1: "Arakawa R.",
    2: "Sumida R.",
    3: "Edo R.",
    4: "Tama R.",
    5: "Tsurumi R.",
    6: "Mama R.",
    7: "Ebi R.",
    8: "Yoro R.",
    9: "Obitsu R.",
    10: "Koito R.",
    11: "Murata R.",
    12: "Hanami R.",
    13: "Shibaura S.",
    14: "Sunamachi S.",
    15: "Ariake S.",
    16: "Kasai S.",
    17: "Morigasaki S.",
    18: "OBC",
    19: "Kisarazu S.",
    20: "GW",
}

# Source type classification
_DEFAULT_MEMBER_SOURCE_TYPES: dict[int, str] = {
    1: "River",
    2: "River",
    3: "River",
    4: "River",
    5: "River",
    6: "River",
    7: "River",
    8: "River",
    9: "River",
    10: "River",
    11: "River",
    12: "River",
    13: "Sewer",
    14: "Sewer",
    15: "Sewer",
    16: "Sewer",
    17: "Sewer",
    18: "OBC",
    19: "Sewer",
    20: "GW",
}


# =============================================================================
# Dynamic accessors (use external config if loaded, else defaults)
# =============================================================================


def _get_config_module():
    """Lazily import config module to avoid circular imports."""
    from xfvcom.config import source_config

    return source_config


def _get_source_names_from_config() -> list[str] | None:
    """Get source names from external config if loaded."""
    config_mod = _get_config_module()
    if not config_mod.is_config_loaded():
        return None
    return config_mod.get_all_subsource_names()


def _get_member_mapping_from_config() -> dict[int, str] | None:
    """Get member-to-label mapping from external config if loaded."""
    config_mod = _get_config_module()
    if not config_mod.is_config_loaded():
        return None
    return config_mod.get_member_source_mapping()


def _get_colors_from_config() -> dict[str, str] | None:
    """Get source colors from external config if loaded."""
    config_mod = _get_config_module()
    if not config_mod.is_config_loaded():
        return None
    return config_mod.get_source_colors()


def _get_member_types_from_config() -> dict[int, str] | None:
    """Get member source types from external config if loaded."""
    config_mod = _get_config_module()
    if not config_mod.is_config_loaded():
        return None

    # Build from member config
    members = config_mod.get_member_config()
    if not members:
        return None

    config = config_mod.get_source_config()
    sewer_names = set(config.get("sewers", {}).keys())

    types = {}
    for m in members:
        member_id = m["id"]
        sources = m.get("sources", [])

        if sources == "all":
            types[member_id] = "All"
        elif m.get("obc_dye") and not sources:
            types[member_id] = "OBC"
        elif m.get("groundwater_dye") and not sources:
            types[member_id] = "GW"
        elif sources:
            # Check if any source is a sewer
            if any(s in sewer_names for s in sources):
                types[member_id] = "Sewer"
            else:
                types[member_id] = "River"
        else:
            types[member_id] = "Unknown"

    return types


# =============================================================================
# Public API (backward-compatible module-level variables)
# =============================================================================

# These are kept for backward compatibility but now delegate to dynamic functions
# when external config is loaded.


# For backward compatibility, expose as simple variables.
# Use get_source_names() etc. for dynamic config-aware access.
DEFAULT_SOURCE_NAMES: list[str] = _DEFAULT_SOURCE_NAMES
MEMBER_SOURCE_NAMES: dict[int, str] = _DEFAULT_MEMBER_SOURCE_NAMES
SOURCE_COLORS: dict[str, str] = _DEFAULT_SOURCE_COLORS
MEMBER_SOURCE_NAMES_FULL: dict[int, str] = _DEFAULT_MEMBER_SOURCE_NAMES_FULL
MEMBER_SOURCE_TYPES: dict[int, str] = _DEFAULT_MEMBER_SOURCE_TYPES


# =============================================================================
# Public functions
# =============================================================================


def get_default_source_names() -> list[str]:
    """Get source names (from external config if loaded, else defaults).

    Returns
    -------
    list[str]
        List of source names in order.
    """
    names = _get_source_names_from_config()
    return names if names else _DEFAULT_SOURCE_NAMES


def get_member_source_names() -> dict[int, str]:
    """Get member-to-source name mapping (from config if loaded, else defaults).

    Returns
    -------
    dict[int, str]
        Mapping from member ID to source label.
    """
    mapping = _get_member_mapping_from_config()
    return mapping if mapping else _DEFAULT_MEMBER_SOURCE_NAMES


def get_source_colors_dict() -> dict[str, str]:
    """Get source colors (from external config if loaded, else defaults).

    Returns
    -------
    dict[str, str]
        Mapping from source name to color hex string.
    """
    colors = _get_colors_from_config()
    if colors:
        merged = _DEFAULT_SOURCE_COLORS.copy()
        merged.update(colors)
        return merged
    return _DEFAULT_SOURCE_COLORS


def get_member_source_types() -> dict[int, str]:
    """Get member source types (from config if loaded, else defaults).

    Returns
    -------
    dict[int, str]
        Mapping from member ID to source type ('River', 'Sewer', 'OBC', 'GW').
    """
    types = _get_member_types_from_config()
    return types if types else _DEFAULT_MEMBER_SOURCE_TYPES


def get_source_name(member_id: int, style: str = "short") -> str:
    """Get source name for a member ID.

    This function checks for external configuration first. If loaded,
    uses labels from the config. Otherwise falls back to hardcoded defaults.

    Parameters
    ----------
    member_id : int
        Member ID (1-18 for individual sources, 0 for baseline)
    style : str
        Name style: "short" (default), "full" (with type suffix)

    Returns
    -------
    str
        Source name

    Examples
    --------
    >>> get_source_name(1)
    'Arakawa'
    >>> get_source_name(13, style="full")
    'Shibaura S.'
    """
    if member_id == 0:
        return "All Sources" if style == "short" else "All Sources (Baseline)"

    # Try external config first
    mapping = _get_member_mapping_from_config()
    if mapping and member_id in mapping:
        label = mapping[member_id]
        if style == "full":
            config_mod = _get_config_module()
            display = config_mod.get_display_config()
            full_names = display.get("full_names", {})
            return full_names.get(label, label)
        return label

    # Fall back to hardcoded defaults
    if style == "full":
        return _DEFAULT_MEMBER_SOURCE_NAMES_FULL.get(member_id, f"Source {member_id}")
    return _DEFAULT_MEMBER_SOURCE_NAMES.get(member_id, f"Source {member_id}")


def get_source_color(source_name: str) -> str:
    """Get color for a source name.

    Parameters
    ----------
    source_name : str
        Source name (e.g., 'Arakawa', 'Shibaura')

    Returns
    -------
    str
        Color hex string (e.g., '#1f77b4')
    """
    colors = get_source_colors_dict()
    return colors.get(source_name, "#404040")  # Default to dark gray


def get_river_sewer_boundary() -> int:
    """Get the index boundary between rivers and sewers.

    Sources with index < boundary are rivers, >= boundary are sewers.

    Returns
    -------
    int
        Boundary index (default 22 for 22 rivers).
    """
    config_mod = _get_config_module()
    if config_mod.is_config_loaded():
        config = config_mod.get_source_config()
        # Count river subsources
        count = 0
        for river_data in config.get("rivers", {}).values():
            count += len(river_data.get("subsources", []))
        return count
    return 22  # Default: 22 rivers


def extract_member_node_mapping(
    nml_dir: str | Path,
    basename: str,
    year: int,
    members: list[int],
    source_names: list[str] | None = None,
) -> pd.DataFrame:
    """Extract member-to-node mapping from namelist files.

    Scans a directory for FVCOM namelist files and extracts which nodes
    are active in each ensemble member.

    Parameters
    ----------
    nml_dir : str or Path
        Directory containing namelist files
    basename : str
        Case basename (e.g., 'tb_w18_r16')
    year : int
        Year of the run
    members : list of int
        List of member IDs to process
    source_names : list of str, optional
        Names of sources. If None, uses get_default_source_names().

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'member': int
        - 'source_index': int (0-based)
        - 'source_name': str
        - 'node_id': int (1-based FVCOM node ID)
        - 'strength': float (dye release rate)
        - 'source_type': str ('River' or 'Sewer')

    Examples
    --------
    >>> df = extract_member_node_mapping(
    ...     '/path/to/TB-FVCOM/goto2023/dye_run',
    ...     'tb_w18_r16',
    ...     2021,
    ...     [0, 1, 2, 3]
    ... )
    >>> # Show nodes for member 1
    >>> print(df[df['member'] == 1])
    >>> # Show all nodes for a specific source
    >>> print(df[df['source_name'] == 'EastArakawa'])
    """
    nml_dir = Path(nml_dir)

    if source_names is None:
        source_names = get_default_source_names()

    # Get river/sewer boundary
    river_sewer_boundary = get_river_sewer_boundary()

    # Lazy import to avoid slow package initialization
    from ..io.nml_parser import parse_member_namelist

    records = []

    for member in members:
        # Construct namelist filename
        nml_file = nml_dir / f"{basename}_{year}_{member}_run.nml"

        if not nml_file.exists():
            print(f"Warning: Namelist file not found: {nml_file}")
            continue

        # Parse namelist
        info = parse_member_namelist(nml_file, source_names=source_names)

        # Extract active sources
        for src in info["active_sources"]:
            # Determine source type using dynamic boundary
            source_idx = src["index"]
            source_type = "River" if source_idx < river_sewer_boundary else "Sewer"

            records.append(
                {
                    "member": member,
                    "source_index": source_idx,
                    "source_name": src.get("source_name", f"Source_{source_idx}"),
                    "node_id": src["node_id"],
                    "strength": src["strength"],
                    "source_type": source_type,
                }
            )

    # Create DataFrame
    df = pd.DataFrame(records)

    # Sort by member and source index
    if not df.empty:
        df = df.sort_values(["member", "source_index"]).reset_index(drop=True)

    return df


def get_member_summary(
    nml_dir: str | Path,
    basename: str,
    year: int,
    members: list[int],
    source_names: list[str] | None = None,
) -> pd.DataFrame:
    """Get summary of active sources for each member.

    Parameters
    ----------
    nml_dir : str or Path
        Directory containing namelist files
    basename : str
        Case basename (e.g., 'tb_w18_r16')
    year : int
        Year of the run
    members : list of int
        List of member IDs to process
    source_names : list of str, optional
        Names of sources. If None, uses get_default_source_names().

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with columns:
        - 'member': int
        - 'n_sources': int (number of active sources)
        - 'n_rivers': int (number of active rivers)
        - 'n_sewers': int (number of active sewers)
        - 'total_strength': float (total dye release rate)
        - 'source_names': str (comma-separated list)
        - 'node_ids': str (comma-separated list)

    Examples
    --------
    >>> summary = get_member_summary(
    ...     '/path/to/TB-FVCOM/goto2023/dye_run',
    ...     'tb_w18_r16',
    ...     2021,
    ...     [0, 1, 2, 3]
    ... )
    >>> print(summary)
    """
    # Get full mapping
    df = extract_member_node_mapping(nml_dir, basename, year, members, source_names)

    if df.empty:
        return pd.DataFrame()

    # Group by member and summarize
    summary_records = []

    for member in df["member"].unique():
        member_df = df[df["member"] == member]

        summary_records.append(
            {
                "member": member,
                "n_sources": len(member_df),
                "n_rivers": len(member_df[member_df["source_type"] == "River"]),
                "n_sewers": len(member_df[member_df["source_type"] == "Sewer"]),
                "total_strength": member_df["strength"].sum(),
                "source_names": ", ".join(member_df["source_name"].tolist()),
                "node_ids": ", ".join(member_df["node_id"].astype(str).tolist()),
            }
        )

    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df.sort_values("member").reset_index(drop=True)

    return summary_df


def export_member_mapping(
    df: pd.DataFrame,
    output_path: str | Path,
    format: str = "csv",
) -> None:
    """Export member-to-node mapping to file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from extract_member_node_mapping()
    output_path : str or Path
        Output file path
    format : str, default 'csv'
        Output format: 'csv', 'json', 'markdown', or 'excel'

    Examples
    --------
    >>> df = extract_member_node_mapping(...)
    >>> export_member_mapping(df, 'member_mapping.csv', format='csv')
    >>> export_member_mapping(df, 'member_mapping.md', format='markdown')
    """
    output_path = Path(output_path)

    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient="records", indent=2)
    elif format == "markdown":
        with open(output_path, "w") as f:
            f.write("# Member-Node Mapping\n\n")
            f.write(df.to_markdown(index=False))
    elif format == "excel":
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

    print(f"Exported to: {output_path}")


def get_node_coordinates(
    nc_file: str | Path,
    node_ids: list[int],
    grid_file: str | Path | None = None,
    utm_zone: int | None = None,
) -> pd.DataFrame:
    """Extract node coordinates from FVCOM NetCDF or grid file.

    Parameters
    ----------
    nc_file : str or Path
        Path to FVCOM NetCDF output file (used if grid_file not provided)
    node_ids : list of int
        List of node IDs (1-based) to extract coordinates for
    grid_file : str or Path, optional
        Path to FVCOM grid file (.dat). If provided, coordinates are
        extracted from the grid file instead of nc_file (recommended).
    utm_zone : int, optional
        UTM zone for coordinate conversion (required if using grid_file)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'node_id': int (1-based)
        - 'x': float (x coordinate in meters)
        - 'y': float (y coordinate in meters)
        - 'lon': float (longitude in degrees)
        - 'lat': float (latitude in degrees)

    Notes
    -----
    When using grid_file (recommended), coordinates are guaranteed to match
    the FVCOM mesh definition. When using nc_file, coordinates are read from
    the NetCDF output file which should contain lon/lat variables.

    Examples
    --------
    >>> # Using grid file (recommended)
    >>> coords = get_node_coordinates(
    ...     nc_file=None,  # Not used when grid_file provided
    ...     node_ids=[310, 241, 312],
    ...     grid_file='TokyoBay18_grd.dat',
    ...     utm_zone=54
    ... )

    >>> # Using NetCDF output file
    >>> coords = get_node_coordinates('output.nc', [310, 241, 312])
    >>> print(coords)
    """
    # If grid file is provided, use it (more reliable)
    if grid_file is not None:
        from ..io.input_loader import FvcomInputLoader

        grid_file = Path(grid_file)
        if not grid_file.exists():
            raise FileNotFoundError(f"Grid file not found: {grid_file}")

        if utm_zone is None:
            raise ValueError("utm_zone is required when using grid_file")

        # Load grid
        loader = FvcomInputLoader(
            grid_path=grid_file,
            utm_zone=utm_zone,
            add_dummy_time=False,
            add_dummy_siglay=False,
        )

        grid_ds = loader.ds

        # Extract coordinates
        records = []
        for node_id in node_ids:
            # Convert to 0-based index
            idx = node_id - 1

            if idx < 0 or idx >= len(grid_ds.lon):
                print(f"Warning: Node {node_id} out of range (1-{len(grid_ds.lon)})")
                continue

            records.append(
                {
                    "node_id": node_id,
                    "x": float(grid_ds.x.values[idx]),
                    "y": float(grid_ds.y.values[idx]),
                    "lon": float(grid_ds.lon.values[idx]),
                    "lat": float(grid_ds.lat.values[idx]),
                }
            )

        return pd.DataFrame(records)

    # Otherwise, use NetCDF file
    import netCDF4 as nc

    nc_file = Path(nc_file)
    if not nc_file.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_file}")

    # Open NetCDF file
    ds = nc.Dataset(nc_file)

    try:
        # Extract coordinates
        x = ds.variables["x"][:]
        y = ds.variables["y"][:]
        lon = ds.variables["lon"][:]
        lat = ds.variables["lat"][:]
    except KeyError as e:
        ds.close()
        raise KeyError(
            f"Required variable not found in NetCDF file: {e}\n"
            "Consider using grid_file parameter instead."
        )

    ds.close()

    # Create DataFrame
    records = []
    for node_id in node_ids:
        # Convert to 0-based index
        idx = node_id - 1

        if idx < 0 or idx >= len(x):
            print(f"Warning: Node {node_id} out of range (1-{len(x)})")
            continue

        records.append(
            {
                "node_id": node_id,
                "x": float(x[idx]),
                "y": float(y[idx]),
                "lon": float(lon[idx]),
                "lat": float(lat[idx]),
            }
        )

    return pd.DataFrame(records)

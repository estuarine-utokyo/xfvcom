"""Extract and analyze member-to-node mappings from FVCOM ensemble runs.

This module provides utilities to extract which nodes are active in each
ensemble member by parsing FVCOM namelist files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

from ..io.nml_parser import parse_member_namelist

# Default source names for TB-FVCOM goto2023 dye runs
# 22 rivers + 7 sewers = 29 sources
DEFAULT_SOURCE_NAMES = [
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
    # Sewers (7)
    "Shibaura",
    "Sunamachi",
    "Ariake",
    "Kasai",
    "AMorigasaki",
    "BMorigasaki",
    "CMorigasaki",
]


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
        Names of sources. If None, uses DEFAULT_SOURCE_NAMES.

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
        source_names = DEFAULT_SOURCE_NAMES

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
            # Determine source type
            source_idx = src["index"]
            source_type = "River" if source_idx < 22 else "Sewer"

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
        Names of sources. If None, uses DEFAULT_SOURCE_NAMES.

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

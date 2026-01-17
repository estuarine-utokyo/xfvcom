"""Dynamic source detection from FVCOM namelist and directory structure.

This module provides utilities to automatically detect dye sources from:
1. Directory structure (which members exist)
2. Namelist files (what sources each member represents)
3. River namelist files (source names and node IDs)
4. OBC node list files (open boundary nodes)

Example usage:
    from xfvcom.ensemble_analysis.source_detection import SourceDetector

    detector = SourceDetector(
        nml_dir="/path/to/dye_run",
        output_dir="/path/to/dye_run/output/2021",
        basename="tb_w18_r16",
        year=2021,
    )

    # Get available members
    members = detector.available_members  # [0, 1, 2, ..., 17, 18]

    # Get source info for a member
    info = detector.get_member_source_info(1)
    # {'source_name': 'Arakawa', 'source_type': 'River', 'nodes': [310, 241, 312, 448]}

    # Get all source nodes for plotting
    source_nodes = detector.get_source_nodes()
    # {'Arakawa': [310, 241, 312, 448], 'Sumida': [4, 2, 12], ...}
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

# Lazy import to avoid slow package initialization
# parse_river_namelist is imported inside _load_rivers_namelist()


# Prefixes used for subsource naming (to be stripped for grouping)
SUBSOURCE_PREFIXES = [
    # Directional prefixes
    "East",
    "Center",
    "West",
    "South",
    "North",
    # Ordinal prefixes (English)
    "First",
    "Second",
    "Third",
    "Fourth",
    "Fifth",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    # Ordinal prefixes (Japanese romanization)
    "Ichi",
    "Ni",
    "San",
    "Yon",
    "Go",
    # Letter prefixes
    "A",
    "B",
    "C",
    "D",
    "E",
]


def extract_group_name(subsource_names: list[str]) -> str:
    """Extract common group name from subsource names.

    Parameters
    ----------
    subsource_names : list of str
        List of subsource names (e.g., ['EastArakawa', 'CenterArakawa'])

    Returns
    -------
    str
        Common group name (e.g., 'Arakawa')

    Examples
    --------
    >>> extract_group_name(['EastArakawa', 'CenterArakawa', 'WestArakawa'])
    'Arakawa'
    >>> extract_group_name(['FirstSumidagawa', 'SecondSumidagawa'])
    'Sumidagawa'
    >>> extract_group_name(['Mamagawa'])
    'Mamagawa'
    """
    if not subsource_names:
        return ""

    if len(subsource_names) == 1:
        # Single source - strip any prefix if present
        name = subsource_names[0]
        for prefix in SUBSOURCE_PREFIXES:
            if name.startswith(prefix) and len(name) > len(prefix):
                # Check if remainder starts with uppercase (actual name)
                remainder = name[len(prefix) :]
                if remainder and remainder[0].isupper():
                    return remainder
        return name

    # Multiple subsources - find common suffix
    # Sort by length to find shortest common part
    names = sorted(subsource_names, key=len)

    # Try stripping each prefix and finding common remainder
    stripped_names = []
    for name in names:
        stripped = name
        for prefix in SUBSOURCE_PREFIXES:
            if name.startswith(prefix) and len(name) > len(prefix):
                remainder = name[len(prefix) :]
                if remainder and remainder[0].isupper():
                    stripped = remainder
                    break
        stripped_names.append(stripped)

    # If all stripped names are the same, use that
    if len(set(stripped_names)) == 1:
        return stripped_names[0]

    # Otherwise, find longest common suffix
    reversed_names = [n[::-1] for n in names]
    common_suffix = ""
    for chars in zip(*reversed_names):
        if len(set(chars)) == 1:
            common_suffix += chars[0]
        else:
            break

    if common_suffix:
        return common_suffix[::-1]

    # Fallback: use first name
    return names[0]


def simplify_source_name(name: str) -> str:
    """Simplify a source name by removing common suffixes.

    Parameters
    ----------
    name : str
        Full source name (e.g., 'Sumidagawa', 'Arakawa', 'koitogawa')

    Returns
    -------
    str
        Simplified name with first letter capitalized (e.g., 'Sumida', 'Arakawa', 'Koito')

    Notes
    -----
    Only removes 'gawa' suffix, not 'kawa', to preserve names like 'Arakawa'.
    Result is always title-cased (first letter uppercase) for consistency.
    """
    # Remove 'gawa' suffix only (not 'kawa' to preserve 'Arakawa', 'Edogawa')
    # But keep the full name if removing suffix leaves less than 3 chars
    result = name
    if name.lower().endswith("gawa") and len(name) > 6:
        base = name[:-4]
        # Only simplify if the base name is meaningful (>= 3 chars)
        if len(base) >= 3:
            result = base

    # Capitalize first letter for consistency
    if result:
        result = result[0].upper() + result[1:]
    return result


class SourceDetector:
    """Detect dye sources from FVCOM configuration files.

    This class automatically detects:
    - Available member directories
    - Source names and types for each member
    - Node IDs for each source
    - OBC (open boundary) sources

    Parameters
    ----------
    nml_dir : str or Path
        Directory containing namelist files (e.g., 'goto2023/dye_run')
    output_dir : str or Path
        Directory containing output subdirectories (e.g., 'output/2021')
    basename : str
        Case basename (e.g., 'tb_w18_r16')
    year : int
        Simulation year
    input_dir : str or Path, optional
        Directory containing input files. If None, tries to find it
        relative to nml_dir.

    Attributes
    ----------
    available_members : list of int
        List of member IDs with existing output directories
    source_nodes : dict
        Mapping of source name to list of node IDs
    member_sources : dict
        Mapping of member ID to source info dict
    """

    def __init__(
        self,
        nml_dir: str | Path,
        output_dir: str | Path,
        basename: str,
        year: int,
        input_dir: str | Path | None = None,
    ):
        self.nml_dir = Path(nml_dir)
        self.output_dir = Path(output_dir)
        self.basename = basename
        self.year = year

        # Try to find input directory
        self.input_dir: Path | None = None
        if input_dir is not None:
            self.input_dir = Path(input_dir)
        else:
            # Common patterns
            candidates = [
                self.nml_dir / "input",
                self.nml_dir.parent / "input",
                self.nml_dir / ".." / "input",
            ]
            for candidate in candidates:
                if candidate.exists():
                    self.input_dir = candidate.resolve()
                    break

        # Cache for parsed data
        self._available_members: list[int] | None = None
        self._rivers_df: pd.DataFrame | None = None
        self._obc_nodes: list[int] | None = None
        self._member_sources: dict[int, dict[str, Any]] | None = None
        self._source_nodes: dict[str, list[int]] | None = None

    @property
    def available_members(self) -> list[int]:
        """List of member IDs with existing output directories."""
        if self._available_members is None:
            self._available_members = self._detect_available_members()
        return self._available_members

    @property
    def source_nodes(self) -> dict[str, list[int]]:
        """Mapping of source name to list of node IDs."""
        if self._source_nodes is None:
            self._build_source_mapping()
        # After _build_source_mapping(), _source_nodes is guaranteed to be a dict
        assert self._source_nodes is not None
        return self._source_nodes

    @property
    def member_sources(self) -> dict[int, dict[str, Any]]:
        """Mapping of member ID to source info."""
        if self._member_sources is None:
            self._build_source_mapping()
        # After _build_source_mapping(), _member_sources is guaranteed to be a dict
        assert self._member_sources is not None
        return self._member_sources

    def _detect_available_members(self) -> list[int]:
        """Scan output directory for available members.

        Supports two directory structures:
        1. Flat structure: output/{case}/{basename}_{member}_0001.nc
        2. Subdirectory structure: output/{year}/{member}/{basename}_{year}_{member}_0001.nc
        """
        members: set[int] = set()
        if not self.output_dir.exists():
            return []

        # First, try flat directory structure (case-based)
        # Pattern: {basename}_{member}_*.nc (excluding restart files)
        flat_pattern = f"{self.basename}_*_*.nc"
        for nc_file in self.output_dir.glob(flat_pattern):
            if "restart" in nc_file.name:
                continue
            # Extract member from filename: tb_w18_r16_0_0001.nc -> 0
            match = re.match(rf"{re.escape(self.basename)}_(\d+)_", nc_file.name)
            if match:
                members.add(int(match.group(1)))

        # If no flat files found, try subdirectory structure (legacy)
        if not members:
            for item in self.output_dir.iterdir():
                if item.is_dir() and item.name.isdigit():
                    # Check if directory has NetCDF files
                    nc_files = list(item.glob("*.nc"))
                    if nc_files:
                        members.add(int(item.name))

        return sorted(members)

    def _parse_run_namelist(self, member: int) -> dict[str, Any]:
        """Parse a member's run namelist file.

        Returns dict with keys:
        - river_info_file: path to rivers namelist
        - obc_node_list_file: path to OBC node list
        - dye_source_term: list of dye source values
        - dye_source_term_obc: list of OBC dye values (if present)
        - m_specify: list of node IDs for dye sources

        Supports two naming patterns:
        1. Case-based: {basename}_{member}_run.nml (new)
        2. Year-based: {basename}_{year}_{member}_run.nml (legacy)
        """
        # Try case-based naming first (new structure)
        nml_file = self.nml_dir / f"{self.basename}_{member}_run.nml"
        if not nml_file.exists():
            # Fall back to year-based naming (legacy)
            nml_file = self.nml_dir / f"{self.basename}_{self.year}_{member}_run.nml"
        if not nml_file.exists():
            return {}

        result: dict[str, Any] = {}

        with open(nml_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract RIVER_INFO_FILE
        match = re.search(r"RIVER_INFO_FILE\s*=\s*['\"]([^'\"]+)['\"]", content, re.I)
        if match:
            result["river_info_file"] = match.group(1)

        # Extract OBC_NODE_LIST_FILE
        match = re.search(
            r"OBC_NODE_LIST_FILE\s*=\s*['\"]([^'\"]+)['\"]", content, re.I
        )
        if match:
            result["obc_node_list_file"] = match.group(1)

        # Extract M_SPECIFY (node IDs)
        match = re.search(r"M_SPECIFY\s*=\s*([0-9,\s]+)", content, re.I)
        if match:
            int_values = [
                int(v.strip()) for v in match.group(1).split(",") if v.strip()
            ]
            result["m_specify"] = int_values

        # Extract DYE_SOURCE_TERM
        match = re.search(r"DYE_SOURCE_TERM\s*=\s*([0-9.,\s]+)", content, re.I)
        if match:
            float_values = [
                float(v.strip()) for v in match.group(1).split(",") if v.strip()
            ]
            result["dye_source_term"] = float_values

        # Extract DYE_SOURCE_TERM_OBC
        match = re.search(r"DYE_SOURCE_TERM_OBC\s*=\s*([0-9.,\s]+)", content, re.I)
        if match:
            obc_values = [
                float(v.strip()) for v in match.group(1).split(",") if v.strip()
            ]
            result["dye_source_term_obc"] = obc_values

        # Extract OBC_DYE_ON (check if OBC dye is actually enabled)
        match = re.search(r"OBC_DYE_ON\s*=\s*([TFtf])", content, re.I)
        if match:
            result["obc_dye_on"] = match.group(1).upper() == "T"

        # Extract GROUNDWATER_DYE_ON
        match = re.search(r"GROUNDWATER_DYE_ON\s*=\s*([TFtf])", content, re.I)
        if match:
            result["groundwater_dye_on"] = match.group(1).upper() == "T"

        # Extract GROUNDWATER_ON (main groundwater switch)
        match = re.search(r"GROUNDWATER_ON\s*=\s*([TFtf])", content, re.I)
        if match:
            result["groundwater_on"] = match.group(1).upper() == "T"

        return result

    def _load_rivers_namelist(self, river_info_file: str) -> pd.DataFrame:
        """Load and parse rivers namelist file."""
        if self._rivers_df is not None:
            return self._rivers_df

        # Lazy import to avoid slow package initialization
        # Use absolute import for compatibility with direct module loading
        import importlib.util
        from pathlib import Path

        module_path = Path(__file__).parent.parent / "io" / "river_nml.py"
        spec = importlib.util.spec_from_file_location("river_nml", str(module_path))
        if spec is None or spec.loader is None:
            return pd.DataFrame()
        river_nml_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(river_nml_module)
        parse_river_namelist = river_nml_module.parse_river_namelist

        # Try to find the file
        candidates = [
            self.input_dir / river_info_file if self.input_dir else None,
            self.nml_dir / "input" / river_info_file,
            self.nml_dir.parent / "input" / river_info_file,
        ]

        for candidate in candidates:
            if candidate and candidate.exists():
                self._rivers_df = parse_river_namelist(candidate, to_zero_based=False)
                return self._rivers_df

        # Return empty DataFrame if not found
        return pd.DataFrame()

    def _load_obc_nodes(self, obc_file: str) -> list[int]:
        """Load OBC node IDs from node list file."""
        if self._obc_nodes is not None:
            return self._obc_nodes

        # Try to find the file
        candidates = [
            self.input_dir / obc_file if self.input_dir else None,
            self.nml_dir / "input" / obc_file,
            self.nml_dir.parent / "input" / obc_file,
        ]

        for candidate in candidates:
            if candidate and candidate.exists():
                nodes = []
                with open(candidate, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("OBC") or not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            # Column 2 is the node ID
                            nodes.append(int(parts[1]))
                self._obc_nodes = nodes
                return self._obc_nodes

        return []

    def _build_source_mapping(self) -> None:
        """Build source mapping from namelist files."""
        self._member_sources = {}
        self._source_nodes = {}

        # Use member 0 (baseline) to get river info
        baseline_info = self._parse_run_namelist(0)
        if not baseline_info:
            # Try first available member
            for m in self.available_members:
                baseline_info = self._parse_run_namelist(m)
                if baseline_info:
                    break

        if not baseline_info:
            return

        # Load rivers namelist
        rivers_df = pd.DataFrame()
        if "river_info_file" in baseline_info:
            rivers_df = self._load_rivers_namelist(baseline_info["river_info_file"])

        # Load OBC nodes
        obc_nodes = []
        if "obc_node_list_file" in baseline_info:
            obc_nodes = self._load_obc_nodes(baseline_info["obc_node_list_file"])

        # Build node-to-name mapping from rivers
        node_to_name: dict[int, str] = {}
        if not rivers_df.empty:
            for _, row in rivers_df.iterrows():
                node_id = int(row["grid_location"])
                name = str(row["name"])
                node_to_name[node_id] = name

        # Process each available member
        for member in self.available_members:
            member_info = self._parse_run_namelist(member)
            if not member_info:
                continue

            # Determine source type and collect active nodes
            active_nodes = []
            active_names = []
            is_obc = False
            is_groundwater = False
            is_baseline = False

            # Check for OBC dye (must have OBC_DYE_ON = T)
            if member_info.get("obc_dye_on", False):
                if "dye_source_term_obc" in member_info:
                    obc_values = member_info["dye_source_term_obc"]
                    if any(v > 0 for v in obc_values):
                        is_obc = True

            # Check for groundwater dye (must have GROUNDWATER_ON = T and GROUNDWATER_DYE_ON = T)
            if member_info.get("groundwater_on", False) and member_info.get(
                "groundwater_dye_on", False
            ):
                is_groundwater = True

            # Check for river/sewer dye
            if "dye_source_term" in member_info and "m_specify" in member_info:
                dye_values = member_info["dye_source_term"]
                m_specify = member_info["m_specify"]

                # Find active sources (non-zero dye values)
                for i, (node, value) in enumerate(zip(m_specify, dye_values)):
                    if value > 0:
                        active_nodes.append(node)
                        if node in node_to_name:
                            active_names.append(node_to_name[node])

                # Check if all sources are active (baseline)
                if len(active_nodes) == len(m_specify) and all(
                    v > 0 for v in dye_values
                ):
                    is_baseline = True

            # Determine source info
            if is_groundwater and not active_nodes and not is_obc:
                # Pure groundwater member
                source_name = "Groundwater"
                source_type = "Groundwater"
                nodes = []  # Groundwater doesn't have specific nodes
            elif is_obc and not active_nodes:
                # Pure OBC member
                source_name = "OBC"
                source_type = "OBC"
                nodes = obc_nodes
            elif is_baseline:
                # Baseline member (all sources)
                source_name = "All"
                source_type = "Baseline"
                nodes = active_nodes
            elif active_names:
                # River/sewer member
                source_name = extract_group_name(active_names)
                # Simplify name (remove 'gawa' suffix)
                source_name = simplify_source_name(source_name)
                # Determine type based on source name patterns
                sewer_names = ["Shibaura", "Sunamachi", "Ariake", "Kasai", "Morigasaki"]
                if any(s.lower() in source_name.lower() for s in sewer_names):
                    source_type = "Sewer"
                else:
                    source_type = "River"
                nodes = active_nodes
            else:
                # Unknown
                source_name = f"Member{member}"
                source_type = "Unknown"
                nodes = active_nodes

            self._member_sources[member] = {
                "source_name": source_name,
                "source_type": source_type,
                "nodes": nodes,
                "is_baseline": is_baseline,
                "is_obc": is_obc,
                "is_groundwater": is_groundwater,
            }

            # Add to source_nodes mapping (skip baseline)
            if not is_baseline and source_name not in self._source_nodes:
                self._source_nodes[source_name] = nodes

    def get_member_source_info(self, member: int) -> dict[str, Any]:
        """Get source info for a specific member.

        Parameters
        ----------
        member : int
            Member ID

        Returns
        -------
        dict
            Source info with keys: source_name, source_type, nodes, is_baseline, is_obc
        """
        if member not in self.member_sources:
            return {
                "source_name": f"Member{member}",
                "source_type": "Unknown",
                "nodes": [],
                "is_baseline": False,
                "is_obc": False,
                "is_groundwater": False,
            }
        return self.member_sources[member]

    def get_source_name(self, member: int) -> str:
        """Get source name for a member.

        Parameters
        ----------
        member : int
            Member ID

        Returns
        -------
        str
            Source name
        """
        info = self.get_member_source_info(member)
        return info["source_name"]

    def get_source_nodes_for_member(self, member: int) -> list[int]:
        """Get node IDs for a member's source.

        Parameters
        ----------
        member : int
            Member ID

        Returns
        -------
        list of int
            Node IDs
        """
        info = self.get_member_source_info(member)
        return info["nodes"]

    def get_non_baseline_members(self) -> list[int]:
        """Get list of non-baseline member IDs."""
        return [
            m
            for m in self.available_members
            if not self.member_sources.get(m, {}).get("is_baseline", False)
        ]

    def print_summary(self) -> None:
        """Print summary of detected sources."""
        print(f"Output directory: {self.output_dir}")
        print(f"Available members: {self.available_members}")
        print()
        print("Member -> Source mapping:")
        print("-" * 60)
        for member in self.available_members:
            info = self.get_member_source_info(member)
            nodes_str = ", ".join(str(n) for n in info["nodes"][:5])
            if len(info["nodes"]) > 5:
                nodes_str += f", ... ({len(info['nodes'])} total)"
            print(
                f"  {member:2d}: {info['source_name']:15s} ({info['source_type']:8s}) nodes=[{nodes_str}]"
            )

"""Parser for FVCOM namelist (.nml) files.

This module provides utilities for parsing FVCOM namelist files to extract
configuration information, particularly for dye release configurations.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class NamelistParser:
    """Parser for FVCOM namelist files.

    Extracts configuration parameters from FVCOM namelist (.nml) files,
    with special support for dye release configurations.

    Examples
    --------
    >>> parser = NamelistParser('run.nml')
    >>> dye_config = parser.parse_dye_release()
    >>> print(dye_config['m_specify'])  # Node IDs
    >>> print(dye_config['dye_source_term'])  # Source strengths
    """

    def __init__(self, filepath: str | Path):
        """Initialize parser with namelist file path.

        Parameters
        ----------
        filepath : str or Path
            Path to FVCOM namelist (.nml) file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Namelist file not found: {filepath}")

        self._content = self.filepath.read_text()

    def parse_dye_release(self) -> dict[str, Any]:
        """Parse dye release configuration from namelist.

        Extracts parameters from &NML_DYE_RELEASE section including:
        - M_SPECIFY: Node IDs where dye is released
        - DYE_SOURCE_TERM: Strength of dye release at each node
        - DYE_ON: Whether dye is enabled
        - DYE_RELEASE_START: Start time
        - DYE_RELEASE_STOP: Stop time
        - MSPE_DYE: Number of dye sources

        Returns
        -------
        dict
            Dictionary with dye configuration parameters:
            - 'dye_on': bool
            - 'm_specify': list of int (node IDs)
            - 'dye_source_term': list of float (source strengths)
            - 'dye_release_start': str
            - 'dye_release_stop': str
            - 'mspe_dye': int (number of sources)

        Examples
        --------
        >>> parser = NamelistParser('member_1_run.nml')
        >>> config = parser.parse_dye_release()
        >>> active_nodes = [
        ...     node for node, strength in
        ...     zip(config['m_specify'], config['dye_source_term'])
        ...     if strength > 0
        ... ]
        """
        # Find &NML_DYE_RELEASE section (match until / at start of line)
        pattern = r'&NML_DYE_RELEASE\s+(.*?)^\s*/'
        match = re.search(pattern, self._content, re.DOTALL | re.IGNORECASE | re.MULTILINE)

        if not match:
            raise ValueError(f"&NML_DYE_RELEASE section not found in {self.filepath}")

        section_content = match.group(1)

        # Parse individual parameters
        config = {}

        # DYE_ON (boolean)
        config['dye_on'] = self._parse_boolean(section_content, 'DYE_ON')

        # M_SPECIFY (node IDs)
        config['m_specify'] = self._parse_int_array(section_content, 'M_SPECIFY')

        # DYE_SOURCE_TERM (source strengths)
        config['dye_source_term'] = self._parse_float_array(
            section_content, 'DYE_SOURCE_TERM'
        )

        # DYE_RELEASE_START (time string)
        config['dye_release_start'] = self._parse_string(
            section_content, 'DYE_RELEASE_START'
        )

        # DYE_RELEASE_STOP (time string)
        config['dye_release_stop'] = self._parse_string(
            section_content, 'DYE_RELEASE_STOP'
        )

        # MSPE_DYE (number of sources)
        config['mspe_dye'] = self._parse_int(section_content, 'MSPE_DYE')

        return config

    def get_active_sources(self) -> list[dict[str, Any]]:
        """Get list of active dye sources with node IDs and strengths.

        Returns only sources where DYE_SOURCE_TERM > 0.

        Returns
        -------
        list of dict
            List of active sources, each containing:
            - 'index': int (0-based index in M_SPECIFY)
            - 'node_id': int (FVCOM node ID, 1-based)
            - 'strength': float (dye release rate)

        Examples
        --------
        >>> parser = NamelistParser('member_1_run.nml')
        >>> active = parser.get_active_sources()
        >>> for src in active:
        ...     print(f"Node {src['node_id']}: strength={src['strength']}")
        """
        config = self.parse_dye_release()

        if not config['dye_on']:
            return []

        active_sources = []
        for i, (node_id, strength) in enumerate(
            zip(config['m_specify'], config['dye_source_term'])
        ):
            if strength > 0:
                active_sources.append({
                    'index': i,
                    'node_id': node_id,
                    'strength': strength,
                })

        return active_sources

    @staticmethod
    def _parse_boolean(content: str, param_name: str) -> bool:
        """Parse boolean parameter from namelist content."""
        pattern = rf'{param_name}\s*=\s*([TF]),'
        match = re.search(pattern, content, re.IGNORECASE)
        if not match:
            raise ValueError(f"Parameter {param_name} not found")
        return match.group(1).upper() == 'T'

    @staticmethod
    def _parse_int(content: str, param_name: str) -> int:
        """Parse integer parameter from namelist content."""
        pattern = rf'{param_name}\s*=\s*(\d+),'
        match = re.search(pattern, content, re.IGNORECASE)
        if not match:
            raise ValueError(f"Parameter {param_name} not found")
        return int(match.group(1))

    @staticmethod
    def _parse_string(content: str, param_name: str) -> str:
        """Parse string parameter from namelist content."""
        pattern = rf"{param_name}\s*=\s*'([^']+)',"
        match = re.search(pattern, content, re.IGNORECASE)
        if not match:
            raise ValueError(f"Parameter {param_name} not found")
        return match.group(1)

    @staticmethod
    def _parse_int_array(content: str, param_name: str) -> list[int]:
        """Parse integer array parameter from namelist content.

        Handles multi-line arrays and comments.
        """
        # Find parameter line - stop at next parameter (word char at line start) or end of section
        pattern = rf'{param_name}\s*=\s*(.*?)(?=\n\s*[A-Z_][A-Z_0-9]*\s*=|\n\s*/|\Z)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if not match:
            raise ValueError(f"Parameter {param_name} not found")

        value_str = match.group(1)

        # Remove comments
        value_str = re.sub(r'!.*', '', value_str)

        # Extract numbers
        numbers = re.findall(r'\d+', value_str)
        return [int(n) for n in numbers]

    @staticmethod
    def _parse_float_array(content: str, param_name: str) -> list[float]:
        """Parse float array parameter from namelist content.

        Handles multi-line arrays and comments.
        """
        # Find parameter line - stop at next parameter (word char at line start) or end of section
        pattern = rf'{param_name}\s*=\s*(.*?)(?=\n\s*[A-Z_][A-Z_0-9]*\s*=|\n\s*/|\Z)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if not match:
            raise ValueError(f"Parameter {param_name} not found")

        value_str = match.group(1)

        # Remove comments
        value_str = re.sub(r'!.*', '', value_str)

        # Extract numbers (including decimals and scientific notation)
        numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', value_str)
        return [float(n) for n in numbers]


def parse_member_namelist(
    filepath: str | Path,
    source_names: list[str] | None = None
) -> dict[str, Any]:
    """Parse a member namelist file and extract dye configuration.

    Convenience function that creates a parser and extracts relevant info.

    Parameters
    ----------
    filepath : str or Path
        Path to namelist file
    source_names : list of str, optional
        Names of sources corresponding to M_SPECIFY indices.
        If provided, adds 'source_name' to each active source.

    Returns
    -------
    dict
        Dictionary with:
        - 'filepath': Path to namelist file
        - 'config': Full dye release configuration
        - 'active_sources': List of active sources with node IDs

    Examples
    --------
    >>> source_names = ['EastArakawa', 'CenterArakawa', ...]
    >>> info = parse_member_namelist('member_1_run.nml', source_names)
    >>> for src in info['active_sources']:
    ...     print(f"{src['source_name']}: node {src['node_id']}")
    """
    parser = NamelistParser(filepath)
    config = parser.parse_dye_release()
    active_sources = parser.get_active_sources()

    # Add source names if provided
    if source_names is not None:
        for src in active_sources:
            idx = src['index']
            if idx < len(source_names):
                src['source_name'] = source_names[idx]

    return {
        'filepath': Path(filepath),
        'config': config,
        'active_sources': active_sources,
    }

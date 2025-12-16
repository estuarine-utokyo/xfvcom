# -*- coding: utf-8 -*-
"""
Data directory for xfvcom package.

This directory stores cached data files such as:
- gwo_correlations_cache.yaml: Pre-computed station correlations for gap filling
"""

from pathlib import Path

DATA_DIR = Path(__file__).parent


def get_data_path(filename: str) -> Path:
    """Get path to a data file in this directory."""
    return DATA_DIR / filename

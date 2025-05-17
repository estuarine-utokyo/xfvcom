"""
Utility helpers for reading external time‑series tables that will be used as forcing
inputs (e.g., river discharge, temperature) before conversion into FVCOM‑compatible
NetCDF.

This module is **stand‑alone**: it does *not* depend on the rest of xfvcom and can be
imported safely by lightweight scripts such as generator CLI tools.  Keeping this
file independent prevents circular‑import issues when higher‑level packages in
xfvcom also need to call back into the generators.

All public symbols are deliberately narrow; at present only
`load_timeseries_table` is intended for external use.

Copyright (c) 2025 University of Tokyo.  All rights reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Iterable, List, Sequence

import chardet
import pandas as pd

__all__: List[str] = ["load_timeseries_table"]

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Ordered list of encodings we attempt when opening a text file.  The first that
# succeeds without a decoding error wins and is recorded in the resulting
# DataFrame's ``attrs`` for provenance.
_FALLBACK_ENCODINGS: Final[tuple[str, ...]] = (
    "utf-8",
    "utf-8-sig",  # Handles BOM if present
    "cp932",  # Common alias for Shift‑JIS on Windows
)

# Default set of strings / numbers that should be interpreted as missing data.
_DEFAULT_NA_VALUES: Final[List[str | float]] = [
    "",
    "NaN",
    "nan",
    "NAN",
    "null",
    "NULL",
    "None",
    "none",
    "-9999",
    "-9999.0",
    "999.99",
    "999.990",  # Support user‑specified sentinel values (real numbers)
]


# ----------------------------------------------------------------------
# Encoding-detection helper
# ----------------------------------------------------------------------
def _detect_encoding(path: Path, nbytes: int = 10_000) -> str:
    """
    Detect text encoding from the first *nbytes* of *path* using chardet.

    Parameters
    ----------
    path : Path
        Target text file.
    nbytes : int, optional
        Number of bytes to inspect. Default is 10 kB.

    Returns
    -------
    str
        Best-guess encoding name (e.g., 'utf-8', 'utf-8-sig', 'cp932').

    Notes
    -----
    If chardet fails to suggest an encoding, the function falls back to
    'utf-8'.  Shift-JIS family names are normalised to 'cp932' so that
    pandas can handle them uniformly on any platform.
    """
    with path.open("rb") as fh:
        raw = fh.read(nbytes)

    result = chardet.detect(raw)
    enc = (result.get("encoding") or "utf-8").lower()

    # Normalise common Shift-JIS aliases
    if enc in {"shift_jis", "shift-jis", "sjis", "windows-31j"}:
        enc = "cp932"

    return enc


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------


def load_timeseries_table(
    path: str | Path,
    *,
    na_values: Iterable[str | float] | None = None,
    strip_inline_comments: bool = True,
) -> pd.DataFrame:
    """Load a CSV or TSV time‑series table with robust encoding / NA handling.

    Parameters
    ----------
    path
        Input file path.
    na_values
        Additional markers that should be recognised as *NA* beyond the default
        set.  Use this to inject project‑specific sentinel values.

    Returns
    -------
    pandas.DataFrame
        Parsed table.  Column ``time`` (required) is converted to
        :class:`~pandas.DatetimeIndex` in UTC internally; original timezone is
        assumed to be *JST (UTC+9)* as per project convention.  The detected file
        encoding is stored under ``df.attrs["encoding"]`` for downstream
        provenance / reproducibility.

    Raises
    ------
    ValueError
        If the filename does not end with ``.csv`` or ``.tsv``;
        if the ``time`` column is missing; or if none of the fallback encodings
        can decode the file.
    """
    path = Path(path)
    if path.suffix.lower() not in (".csv", ".tsv"):
        raise ValueError(
            f"Unsupported file extension: '{path.suffix}'. Expected .csv or .tsv."
        )

    delimiter: str = "," if path.suffix.lower() == ".csv" else "\t"

    # Compose the effective NA list.
    na_markers: list[str | float] = list(_DEFAULT_NA_VALUES)
    if na_values is not None:
        na_markers.extend(na_values)

    # ------------------------------------------------------------------
    #  Attempt to read the file with the detected encoding
    # ------------------------------------------------------------------
    enc = _detect_encoding(path)
    try:
        with path.open("r", encoding=enc, newline="") as fp:
            # pandas-stubs expects Sequence[str]; cast float→str for type-checker
            na_values_text: Sequence[str] = [str(v) for v in na_markers]
            df: pd.DataFrame = pd.read_csv(
                fp,
                sep=delimiter,
                na_values=na_values_text,
                keep_default_na=False,
                parse_dates=["time"],
            )
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Failed to decode '{path}' with detected encoding '{enc}'."
        ) from exc

    # Sanity‑check mandatory column.
    if "time" not in df.columns:
        raise ValueError("Column 'time' not found in the input table.")

    # Standardise timezone: input is JST (UTC+9) → convert to UTC naive.
    df["time"] = (
        df["time"].dt.tz_localize("Asia/Tokyo", ambiguous="NaT").dt.tz_convert("UTC")
    )

    # Store provenance info.
    df.attrs["encoding"] = enc
    df.attrs["delimiter"] = "," if delimiter == "," else "TAB"

    # ---------------------------------------------------------------
    #  Strip inline comments like "   # some remark" (optional)
    # ---------------------------------------------------------------
    if strip_inline_comments:
        # Work only on string/object columns
        for col in df.select_dtypes(include="object").columns:
            # Remove "whitespace + # + everything" at end of cell
            df[col] = (
                df[col]
                .str.replace(r"\s*#.*$", "", regex=True)
                .replace({"": pd.NA})  # empty string -> NA
            )
            # Try to convert the entire column to numeric.
            # If conversion fails (mixed strings, etc.), keep the original.
            try:
                df[col] = pd.to_numeric(df[col])  # errors='raise' is default
            except (ValueError, TypeError):
                # At least one value cannot be parsed as number -> leave as-is
                pass

    return df

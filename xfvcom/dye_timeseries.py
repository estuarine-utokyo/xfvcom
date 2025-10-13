"""Multi-year FVCOM dye time-series extractor and analyzer.

This module provides robust tools for extracting, aggregating, and analyzing
dye concentration time series from multi-year, multi-member FVCOM ensemble runs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import xarray as xr

# Configure logging
logger = logging.getLogger(__name__)


def decode_fvcom_time(ds: xr.Dataset, time_key: str = "time") -> xr.Dataset:
    """Manually decode FVCOM time from MJD to datetime.

    FVCOM uses Modified Julian Day (MJD) format which xarray can't decode automatically.
    This function converts the time coordinate from MJD to proper datetime64 objects.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with undecoded time coordinate
    time_key : str
        Name of the time coordinate (default: "time")

    Returns
    -------
    xr.Dataset
        Dataset with decoded time coordinate
    """
    if time_key not in ds.coords:
        logger.warning(f"Time coordinate '{time_key}' not found in dataset")
        return ds

    # Get time values (should be in MJD format)
    time_values = ds[time_key].values

    # Get time attributes to determine the reference date
    time_attrs = ds[time_key].attrs
    units = time_attrs.get('units', '')

    if 'since' in units:
        # Parse units string: "days since 1858-11-17 00:00:00"
        try:
            # Extract reference date from units
            parts = units.split('since')
            if len(parts) == 2:
                ref_date_str = parts[1].strip()
                ref_date = pd.Timestamp(ref_date_str)

                # Convert to datetime
                if 'days' in parts[0]:
                    time_decoded = pd.to_datetime(time_values, unit='D', origin=ref_date)
                elif 'hours' in parts[0]:
                    time_decoded = pd.to_datetime(time_values, unit='h', origin=ref_date)
                elif 'seconds' in parts[0]:
                    time_decoded = pd.to_datetime(time_values, unit='s', origin=ref_date)
                elif 'milliseconds' in parts[0] or 'msec' in parts[0]:
                    time_decoded = pd.to_datetime(time_values, unit='ms', origin=ref_date)
                else:
                    logger.warning(f"Unknown time unit in '{units}', keeping original values")
                    return ds

                # Update dataset with decoded time
                ds = ds.assign_coords({time_key: time_decoded})
                logger.debug(f"Decoded time from {time_values[0]} to {time_decoded[0]}")
            else:
                logger.warning(f"Could not parse units '{units}', keeping original values")
        except Exception as e:
            logger.warning(f"Failed to decode time: {e}, keeping original values")
    else:
        logger.warning(f"Time units '{units}' don't contain 'since', keeping original values")

    return ds


@dataclass
class DyeCase:
    """Configuration for a dye simulation case.

    Attributes:
        basename: Base name of output files (e.g., "tb_w18_r16")
        years: List of years to process (e.g., [2021, 2022])
        members: List of member IDs to process (e.g., [0, 1, 2, ..., 18])
        var_name: FVCOM variable name for dye concentration
        sigma_key: Vertical coordinate name ("siglay" or "siglev")
        time_key: Time coordinate name
    """

    basename: str
    years: list[int]
    members: list[int]
    var_name: str = "DYE"
    sigma_key: str = "siglay"
    time_key: str = "time"


@dataclass
class Selection:
    """Spatial and vertical selection for dye extraction.

    Attributes:
        nodes: Node index or list of node indices (0-based)
        sigmas: Sigma layer index or list of indices (0-based)
    """

    nodes: list[int] | int
    sigmas: list[int] | int

    def __post_init__(self) -> None:
        """Normalize inputs to lists."""
        if isinstance(self.nodes, int):
            self.nodes = [self.nodes]
        if isinstance(self.sigmas, int):
            self.sigmas = [self.sigmas]


@dataclass
class Paths:
    """Path configuration for locating FVCOM outputs.

    Attributes:
        tb_fvcom_dir: Root directory of TB-FVCOM repository
        output_root_rel: Relative path from tb_fvcom_dir to output directory
    """

    tb_fvcom_dir: Path
    output_root_rel: Path = Path("goto2023/dye_run/output")

    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        self.tb_fvcom_dir = Path(self.tb_fvcom_dir)
        self.output_root_rel = Path(self.output_root_rel)


@dataclass
class NegPolicy:
    """Policy for handling negative dye values (undershoots).

    Attributes:
        mode: "keep" to preserve negatives, "clip_zero" to set them to 0
    """

    mode: Literal["keep", "clip_zero"] = "keep"


@dataclass
class AlignPolicy:
    """Policy for time alignment across years and members.

    Attributes:
        mode: Time alignment strategy:
            - "native_intersection": Strict intersection of exact timestamps
            - "same_calendar": Align by (month, day, hour) across years
            - "climatology": Average over years by (month, day, hour)
    """

    mode: Literal["native_intersection", "same_calendar", "climatology"] = "native_intersection"


def collect_member_files(
    paths: Paths,
    case: DyeCase
) -> dict[tuple[int, int], list[Path]]:
    """Collect NetCDF files for all requested (year, member) combinations.

    Args:
        paths: Path configuration
        case: Dye case configuration

    Returns:
        Dictionary mapping (year, member) to list of file paths

    Raises:
        FileNotFoundError: If no files found for any requested (year, member)
    """
    member_map: dict[tuple[int, int], list[Path]] = {}
    missing_pairs: list[tuple[int, int]] = []

    for year in case.years:
        for member in case.members:
            # Construct expected directory and glob pattern
            member_dir = paths.tb_fvcom_dir / paths.output_root_rel / str(year) / str(member)
            pattern = f"{case.basename}_{year}_{member}_*.nc"

            # Exclude restart files
            files = [
                f for f in sorted(member_dir.glob(pattern))
                if "restart" not in f.name
            ]

            if not files:
                missing_pairs.append((year, member))
                logger.warning(
                    f"No files found for (year={year}, member={member}). "
                    f"Expected pattern: {member_dir / pattern}"
                )
            else:
                member_map[(year, member)] = files
                logger.info(
                    f"Found {len(files)} file(s) for (year={year}, member={member})"
                )

    if missing_pairs:
        error_msg = (
            f"No files found for {len(missing_pairs)} (year, member) pair(s):\n"
            f"  Missing: {missing_pairs}\n"
            f"  Expected location: {paths.tb_fvcom_dir / paths.output_root_rel}/<YEAR>/<MEMBER>/\n"
            f"  Expected pattern: {case.basename}_<YEAR>_<MEMBER>_*.nc\n"
            f"  Hint: Check that outputs exist and paths are correct."
        )
        raise FileNotFoundError(error_msg)

    logger.info(
        f"Successfully collected files for {len(member_map)} (year, member) pairs"
    )
    return member_map


def load_member_series(
    files: list[Path],
    case: DyeCase,
    sel: Selection,
    neg: NegPolicy,
    year: int,
    member: int,
) -> xr.DataArray:
    """Load and process dye time series for one (year, member).

    Args:
        files: List of NetCDF files for this member
        case: Dye case configuration
        sel: Spatial/vertical selection
        neg: Negative value handling policy
        year: Year identifier
        member: Member identifier

    Returns:
        DataArray with DYE(time) after averaging over selected nodes/sigmas

    Raises:
        ValueError: If NaN values detected in dye data with detailed diagnostics
    """
    logger.info(
        f"Loading (year={year}, member={member}): {len(files)} file(s), "
        f"nodes={sel.nodes}, sigmas={sel.sigmas}"
    )

    # Open multi-file dataset
    # Note: FVCOM uses non-standard time encoding (e.g., "msec since 00:00:00")
    # which causes xarray decode errors. We use decode_times=False to avoid this.
    if len(files) == 1:
        # Single file - open directly
        ds = xr.open_dataset(files[0], decode_times=False)
    else:
        # Multiple files - concatenate along time dimension
        ds = xr.open_mfdataset(
            files,
            combine="nested",
            concat_dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
            parallel=False,
            decode_times=False,  # FVCOM time encoding is non-standard
        )

    # Manually decode FVCOM time from MJD to datetime
    ds = decode_fvcom_time(ds, time_key=case.time_key)

    # Extract DYE variable
    if case.var_name not in ds:
        raise ValueError(
            f"Variable '{case.var_name}' not found in dataset. "
            f"Available variables: {list(ds.data_vars)}"
        )

    dye_full = ds[case.var_name]

    # Expected dimensions: (time, sigma, node) or similar
    # FVCOM typically uses (time, siglay, node)
    if dye_full.dims != (case.time_key, case.sigma_key, "node"):
        logger.warning(
            f"Unexpected dimensions: {dye_full.dims}. "
            f"Expected: ({case.time_key}, {case.sigma_key}, 'node')"
        )

    # Select nodes and sigma layers
    dye_sel = dye_full.isel(node=sel.nodes, **{case.sigma_key: sel.sigmas})

    # Check for NaNs - STRICT enforcement
    nan_mask = dye_sel.isnull()
    if nan_mask.any():
        # Collect detailed diagnostics for first 10 NaN occurrences
        nan_coords = np.argwhere(nan_mask.values)
        diagnostics = []

        for idx, (t_idx, s_idx, n_idx) in enumerate(nan_coords[:10]):
            time_val = dye_sel[case.time_key].isel({case.time_key: int(t_idx)}).values
            time_str = pd.Timestamp(time_val).isoformat()

            node_actual = sel.nodes[int(n_idx)]
            sigma_actual = sel.sigmas[int(s_idx)]

            diagnostics.append(
                f"  [{idx+1}] time_idx={t_idx} ({time_str}), "
                f"node={node_actual}, sigma={sigma_actual}"
            )

        error_msg = (
            f"NaN values detected in DYE data for (year={year}, member={member})!\n"
            f"  Files: {[f.name for f in files]}\n"
            f"  Total NaNs: {nan_mask.sum().item()}\n"
            f"  First occurrences:\n" + "\n".join(diagnostics) + "\n"
            f"  Remediation hints:\n"
            f"    - Check for dry nodes (land mask)\n"
            f"    - Verify DYE initialization and boundary conditions\n"
            f"    - Review upstream preprocessing steps\n"
            f"    - Check model convergence and output writing"
        )
        raise ValueError(error_msg)

    # Handle negative values based on policy
    if neg.mode == "clip_zero":
        # Clip BEFORE averaging
        dye_sel = dye_sel.clip(min=0.0)
        logger.debug(f"Clipped negative values to zero for (year={year}, member={member})")

    # Average over nodes and sigma layers
    dye_mean = dye_sel.mean(dim=["node", case.sigma_key], skipna=False)

    # Attach metadata
    dye_mean.attrs.update({
        "units": dye_full.attrs.get("units", "dimensionless"),
        "long_name": f"Spatially averaged {case.var_name}",
        "year": year,
        "member": member,
        "nodes_selected": sel.nodes,
        "sigmas_selected": sel.sigmas,
        "negative_policy": neg.mode,
    })

    ds.close()

    logger.info(
        f"Loaded (year={year}, member={member}): {len(dye_mean[case.time_key])} time steps"
    )

    return dye_mean


def aggregate(
    member_map: dict[tuple[int, int], list[Path]],
    case: DyeCase,
    sel: Selection,
    neg: NegPolicy,
    align: AlignPolicy,
) -> xr.Dataset:
    """Aggregate dye time series across all (year, member) combinations.

    Args:
        member_map: Dictionary mapping (year, member) to file lists
        case: Dye case configuration
        sel: Spatial/vertical selection
        neg: Negative value handling policy
        align: Time alignment policy

    Returns:
        Dataset with dye(time, ensemble) and appropriate coordinates

    Raises:
        ValueError: If time alignment fails (e.g., empty intersection)
    """
    logger.info(
        f"Aggregating {len(member_map)} (year, member) pairs with align={align.mode}"
    )

    # Load all member series
    series_dict: dict[tuple[int, int], xr.DataArray] = {}
    for (year, member), files in member_map.items():
        series = load_member_series(files, case, sel, neg, year, member)
        series_dict[(year, member)] = series

    # Time alignment strategy
    if align.mode == "native_intersection":
        return _aggregate_native_intersection(series_dict, case, sel, neg, align)
    elif align.mode == "same_calendar":
        return _aggregate_same_calendar(series_dict, case, sel, neg, align)
    elif align.mode == "climatology":
        return _aggregate_climatology(series_dict, case, sel, neg, align)
    else:
        raise ValueError(f"Unknown alignment mode: {align.mode}")


def _aggregate_native_intersection(
    series_dict: dict[tuple[int, int], xr.DataArray],
    case: DyeCase,
    sel: Selection,
    neg: NegPolicy,
    align: AlignPolicy,
) -> xr.Dataset:
    """Aggregate using strict time intersection."""
    # Convert to common dataset
    datasets = []
    for (year, member), series in series_dict.items():
        ds_temp = series.to_dataset(name="dye")
        ds_temp = ds_temp.assign_coords(year=year, member=member)
        ds_temp = ds_temp.expand_dims({"ensemble": 1})
        datasets.append(ds_temp)

    # Align on time with inner join (strict intersection)
    aligned = xr.align(*datasets, join="inner", copy=False)

    # Check if intersection is empty
    if len(aligned[0][case.time_key]) == 0:
        # Diagnostic information
        time_ranges = []
        for (year, member), series in series_dict.items():
            t_start = series[case.time_key].values[0]
            t_end = series[case.time_key].values[-1]
            n_steps = len(series[case.time_key])
            time_ranges.append(
                f"  (year={year}, member={member}): "
                f"{pd.Timestamp(t_start)} to {pd.Timestamp(t_end)} ({n_steps} steps)"
            )

        error_msg = (
            f"Time alignment failed: empty intersection!\n"
            f"Time ranges for each (year, member):\n" + "\n".join(time_ranges) + "\n"
            f"Hint: Consider using align='same_calendar' or 'climatology' instead."
        )
        raise ValueError(error_msg)

    # Concatenate along ensemble dimension
    combined = xr.concat(aligned, dim="ensemble")

    # Build ensemble MultiIndex
    ensemble_coords = [(ds.year.item(), ds.member.item()) for ds in aligned]
    mindex = pd.MultiIndex.from_tuples(ensemble_coords, names=["year", "member"])
    mindex_coords = xr.Coordinates.from_pandas_multiindex(mindex, "ensemble")
    combined = combined.assign_coords(mindex_coords)

    # Add metadata
    combined.attrs.update({
        "basename": case.basename,
        "nodes_selected": sel.nodes,
        "sigmas_selected": sel.sigmas,
        "negative_policy": neg.mode,
        "alignment_mode": align.mode,
        "n_years": len(case.years),
        "n_members": len(case.members),
        "tool": "xfvcom.dye_timeseries",
        "time_steps": len(combined[case.time_key]),
    })

    logger.info(
        f"Native intersection complete: {len(combined[case.time_key])} time steps, "
        f"{len(combined.ensemble)} ensemble members"
    )

    return combined


def _aggregate_same_calendar(
    series_dict: dict[tuple[int, int], xr.DataArray],
    case: DyeCase,
    sel: Selection,
    neg: NegPolicy,
    align: AlignPolicy,
) -> xr.Dataset:
    """Aggregate by grouping on (month, day, hour) calendar position."""
    logger.info("Aggregating by same calendar position (month, day, hour)")

    # For each series, add calendar coordinates
    grouped_data = []
    for (year, member), series in series_dict.items():
        time_pd = pd.DatetimeIndex(series[case.time_key].values)

        # Create calendar coordinates
        series_df = series.to_dataframe(name="dye")
        series_df["month"] = time_pd.month
        series_df["day"] = time_pd.day
        series_df["hour"] = time_pd.hour
        series_df["year"] = year
        series_df["member"] = member

        grouped_data.append(series_df)

    # Combine all data
    combined_df = pd.concat(grouped_data, ignore_index=True)

    # Group by calendar position and ensemble
    grouped = combined_df.groupby(["month", "day", "hour", "year", "member"])["dye"].mean()

    # Convert back to xarray
    result_ds = grouped.to_xarray()
    result_ds = result_ds.rename({"dye": "dye"})

    # Create ensemble MultiIndex
    if "year" in result_ds.coords and "member" in result_ds.coords:
        # Stack year and member into ensemble dimension
        result_ds = result_ds.stack(ensemble=("year", "member"))

    # Add metadata
    result_ds.attrs.update({
        "basename": case.basename,
        "nodes_selected": sel.nodes,
        "sigmas_selected": sel.sigmas,
        "negative_policy": neg.mode,
        "alignment_mode": align.mode,
        "n_years": len(case.years),
        "n_members": len(case.members),
        "tool": "xfvcom.dye_timeseries",
        "calendar_coords": ["month", "day", "hour"],
    })

    logger.info(f"Same calendar aggregation complete: {len(result_ds.ensemble)} ensemble members")

    return result_ds


def _aggregate_climatology(
    series_dict: dict[tuple[int, int], xr.DataArray],
    case: DyeCase,
    sel: Selection,
    neg: NegPolicy,
    align: AlignPolicy,
) -> xr.Dataset:
    """Aggregate as climatological mean over years by (month, day, hour)."""
    logger.info("Aggregating as climatology (average over years)")

    # For each series, add calendar coordinates
    grouped_data = []
    for (year, member), series in series_dict.items():
        time_pd = pd.DatetimeIndex(series[case.time_key].values)

        series_df = series.to_dataframe(name="dye")
        series_df["month"] = time_pd.month
        series_df["day"] = time_pd.day
        series_df["hour"] = time_pd.hour
        series_df["member"] = member

        grouped_data.append(series_df)

    # Combine all data
    combined_df = pd.concat(grouped_data, ignore_index=True)

    # Group by calendar position and member, average over years
    grouped = combined_df.groupby(["month", "day", "hour", "member"])["dye"].mean()

    # Convert back to xarray
    result_ds = grouped.to_xarray()

    # Add overall climatological mean (averaged over members too)
    clim_mean = combined_df.groupby(["month", "day", "hour"])["dye"].mean()
    result_ds["clim_mean"] = clim_mean.to_xarray()

    # Add metadata
    result_ds.attrs.update({
        "basename": case.basename,
        "nodes_selected": sel.nodes,
        "sigmas_selected": sel.sigmas,
        "negative_policy": neg.mode,
        "alignment_mode": align.mode,
        "n_years": len(case.years),
        "n_members": len(case.members),
        "tool": "xfvcom.dye_timeseries",
        "calendar_coords": ["month", "day", "hour"],
        "note": "Averaged over years; clim_mean is average over both years and members",
    })

    logger.info("Climatology aggregation complete")

    return result_ds


def negative_stats(ds: xr.Dataset, series_dict: dict[tuple[int, int], xr.DataArray] | None = None) -> dict:
    """Compute statistics on negative values BEFORE clipping.

    Args:
        ds: Aggregated dataset
        series_dict: Optional raw series before clipping (for accurate stats)

    Returns:
        Dictionary with negative value statistics
    """
    logger.info("Computing negative value statistics")

    stats: dict = {
        "per_member": {},
        "global": {},
    }

    # If we have raw series (before clipping), use those
    if series_dict is not None:
        for (year, member), series in series_dict.items():
            neg_mask = series < 0
            count_neg = int(neg_mask.sum().item())
            min_val = float(series.min().item())
            total_samples = int(series.size)
            share_neg = count_neg / total_samples if total_samples > 0 else 0.0

            stats["per_member"][f"{year}_{member}"] = {
                "count_neg": count_neg,
                "min_value": min_val,
                "share_neg": share_neg,
                "total_samples": total_samples,
            }

        # Global statistics
        all_values = np.concatenate([s.values.flatten() for s in series_dict.values()])
        global_min = float(np.min(all_values))
        global_count_neg = int(np.sum(all_values < 0))
        global_total = len(all_values)
        global_share = global_count_neg / global_total if global_total > 0 else 0.0

        stats["global"] = {
            "min_value": global_min,
            "count_neg": global_count_neg,
            "share_neg": global_share,
            "total_samples": global_total,
        }
    else:
        # Fallback: analyze from dataset (may be post-clipping)
        logger.warning("No raw series provided; computing stats from aggregated dataset")

        if "dye" in ds:
            dye_values = ds["dye"].values
            neg_mask = dye_values < 0
            count_neg = int(np.sum(neg_mask))
            min_val = float(np.min(dye_values))
            total_samples = dye_values.size
            share_neg = count_neg / total_samples if total_samples > 0 else 0.0

            stats["global"] = {
                "min_value": min_val,
                "count_neg": count_neg,
                "share_neg": share_neg,
                "total_samples": total_samples,
            }

    # Attach summary to dataset attributes
    ds.attrs["negative_stats"] = json.dumps(stats, indent=2)

    logger.info(
        f"Negative stats: global_min={stats.get('global', {}).get('min_value', 'N/A')}, "
        f"global_count={stats.get('global', {}).get('count_neg', 'N/A')}"
    )

    return stats


def verify_linearity(
    ds: xr.Dataset,
    ref_member: int = 0,
    parts: list[int] | None = None,
) -> dict:
    """Verify linearity assumption: ref_member = sum(parts).

    Args:
        ds: Aggregated dataset with ensemble dimension
        ref_member: Reference member index (expected to equal sum of parts)
        parts: List of member indices to sum (if None, use all except ref_member)

    Returns:
        Dictionary with linearity metrics (RMSE, MAE, NSE, etc.)
    """
    logger.info(f"Verifying linearity: ref_member={ref_member}, parts={parts}")

    # Determine parts if not specified
    if parts is None:
        ensemble_idx = ds.ensemble.to_index() if hasattr(ds.ensemble, 'to_index') else ds.ensemble.values
        if isinstance(ensemble_idx, pd.MultiIndex):
            all_members = ensemble_idx.get_level_values("member").unique().tolist()
        else:
            # Assume ensemble index corresponds to member
            all_members = list(range(ds.sizes["ensemble"]))
        parts = [m for m in all_members if m != ref_member]

    # Extract reference and parts
    # Handle both MultiIndex and flat ensemble
    ensemble_idx = ds.ensemble.to_index() if hasattr(ds.ensemble, 'to_index') else ds.ensemble.values

    if isinstance(ensemble_idx, pd.MultiIndex):
        # MultiIndex case
        member_values = ensemble_idx.get_level_values("member")
        ref_mask = member_values == ref_member
        parts_mask = member_values.isin(parts)

        ref_data = ds["dye"].isel(ensemble=np.where(ref_mask)[0])
        parts_data = ds["dye"].isel(ensemble=np.where(parts_mask)[0])

        # Sum over parts
        parts_sum = parts_data.sum(dim="ensemble")
        ref_series = ref_data.isel(ensemble=0) if ref_data.sizes.get("ensemble", 0) > 0 else ref_data
    else:
        # Flat ensemble case - assume ensemble index corresponds to member
        logger.warning("Ensemble is not MultiIndex; assuming ensemble index = member index")

        # Find indices for ref and parts
        ref_idx = ref_member if ref_member < ds.sizes["ensemble"] else None
        parts_idx = [p for p in parts if p < ds.sizes["ensemble"]]

        if ref_idx is None:
            raise ValueError(f"ref_member {ref_member} not found in ensemble (size={ds.sizes['ensemble']})")

        ref_series = ds["dye"].isel(ensemble=ref_idx)
        parts_data = ds["dye"].isel(ensemble=parts_idx)
        parts_sum = parts_data.sum(dim="ensemble")

    # Align time coordinates
    ref_series, parts_sum = xr.align(ref_series, parts_sum, join="inner")

    # Compute metrics
    diff = ref_series - parts_sum
    rmse = float(np.sqrt((diff**2).mean()).item())
    mae = float(np.abs(diff).mean().item())
    max_abs_diff = float(np.abs(diff).max().item())

    # Nash-Sutcliffe Efficiency
    mean_ref = float(ref_series.mean().item())
    nse = 1.0 - float(((diff**2).sum() / ((ref_series - mean_ref)**2).sum()).item())

    # Count of compared samples
    n_samples = int(ref_series.size)

    metrics = {
        "ref_member": ref_member,
        "parts": parts,
        "rmse": rmse,
        "mae": mae,
        "max_abs_diff": max_abs_diff,
        "nse": nse,
        "n_samples": n_samples,
        "summary": (
            f"Linearity check: RMSE={rmse:.6e}, MAE={mae:.6e}, "
            f"Max|Δ|={max_abs_diff:.6e}, NSE={nse:.6f} ({n_samples} samples)"
        ),
    }

    # Attach to dataset
    ds.attrs["linearity_check"] = json.dumps(metrics, indent=2)

    logger.info(metrics["summary"])

    return metrics

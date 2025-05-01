import inspect

import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score


def clean_kwargs(func, kwargs):
    """
    Filter kwargs so that only arguments accepted by *func* survive.

    Parameters
    ----------
    func : Callable
        Target function whose signature is inspected.
    kwargs : dict
        Original keyword arguments.

    Returns
    -------
    dict
        Filtered kwargs that can be safely expanded with ** when
        calling *func*.
    """
    func_args = inspect.signature(func).parameters
    # Keep keys that appear in the callee's signature
    return {k: v for k, v in kwargs.items() if k in func_args}


def unpack_plot_kwargs(kwargs):
    """
    Unpack nested 'plot_kwargs' dictionary if present.

    Parameters:
    - kwargs: Dictionary of keyword arguments.

    Returns:
    - A flat dictionary with 'plot_kwargs' unpacked.
    """

    if "plot_kwargs" in kwargs and isinstance(kwargs["plot_kwargs"], dict):
        return {
            **kwargs["plot_kwargs"],
            **{k: v for k, v in kwargs.items() if k != "plot_kwargs"},
        }
    return kwargs


def parse_coordinate(coord):
    """
    Convert a coordinate in degrees:minutes:seconds or degrees:minutes format to a float (decimal degrees).

    Parameters:
    - coord (str or float): Coordinate as a string in degrees:minutes:seconds (e.g., "139:30:25")
      or degrees:minutes (e.g., "139:30"), or a float.

    Returns:
    - float: Coordinate in decimal degrees.
    """
    if isinstance(coord, (int, float)):
        return float(coord)

    parts = coord.split(":")
    if len(parts) == 3:  # degrees:minutes:seconds format
        degrees, minutes, seconds = map(float, parts)
        return degrees + minutes / 60 + seconds / 3600
    elif len(parts) == 2:  # degrees:minutes format
        degrees, minutes = map(float, parts)
        return degrees + minutes / 60
    else:
        raise ValueError(f"Invalid coordinate format: {coord}")


def apply_xlim_ylim(ax, xlim, ylim, is_cartesian=False):
    """
    Apply xlim and ylim to a Matplotlib axis, supporting both Cartesian and geographic (lon/lat) coordinates.

    Parameters:
    - ax (matplotlib.axes._subplots.AxesSubplot): Matplotlib axis object.
    - xlim (tuple): Range for the x-axis, in longitude/Cartesian x as (min, max).
    - ylim (tuple): Range for the y-axis, in latitude/Cartesian y as (min, max).
    - is_cartesian (bool): If True, the input is Cartesian coordinates and no CRS transformation is applied.

    Returns:
    - None
    """

    if is_cartesian:
        # Cartesian coordinates: use xlim and ylim directly
        x_min, x_max = xlim
        y_min, y_max = ylim
        # if ax is not None:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        # Default CRS for geographic data
        src_crs = ccrs.PlateCarree()
        x_min, x_max = map(parse_coordinate, xlim)
        y_min, y_max = map(parse_coordinate, ylim)
        # if ax is not None:
        ax.set_extent([x_min, x_max, y_min, y_max], crs=src_crs)
        xlim = (x_min, x_max)
        ylim = (y_min, y_max)

    print(f"Set extent: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
    # return xlim, ylim


def evaluate_model_scores(sim_list, obs_list):
    """
    Evaluate model performance metrics including R², Pearson r, RMSE, and bias.
    Normalize Combined Score to 0-1, where 1 indicates perfect match and 0 indicates poor performance.

    Parameters:
    - sim_list: List of simulation DataArray (or numpy arrays).
    - obs_list: List of observation DataArray (or numpy arrays).

    Returns:
    - scores: A dictionary containing individual scores for each layer and a combined score.

    Usage:
    ```
    from ..utils.helpers_utils import evaluate_model_scores, generate_test_data
    sim_list, obs_list = generate_test_data()
    ```
    """

    def to_numpy(data):
        if isinstance(data, xr.DataArray):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise ValueError("Input data must be xarray.DataArray or numpy.ndarray")

    individual_scores = []
    for sim, obs in zip(sim_list, obs_list):
        sim = to_numpy(sim)
        obs = to_numpy(obs)

        # Ensure matching shapes and remove NaNs
        mask = ~np.isnan(sim) & ~np.isnan(obs)
        sim, obs = sim[mask], obs[mask]

        if len(sim) == 0 or len(obs) == 0:
            raise ValueError(
                "Simulation and observation data contain no valid values after removing NaNs.\n"
                "Maybe observed data corresponding to the siglay depths does not exist \n"
                "because the total depth of observed is less than that of simulated."
            )

        # Calculate metrics
        r2 = r2_score(obs, sim)
        r, _ = pearsonr(obs, sim)
        rmse = np.sqrt(mean_squared_error(obs, sim))
        bias = np.mean(sim - obs)

        individual_scores.append(
            {"R²": r2, "r": r.item(), "RMSE": rmse.item(), "Bias": bias.item()}
        )

    # Normalize scores for Combined Score calculation
    normalized_scores = []
    for score in individual_scores:
        normalized_r2 = max(0, score["R²"])  # Clamp R² to [0, 1]
        normalized_r = (score["r"] + 1) / 2  # Convert r from [-1, 1] to [0, 1]
        normalized_rmse = 1 / (
            1 + score["RMSE"]
        )  # Smaller RMSE is better, normalize to (0, 1]
        normalized_bias = 1 / (
            1 + abs(score["Bias"])
        )  # Smaller bias is better, normalize to (0, 1]

        normalized_scores.append(
            {
                "R²": normalized_r2,
                "r": normalized_r,
                "RMSE": normalized_rmse,
                "Bias": normalized_bias,
            }
        )

    # Combine scores into a single score using weights
    weights = {"R²": 0.4, "r": 0.4, "RMSE": 0.1, "Bias": 0.1}
    combined_score = sum(
        weights["R²"] * score["R²"]
        + weights["r"] * score["r"]
        + weights["RMSE"] * score["RMSE"]
        + weights["Bias"] * score["Bias"]
        for score in normalized_scores
    ) / len(normalized_scores)

    return {"individual_scores": individual_scores, "combined_score": combined_score}


# Test Data
def generate_test_data():
    """
    Generate an ideal test data (perfect) for evaluate_model_scores().
    """

    time = np.arange("2023-01-01", "2023-01-11", dtype="datetime64[h]").astype(
        "datetime64[ns]"
    )  # Ensure nanosecond precision
    sim_list = [
        xr.DataArray(
            np.sin(np.linspace(0, 10, len(time))) + np.random.normal(0, 1, len(time)),
            dims="time",
            coords={"time": time},
        ),
        xr.DataArray(
            np.cos(np.linspace(0, 10, len(time))) + np.random.normal(0, 1, len(time)),
            dims="time",
            coords={"time": time},
        ),
    ]
    obs_list = [
        xr.DataArray(
            np.sin(np.linspace(0, 10, len(time))), dims="time", coords={"time": time}
        ),
        xr.DataArray(
            np.cos(np.linspace(0, 10, len(time))), dims="time", coords={"time": time}
        ),
    ]
    return sim_list, obs_list


def ensure_time_index(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """
    Guarantee datetime64 type in `time` coord.
    If already ok, just return the original object.
    """
    if "time" not in ds.coords:
        return ds  # time 軸が無ければ何もしない
    if ds.time.dtype.kind != "M":  # not datetime64
        ds = ds.assign_coords(time=ds.time.values.astype("datetime64[ns]"))
    return ds

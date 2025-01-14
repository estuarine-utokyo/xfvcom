import inspect
import cartopy.crs as ccrs

def clean_kwargs(func, kwargs):
    """
    Clean kwargs to avoid conflicts with the explicit arguments of the given function.
    """
    func_args = inspect.signature(func).parameters
    clean_kwargs = {}
    for key, value in kwargs.items():
        if key in func_args:
            clean_kwargs[key] = value  # 関数の引数として渡すキーは保持
        elif key not in func_args:
            clean_kwargs[key] = value  # その他のキーも保持
    return clean_kwargs

def unpack_plot_kwargs(kwargs):
    """
    Unpack nested 'plot_kwargs' dictionary if present.

    Parameters:
    - kwargs: Dictionary of keyword arguments.

    Returns:
    - A flat dictionary with 'plot_kwargs' unpacked.
    """

    if 'plot_kwargs' in kwargs and isinstance(kwargs['plot_kwargs'], dict):
        return {**kwargs['plot_kwargs'], **{k: v for k, v in kwargs.items() if k != 'plot_kwargs'}}
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
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    else:
        # Default CRS for geographic data
        src_crs = ccrs.PlateCarree()
        x_min, x_max = map(parse_coordinate, xlim)
        y_min, y_max = map(parse_coordinate, ylim)
        ax.set_extent([x_min, x_max, y_min, y_max], crs=src_crs)

    print(f"Set extent: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")


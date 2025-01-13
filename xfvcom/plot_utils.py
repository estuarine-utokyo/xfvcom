import pandas as pd
from .helpers import FrameGenerator, create_gif

def create_gif_anim_2d_plot(plotter, var_name, siglay=None, fps=10, post_process_func=None, plot_kwargs=None):
    """
    Generate a 2D plot animation as a GIF.

    Parameters:
    - plotter: FvcomPlotter instance used for plotting.
    - var_name: Name of the variable to plot.
    - siglay: Index of the vertical layer (optional).
    - fps: Frames per second for the GIF animation.
    - post_process_func: Function to apply custom styling to the plot (optional).
    - plot_kwargs: Additional plotting arguments passed to the plotter.

    Returns:
    - None
    """

    if siglay is None:
        da = plotter.ds[var_name]
    else:
        da = plotter.ds[var_name].isel(siglay=siglay)

    start_date = da.time.isel(time=0).values
    start_date = pd.to_datetime(start_date).strftime("%Y%m%d")
    end_date = da.time.isel(time=-1).values
    end_date = pd.to_datetime(end_date).strftime("%Y%m%d")
    
    suffix = "_frame"
    len_suffix = len(suffix)
    base_name = f"{da.long_name}_{start_date}-{end_date}{suffix}"
    
    # Generate movie frames using helper methods in helpers.py
    output_dir = "frames"
    
    frames = FrameGenerator.generate_frames(data_array=da, output_dir=output_dir, plotter=plotter, base_name=base_name,
        post_process_func=post_process_func, **plot_kwargs)
    
    # Create GIF animation
    output_gif = f"{base_name[:-len_suffix]}.gif"
    #plotter.create_gif(frames, output_gif, fps=fps)

    create_gif(frames, output_gif, fps=fps)

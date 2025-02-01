import pandas as pd
import re
from .helpers import FrameGenerator, create_gif, convert_gif_to_mp4

def create_anim_2d_plot(plotter, processes, var_name, siglay=None, fps=10, generate_gif=True, generate_mp4=False,
                        batch_size=500, cleanup=False, post_process_func=None, plot_kwargs=None):
    """
    Generate a 2D plot animation as a GIF/MP4.

    Parameters:
    - plotter: FvcomPlotter instance used for plotting.
    - processes: Number of maximum processes.
    - var_name: Name of the variable to plot.
    - siglay: Index of the vertical layer (optional).
    - fps: Frames per second for the GIF animation.
    - generate_gif: If True, generate a GIF animation.
    - generate_mp4: If True, generate an MP4 animation.
    - cleanup: If True, delete the frame files after creating the animation.
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
    # Convert subscripts $_x$ -> x
    long_name = re.sub(r"\$_(\d+)\$", r"\1", da.long_name)
    base_name = f"{long_name}_{start_date}-{end_date}{suffix}"
    
    # Generate movie frames using helper methods in helpers.py
    output_dir = "frames"
    
    frames = FrameGenerator.generate_frames(data_array=da, output_dir=output_dir, plotter=plotter, processes=processes, 
                                            base_name=base_name, post_process_func=post_process_func, **plot_kwargs)
    
    anim_base_name = f"{base_name[:-len_suffix]}"
    if not generate_gif and not generate_mp4:
        print(f"Frames have been generated and saved as PNG files. No animation created.")
        return
    # Create GIF animation
    if generate_gif:
        output_gif = f"{anim_base_name}.gif"
        #clip = ImageSequenceClip(frames, fps=fps)
        #clip.write_gif(output_gif, fps=fps)
        #create_gif(frames, output_gif, fps=fps, cleanup=cleanup)
        #create_gif_with_batch(frames, output_gif=output_gif, fps=fps, batch_size=batch_size, cleanup=cleanup)
        create_gif(frames, output_gif=output_gif, fps=fps, cleanup=cleanup)
    # Create MP4 animation
    if generate_mp4:
        output_mp4 = f"{anim_base_name}.mp4"
        #create_mp4(frames, output_mp4, fps=fps, cleanup=cleanup) # does not work
        convert_gif_to_mp4(output_gif, output_mp4)
    return anim_base_name


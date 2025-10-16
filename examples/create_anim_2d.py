# Create GIF animation for 2D horizontal plots
# **Author: Jun Sasaki  Coded on 2025-01-12  Updated on 2025-02-04**<br>
# Create a GIF animation. Defining `post_process_func` for customization or `post_process_func=None` without customization.
#
import os
import warnings

import pandas as pd

from xfvcom import FvcomDataLoader, FvcomPlotConfig, FvcomPlotter
from xfvcom.plot_utils import create_anim_2d_plot

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Define `post_process_func` for customizing plot by modifying ax.
def post_process_func(ax, da, time):
    """
    Plot the corresponding datetime at each frame

    Parameters:
    - ax: matplotib axis.
    - da: DataArray.
    - time: Frame time.
    """
    datetime = pd.Timestamp(da.time.item()).strftime("%Y-%m-%d %H:%M:%S")
    ax.set_title(f"Time: {datetime}")


def anim(
    nprocs=16,
    var_name=None,
    siglay=0,
    plot_kwargs=None,
    plotter=None,
    fps=10,
    generate_gif=True,
    generate_mp4=False,
    cleanup=True,
    post_process_func=post_process_func,
):
    """
    Generate animation frames using multiprocessing and create a GIF animation.

    Parameters:
    - nprocs: Number of processes.
    - var_name: Variable name to plot.
    - siglay: Sigma layer index.
    - plot_kwargs: Dictionary of plot settings.
    - plotter: FvcomPlotter instance.
    - fps: Frames per second.
    - generate_gif: True for generating.
    - generate_mp4: True for generating.
    - cleanup: True for cleaning up.
    """
    create_anim_2d_plot(
        plotter=plotter,
        processes=nprocs,
        var_name=var_name,
        siglay=siglay,
        fps=fps,
        generate_gif=generate_gif,
        generate_mp4=generate_mp4,
        cleanup=cleanup,
        post_process_func=post_process_func,
        plot_kwargs=plot_kwargs,
    )


### Load FVCOM output netcdf into FvcomDataLoader instance of `fvcom` where `fvcom.ds` is Dataset.
# base_path = "~/Github/TB-FVCOM/goto2023/output"
base_path = "/home/pj24001722/ku40003295/Ersem_TokyoBay/output_5times/"
# base_path = os.path.expanduser(base_path) if base_path else None  # expand ~/
# base_path = os.path.expanduser(base_path) if base_path else None  # example: "~/data"
base_path = os.path.expanduser(base_path)
# base_path: str | None = (
#    os.path.expanduser(base_path) if base_path else None
# )  # example: "~/data"
### List of netcdf files convenient to switch to another netcdf by specifying its index
# ncfiles = ["TokyoBay18_r16_crossed_0001.nc"]
ncfiles = ["tb_0001.nc"]
index_ncfile = 0
ncfile_path = f"{base_path}/{ncfiles[index_ncfile]}"
fvcom = FvcomDataLoader(ncfile_path=ncfile_path, chunks=None, time_tolerance=5)

# Set the number of processes and dpi.
nprocs = 10  # Between 10 and 20 may be the best considering the overhead of parallel computing.
figsize = (6, 8)  # (6, 8) recommended.
dpi = 150  # dpi=60, 150, 300 may be suitable. dpi=60 fits for two-row in PPTX.
fps = 10  # Frames per seconds.

# Set var_names and sigma layers to be plotted.
var_names = [
    "temp",
    "salinity",
    "O2_o",
    "N1_p",
    "N3_n",
    "N4_n",
    "N5_s",
    "P1_Chl",
    "P2_Chl",
    "P3_Chl",
    "P4_Chl",
]
var_names = ["O2_o"]
siglays = [0]  # Specify sigma layers as a list: =0 for surface

# Set plot_kwargs for `ax.tricontourf(**kwargs)`
# plot_kwargs={"verbose": False, "vmin": 10, "vmax": 20, "levels": [9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 15, 16, 17, 18, 19, 20]}
# plot_kwargs={"verbose": False, "vmin": 20, "vmax": 34, "levels": 29, "cmap": "jet"}
plot_kwargs_dict = {
    "salinity": {"vmin": 20, "vmax": 34, "levels": 29, "cmap": "jet"},
    "temp": {"vmin": 10, "vmax": 30, "levels": 21, "cmap": "jet"},
    "O2_o": {"vmin": 0, "vmax": 15, "levels": 16, "cmap": "jet"},
    "N1_p": {"vmin": 0, "vmax": 6, "levels": 19, "cmap": "jet"},
    "N3_n": {"vmin": 0, "vmax": 80, "levels": 21, "cmap": "jet"},
    "N4_n": {"vmin": 0, "vmax": 40, "levels": 21, "cmap": "jet"},
    "N5_s": {"vmin": 0, "vmax": 80, "levels": 21, "cmap": "jet"},
    "P1_Chl": {"vmin": 0, "vmax": 80, "levels": 21, "cmap": "jet"},
    "P2_Chl": {"vmin": 0, "vmax": 5, "levels": 21, "cmap": "jet"},
    "P3_Chl": {"vmin": 0, "vmax": 5, "levels": 21, "cmap": "jet"},
    "P4_Chl": {"vmin": 0, "vmax": 5, "levels": 21, "cmap": "jet"},
}
plot_kwargs_dict = {
    "salinity": {"vmin": 20, "vmax": 34, "levels": 29, "cmap": "jet"},
    "temp": {"vmin": 10, "vmax": 30, "levels": 21, "cmap": "jet"},
    "O2_o": {"vmin": 0, "cmap": "jet"},
    "N1_p": {"vmin": 0, "cmap": "jet"},
    "N3_n": {"vmin": 0, "cmap": "jet"},
    "N4_n": {"vmin": 0, "cmap": "jet"},
    "N5_s": {"vmin": 0, "cmap": "jet"},
    "P1_Chl": {"vmin": 0, "cmap": "jet"},
    "P2_Chl": {"vmin": 0, "cmap": "jet"},
    "P3_Chl": {"vmin": 0, "cmap": "jet"},
    "P4_Chl": {"vmin": 0, "cmap": "jet"},
}
plot_kwargs = {"cmap": "jet"}  # automated scaling

# Slice by the time index range, e.g., `time=slice(0,100)`, `time=slice(0, None)` for the whole range.
time_slice = slice(0, None)
# time_index = pd.DatetimeIndex(fvcom.ds['time'].values)
# Example of filtering time index at 12:00 on 15th every month.
# target_times = time_index[(time_index.day == 15) & (time_index.hour == 12)]
# time_slice = [i for i, t in enumerate(fvcom.ds['time'].values) if t in target_times]
# Check the time_slice is within the valid range.
time_size = fvcom.ds.sizes["time"]
print(f"Time dimension size: {time_size}")
start_index = time_slice.start if time_slice.start is not None else 0
end_index = time_slice.stop if time_slice.stop is not None else time_size
if start_index < 0 or end_index > time_size:
    raise IndexError(
        f"Time slice is out of bounds. Valid range is [0, {time_size}), "
        f"but got slice({start_index}, {end_index})."
    )

dataset = fvcom.ds.isel(time=time_slice)
cfg = FvcomPlotConfig(figsize=figsize, dpi=dpi)
plotter = FvcomPlotter(dataset, cfg)


# Invoke xfvcom.plot_utils.create_gif_anim_2d_plot
for siglay in siglays:
    for var_name in var_names:
        anim(
            nprocs=nprocs,
            var_name=var_name,
            siglay=siglay,
            plot_kwargs=plot_kwargs_dict[var_name],
            plotter=plotter,
            fps=fps,
        )

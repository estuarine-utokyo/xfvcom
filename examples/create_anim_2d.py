# # Create GIF animation for 2D horizontal plots
# **Author: Jun Sasaki  Coded on 2025-01-12  Updated on 2025-02-01**<br>
# Create a GIF animation. Customization can be made by defining `post_process_func` statically (no time change) or dynamically (change with time), or `post_process_func=None` without customization.
# 
from xfvcom import FvcomDataLoader, FvcomPlotConfig, FvcomPlotter
from xfvcom.helpers import FrameGenerator
from xfvcom.plot_utils import create_anim_2d_plot
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ### Prepare FvcomDataLoader instance of `fvcom` using FVCOM output netcdf.
# Dataset is `fvcom.ds`.
# Loading FVCOM output netcdf
#base_path = "~/Github/TB-FVCOM/goto2023/output"
base_path =  "/home/pj24001722/ku40003295/Ersem_TokyoBay/output_5times/"
# List of netcdf files convenient to switch to another netcdf by specifying its index
#ncfiles = ["TokyoBay18_r16_crossed_0001.nc"]
ncfiles = ["tb_0001.nc"]
index_ncfile = 0

# Specify the number of processes
nprocs = 16
# Create an instance of FvcomDataLoader where fvcom.ds is a Dataset
# 1. メタデータのみを読み込んで time のサイズを取得
ncfile_path = f"{base_path}/{ncfiles[index_ncfile]}"

fvcom = FvcomDataLoader(base_path=base_path, ncfile=ncfiles[index_ncfile], chunks=None, time_tolerance=5)

# ### Create GIF animation with dynamic customizing.
# - 2-D horizontal plot with dynamic customizing by updating `ax`, which changes with time.
# - Prepare `dynamic_custom_plot` for dynamic customizing.  
def custome_plot(ax, da, time):
    """
    Plot the corresponding datetime at each frame

    Parameters:
    - ax: matplotib axis.
    - da: DataArray.
    - time: Frame time.    
    """
    datetime = pd.Timestamp(da.time.item()).strftime('%Y-%m-%d %H:%M:%S')
    ax.set_title(f"Time: {datetime}")

def anim(nprocs=16, var_name=None, siglay=0, plot_kwargs=None, plotter=None):
    # Invoke xfvcom.plot_utils.create_gif_anim_2d_plot
    create_anim_2d_plot(plotter, nprocs, var_name, siglay=siglay, fps=10, generate_gif=True, generate_mp4=False,
                        cleanup=True, post_process_func=custome_plot, plot_kwargs=plot_kwargs)

var_names = ["salinity", "temp", "O2_o", "N1_p", "N3_n", "N4_n", "N5_s", "P1_Chl", "P2_Chl", "P3_Chl", "P4_Chl"]
var_names = ["O2_o"]
siglays =[1]

# You may slice by the time index range, e.g., `time=slice(0,10)`, `time=slice(0, None)` for the whole range.
time = slice(0, None)
#time_index = pd.DatetimeIndex(fvcom.ds['time'].values)
# 毎月15日の12:00を条件にフィルタリング
#target_times = time_index[(time_index.day == 15) & (time_index.hour == 12)]
#time = [i for i, t in enumerate(fvcom.ds['time'].values) if t in target_times]
# Check the time slice is within the valid range.
time_size = fvcom.ds.sizes['time']
print(f"Time dimension size: {time_size}")
start_index = time.start if time.start is not None else 0
end_index = time.stop if time.stop is not None else time_size
if start_index < 0 or end_index > time_size:
    raise IndexError(f"Time slice is out of bounds. Valid range is [0, {time_size}), "
                     f"but got slice({start_index}, {end_index}).")

dataset = fvcom.ds.isel(time=time)
cfg = FvcomPlotConfig(figsize=(6, 8), dpi=60)
plotter = FvcomPlotter(dataset, cfg)

# Set plot_kwargs for `ax.tricontourf(**kwargs)`
#plot_kwargs={"verbose": False, "vmin": 10, "vmax": 20, "levels": [9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 15, 16, 17, 18, 19, 20]}
#plot_kwargs={"verbose": False, "vmin": 20, "vmax": 34, "levels": 29, "cmap": "jet"}
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

plot_kwargs={"cmap": "jet"}

# Invoke xfvcom.plot_utils.create_gif_anim_2d_plot
for siglay in siglays:
    for var_name in var_names:
        anim(nprocs=nprocs, var_name=var_name, siglay=siglay, plot_kwargs=plot_kwargs_dict[var_name], plotter=plotter)

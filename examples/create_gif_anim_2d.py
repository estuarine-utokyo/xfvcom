# coding: utf-8

# Set the following environment on Linux to avoid warning of Qt, or install required packages such as qtwayland5.
# $ export QT_QPA_PLATFORM=xcb

# # Create GIF animation for 2D horizontal plots
# **Author: Jun Sasaki  Coded on 2025-01-12  Updated on 2025-01-13**<br>
# Create a GIF animation. Customization can be made by defining `post_process_func` statically (no time change) or dynamically (change with time), or `post_process_func=None` without customization.
# 
# ```Python
# def post_process_func(ax, da=None, time=None):
#     """
#     Example of post_process_func for customizing plot (e.g., add text or markers)
#     
#     Parameters:
#     - ax: matplotlib axis.
#     - da: DataArray (optional and used for dynamic customizing).
#     - time: Frame time (optional and used for dynamic customizing).
#     """
# ```

# In[ ]:


from xfvcom import FvcomDataLoader, FvcomPlotConfig, FvcomPlotter
from xfvcom.helpers import FrameGenerator
from xfvcom.plot_utils import create_gif_anim_2d_plot
import pandas as pd
from functools import partial
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ### Prepare FvcomDataLoader instance of `fvcom` using FVCOM output netcdf.
# Dataset is `fvcom.ds`.

# In[ ]:


# Loading FVCOM output netcdf
base_path = "~/Github/TB-FVCOM/goto2023/output"
# List of netcdf files convenient to switch to another netcdf by specifying its index
ncfiles = ["TokyoBay18_r16_crossed_0001.nc"]
index_ncfile = 0
# Create an instance of FvcomDataLoader where fvcom.ds is a Dataset
fvcom = FvcomDataLoader(base_path=base_path, ncfile=ncfiles[index_ncfile], time_tolerance=5)


# ### Create GIF animation with static customizing.
# - 2-D horizontal plot with static customization by updating `ax`, which does not change with time.
# - Prepare `add_custom_plot` for customizing or `post_process_func=None` without customizing.  

# In[ ]:


def static_custom_plot(ax):
    """
    Customizing plot by updating ax

    Parameters:
    - ax: matplotlib axis
    """

    # Further customization can be added.
    ax.set_title("Title with Custom Plot")

'''
# Loading FVCOM output netcdf
base_path = "~/Github/TB-FVCOM/goto2023/output"
# List of netcdf files convenient to switch to another netcdf by specifying its index
ncfiles = ["TokyoBay18_r16_crossed_0001.nc"]
index_ncfile = 0
# Create an instance of FvcomDataLoader where fvcom.ds is a Dataset
fvcom = FvcomDataLoader(base_path=base_path, ncfile=ncfiles[index_ncfile], time_tolerance=5)
'''
# Create an instance of FvcomPlotter
dataset = fvcom.ds.isel(time=slice(0, 20)) # You may slice by the time index range. The whole range is `time=slice(0, None)`.
plotter = FvcomPlotter(dataset, FvcomPlotConfig(figsize=(6, 8)))
# Specify var_name and siglay if any
var_name = "salinity"
siglay = 0
# Set plot_kwargs for `ax.tricontourf(**kwargs)`.
plot_kwargs={"verbose": False, "vmin": 10, "vmax": 20, "levels": [9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 15, 16, 17, 18, 19, 20]}
plot_kwargs={"verbose": False, "vmin": 28, "vmax": 34, "levels": 20, "cmap": "jet"}
#plot_kwargs={"verbose": False, "vmin": 28, "vmax": 34, "levels": 20, "cmap": "jet", "with_mesh": True}
#plot_kwargs={}

# Invoke xfvcom.plot_utils.create_gif_anim_2d_plot
create_gif_anim_2d_plot(plotter, var_name, siglay=siglay, fps=10, post_process_func=static_custom_plot, plot_kwargs=plot_kwargs)


# ### Create GIF animation with dynamic customizing.
# - 2-D horizontal plot with dynamic customizing by updating `ax`, which changes with time.
# - Prepare `dynamic_custom_plot` for dynamic customizing.  

# In[ ]:


def dynamic_custom_plot(ax, da, time):
    """
    Plot the corresponding datetime at each frame

    Parameters:
    - ax: matplotib axis.
    - da: DataArray.
    - time: Frame time.    
    """
    datetime = pd.Timestamp(da.time.item()).strftime('%Y-%m-%d %H:%M:%S')
    ax.set_title(f"Time: {datetime}")

dataset = fvcom.ds.isel(time=slice(0, 10)) # You may slice by the time index range. The whole range is `time=slice(0, None)`.
plotter = FvcomPlotter(dataset, FvcomPlotConfig(figsize=(6, 8)))
# Specify var_name and siglay if any
var_name = "salinity"
siglay = 0
# Set plot_kwargs for `ax.tricontourf(**kwargs)`
plot_kwargs={"verbose": False, "vmin": 10, "vmax": 20, "levels": [9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 15, 16, 17, 18, 19, 20]}
plot_kwargs={"verbose": False, "vmin": 28, "vmax": 34, "levels": 20, "cmap": "jet"}
#plot_kwargs={"verbose": False, "vmin": 28, "vmax": 34, "levels": 20, "cmap": "jet", "with_mesh": True}
#plot_kwargs={}

# Invoke xfvcom.plot_utils.create_gif_anim_2d_plot
create_gif_anim_2d_plot(plotter, var_name, siglay=siglay, fps=10, post_process_func=dynamic_custom_plot, plot_kwargs=plot_kwargs)


# In[ ]:





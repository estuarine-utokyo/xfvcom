from __future__ import annotations

import glob
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import matplotlib.colorbar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from ..plot_options import FvcomPlotOptions
from ..utils.helpers import FrameGenerator, convert_gif_to_mp4, create_gif

# import time
# from pathlib import Path
if TYPE_CHECKING:
    from ..plot.core import FvcomPlotter

# compile once
_FRAME_RE = re.compile(r"_(\d+)\.png$")


def _extract_index(path: str) -> int:
    """
    Return numeric frame index parsed from file name.
    If the pattern is not found, return -1 so that such
    files are sorted to the beginning and can be skipped.
    """
    m = _FRAME_RE.search(path)
    return int(m.group(1)) if m else -1


def prepare_contourf_args(
    data: np.ndarray | pd.DataFrame,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    levels: int | Sequence[float] | None = None,
    cmap: str = "viridis",
) -> dict[str, Any]:
    """
    Return kwargs dict for tricontourf / contourf.

    Notes
    -----
    * When *levels* is an int, it is passed directly to the colormap.
    * vmin / vmax fall back to data min / max if None.
    """
    vmin = float(np.nanmin(data)) if vmin is None else vmin
    vmax = float(np.nanmax(data)) if vmax is None else vmax
    if isinstance(levels, int):
        levels = np.linspace(vmin, vmax, levels + 1)
    kwargs = dict(levels=levels, cmap=cmap, norm=Normalize(vmin, vmax))
    return kwargs


def add_colorbar(
    fig: plt.Figure,
    mappable: Any,
    *,
    cax: plt.Axes | None = None,
    label: str | None = None,
    **cbar_opts: Any,
) -> matplotlib.colorbar.Colorbar:
    """
    Attach a colorbar to *fig* and return the Colorbar instance.
    """
    cbar = fig.colorbar(mappable, cax=cax, **cbar_opts)
    if label is not None:
        cbar.set_label(label)
    return cbar


def create_anim_2d_plot(
    plotter: FvcomPlotter,
    processes: int,
    var_name: str,
    *,
    siglay: int | None = None,
    fps: int = 10,
    generate_gif: bool = True,
    generate_mp4: bool = False,
    cleanup: bool = False,
    post_process_func: Callable | None = None,
    opts: FvcomPlotOptions | None = None,
    plot_kwargs: dict[str, Any] | None = None,
) -> str | None:
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
    - opts: FvcomPlotOptions instance for plot options (optional).
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
    # base_name = f"{long_name}_{start_date}-{end_date}{suffix}"
    base_name = f"{var_name}_{start_date}-{end_date}{suffix}"

    output_dir = Path(f"frames_{var_name}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir()

    # ------------------------------------------------------------
    # 0.  Unify option source  (old-style kwargs  /  new-style opts)
    # ------------------------------------------------------------
    plot_kwargs = plot_kwargs or {}

    if opts is None:  # --- old style only
        opts = FvcomPlotOptions.from_kwargs(**plot_kwargs)
    else:  # --- new style + extra kwargs
        opts.extra.update(plot_kwargs)

    # `plot_kwargs_final` は FrameGenerator → plot_2d にそのまま流れる
    plot_kwargs_final = {"opts": opts}

    frames = FrameGenerator.generate_frames(
        da=da,
        output_dir=output_dir,
        plotter=plotter,
        processes=processes,
        base_name=base_name,
        post_process_func=post_process_func,
        **plot_kwargs_final,
    )
    # print(f"frames={frames}")
    # `proc_*` 内のフレームを `frames/` に統合（上書きする）
    proc_dirs = sorted(glob.glob(f"{output_dir}/proc_*"))

    for proc_dir in proc_dirs:
        # サブディレクトリ内から直接フレームを取得
        for frame_path in glob.glob(f"{proc_dir}/{base_name}_*.png"):
            dest_path = output_dir / Path(frame_path).name
            # 上書き防止（あれば削除）
            if dest_path.exists():
                dest_path.unlink()
            shutil.move(frame_path, dest_path)
        # 空になったproc_*ディレクトリは削除
        shutil.rmtree(proc_dir)

    all_frames = sorted(
        (
            p
            for p in glob.glob(f"{output_dir}/{base_name}_*.png")
            if _FRAME_RE.search(p)
        ),
        key=_extract_index,
    )

    # If no frames are found, raise an error
    if not all_frames:
        raise FileNotFoundError(f"No frames found in {output_dir}/ for animation.")

    anim_base_name = f"{base_name[:-len_suffix]}"

    if not generate_gif and not generate_mp4:
        print(
            f"Frames have been generated and saved as PNG files. No animation created."
        )
        return None
    # Create GIF animation
    if generate_gif:
        output_gif = f"{anim_base_name}.gif"
        create_gif(all_frames, output_gif=output_gif, fps=fps, cleanup=cleanup)
    # Create MP4 animation
    if generate_mp4:
        output_mp4 = f"{anim_base_name}.mp4"
        # create_mp4(frames, output_mp4, fps=fps, cleanup=cleanup) # does not work
        convert_gif_to_mp4(output_gif, output_mp4)
    return anim_base_name

import glob
import os
import shutil
import pandas as pd
import re
import time
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from ..utils.helpers import FrameGenerator, create_gif, convert_gif_to_mp4

def prepare_contourf_args(
    data, *, vmin=None, vmax=None, levels=None, cmap="viridis"
):
    """Return kwargs dict for tricontourf / contourf.

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


def add_colorbar(fig: plt.Figure, mappable, *, cax=None, label=None, **cbar_opts):
    """Attach a colorbar to *fig* and return the Colorbar instance."""
    cbar = fig.colorbar(mappable, cax=cax, **cbar_opts)
    if label is not None:
        cbar.set_label(label)
    return cbar


def create_anim_2d_plot(plotter, processes, var_name, *, siglay=None, fps=10, generate_gif=True, generate_mp4=False,
                        cleanup=False, post_process_func=None,
                        opts: "FvcomPlotOptions | None" =None,  plot_kwargs: dict | None =None):
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
    
    # Generate movie frames using helper methods in helpers.py
    output_dir = f"frames_{var_name}"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # サブディレクトリを含めて削除
    os.makedirs(output_dir) 

    # ------------------------------------------------------------
    # 0.  Unify option source  (old-style kwargs  /  new-style opts)
    # ------------------------------------------------------------
    plot_kwargs = plot_kwargs or {}

    if opts is None:                           # --- old style only
        opts = FvcomPlotOptions.from_kwargs(**plot_kwargs)
    else:                                      # --- new style + extra kwargs
        opts.extra.update(plot_kwargs)

    # `plot_kwargs_final` は FrameGenerator → plot_2d にそのまま流れる
    plot_kwargs_final = {"opts": opts}

    frames = FrameGenerator.generate_frames(da=da, output_dir=output_dir, plotter=plotter, processes=processes, 
                                            base_name=base_name, post_process_func=post_process_func, **plot_kwargs_final)
    #print(f"frames={frames}") 
    # `proc_*` 内のフレームを `frames/` に統合（上書きする）
    proc_dirs = sorted(glob.glob(f"{output_dir}/proc_*"))
    #print(f"proc_dirs={proc_dirs}") 
    for proc_dir in proc_dirs:
        for frame in glob.glob(f"{proc_dir}/{base_name}_*.png"):
            dest_path = os.path.join(output_dir, os.path.basename(frame))
            #print(f"dest_path={dest_path}")

            # 既存のファイルがあれば削除してから移動
            if os.path.exists(dest_path):
                os.remove(dest_path)  # 既存ファイルを削除して上書き

            # 上書き可能なので、直接 `shutil.move()` を実行
            shutil.move(frame, dest_path)

    # `proc_*` フォルダが空であれば削除（競合を防ぐ）
    #if not os.listdir(proc_dir):  # `proc_*` が空かチェック
    #    os.rmdir(proc_dir)

    #for proc_dir in proc_dirs:
    #    files = glob.glob(f"{proc_dir}/*.png")
    #    print(f"proc_dir={proc_dir}, files={files}") 

    #os.rmdir(proc_dir)  # `proc_{rank}/` を削除
    
    # `frames/` 内のフレームを収集
    #all_frames = sorted(glob.glob(f"{output_dir}/{base_name}_*.png"))
    all_frames = sorted(glob.glob(f"{output_dir}/{base_name}_*.png"), key=lambda x: int(re.search(r'_(\d+)\.png$', x).group(1)))
    # **デバッグ用: `all_frames` の中身を確認**
    #print(f"Collected {len(all_frames)} frames for GIF animation.")

    # **リストのリストになっていないかチェック**
    #if any(isinstance(f, list) for f in all_frames):
    #    pring(f"duplicated list exists.")
    #    all_frames = [item for sublist in all_frames for item in sublist]  # 二重リストを展開

    # `frames/` 内にフレームがない場合、エラーを出力
    if not all_frames:
        raise FileNotFoundError(f"No frames found in {output_dir}/ for animation.")


    # フレームが見つからない場合の処理
    #if not all_frames:
    #    print(f"Warning: No frames found in {output_dir}/")
    #    return
    
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
        create_gif(all_frames, output_gif=output_gif, fps=fps, cleanup=cleanup)
    # Create MP4 animation
    if generate_mp4:
        output_mp4 = f"{anim_base_name}.mp4"
        #create_mp4(frames, output_mp4, fps=fps, cleanup=cleanup) # does not work
        convert_gif_to_mp4(output_gif, output_mp4)
    return anim_base_name

import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import os
import imageio.v2 as imageio
from moviepy.editor import ImageSequenceClip
import multiprocessing
from multiprocessing import Pool
from .helpers_utils import clean_kwargs, unpack_plot_kwargs
import inspect
import subprocess
import cartopy.crs as ccrs
from tqdm import tqdm
#import dask
#from dask.delayed import delayed

def create_gif(frames, output_gif=None, fps=10, cleanup=False):
    """
    Create a GIF animation from a list of frames using memory-efficient processing.

    Parameters:
    - frames: List of file paths to the frames.
    - output_gif: Output file path for the GIF. Defaults to "output.gif".
    - fps: Frames per second for the GIF.
    - cleanup: If True, delete the frame files after creating the GIF.

    Returns:
    - None
    """
    if output_gif is None:
        output_gif = "Animation.gif"
    output_gif = os.path.expanduser(output_gif)

    # GIF作成の逐次処理（バッチなし）
    with imageio.get_writer(output_gif, mode="I", fps=fps) as writer:
        for frame in tqdm(frames, desc="Creating GIF", unit="frame"):
            writer.append_data(imageio.imread(frame))

    # フレーム削除処理
    if cleanup:
        for frame in tqdm(frames, desc="Cleaning up frames", unit="frame"):
            os.remove(frame)

    print(f"GIF animation saved at: {output_gif}")


def create_gif_with_batch(frames, output_gif=None, fps=10, batch_size=500, cleanup=False):
    """
    Obsolete: This may not be superior to create_gif.
    Create a GIF animation from a list of frames with memory-efficient batch processing.

    Parameters:
    - frames: List of file paths to the frames.
    - output_gif: Output file path for the GIF. Defaults to "output.gif".
    - fps: Frames per second for the GIF.
    - cleanup: If True, delete the frame files after creating the GIF.
    - batch_size: Number of frames to process at a time to reduce memory usage.

    Returns:
    - None
    """
    if output_gif is None:
        output_gif = "Animation.gif"  # Default GIF file name
    output_gif = os.path.expanduser(output_gif)

    # Temporary files for batched GIFs
    temp_gifs = []
    total_batches = (len(frames) + batch_size - 1) // batch_size  # Calculate total number of batches

    # Process frames in batches
    for i, batch_start in enumerate(range(0, len(frames), batch_size), start=1):
        batch_frames = frames[batch_start:batch_start + batch_size]
        temp_gif = f"{output_gif}_batch_{i}.gif"
        temp_gifs.append(temp_gif)

        with imageio.get_writer(temp_gif, mode="I", fps=fps) as writer:
            for frame in tqdm(batch_frames, desc=f"Batch {i}/{total_batches}", unit="frame"):
                writer.append_data(imageio.imread(frame))

    # Combine batched GIFs into final GIF
    with imageio.get_writer(output_gif, mode="I", fps=fps) as writer:
        for temp_gif in tqdm(temp_gifs, desc="Combining Batches", unit="batch"):
            with imageio.get_reader(temp_gif) as reader:
                for frame in reader:
                    writer.append_data(frame)

    # Cleanup temporary files
    if cleanup:
        for temp_gif in temp_gifs:
            os.remove(temp_gif)
        for frame in frames:
            os.remove(frame)

    print(f"GIF animation saved at: {output_gif}")

def create_mp4(frames, output_mp4=None, fps=10, cleanup=True):
    """
    Create an MP4 animation from a list of frames.
    Currently this function does not work as expected. Use convert_gif_to_mp4 instead.

    Parameters:
    - frames: List of file paths to the frames (e.g., PNG files).
    - output_mp4: Output file path for the MP4. Defaults to "output.mp4".
    - fps: Frames per second for the animation.
    - cleanup: If True, delete the frame files after creating the MP4.

    Returns:
    - None
    """
    if output_mp4 is None:
        output_mp4 = "output.mp4"  # Default MP4 file name

    # Generate MP4 using moviepy
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(output_mp4, codec="libx264", fps=fps, pix_fmt="yuv420p")

    if cleanup:
        for frame in frames:
            os.remove(frame)

    #print(f"Saved the MP4 animation as '{output_mp4}'.")

def convert_gif_to_mp4(input_gif, output_mp4):
    """
    Convert a GIF animation to an MP4 video using ffmpeg.

    Parameters:
    - input_gif: Path to the input GIF file.
    - output_mp4: Path to the output MP4 file.

    Returns:
    - None
    """
    command = [
        "ffmpeg",
        "-y",                      # Skip overwrite confirmation
        "-i", input_gif,           # Input GIF file
        "-movflags", "+faststart", # Enable fast start for streaming
        "-pix_fmt", "yuv420p",     # Pixel format for compatibility
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure even dimensions
        output_mp4
    ]
    # subprocess.run(command, check=True)
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Converted from {input_gif} to {output_mp4}.")


def create_gif_from_frames(frames_dir, output_gif, fps=10, prefix=None, batch_size=500, cleanup=False):
    """
    Create a GIF animation from PNG frames in batches to handle memory constraints.

    Parameters:
    - frames_dir: Directory containing PNG frames.
    - output_gif: Path to save the GIF animation.
    - fps: Frames per second for the GIF.
    - prefix: Filter files that start with this prefix (e.g., 'salinity').
    - batch_size: Number of frames to process in each batch.
    - cleanup: If True, delete PNG frames after creating the GIF.

    Returns:
    - None
    """
    frames = sorted([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.endswith('.png') and (prefix is None or f.startswith(prefix))
    ])
    #frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
    temp_gifs = []  # Temporary GIFs for each batch

    total_batches = (len(frames) + batch_size - 1) // batch_size  # 総バッチ数を計算

    for i, batch_start in enumerate(range(0, len(frames), batch_size), start=1):
        batch_frames = frames[batch_start:batch_start+batch_size]
        temp_gif = f"{output_gif}_batch_{i}.gif"
        temp_gifs.append(temp_gif)

        with imageio.get_writer(temp_gif, mode="I", fps=fps) as writer:
            for frame in tqdm(batch_frames, desc=f"Batch {i}/{total_batches}", unit="frame"):
                writer.append_data(imageio.imread(frame))
    print("Temporary GIFs created successfully.")
    
    # Combine temporary GIFs into the final GIF
    #with imageio.get_writer(output_gif, mode="I", fps=fps) as writer:
    #    for temp_gif in temp_gifs:
    #        gif_reader = imageio.get_reader(temp_gif)
    #        for frame in gif_reader:
    #            writer.append_data(frame)
    #        gif_reader.close()
    #        if cleanup:
    #            os.remove(temp_gif)
    
    # Combine temporary GIFs into the final GIF
    # Robust version of the above code
    with imageio.get_writer(output_gif, mode="I", fps=fps) as writer:
        for temp_gif in tqdm(temp_gifs, desc="Combining Batches", unit="batch"):
            with imageio.get_reader(temp_gif) as reader:
                for frame in reader:
                    writer.append_data(frame)

    if cleanup:
        for frame in frames:
            os.remove(frame)

    print(f"GIF Animation created successfully at: {output_gif}")


class FrameGenerator:
    @staticmethod
    def generate_frame(args):
        """
        Default frame generation logic.
        """
        cls, time, data_array, plotter, output_dir, base_name, post_process_func, plot_kwargs = args
        
        # Unpack and clean plot_kwargs
        #print("Original plot_kwargs:", plot_kwargs)
        plot_kwargs = unpack_plot_kwargs(plot_kwargs)
        #print("Unpacked plot_kwargs:", plot_kwargs)
        plot_kwargs = clean_kwargs(plotter.plot_2d, plot_kwargs)
        #print("Cleaned plot_kwargs:", plot_kwargs)
        save_path = os.path.join(output_dir, f"{base_name}_{time}.png")

        cls.plot_data(data_array, time, plotter, save_path, post_process_func, plot_kwargs)
        #da = data_array.isel(time=time)
        #plotter.plot_2d(da=da, save_path=save_path, post_process_func=post_process_func, **plot_kwargs)
        return save_path

    @staticmethod
    def plot_data(data_array, time=None, plotter=None, save_path=None, post_process_func=None, plot_kwargs=None):
        """
        Generate a single frame with the given parameters.

        Parameters:
        - data_array: DataArray to plot.
        - time: Time index to select from the DataArray.
        - plotter: FvcomPlotter instance used for plotting.
        - save_path: Path to save the generated frame.
        - post_process_func: Function to apply custom processing to the plot.
        - plot_kwargs: Additional arguments for the plot.
        """
        # Extract the data_array for the given time index if specified.
        if time is not None:
            da = data_array.isel(time=time)
        else:
            da = data_array

        # Call the plotter's plot_2d method with the given arguments.
        def wrapped_post_process_func(ax):
            if post_process_func:
                # This function is used to dynamically pass the required arguments if specified.
                func_args = inspect.signature(post_process_func).parameters

                # Dynamically pass the required arguments (da, time).
                kwargs = {"ax": ax}
                if "da" in func_args:
                    kwargs["da"] = da
                if "time" in func_args:
                    kwargs["time"] = time
                
                post_process_func(**kwargs)
        
        return plotter.plot_2d(da=da, save_path=save_path, post_process_func=wrapped_post_process_func, **plot_kwargs)
    
    @classmethod
    def generate_frames(cls, data_array, output_dir, plotter, processes, base_name="frame", post_process_func=None, **plot_kwargs):
        """
        Generate frames using multiprocessing with the class's generate_frame method.
        """
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        time_indices = range(data_array.sizes["time"])
        args_list = [(cls, time, data_array, plotter, output_dir, base_name, post_process_func, plot_kwargs) for time in time_indices]

        # Check the number of available processes
        max_procs = multiprocessing.cpu_count()
        # スパコンのジョブ環境で設定されたプロセス数を取得（なければNone）
        job_procs = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("PBS_NP") or os.environ.get("LSB_DJOB_NUMPROC")

        # 取得した環境変数を整数に変換（環境変数が None の場合はデフォルトを max_procs に）
        if job_procs:
            job_procs = int(job_procs)
        else:
            job_procs = max_procs  # 環境変数がない場合は最大コア数を仮の値とする

        print(f"Total available cores: {job_procs}")
        
        # 指定した processes をチェック
        if processes > job_procs:
            raise ValueError(f"Error: The specified processes ({processes}) exceed the available job processes ({job_procs}).")

        # Use the class's generate_frame method
        with Pool(processes=processes) as pool:
            frames = pool.map(cls.generate_frame, args_list)

        return frames


class PlotHelperMixin:
    """
    A mixin class to provide helper methods for batch plotting and other common operations.
    """

    def plot_timeseries_in_batches(self, plotter, vars, index, log=None, batch_size=4, k=None, start=None, end=None, save_prefix="plot", **kwargs):
        """
        Plot variables in batches.

        Parameters:
        - plotter: FvcomPlotter object
        - vars: List of variable names to plot.
        - index: Index of the node or nele to plot.
        - batch_size: Number of variables per figure.
        - start, end: Start and end times for the time series.
        - save_prefix: Prefix for saved file names (e.g., "plot").
        - **kwargs: Additional arguments for customization.
        """

        if not isinstance(vars, list) or len(vars) == 0:
            print("ERROR: Variable names are not included in 'vars' list.")
            return None
        
        # 分割数を計算
        num_batches = ceil(len(vars) / batch_size)

        for batch_num in range(num_batches):
            # 対象の変数を抽出
            batch_vars = vars[batch_num * batch_size : (batch_num + 1) * batch_size]

            # 図の作成
            fig, axes = plt.subplots(len(batch_vars), 1, figsize=(10, 3 * len(batch_vars)), sharex=True)
            if len(batch_vars) == 1:
                axes = [axes]  # 変数が1つの場合、axesをリストにする

            # 各変数のプロット
            for var, ax in zip(batch_vars, axes):
                plotter.plot_timeseries(var_name=var, index=index, log=log, k=k, start=start, end=end, ax=ax, **kwargs)
                ax.set_title(var, fontsize=14)

            # 図全体の調整
            fig.suptitle(f"Time Series Batch {batch_num + 1}", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # タイトルとプロット間のスペース調整

            # 保存または表示
            save_path = f"{save_prefix}_batch_{batch_num + 1}.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        print(f"Saved {num_batches} figures as '{save_prefix}_batch_#.png'.")

    def plot_timeseries_for_river_in_batches(self, plotter, var_name, batch_size=4, start=None, end=None, save_prefix="river_plot", **kwargs):
        """
        Plot a single variable for all rivers in batches.

        Parameters:
        - plotter: FvcomPlotter object
        - var_name: Variable name to plot.
        - batch_size: Number of rivers per figure.
        - start, end: Start and end times for the time series.
        - save_prefix: Prefix for saved file names (e.g., "river_plot").
        - **kwargs: Additional arguments for customization.
        """

        # Rivers の次元サイズを取得
        if "river_names" in self.ds:
            num_rivers = self.ds["river_names"].sizes["rivers"]
        else:
            print("ERROR: No 'river_names' variable found.")
            return None

        # バッチの数を計算
        num_batches = ceil(num_rivers / batch_size)

        for batch_num in range(num_batches):
            # バッチごとのインデックス範囲を計算
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, num_rivers)
            river_indices = range(start_idx, end_idx)

            # 図の作成
            fig, axes = plt.subplots(len(river_indices), 1, figsize=(10, 3 * len(river_indices)), sharex=True)
            if len(river_indices) == 1:
                axes = [axes]  # rivers が1つの場合でもリスト化

            # 各 river のプロット
            for river_index, ax in zip(river_indices, axes):
                plotter.plot_timeseries_for_river(
                    var_name=var_name, river_index=river_index, start=start, end=end, ax=ax, **kwargs
                )
                #ax.set_title(f"{var_name} for river {river_index}", fontsize=14)

            # 図全体の調整
            fig.suptitle(f"Time Series of {var_name} (Batch {batch_num + 1})", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # タイトルとプロット間のスペース調整

            # 保存または表示
            save_path = f"{save_prefix}_batch_{batch_num + 1}.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        print(f"Saved {num_batches} figures as '{save_prefix}_batch_#.png'.")

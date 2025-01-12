import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import os
import imageio.v2 as imageio
from multiprocessing import Pool
from .helpers_utils import clean_kwargs, unpack_plot_kwargs

'''
def generate_frame(args):
    """
    Generate a frame for a GIF animation.

    Parameters:
    - args: Tuple containing (time, data_array, plotter, output_dir, base_name, post_process_func, plot_kwargs).

    Returns:
    - File path to the generated frame.
    """

    time, data_array, plotter, output_dir, base_name, post_process_func, plot_kwargs = args
    # Unpack and clean plot_kwargs
    #print("Original plot_kwargs:", plot_kwargs)
    plot_kwargs = unpack_plot_kwargs(plot_kwargs)
    #print("Unpacked plot_kwargs:", plot_kwargs)
    plot_kwargs = clean_kwargs(plotter.plot_2d, plot_kwargs)
    #print("Cleaned plot_kwargs:", plot_kwargs)


    da = data_array.isel(time=time)
    save_path = os.path.join(output_dir, f"{base_name}_{time}.png")

    plotter.plot_2d(da=da, save_path=save_path, post_process_func=post_process_func, **plot_kwargs)

    return save_path
'''

def create_gif(frames, output_gif=None, fps=10, cleanup=True):
    """
    Create a GIF animation from a list of frames.

    Parameters:
    - frames: List of file paths to the frames.
    - output_gif: Output file path for the GIF. - output_gif: Output file path for the GIF. 
        If None, defaults to "output.gif". `~/` will be expanded to the user's home directory.
    - fps: Frames per second for the GIF.
    - cleanup: If True, delete the frame files after creating the GIF.

    Returns:
    - None
    """

    if output_gif is None:
        output_gif = "output.gif"  # Default GIF file name
    output_gif = os.path.expanduser(output_gif)
    
    with imageio.get_writer(output_gif, mode="I", fps=fps) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)

    if cleanup:
        for frame in frames:
            os.remove(frame)

    print(f"Saved the GIF animation as '{output_gif}'.")


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
    def plot_data(data_array, time, plotter, save_path, post_process_func, plot_kwargs):
        """
        プロット処理を分離
        """
        da = data_array.isel(time=time)
        #plotter.plot_2d(da=da, save_path=save_path, **plot_kwargs)
        plotter.plot_2d(da=da, save_path=save_path, post_process_func=post_process_func, **plot_kwargs)

    @classmethod
    def generate_frames(cls, data_array, output_dir, plotter, base_name="frame", post_process_func=None, **plot_kwargs):
        """
        Generate frames using multiprocessing with the class's generate_frame method.
        """
        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        time_indices = range(data_array.sizes["time"])
        args_list = [
            (cls, time, data_array, plotter, output_dir, base_name, post_process_func, plot_kwargs) for time in time_indices
        ]

        # Use the class's generate_frame method
        with Pool() as pool:
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

    # Create a GIF animation using FvcomPlotter.plot_2d method
    '''
    def generate_frames(self, data_array, output_dir, plotter, base_name="frame", post_process_func=None, **plot_kwargs):
        """
        Generate frames for a GIF animation.

        Parameters:
        - data_array: xarray.DataArray (dimensions: time, node) to plot.
        - output_dir: Directory to save the frames. `~` will be expanded to the user's home directory.
        - plotter: An instance of FvcomPlotter.
        - base_name: Base name for frame files (default: "frame").
        - post_process_func: Function to apply custom plots or decorations to the Axes.
        - **kwargs: Additional arguments passed to the plotter.

        Returns:
        - A list of file paths to the generated frames.
        """

        output_dir = os.path.expanduser(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
        time_indices = range(data_array.sizes["time"])  # 全timeステップ

        # Arguments for the generate_frame function
        args_list = [
            (time, data_array, plotter, output_dir, base_name, post_process_func, plot_kwargs) for time in time_indices
        ]

        # Generate frames in parallel
        with Pool() as pool:
            frames = pool.map(generate_frame, args_list)

        return frames
    '''

    def create_gif(self, frames, output_gif=None, fps=10, cleanup=True):
        """
        Create a GIF animation from a list of frames.

        Parameters:
        - frames: List of file paths to the frames.
        - output_gif: Output file path for the GIF. - output_gif: Output file path for the GIF. 
          If None, defaults to "output.gif". `~/` will be expanded to the user's home directory.
        - fps: Frames per second for the GIF.
        - cleanup: If True, delete the frame files after creating the GIF.

        Returns:
        - None
        """

        if output_gif is None:
            output_gif = "output.gif"  # Default GIF file name
        output_gif = os.path.expanduser(output_gif)
        
        with imageio.get_writer(output_gif, mode="I", fps=fps) as writer:
            for frame in frames:
                image = imageio.imread(frame)
                writer.append_data(image)

        if cleanup:
            for frame in frames:
                os.remove(frame)

        print(f"Saved the GIF animation as '{output_gif}'.")


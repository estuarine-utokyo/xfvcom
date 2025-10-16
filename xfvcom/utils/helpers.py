from __future__ import annotations

"""
Helper functions and classes for creating GIF/MP4 animations,
generating plot frames in parallel, and cleaning up temporary files.

This module provides:
- Listing PNG frame files from a directory.
- Creating GIF animations from image frames (including batched processing).
- Converting GIF to MP4 via ffmpeg.
- Parallel frame generation via FrameGenerator.
- Centralized cleanup of files with optional progress display.
- PlotHelperMixin for batch‐plot utilities.

Requires Python 3.10+ with `from __future__ import annotations`.
Uses `pathlib.Path` for I/O and standard `logging` for status messages.
"""

from typing import TYPE_CHECKING, Any, Callable, Sequence

if TYPE_CHECKING:
    # Forward-reference only for static type-checking; avoids import cycle
    from ..plot.core import FvcomPlotter

import inspect
import logging
import multiprocessing
import os
import shutil
import subprocess
from math import ceil
from multiprocessing import Pool
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

from ..plot_options import FvcomPlotOptions
from ..utils.helpers_utils import clean_kwargs, unpack_plot_kwargs

# import dask
# from dask.delayed import delayed


# Create module‐level logger
logger = logging.getLogger(__name__)


def _cleanup_files(paths: list[Path], show_progress: bool = False) -> None:
    """
    Delete each file in `paths`.

    Parameters
    ----------
    paths : list[Path]
        File paths to remove.
    show_progress : bool, default False
        If True, show a progress bar.
    """
    iterator = (
        tqdm(paths, desc="Cleaning up files", unit="file") if show_progress else paths
    )
    for p in iterator:
        p.unlink()


# -----------------------------------------------------------------------------
# Directory Utility: ensure directory exists (with optional clean)
# -----------------------------------------------------------------------------
def ensure_dir(path: Path, clean: bool = False) -> None:
    """
    Create directory at `path`. If clean=True and it already exists, remove it first.
    """
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# File/Directory Utilities
# -----------------------------------------------------------------------------
def list_png_files(frames_dir: Path, prefix: str | None = None) -> list[Path]:
    """
    Return a sorted list of PNG files in the given directory.

    Parameters
    ----------
    frames_dir : Path
        Directory to search for `.png` files.
    prefix : str | None, optional
        If given, only include files starting with this prefix.

    Returns
    -------
    list[Path]
        Sorted list of matching PNG file paths.
    """
    if not frames_dir.exists():
        raise FileNotFoundError(f"{frames_dir} does not exist")
    pattern = f"{prefix}_*.png" if prefix else "*.png"
    return sorted(frames_dir.glob(pattern))


def generate_frames_and_collect(
    plotter: FvcomPlotter,
    processes: int,
    var_name: str,
    output_dir: Path,
    siglay: int | None = None,
    post_process_func: Callable | None = None,
    opts: Any = None,
    **plot_kwargs: Any,
) -> list[Path]:
    """
    Wrapper that:
      1) Cleans and prepares `output_dir`
      2) Calls the existing FrameGenerator.parallel logic unchanged
      3) Collects and returns a sorted list of generated PNG frame paths

    This isolates I/O from the high-performance core.
    """
    # 1) prepare output directory (clean old frames)
    ensure_dir(output_dir, clean=True)

    # 2) invoke original parallel frame generation
    #    (FrameGenerator.generate_frames returns list of file-path strings)
    raw_frames = FrameGenerator.generate_frames(
        plotter=plotter,
        processes=processes,
        var_name=var_name,
        siglay=siglay,
        post_process_func=post_process_func,
        opts=opts,
        **plot_kwargs,
    )

    # 3) convert to Path and sort
    frame_paths = [Path(fp) for fp in raw_frames]
    return sorted(frame_paths)


def create_gif(
    frames: Sequence[str | Path],
    output_gif: str | Path | None = None,
    fps: int = 10,
    cleanup: bool = False,
) -> None:
    """
    Create a GIF animation from a list of image frames.

    Parameters
    ----------
    frames : list[str | Path]
        Paths to input image files.
    output_gif : str | Path, optional
        Path to the output GIF file. Defaults to “Animation.gif”.
    fps : int, default 10
        Frames per second for the GIF.
    cleanup : bool, default False
        If True, delete source frames after creating the GIF.
    """

    if output_gif is None:
        output_gif = "Animation.gif"
    # Expand user and wrap as Path
    output_gif = Path(output_gif).expanduser()
    # Create GIF using imageio without batch processing
    duration = int(1000 / fps)
    with imageio.get_writer(str(output_gif), mode="I", duration=duration) as writer:  # type: ignore[attr-defined]
        # with imageio.get_writer(output_gif, mode="I", fps=fps) as writer:
        for frame in tqdm(frames, desc="Creating GIF", unit="frame"):
            # ensure str for imageio
            writer.append_data(imageio.imread(str(frame)))  # type: ignore[attr-defined]
    #  - remove via pathlib
    if cleanup:
        # for frame in tqdm(frames, desc="Cleaning up frames", unit="frame"):
        #     # remove via pathlib
        #     Path(frame).unlink()
        _cleanup_files([Path(f) for f in frames], show_progress=True)
    logger.info("GIF animation saved at: %s", output_gif)


def create_gif_with_batch(
    frames: Sequence[str | Path],
    output_gif: str | Path | None = None,
    fps: int = 10,
    batch_size: int = 500,
    cleanup: bool = False,
) -> None:
    """
    Create a GIF animation by processing frames in batches.

    Parameters
    ----------
    frames : list[str | Path]
        Paths to input image files.
    output_gif : str | Path, optional
        Path to the final output GIF. Defaults to “Animation.gif”.
    fps : int, default 10
        Frames per second for both batch GIFs and the final GIF.
    batch_size : int, default 500
        Number of frames per intermediate batch GIF.
    cleanup : bool, default False
        If True, delete intermediate batch GIFs and source frames.
    """

    if output_gif is None:
        output_gif = "Animation.gif"  # Default GIF file name
    # expand user and wrap as Path
    output_gif = Path(output_gif).expanduser()

    # Temporary files for batched GIFs
    temp_gifs = []
    total_batches = (
        len(frames) + batch_size - 1
    ) // batch_size  # Calculate total number of batches

    # Process frames in batches
    duration = int(1000 / fps)
    for i, batch_start in enumerate(range(0, len(frames), batch_size), start=1):
        batch_frames = frames[batch_start : batch_start + batch_size]
        temp_gif = f"{output_gif}_batch_{i}.gif"
        temp_gifs.append(temp_gif)
        with imageio.get_writer(str(temp_gif), mode="I", duration=duration) as writer:  # type: ignore[attr-defined]
            # with imageio.get_writer(temp_gif, mode="I", fps=fps) as writer:
            for frame in tqdm(
                batch_frames, desc=f"Batch {i}/{total_batches}", unit="frame"
            ):
                writer.append_data(imageio.imread(frame))  # type: ignore[attr-defined]

    # Combine batched GIFs into final GIF
    # with imageio.get_writer(output_gif, mode="I", fps=fps) as writer:
    with imageio.get_writer(str(output_gif), mode="I", duration=duration) as writer:  # type: ignore[attr-defined]
        for temp_gif in tqdm(temp_gifs, desc="Combining Batches", unit="batch"):
            with imageio.get_reader(temp_gif) as reader:
                for frame in reader:  # type: ignore[attr-defined]
                    writer.append_data(frame)  # type: ignore[attr-defined]

    # Cleanup temporary files
    if cleanup:
        # for temp_gif in temp_gifs:
        #     # remove file via pathlib
        #     Path(temp_gif).unlink()
        # for frame in frames:
        #     Path(frame).unlink()
        _cleanup_files([Path(p) for p in temp_gifs] + [Path(f) for f in frames])
    logger.info("GIF animation saved at: %s", output_gif)


def create_mp4(
    frames: Sequence[str | Path],
    output_mp4: str | Path | None = None,
    fps: int = 10,
    cleanup: bool = True,
) -> None:
    """
    Create an MP4 animation from a list of image frames.

    Note
    ----
    This function currently relies on moviepy and may not work as expected;
    consider using `convert_gif_to_mp4` for more robust GIF→MP4 conversion.

    Parameters
    ----------
    frames : list[str | Path]
        Paths to input image files (e.g., PNG frames).
    output_mp4 : str | Path, optional
        Path to write the MP4 file. Defaults to "output.mp4".
    fps : int, default 10
        Frames per second for the resulting video.
    cleanup : bool, default True
        If True, delete source frame files after writing the MP4.
    """

    # Lazy import – give clear guidance if moviepy is missing
    try:
        from moviepy.editor import ImageSequenceClip
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "moviepy is required for MP4 generation.\n"
            "Install with:  pip install 'xfvcom[video]'   または   pip install moviepy"
        ) from e

    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(
        output_mp4, codec="libx264", fps=fps, pix_fmt="yuv420p", logger=None
    )

    # if output_mp4 is None:
    #     output_mp4 = "output.mp4"  # Default MP4 file name

    # # Generate MP4 using moviepy
    # clip = ImageSequenceClip(frames, fps=fps)
    # clip.write_videofile(output_mp4, codec="libx264", fps=fps, pix_fmt="yuv420p")

    if cleanup:
        # for frame in frames:
        #     Path(frame).unlink()
        _cleanup_files([Path(f) for f in frames])


def convert_gif_to_mp4(
    input_gif: str | Path,
    output_mp4: str | Path,
) -> None:
    """
    Convert an existing GIF animation to MP4 format via ffmpeg.

    Parameters
    ----------
    input_gif : str | Path
        Path to source GIF file.
    output_mp4 : str | Path
        Desired path for output MP4 file.

    Raises
    ------
    Mp4ConversionError
        If ffmpeg returns a non-zero exit code.
    """

    command = [
        "ffmpeg",
        "-y",  # Skip overwrite confirmation
        "-i",
        input_gif,  # Input GIF file
        "-movflags",
        "+faststart",  # Enable fast start for streaming
        "-pix_fmt",
        "yuv420p",  # Pixel format for compatibility
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure even dimensions
        output_mp4,
    ]
    # subprocess.run(command, check=True)
    subprocess.run(
        command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    logger.info("Converted GIF %s to MP4 %s", input_gif, output_mp4)


def create_gif_from_frames(
    frames_dir: str | Path,
    output_gif: str | Path,
    fps: int = 10,
    prefix: str | None = None,
    batch_size: int = 500,
    cleanup: bool = False,
) -> None:
    """
    Create a GIF animation by reading all PNG frames in a directory.

    Parameters
    ----------
    frames_dir : str | Path
        Directory containing `.png` frames.
    output_gif : str | Path
        Path to the output GIF file.
    fps : int, default 10
        Frames per second for the GIF.
    cleanup : bool, default False
        If True, delete source frames after creating the GIF.
    """

    # Convert frames_dir to Path object and gather PNG frames
    frames_path = Path(frames_dir).expanduser()
    frames = list_png_files(frames_path, prefix)

    temp_gifs = []  # Temporary GIFs for each batch

    total_batches = (len(frames) + batch_size - 1) // batch_size  # 総バッチ数を計算

    for i, batch_start in enumerate(range(0, len(frames), batch_size), start=1):
        batch_frames = frames[batch_start : batch_start + batch_size]
        temp_gif = f"{output_gif}_batch_{i}.gif"
        temp_gifs.append(temp_gif)

        duration = int(1000 / fps)
        with imageio.get_writer(str(output_gif), mode="I", duration=duration) as writer:  # type: ignore[assignment]
            # with imageio.get_writer(temp_gif, mode="I", fps=fps) as writer:
            for frame in tqdm(
                batch_frames, desc=f"Batch {i}/{total_batches}", unit="frame"
            ):
                writer.append_data(imageio.imread(frame))  # type: ignore[attr-defined]
    logger.info("Temporary GIFs created successfully.")

    # Combine temporary GIFs into the final GIF
    # Robust version of the above code
    with imageio.get_writer(output_gif, mode="I", fps=fps) as writer:  # type: ignore[attr-defined]
        for temp_gif in tqdm(temp_gifs, desc="Combining Batches", unit="batch"):
            with imageio.get_reader(temp_gif) as reader:
                for frame in reader:  # type: ignore[attr-defined]
                    writer.append_data(frame)  # type: ignore[attr-defined]

    if cleanup:
        # for frame in frames:
        #     # Remove frame file to clean up
        #     Path(frame).unlink()
        _cleanup_files([Path(f) for f in frames])
    logger.info("GIF Animation created successfully at: %s", output_gif)


class FrameGenerator:
    @staticmethod
    def generate_frame(args):
        """
        Default frame generation logic.
        """
        (
            cls,
            time,
            da,
            plotter,
            output_dir,
            base_name,
            post_process_func,
            plot_kwargs,
        ) = args

        # verbose flag (default: False)
        verbose = plot_kwargs.pop("verbose", False)

        # Unpack and clean plot_kwargs
        plot_kwargs = clean_kwargs(plotter.plot_2d, unpack_plot_kwargs(plot_kwargs))

        if verbose:
            print(f"[FrameGenerator] time={time}, kwargs={plot_kwargs}")

        # use pathlib for constructing the file path
        save_path = Path(output_dir) / f"{base_name}_{time}.png"

        cls.plot_data(
            da=da,
            time=time,
            plotter=plotter,
            save_path=save_path,
            post_process_func=post_process_func,
            **plot_kwargs,
        )
        return save_path

    @staticmethod
    def generate_frame_batch(args):
        """
        各プロセスが `time_slices` に含まれる複数の time ステップを一括処理
        """

        (
            cls,
            time_slice,
            da,
            plotter,
            output_dir,
            base_name,
            post_process_func,
            plot_kwargs,
        ) = args
        verbose = plot_kwargs.pop("verbose", False)

        # Unpack and clean plot_kwargs
        plot_kwargs = unpack_plot_kwargs(plot_kwargs)
        plot_kwargs = clean_kwargs(plotter.plot_2d, plot_kwargs)

        # 各プロセス専用フォルダを作成（ディスク I/O 競争を防止）
        rank = multiprocessing.current_process()._identity[0]  # プロセスID
        # use pathlib for process‐specific output directory
        proc_output_dir = Path(output_dir) / f"proc_{rank}"
        proc_output_dir.mkdir(parents=True, exist_ok=True)

        # メモリ上に確保（I/O オーバーヘッド削減）
        da = da.isel(time=time_slice).load()

        frames = []
        class_label = cls.__name__  # e.g., "FrameGenerator"
        for i, t in enumerate(time_slice):
            if verbose:
                print(f"[{class_label}] rank={rank}  time={t}")
            save_path = Path(proc_output_dir) / f"{base_name}_{t}.png"
            frame_data = da.isel(time=i) if "time" in da.dims else da

            cls.plot_data(
                da=frame_data,
                time=t,
                plotter=plotter,
                save_path=save_path,
                post_process_func=post_process_func,
                **plot_kwargs,
            )
            frames.append(save_path)

        return frames

    @staticmethod
    def plot_data(
        *,
        da: xr.DataArray | None = None,
        time: int | None = None,
        plotter: "FvcomPlotter",
        save_path: str | None = None,
        post_process_func: Callable[..., None] | None = None,
        opts: FvcomPlotOptions | None = None,
        **plot_kwargs,
    ):
        """
        Generate a single frame with the given parameters.

        Parameters:
        - da: DataArray to plot. Plot only mesh if da is None (default).
        - time: Time index to select from the DataArray.
        - plotter: FvcomPlotter instance used for plotting.
        - save_path: Path to save the generated frame.
        - post_process_func: Function to apply custom processing to the plot.
        - opts: FvcomPlotOptions instance for plot options.
        - **plot_kwargs: Additional arguments for the plot.

        Returns:
        - FvcomPlotter.plot_2d: The plotter instance used for plotting.
        """

        # --- unify option source --------------------------------
        if opts is None:
            # still old-style → convert
            opts = FvcomPlotOptions.from_kwargs(**plot_kwargs)
        else:
            # new-style + extra kwargs → merge into opts.extra
            opts.extra.update(plot_kwargs)

        if da is None and not opts.with_mesh:
            raise ValueError("'da' is None and 'with_mesh' is False — nothing to plot.")

        if da is not None and "time" in da.dims and time is not None:
            da = da.isel(time=time)

        # Wrap post-process callback
        def _wrapped(ax: plt.Axes) -> None:
            if post_process_func is None:
                return
            sig = inspect.signature(post_process_func).parameters
            kw: dict[str, Any] = {"ax": ax}
            if "da" in sig:
                kw["da"] = da
            if "time" in sig:
                kw["time"] = time
            post_process_func(**kw)

        return plotter.plot_2d(
            da=da, save_path=save_path, post_process_func=_wrapped, opts=opts
        )

    @classmethod
    def generate_frames(
        cls,
        da,
        output_dir,
        plotter,
        processes,
        base_name="frame",
        post_process_func=None,
        **plot_kwargs,
    ):
        """
        Generate frames using multiprocessing with the class's generate_frame method.
        """
        # prepare output directory
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        # time_indices = range(da.sizes["time"])
        # args_list = [(cls, time, da, plotter, output_dir, base_name, post_process_func, plot_kwargs) for time in time_indices]

        # Check the number of available processes
        max_procs = multiprocessing.cpu_count()
        # スパコンのジョブ環境で設定されたプロセス数を取得（なければNone）
        job_procs = (
            os.environ.get("SLURM_CPUS_PER_TASK")
            or os.environ.get("PBS_NP")
            or os.environ.get("LSB_DJOB_NUMPROC")
        )

        # 取得した環境変数を整数に変換（環境変数が None の場合はデフォルトを max_procs に）
        if job_procs:
            job_procs = int(job_procs)
        else:
            job_procs = max_procs  # 環境変数がない場合は最大コア数を仮の値とする

        print(f"Total available cores: {job_procs}")

        # 指定した processes をチェック
        if processes > job_procs:
            raise ValueError(
                f"Error: The specified processes ({processes}) exceed the available job processes ({job_procs})."
            )

        time_size = da.sizes["time"]
        time_slices = np.array_split(range(time_size), processes)
        args_list = [
            (
                cls,
                time_slice,
                da,
                plotter,
                output_dir,
                base_name,
                post_process_func,
                plot_kwargs,
            )
            for time_slice in time_slices
        ]

        # Use the class's generate_frame method
        # with Pool(processes=processes) as pool:
        #    frames = pool.map(cls.generate_frame, args_list)

        with Pool(processes=processes) as pool:
            frames = pool.map(cls.generate_frame_batch, args_list)

        # return frames
        # --------------------------------------------------------------------
        # Merge proc_* subdirectories into output_dir and remove them
        # --------------------------------------------------------------------
        merged = []
        # for each proc_X folder
        for proc_dir in output_path.glob("proc_*"):
            for fp in proc_dir.glob(f"{base_name}_*.png"):
                dst = output_path / fp.name
                if dst.exists():
                    dst.unlink()  # remove old copy
                fp.rename(dst)  # move into output_dir
                merged.append(dst)
            proc_dir.rmdir()  # remove empty proc_* dir

        # If user code expected the flat list of frame paths, return that:
        #  - use merged list if non-empty, else fall back to original frames
        final_paths = merged if merged else [Path(f) for f in frames]
        # sort by the numeric suffix in file name
        # from .utils import _extract_index

        # final_paths.sort(key=lambda p: _extract_index(str(p)))
        # return list of path strings for backward compatibility
        # return [str(p) for p in final_paths]
        # Inline sort by numeric suffix (e.g. "..._123.png")
        import re

        FRAME_RE = re.compile(r"_(\d+)\.png$")

        def _extract_index(path_obj):
            m = FRAME_RE.search(path_obj.name)
            return int(m.group(1)) if m else -1

        final_paths.sort(key=_extract_index)
        # Return list of path‐strings for backward compatibility
        return [str(p) for p in final_paths]


class PlotHelperMixin:
    """
    A mixin class to provide helper methods for batch plotting and other common operations.
    """

    ds: Any  # added for static-type checkers
    cfg: Any
    ts_river: Callable[..., Any]
    ts_plot: Callable[..., Any]

    def ts_plot_in_batches(
        self,
        varnames: list[str],
        index: int,
        batch_size: int = 4,
        k: int | None = None,
        png_prefix: str = "plot",
        **kwargs: Any,
    ) -> None:
        """
        Batch-plot multiple variables (time-series) in groups of `batch_size`,
        delegating each subplot to self.ts_plot().

        Parameters:
        valnames: List of variable names to plot.
        index: Index of the node or nele to plot.
        batch_size: Number of variables per figure.
        png_prefix: Prefix for saved PNG file names (e.g., "PNG/time_node").
        k: Optional index for the variable to plot.
        **kwargs: Transferred to self.ts_plot(**kwargs).
        """
        # ------------------------------------------------------------
        # 0) Basic validation
        # ------------------------------------------------------------
        if not isinstance(varnames, list) or len(varnames) == 0:
            raise ValueError("'varnames' must be a non-empty list.")

        # ------------------------------------------------------------
        # 1) Split into batches
        # ------------------------------------------------------------
        num_batches = ceil(len(varnames) / batch_size)

        # ------------------------------------------------------------
        # 2) Pop plotting-specific kwargs
        # ------------------------------------------------------------
        dpi = kwargs.pop("dpi", self.cfg.dpi)
        sharex = kwargs.pop("sharex", True)
        legend = kwargs.pop("legend", True)
        title = kwargs.pop("title", True)
        suptitle = kwargs.pop("suptitle", True)
        # ------------------------------------------------------------

        for b in range(num_batches):
            # ---- 3-1) Variables in this batch ----------------------
            batch_vars = varnames[b * batch_size : (b + 1) * batch_size]

            # ---- 3-2) Figure & axes --------------------------------
            width, height = self.cfg.figsize  # e.g., (8, 2)
            batch_figsize = (width, height * len(batch_vars))  # scale height by n rows
            # fig_height    = height * len(batch_vars)  # scale by rows
            fig, axes = plt.subplots(
                len(batch_vars),
                1,
                figsize=batch_figsize,
                sharex=sharex,
                dpi=self.cfg.dpi,  # Figure dpi follows config
            )
            if len(batch_vars) == 1:
                axes = [axes]

            # ---- 3-3) Plot each variable ---------------------------
            for var, ax in zip(batch_vars, axes):
                self.ts_plot(varname=var, index=index, k=k, label=var, ax=ax, **kwargs)
                # ax.set_title(var, fontsize=self.cfg.fontsize_title)
                if legend:
                    ax.legend(fontsize=self.cfg.fontsize_legend)
            # ---- 3-4) Layout & save -------------------------------
            if title and suptitle:
                fig.suptitle(
                    f"Time-Series Batch {b + 1}/{num_batches}",
                    fontsize=self.cfg.fontsize_suptitle,
                )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            save_path = f"{png_prefix}_batch_{b + 1}.png"
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        logger.info("Saved %d figure(s) as '%s_batch_#.png'", num_batches, png_prefix)

    def ts_river_in_batches(
        self,
        varname: str,
        batch_size: int = 4,
        png_prefix: str = "river_plot",
        **kwargs,
    ):
        """
        Plot one river variable for *all* rivers in batches, delegating to self.ts_river().

        Parameters
        ----------
        varname : str
            Variable name to plot (must have 'rivers' and 'time' dims).
        batch_size : int, default 4
            Number of rivers per figure.
        xlim : tuple | None, optional
            (start, end) time limits forwarded to ts_river / ts_plot.
        png_prefix : str, default "river_plot"
            Prefix for saved PNG files (e.g. "river_plot_batch_1.png").
        **kwargs
            Additional keyword arguments passed straight to self.ts_river()
            (e.g. rolling_window=24, color="k", linestyle="--").
        """

        # ---- 1) Determine river count and batches------------------------------
        if "river_names" not in self.ds:
            raise ValueError("Dataset lacks 'river_names'; cannot infer river count.")
        num_rivers = int(self.ds["river_names"].sizes["rivers"])
        num_batches = ceil(num_rivers / batch_size)

        # ---- 2) Clean up kwargs for plotting -------------------------------
        dpi = kwargs.pop("dpi", self.cfg.dpi)
        sharex = kwargs.pop("sharex", True)
        legend = kwargs.pop("legend", True)
        title = kwargs.pop("title", True)
        suptitle = kwargs.pop("suptitle", True)

        for b in range(num_batches):
            start_i = b * batch_size
            end_i = min((b + 1) * batch_size, num_rivers)
            river_idxs = range(start_i, end_i)

            # ---- 3) Make one figure per batch --------------------------------
            # --- determine base figsize -----------------------------------
            width, height = self.cfg.figsize  # e.g., (8, 2)
            batch_figsize = (width, height * len(river_idxs))  # scale height by n rows

            fig, axes = plt.subplots(
                len(river_idxs),
                1,
                figsize=batch_figsize,  # (10, 3 * len(river_idxs)),
                sharex=sharex,
            )
            if len(river_idxs) == 1:  # axes is Axes if n==1 → listify
                axes = [axes]

            # ---- 4) Plot each river -----------------------------------------
            for idx, ax in zip(river_idxs, axes):
                _, ax = self.ts_river(varname=varname, river_index=idx, ax=ax, **kwargs)  # type: ignore[attr-defined]
                # Plot legend
                if legend:
                    ax.legend(fontsize=self.cfg.fontsize_legend)
            # ---- 5) Layout & save -------------------------------------------
            if title and suptitle:
                fig.suptitle(
                    f"{varname} - Rivers {start_i}-{end_i-1} "
                    f"(batch {b+1}/{num_batches})",
                    fontsize=self.cfg.fontsize_suptitle,
                )
            fig.tight_layout(rect=(0, 0, 1, 0.95))

            save_path = f"{png_prefix}_batch_{b+1}.png"
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

        logger.info("Saved %d figure(s) as '%s_batch_#.png'", num_batches, png_prefix)

    def plot_timeseries_for_river_in_batches(
        self,
        plotter,
        var_name,
        batch_size=4,
        start=None,
        end=None,
        save_prefix="river_plot",
        **kwargs,
    ):
        """
        Plot a single variable for all rivers in batches.

        Parameters:
        - plotter: FvcomPlotter object
        - var_name: Variable name to plot.
        - batch_size: Number of rivers per figure.
        - start, end: Start and end times for the time series.
        - png_prefix: Prefix for saved file names (e.g., "river_plot").
        - **kwargs: Additional arguments for customization.
        """

        # Rivers の次元サイズを取得
        if "river_names" in self.ds:
            num_rivers = self.ds["river_names"].sizes["rivers"]
        else:
            logger.error("No 'river_names' variable found.")
            return None

        # バッチの数を計算
        num_batches = ceil(num_rivers / batch_size)

        for batch_num in range(num_batches):
            # バッチごとのインデックス範囲を計算
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, num_rivers)
            river_indices = range(start_idx, end_idx)

            # 図の作成
            fig, axes = plt.subplots(
                len(river_indices), 1, figsize=(10, 3 * len(river_indices)), sharex=True
            )
            if len(river_indices) == 1:
                axes = [axes]  # rivers が1つの場合でもリスト化

            # 各 river のプロット
            for river_index, ax in zip(river_indices, axes):
                plotter.plot_timeseries_for_river(
                    var_name=var_name,
                    river_index=river_index,
                    start=start,
                    end=end,
                    ax=ax,
                    **kwargs,
                )
                # ax.set_title(f"{var_name} for river {river_index}", fontsize=14)

            # 図全体の調整
            fig.suptitle(
                f"Time Series of {var_name} (Batch {batch_num + 1})", fontsize=16
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))  # タイトルとプロット間のスペース調整

            # 保存または表示
            save_path = f"{save_prefix}_batch_{batch_num + 1}.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        logger.info("Saved %d figures as '%s_batch_#.png'", num_batches, save_prefix)


# Other helper functions can be added here as needed.


def get_index_by_value(array, value):
    """
    Get the index of a value in a list or numpy array.
    This function works for both lists and numpy arrays.

    Parameters:
        array (list or numpy.ndarray): The list or numpy array to search.
        value (int or float): The value to find in the array.

    Returns:
        int: The index of the value in the array.

    Raises:
        ValueError: If the value is not found in the array.
        TypeError: If the provided array type is not supported.

    Example:
        >>> lst = [10, 20, 30, 40]
        >>> get_index_by_value(lst, 30)
        2

        >>> arr = np.array([10, 20, 30, 40])
        >>> get_index_by_value(arr, 30)
        2
        >>> get_index_by_value(lst, 50)
        ValueError: 50 does not exist in the list.
        >>> get_index_by_value(arr, 50)
        ValueError: 50 does not exist in the numpy array.
    """
    # If the array is a list, attempt to find the index using the list.index() method.
    if isinstance(array, list):
        try:
            return array.index(value)
        except ValueError:
            raise ValueError(f"{value} does not exist in the list.")

    # If the array is a numpy array, use np.where to find the index.
    elif isinstance(array, np.ndarray):
        index_array = np.where(array == value)[0]
        if index_array.size > 0:
            return int(index_array[0])
        else:
            raise ValueError(f"{value} does not exist in the numpy array.")

    # If the array is neither a list nor a numpy array, raise a TypeError.
    else:
        raise TypeError(
            f"Unsupported type: {type(array)}. Expected list or numpy array."
        )


def pick_first(*values, default=None):
    """Return first non-None value; otherwise *default*."""
    for v in values:
        if v is not None:
            return v
    return default

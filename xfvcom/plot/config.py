from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class FvcomPlotConfig:
    """
    Stores plot configuration settings.
    """

    # ----------------------------------------------------------------
    # Default figure settings
    # ----------------------------------------------------------------
    DEFAULT_FIGURE = {"figsize": (8, 2), "dpi": 100, "facecolor": "white"}

    # ----------------------------------------------------------------
    # Default color cycle & colormap
    # ----------------------------------------------------------------
    DEFAULT_COLOR = {
        # Matplotlib “tab10” colors
        "color_cycle": [
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
            "#e377c2",  # pink
            "#7f7f7f",  # gray
            "#bcbd22",  # olive
            "#17becf",  # cyan
        ],
        "cmap": "jet",
        "norm": None,
    }

    # ----------------------------------------------------------------
    # Default colorbar settings
    # ----------------------------------------------------------------
    DEFAULT_COLORBAR = {
        "size": "2.0%",  # width of cbar relative to main axes
        "pad": 0.10,  # gap between axes and cbar (axes fraction)
        "kwargs": {},  # → forwarded verbatim to `figure.colorbar`
    }

    # ----------------------------------------------------------------
    # Default grid settings
    # ----------------------------------------------------------------
    DEFAULT_GRID = {"linestyle": "--", "linewidth": 0.5, "color": "gray", "alpha": 0.7}

    # ----------------------------------------------------------------
    # Class‐level defaults for font sizes (can be overridden)
    # ----------------------------------------------------------------
    DEFAULT_FONT_SIZES = {
        "xticks": 11,  # size of x-axis tick labels
        "yticks": 11,  # size of y-axis tick labels
        "xlabel": 12,  # size of x-axis label
        "ylabel": 12,  # size of y-axis label
        "title": 12,  # size of axes title
        "suptitle": 12,  # size of figure suptitle
        "legend": 10,  # size of legend text
        "legend_title": 11,  # size of legend title
        "annotation": 12,  # size of annotation text
        "text": 12,  # size of generic text
        "colorbar": 11,  # size of colorbar tick labels
        "cbar_label": 12,  # size of colorbar axis label
        "cbar_title": 12,  # size of colorbar title
        "tick_params": 11,  # size applied when using ax.tick_params
    }
    # ----------------------------------------------------------------
    # Class‐level defaults for line widths (can be overridden)
    # ----------------------------------------------------------------
    DEFAULT_LINE_WIDTHS = {
        "plot": 1.5,  # linewidth for ax.plot
        "contour": 1.0,  # linewidths for ax.contour
        "grid": 0.8,  # linewidth for ax.grid
        "axes": 1.2,  # overall axes line width (spines & ticks)
        "spines": 1.2,  # line width for each spine
        "tick_params": 0.8,  # width parameter in ax.tick_params
        "legend": 1.0,  # frame line width for legend
        "colorbar": 1.0,  # outline line width for colorbar
        "errorbar": 1.0,  # elinewidth for ax.errorbar
        "hist": 1.0,  # edge line width for ax.hist
        "bar": 1.0,  # edge line width for ax.bar
        "boxplot": 1.0,  # line width for ax.boxplot
        "scatter": 0.5,  # edge width for ax.scatter markers
        "annotation": 0.8,  # bbox line width for ax.annotate
    }

    # Obsolete for the past code. After updating, delete this.
    """
    DEFAULT_ARROW_OPTIONS = {
        "arrow_scale": 1.0,
        "arrow_scale": 1.0,
        "arrow_width": 0.002,
        "arrow_color": "blue",
        "arrow_alpha": 0.7,
        "arrow_angles": "uv",
        "arrow_headlength": 5,
        "arrow_headwidth": 3,
        "arrow_headaxislength": 4.5
    }
    """
    # ----------------------------------------------------------------
    # Default arrow properties (can be overridden via arrow_options)
    # ----------------------------------------------------------------
    DEFAULT_ARROW_OPTIONS = {
        "scale": 1.0,
        "width": 0.002,
        "color": "blue",
        "alpha": 0.7,
        "angles": "uv",
        "headlength": 5,
        "headwidth": 3,
        "headaxislength": 4.5,
    }

    # ----- internal cache (not in __init__, not in repr) -------------
    _private_cache: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __init__(
        self,
        figsize=None,
        dpi=None,
        facecolor=None,
        width=800,
        height=200,
        cmap="jet",
        levels=21,
        title_fontsize=14,
        label_fontsize=12,
        tick_fontsize=11,
        figure=None,
        color=None,
        grid=None,
        fontsize=None,
        linewidth=None,
        arrow_options=None,
        colorbar: dict | None = None,
        cbar_size: str | None = None,
        cbar_pad: float | None = None,
        cbar_kwargs: dict | None = None,
        **kwargs,
    ):
        """
        Initialize the FvcomPlotConfig instance.
        Parameters:
        - figsize: Figure size in inches (width, height).
        - width: Width of the plot in pixels.
        - height: Height of the plot in pixels.
        - dpi: Dots per inch for the plot.
        - cbar_size: Size of the colorbar.
        - cbar_pad: Padding between the colorbar and the plot.
        - cmap: Colormap to use for the plot.
        - levels: Number of levels for the colormap.
        - title_fontsize: Font size for the plot title.
        - label_fontsize: Font size for the x and y labels.
        - tick_fontsize: Font size for the tick labels.
        - fontsize: Font size settings for various plot elements.
        - linewidth: Line width for the plot lines.
        - **kwargs: Additional keyword arguments for customization.
        """
        # Merge figure settings
        fig_opts = {**self.DEFAULT_FIGURE, **(figure or {})}
        # Override by individual args if provided
        if figsize is not None:
            fig_opts["figsize"] = figsize
        if dpi is not None:
            fig_opts["dpi"] = dpi
        if facecolor is not None:
            fig_opts["facecolor"] = facecolor
        for name, val in fig_opts.items():
            setattr(self, name, val)

        # Merge user-provided color settings
        col_opts = {**self.DEFAULT_COLOR, **(color or {})}
        self.color_cycle = col_opts["color_cycle"]
        self.cmap = col_opts["cmap"]
        self.norm = col_opts["norm"]

        # ------------------------------------------------------------
        # Color-bar defaults
        # ------------------------------------------------------------
        cb_base = {**self.DEFAULT_COLORBAR, **(colorbar or {})}
        if cbar_size is not None:
            cb_base["size"] = cbar_size
        if cbar_pad is not None:
            cb_base["pad"] = cbar_pad
        if cbar_kwargs is not None:
            cb_base["kwargs"] = {**cb_base["kwargs"], **cbar_kwargs}
        self.cbar_size = cb_base["size"]
        self.cbar_pad = cb_base["pad"]
        self.cbar_kwargs = cb_base["kwargs"]

        # Merge grid settings
        grid_opts = {**self.DEFAULT_GRID, **(grid or {})}
        self.grid_linestyle = grid_opts["linestyle"]
        self.grid_linewidth = grid_opts["linewidth"]
        self.grid_color = grid_opts["color"]
        self.grid_alpha = grid_opts["alpha"]

        # ------------------------------------------------------------
        # Merge and assign font sizes
        # ------------------------------------------------------------
        merged_fs = {**self.DEFAULT_FONT_SIZES, **(fontsize or {})}
        self.fontsize = merged_fs
        # Also expose as individual attributes: fontsize_xticks, fontsize_xlabel, …
        for name, size in merged_fs.items():
            setattr(self, f"fontsize_{name}", size)
        # ------------------------------------------------------------
        # Merge and assign line widths
        # ------------------------------------------------------------
        merged_lw = {**self.DEFAULT_LINE_WIDTHS, **(linewidth or {})}
        self.linewidth = merged_lw
        # Also expose as individual attributes: linewidth_plot, linewidth_grid, …
        for name, lw in merged_lw.items():
            setattr(self, f"linewidth_{name}", lw)

        merged_ao = {**self.DEFAULT_ARROW_OPTIONS, **(arrow_options or {})}
        self.arrow_options = merged_ao
        # Assign each arrow option as an attribute with 'arrow_' prefix
        for name, val in merged_ao.items():
            setattr(self, f"arrow_{name}", val)

        # Obsolete. After updating, delete this part.
        # self.figsize = figsize
        self.width = width
        self.height = height
        # self.dpi = dpi
        # self.cbar_size = cbar_size
        # self.cbar_pad = cbar_pad
        # self.cmap = cmap
        self.levels = levels
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.date_format = kwargs.get("date_format", "%Y-%m-%d")
        self.plot_color = kwargs.get("plot_color", "red")

        # ------------------------------------------------------------
        # Internal cache (dataclass default is skipped because we define __init__)
        # ------------------------------------------------------------
        self._private_cache: Dict[str, Any] = {}

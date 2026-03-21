"""
Improved visualization for seasonal extension that shows the extension clearly
even when data is constant.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from xfvcom.utils.timeseries_utils import extend_timeseries_seasonal


def plot_seasonal_extension_improved(
    orig_data, extended_data, title="Seasonal Extension"
):
    """
    Create an improved plot that clearly shows the extension even for constant data.

    Parameters
    ----------
    orig_data : pd.DataFrame
        Original data with DatetimeIndex
    extended_data : pd.DataFrame
        Extended data with DatetimeIndex
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    # Get column name
    col = orig_data.columns[0]

    # Plot original data
    ax.plot(
        orig_data.index,
        orig_data[col],
        "b-",
        label="Original Data",
        linewidth=2,
        zorder=3,
    )

    # Get the extended portion only
    orig_end = orig_data.index[-1]
    extended_portion = extended_data.loc[extended_data.index > orig_end]

    if len(extended_portion) > 0:
        # Plot ONLY the extended portion to make it visible
        ax.plot(
            extended_portion.index,
            extended_portion[col],
            "m--",
            label=f"Extension ({len(extended_portion)} points)",
            linewidth=2,
            alpha=0.8,
            zorder=2,
        )

        # Add a connector line between original and extension
        connector_x = [orig_data.index[-1], extended_portion.index[0]]
        connector_y = [orig_data[col].iloc[-1], extended_portion[col].iloc[0]]
        ax.plot(connector_x, connector_y, "g-", linewidth=1, alpha=0.5, zorder=1)

    # Mark the extension point clearly
    ax.axvline(
        x=orig_end,
        color="red",
        linestyle=":",
        alpha=0.7,
        label="Extension Start",
        linewidth=2,
    )

    # Add shaded region for extended period
    if len(extended_portion) > 0:
        ax.axvspan(
            orig_end,
            extended_data.index[-1],
            alpha=0.1,
            color="magenta",
            label="Extended Period",
        )

    # Add text annotation
    mid_point = orig_end + (extended_data.index[-1] - orig_end) / 2
    y_pos = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95
    ax.text(
        mid_point,
        y_pos,
        f"Extended: {len(extended_portion)} points",
        ha="center",
        va="top",
        fontsize=10,
        color="magenta",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    return fig, ax


# Test with constant data
if __name__ == "__main__":
    # Create constant data (like in the notebook)
    dates = pd.date_range("2020-01-01", periods=365, freq="D")
    const_data = pd.DataFrame(
        {"discharge": np.full(365, 6.60580921, dtype=np.float32)}, index=dates
    )

    # Extend the data
    const_extended = extend_timeseries_seasonal(const_data, "2021-12-31", period="1Y")

    # Create improved plot
    fig, ax = plot_seasonal_extension_improved(
        const_data,
        const_extended,
        title="Constant Discharge - Seasonal Extension (Improved Visualization)",
    )

    plt.savefig("/tmp/seasonal_improved.png", dpi=150)
    print("Improved plot saved to /tmp/seasonal_improved.png")

    # Also test with varying data
    t = np.arange(365)
    varying_values = 10 + 5 * np.sin(2 * np.pi * t / 365)
    vary_data = pd.DataFrame(
        {"discharge": varying_values.astype(np.float32)}, index=dates
    )
    vary_extended = extend_timeseries_seasonal(vary_data, "2021-12-31", period="1Y")

    fig2, ax2 = plot_seasonal_extension_improved(
        vary_data,
        vary_extended,
        title="Varying Discharge - Seasonal Extension (Improved Visualization)",
    )

    plt.savefig("/tmp/seasonal_varying_improved.png", dpi=150)
    print("Varying data plot saved to /tmp/seasonal_varying_improved.png")

    plt.show()

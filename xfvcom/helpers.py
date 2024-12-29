import matplotlib.pyplot as plt
from math import ceil
class PlotHelperMixin:
    """
    A mixin class to provide helper methods for batch plotting and other common operations.
    """

    def plot_variables_in_batches(self, plotter, vars, index, batch_size=4, start=None, end=None, save_prefix="plot", **kwargs):
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
                plotter.plot_time_series(var_name=var, index=index, start=start, end=end, ax=ax, **kwargs)
                ax.set_title(var, fontsize=14)

            # 図全体の調整
            fig.suptitle(f"Time Series Batch {batch_num + 1}", fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # タイトルとプロット間のスペース調整

            # 保存または表示
            save_path = f"{save_prefix}_batch_{batch_num + 1}.png"
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        print(f"Saved {num_batches} figures as '{save_prefix}_batch_#.png'.")

    def plot_rivers_in_batches(self, plotter, var_name, batch_size=4, start=None, end=None, save_prefix="river_plot", **kwargs):
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
                plotter.plot_time_series_for_river(
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

# Plotting with xfvcom

[Back to README](../README.md)

## Setup

```python
from xfvcom import FvcomDataLoader, FvcomPlotter, FvcomPlotConfig, FvcomPlotOptions

loader = FvcomDataLoader(base_path="/path/to/data", ncfile="output.nc")
ds = loader.ds

cfg = FvcomPlotConfig(figsize=(12, 8))
plotter = FvcomPlotter(ds, cfg)
```

---

## 2D Horizontal Plots

### Basic Plot

```python
fig = plotter.plot_2d("temp", time="2020-07-01", siglay=0)
```

### With Options

```python
opts = FvcomPlotOptions(
    add_tiles=True,
    tile_provider="satellite",
    with_mesh=True,
    mesh_color="white",
    cmap="RdYlBu_r",
    vmin=15, vmax=30,
)
fig = plotter.plot_2d("temp", time="2020-07-01", siglay=0, opts=opts)
```

### Vector Overlay

```python
opts = FvcomPlotOptions(
    plot_vec2d=True,
    vec_siglay=0,
    arrow_color="black",
    with_mesh=False,
)
fig = plotter.plot_2d("temp", time="2020-07-01", siglay=0, opts=opts)
```

### FvcomPlotOptions Reference

| Option | Type | Description |
|--------|------|-------------|
| `figsize` | `tuple` | Figure size |
| `cmap` | `str` | Colormap name |
| `vmin`, `vmax` | `float` | Color range |
| `levels` | `int` | Number of contour levels |
| `with_mesh` | `bool` | Draw mesh edges |
| `mesh_color` | `str` | Mesh edge color |
| `add_tiles` | `bool` | Add map tiles |
| `tile_provider` | `str` | `"osm"`, `"satellite"`, or Cartopy tile object |
| `plot_vec2d` | `bool` | Overlay velocity vectors |
| `vec_siglay` | `int` | Sigma layer for vectors |
| `arrow_color` | `str` | Vector arrow color |
| `xlim`, `ylim` | `tuple` | Map extent |
| `coastlines` | `bool` | Draw coastlines |

---

## Time Series

### Single Node

```python
fig = plotter.plot_timeseries("temp", index=100)
```

### Ensemble Plots

```python
from xfvcom.plot import plot_ensemble_timeseries, plot_dye_timeseries_stacked

# Line plot (auto colormap: tab20 for <=20 members, hsv otherwise)
fig, ax = plot_ensemble_timeseries(ds, var_name="dye", cfg=cfg)

# Stacked area plot
result = plot_dye_timeseries_stacked(ds, cfg=cfg, output="stacked.png")
```

---

## Animations

```python
from xfvcom.plot.utils import create_anim_2d_plot

create_anim_2d_plot(
    plotter=plotter,
    var_name="temp",
    siglay=0,
    fps=10,
    output_format="gif",  # or "mp4"
    plot_kwargs={"vmin": 15, "vmax": 30, "cmap": "RdYlBu_r"}
)
```

---

## Post-Processing Hooks

```python
def add_markers(ax, da, time):
    ax.plot([139.8], [35.4], 'ro', markersize=10)
    ax.text(139.8, 35.41, 'Station A', color='red')

fig = plotter.plot_2d(
    "temp", time="2020-07-01", siglay=0,
    post_process_func=add_markers
)
```

### Node Markers

```python
from xfvcom import make_node_marker_post

pp = make_node_marker_post(
    nodes=[100, 200, 300],
    plotter=plotter,
    marker_kwargs={"color": "red", "markersize": 8},
    index_base=1,
)
fig = plotter.plot_2d("temp", post_process_func=pp)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Map flipped/shifted | Set `projection=ccrs.PlateCarree()` in options |
| Vector scale wrong | Use `vec_legend_speed` to fix reference speed |
| Tiles not loading | Check internet connection; try `tile_provider="osm"` |

[Back to README](../README.md)

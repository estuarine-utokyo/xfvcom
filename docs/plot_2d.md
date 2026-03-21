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
    tile_provider=GoogleTiles(style="satellite"),
    with_mesh=True,
    mesh_color="white",
    cmap="RdYlBu_r",
    vmin=15, vmax=30,
)
fig = plotter.plot_2d("temp", time="2020-07-01", siglay=0, opts=opts)
```

### With Coast Masking (OSM-derived)

```python
from xfvcom.coastmask import load

mask = load("tokyo_bay")
opts = FvcomPlotOptions(coastmask=mask)
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

#### Color & Scaling

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `cmap` | `str` | `"viridis"` | Colormap name |
| `vmin`, `vmax` | `float \| None` | `None` | Color range limits |
| `levels` | `int \| list[float]` | `20` | Number of contour levels |
| `extend` | `str` | `"both"` | Colorbar extension (`"both"`, `"neither"`, `"min"`, `"max"`) |
| `norm` | `Normalize \| None` | `None` | Custom matplotlib normalization |
| `log_scale` | `bool` | `False` | Use logarithmic color scale |

#### Figure & Axes

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `figsize` | `tuple \| None` | `None` | Figure size (width, height) |
| `dpi` | `int \| None` | `None` | Figure resolution |
| `title` | `str \| None` | `None` | Plot title |
| `xlabel`, `ylabel` | `str \| None` | `None` | Axis labels |
| `xlim`, `ylim` | `tuple \| None` | `None` | Map extent |
| `date_fmt` | `str` | `"%Y-%m-%d"` | Date format for time axes |
| `projection` | `ccrs.Projection` | `ccrs.Mercator()` | Cartopy projection |

#### Mesh & Map

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `with_mesh` | `bool` | `False` | Draw mesh edges |
| `mesh_color` | `str` | `"#36454F"` | Mesh edge color |
| `mesh_linewidth` | `float` | `0.5` | Mesh edge width |
| `coastlines` | `bool` | `False` | Draw Cartopy coastlines |
| `coastline_color` | `str` | `"gray"` | Coastline color |
| `add_tiles` | `bool` | `False` | Add map tiles |
| `tile_provider` | `GoogleTiles` | `GoogleTiles(style="satellite")` | Map tile source |
| `tile_zoom` | `int` | `12` | Tile zoom level |
| `plot_grid` | `bool` | `False` | Draw lat/lon grid lines |

#### Coast Masking

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `coastmask` | `CoastMask \| None` | `None` | OSM-derived land mask object |
| `coastmask_facecolor` | `str` | `"lightgray"` | Land fill color |
| `coastmask_edgecolor` | `str` | `"black"` | Land boundary color |
| `coastmask_linewidth` | `float` | `0.5` | Land boundary width |
| `coastmask_zorder` | `int` | `5` | Drawing order |

#### Colorbar

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `colorbar` | `bool` | `True` | Show colorbar |
| `cbar_label` | `str \| None` | `None` | Colorbar label |
| `cbar_size` | `str \| None` | `None` | Colorbar size (e.g. `"3%"`) |
| `cbar_pad` | `float \| None` | `None` | Colorbar padding |

#### Vector Overlay

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `plot_vec2d` | `bool` | `False` | Overlay velocity vectors |
| `vec_siglay` | `int \| None` | `None` | Sigma layer for vectors |
| `arrow_color` | `str` | `"k"` | Vector arrow color |
| `arrow_scale` | `float \| None \| "auto"` | `"auto"` | Arrow scaling factor |
| `arrow_width` | `float` | `0.002` | Arrow width |
| `arrow_alpha` | `float` | `0.7` | Arrow transparency |
| `skip` | `int \| None` | `None` | Arrow sampling interval |
| `show_vec_legend` | `bool` | `True` | Show vector legend |
| `vec_legend_speed` | `float \| None` | `None` | Reference speed (None → 0.3×max) |
| `with_magnitude` | `bool` | `True` | Color arrows by magnitude |

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
    processes=4,              # Max parallel processes for frame rendering
    var_name="temp",
    siglay=0,
    fps=10,
    generate_gif=True,        # Generate GIF (default: True)
    generate_mp4=False,       # Generate MP4 (default: False)
    cleanup=False,            # Delete frame PNGs after animation
    plot_kwargs={"vmin": 15, "vmax": 30, "cmap": "RdYlBu_r"},
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
    text_kwargs={"fontsize": 8},  # Optional: label styling
    index_base=1,                 # 1-based (FVCOM convention)
)
fig = plotter.plot_2d("temp", post_process_func=pp)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Map flipped/shifted | Set `projection=ccrs.PlateCarree()` in options |
| Vector scale wrong | Use `vec_legend_speed` to fix reference speed |
| Tiles not loading | Check internet connection; try a different `tile_provider` |
| Text clipped by mask | Use `make_node_marker_post(..., text_clip_buffer=-0.001)` |

[Back to README](../README.md)

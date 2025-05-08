# Plotting 2-D Horizontal (sigma layer) Fields with **xfvcom**

[← Back to README](../README.md)

This document is the **definitive guide** to the `plot_2d` method in the `xfvcom.plot` sub‑package.
Starting from a data preparation, it walks through mesh overlays, map tiles,
scalar fields, vector fields, and advanced hooks.

---

## 1. Prepare data and plotter instance

```python
import cartopy.crs as ccrs
from xfvcom import FvcomDataLoader
from xfvcom import FvcomPlotter, FvcomPlotConfig, FvcomPlotOptions

fvcom = FvcomDataLoader(
    base_path="/path/to/data",
    ncfile="sample.nc",
)
da_s = fvcom.ds["temp"]   # scalar  (time, siglay, node)
da_u = fvcom.ds["u"]      # vector-x(time, siglay, nele)
da_v = fvcom.ds["v"]      # vector-y(time, siglay, nele)

plotter = FvcomPlotter(fvcom.ds, FvcomPlotConfig())

```

---

## 2. Option cheat‑sheet

| Category       | Key fields (type)                                       | Example                         |
| -------------- | ------------------------------------------------------- | ------------------------------- |
| Colour & range | `cmap`, `vmin/vmax`, `levels`                           | `levels=30`, `cmap="jet"`       |
| Mesh           | `with_mesh`, `mesh_color`, `mesh_lw` 　　　　            | `with_mesh=True`                |
| Coast / OBC    | `coastlines`, `obclines`, `coastline_color`             | `obclines=True`                 |
| Projection     | `use_latlon`, `projection`, `add_tiles`, `tile_*`       | `projection=ccrs.PlateCarree()` |
| Vectors        | `plot_vec2d`, `vec_siglay`, `arrow_color`, `vec_reduce` | `vec_reduce={"time": "mean"}`   |

| Option | Type / default | Effect |
| ------ | -------------- | ------ |
| `figsize`    | `tuple[float, float] \| None = None` | Matplotlib Figure size| 
| `use_latlon` | `bool, auto‑detected` | Treat coords as lon/lat |
| `with_mesh` | `bool = True` | Draw element edges |
| `plot_vec2d` | `bool = False` | Overlay 2‑D velocity vectors |
| `vec_siglay` | `int \| slice = "thickness"` | Which σ‑layer for vectors |
| `arrow_color` | CSS color str | Quiver arrow colour |
| `coastlines` | `bool = False` | Draw land–sea boundary |
| `obclines` | `bool = False` | Draw open‑boundary segments |
| `add_tiles` | `"terrain" \| "toner" \| None` | Fetch Stamen background tiles |
| `levels` | `int \| list` | Contour levels |
| `cmap` | Matplotlib colormap name or object | Colour map for scalar field |
| `scalar_reduce` | `dict[str, str] \| None` | `None` | Reduction to apply to the scalar - field DataArray before plotting.<br/>Keys = dimension names (`"time"`, `"siglay"`…), values = NumPy reduction functions (`"mean"`, `"max"`, `"min"`, `"sum"`…). |
| `vec_reduce`    | `dict[str, str] \| None` | `None` | Same as `scalar_reduce`, but applied to the vector components (`u`, `v`) **and** the derived magnitude \|U\|. |
| `xlim`| `tuple[float \| str, float \| str] \| None = None` | x coordinate extent |
| `ylim`| `tuple[float \| str, float \| str] \| None = None` | y coordinate extent |
| `lon_tick_skip` | `int \| None = None` | Skip longitute axis label  (e.g., =2 → 1/2 ) |
| `lat_tick_skip` | `int \| None = None` | Skip latitude axis label (e.g., =2 → 1/2 ) |


*All* fields are documented in
[`xfvcom.plot_options.FvcomPlotOptions`](../../xfvcom/plot_options.py).

---

## 3. Scalar + vector overlay (quick style with u and v retrieved in plotter instance)

```python
opts = FvcomPlotOptions(
    plot_vec2d=True,
    vec_siglay=0,
    arrow_color="k",
    with_mesh=True,
    coastlines=True,
)

da = da_s.isel(time=2, siglay=0)
ax = plotter.plot_2d(da=da, opts=opts)
ax.figure.savefig("scalar_vec.png")
```

*To average scalar and vector time‑series independently, use
`scalar_reduce` and `vec_reduce` separately.*

---

## 4. Mesh and map tiles only

```python
from cartopy.io.img_tiles import GoogleTiles

opts = FvcomPlotOptions(
    add_tiles=True,
    tile_provider=GoogleTiles(style="satellite"),
    mesh_color="#ffffff",
    mesh_linewidth=0.3,
)

plotter.plot_2d(da=None, opts=opts)
```

---

## 5. Custom post‑processing hook

`plot_2d()` accepts a callable that is executed **after** the figure is
completed:

```python
def add_timestamp(ax, da, time):
    import pandas as pd
    txt = pd.to_datetime(time).strftime("%Y-%m-%d %H:%M")
    ax.text(
        0.02, 0.95, txt,
        transform=ax.transAxes,
        ha="left", va="top",
        color="white",
        fontsize=10,
        bbox=dict(fc="0.2", ec="none", alpha=0.7, pad=2),
    )

plotter.plot_2d(
    da=da_s.isel(time=5, siglay=0),
    opts=FvcomPlotOptions(),
    post_process_func=add_timestamp,
)
```

---

## 6. Scalar + vector overlay with time/vertical averaging
Specify DataArrays explicitly.

```python
opts = FvcomPlotOptions(
    # scalar: 2 day-mean of surface layer
    scalar_time   = slice("2020-01-03", "2020-01-04"),
    scalar_siglay = slice(None),                 # all layers
    scalar_reduce = {"time": "mean", "siglay": "mean"},

    # vector overlay: same range, all layers mean
    plot_vec2d    = True,
    vec_time      = slice("2020-01-03", "2020-01-04"),
    vec_siglay    = slice(None),
    vec_reduce    = {"time": "mean", "siglay": "mean"},
    arrow_color   = "k",

    # decorations
    with_mesh     = True,
    coastlines    = True,
)

# ------------------------------------------------------------
# Draw & save
# ------------------------------------------------------------
ax = plotter.plot_2d(
    da     = da_s,   # scalar DataArray
    da_u  = da_u,   # vector-U DataArray
    da_v  = da_v,   # vector-V DataArray
    opts   = opts,
)

ax.figure.savefig("scalar_vec.png", dpi=150)
```

## 7. Batch frames for animation

```python
from pathlib import Path
outdir = Path("frames"); outdir.mkdir(exist_ok=True)

for ti in range(len(ds.time)):
    ax = plotter.plot_2d(
        da   = da_s.isel(time=ti, siglay=0),
        opts = FvcomPlotOptions(),
        post_process_func = add_timestamp,
    )
    ax.figure.savefig(outdir / f"frame_{ti:04d}.png", dpi=120, bbox_inches="tight")
    ax.figure.clf()
```

---

## 8. Troubleshooting and tips

| Symptom                                       | Cause & fix                                                                                        |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `ValueError: z array must have same length …` | Ensure the DataArray is nodal (`node` dimension) and the triangulation uses nodal coords.          |
| Map is flipped or shifted                     | `use_latlon=True` but `projection` is `None`. Pass `ccrs.PlateCarree()` or set `use_latlon=False`. |
| Vector legend size changes too much           | Fix the reference speed with `vec_legend_speed=<float in m/s>`.                                    |

---

## 9. API reference

* **`FvcomPlotter.plot_2d`** — combined scalar/vector/mesh routine.
* **`FvcomPlotter.plot_vector2d`** — depth‑averaged (or layer‑specific) vector field.
* **`FvcomPlotOptions`** — dataclass holding all plotting options.

---

[← Back to README](../README.md)

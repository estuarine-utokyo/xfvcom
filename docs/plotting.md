# Plotting with **xfvcom**

[← Back to README](../README.md)

This document is the **definitive guide** to the `xfvcom.plot` sub‑package.
Starting from a 3‑line quick start, it walks through mesh overlays, map tiles,
vector fields, and advanced hooks.

---

## 1. Quick start (contour map in three lines)

```python
from xfvcom import FvcomPlotConfig, FvcomPlotOptions
from xfvcom.plot import FvcomPlotter
import xarray as xr

ds = xr.open_dataset("sample_fvcom.nc")
plotter = FvcomPlotter(ds, FvcomPlotConfig())

da = ds["temp"].isel(time=0, siglay=0)
ax = plotter.plot_2d(da=da, opts=FvcomPlotOptions())
ax.figure.savefig("temp_t000.png", dpi=150)
```

---

## 2. Option cheat‑sheet

| Category       | Key fields (type)                                       | Example                         |
| -------------- | ------------------------------------------------------- | ------------------------------- |
| Colour & range | `cmap`, `vmin/vmax`, `levels`                           | `levels=30`, `cmap="jet"`       |
| Mesh           | `with_mesh`, `mesh_color`, `mesh_linewidth`             | `with_mesh=True`                |
| Coast / OBC    | `coastlines`, `obclines`, `coastline_color`             | `obclines=True`                 |
| Projection     | `use_latlon`, `projection`, `add_tiles`, `tile_*`       | `projection=ccrs.PlateCarree()` |
| Vectors        | `plot_vec2d`, `vec_siglay`, `arrow_color`, `vec_reduce` | `vec_reduce={"time": "mean"}`   |

*All* fields are documented in
[`xfvcom.plot_options.FvcomPlotOptions`](../../xfvcom/plot_options.py).

---

## 3. Scalar + vector overlay

```python
opts = FvcomPlotOptions(
    plot_vec2d=True,
    vec_siglay=0,
    arrow_color="k",
    with_mesh=True,
    coastlines=True,
)

da = ds["temp"].isel(time=2, siglay=0)
ax = plotter.plot_2d(da=da, opts=opts)
ax.figure.savefig("scalar_vec.png")
```

*To average scalar and vector time‑series independently, use
`scalar_reduce` and `vec_reduce` separately.*

---

## 4. Adding map tiles

```python
from cartopy.io.img_tiles import GoogleTiles

opts = FvcomPlotOptions(
    add_tiles=True,
    tile_provider=GoogleTiles(style="satellite"),
    tile_zoom=11,
    mesh_color="#ffffff",
    mesh_linewidth=0.3,
)

plotter.plot_2d(da=None, opts=opts)  # mesh + tiles only
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
        bbox=dict(fc="0.2", ec="none", alpha=0.7),
    )

plotter.plot_2d(
    da=ds["salinity"].isel(time=5, siglay=0),
    opts=FvcomPlotOptions(),
    post_process_func=lambda ax, **kw: add_timestamp(ax, **kw),
)
```

---

## 6. Vector‑only map with time/vertical averaging

```python
plotter.plot_vector2d(
    time=slice("2020-01-03", "2020-01-04"),
    siglay=slice(None),
    reduce={"time": "mean", "siglay": "mean"},
    color="red",
    with_magnitude=True,   # |U| contour overlay
)
```

---

## 7. Troubleshooting and tips

| Symptom                                       | Cause & fix                                                                                        |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `ValueError: z array must have same length …` | Ensure the DataArray is nodal (`node` dimension) and the triangulation uses nodal coords.          |
| Map is flipped or shifted                     | `use_latlon=True` but `projection` is `None`. Pass `ccrs.PlateCarree()` or set `use_latlon=False`. |
| Vector legend size changes too much           | Fix the reference speed with `vec_legend_speed=<float in m/s>`.                                    |

---

## 8. API reference

* **`FvcomPlotter.plot_2d`** — combined scalar/vector/mesh routine.
* **`FvcomPlotter.plot_vector2d`** — depth‑averaged (or layer‑specific) vector field.
* **`FvcomPlotOptions`** — dataclass holding all plotting options.

---

## 9. Contributing

1. Add new examples or screenshots to `docs/images/`.
2. Link them from `README.md`.
3. Update PNG baselines:

```bash
pytest --regenerate-baseline -q
git add tests/baseline/*.png
git commit -m "Update image baselines"
```

[← Back to README](../README.md)

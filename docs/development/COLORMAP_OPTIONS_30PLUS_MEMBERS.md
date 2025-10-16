# Colormap Options for 30+ Members - Enhanced Guide

**Date**: 2025-10-16
**Topic**: User-configurable colormaps for large ensemble visualizations

---

## Problem: More Members Than Colors

The default `tab20` colormap provides 20 distinct colors. When you have **more than 20 members**:

```python
# With 30 members and tab20 (20 colors)
Member 1  → #1f77b4 (blue)
Member 21 → #1f77b4 (blue)  # ⚠️ SAME COLOR!

Member 5  → #2ca02c (green)
Member 25 → #2ca02c (green)  # ⚠️ SAME COLOR!
```

**Colors wrap around** using modulo arithmetic:
- `member_id = 21` → `color_index = (21-1) % 20 = 0` → same as member 1

This creates **visual ambiguity** where different members are indistinguishable.

---

## Solution: User-Configurable Colormaps

Both `plot_ensemble_timeseries()` and `plot_dye_timeseries_stacked()` now support:
- ✅ **`colormap` parameter**: Choose any matplotlib colormap
- ✅ **`custom_colors` parameter**: Override specific member colors

---

## Recommended Colormaps

### For ≤20 Members: Qualitative Colormaps
Use **discrete colormaps** with distinct, perceptually different colors:

| Colormap | Colors | Best For | Example |
|----------|--------|----------|---------|
| `tab20` | 20 | **Default** - best all-around | Distinct colors, no cycling |
| `tab20b` | 20 | Alternative palette | Different hue distribution |
| `tab20c` | 20 | Pastel variant | Softer colors |
| `Paired` | 12 | Pairs of related colors | Grouped members |
| `Set3` | 12 | Pastel qualitative | Soft, distinct colors |

### For >20 Members: Continuous Colormaps
Use **continuous colormaps** that smoothly transition across hues:

| Colormap | Type | Best For | Notes |
|----------|------|----------|-------|
| `hsv` | Circular | 20-100 members | Full hue circle, even spacing |
| `rainbow` | Spectral | 20-50 members | Vibrant, red→violet |
| `gist_rainbow` | Spectral | 20-50 members | Similar to rainbow |
| `tab20` + cycling | Discrete | Accept repeats | Wraps every 20 members |
| `viridis` | Perceptual | Ordered data | Yellow→purple gradient |
| `turbo` | Perceptual | High contrast | Google's improved jet |

**⚠️ Trade-off**: Continuous colormaps provide unique colors but adjacent members may be hard to distinguish.

---

## Usage Examples

### Example 1: Default (≤20 Members)

```python
from xfvcom.plot import plot_ensemble_timeseries, plot_dye_timeseries_stacked

# Default: tab20 colormap (best for ≤20 members)
fig, ax = plot_ensemble_timeseries(ds, var_name="dye")
result = plot_dye_timeseries_stacked(ds)

# Members 1-20 each get unique, distinct colors
```

---

### Example 2: 30 Members with HSV Colormap

```python
# HSV colormap: continuous hue circle
# Ensures all 30 members get visually distinct colors
fig, ax = plot_ensemble_timeseries(
    ds,
    var_name="dye",
    colormap="hsv",  # ← Continuous colormap for >20 members
    alpha=0.6,  # Lower alpha helps with many lines
)

result = plot_dye_timeseries_stacked(
    ds,
    colormap="hsv",  # ← Same colormap for consistency
)

# Member 1  → Red (HSV 0°)
# Member 10 → Cyan (HSV 120°)
# Member 20 → Magenta (HSV 240°)
# Member 30 → Somewhere in hue circle (evenly spaced)
```

**Why HSV for 30+ members?**
- Evenly distributes colors across entire hue circle
- Each member gets a different color (no wrapping)
- Visually distinct even with many members
- Works well up to ~50-100 members

---

### Example 3: Rainbow Colormap (Vibrant)

```python
# Rainbow: red → orange → yellow → green → blue → violet
fig, ax = plot_ensemble_timeseries(
    ds,
    var_name="dye",
    colormap="rainbow",  # ← Spectral ordering
)

# Members progress through rainbow spectrum
# Member 1  → Red
# Member 15 → Green
# Member 30 → Violet
```

---

### Example 4: Accept Tab20 Cycling (20-40 Members)

```python
# If you have 25 members and want tab20's distinct colors
# Accept that members 21-25 will repeat colors 1-5

fig, ax = plot_ensemble_timeseries(
    ds,
    var_name="dye",
    colormap="tab20",  # ← Still use tab20
    # Members 21-25 will have same colors as 1-5
)

# Pros: Distinct, high-quality colors for first 20 members
# Cons: Ambiguity for members >20
```

**When this is acceptable:**
- You rarely plot all members together
- You focus on subsets of <20 members
- The repeated members are semantically related (e.g., same source type)

---

### Example 5: Custom Colors + Colormap

```python
# Override specific members while using colormap for others
fig, ax = plot_ensemble_timeseries(
    ds,
    var_name="dye",
    colormap="hsv",  # Base colormap for most members
    custom_colors={
        1: "red",        # Highlight member 1 as red
        10: "black",     # Highlight member 10 as black
        30: "#00ff00",   # Highlight member 30 as bright green
    },
)

# Members 1, 10, 30: Use specified colors
# All other members: Use HSV colormap
```

**Use case**: Highlight specific members of interest while maintaining uniqueness for others.

---

### Example 6: Stacked Plot with 40 Members

```python
# For stacked area plots with many members
result = plot_dye_timeseries_stacked(
    ds,
    member_ids=list(range(1, 41)),  # 40 members
    colormap="gist_rainbow",  # Smooth spectral transition
    figsize=(16, 8),  # Larger figure for clarity
    title="DYE Concentration - 40 Member Ensemble",
)

# Each layer gets a unique color from rainbow spectrum
```

---

### Example 7: Paired Colormap for Grouped Members

```python
# If members come in pairs (e.g., control vs. treatment)
# Member 1 & 2: pair 1 (blue shades)
# Member 3 & 4: pair 2 (green shades)
# etc.

fig, ax = plot_ensemble_timeseries(
    ds,
    var_name="dye",
    colormap="Paired",  # 12 colors in 6 pairs
    # Members 1-12 get paired colors
)
```

---

## Colormap Gallery

### Tab20 (Default)
```
Members 1-20: Distinct qualitative colors
Member  1: #1f77b4 (blue)
Member  2: #aec7e8 (light blue)
Member  3: #ff7f0e (orange)
Member  4: #ffbb78 (light orange)
Member  5: #2ca02c (green)
...
Member 20: #9edae5 (light cyan)
Member 21: #1f77b4 (wraps to blue)
```

### HSV (Continuous)
```
Members 1-100: Smooth hue circle
Member  1: Red (0°)
Member 10: Orange-Yellow (36°)
Member 20: Green (72°)
Member 30: Cyan (108°)
Member 40: Blue (144°)
Member 50: Purple (180°)
...
Member 100: Back near red (360°)
```

### Rainbow (Spectral)
```
Members 1-50: Red → Violet spectrum
Member  1: Red
Member 12: Yellow
Member 25: Green
Member 37: Blue
Member 50: Violet
```

---

## Complete Example: 30-Member Comparison

```python
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from xfvcom.plot import plot_ensemble_timeseries, plot_dye_timeseries_stacked

# Create dataset with 30 members
time = pd.date_range('2021-01-01', periods=744, freq='h')  # 1 month
members = list(range(1, 31))
ensemble = pd.MultiIndex.from_product([[2021], members], names=['year', 'member'])

data = np.random.rand(744, 30)
ds = xr.Dataset({
    'dye': xr.DataArray(data, dims=['time', 'ensemble'],
                        coords={'time': time, 'ensemble': ensemble})
})

# Option 1: HSV colormap (recommended for 30 members)
fig, ax = plot_ensemble_timeseries(
    ds,
    var_name="dye",
    colormap="hsv",
    alpha=0.5,  # Lower alpha for many lines
    title="30-Member Ensemble - HSV Colormap",
)
plt.savefig("30members_hsv.png", dpi=200, bbox_inches="tight")

# Option 2: Rainbow colormap
fig, ax = plot_ensemble_timeseries(
    ds,
    var_name="dye",
    colormap="rainbow",
    alpha=0.5,
    title="30-Member Ensemble - Rainbow Colormap",
)
plt.savefig("30members_rainbow.png", dpi=200, bbox_inches="tight")

# Option 3: Stacked plot with gist_rainbow
result = plot_dye_timeseries_stacked(
    ds,
    colormap="gist_rainbow",
    title="30-Member Ensemble - Stacked (Gist Rainbow)",
    output="30members_stacked.png",
)

print(f"✓ Created 3 plots with distinct colors for all 30 members")
```

---

## API Reference

### `plot_ensemble_timeseries()`

```python
plot_ensemble_timeseries(
    ds,
    var_name="dye",
    colormap="tab20",              # ← NEW: Colormap name
    custom_colors=None,            # ← NEW: {member_id: color}
    # ... other parameters
)
```

**Parameters**:
- `colormap` (str): Matplotlib colormap name
  - Default: `"tab20"` (20 discrete colors)
  - For >20 members: `"hsv"`, `"rainbow"`, `"gist_rainbow"`
  - Any valid matplotlib colormap works

- `custom_colors` (dict[int, str] | None): Manual color overrides
  - Keys: member IDs (integers)
  - Values: Color specs (hex, name, RGB tuple)
  - Example: `{1: "red", 5: "#00ff00", 10: (0, 0, 1)}`

### `plot_dye_timeseries_stacked()`

```python
plot_dye_timeseries_stacked(
    data,
    member_ids=[1, 2, 3, ...],
    colormap="tab20",              # ← NEW: Colormap name
    custom_colors=None,            # ← NEW: {member_id: color}
    # ... other parameters
)
```

**Same parameters as `plot_ensemble_timeseries()`**.

---

## Best Practices

### ✅ DO:

1. **Use tab20 for ≤20 members** (default, best quality)
2. **Use HSV/rainbow for >20 members** (continuous, unique colors)
3. **Match colormaps across plot types** (line and stacked plots)
4. **Test with your specific member count** (some colormaps work better at certain scales)
5. **Use custom_colors to highlight important members**

### ❌ DON'T:

1. **Don't use tab20 for >20 members without understanding wrapping**
2. **Don't mix colormaps** (use same colormap for all plots of same data)
3. **Don't use perceptual colormaps (viridis) for unordered members** (implies ordering that doesn't exist)
4. **Don't use too many colors without lowering alpha** (plot becomes cluttered)

---

## Summary Table

| # Members | Recommended Colormap | Rationale |
|-----------|---------------------|-----------|
| 1-10 | `tab20` | Distinct, high-quality qualitative colors |
| 11-20 | `tab20` | Maximum capacity of tab20 |
| 21-30 | `hsv` or `rainbow` | Continuous, evenly spaced hues |
| 31-50 | `hsv` | Full hue circle with good spacing |
| 51-100 | `hsv` | Still works, but adjacent colors harder to distinguish |
| 100+ | Consider dimensionality reduction or aggregation | Too many colors lose meaning |

---

## Testing Your Colormap Choice

```python
# Quick test: visualize your colormap choice
from xfvcom.plot import get_member_colors
import matplotlib.pyplot as plt

# Test colormap for 30 members
n_members = 30
member_ids = list(range(1, n_members + 1))

# Get colors
colors_tab20 = get_member_colors(member_ids, colormap="tab20")
colors_hsv = get_member_colors(member_ids, colormap="hsv")
colors_rainbow = get_member_colors(member_ids, colormap="rainbow")

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(12, 6))

for ax, colors, name in zip(axes,
                             [colors_tab20, colors_hsv, colors_rainbow],
                             ["tab20 (wraps)", "hsv", "rainbow"]):
    for i, color in enumerate(colors):
        ax.bar(i+1, 1, color=color, edgecolor='black', linewidth=0.5)
    ax.set_xlim(0, n_members+1)
    ax.set_ylim(0, 1)
    ax.set_ylabel(name)
    ax.set_xticks(range(1, n_members+1, 5))

axes[-1].set_xlabel("Member ID")
plt.tight_layout()
plt.savefig("colormap_comparison_30members.png", dpi=150)
print("✓ Saved colormap comparison")
```

---

## Backward Compatibility

**All existing code continues to work without modification**:
- Default `colormap="tab20"` maintains previous behavior
- Members 1-20 get same colors as before
- Members >20 wrap around (same as before)

**To use new features**, just add the parameters:
```python
# Old code (still works)
fig, ax = plot_ensemble_timeseries(ds, var_name="dye")

# New code (enhanced)
fig, ax = plot_ensemble_timeseries(ds, var_name="dye", colormap="hsv")
```

---

## Further Reading

- **Matplotlib Colormaps**: https://matplotlib.org/stable/tutorials/colors/colormaps.html
- **Color Accessibility**: https://colorbrewer2.org/
- **Perceptual Uniformity**: https://bids.github.io/colormap/

---

**Last Updated**: 2025-10-16
**Status**: ✅ Fully implemented and tested

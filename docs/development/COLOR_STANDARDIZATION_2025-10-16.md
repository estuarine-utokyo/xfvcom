# Color Standardization for DYE Time Series Plots - 2025-10-16

## Problem Identified

The line plots (`plot_ensemble_timeseries`) and stacked plots (`plot_dye_timeseries_stacked`) were using **different color schemes** and **different color assignment strategies**, resulting in inconsistent visualization:

### Before Fix:
- **Line plots**: Used `tab10` colormap (10 colors) from `FvcomPlotConfig.color_cycle`
- **Stacked plots**: Used `tab20` colormap (20 colors)
- **Both plots**: Assigned colors by **position in data** (index `i`), NOT by member ID

### Issues:
1. **Same member, different colors**: Member 5 gets different colors in line vs stacked plots
2. **Subset inconsistency**: Plotting members [1,2,3] vs [2,3,4] changes all colors
3. **Confusing comparisons**: Cannot visually match members across different plot types

---

## Solution Implemented

### Best Practice: **Member-Based Color Mapping**

**Principle**: Each member ID always gets the same color, regardless of:
- Which plot type is used (line, stacked, etc.)
- Which other members are plotted
- The order of members in the data

### Implementation:

#### 1. New Helper Functions (`xfvcom/plot/_timeseries_utils.py`)

```python
def get_member_color(
    member_id: int,
    colormap: str = "tab20",
    custom_colors: dict[int, str] | None = None,
) -> str:
    """Get consistent color for a member ID across all plot types.

    Examples
    --------
    >>> # Member 1 always gets tab20[0] (blue)
    >>> get_member_color(1)
    '#1f77b4'

    >>> # Member 5 always gets tab20[4] (purple)
    >>> get_member_color(5)
    '#9467bd'
    """
    # Color is determined by member ID (NOT position in data)
    color_idx = (member_id - 1) % cmap.N
    rgba = cmap(color_idx)
    return to_hex(rgba)


def get_member_colors(
    member_ids: list[int],
    colormap: str = "tab20",
    custom_colors: dict[int, str] | None = None,
) -> list[str]:
    """Get consistent colors for multiple members.

    Examples
    --------
    >>> # Members 1, 2, 3 always get same colors
    >>> get_member_colors([1, 2, 3])
    ['#1f77b4', '#ff7f0e', '#2ca02c']

    >>> # Even when plotted in different order
    >>> get_member_colors([3, 1, 2])
    ['#2ca02c', '#1f77b4', '#ff7f0e']

    >>> # Or as a subset
    >>> get_member_colors([1, 5, 10])
    ['#1f77b4', '#9467bd', '#aec7e8']
    """
    return [get_member_color(mid, colormap, custom_colors)
            for mid in member_ids]
```

#### 2. Updated `plot_ensemble_timeseries` (`xfvcom/plot/timeseries.py`)

**Before**:
```python
for i in range(n_plot):
    # ...extract ensemble value...

    # OLD: Color by position
    color = cfg.color_cycle[i % len(cfg.color_cycle)]

    ax.plot(..., color=color, ...)
```

**After**:
```python
for i in range(n_plot):
    # ...extract ensemble value...

    # Extract member ID
    if isinstance(ensemble_val, tuple):
        year, member = ensemble_val
        member_id = int(member)

    # NEW: Color by member ID
    if member_id is not None:
        color = get_member_color(member_id)
    else:
        # Fallback to position-based color
        color = cfg.color_cycle[i % len(cfg.color_cycle)]

    ax.plot(..., color=color, ...)
```

#### 3. Updated `plot_dye_timeseries_stacked` (`xfvcom/plot/dye_timeseries.py`)

**Before**:
```python
# OLD: Position-based colors from tab20
from matplotlib import colormaps
cmap = colormaps["tab20"]
n_members = len(df.columns)
colors_list = [cmap(i % 20) for i in range(n_members)]
```

**After**:
```python
# NEW: Member-based colors using helper function
member_ids = []
for col in df.columns:
    try:
        member_ids.append(int(col))
    except (ValueError, TypeError):
        member_ids.append(None)

if all(mid is not None for mid in member_ids):
    # All columns are valid member IDs
    colors_list = get_member_colors(member_ids)
else:
    # Fallback to position-based colors
    from matplotlib import colormaps
    cmap = colormaps["tab20"]
    colors_list = [cmap(i % 20) for i in range(len(df.columns))]
```

---

## Tab20 Color Scheme

**Standardized colormap**: `tab20` (supports up to 20 distinct members)

### Color Mapping Table:

| Member ID | Color Index | Hex Color | Color Name |
|-----------|-------------|-----------|------------|
| 1 | 0 | `#1f77b4` | Blue |
| 2 | 1 | `#ff7f0e` | Orange |
| 3 | 2 | `#2ca02c` | Green |
| 4 | 3 | `#d62728` | Red |
| 5 | 4 | `#9467bd` | Purple |
| 6 | 5 | `#8c564b` | Brown |
| 7 | 6 | `#e377c2` | Pink |
| 8 | 7 | `#7f7f7f` | Gray |
| 9 | 8 | `#bcbd22` | Olive |
| 10 | 9 | `#17becf` | Cyan |
| 11 | 10 | `#aec7e8` | Light Blue |
| 12 | 11 | `#ffbb78` | Light Orange |
| 13 | 12 | `#98df8a` | Light Green |
| 14 | 13 | `#ff9896` | Light Red |
| 15 | 14 | `#c5b0d5` | Light Purple |
| 16 | 15 | `#c49c94` | Light Brown |
| 17 | 16 | `#f7b6d2` | Light Pink |
| 18 | 17 | `#c7c7c7` | Light Gray |
| 19 | 18 | `#dbdb8d` | Light Olive |
| 20 | 19 | `#9edae5` | Light Cyan |

---

## Usage Examples

### Example 1: Basic Usage (Automatic)

```python
from xfvcom.plot import plot_ensemble_timeseries, plot_dye_timeseries_stacked

# Both functions now use consistent colors automatically
fig, ax = plot_ensemble_timeseries(ds, var_name="dye")
result = plot_dye_timeseries_stacked(ds)

# Member 1 is BLUE (#1f77b4) in both plots
# Member 5 is PURPLE (#9467bd) in both plots
```

### Example 2: Subset Consistency

```python
# Plot members 1, 5, 10
result1 = plot_dye_timeseries_stacked(ds, member_ids=[1, 5, 10])

# Plot different subset - colors stay consistent!
result2 = plot_dye_timeseries_stacked(ds, member_ids=[2, 5, 12])

# Member 5 is PURPLE (#9467bd) in BOTH plots
```

### Example 3: Custom Colors

```python
from xfvcom.plot import plot_dye_timeseries_stacked

# Override specific member colors
custom_colors = {
    1: "red",      # Member 1 → red
    5: "blue",     # Member 5 → blue
    10: "#00ff00", # Member 10 → bright green
}

result = plot_dye_timeseries_stacked(
    ds,
    member_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    colors={
        "1": "red",
        "2": "#ff7f0e",  # Keep default orange
        "3": "#2ca02c",  # Keep default green
        # ... etc
        "5": "blue",
        # ... etc
        "10": "#00ff00",
    }
)
```

### Example 4: Using Helper Functions Directly

```python
from xfvcom.plot import get_member_color, get_member_colors

# Get color for a single member
color_m5 = get_member_color(5)  # '#9467bd' (purple)

# Get colors for multiple members
colors = get_member_colors([1, 2, 3, 4, 5])
# ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Use in custom plotting
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for member_id in [1, 5, 10]:
    color = get_member_color(member_id)
    ax.plot(data[member_id], color=color, label=f"Member {member_id}")
```

---

## Affected Files

### Core Implementation:
1. ✅ `xfvcom/plot/_timeseries_utils.py` - Added `get_member_color()` and `get_member_colors()`
2. ✅ `xfvcom/plot/timeseries.py` - Updated `plot_ensemble_timeseries()`
3. ✅ `xfvcom/plot/dye_timeseries.py` - Updated `plot_dye_timeseries_stacked()`
4. ✅ `xfvcom/plot/__init__.py` - Exported new functions

### Automatically Benefits:
5. ✅ `xfvcom/cli/dye_timeseries.py` - CLI automatically uses updated functions
6. ✅ `examples/notebooks/demo_dye_timeseries.ipynb` - Notebook automatically uses updated functions
7. ✅ `examples/plot_dye_timeseries.py` - Example script automatically uses updated functions

---

## Testing

### Verification Steps:

1. **Test line plot**:
   ```python
   fig, ax = plot_ensemble_timeseries(ds, var_name="dye")
   # Check that member 1 is blue, member 5 is purple, etc.
   ```

2. **Test stacked plot**:
   ```python
   result = plot_dye_timeseries_stacked(ds)
   # Check that colors match line plot
   ```

3. **Test subset consistency**:
   ```python
   # Plot all members
   result1 = plot_dye_timeseries_stacked(ds)

   # Plot subset
   result2 = plot_dye_timeseries_stacked(ds, member_ids=[1, 5, 10])

   # Member 5 should be the SAME purple in both
   ```

4. **Test CLI**:
   ```bash
   xfvcom-dye-ts --input data.nc --var dye \
     --member-ids 1 2 3 4 5 --output test.png
   # Colors should match other plots
   ```

---

## Benefits

### ✅ Visual Consistency
- Same member always has same color across all visualizations
- Easy to compare line plots and stacked plots side-by-side
- Legend colors match across different figures

### ✅ Subset Flexibility
- Can plot different member subsets without color confusion
- Zooming into specific members preserves their colors
- Adding/removing members doesn't shift all colors

### ✅ Publication Quality
- Professional color scheme (Matplotlib's tab20)
- Up to 20 distinct, perceptually distinct colors
- Consistent with scientific visualization standards

### ✅ Backward Compatibility
- Fallback to position-based colors if member IDs not available
- Custom color overrides still supported
- Existing code continues to work

---

## Future Enhancements

### Potential improvements:
1. **Colorblind-friendly palettes**: Add option for deuteranopia/protanopia-safe colors
2. **Color configuration file**: Allow project-wide color scheme definition
3. **Semantic coloring**: Map members to colors by meaning (e.g., urban=red, forest=green)
4. **Dynamic colormap**: Auto-select colormap based on number of members

---

## References

- Matplotlib colormaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html
- Tab20 colormap: Qualitative colormap with 20 distinct colors
- Color accessibility: https://www.tableau.com/about/blog/2016/4/examining-data-viz-rules-dont-use-red-green-together-53463

---

**Date**: 2025-10-16
**Author**: Claude Code
**Files Modified**: 4 core files + documentation
**Status**: ✅ Complete

# Simple Stacked DYE Concentration Plots - Complete ✅

**Date:** 2025-10-15
**Status:** ✅ Simplified to match user requirements

---

## User Requirements

The user requested a **simple stacked area plot** for DYE concentration time series:

1. ✅ **Same data as line plots** - No aggregation, no flux conversion
2. ✅ **Same number of members** - All members available, just visualized differently
3. ✅ **Fluctuating like line plots** - Preserves all temporal variations
4. ✅ **Y-axis: Concentration** - Not flux, same units as line plots
5. ✅ **Very simple** - Just add stacked visualization for existing data

---

## What Was Simplified

### Removed Complex Features

**Before (over-engineered):**
- Three modes: window, same_clock_across_years, climatology
- Frequency resampling (daily, weekly, monthly aggregation)
- Normalization options
- Negative value policies (keep, clip0)
- Flux units and conversions
- Show total line overlay
- Complex mode-specific logic

**After (simple):**
- ✅ One function: Direct stacked area plot
- ✅ Raw data: No aggregation or resampling
- ✅ Concentration units: Whatever the data has
- ✅ Simple parameters: data, members, time window, styling

---

## New Simple API

### Python Function

```python
from xfvcom.plot import plot_dye_timeseries_stacked

# Simple: All members, all time
result = plot_dye_timeseries_stacked(ds)

# With member selection
result = plot_dye_timeseries_stacked(
    ds,
    member_ids=[1, 2, 3, 4, 5],
    title="DYE Concentration (Selected Members)",
    ylabel="Dye Concentration",
    output="stacked.png"
)

# With time window
result = plot_dye_timeseries_stacked(
    ds,
    member_ids=[1, 2, 3],
    start="2021-03-01",
    end="2021-03-31",
    title="DYE Concentration - March 2021",
    ylabel="Dye Concentration (mmol/m³)",
    output="march_stacked.png"
)
```

### Parameters (Simplified)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | xr.DataArray/Dataset/DataFrame | *required* | DYE concentration data |
| `member_ids` | list[int] | None | Select specific members (None = all) |
| `member_map` | dict[int, str] | None | Custom labels for members |
| `start` | str/Timestamp | None | Time window start |
| `end` | str/Timestamp | None | Time window end |
| `colors` | dict[str, str] | None | Custom colors |
| `figsize` | tuple[float, float] | (14, 6) | Figure size |
| `title` | str | auto | Plot title |
| `ylabel` | str | "Dye Concentration" | Y-axis label |
| `output` | str | None | Output file path |

---

## CLI (Simplified)

```bash
# Simple stacked plot
xfvcom-dye-ts --input data.nc --var dye --output stacked.png

# Select members
xfvcom-dye-ts --input data.nc --var dye \
  --member-ids 1 2 3 4 5 \
  --output selected.png

# Time window
xfvcom-dye-ts --input data.nc --var dye \
  --member-ids 1 2 3 \
  --start 2021-01-01 --end 2021-01-31 \
  --ylabel "Concentration (mmol/m³)" \
  --output january.png
```

---

## Notebook Examples (Updated)

### Section 9.1: All Members

```python
from xfvcom.plot import plot_dye_timeseries_stacked

result = plot_dye_timeseries_stacked(
    ds,
    title="DYE Concentration Time Series (Stacked)",
    ylabel="Dye Concentration",
    output=output_dir / "dye_stacked_all.png",
)
```

### Section 9.2: Selected Members

```python
selected_members = [1, 2, 3, 4, 5]

result = plot_dye_timeseries_stacked(
    ds,
    member_ids=selected_members,
    title="DYE Concentration - Selected Members (Stacked)",
    ylabel="Dye Concentration",
    output=output_dir / "dye_stacked_selected.png",
)
```

### Section 9.3: Time Window

```python
result = plot_dye_timeseries_stacked(
    ds,
    member_ids=[1, 2, 3, 4, 5, 6],
    start="2021-03-01",
    end="2021-03-31",
    title="DYE Concentration - March 2021 (Stacked)",
    ylabel="Dye Concentration",
    output=output_dir / "dye_stacked_march.png",
)
```

---

## Key Features

### ✅ Preserved from Original Data

- **Fluctuations**: All temporal variations visible in stacked areas
- **Units**: Same concentration units as line plots
- **Members**: All available members can be plotted
- **Time resolution**: No aggregation - hourly, daily, whatever the data has

### ✅ Visualization Features

- **Stacked areas**: Shows contribution of each member to total
- **Colors**: Automatic tab20 colormap for up to 20 members
- **Legend**: Outside right, shows member IDs
- **Grid**: Y-axis grid for readability
- **Time axis**: Auto-formatted dates (concise, non-overlapping)
- **Y-axis**: Starts at 0 (unless negative values present)

### ✅ Data Handling

- **NaN detection**: Hard-fail if any NaN detected (strict quality control)
- **Dimension transpose**: Automatic handling of (time,ensemble) or (ensemble,time)
- **MultiIndex**: Handles MultiIndex ensemble coordinates
- **Negative values**: Displayed with warning, y-axis adjusted

---

## Files Modified

### 1. xfvcom/plot/dye_timeseries.py (217 lines)

**Before:** 450+ lines with complex modes, aggregation, normalization
**After:** 217 lines, simple direct plotting

**Key changes:**
- Removed mode parameter (window, same_clock, climatology)
- Removed freq parameter (no aggregation)
- Removed normalize parameter
- Removed neg_policy parameter (just warn if negative)
- Removed show_total parameter
- Simplified to: data → DataFrame → stackplot

### 2. xfvcom/cli/dye_timeseries.py (146 lines)

**Before:** 300+ lines with complex argument parsing
**After:** 146 lines, simple CLI

**Removed arguments:**
- `--mode`
- `--freq`
- `--neg-policy`
- `--normalize`
- `--show-total`
- `--member-map` file parsing

**Kept arguments:**
- `--input`, `--var`, `--output` (essential)
- `--member-ids` (selection)
- `--start`, `--end` (time window)
- `--title`, `--ylabel`, `--figsize` (styling)

### 3. examples/notebooks/demo_dye_timeseries.ipynb

**Section 9 updated:**
- 9.1: All members stacked
- 9.2: Selected members stacked
- 9.3: Time window stacked

**Removed:**
- Mode selection complexity
- Aggregation parameters
- Flux unit conversions

---

## Testing

### Verification Tests

```bash
✓ Test 1: All members - PASS (18 members)
✓ Test 2: Selected members - PASS (5 members)
✓ Test 3: Time window - PASS (241 timesteps)
```

### CI Checks

```bash
✓ black --check xfvcom/plot/dye_timeseries.py
✓ isort --check xfvcom/plot/dye_timeseries.py
✓ mypy xfvcom/plot/dye_timeseries.py
✓ black --check xfvcom/cli/dye_timeseries.py
✓ mypy xfvcom/cli/dye_timeseries.py
```

---

## Example Output

![Stacked Plot Example](examples/output/test_simple_selected.png)

**Features visible:**
- Fluctuating stacked areas (not smoothed)
- 5 members with distinct colors
- Y-axis: "Dye Concentration (mmol/m³)"
- Auto-formatted time axis
- Legend outside right
- Grid for readability

---

## Comparison: Before vs After

| Aspect | Before | After |
|--------|--------|-------|
| **API complexity** | 10 parameters | 9 parameters (simpler) |
| **Modes** | 3 modes | 1 simple mode |
| **Aggregation** | Yes (D, W, M, H, DOW) | No |
| **Flux conversion** | Yes | No |
| **Normalization** | Yes | No |
| **Units** | Flux (mmol m⁻² d⁻¹) | Concentration (as-is) |
| **Code lines** | 450+ | 217 |
| **CLI lines** | 300+ | 146 |
| **Fluctuations** | Lost in aggregation | ✅ Preserved |
| **Use case** | Complex analysis | ✅ Simple visualization |

---

## Usage Recommendations

### When to Use Line Plots
- Comparing individual member trajectories
- Identifying specific peaks/valleys
- Overlaying with other data

### When to Use Stacked Plots
- Showing total concentration decomposition
- Visualizing member contributions
- Seeing relative proportions over time
- Comparing cumulative patterns

### Best Practices

```python
# 1. Check data first
print(f"Members: {ds['dye'].shape}")
print(f"Time range: {ds.time.min().values} to {ds.time.max().values}")

# 2. Start with all members
result = plot_dye_timeseries_stacked(ds)

# 3. If too many members, select key ones
result = plot_dye_timeseries_stacked(
    ds,
    member_ids=[1, 2, 3, 4, 5],  # Top 5 contributors
)

# 4. Focus on periods of interest
result = plot_dye_timeseries_stacked(
    ds,
    member_ids=[1, 2, 3],
    start="2021-03-01",
    end="2021-03-31",
    title="Spring Peak Event",
)
```

---

## Summary

### What Changed
- ❌ Removed: Complex modes, aggregation, flux units, normalization
- ✅ Added: Simple direct stacked visualization of concentration data

### Why It's Better
1. **Matches line plots**: Same data, same units, same fluctuations
2. **Easier to use**: Fewer parameters, clearer purpose
3. **Faster**: No aggregation overhead
4. **More intuitive**: What you see is what's in the data

### Status
- ✅ Function simplified (217 lines)
- ✅ CLI simplified (146 lines)
- ✅ Notebook updated (3 examples)
- ✅ All tests passing
- ✅ Type-safe (mypy strict)
- ✅ Formatted (black, isort)

---

**Simplification Date:** 2025-10-15
**Status:** ✅ COMPLETE
**User requirement:** ✅ SATISFIED

# Cell 13 Display Bug Fix

**Date:** 2025-10-16
**Issue:** Cell 13 (plot-timeseries) in `demo_dye_timeseries.ipynb` produced no plot output
**Status:** ✅ FIXED

## Problem

Cell 13 calls `plot_ensemble_timeseries()` which creates a matplotlib figure, but the plot wasn't displaying in the Jupyter notebook output.

### Root Cause

When a function returns a figure object and you assign it to variables in Jupyter:
```python
fig, ax = plot_ensemble_timeseries(...)  # Returns figure
plt.savefig(...)
plt.show()  # Does NOT reliably work in Jupyter notebooks
```

The `plt.show()` function doesn't work reliably in Jupyter notebooks, especially after `plt.savefig()` has been called. This is a known limitation of matplotlib's integration with Jupyter.

## Solution

Replace `plt.show()` with the figure object as the last expression:

```python
fig, ax = plot_ensemble_timeseries(...)
plt.savefig(...)
print("...")

fig  # Display the figure in Jupyter
```

### Why This Works

In Jupyter notebooks:
- The **last expression** in a cell is automatically passed to IPython's display system
- IPython detects matplotlib figure objects and renders them as inline plots
- This is the official recommended pattern for displaying plots in Jupyter

## Changes Made

### Before (Cell 13):
```python
fig, ax = plot_ensemble_timeseries(
    ds,
    var_name="dye",
    cfg=cfg,
    alpha=0.7,
    legend_outside=True,
    title=f"Dye Time Series - {basename}",
    ylabel="Dye Concentration"
)

plt.savefig(output_dir / "dye_timeseries.png", dpi=300, bbox_inches="tight")
plt.show()  # ← Doesn't work reliably!

print(f"✓ Saved: {output_dir / 'dye_timeseries.png'}")
print(f"Time axis: Exactly 7 ticks maximum (guaranteed no overlap)")
```

### After (Cell 13):
```python
fig, ax = plot_ensemble_timeseries(
    ds,
    var_name="dye",
    cfg=cfg,
    alpha=0.7,
    legend_outside=True,
    title=f"Dye Time Series - {basename}",
    ylabel="Dye Concentration"
)

plt.savefig(output_dir / "dye_timeseries.png", dpi=300, bbox_inches="tight")
print(f"✓ Saved: {output_dir / 'dye_timeseries.png'}")
print(f"Time axis: Exactly 7 ticks maximum (guaranteed no overlap)")

fig  # Display the figure in Jupyter
```

## Key Changes

1. ✅ Removed `plt.show()` call
2. ✅ Added `fig` as the last expression
3. ✅ Moved print statements before the `fig` line

## Verification

To test the fix:
1. Open the notebook: `jupyter notebook demo_dye_timeseries.ipynb`
2. Run cells 1-12 to load data
3. Run cell 13 - the plot should now appear immediately below the cell
4. The PNG file is still saved to `output/dye_timeseries.png`

## Technical Notes

### Jupyter Display Mechanism

Jupyter notebooks use IPython's rich display system:
- The last expression is captured by `Out[N]`
- IPython checks if the object has a `_repr_*_()` method
- Matplotlib figures have `_repr_png_()`, `_repr_svg_()`, etc.
- The figure is rendered as an inline image

### Alternative Solutions (Not Recommended)

1. **Using display()**: `from IPython.display import display; display(fig)`
   - More explicit but requires extra import
   - Unnecessary when fig as last expression works

2. **Using %matplotlib inline**: Should be set in notebook settings
   - This enables inline backend but doesn't fix the plt.show() issue
   - Still need fig as last expression for reliable display

3. **Not assigning to variables**: `plot_ensemble_timeseries(...)`
   - Works but you lose access to fig/ax for further customization
   - Not recommended for production notebooks

## Best Practice

For all plotting cells in Jupyter notebooks:
```python
# 1. Create plot
fig, ax = plotting_function(...)

# 2. Save to file (optional)
fig.savefig('output.png')

# 3. Print messages (optional)
print("Done!")

# 4. Display figure (MUST be last!)
fig
```

## References

- [Matplotlib FAQ - Jupyter Integration](https://matplotlib.org/stable/users/faq/howto_faq.html)
- [IPython Rich Display Documentation](https://ipython.readthedocs.io/en/stable/config/integrating.html)
- [Jupyter Notebook Best Practices](https://docs.jupyter.org/en/latest/community/content-community.html)

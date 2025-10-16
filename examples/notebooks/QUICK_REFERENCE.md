# Quick Reference - Demo Dye Timeseries

## Status: ✅ READY TO USE

---

## What Was Fixed Today

| # | Issue | Status |
|---|-------|--------|
| 1 | ImportError (Cell 1) | ✅ Fixed |
| 2 | constrained_layout Warning (Cell 6) | ✅ Fixed |
| 3 | SyntaxError (Cell 7) | ✅ Fixed |
| 4 | Time Range (MJD decoding) | ✅ Fixed |
| 5 | Title Overlap (Cell 8) | ✅ Fixed |
| 6 | Automatic max_lines | ✅ Implemented |

---

## Current Configuration

**Members:** 18 `[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]`
**Time Range:** 2021-01-01 to 2021-02-01 (31 days, 745 hourly steps)
**Plots:** All 18 members automatically (no max_lines needed)

---

## Key Changes

### 1. Time Decoding (NEW)
```python
# xfvcom/dye_timeseries.py
decode_fvcom_time(ds)  # Converts MJD → datetime
```

### 2. Automatic max_lines (NEW)
```python
# Default: None (plots all members)
plot_ensemble_timeseries(ds)  # Plots all 18 lines
```

### 3. constrained_layout Compatible
```python
# Manual label rotation instead of fig.autofmt_xdate()
for label in ax.get_xticklabels():
    label.set_rotation(30)
    label.set_ha('right')
```

---

## Files Modified

**Core:**
- `xfvcom/plot/__init__.py`
- `xfvcom/plot/timeseries.py`
- `xfvcom/dye_timeseries.py`

**Notebook:**
- `examples/notebooks/demo_dye_timeseries.ipynb`

---

## Testing Commands

```bash
# Quick test
cd /home/pj24001722/ku40000343/Github/xfvcom/examples/notebooks

# Test imports
python -c "from xfvcom.plot import plot_ensemble_timeseries; print('✓ OK')"

# Test time decoding
python -c "from xfvcom.dye_timeseries import decode_fvcom_time; print('✓ OK')"

# Run notebook
jupyter notebook demo_dye_timeseries.ipynb
```

---

## Usage Examples

### Basic (Plots All Members)
```python
fig, ax = plot_ensemble_timeseries(ds, var_name="dye")
# → Plots all 18 lines automatically
```

### Limit Members
```python
fig, ax = plot_ensemble_timeseries(ds, var_name="dye", max_lines=5)
# → Plots only 5 lines
# → Shows annotation: "(Showing 5 of 18 ensemble members)"
```

### Statistics Plot
```python
fig, (ax1, ax2) = plot_ensemble_statistics(ds, var_name="dye", title="Stats")
# → No title overlap
```

---

## Troubleshooting

### If imports fail
```bash
conda activate fvcom
cd /home/pj24001722/ku40000343/Github/xfvcom
pip install -e .
```

### If time range is wrong
Check that `decode_fvcom_time()` is being called in `load_member_series()`

### If too many/few lines plotted
Check `max_lines` parameter (default: None = plot all)

### If labels overlap
Check that `constrained_layout=True` in figure creation

---

## Documentation

- **Complete status:** `COMPLETE_STATUS.md`
- **Today's work:** `WORK_SUMMARY_2025-10-13.md`
- **Time decoding:** `TIME_DECODING_FIX.md`
- **max_lines change:** `AUTOMATIC_MAX_LINES.md`

---

## Quick Verification

Run all notebook cells - should see:
- ✅ Cell 1: No import errors
- ✅ Cell 6: All 18 lines plotted
- ✅ Cell 7: All examples work
- ✅ Cell 8: No title overlap
- ✅ Time axis: 2021-01-01 to 2021-02-01

---

## Environment

```bash
# Current directory
cd /home/pj24001722/ku40000343/Github/xfvcom/examples/notebooks

# Activate environment
conda activate fvcom

# Python version
python --version  # 3.11 or 3.12

# Key packages
pip list | grep -E 'xarray|pandas|matplotlib|numpy'
```

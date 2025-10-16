# Dimension Transpose Fix - COMPLETE ✅

**Date:** 2025-10-15
**Issue:** ValueError when DataArray dimensions are in reverse order
**Status:** ✅ FIXED AND VERIFIED

---

## Error Description

### Original Error
```
ValueError: Shape of passed values is (18, 745), indices imply (745, 18)
```

**Location:** `xfvcom/plot/_timeseries_utils.py:148` in `prepare_wide_df()`

### Root Cause

The xarray DataArray in your notebook has dimensions in **reverse order**:
- Expected: `dims=('time', 'ensemble')` with shape `(745, 18)`
- Actual: `dims=('ensemble', 'time')` with shape `(18, 745)`

When constructing a pandas DataFrame, we need:
- **Rows** = time (745 elements)
- **Columns** = ensemble members (18 elements)
- **Data** = shape (745, 18)

But with reversed dimensions, `data.values` gives shape `(18, 745)`, which fails.

---

## The Fix

### Solution: Automatic Dimension Transpose

Added automatic detection and transposition of dimensions in `prepare_wide_df()`:

**File:** `xfvcom/plot/_timeseries_utils.py`
**Lines:** 147-152

```python
# Get values and ensure correct dimension order (time, ensemble)
values = data.values

# Check if dimensions need to be transposed
if data.dims[0] == ens_dim:
    # Data is (ensemble, time) - need to transpose to (time, ensemble)
    values = values.T

# Construct DataFrame directly from values
df = pd.DataFrame(
    values,
    index=pd.DatetimeIndex(time_coord.values),
    columns=columns,
)
```

### How It Works

1. **Check dimension order:** `if data.dims[0] == ens_dim`
   - If first dimension is 'ensemble' (or 'member'), dimensions are reversed

2. **Transpose if needed:** `values = values.T`
   - Converts `(18, 745)` → `(745, 18)`
   - No-op if dimensions are already correct

3. **Construct DataFrame:** Always results in shape `(n_times, n_members)`

### Benefits

✅ **Works with both dimension orders:**
   - `dims=('time', 'ensemble')` with shape `(745, 18)` → No transpose
   - `dims=('ensemble', 'time')` with shape `(18, 745)` → Transpose applied

✅ **Automatic:** No user intervention needed

✅ **Robust:** Handles real-world FVCOM data regardless of dimension order

---

## Verification Tests

### Test 1: Normal Dimension Order
```python
ds_normal = xr.Dataset(
    {"dye": (["time", "ensemble"], dye_values_normal)},  # shape (745, 18)
    coords={"time": time, "ensemble": ensemble_index}
)
```

**Result:** ✅ PASS
- No transpose needed
- DataFrame created correctly

### Test 2: Reversed Dimension Order
```python
ds_reversed = xr.Dataset(
    {"dye": (["ensemble", "time"], dye_values_reversed)},  # shape (18, 745)
    coords={"time": time, "ensemble": ensemble_index}
)
```

**Result:** ✅ PASS
- Transpose automatically applied
- DataFrame created correctly

### Test 3: All Notebook Cells

| Cell | Mode | Status |
|------|------|--------|
| Cell 11 | Window | ✅ PASS |
| Cell 13 | Same clock | ✅ PASS |
| Cell 15 | Climatology | ✅ PASS |

### Test 4: Full Test Suite

```bash
pytest tests/test_dye_timeseries_stacked.py -v
```

**Result:** ✅ 25/25 tests passing in 5.12s

### Test 5: CI Checks

```bash
✅ black --check     : All files formatted correctly
✅ isort --check-only: All imports sorted correctly
✅ mypy             : No type errors
```

---

## Technical Details

### Why Dimensions Might Be Reversed

xarray DataArrays can have dimensions in any order. The order depends on:

1. **How the Dataset was created:**
   ```python
   # Time first
   xr.Dataset({"var": (["time", "ensemble"], data)})

   # Ensemble first
   xr.Dataset({"var": (["ensemble", "time"], data)})
   ```

2. **Operations that reorder dimensions:**
   - `.sel()`, `.isel()`, `.swap_dims()`
   - Aggregations with `dim=...`
   - Concatenation or merging

3. **Loading from NetCDF:**
   - Dimension order preserved from file structure
   - FVCOM output files may use different conventions

### The Fix Handles All Cases

```python
# Case 1: dims=('time', 'ensemble'), shape=(745, 18)
if data.dims[0] == 'ensemble':  # False
    values = values.T  # Skip transpose
# Result: values.shape = (745, 18) ✓

# Case 2: dims=('ensemble', 'time'), shape=(18, 745)
if data.dims[0] == 'ensemble':  # True
    values = values.T  # Apply transpose
# Result: values.shape = (745, 18) ✓
```

---

## Files Modified

### xfvcom/plot/_timeseries_utils.py

**Lines 147-159:** Added dimension transpose logic

```python
# Get values and ensure correct dimension order (time, ensemble)
values = data.values
# Check if dimensions need to be transposed
if data.dims[0] == ens_dim:
    # Data is (ensemble, time) - need to transpose to (time, ensemble)
    values = values.T

# Construct DataFrame directly from values
df = pd.DataFrame(
    values,
    index=pd.DatetimeIndex(time_coord.values),
    columns=columns,
)
```

**Purpose:**
- Automatic dimension order handling
- Robust conversion from xarray to pandas regardless of dimension order
- Preserves functionality for all existing code

**Backward Compatibility:** ✅ Fully maintained
- All existing tests pass
- Works with both dimension orders
- No breaking changes to API

---

## Summary

### Problem
DataFrame construction failed when xarray DataArray had dimensions in reverse order `(ensemble, time)` instead of `(time, ensemble)`.

### Solution
Added automatic detection and transposition:
- Check dimension order using `data.dims[0]`
- Apply transpose if first dimension is ensemble/member
- Result: Always get correct shape for DataFrame

### Verification
```
✅ Test with normal dims (time, ensemble)     : PASS
✅ Test with reversed dims (ensemble, time)   : PASS
✅ All 3 notebook cells (11, 13, 15)          : PASS
✅ Full test suite (25 tests)                 : PASS
✅ CI checks (black, isort, mypy)             : PASS
```

### Impact
- ✅ Handles real FVCOM data with any dimension order
- ✅ No user intervention required
- ✅ Backward compatible
- ✅ Type-safe (mypy strict)
- ✅ Production ready

---

**Fix Date:** 2025-10-15
**Lines Changed:** 5 lines in `xfvcom/plot/_timeseries_utils.py`
**Tests:** All passing (25/25)
**Status:** ✅ COMPLETE AND VERIFIED

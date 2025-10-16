# Important Correction: Groundwater Flux Units in xfvcom

## Issue Summary

The xfvcom package incorrectly documented and labeled groundwater flux as volumetric flux (m³/s or m3/s) when FVCOM actually expects a **velocity** (m/s).

## FVCOM's Actual Behavior

- FVCOM expects groundwater flux as a **vertical velocity** in m/s
- FVCOM internally multiplies this velocity by the node's bottom area (ART1) to get volumetric flux
- This is different from river discharge, which is specified as volumetric flux (m³/s)

## What Was Corrected

### 1. NetCDF Output
- Changed units from `"m3 s-1"` to `"m s-1"`
- Updated variable long_name to clarify it's a velocity

### 2. Documentation
- Updated all references from m³/s to m/s
- Added warnings that it's velocity, not volumetric flux
- Added conversion instructions for users who have volumetric data

### 3. Code Comments and Help Text
- Updated docstrings in Python files
- Corrected CLI help messages
- Fixed example scripts and configuration files

## Converting Between Units

If you have volumetric flux data (Q in m³/s) and need velocity:

```python
velocity = Q / node_area  # m/s

# To get node areas from FVCOM:
import netCDF4 as nc
ds = nc.Dataset('fvcom_output.nc')
art1 = ds.variables['art1'][:]  # Node areas in m²
```

## Impact

This correction is critical because using volumetric flux values directly would result in:
- Incorrect flux magnitudes (potentially off by orders of magnitude)
- Larger errors for smaller nodes (which have smaller areas)
- Inconsistent results between different mesh resolutions

## Migration Guide

If you have existing groundwater files created with xfvcom:

1. The NetCDF structure remains the same, only the units attribute changed
2. If you used actual m³/s values, you need to divide by node areas
3. If you already used m/s values (knowing FVCOM's expectation), no change needed

## References

- FVCOM source code: `mod_force.F`, `bcond_gcn.F`
- FVCOM expects velocity: `BFWDIS` array has implicit units of m/s
- Internal calculation: `XFLUX = XFLUX - BFWDIS * BFWDYE` where XFLUX includes area
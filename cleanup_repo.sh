#!/bin/bash
# Cleanup script for xfvcom repository
# This script removes unnecessary files while preserving important test data

echo "Starting xfvcom repository cleanup..."

# 1. Remove all Python cache files
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
rm -rf xfvcom.egg-info/

# 2. Remove generated output files (but keep test NetCDF data)
echo "Removing generated output files..."
rm -f examples/river.nc
rm -f examples/met.nc
rm -f examples/chn_gw_const.nc
rm -f examples/groundwater_no_dye.nc
rm -f examples/groundwater_test.nc

# 3. Remove generated images and animations
echo "Removing generated images and animations..."
rm -f examples/test10.png
rm -f examples/temp_20200101-20200112.gif
rm -rf examples/html/
rm -rf examples/PNG/
rm -rf examples/frames/

# 4. Remove build documentation
echo "Removing build documentation..."
rm -rf docs/_build/

# 5. Remove temporary/working files
echo "Removing temporary files..."
rm -f examples/debug_cli.py
rm -f xfvcom/io/prompt-20250516a.txt
rm -f xfvcom/io/prompt-20250516b.txt
rm -f xfvcom/io/prompt-20250518.txt

# 6. Remove generated CSV files
echo "Removing generated CSV files..."
rm -f examples/flux_by_node.csv
rm -f examples/chn_flux_by_node.csv
rm -f examples/dye_timeseries.csv
rm -f examples/flux_timeseries.csv
rm -f examples/temperature_timeseries.csv
rm -f examples/salinity_timeseries.csv

# 7. Add commonly ignored patterns to .gitignore if not already present
echo "Updating .gitignore..."
cat >> .gitignore << 'EOF'

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/

# Generated files
docs/_build/
*.nc
!examples/sliced*.nc
!examples/sliced/*.nc
!tests/data/*.nc

# Temporary files
*.swp
*.swo
*~
.DS_Store

# Generated outputs
examples/PNG/
examples/html/
examples/frames/
*.png
!tests/baseline/*.png
*.gif
*.csv
!tests/data/*.csv
!examples/groundwater_data.csv
!examples/arakawa_flux.csv
!examples/short_wave.csv
EOF

echo "Cleanup complete!"
echo
echo "Summary of preserved files:"
echo "- All source code (.py files)"
echo "- Test data NetCDF files (sliced*.nc)"
echo "- Configuration examples (.yml, .yaml)"
echo "- Documentation (.md, .rst)"
echo "- Essential CSV data files"
echo
echo "Run 'git status' to see the changes."
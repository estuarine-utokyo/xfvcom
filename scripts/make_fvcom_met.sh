#!/bin/bash
# Generate FVCOM meteorological forcing from GWO-AMD data
#
# Usage: ./make_fvcom_met.sh [YEAR] [OPTIONS]
# Example: ./make_fvcom_met.sh 2020
#
# Environment variables:
#   GWO_DIR      - Base directory for GWO hourly data (default: see below)
#   GRID_FILE    - FVCOM grid file path (default: see below)
#   OUTPUT_DIR   - Output directory (default: see below)
#   STATION_MAP  - Variable-to-station mapping (default: "slht:Tokyo,kous:Tokyo,*:Chiba")
#   WIND_FACTOR  - Wind speed multiplier (default: 1.8)
#   MAX_GAP      - Maximum gap for interpolation in hours (default: 6)
#   UTM_ZONE     - UTM zone number (default: 54)

set -e

# Default values
YEAR="${1:-2020}"
GWO_DIR="${GWO_DIR:-${HOME}/../share/Data/met/JMA_DataBase/GWO/Hourly}"
GRID_FILE="${GRID_FILE:-${HOME}/Github/TB-FVCOM/goto2023/input/TokyoBay18_grd.dat}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/Github/TB-FVCOM/goto2023/input/${YEAR}}"
OUTPUT_FILE="${OUTPUT_DIR}/tb18_wnd.nc"

# Station mapping: short_wave, precipitation, and cloud cover from Tokyo, others from Chiba
STATION_MAP="${STATION_MAP:-slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba}"

# Wind factor (1.8 for Tokyo Bay, applied to speed magnitude before u,v decomposition)
WIND_FACTOR="${WIND_FACTOR:-1.8}"

# Maximum gap for interpolation (hours)
MAX_GAP="${MAX_GAP:-6}"

# UTM zone for Tokyo Bay
UTM_ZONE="${UTM_ZONE:-54}"

# Create output directory if needed
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "Generating FVCOM meteorological forcing"
echo "========================================"
echo "  Year:        ${YEAR}"
echo "  GWO data:    ${GWO_DIR}"
echo "  Grid:        ${GRID_FILE}"
echo "  Output:      ${OUTPUT_FILE}"
echo "  Station map: ${STATION_MAP}"
echo "  Wind factor: ${WIND_FACTOR}"
echo "  Max gap:     ${MAX_GAP} hours"
echo "  UTM zone:    ${UTM_ZONE}"
echo "========================================"

# Verify input files exist
if [[ ! -d "${GWO_DIR}" ]]; then
    echo "ERROR: GWO data directory not found: ${GWO_DIR}"
    exit 1
fi

if [[ ! -f "${GRID_FILE}" ]]; then
    echo "ERROR: Grid file not found: ${GRID_FILE}"
    exit 1
fi

# Check if stations exist
for station in Tokyo Chiba; do
    station_dir="${GWO_DIR}/${station}"
    if [[ ! -d "${station_dir}" ]]; then
        echo "WARNING: Station directory not found: ${station_dir}"
    fi
done

# Run the generator
xfvcom-make-met-nc "${GRID_FILE}" \
    --gwo-dir "${GWO_DIR}" \
    --station-map "${STATION_MAP}" \
    --start "${YEAR}" \
    --wind-factor "${WIND_FACTOR}" \
    --max-gap-hours "${MAX_GAP}" \
    --utm-zone ${UTM_ZONE} \
    --output "${OUTPUT_FILE}"

echo "========================================"
echo "Done: ${OUTPUT_FILE}"
echo "========================================"

# Show output file info
if command -v ncdump &> /dev/null; then
    echo ""
    echo "Output file dimensions and variables:"
    ncdump -h "${OUTPUT_FILE}" | grep -E "^\s+(nele|node|time|[a-z_]+\()" | head -20
fi

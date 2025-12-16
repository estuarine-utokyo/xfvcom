#!/bin/bash
# Generate FVCOM meteorological forcing from GWO-AMD data
#
# Usage: ./make_fvcom_met.sh [YEAR]
# Example: ./make_fvcom_met.sh 2020
#
# Edit default values below or override with environment variables.

set -e

# ============================================================
# Default values (edit as needed or override with env variables)
# ============================================================
YEAR="${1:-2020}"
GRID="${GRID:-${HOME}/Github/TB-FVCOM/goto2023/input/TokyoBay18_grd.dat}"
UTM_ZONE="${UTM_ZONE:-54}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/Github/TB-FVCOM/goto2023/input/${YEAR}}"
OUTPUT="${OUTPUT_DIR}/tb18_wnd.nc"

# GWO-AMD options
GWO_DIR="${GWO_DIR:-${HOME}/../share/Data/met/JMA_DataBase/GWO/Hourly}"
STATION_MAP="${STATION_MAP:-slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba}"
WIND_FACTOR="${WIND_FACTOR:-1.8}"
MAX_GAP_HOURS="${MAX_GAP_HOURS:-6}"

# Gap filling options
FILL_GAPS="${FILL_GAPS:-true}"
FALLBACK_STATIONS="${FALLBACK_STATIONS:-Chiba:Tokyo,Yokohama,Tateyama}"
SOLAR_MODEL="${SOLAR_MODEL:-empirical}"
# ============================================================

# Create output directory if needed
mkdir -p "${OUTPUT_DIR}"

echo "========================================"
echo "Generating FVCOM meteorological forcing"
echo "========================================"
echo "  Year:             ${YEAR}"
echo "  Grid:             ${GRID}"
echo "  UTM zone:         ${UTM_ZONE}"
echo "  Output:           ${OUTPUT}"
echo "  GWO dir:          ${GWO_DIR}"
echo "  Station map:      ${STATION_MAP}"
echo "  Wind factor:      ${WIND_FACTOR}"
echo "  Max gap hours:    ${MAX_GAP_HOURS}"
echo "  Fill gaps:        ${FILL_GAPS}"
echo "  Fallback stations:${FALLBACK_STATIONS}"
echo "  Solar model:      ${SOLAR_MODEL}"
echo "========================================"

# Verify input files exist
if [[ ! -d "${GWO_DIR}" ]]; then
    echo "ERROR: GWO data directory not found: ${GWO_DIR}"
    exit 1
fi

if [[ ! -f "${GRID}" ]]; then
    echo "ERROR: Grid file not found: ${GRID}"
    exit 1
fi

# Run the generator
xfvcom-make-met-nc "${GRID}" --start "${YEAR}" --utm-zone "${UTM_ZONE}" \
    --gwo-dir "${GWO_DIR}" --station-map "${STATION_MAP}" \
    --wind-factor "${WIND_FACTOR}" --max-gap-hours "${MAX_GAP_HOURS}" \
    $($FILL_GAPS && echo "--fill-gaps") \
    --fallback-stations "${FALLBACK_STATIONS}" \
    --solar-model "${SOLAR_MODEL}" \
    -o "${OUTPUT}"

echo "========================================"
echo "Done: ${OUTPUT}"
echo "========================================"

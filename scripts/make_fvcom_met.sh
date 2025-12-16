#!/bin/bash
# Generate FVCOM meteorological forcing from GWO-AMD data
#
# Usage: Edit the default values below, then run:
#   ./make_fvcom_met.sh
#
# Or override with environment variables:
#   START=2021 END=2022 ./make_fvcom_met.sh

set -e

# ============================================================
# Default values (edit as needed or override with env variables)
# ============================================================
# Grid and time
GRID=${GRID:-grid.dat}
START=${START:-2020}
END=${END:-}
UTM_ZONE=${UTM_ZONE:-54}
OUTPUT=${OUTPUT:-met_forcing.nc}

# GWO-AMD options
GWO_DIR=${GWO_DIR:-/path/to/GWO/Hourly}
STATION_MAP=${STATION_MAP:-"slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba"}
WIND_FACTOR=${WIND_FACTOR:-1.8}
MAX_GAP_HOURS=${MAX_GAP_HOURS:-6}

# Gap filling options
FILL_GAPS=${FILL_GAPS:-true}
FALLBACK_STATIONS=${FALLBACK_STATIONS:-"Chiba:Tokyo,Yokohama,Tateyama"}
SOLAR_MODEL=${SOLAR_MODEL:-empirical}
# ============================================================

echo "========================================"
echo "Generating FVCOM meteorological forcing"
echo "========================================"
echo "  Grid:             ${GRID}"
echo "  Start:            ${START}"
echo "  End:              ${END:-auto}"
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
xfvcom-make-met-nc "${GRID}" \
    --start "${START}" ${END:+--end "${END}"} \
    --utm-zone "${UTM_ZONE}" \
    --gwo-dir "${GWO_DIR}" \
    --station-map "${STATION_MAP}" \
    --wind-factor "${WIND_FACTOR}" \
    --max-gap-hours "${MAX_GAP_HOURS}" \
    $($FILL_GAPS && echo "--fill-gaps") \
    --fallback-stations "${FALLBACK_STATIONS}" \
    --solar-model "${SOLAR_MODEL}" \
    -o "${OUTPUT}"

echo "========================================"
echo "Done: ${OUTPUT}"
echo "========================================"

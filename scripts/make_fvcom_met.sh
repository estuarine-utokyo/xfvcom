#!/bin/bash
# Save environment variables before defaults overwrite them
_E_GRID="$GRID" _E_START="$START" _E_END="$END" _E_UTM_ZONE="$UTM_ZONE"
_E_OUTPUT="$OUTPUT" _E_GWO_DIR="$GWO_DIR" _E_STATION_MAP="$STATION_MAP"
_E_WIND_FACTOR="$WIND_FACTOR" _E_MAX_GAP_HOURS="$MAX_GAP_HOURS"
_E_FILL_GAPS="$FILL_GAPS" _E_FALLBACK_STATIONS="$FALLBACK_STATIONS"
_E_SOLAR_MODEL="$SOLAR_MODEL"

# ============================================================
# Default values (edit as needed)
# ============================================================
# Grid and time
GRID=~/Github/TB-FVCOM/goto2023/input/TokyoBay18_grd.dat
START=2020
END=
UTM_ZONE=54
OUTPUT=met_forcing.nc

# GWO-AMD options
GWO_DIR=${DATA_DIR}/met/JMA_DataBase/GWO/Hourly
STATION_MAP="slht:Tokyo,kous:Tokyo,clod:Tokyo,*:Chiba"
WIND_FACTOR=1.8
MAX_GAP_HOURS=6

# Gap filling options
FILL_GAPS=true
FALLBACK_STATIONS="Chiba:Tokyo,Yokohama,Tateyama"
SOLAR_MODEL=empirical
# ============================================================
# End of user configuration
# ============================================================

set -e

# Apply environment variable overrides
[ -n "$_E_GRID" ] && GRID="$_E_GRID"
[ -n "$_E_START" ] && START="$_E_START"
[ -n "$_E_END" ] && END="$_E_END"
[ -n "$_E_UTM_ZONE" ] && UTM_ZONE="$_E_UTM_ZONE"
[ -n "$_E_OUTPUT" ] && OUTPUT="$_E_OUTPUT"
[ -n "$_E_GWO_DIR" ] && GWO_DIR="$_E_GWO_DIR"
[ -n "$_E_STATION_MAP" ] && STATION_MAP="$_E_STATION_MAP"
[ -n "$_E_WIND_FACTOR" ] && WIND_FACTOR="$_E_WIND_FACTOR"
[ -n "$_E_MAX_GAP_HOURS" ] && MAX_GAP_HOURS="$_E_MAX_GAP_HOURS"
[ -n "$_E_FILL_GAPS" ] && FILL_GAPS="$_E_FILL_GAPS"
[ -n "$_E_FALLBACK_STATIONS" ] && FALLBACK_STATIONS="$_E_FALLBACK_STATIONS"
[ -n "$_E_SOLAR_MODEL" ] && SOLAR_MODEL="$_E_SOLAR_MODEL"

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

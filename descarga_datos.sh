#!/usr/bin/env bash
# =============================================================================
# descarga_datos.sh — Descarga catálogo sísmico USGS a archivos CSV locales
#
# Uso:
#   ./descarga_datos.sh              # descarga todo (histórico + reciente)
#   ./descarga_datos.sh --historico  # solo catálogo 1900-hoy M≥6.0
#   ./descarga_datos.sh --reciente   # solo últimos 90 días M≥2.5
#
# Los archivos quedan en ./data/ y main.py los carga automáticamente.
# Volver a ejecutar actualiza solo el reciente (el histórico cambia poco).
# =============================================================================

set -euo pipefail

BASE_URL="https://earthquake.usgs.gov/fdsnws/event/1/query"
DATA_DIR="$(dirname "$0")/data"
mkdir -p "$DATA_DIR"

MODE="${1:-}"
HISTORICO=true
RECIENTE=true
[[ "$MODE" == "--historico" ]] && RECIENTE=false
[[ "$MODE" == "--reciente"  ]] && HISTORICO=false

SEP="================================================================="

# ── CATÁLOGO HISTÓRICO M≥6.0 (1900 → hoy) ────────────────────────
if $HISTORICO; then
    echo ""
    echo "$SEP"
    echo "  CATÁLOGO HISTÓRICO  M≥6.0  |  1900 → hoy"
    echo "$SEP"

    HIST_FILE="$DATA_DIR/historico_M60_1900.csv"
    TMP_FILE="$DATA_DIR/_tmp_hist.csv"
    HEADER_WRITTEN=false

    for YEAR in $(seq 1900 10 2030); do
        END_YEAR=$((YEAR + 10))
        # No descargar décadas futuras
        CURRENT_YEAR=$(date +%Y)
        [[ $YEAR -gt $CURRENT_YEAR ]] && break

        printf "  Descargando %d-%d ... " "$YEAR" "$((END_YEAR - 1))"

        HTTP_CODE=$(curl -s -w "%{http_code}" \
            "${BASE_URL}?format=csv&starttime=${YEAR}-01-01&endtime=${END_YEAR}-01-01&minmagnitude=6.0&orderby=time-asc&limit=20000" \
            -o "$TMP_FILE" \
            --connect-timeout 30 \
            --max-time 90)

        if [[ "$HTTP_CODE" != "200" ]]; then
            echo "ERROR HTTP $HTTP_CODE — omitiendo"
            continue
        fi

        LINES=$(wc -l < "$TMP_FILE")
        EVENTS=$((LINES - 1))
        echo "$EVENTS eventos"

        if ! $HEADER_WRITTEN; then
            # Primera decade: copiar con cabecera
            cat "$TMP_FILE" >> "$HIST_FILE.new"
            HEADER_WRITTEN=true
        else
            # Décadas siguientes: omitir línea de cabecera (tail -n +2)
            tail -n +2 "$TMP_FILE" >> "$HIST_FILE.new"
        fi

        sleep 0.4  # Respetar rate limit USGS (≤60 req/min)
    done

    rm -f "$TMP_FILE"

    if [[ -f "$HIST_FILE.new" ]]; then
        mv "$HIST_FILE.new" "$HIST_FILE"
        TOTAL=$(( $(wc -l < "$HIST_FILE") - 1 ))
        echo ""
        echo "  ✓ Guardado: $HIST_FILE"
        echo "  ✓ Total eventos históricos M≥6.0: $TOTAL"
    fi
fi

# ── CATÁLOGO RECIENTE M≥2.5 (últimos 90 días) ─────────────────────
if $RECIENTE; then
    echo ""
    echo "$SEP"
    echo "  CATÁLOGO RECIENTE  M≥2.5  |  últimos 90 días"
    echo "$SEP"

    # Calcular fecha de hace 90 días (compatible macOS y Linux)
    START_DATE=$(python3 -c "
from datetime import datetime, timedelta, timezone
print((datetime.now(timezone.utc) - timedelta(days=90)).strftime('%Y-%m-%d'))
")

    RECENT_FILE="$DATA_DIR/reciente_M25_90d.csv"
    echo "  Período: $START_DATE → hoy"
    echo "  Descargando..."

    HTTP_CODE=$(curl -s -w "%{http_code}" \
        "${BASE_URL}?format=csv&starttime=${START_DATE}&minmagnitude=2.5&orderby=time-asc&limit=20000" \
        -o "$RECENT_FILE" \
        --connect-timeout 30 \
        --max-time 120)

    if [[ "$HTTP_CODE" == "200" ]]; then
        TOTAL=$(( $(wc -l < "$RECENT_FILE") - 1 ))
        echo "  ✓ Guardado: $RECENT_FILE"
        echo "  ✓ Total eventos recientes M≥2.5: $TOTAL"
    else
        echo "  ERROR HTTP $HTTP_CODE"
        rm -f "$RECENT_FILE"
    fi
fi

# ── RESUMEN FINAL ─────────────────────────────────────────────────
echo ""
echo "$SEP"
echo "  ARCHIVOS DISPONIBLES EN ./data/"
echo "$SEP"
ls -lh "$DATA_DIR/"*.csv 2>/dev/null | awk '{print "  " $NF "  (" $5 ")"}'
echo ""
echo "  Ejecutar ahora:  python3.10 main.py"
echo "$SEP"

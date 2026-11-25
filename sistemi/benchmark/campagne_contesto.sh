#!/bin/sh
# Cinque campagne dello studio sul contesto, ciascuna avviata a sistema scarico
# (media di carico a 1 minuto sotto SOGLIA_CARICO, poi PAUSA secondi). Uso: PYTHON=... sh campagne_mlx.sh <uscita> [n]
set -eu
cd "$(dirname "$0")/../.."
OUT=$1; N=${2:-5}; SOGLIA=${SOGLIA_CARICO:-1.5}
P=${PYTHON:-python3}; P=$(cd "$(dirname "$P")" && pwd)/$(basename "$P")
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
carico() { sysctl -n vm.loadavg | awk '{print $2}'; }
for c in $(seq 1 "$N"); do
  while [ "$(echo "$(carico) >= $SOGLIA" | bc)" -eq 1 ]; do sleep 20; done
  sleep "${PAUSA:-60}"
  echo "campagna $c pesi ${STERA_PESI:-H122000} inizio $(date +%FT%T) carico $(sysctl -n vm.loadavg) alimentazione $(pmset -g batt | sed -n 1p | cut -d"'" -f2)" >> "$OUT/condizioni.log"
  PYTHON=$P sh sistemi/benchmark/TinyStories/esegui_contesto_mlx.sh "$OUT/c${c}"
  echo "campagna $c fine $(date +%FT%T)" >> "$OUT/condizioni.log"
done

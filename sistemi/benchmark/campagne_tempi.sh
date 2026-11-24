#!/bin/sh
# Campagne ripetute delle misure di tempo del Capitolo 8 (Mac con Apple Silicon).
# Ogni campagna esegue il confronto appaiato compatto/denso (6 input) e il confronto dei tre modelli (3 input).
# Ogni campagna attende che la media di carico a 1 minuto scenda sotto SOGLIA_CARICO (predefinito 1,5), poi PAUSA secondi (predefinito 60).
# Uso: PYTHON=.venv-sistemi/bin/python sh sistemi/benchmark/campagne_tempi.sh <cartella_uscita> [numero_campagne]
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
  d=$(PYTHON=$P sh sistemi/benchmark/confronto_appaiato.sh 2>/dev/null | tail -1)
  mv "sistemi/$(dirname "$d")" "$OUT/c${c}_appaiato"
  PYTHON_MLX=$P PYTHON_TORCH=$P sh sistemi/benchmark/TinyStories/esegui_confronto_tesi.sh "$OUT/c${c}_tre_modelli"
  echo "campagna $c fine $(date +%FT%T)" >> "$OUT/condizioni.log"
done

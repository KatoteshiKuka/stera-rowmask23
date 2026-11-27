#!/bin/sh
# Five campaigns of the prefill study, each started on an idle system
# (one-minute load average below LOAD_THRESHOLD, then PAUSE seconds). Usage: PYTHON=... sh campaigns_prefill.sh <output_dir> [n]
set -eu
cd "$(dirname "$0")/../.."
OUT=$1; N=${2:-5}; THRESHOLD=${LOAD_THRESHOLD:-${SOGLIA_CARICO:-1.5}}
P=${PYTHON:-python3}; P=$(cd "$(dirname "$P")" && pwd)/$(basename "$P")
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
load_avg() { sysctl -n vm.loadavg | awk '{print $2}'; }
for c in $(seq 1 "$N"); do
  while [ "$(echo "$(load_avg) >= $THRESHOLD" | bc)" -eq 1 ]; do sleep 20; done
  sleep "${PAUSE:-${PAUSA:-60}}"
  echo "campagna $c pesi ${STERA_WEIGHTS:-${STERA_PESI:-H122000}} inizio $(date +%FT%T) carico $(sysctl -n vm.loadavg) alimentazione $(pmset -g batt | sed -n 1p | cut -d"'" -f2)" >> "$OUT/condizioni.log"
  PYTHON=$P sh systems/benchmark/TinyStories/run_prefill.sh "$OUT/c${c}"
  echo "campagna $c fine $(date +%FT%T)" >> "$OUT/condizioni.log"
done

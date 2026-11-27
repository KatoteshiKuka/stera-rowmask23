#!/bin/sh
# Repeated campaigns of the timing measurements of the thesis (Mac with Apple Silicon).
# Every campaign runs the paired compact/dense comparison (6 inputs) and the comparison of the three models (3 inputs).
# Every campaign waits until the one-minute load average drops below LOAD_THRESHOLD (default 1.5), then PAUSE seconds (default 60).
# Usage: PYTHON=.venv-systems/bin/python sh systems/benchmark/campaigns_timing.sh <output_dir> [number_of_campaigns]
# The log lines written to condizioni.log keep the Italian wording of the archived logs.
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
  d=$(PYTHON=$P sh systems/benchmark/paired_comparison.sh 2>/dev/null | tail -1)
  mv "systems/$(dirname "$d")" "$OUT/c${c}_appaiato"
  PYTHON_MLX=$P PYTHON_TORCH=$P sh systems/benchmark/TinyStories/run_thesis_comparison.sh "$OUT/c${c}_tre_modelli"
  echo "campagna $c fine $(date +%FT%T)" >> "$OUT/condizioni.log"
done

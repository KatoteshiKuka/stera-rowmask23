#!/bin/bash
# Campagne di decode del modello fittizio F10 (generato da sistemi/f10/genera_f10.py dal checkpoint finale H122000)
# al variare del contesto. Ogni voce di CONTESTI è "token:precisione_kv".
# Ogni esecuzione parte solo con media di carico a un minuto sotto SOGLIA_CARICO, dopo PAUSA secondi.
set -u
cd "$(dirname "$0")"
OUT=${OUT:-../risultati/campagne_20261127/f10_contesto}; mkdir -p "$OUT"
PY=${PY:-../../.venv-sistemi/bin/python}; SOGLIA=${SOGLIA_CARICO:-1.5}; PAUSA=${PAUSA:-30}
CONTESTI=${CONTESTI:-"128:fp32 512:fp32 2000:fp32 8000:fp32"}; RIP=${RIP:-3}
for r in $(seq 1 $RIP); do for voce in $CONTESTI; do
  L=${voce%%:*}; KV=${voce##*:}
  sleep $PAUSA
  while :; do c=$(sysctl -n vm.loadavg | awk '{print $2}'); awk -v c=$c -v s=$SOGLIA 'BEGIN{exit !(c<s)}' && break; sleep 10; done
  echo "$(date +%T) contesto $L kv $KV ripetizione $r carico $c swap $(sysctl -n vm.swapusage | awk '{print $6}')" | tee -a "$OUT/condizioni.txt"
  STERA_PESI=H122000 $PY f10_contesto.py --contesto $L --kv $KV --out "$OUT/f10_L${L}_${KV}_r${r}.json" > "$OUT/f10_L${L}_${KV}_r${r}.log" 2>&1
done; done
echo FINE | tee -a "$OUT/condizioni.txt"

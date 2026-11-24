#!/bin/bash
# Addestramento di STERA 30M da zero fino a 122.000 aggiornamenti (1.529.514.422 token bersaglio).
# Pensato per Slurm con 2 GPU (es. NVIDIA P100 16 GB); funziona anche come semplice script bash.
# La corsa riprende automaticamente dall'ultimo checkpoint salvato nella cartella di uscita:
# nella tesi e stata eseguita in tre job consecutivi dello stesso comando (0-80k, 80k-100k, 100k-122k).
#SBATCH --job-name=stera30m
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
set -euo pipefail
: "${STERA_DATI:?impostare STERA_DATI alla cartella prodotta da prepara_dati.py}"
: "${STERA_RUN:?impostare STERA_RUN alla cartella dei checkpoint}"
TOTALE=${TOTALE:-122000}
# percorsi assoluti, perche lo script cambia cartella
mkdir -p "$STERA_RUN"
STERA_DATI=$(cd "$STERA_DATI" && pwd); STERA_RUN=$(cd "$STERA_RUN" && pwd)
# sotto sbatch $0 punta a una copia dello script: in quel caso si risale al file originale
QUI=$(dirname "$0")
if [ ! -f "$QUI/ts68_v3.py" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  QUI=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')")
fi
cd "$QUI"
ULTIMO=$(ls -1 "$STERA_RUN"/TS68V2_STEP*.ckpt.gz 2>/dev/null | sort -V | tail -1 || true)
if [ -z "$ULTIMO" ]; then
  INIZIO="--scratch"; PASSI=$TOTALE
else
  S=$(basename "$ULTIMO" | sed -E 's/TS68V2_STEP([0-9]+).ckpt.gz/\1/')
  INIZIO="--ckpt $ULTIMO --sha256 $(shasum -a 256 "$ULTIMO" | cut -d' ' -f1)"; PASSI=$((TOTALE - S))
fi
[ "$PASSI" -gt 0 ] || { echo "addestramento gia completato"; exit 0; }
# Profilo del tasso di apprendimento: warmup 300 passi fino a 6e-4, costante fino a 64.000,
# discesa lineare fino a 6e-5 al passo 80.000, poi costante a 6e-5.
torchrun --standalone --nproc_per_node=2 ts68_v3.py --data "$STERA_DATI" --output "$STERA_RUN" $INIZIO \
  --layers 6 --gains --rank 96 --steps "$PASSI" --batch 64 --micro 8 --ctx 512 \
  --lr 6e-4 --warmup 300 --budget-steps 80000 --decay-frac 0.2 --sched-offset 0 \
  --wd 0.1 --beta2 0.95 --save-steps 2000 --nval 512

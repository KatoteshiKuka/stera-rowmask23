#!/bin/bash
# Train STERA 30M from scratch up to 122,000 updates (1,529,514,422 target tokens).
# Written for Slurm with 2 GPUs (e.g. NVIDIA P100 16 GB); it also works as a plain bash script.
# The run resumes automatically from the last checkpoint in the output folder:
# for the thesis it ran as three consecutive jobs of the same command (0-80k, 80k-100k, 100k-122k).
#SBATCH --job-name=stera30m
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
set -euo pipefail
: "${STERA_DATA:=${STERA_DATI:-}}"
: "${STERA_DATA:?set STERA_DATA to the folder produced by prepare_data.py}"
: "${STERA_RUN:?set STERA_RUN to the checkpoint folder}"
TOTAL=${TOTAL:-${TOTALE:-122000}}
# absolute paths, because the script changes directory
mkdir -p "$STERA_RUN"
STERA_DATA=$(cd "$STERA_DATA" && pwd); STERA_RUN=$(cd "$STERA_RUN" && pwd)
# under sbatch $0 points to a copy of the script: in that case recover the original file
HERE=$(dirname "$0")
if [ ! -f "$HERE/ts68_v3.py" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  HERE=$(dirname "$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')")
fi
cd "$HERE"
LAST=$(ls -1 "$STERA_RUN"/TS68V2_STEP*.ckpt.gz 2>/dev/null | sort -V | tail -1 || true)
if [ -z "$LAST" ]; then
  START="--scratch"; STEPS=$TOTAL
else
  S=$(basename "$LAST" | sed -E 's/TS68V2_STEP([0-9]+).ckpt.gz/\1/')
  START="--ckpt $LAST --sha256 $(shasum -a 256 "$LAST" | cut -d' ' -f1)"; STEPS=$((TOTAL - S))
fi
[ "$STEPS" -gt 0 ] || { echo "training already complete"; exit 0; }
# Learning-rate schedule: 300 warmup steps up to 6e-4, constant until 64,000,
# linear decay to 6e-5 at step 80,000, then constant at 6e-5.
torchrun --standalone --nproc_per_node=2 ts68_v3.py --data "$STERA_DATA" --output "$STERA_RUN" $START \
  --layers 6 --gains --rank 96 --steps "$STEPS" --batch 64 --micro 8 --ctx 512 \
  --lr 6e-4 --warmup 300 --budget-steps 80000 --decay-frac 0.2 --sched-offset 0 \
  --wd 0.1 --beta2 0.95 --save-steps 2000 --nval 512

#!/bin/sh
# Timing of the compact engine with the low-rank term in FP32 and at 8 bits (engines/stera_rank8_mlx.py): same three
# 256-token inputs, 100 generated tokens, prefill block 64, alternating order.
set -eu
cd "$(dirname "$0")/../.."
out="$1"; mkdir -p "$out"
for offset in 5 35 65; do
    case "$offset" in 35) order="STERA-R8 STERA" ;; *) order="STERA STERA-R8" ;; esac
    for model in $order; do
        base="$out/${model}_${offset}"
        PROMPT_OFFSET="$offset" STERA_PREFILL_BLOCK=64 /usr/bin/time -l "${PYTHON:-python3}" benchmark/TinyStories/measure.py "$model" "$offset" > "$base.log" 2> "$base.stderr"
        tail -1 "$base.log" >> "$out/misure.jsonl"
    done
done

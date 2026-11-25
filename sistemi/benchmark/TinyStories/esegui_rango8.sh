#!/bin/sh
# Tempi del motore compatto con basso rango FP32 e a 8 bit (motori/stera_rango8_mlx.py): stessi tre input da 256 token,
# 100 token generati, blocco di prefill 64, ordine alternato.
set -eu
cd "$(dirname "$0")/../.."
out="$1"; mkdir -p "$out"
for offset in 5 35 65; do
    case "$offset" in 35) order="STERA-R8 STERA" ;; *) order="STERA STERA-R8" ;; esac
    for model in $order; do
        base="$out/${model}_${offset}"
        PROMPT_OFFSET="$offset" STERA_PREFILL_BLOCK=64 /usr/bin/time -l "${PYTHON:-python3}" benchmark/TinyStories/misura.py "$model" "$offset" > "$base.log" 2> "$base.stderr"
        tail -1 "$base.log" >> "$out/misure.jsonl"
    done
done

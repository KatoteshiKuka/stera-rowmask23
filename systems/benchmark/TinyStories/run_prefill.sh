#!/bin/sh
# Prefill of STERA 30M: compact kernel (block 64), on-the-fly expansion layer by layer (FP32 and with the 8-bit
# low-rank term) and dense engine, with inputs from 32 to 384 tokens and 100 generated tokens; rotated order.
set -eu
cd "$(dirname "$0")/../.."
out="$1"; mkdir -p "$out"
for len in 32 64 128 256 384; do
  bc=$(( len < 64 ? len : 64 )); bd=$(( len < 256 ? len : 256 ))
  for offset in 5 35 65; do
    case "$offset" in
        5) order="STERA STERA-ESP STERA-R8-ESP STERA-DENSO-MLX" ;;
        35) order="STERA-ESP STERA-R8-ESP STERA-DENSO-MLX STERA" ;;
        65) order="STERA-R8-ESP STERA-DENSO-MLX STERA STERA-ESP" ;;
    esac
    for model in $order; do
        base="$out/${model}_L${len}_${offset}"
        PROMPT_LEN="$len" PROMPT_OFFSET="$offset" STERA_PREFILL_BLOCK="$bc" MLX_PREFILL_BLOCK="$bd" \
            /usr/bin/time -l "${PYTHON:-python3}" benchmark/TinyStories/measure.py "$model" "$offset" > "$base.log" 2> "$base.stderr"
        tail -1 "$base.log" >> "$out/misure.jsonl"
    done
  done
done

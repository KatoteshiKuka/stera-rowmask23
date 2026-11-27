#!/bin/sh
# Study over the input length (32, 64, 128, 256, 384 tokens; always 100 generated tokens),
# same four models and same MLX engine as the like-for-like comparison. Prefill block: 64 for the compact engine
# (the best measured), up to 256 for the dense one.
set -eu
cd "$(dirname "$0")/../.."
out="$1"
mkdir -p "$out"
for len in 32 64 128 256 384; do
  bc=$(( len < 64 ? len : 64 )); bd=$(( len < 256 ? len : 256 ))
  for offset in 5 35 65; do
    case "$offset" in
        5) order="STERA 8M-MLX 33M-MLX STERA-DENSO-MLX" ;;
        35) order="8M-MLX 33M-MLX STERA-DENSO-MLX STERA" ;;
        65) order="33M-MLX STERA-DENSO-MLX STERA 8M-MLX" ;;
    esac
    for model in $order; do
        base="$out/${model}_L${len}_${offset}"
        PROMPT_LEN="$len" PROMPT_OFFSET="$offset" STERA_PREFILL_BLOCK="$bc" MLX_PREFILL_BLOCK="$bd" \
            /usr/bin/time -l "${PYTHON:-python3}" benchmark/TinyStories/measure.py "$model" "$offset" \
            > "$base.log" 2> "$base.stderr"
        tail -1 "$base.log" >> "$out/misure.jsonl"
    done
  done
done

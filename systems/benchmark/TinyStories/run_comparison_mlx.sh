#!/bin/sh
# Like-for-like comparison on the same MLX engine: compact STERA 30M, dense STERA 30M and the two TinyStories models
# run with the dense MLX engine (engines/gptneo_dense_mlx.py). Same three inputs and same rotated order
# as the comparison in the thesis; every process runs one cold and one warm request.
set -eu
cd "$(dirname "$0")/../.."
out="$1"
mkdir -p "$out"
for offset in 5 35 65; do
    case "$offset" in
        5) order="STERA 8M-MLX 33M-MLX STERA-DENSO-MLX" ;;
        35) order="8M-MLX 33M-MLX STERA-DENSO-MLX STERA" ;;
        65) order="33M-MLX STERA-DENSO-MLX STERA 8M-MLX" ;;
    esac
    for model in $order; do
        base="$out/${model}_${offset}"
        PROMPT_OFFSET="$offset" STERA_PREFILL_BLOCK=16 MLX_PREFILL_BLOCK=256 \
            /usr/bin/time -l "${PYTHON:-python3}" benchmark/TinyStories/measure.py "$model" "$offset" \
            > "$base.log" 2> "$base.stderr"
        tail -1 "$base.log" >> "$out/misure.jsonl"
    done
done

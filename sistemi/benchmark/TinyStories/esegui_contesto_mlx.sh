#!/bin/sh
# Studio al variare della lunghezza dell'input (32, 64, 128, 256, 384 token; sempre 100 token generati),
# stessi quattro modelli e stesso motore MLX del confronto alla pari. Blocco di prefill: 64 per il compatto
# (il migliore misurato), fino a 256 per il denso.
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
            /usr/bin/time -l "${PYTHON:-python3}" benchmark/TinyStories/misura.py "$model" "$offset" \
            > "$base.log" 2> "$base.stderr"
        tail -1 "$base.log" >> "$out/misure.jsonl"
    done
  done
done

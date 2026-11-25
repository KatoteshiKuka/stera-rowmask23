#!/bin/sh
# Confronto alla pari sullo stesso motore MLX: STERA 30M compatto, STERA 30M denso e i due TinyStories
# eseguiti con il motore denso MLX (motori/gptneo_denso_mlx.py). Stessi tre input e stesso ordine ruotato
# del confronto della tesi; ogni processo esegue una richiesta a freddo e una a caldo.
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
            /usr/bin/time -l "${PYTHON:-python3}" benchmark/TinyStories/misura.py "$model" "$offset" \
            > "$base.log" 2> "$base.stderr"
        tail -1 "$base.log" >> "$out/misure.jsonl"
    done
done

#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
out="$1"
mkdir "$out"
for offset in 5 35 65; do
    case "$offset" in
        5) order="STERA 8M 33M" ;;
        35) order="8M 33M STERA" ;;
        65) order="33M STERA 8M" ;;
    esac
    for model in $order; do
        case "$model" in
            STERA) runner=${PYTHON_MLX:-python3} ;;
            *) runner=${PYTHON_TORCH:-python3} ;;
        esac
        base="$out/${model}_${offset}"
        PROMPT_OFFSET="$offset" STERA_PREFILL_BLOCK=16 GPTNEO_SDPA=1 GPTNEO_LAST_LOGIT=1 \
            /usr/bin/time -l "$runner" benchmark/TinyStories/misura.py "$model" "$offset" \
            > "$base.log" 2> "$base.stderr"
        tail -1 "$base.log" >> "$out/ottimizzati.jsonl"
    done
done
for offset in 5 35 65; do
    for model in 8M 33M; do
        base="$out/${model}_${offset}_eager"
        PROMPT_OFFSET="$offset" GPTNEO_SDPA=0 GPTNEO_LAST_LOGIT=0 \
            ${PYTHON_TORCH:-python3} benchmark/TinyStories/misura.py "$model" "$offset" \
            > "$base.log" 2> "$base.stderr"
        tail -1 "$base.log" >> "$out/eager_parita.jsonl"
    done
done

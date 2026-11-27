#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p runs
target="$(mktemp -d runs/confronto.XXXXXX)"
for offset in 10 25 40 55 70 85; do
    case "$offset" in
        10|40|70) order="dense stera";;
        *) order="stera dense";;
    esac
    for model in $order; do
        block=128
        [ "$model" != dense ] || block=256
        ${PYTHON:-python3} benchmark/measure_dense_stera.py "$model" "$block" "$offset" > "$target/${model}_${offset}.log"
    done
done
${PYTHON:-python3} - "$target" <<'PY'
import json, statistics, sys
from pathlib import Path
p = Path(sys.argv[1])
pairs = []
for offset in (10,25,40,55,70,85):
    dense, stera = [json.loads((p/f"{model}_{offset}.log").read_text().splitlines()[-1])
                    for model in ("dense","stera")]
    assert dense["prompt_sha256"] == stera["prompt_sha256"]
    for phase in ("first", "warm"):
        assert len(dense[phase]["outputs"]) == 101
        assert dense[phase]["outputs"] == stera[phase]["outputs"]
    pairs.append({"dense":dense,"stera":stera})
results = {model:{phase:{key:statistics.median(r[model][phase][key] for r in pairs)
    for key in ("ttft_s","decode_s","decode_tps","total_s")} for phase in ("first","warm")}
    for model in ("dense","stera")}
results["paired_decode_time_change_pct"] = 100*(statistics.median(
    r["stera"]["warm"]["decode_s"]/r["dense"]["warm"]["decode_s"] for r in pairs)-1)
(p/"riepilogo.json").write_text(json.dumps(results,indent=2)+"\n")
print(p/"riepilogo.json")
PY

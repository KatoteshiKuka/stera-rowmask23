"""Summary of the timing campaigns of the thesis.

Reads the folder produced by campaigns_timing.sh and writes riepilogo.json ("summary") into the same folder.
It first checks that, in every campaign, inputs and generated tokens match:
  - compact and dense engine: the same 101 greedy IDs on every input;
  - TinyStories: the same IDs as the eager implementation;
  - across campaigns: the same IDs for every model and input.
Then it computes medians and ranges over all warm measurements. Output keys are kept as in the archived file.

Usage: python summarize_campaigns.py systems/results/campagne_20261124/protocollo
"""
import json, re, statistics as st, sys
from pathlib import Path

PAIRED_INPUTS = (10, 25, 40, 55, 70, 85)
THREE_MODEL_INPUTS = (5, 35, 65)
MODELS = ("8M", "STERA", "33M")


def last_line(p):
    return json.loads(Path(p).read_text().splitlines()[-1])


def stats(v):
    return {"mediana": st.median(v), "min": min(v), "max": max(v), "n": len(v)}


def paired(folders):
    pairs, outputs = [], {}
    for c, d in enumerate(folders, 1):
        for o in PAIRED_INPUTS:
            den, com = last_line(d / f"dense_{o}.log"), last_line(d / f"stera_{o}.log")
            assert den["prompt_sha256"] == com["prompt_sha256"]
            for phase in ("first", "warm"):
                assert len(den[phase]["outputs"]) == 101 and den[phase]["outputs"] == com[phase]["outputs"]
            assert outputs.setdefault(o, den["warm"]["outputs"]) == den["warm"]["outputs"]
            pairs.append({"campagna": c, "input": o, "denso": den["warm"], "compatto": com["warm"],
                           "variazione_decode_pct": 100 * (com["warm"]["decode_s"] / den["warm"]["decode_s"] - 1)})
    r = {"campagne": len(folders), "coppie": len(pairs), "token_identici_compatto_denso": True,
         "token_identici_fra_campagne": True}
    for m, k in (("denso", "denso"), ("compatto", "compatto")):
        r[m] = {"primo_token_ms": stats([1000 * x[k]["ttft_s"] for x in pairs]),
                "decode_token_s": stats([x[k]["decode_tps"] for x in pairs]),
                "totale_s": stats([x[k]["total_s"] for x in pairs])}
    var = [x["variazione_decode_pct"] for x in pairs]
    r["variazione_decode_pct"] = stats(var)
    r["variazione_decode_pct_per_campagna"] = [st.median(x["variazione_decode_pct"] for x in pairs if x["campagna"] == c)
                                               for c in range(1, len(folders) + 1)]
    r["variazione_decode_pct_per_input"] = {o: st.median(x["variazione_decode_pct"] for x in pairs if x["input"] == o)
                                            for o in PAIRED_INPUTS}
    r["totale_s_per_input"] = {o: {m: st.median(x[m]["total_s"] for x in pairs if x["input"] == o)
                                   for m in ("denso", "compatto")} for o in PAIRED_INPUTS}
    r["coppie_compatto_piu_rapido_in_decode"] = sum(v < 0 for v in var)
    r["coppie_denso_primo_token_prima"] = sum(x["denso"]["ttft_s"] < x["compatto"]["ttft_s"] for x in pairs)
    r["coppie_compatto_totale_minore"] = sum(x["compatto"]["total_s"] < x["denso"]["total_s"] for x in pairs)
    return r


def three_models(folders):
    rows, outputs = {m: [] for m in MODELS}, {}
    for d in folders:
        opt = {(x["model"], x["rep"]): x for x in map(json.loads, (d / "ottimizzati.jsonl").read_text().splitlines())}
        eager = {(x["model"], x["rep"]): x for x in map(json.loads, (d / "eager_parita.jsonl").read_text().splitlines())}
        for o in THREE_MODEL_INPUTS:
            assert len({opt[m, o]["prompt_sha256"] for m in MODELS}) == 1
        for k, e in eager.items():
            assert opt[k]["outputs"] == e["outputs"] and len(e["outputs"]) == 101
        for (m, o), x in opt.items():
            assert outputs.setdefault((m, o), x["outputs"]) == x["outputs"]
            peak = int(re.search(r"(\d+)\s+peak memory footprint", (d / f"{m}_{o}.stderr").read_text())[1])
            rows[m].append({**x, "picco_processo": peak})
    return {m: {"misure": len(v), "motore": v[0]["framework"],
                "primo_token_ms": stats([1000 * x["warm"]["ttft_s"] for x in v]),
                "decode_token_s": stats([x["warm"]["decode_tps"] for x in v]),
                "totale_s": stats([x["warm"]["total_s"] for x in v]),
                "picco_richiesta_mib": stats([x["compiled_request_memory"]["peak_footprint_sampled_bytes"] / 2**20 for x in v]),
                "picco_processo_mib": stats([x["picco_processo"] / 2**20 for x in v]),
                "parita_eager": m != "STERA"} for m, v in rows.items()}


def main(base):
    base = Path(base)
    ap = sorted(base.glob("c*_appaiato"), key=lambda p: int(p.name[1:].split("_")[0]))
    tm = sorted(base.glob("c*_tre_modelli"), key=lambda p: int(p.name[1:].split("_")[0]))
    out = {"condizioni": (base / "condizioni.log").read_text().splitlines(),
           "confronto_appaiato": paired(ap), "tre_modelli": three_models(tm)}
    (base / "riepilogo.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(base / "riepilogo.json")


if __name__ == "__main__":
    main(sys.argv[1])

"""Riepilogo delle campagne di misura dei tempi (Capitolo 8).

Legge la cartella prodotta da campagne_tempi.sh e scrive riepilogo.json nella stessa cartella.
Controlla prima che in ogni campagna gli ingressi e i token generati coincidano:
  - motore compatto e motore denso: stessi 101 ID greedy su ciascun input;
  - TinyStories: stessi ID dell'implementazione eager;
  - tutte le campagne: stessi ID per ogni modello e input.
Poi calcola mediane e intervalli su tutte le misure a caldo.

Uso: python riepiloga_campagne.py sistemi/risultati/campagne_20261124/protocollo
"""
import json, re, statistics as st, sys
from pathlib import Path

INPUT_APPAIATI = (10, 25, 40, 55, 70, 85)
INPUT_TRE = (5, 35, 65)
MODELLI = ("8M", "STERA", "33M")


def ultima_riga(p):
    return json.loads(Path(p).read_text().splitlines()[-1])


def sintesi(v):
    return {"mediana": st.median(v), "min": min(v), "max": max(v), "n": len(v)}


def appaiato(cartelle):
    coppie, uscite = [], {}
    for c, d in enumerate(cartelle, 1):
        for o in INPUT_APPAIATI:
            den, com = ultima_riga(d / f"dense_{o}.log"), ultima_riga(d / f"stera_{o}.log")
            assert den["prompt_sha256"] == com["prompt_sha256"]
            for fase in ("first", "warm"):
                assert len(den[fase]["outputs"]) == 101 and den[fase]["outputs"] == com[fase]["outputs"]
            assert uscite.setdefault(o, den["warm"]["outputs"]) == den["warm"]["outputs"]
            coppie.append({"campagna": c, "input": o, "denso": den["warm"], "compatto": com["warm"],
                           "variazione_decode_pct": 100 * (com["warm"]["decode_s"] / den["warm"]["decode_s"] - 1)})
    r = {"campagne": len(cartelle), "coppie": len(coppie), "token_identici_compatto_denso": True,
         "token_identici_fra_campagne": True}
    for m, k in (("denso", "denso"), ("compatto", "compatto")):
        r[m] = {"primo_token_ms": sintesi([1000 * x[k]["ttft_s"] for x in coppie]),
                "decode_token_s": sintesi([x[k]["decode_tps"] for x in coppie]),
                "totale_s": sintesi([x[k]["total_s"] for x in coppie])}
    var = [x["variazione_decode_pct"] for x in coppie]
    r["variazione_decode_pct"] = sintesi(var)
    r["variazione_decode_pct_per_campagna"] = [st.median(x["variazione_decode_pct"] for x in coppie if x["campagna"] == c)
                                               for c in range(1, len(cartelle) + 1)]
    r["variazione_decode_pct_per_input"] = {o: st.median(x["variazione_decode_pct"] for x in coppie if x["input"] == o)
                                            for o in INPUT_APPAIATI}
    r["totale_s_per_input"] = {o: {m: st.median(x[m]["total_s"] for x in coppie if x["input"] == o)
                                   for m in ("denso", "compatto")} for o in INPUT_APPAIATI}
    r["coppie_compatto_piu_rapido_in_decode"] = sum(v < 0 for v in var)
    r["coppie_denso_primo_token_prima"] = sum(x["denso"]["ttft_s"] < x["compatto"]["ttft_s"] for x in coppie)
    r["coppie_compatto_totale_minore"] = sum(x["compatto"]["total_s"] < x["denso"]["total_s"] for x in coppie)
    return r


def tre_modelli(cartelle):
    righe, uscite = {m: [] for m in MODELLI}, {}
    for d in cartelle:
        ott = {(x["model"], x["rep"]): x for x in map(json.loads, (d / "ottimizzati.jsonl").read_text().splitlines())}
        eag = {(x["model"], x["rep"]): x for x in map(json.loads, (d / "eager_parita.jsonl").read_text().splitlines())}
        for o in INPUT_TRE:
            assert len({ott[m, o]["prompt_sha256"] for m in MODELLI}) == 1
        for k, e in eag.items():
            assert ott[k]["outputs"] == e["outputs"] and len(e["outputs"]) == 101
        for (m, o), x in ott.items():
            assert uscite.setdefault((m, o), x["outputs"]) == x["outputs"]
            picco = int(re.search(r"(\d+)\s+peak memory footprint", (d / f"{m}_{o}.stderr").read_text())[1])
            righe[m].append({**x, "picco_processo": picco})
    return {m: {"misure": len(v), "motore": v[0]["framework"],
                "primo_token_ms": sintesi([1000 * x["warm"]["ttft_s"] for x in v]),
                "decode_token_s": sintesi([x["warm"]["decode_tps"] for x in v]),
                "totale_s": sintesi([x["warm"]["total_s"] for x in v]),
                "picco_richiesta_mib": sintesi([x["compiled_request_memory"]["peak_footprint_sampled_bytes"] / 2**20 for x in v]),
                "picco_processo_mib": sintesi([x["picco_processo"] / 2**20 for x in v]),
                "parita_eager": m != "STERA"} for m, v in righe.items()}


def main(base):
    base = Path(base)
    ap = sorted(base.glob("c*_appaiato"), key=lambda p: int(p.name[1:].split("_")[0]))
    tm = sorted(base.glob("c*_tre_modelli"), key=lambda p: int(p.name[1:].split("_")[0]))
    out = {"condizioni": (base / "condizioni.log").read_text().splitlines(),
           "confronto_appaiato": appaiato(ap), "tre_modelli": tre_modelli(tm)}
    (base / "riepilogo.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(base / "riepilogo.json")


if __name__ == "__main__":
    main(sys.argv[1])

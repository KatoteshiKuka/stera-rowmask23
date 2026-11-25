"""Aggrega il giudizio cieco dei racconti con rango ridotto e li confronta, incipit per incipit, con STERA 30M
completo (rango 96, giudizio della tesi). Stessi criteri della tesi: corretto = categoria S (tutti i voti >= 6),
intervallo di Wilson, differenza appaiata con bootstrap (25.000 repliche), McNemar esatto con correzione di Holm."""
import collections, hashlib, json, math, pathlib, random

O = pathlib.Path(__file__).resolve().parent
GC = O.parent.parent / "giudizio_cieco"
ratings = []
for f in sorted((O / "giudizio").glob("sessione*/giudizi_blocchi/*.json")):
    ratings += json.loads(f.read_text())["ratings"]
ratings.sort(key=lambda r: r["id"])
assert [r["id"] for r in ratings] == [f"R{i:04d}" for i in range(1800)], "giudizi incompleti"
for r in ratings:
    assert (r["category"] == "S") == (min(r["grammar"], r["consistency"], r["plot"]) >= 6), r["id"]
rsha = hashlib.sha256(json.dumps(ratings, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
(O / "giudizio" / "giudizi_1800.json").write_text(json.dumps({"ratings": ratings}, ensure_ascii=False, indent=1))
mp = {m["id"]: m for m in json.loads((O / "abbinamento.json").read_text())["map"]}
by = collections.defaultdict(dict)
for r in ratings:
    by[mp[r["id"]]["rango"]][mp[r["id"]]["prompt_id"]] = r
orig = {m["id"]: m for m in json.loads((GC / "abbinamento_modelli.json").read_text())["map"]}
by[96] = {orig[r["id"]]["prompt_id"]: r for r in json.loads((GC / "giudizi_900.json").read_text())["ratings"] if orig[r["id"]]["model"] == "STERA30M"}
pids = sorted(by[96]); assert all(sorted(v) == pids for v in by.values())
nll = {o["rango"]: o["nll_test"] for o in json.loads((O / "risultati_rango_ridotto.json").read_text())["risultati"]}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


rng = random.Random(20261125); out = {"giudizi_sha256": rsha, "giudice": "Claude Opus 5.5, cieco, 6 sessioni su testi mescolati di tutti i ranghi",
                                     "nota": "rango 96 = giudizio della tesi (sessione dei 900 testi)", "ranghi": {}}
base = [by[96][p]["category"] == "S" for p in pids]
for k in sorted(by, reverse=True):
    rr = [by[k][p] for p in pids]; x = [r["category"] == "S" for r in rr]; c = sum(x)
    e = {"nll_test": nll.get(k), "corretti": c, "pct": 100 * c / 300, "wilson95": wilson(c, 300),
         "categorie": dict(collections.Counter(r["category"] for r in rr)),
         "medie": {d: sum(r[d] for r in rr) / 300 for d in ("grammar", "consistency", "plot")}}
    if k != 96:
        ds = [int(i) - int(j) for i, j in zip(x, base)]
        bo = sum(i and not j for i, j in zip(x, base)); bt = sum(j and not i for i, j in zip(x, base)); n = bo + bt
        boot = sorted(sum(rng.choices(ds, k=300)) / 3 for _ in range(25000))
        e["vs_rango96"] = {"delta_pp": 100 * sum(ds) / 300, "ci95_pp": [boot[624], boot[24374]], "solo_ridotto": bo, "solo_96": bt,
                           "mcnemar_p": min(1, 2 * sum(math.comb(n, i) for i in range(min(bo, bt) + 1)) / 2 ** n) if n else 1.0}
    out["ranghi"][str(k)] = e
conf = sorted((k for k in out["ranghi"] if k != "96"), key=lambda k: out["ranghi"][k]["vs_rango96"]["mcnemar_p"]); prev = 0
for i, k in enumerate(conf):
    prev = max(prev, min(1, (len(conf) - i) * out["ranghi"][k]["vs_rango96"]["mcnemar_p"])); out["ranghi"][k]["vs_rango96"]["holm_p"] = prev
(O / "risultati_giudizio_rango_ridotto.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
for k, e in out["ranghi"].items():
    v = e.get("vs_rango96", {})
    print(f"rango {k:>2}: NLL {e['nll_test']:.3f}  corretti {e['corretti']:3d}/300 ({e['pct']:.1f}%)  " +
          (f"delta {v['delta_pp']:+.1f} pp  Holm p={v['holm_p']:.2g}" if v else "(tesi)"))

"""Aggregate the blind judgment of the STERA 30M variant with an 8-bit low-rank term and compare it,
prompt by prompt, with the judgment of STERA 30M FP32 in the thesis. Same criteria as aggregate_and_decode.py:
correct = category S (all scores >= 6), Wilson interval, paired difference with bootstrap (25,000 resamples),
exact McNemar test. Writes risultati_rango8.json (output keys are kept as in the archived file)."""
import collections, hashlib, json, math, pathlib, random

O = pathlib.Path(__file__).resolve().parent
GC = O.parent / "blind_judgment"
ratings = []
for f in sorted((O / "giudizio_rango8" / "giudizi_blocchi").glob("*.json")):
    ratings += json.loads(f.read_text())["ratings"]
assert [r["id"] for r in ratings] == [f"V{i:03d}" for i in range(300)], "incomplete judgments"
for r in ratings:
    assert (r["category"] == "S") == (min(r["grammar"], r["consistency"], r["plot"]) >= 6), r["id"]
rsha = hashlib.sha256(json.dumps(ratings, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
(O / "giudizio_rango8" / "giudizi_300.json").write_text(json.dumps({"ratings": ratings}, ensure_ascii=False, indent=1))

mp = {m["id"]: m["prompt_id"] for m in json.loads((O / "abbinamento_rango8.json").read_text())["map"]}
q = {mp[r["id"]]: r for r in ratings}
orig = {m["id"]: m for m in json.loads((GC / "abbinamento_modelli.json").read_text())["map"]}
f = {orig[r["id"]]["prompt_id"]: r for r in json.loads((GC / "giudizi_900.json").read_text())["ratings"] if orig[r["id"]]["model"] == "STERA30M"}
pids = sorted(f); assert sorted(q) == pids and len(pids) == 300


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [100 * (c - h), 100 * (c + h)]


def summary(d):
    rr = [d[p] for p in pids]; k = sum(r["category"] == "S" for r in rr)
    return {"n": 300, "corretti": k, "pct": 100 * k / 300, "wilson95": wilson(k, 300),
            "categorie": dict(collections.Counter(r["category"] for r in rr)),
            "medie": {x: sum(r[x] for r in rr) / 300 for x in ("grammar", "consistency", "plot")}}


x = [q[p]["category"] == "S" for p in pids]; y = [f[p]["category"] == "S" for p in pids]
ds = [int(i) - int(j) for i, j in zip(x, y)]
bo = sum(i and not j for i, j in zip(x, y)); bt = sum(j and not i for i, j in zip(x, y)); n = bo + bt
pv = min(1, 2 * sum(math.comb(n, i) for i in range(min(bo, bt) + 1)) / 2 ** n) if n else 1.0
rng = random.Random(20261125)
boot = sorted(sum(rng.choices(ds, k=300)) / 3 for _ in range(25000))
out = {"giudizi_sha256": rsha, "giudice": "Claude Opus 5.5, singolo giudice cieco, sessione separata",
       "rango_8bit": summary(q), "fp32_tesi": summary(f),
       "appaiato": {"delta_pp": 100 * sum(ds) / 300, "ci95_pp": [boot[624], boot[24374]], "solo_rango8": bo, "solo_fp32": bt,
                    "entrambi": sum(i and j for i, j in zip(x, y)), "nessuno": sum(not i and not j for i, j in zip(x, y)), "mcnemar_p": pv}}
(O / "risultati_rango8.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
print(json.dumps(out, indent=1, ensure_ascii=False))

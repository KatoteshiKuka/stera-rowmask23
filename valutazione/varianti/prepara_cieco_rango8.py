"""Insieme cieco per la variante con basso rango in 8 bit: 300 continuazioni, identificativi V000-V299
in ordine casuale (seme 202611251200), 10 blocchi da 30. L'abbinamento agli incipit e scritto a parte."""
import json, random, hashlib, pathlib
O = pathlib.Path(__file__).resolve().parent
rows = [json.loads(l) for l in (O / "stera30m_rango8.jsonl").read_text().splitlines()]
assert len(rows) == 300
items = [(r["id"], r["prompt"], r["continuation"]) for r in rows]
random.Random(202611251200).shuffle(items)
stories, mapping = [], []
for i, (pid, pr, co) in enumerate(items):
    sid = f"V{i:03d}"; stories.append({"id": sid, "prompt": pr, "continuation": co}); mapping.append({"id": sid, "prompt_id": pid})
D = O / "giudizio_rango8"; (D / "blocchi").mkdir(parents=True, exist_ok=True); (D / "giudizi_blocchi").mkdir(exist_ok=True)
blob = json.dumps({"stories": stories}, ensure_ascii=False, indent=1)
(D / "racconti_anonimi_300.json").write_text(blob)
for b in range(10):
    s = stories[30 * b:30 * b + 30]
    (D / "blocchi" / f"{s[0]['id']}_{s[-1]['id']}.json").write_text(json.dumps({"stories": s}, ensure_ascii=False, indent=1))
(O / "abbinamento_rango8.json").write_text(json.dumps({"map": mapping}, indent=1))
print("POOL", len(stories), "SHA", hashlib.sha256(blob.encode()).hexdigest())

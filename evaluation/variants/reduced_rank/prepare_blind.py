"""Blind set of the stories generated with the low-rank term reduced after training (k = 64, 48, 32, 16, 8, 0).
1,800 texts shuffled together (seed 202611251500), identifiers R0000-R1799, 60 batches of 30
split into 6 judging sessions of 10 batches: every session sees texts of all ranks.
The assignment to rank and prompt is written separately (abbinamento.json)."""
import hashlib, json, pathlib, random

O = pathlib.Path(__file__).resolve().parent
RANKS = (64, 48, 32, 16, 8, 0)
items = []
for k in RANKS:
    rows = [json.loads(l) for l in (O / f"gen_k{k}.jsonl").read_text().splitlines()]
    assert len(rows) == 300
    items += [(k, r["id"], r["prompt"], r["continuation"]) for r in rows]
random.Random(202611251500).shuffle(items)
stories, mapping = [], []
for i, (k, pid, pr, co) in enumerate(items):
    sid = f"R{i:04d}"; stories.append({"id": sid, "prompt": pr, "continuation": co}); mapping.append({"id": sid, "rango": k, "prompt_id": pid})
D = O / "giudizio"
for s in range(6):
    (D / f"sessione{s + 1}" / "blocchi").mkdir(parents=True, exist_ok=True); (D / f"sessione{s + 1}" / "giudizi_blocchi").mkdir(exist_ok=True)
for b in range(60):
    blk = stories[30 * b:30 * b + 30]
    (D / f"sessione{b // 10 + 1}" / "blocchi" / f"{blk[0]['id']}_{blk[-1]['id']}.json").write_text(json.dumps({"stories": blk}, ensure_ascii=False, indent=1))
blob = json.dumps({"stories": stories}, ensure_ascii=False, indent=1)
(D / "racconti_anonimi_1800.json").write_text(blob)
(O / "abbinamento.json").write_text(json.dumps({"map": mapping}, indent=1))
print("POOL", len(stories), "SHA", hashlib.sha256(blob.encode()).hexdigest())

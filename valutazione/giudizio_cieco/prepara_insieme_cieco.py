"""Costruisce l'insieme cieco di 900 testi (3 modelli x 300 incipit), con identificativi T000-T899
in ordine casuale (seme 202611241500). L'abbinamento ai modelli viene scritto a parte e non stampato."""
import json,random,hashlib,pathlib
O=pathlib.Path(__file__).resolve().parent
G=O.parent/"generazioni"
src={"A":G/"stera30m.jsonl","B":G/"tinystories33m.jsonl","C":G/"tinystories8m.jsonl"}
names={"A":"STERA30M","B":"TinyStories-33M","C":"TinyStories-8M"}
items=[]
for k,p in src.items():
    rows=[json.loads(l) for l in p.read_text().splitlines()]
    assert len(rows)==300
    for r in rows: items.append((k,r["id"],r["prompt"],r["continuation"]))
rng=random.Random(202611241500); rng.shuffle(items)
stories=[];mapping=[]
for i,(k,pid,pr,co) in enumerate(items):
    rid=f"T{i:03d}"; stories.append({"id":rid,"prompt":pr,"continuation":co}); mapping.append({"id":rid,"model":names[k],"prompt_id":pid})
blob=json.dumps({"stories":stories},ensure_ascii=False,indent=1)
(O/"racconti_anonimi_900.json").write_text(blob)
(O/"blocchi").mkdir(exist_ok=True)
for b in range(30):
    s=stories[30*b:30*b+30]
    (O/"blocchi"/f"{s[0]['id']}_{s[-1]['id']}.json").write_text(json.dumps({"stories":s},ensure_ascii=False,indent=1))
(O/"abbinamento_modelli.json").write_text(json.dumps({"map":mapping},indent=1))
print("POOL",len(stories),"SHA",hashlib.sha256(blob.encode()).hexdigest())

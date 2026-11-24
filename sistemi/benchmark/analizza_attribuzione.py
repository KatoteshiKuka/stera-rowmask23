import collections
import itertools
import json
import statistics
from pathlib import Path

root=Path(__file__).resolve().parent.parent / "runs"
rows=[json.loads(line) for line in (root/"attribuzione_grezzi.jsonl").read_text().splitlines()]
groups=("attention","ffn","vocab")
batches=collections.defaultdict(dict)
for r in rows:
    k=r["offset"],r["rep"]
    batches[k][frozenset(r["compact"])]=r["ms_per_token"]
assert len(rows)==96 and len(batches)==12 and all(len(x)==8 for x in batches.values())

savings=[]
for key, costs in batches.items():
    gains={g:[] for g in groups}
    for order in itertools.permutations(groups):
        previous=frozenset()
        for g in order:
            following=previous|{g}
            gains[g].append(costs[previous]-costs[following])
            previous=following
    effects={g:statistics.mean(v) for g,v in gains.items()}
    total=costs[frozenset()]-costs[frozenset(groups)]
    assert abs(sum(effects.values())-total)<1e-9
    savings.append({"offset":key[0],"rep":key[1],"ms_saved":effects,"total_ms_saved":total})

summary={"mean_ms_saved":{g:statistics.mean(r["ms_saved"][g] for r in savings) for g in groups},
         "mean_total_ms_saved":statistics.mean(r["total_ms_saved"] for r in savings),
         "endpoint_mean_ms":{name:statistics.mean(c[k] for c in batches.values())
                            for name,k in (("dense",frozenset()),("stera",frozenset(groups)))},
         "median_total_ms_saved":statistics.median(r["total_ms_saved"] for r in savings),
         "median_ms_by_configuration":{
             "+".join(sorted(k)) or "dense":statistics.median(c[k] for c in batches.values())
             for k in next(iter(batches.values()))},
         "paired_blocks":savings}
(root/"attribuzione_effetti.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({k:v for k,v in summary.items() if k!="paired_blocks"},indent=2))

import json
import random
import statistics
from pathlib import Path

from attribuzione import make_decoder,decode
from misura_dense_stera import CompressedCompiled,load_dense,prompt,mx
from prefill_dense import build_prefill

root=Path(__file__).resolve().parent.parent / "runs"
s,d=CompressedCompiled(512),load_dense()
preload=build_prefill(d,256,512)
styles={"dense":set(),"embedding":{"embedding"},"head":{"head"},"both":{"vocab"}}
steps={name:make_decoder(s,d,groups|{"ffn","attention"}) for name,groups in styles.items()}
rng=random.Random(804)
rows=[]
with (root/"head_grezzi.jsonl").open("x") as out:
    for offset in (10,25,40,55,70,85):
        logits=preload(prompt(offset),0)
        first=mx.argmax(logits).astype(mx.int32)
        mx.eval(first,d.k,d.v)
        state=first,d.k,d.v
        _,ref=decode(steps["dense"],state,256,100)
        for step in steps.values():
            _,tokens=decode(step,state,256,16)
            assert tokens==ref[:16]
        order=list(steps)
        rng.shuffle(order)
        measures={}
        for name in order:
            ms,tokens=decode(steps[name],state,256,100)
            assert tokens==ref
            measures[name]=ms
        entry=0.5*((measures["dense"]-measures["embedding"])+(measures["head"]-measures["both"]))
        head=0.5*((measures["dense"]-measures["head"])+(measures["embedding"]-measures["both"]))
        row={"offset":offset,"ms":measures,"embedding_saved_ms":entry,"head_saved_ms":head}
        rows.append(row)
        out.write(json.dumps(row)+"\n")
        out.flush()
        print("VERIFIED",offset,flush=True)
summary={k:statistics.mean(r[k] for r in rows) for k in ("embedding_saved_ms","head_saved_ms")}
(root/"head_riepilogo.json").write_text(json.dumps(summary,indent=2)+"\n")
print(summary,flush=True)

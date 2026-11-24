import json
import random
import statistics
import time
from itertools import combinations
from pathlib import Path

from misura_dense_stera import load_dense, prompt, CompressedCompiled, mx

ROOT = Path(__file__).resolve().parent.parent / "runs"
GROUPS = ("attention", "ffn", "vocab")

def make_decoder(stera, dense, enabled):
    m = stera.m
    from stera4b_wht_splitk_transposed_v28 import KEMB_PACKED
    positions = mx.arange(512, dtype=mx.int32)
    visible = mx.arange(50304, dtype=mx.int32) < 50257
    w, p = dense.m.w, dense.m.p

    def ln(x, name):
        gain, bias = dense.norms[name]
        d = x - mx.mean(x)
        return d * mx.rsqrt(mx.mean(d*d) + 1e-5) * gain + bias

    def step(token, t, keys, vals):
        token = token.reshape(1)
        if "vocab" in enabled or "embedding" in enabled:
            co, sc, gout, gin, u, v = m.w["embedding"]
            x = KEMB_PACKED(inputs=[co.reshape(-1), sc, gout, gin, m.map, token],
                            grid=(768,1,1), threadgroup=(256,1,1),
                            output_shapes=[(768,)], output_dtypes=[mx.float32])[0]
            x = x + u[token][0] @ v.T
        else:
            x = dense.E[token][0]
        x = x + dense.pos[t.reshape(1)][0]
        nk, nv = [], []
        for layer in range(6):
            n = f"layers.{layer}."
            z = ln(x, n+"ln1")
            if "attention" in enabled:
                q, k, v = (m.matvec(n+name, z).reshape(16,48) for name in ("q","k","v"))
            else:
                qkv = (dense.qkv[layer] @ z).reshape(3,16,48)
                q, k, v = qkv[0], qkv[1], qkv[2]
            k = mx.slice_update(keys[layer], k.reshape(1,16,48), t.reshape(1), axes=[0])
            v = mx.slice_update(vals[layer], v.reshape(1,16,48), t.reshape(1), axes=[0])
            nk.append(k)
            nv.append(v)
            mask = positions <= t
            if layer % 2:
                mask = mask & (positions > t - 256)
            mask = mx.where(mask,0.0,-1e9)[None,None,None,:]
            a = mx.fast.scaled_dot_product_attention(
                q[None,:,None,:], k.transpose(1,0,2)[None], v.transpose(1,0,2)[None],
                scale=1.0, mask=mask)[0,:,0,:].reshape(768)
            x = x + (m.matvec(n+"o",a) if "attention" in enabled else w[n+"o"]@a+p[n+"o.bias"])
            h = ln(x,n+"ln2")
            y = m.matvec(n+"fc",h) if "ffn" in enabled else w[n+"fc"]@h+p[n+"fc.bias"]
            y = 0.5*y*(1+mx.tanh(0.7978845608028654*(y+0.044715*y*y*y)))
            x = x + (m.matvec(n+"proj",y) if "ffn" in enabled else w[n+"proj"]@y+p[n+"proj.bias"])
        x = ln(x,"final_norm")
        logits = m.matvec("embedding",x) if "vocab" in enabled or "head" in enabled else dense.E@x
        return mx.argmax(mx.where(visible,logits,-1e9)).astype(mx.int32),tuple(nk),tuple(nv)

    return mx.compile(step)

def decode(step, state, position, measured):
    token, k, v = state
    outputs = []
    start = time.perf_counter()
    for idx in range(position, position + measured):
        token,k,v = step(token,mx.array(idx,mx.int32),k,v)
        mx.eval(token)
        outputs.append(token)
    return (time.perf_counter()-start)*1000/measured, list(map(int,outputs))

def main():
    s,d = CompressedCompiled(512),load_dense()
    from prefill_dense import build_prefill
    preload = build_prefill(d,256,512)
    choices = [frozenset(x) for n in range(4) for x in combinations(GROUPS,n)]
    steps = {c:make_decoder(s,d,c) for c in choices}
    rng = random.Random(20261123)
    observations = []
    with (ROOT/"attribuzione_grezzi.jsonl").open("x") as out:
        for offset in (10,25,40,55,70,85):
            logits=preload(prompt(offset),0)
            first=mx.argmax(logits).astype(mx.int32)
            mx.eval(first,d.k,d.v)
            state=first,d.k,d.v
            ref,dense_tokens=decode(steps[frozenset()],state,256,100)
            for c in choices:
                _,tokens=decode(steps[c],state,256,32)
                assert tokens==dense_tokens[:32], (offset,sorted(c))
            order=choices[:]
            for rep in range(2):
                rng.shuffle(order)
                for c in order:
                    ms,tokens=decode(steps[c],state,256,100)
                    assert tokens==dense_tokens,(offset,sorted(c))
                    row={"offset":offset,"rep":rep,"compact":sorted(c),"ms_per_token":ms}
                    observations.append(row)
                    out.write(json.dumps(row)+"\n")
                out.flush()
            print("VERIFIED",offset,flush=True)
    summary = {"+".join(sorted(c)) or "dense": statistics.median(
        r["ms_per_token"] for r in observations if set(r["compact"])==c) for c in choices}
    (ROOT/"attribuzione_riepilogo.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(summary,flush=True)

if __name__=="__main__":
    main()

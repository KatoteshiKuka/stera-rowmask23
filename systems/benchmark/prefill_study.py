"""Why the first token of the compact engine is slower than the dense one (256 input tokens, final-checkpoint weights).

A) first token of the compact engine as a function of the prefill block (8 ... 256 tokens per pass);
B) time of the projections alone on 256 tokens: compact block kernel versus dense matrix-matrix product;
C) hybrid strategy: expand the coordinates into dense matrices once, then run a dense prefill.
Every measurement is the median of 7 warm repetitions. Writes a JSON to stdout (keys as in the archived results).
Usage: STERA_WEIGHTS=H122000 python prefill_study.py
"""
import json, os, statistics as st, sys, time
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "engines")); sys.path.insert(0, str(HERE.parent.parent / "training"))
from stera_h80000_fp32_tuned_sdpa_v63 import CompressedCompiled
from prefill_f1 import BatchedPrefill
import stera_batch_rowmask_fp32_v38 as BK
import gptneo_dense_mlx as G
sys.path.insert(0, str(HERE / "TinyStories"))

REPS = 7


def median_ms(f):
    t = []
    for _ in range(REPS + 1):
        s = time.perf_counter(); f(); t.append(time.perf_counter() - s)
    return st.median(t[1:]) * 1000


def ids_prompt():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(HERE.parent / "models" / "tokenizer.json"))
    ids = [50256]
    for r in json.loads((HERE.parent / "models" / "PROMPTS_FRESH_TEST100.json").read_text())[5:]:
        ids += tok.encode(r["prompt"], add_special_tokens=False).ids
        if len(ids) >= 256:
            return ids[:256]


def main():
    ids = ids_prompt(); out = {"token_ingresso": len(ids)}
    model = CompressedCompiled(512)
    # A) prefill block
    res_a = {}; first = None
    for B in ((8, 16, 32, 64, 128, 256) if not os.getenv('SOLO_C') else ()):
        try:
            pf = BatchedPrefill(capacity=512, BLOCK=B, shared_base=model)
            def run():
                model.k = tuple(mx.zeros_like(a) for a in model.k); model.v = tuple(mx.zeros_like(a) for a in model.v)
                for p in range(0, len(ids), B):
                    lg = pf.process(ids[p:p + B], p)
                nxt = mx.argmax(lg); mx.eval(nxt); return int(nxt)
            tok = run(); first = first if first is not None else tok
            res_a[B] = {"primo_token_ms": median_ms(run), "stesso_token": tok == first}
        except Exception as e:
            res_a[B] = {"errore": f"{type(e).__name__}: {str(e)[:120]}"}
        print("A", B, res_a[B], flush=True)
    out["A_blocco_prefill_compatto"] = res_a
    # B) projections alone on 256 tokens
    if os.getenv('SOLO_C'): names = []
    x = {d: mx.random.normal((256, d)) for d in (768, 3072)}; mx.eval(x)
    names = [] if os.getenv('SOLO_C') else [n for n in model.m.w if n != "embedding"]
    dims = {n: (int(model.m.w[n][2].size), int(model.m.w[n][3].size)) for n in names}   # (rows, columns)
    dense = {n: mx.random.normal(dims[n]) for n in names}; mx.eval(dense)
    def compact():
        ys = [BK.packed_matmul(model.m, n, x[dims[n][1]], 1) for n in names]; mx.eval(ys)
    def dense_mm():
        ys = [x[dims[n][1]] @ dense[n].T for n in names]; mx.eval(ys)
    out["B_proiezioni_256_token_ms"] = {"compatto": median_ms(compact), "denso": median_ms(dense_mm), "matrici": len(names)}
    print("B", out["B_proiezioni_256_token_ms"], flush=True)
    # C) expansion of the coordinates
    from ts68_core import RowMask23
    import torch
    rm = RowMask23("cpu"); base = mx.array(rm.base.numpy().astype(np.int32)); res = mx.array(rm.residual.numpy().astype(np.int32))
    mask = mx.array(rm.mask.numpy().reshape(-1))
    def expand_one(co, sc, gout, gin, U, V):
        nt = co.shape[0]; r, c = gout.size, gin.size
        tile = (mx.take(co, base, axis=1) + mx.take(co, res, axis=1) * mask).reshape(nt, 128, 128) * sc.reshape(nt, 1, 1)
        W = tile.reshape(r // 128, c // 128, 128, 128).transpose(0, 2, 1, 3).reshape(r, c)
        return W * gout[:, None] * gin[None, :] + U @ V.T
    expand_c = mx.compile(expand_one)
    # coordinates in the training format (file H122000_fp32_weights.npz), already resident in GPU memory
    z = np.load(HERE.parent / "models" / f"{(os.getenv('STERA_WEIGHTS') or os.getenv('STERA_PESI', 'H122000'))}_fp32_weights.npz")
    all_names = [n[:-7] for n in z.files if n.endswith(".coords")]
    comp = {n: tuple(mx.array(z[n + k]) for k in (".coords", ".scales", ".gout", ".gin", ".U", ".V")) for n in all_names}
    comp = {n: (c[0].reshape(-1, 3176),) + c[1:] for n, c in comp.items()}; mx.eval(comp)
    def expand():
        Ws = [expand_c(*comp[n]) for n in all_names]; mx.eval(Ws); return Ws
    Ws = expand()
    ref = mx.load(str(HERE.parent / "models" / f"{(os.getenv('STERA_WEIGHTS') or os.getenv('STERA_PESI', 'H122000'))}_DENSE_FP32.safetensors"))
    err = max(float(mx.max(mx.abs(W - ref[n]))) for n, W in zip(all_names, Ws))
    nbytes = sum(W.nbytes for W in Ws); del Ws
    out["C_espansione"] = {"ms": median_ms(expand), "byte_matrici_dense": nbytes, "errore_massimo_vs_file_denso": err}
    print("C", out["C_espansione"], flush=True)
    print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    main()

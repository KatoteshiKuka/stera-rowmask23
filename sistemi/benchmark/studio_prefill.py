"""Perche il primo token del motore compatto e piu lento di quello denso (256 token in ingresso, pesi del checkpoint finale).

A) primo token del motore compatto al variare del blocco di prefill (8 ... 256 token per passaggio);
B) tempo delle sole proiezioni su 256 token: kernel compatto a blocchi contro prodotto matrice per matrice denso;
C) strategia ibrida: espansione delle coordinate in matrici dense una sola volta, poi prefill denso.
Ogni misura e la mediana di 7 ripetizioni a caldo. Scrive un JSON su stdout.
Uso: STERA_PESI=H122000 python studio_prefill.py
"""
import json, os, statistics as st, sys, time
from pathlib import Path

import mlx.core as mx
import numpy as np

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI.parent / "motori")); sys.path.insert(0, str(QUI.parent.parent / "addestramento"))
from stera_h80000_fp32_tuned_sdpa_v63 import CompressedCompiled
from prefill_f1 import BatchedPrefill
import stera_batch_rowmask_fp32_v38 as BK
import gptneo_denso_mlx as G
sys.path.insert(0, str(QUI / "TinyStories"))

RIP = 7


def mediana(f):
    t = []
    for _ in range(RIP + 1):
        s = time.perf_counter(); f(); t.append(time.perf_counter() - s)
    return st.median(t[1:]) * 1000


def ids_prompt():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(QUI.parent / "modelli" / "tokenizer.json"))
    ids = [50256]
    for r in json.loads((QUI.parent / "modelli" / "PROMPTS_FRESH_TEST100.json").read_text())[5:]:
        ids += tok.encode(r["prompt"], add_special_tokens=False).ids
        if len(ids) >= 256:
            return ids[:256]


def main():
    ids = ids_prompt(); out = {"token_ingresso": len(ids)}
    model = CompressedCompiled(512)
    # A) blocco di prefill
    ris_a = {}; primo = None
    for B in ((8, 16, 32, 64, 128, 256) if not os.getenv('SOLO_C') else ()):
        try:
            pf = BatchedPrefill(capacity=512, BLOCK=B, shared_base=model)
            def run():
                model.k = tuple(mx.zeros_like(a) for a in model.k); model.v = tuple(mx.zeros_like(a) for a in model.v)
                for p in range(0, len(ids), B):
                    lg = pf.process(ids[p:p + B], p)
                nxt = mx.argmax(lg); mx.eval(nxt); return int(nxt)
            tok = run(); primo = primo if primo is not None else tok
            ris_a[B] = {"primo_token_ms": mediana(run), "stesso_token": tok == primo}
        except Exception as e:
            ris_a[B] = {"errore": f"{type(e).__name__}: {str(e)[:120]}"}
        print("A", B, ris_a[B], flush=True)
    out["A_blocco_prefill_compatto"] = ris_a
    # B) sole proiezioni su 256 token
    if os.getenv('SOLO_C'): nomi = []
    x = {d: mx.random.normal((256, d)) for d in (768, 3072)}; mx.eval(x)
    nomi = [] if os.getenv('SOLO_C') else [n for n in model.m.w if n != "embedding"]
    dims = {n: (int(model.m.w[n][2].size), int(model.m.w[n][3].size)) for n in nomi}   # (righe, colonne)
    dense = {n: mx.random.normal(dims[n]) for n in nomi}; mx.eval(dense)
    def compatto():
        ys = [BK.packed_matmul(model.m, n, x[dims[n][1]], 1) for n in nomi]; mx.eval(ys)
    def denso():
        ys = [x[dims[n][1]] @ dense[n].T for n in nomi]; mx.eval(ys)
    out["B_proiezioni_256_token_ms"] = {"compatto": mediana(compatto), "denso": mediana(denso), "matrici": len(nomi)}
    print("B", out["B_proiezioni_256_token_ms"], flush=True)
    # C) espansione delle coordinate
    from ts68_core import RowMask23
    import torch
    rm = RowMask23("cpu"); base = mx.array(rm.base.numpy().astype(np.int32)); res = mx.array(rm.residual.numpy().astype(np.int32))
    mask = mx.array(rm.mask.numpy().reshape(-1))
    def espandi_una(co, sc, gout, gin, U, V):
        nt = co.shape[0]; r, c = gout.size, gin.size
        tile = (mx.take(co, base, axis=1) + mx.take(co, res, axis=1) * mask).reshape(nt, 128, 128) * sc.reshape(nt, 1, 1)
        W = tile.reshape(r // 128, c // 128, 128, 128).transpose(0, 2, 1, 3).reshape(r, c)
        return W * gout[:, None] * gin[None, :] + U @ V.T
    esp = mx.compile(espandi_una)
    # coordinate nel formato dell'addestramento (file H122000_fp32_weights.npz), gia residenti in memoria GPU
    z = np.load(QUI.parent / "modelli" / f"{os.getenv('STERA_PESI', 'H122000')}_fp32_weights.npz")
    tutte = [n[:-7] for n in z.files if n.endswith(".coords")]
    comp = {n: tuple(mx.array(z[n + k]) for k in (".coords", ".scales", ".gout", ".gin", ".U", ".V")) for n in tutte}
    comp = {n: (c[0].reshape(-1, 3176),) + c[1:] for n, c in comp.items()}; mx.eval(comp)
    def espandi():
        Ws = [esp(*comp[n]) for n in tutte]; mx.eval(Ws); return Ws
    Ws = espandi()
    rif = mx.load(str(QUI.parent / "modelli" / f"{os.getenv('STERA_PESI', 'H122000')}_DENSE_FP32.safetensors"))
    err = max(float(mx.max(mx.abs(W - rif[n]))) for n, W in zip(tutte, Ws))
    byte = sum(W.nbytes for W in Ws); del Ws
    out["C_espansione"] = {"ms": mediana(espandi), "byte_matrici_dense": byte, "errore_massimo_vs_file_denso": err}
    print("C", out["C_espansione"], flush=True)
    print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    main()

"""Prova di controllo per le misure a contesto lungo del modello fittizio F10.

Il modello genera da solo N token di fila, partendo da un solo token: la KV cache si riempie davvero un token
alla volta, come in una conversazione che si allunga. Si registra il tempo di ogni passo e si confronta il tempo
per token attorno a N con quello misurato da f10_contesto.py con la cache riempita di valori casuali.
Uso: STERA_PESI=H122000 python f10_controllo_reale.py --token 8000 --out file.json
"""
import argparse, json, statistics, sys, time
from pathlib import Path
import mlx.core as mx
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "motori")); sys.path.insert(0, str(R / "f10"))
import stera_independent_fp32_longctx_v47 as LC
import stera4b_stream_mlx as S
import carica_f10
LC.M.DirectIndependentModel = carica_f10.F10DaFile

ap = argparse.ArgumentParser(); ap.add_argument("--token", type=int, default=8000); ap.add_argument("--out", required=True)
a = ap.parse_args(); N = a.token
cap = next(c for c in (512, 2048, 4096, 8192, 16384, 32768, 65536) if c >= N + 1)
m = LC.LongCompiled(cap)
kk, vv = m.k, m.v; m.k = m.v = None
tok = mx.array(50256, mx.int32); tempi = []; t0 = time.time()
for t in range(N):
    st = time.perf_counter()
    logits, nk, nv = m.fn(tok, mx.array(t, mx.int32), kk, vv)
    del kk, vv
    tok = mx.argmax(logits).astype(mx.int32); mx.eval(tok, nk, nv)
    kk, vv = nk, nv; del nk, nv
    tempi.append(time.perf_counter() - st)
    if t % 1000 == 0: print("passo", t, "ms", round(1000 * tempi[-1], 1), "trascorsi", round(time.time() - t0), "s", flush=True)
fine = tempi[-24:]
ris = dict(token=N, capacita_cache=cap, modello=carica_f10.FILE.name, sha256_modello=m.m.sourcehash,
           ms_mediana_ultimi_24=1000 * statistics.median(fine), decode_tok_s_ultimi_24=len(fine) / sum(fine),
           ms_mediana_per_blocco_di_1000={str(i): 1000 * statistics.median(tempi[i:i + 1000]) for i in range(0, N, 1000)},
           durata_s=time.time() - t0, memoria=S.get_mem())
Path(a.out).write_text(json.dumps(ris, indent=1)); print(json.dumps(ris))

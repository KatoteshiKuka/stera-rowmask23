"""Decode del modello fittizio STERA F10 (non addestrato, generato da sistemi/f10/genera_f10.py a partire dal
checkpoint finale H122000) al variare del contesto.

Prova meccanica: la cache KV viene riempita con valori casuali (i layer globali fino a L posizioni
(quella dei layer locali fino a 256), poi si misurano i passi di decode con t = L, ..., L+23.
L'attention legge effettivamente tutte le posizioni occupate. Le posizioni oltre 2047 usano l'ultima
riga della tabella delle posizioni: la prova misura il costo di esecuzione, non la qualità a contesto lungo.
Uso: STERA_PESI=H122000 python f10_contesto.py --contesto 8192 --out file.json
"""
import argparse, json, statistics, sys, time
from pathlib import Path
import mlx.core as mx
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "motori"))
import stera_independent_fp32_longctx_v47 as LC
import stera4b_stream_mlx as S
sys.path.insert(0, str(R / "f10"))
import carica_f10
LC.M.DirectIndependentModel = carica_f10.F10DaFile   # il modello si legge dal file generato

ap = argparse.ArgumentParser(); ap.add_argument("--contesto", type=int, required=True)
ap.add_argument("--passi", type=int, default=24); ap.add_argument("--kv", default="fp32"); ap.add_argument("--out", required=True)
a = ap.parse_args()
L = a.contesto
cap = next(c for c in (512, 2048, 4096, 8192, 16384, 32768, 65536) if c >= L + a.passi)
m = LC.LongCompiled(cap, kv_precision=a.kv)
mx.random.seed(0)
# cache riempita con valori casuali; i riferimenti alla cache vengono rilasciati prima di ogni passo, così MLX
# aggiorna la cache sul posto invece di copiarla (come farebbe un motore in uso reale)
kk = tuple(mx.random.normal(x.shape, dtype=x.dtype) * 0.5 for x in m.k)
vv = tuple(mx.random.normal(x.shape, dtype=x.dtype) * 0.5 for x in m.v)
m.k = m.v = None; mx.eval(kk, vv)
if hasattr(mx, "reset_peak_memory"): mx.reset_peak_memory()
tok = mx.array(50256, mx.int32); tempi = []
for i in range(a.passi + 3):
    t = L + i
    st = time.perf_counter()
    logits, nk, nv = m.fn(tok, mx.array(t, mx.int32), kk, vv)
    del kk, vv
    tok = mx.argmax(logits).astype(mx.int32); mx.eval(tok, nk, nv)
    kk, vv = nk, nv; del nk, nv
    tempi.append(time.perf_counter() - st)
dec = tempi[3:]
ris = dict(contesto=L, capacita_cache=cap, kv=a.kv, passi=a.passi, modello=carica_f10.FILE.name, sha256_modello=m.m.sourcehash, kv_byte=sum(x.nbytes for x in kk + vv),
           decode_tok_s=len(dec) / sum(dec), ms_mediana=1000 * statistics.median(dec),
           ms_p90=1000 * sorted(dec)[int(.9 * len(dec)) - 1], primo_passo_s=tempi[0],
           memoria=S.get_mem(), mlx_picco_mib=round(mx.get_peak_memory() / 2**20, 1) if hasattr(mx, "get_peak_memory") else None)
Path(a.out).write_text(json.dumps(ris, indent=1)); print(json.dumps(ris))

"""Verifica termica: decode a contesto fisso L (cache casuale) ripetuto per una durata prolungata, per
distinguere l'effetto della lunghezza della cache da quello del riscaldamento del MacBook Air (senza ventola).
Uso: STERA_PESI=H122000 python f10_contesto_prolungato.py --contesto 8000 --secondi 800 --out file.json
"""
import argparse, json, statistics, sys, time
from pathlib import Path
import mlx.core as mx
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "motori")); sys.path.insert(0, str(R / "f10"))
import stera_independent_fp32_longctx_v47 as LC, carica_f10
LC.M.DirectIndependentModel = carica_f10.F10DaFile
ap = argparse.ArgumentParser(); ap.add_argument("--contesto", type=int, default=8000); ap.add_argument("--secondi", type=int, default=800); ap.add_argument("--out", required=True)
a = ap.parse_args(); L = a.contesto
m = LC.LongCompiled(next(c for c in (512, 2048, 4096, 8192, 16384, 32768) if c > L))
mx.random.seed(0)
kk = tuple(mx.random.normal(x.shape, dtype=x.dtype) * 0.5 for x in m.k); vv = tuple(mx.random.normal(x.shape, dtype=x.dtype) * 0.5 for x in m.v)
m.k = m.v = None; mx.eval(kk, vv)
tok = mx.array(50256, mx.int32); t0 = time.time(); serie = []
while time.time() - t0 < a.secondi:
    st = time.perf_counter()
    logits, nk, nv = m.fn(tok, mx.array(L, mx.int32), kk, vv)   # stessa posizione: contesto costante
    del kk, vv
    tok = mx.argmax(logits).astype(mx.int32); mx.eval(tok, nk, nv); kk, vv = nk, nv; del nk, nv
    serie.append((time.time() - t0, time.perf_counter() - st))
per_minuto = {}
for s, d in serie[3:]:
    per_minuto.setdefault(int(s // 60), []).append(d)
ris = dict(contesto=L, secondi=a.secondi, ms_mediana_per_minuto={k: 1000 * statistics.median(v) for k, v in per_minuto.items()})
Path(a.out).write_text(json.dumps(ris, indent=1)); print(json.dumps(ris))

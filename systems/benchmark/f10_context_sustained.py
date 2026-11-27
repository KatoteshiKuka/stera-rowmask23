"""Thermal check: decode at a fixed context L (random cache) repeated for a long time, to separate the
effect of the cache length from that of the heating of the fanless MacBook Air.
Usage: STERA_WEIGHTS=H122000 python f10_context_sustained.py --context 8000 --seconds 800 --out file.json
Output keys are kept as in the archived result (contesto, secondi, ms_mediana_per_minuto).
"""
import argparse, json, statistics, sys, time
from pathlib import Path
import mlx.core as mx
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "engines")); sys.path.insert(0, str(R / "f10"))
import stera_independent_fp32_longctx_v47 as LC, load_f10
LC.M.DirectIndependentModel = load_f10.F10FromFile
ap = argparse.ArgumentParser(); ap.add_argument("--context", "--contesto", dest="context", type=int, default=8000); ap.add_argument("--seconds", "--secondi", dest="seconds", type=int, default=800); ap.add_argument("--out", required=True)
a = ap.parse_args(); L = a.context
m = LC.LongCompiled(next(c for c in (512, 2048, 4096, 8192, 16384, 32768) if c > L))
mx.random.seed(0)
kk = tuple(mx.random.normal(x.shape, dtype=x.dtype) * 0.5 for x in m.k); vv = tuple(mx.random.normal(x.shape, dtype=x.dtype) * 0.5 for x in m.v)
m.k = m.v = None; mx.eval(kk, vv)
tok = mx.array(50256, mx.int32); t0 = time.time(); series = []
while time.time() - t0 < a.seconds:
    st = time.perf_counter()
    logits, nk, nv = m.fn(tok, mx.array(L, mx.int32), kk, vv)   # same position: constant context
    del kk, vv
    tok = mx.argmax(logits).astype(mx.int32); mx.eval(tok, nk, nv); kk, vv = nk, nv; del nk, nv
    series.append((time.time() - t0, time.perf_counter() - st))
per_minute = {}
for s, d in series[3:]:
    per_minute.setdefault(int(s // 60), []).append(d)
res = dict(contesto=L, secondi=a.seconds, ms_mediana_per_minuto={k: 1000 * statistics.median(v) for k, v in per_minute.items()})
Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res))

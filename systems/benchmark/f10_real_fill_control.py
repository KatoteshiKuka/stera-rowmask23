"""Control test for the long-context measurements of the fictitious F10 model.

The model generates N tokens in a row by itself, starting from a single token: the KV cache really fills one
token at a time, as in a conversation that grows. The time of every step is recorded, and the time per token
around N is compared with the one measured by f10_context.py with the cache filled with random values.
Usage: STERA_WEIGHTS=H122000 python f10_real_fill_control.py --tokens 8000 --out file.json
Output keys are kept as in the archived result controllo_reale_8000.json.
"""
import argparse, json, statistics, sys, time
from pathlib import Path
import mlx.core as mx
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "engines")); sys.path.insert(0, str(R / "f10"))
import stera_independent_fp32_longctx_v47 as LC
import stera4b_stream_mlx as S
import load_f10
LC.M.DirectIndependentModel = load_f10.F10FromFile

ap = argparse.ArgumentParser(); ap.add_argument("--tokens", "--token", dest="token", type=int, default=8000); ap.add_argument("--out", required=True)
a = ap.parse_args(); N = a.token
cap = next(c for c in (512, 2048, 4096, 8192, 16384, 32768, 65536) if c >= N + 1)
m = LC.LongCompiled(cap)
kk, vv = m.k, m.v; m.k = m.v = None
tok = mx.array(50256, mx.int32); times = []; t0 = time.time()
for t in range(N):
    st = time.perf_counter()
    logits, nk, nv = m.fn(tok, mx.array(t, mx.int32), kk, vv)
    del kk, vv
    tok = mx.argmax(logits).astype(mx.int32); mx.eval(tok, nk, nv)
    kk, vv = nk, nv; del nk, nv
    times.append(time.perf_counter() - st)
    if t % 1000 == 0: print("step", t, "ms", round(1000 * times[-1], 1), "elapsed", round(time.time() - t0), "s", flush=True)
last = times[-24:]
res = dict(token=N, capacita_cache=cap, modello=load_f10.FILE.name, sha256_modello=m.m.sourcehash,
           ms_mediana_ultimi_24=1000 * statistics.median(last), decode_tok_s_ultimi_24=len(last) / sum(last),
           ms_mediana_per_blocco_di_1000={str(i): 1000 * statistics.median(times[i:i + 1000]) for i in range(0, N, 1000)},
           durata_s=time.time() - t0, memoria=S.get_mem())
Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res))

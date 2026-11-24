import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
MODELLI = WORKSPACE / "modelli"
sys.path.insert(0, str(WORKSPACE / "motori"))
import dense_clone_fp32_v8 as dense
from prefill_dense import build_prefill
from prefill_f1 import BatchedPrefill
from stera_h80000_fp32_dense_sdpa_v62 import CompiledStatic
from stera_h80000_fp32_tuned_sdpa_v63 import CompressedCompiled

def load_dense():
    weights = mx.load(str(MODELLI/"H80000_DENSE_FP32.safetensors"))
    matrices = {name for name, _, _ in dense.BASE}
    def from_file(factor):
        assert factor == 1
        return SimpleNamespace(
            w={k: v for k, v in weights.items() if k in matrices},
            p={k: v for k, v in weights.items() if k not in matrices})
    with patch.object(dense, "Model", from_file):
        return CompiledStatic(512)

def prompt(offset):
    tok = Tokenizer.from_file(str(MODELLI/"tokenizer.json"))
    ids = [50256]
    for row in json.loads((MODELLI/"PROMPTS_FRESH_TEST100.json").read_text())[offset:]:
        ids.extend(tok.encode(row["prompt"], add_special_tokens=False).ids)
        if len(ids) >= 256:
            return ids[:256]
    raise RuntimeError("Insufficient input tokens")

def measure(model, ids, block, prefill):
    model.k = tuple(mx.zeros_like(k) for k in model.k)
    model.v = tuple(mx.zeros_like(v) for v in model.v)
    mx.eval(model.k, model.v)
    tokens = [mx.array(i,mx.int32) for i in ids]
    start = time.perf_counter()
    if block:
        for offset in range(0, len(ids), block):
            logits = prefill(ids[offset:offset+block], offset)
        nxt = mx.argmax(logits).astype(mx.int32)
        mx.eval(nxt)
    else:
        for position, token in enumerate(tokens):
            nxt = mx.argmax(model.step(token, position)).astype(mx.int32)
            mx.eval(nxt)
    first = time.perf_counter()
    outputs = [nxt]
    for position in range(len(ids), len(ids) + 100):
        nxt = mx.argmax(model.step(nxt, position)).astype(mx.int32)
        mx.eval(nxt)
        outputs.append(nxt)
    end = time.perf_counter()
    return {"ttft_s": first-start, "decode_s": end-first, "total_s":end-start,
            "decode_tps":100/(end-first), "outputs":list(map(int,outputs))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=["dense","stera"])
    ap.add_argument("block", type=int, choices=[0,16,32,64,128,256])
    ap.add_argument("offset", type=int)
    args = ap.parse_args()
    ids = prompt(args.offset)
    model = load_dense() if args.model == "dense" else CompressedCompiled(512)
    prefill = None
    if args.block:
        prefill = (build_prefill(model,args.block,512) if args.model=="dense"
                   else BatchedPrefill(capacity=512,BLOCK=args.block,shared_base=model).process)
    cold = measure(model, ids,args.block,prefill)
    warm = measure(model, ids,args.block,prefill)
    assert cold["outputs"] == warm["outputs"]
    print(json.dumps({"model":args.model,"block":args.block,"offset":args.offset,
                      "prompt_sha256":hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                      "first":cold,"warm":warm}),flush=True)

if __name__ == "__main__":
    main()

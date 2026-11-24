import argparse
import hashlib
import json
import os
from types import MethodType
import statistics
import sys
import time
from pathlib import Path
from footprint import observe_request

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "modelli"
KERNELS = ROOT / "motori"
PUBLIC = BASE / "TinyStories"

def prompt_ids(model, offset=0):
    from tokenizers import Tokenizer
    ref = Tokenizer.from_file(str(BASE / "tokenizer.json"))
    tok = ref if model == "STERA" else Tokenizer.from_file(str(PUBLIC / model / "tokenizer.json"))
    rows = json.loads((BASE / "PROMPTS_FRESH_TEST100.json").read_text())
    ids = [50256]
    for row in rows[offset:]:
        a = ref.encode(row["prompt"], add_special_tokens=False).ids
        b = tok.encode(row["prompt"], add_special_tokens=False).ids
        if a != b:
            raise ValueError("Different prompt tokenization")
        ids.extend(a)
        if len(ids) >= 256:
            return ids[:256]
    raise ValueError("Prompt too short")

def optimized_attention(self, query, key, value, attention_mask=None):
    import torch
    nq, nk = query.size(-2), key.size(-2)
    mask = self.bias[:, :, nk - nq:nk, :nk]
    if attention_mask is not None:
        mask = attention_mask.masked_fill(~mask, torch.finfo(query.dtype).min)
    y = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, dropout_p=0.0, scale=1.0,
    )
    return y, None

def stera(ids):
    import mlx.core as mx
    sys.path.insert(0, str(KERNELS))
    from stera_h80000_fp32_tuned_sdpa_v63 import CompressedCompiled
    start = time.perf_counter()
    model = CompressedCompiled(512)
    mx.synchronize()
    batch = int(os.getenv("STERA_PREFILL_BLOCK", "0"))
    if batch:
        from prefill_f1 import BatchedPrefill
        prefill = BatchedPrefill(capacity=512, BLOCK=batch, shared_base=model)
    load = time.perf_counter() - start
    inputs = [mx.array(i, mx.int32) for i in ids]
    def next_token(inp, position):
        out = mx.argmax(model.step(inp, position)).astype(mx.int32)
        mx.eval(out)
        return out
    def measure():
        start = time.perf_counter()
        if batch:
            for position in range(0, len(ids), batch):
                logits = prefill.process(ids[position:position+batch], position)
            nxt = mx.argmax(logits).astype(mx.int32)
            mx.eval(nxt)
        else:
            for position, inp in enumerate(inputs):
                nxt = next_token(inp, position)
        ttft = time.perf_counter() - start
        outputs = [nxt]
        times = []
        start = time.perf_counter()
        for position in range(len(ids), len(ids) + 100):
            t = time.perf_counter()
            nxt = next_token(nxt, position)
            times.append(time.perf_counter() - t)
            outputs.append(nxt)
        decode = time.perf_counter() - start
        return load, ttft, decode, times, {"framework": "MLX/Metal v63", "prefill": f"block{batch}" if batch else "sequential", "outputs": [int(v) for v in outputs],
            "active_mib": mx.get_active_memory() / 2**20}
    with observe_request() as cold_memory:
        cold = measure()
    model.k = tuple(mx.zeros_like(a) for a in model.k)
    model.v = tuple(mx.zeros_like(a) for a in model.v)
    mx.eval(model.k, model.v)
    with observe_request() as warm_memory:
        warm = measure()
    assert cold[-1]["outputs"] == warm[-1]["outputs"]
    cold[-1]["warm"] = {"ttft_s": warm[1], "decode_tps": 100 / warm[2], "total_s": warm[1] + warm[2]}
    cold[-1]["compiled_request_memory"] = warm_memory
    cold[-1]["first_request_memory"] = cold_memory
    return cold


def dense(ids, name):
    import torch
    from transformers import AutoModelForCausalLM
    with torch.inference_mode():
        start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            PUBLIC / name, local_files_only=True, dtype=torch.float32,
            attn_implementation="eager",
        ).eval().to("mps")
        use_sdpa = os.getenv("GPTNEO_SDPA") == "1"
        last_logit = os.getenv("GPTNEO_LAST_LOGIT") == "1"
        if use_sdpa:
            for layer in model.transformer.h:
                attention = layer.attn.attention
                attention._attn = MethodType(optimized_attention, attention)
        torch.mps.synchronize()
        load = time.perf_counter() - start
        prompt = torch.tensor([ids], device="mps")
        def measure():
            start = time.perf_counter()
            out = model(input_ids=prompt, use_cache=True, logits_to_keep=int(last_logit))
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            torch.mps.synchronize()
            ttft = time.perf_counter() - start
            outputs = [nxt]
            times = []
            start = time.perf_counter()
            for _ in range(100):
                t = time.perf_counter()
                out = model(input_ids=nxt, cache=out.past_key_values, use_cache=True)
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                torch.mps.synchronize()
                times.append(time.perf_counter() - t)
                outputs.append(nxt)
            decode = time.perf_counter() - start
            return load, ttft, decode, times, {"framework": f"Transformers/PyTorch {torch.__version__} MPS",
                "prefill": "parallel", "attention": "SDPA adapter" if use_sdpa else "eager",
                "last_logit_only": last_logit,
                "outputs": torch.cat(outputs,dim=1).cpu().flatten().tolist(),
                "parameters": sum(p.numel() for p in model.parameters()),
                "active_mib": torch.mps.current_allocated_memory() / 2**20}
        with observe_request() as cold_memory:
            cold = measure()
        with observe_request() as warm_memory:
            warm = measure()
        assert cold[-1]["outputs"] == warm[-1]["outputs"]
        cold[-1]["warm"] = {"ttft_s": warm[1], "decode_tps": 100 / warm[2], "total_s": warm[1] + warm[2]}
        cold[-1]["compiled_request_memory"] = warm_memory
        cold[-1]["first_request_memory"] = cold_memory
        return cold


parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["STERA", "8M", "28M", "33M"])
parser.add_argument("rep", type=int)
args = parser.parse_args()
ids = prompt_ids(args.model, int(os.getenv("PROMPT_OFFSET", "0")))
load, ttft, decode, times, detail = stera(ids) if args.model == "STERA" else dense(ids, args.model)
print(json.dumps({"model": args.model, "rep": args.rep, "dtype": "float32", "batch": 1,
    "prompt_tokens": 256, "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    "decode_steps": 100, "load_s": load, "ttft_after_load_s": ttft, "decode_s": decode,
    "decode_tps": 100 / decode, "total_after_load_s": ttft + decode,
    "median_step_ms": statistics.median(times) * 1000, **detail}), flush=True)

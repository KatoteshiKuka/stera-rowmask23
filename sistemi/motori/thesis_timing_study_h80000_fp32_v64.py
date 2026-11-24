#!/usr/bin/env python3
"""FAIR STERA thesis study: same H80000 trained FP32 function, 3 independent
decoders (old RowMask23, optimized RowMask23, reconstructed dense), same prompts,
context, deterministic greedy tokens. Explicit prefill, decode, cold model init,
memory, attention. Every run isolated in a NEW process."""
import argparse,hashlib,json,time,statistics
from pathlib import Path
import mlx.core as mx
from tokenizers import Tokenizer
from stera4b_fp32_compressed_jit_v11 import CompressedCompiled as STERA_V11
from stera_h80000_fp32_optimized_v56 import CompressedCompiled as STERA_V56
from stera_h80000_fp32_tuned_v59 import CompressedCompiled as STERA_V59
from stera_h80000_fp32_tuned_sdpa_v63 import CompressedCompiled as STERA_V63
from stera_h80000_fp32_dense_sdpa_v62 import CompiledStatic as DENSE_V62
from stera4b_fp32_compiled_static_v10 import CompiledStatic as DENSE
ROOT=Path(__file__).resolve().parent
def prompt_tokens(length):
 path=ROOT.parent/'PROMPTS_FRESH_TEST100.json'
 prompts=json.loads(path.read_text())
 tokenizer=Tokenizer.from_file(str(ROOT.parent/'STERA_FINAL_H80000_LOSSLESS_FROZEN_20261122/tokenizer.json'))
 ids=[50256]
 while len(ids)<length:
  for row in prompts:
   ids+=tokenizer.encode(row['prompt'],add_special_tokens=False).ids
   if len(ids)>=length:break
 return ids[:length]
def run():
 p=argparse.ArgumentParser()
 p.add_argument('--mode',choices=('stera_old','stera_optimized','stera_tuned','stera_tuned_sdpa','dense','dense_sdpa'),required=True)
 p.add_argument('--prompt',type=int,default=256)
 p.add_argument('--decode',type=int,default=100)
 p.add_argument('--rep',type=int,default=1)
 a=p.parse_args()
 assert a.prompt+a.decode<=2048
 ids=prompt_tokens(a.prompt)
 capacity=(512 if a.prompt+a.decode<=512 else 1024 if a.prompt+a.decode<=1024 else 2048)
 sha=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
 start=time.perf_counter()
 if a.mode=='stera_old':model=STERA_V11(capacity)
 elif a.mode=='stera_optimized':model=STERA_V56(capacity)
 elif a.mode=='stera_tuned':model=STERA_V59(capacity)
 elif a.mode=='stera_tuned_sdpa':model=STERA_V63(capacity)
 elif a.mode=='dense_sdpa':model=DENSE_V62(capacity)
 else:model=DENSE(capacity)
 construction=time.perf_counter()-start
 prefill_durations=[]; decode_durations=[]
 next_token=mx.array(50256,mx.int32);output=[]
 for i in range(a.prompt+a.decode):
  st=time.perf_counter()
  inp=mx.array(ids[i],mx.int32) if i<a.prompt else next_token
  lg=model.step(inp,i)
  next_token=mx.argmax(lg).astype(mx.int32)
  mx.eval(next_token)
  took=time.perf_counter()-st
  if i<a.prompt:prefill_durations.append(took)
  else:decode_durations.append(took)
  if i>=a.prompt-1:output.append(int(next_token))
 prefill_s=sum(prefill_durations)
 dec_s=sum(decode_durations)
 obj={'mode':a.mode,'rep':a.rep,'hardware':'Apple Mac M1 16GiB',
  'trained_checkpoint':'STERA H80000 FP32 lossless frozen',
  'prompt_sha256':sha,'prompt_tokens':a.prompt,'decode_steps':a.decode,'kv_capacity':capacity,
  'dtype_weights':'float32','dtype_cache':'float32','batch_size':1,'greedy':True,
  'init_s':construction,'prefill_s':prefill_s,'ttft_after_model_load_s':prefill_s,
  'prefill_tokens_per_second':a.prompt/prefill_s,
  'decode_s':dec_s,'decode_tokens_per_second':a.decode/dec_s,
  'time_prompt_plus_decode_s':prefill_s+dec_s,
  'decode_median_ms':statistics.median(decode_durations)*1000,
  'decode_p90_ms':sorted(decode_durations)[int(.9*len(decode_durations))]*1000,
  'mlx_active_mib':mx.get_active_memory()/2**20,
  'mlx_peak_mib':mx.get_peak_memory()/2**20,
  'output_ids':output[:a.decode],
  'note':'Includes MLX first-step compilation in prefill; excludes tokenization and process launch. Same trained H80000; not untrained 4.63B.'}
 out=ROOT/f'THESIS_TIMING_STUDY_F1_{a.mode}_P{a.prompt}_D{a.decode}_R{a.rep:02d}.json'
 out.write_text(json.dumps(obj,indent=2))
 print('RESULT',out.name, 'prefill',round(prefill_s,4),'decode_tps',round(a.decode/dec_s,3),'mem',round(obj['mlx_active_mib'],2),'peak',round(obj['mlx_peak_mib'],2),'sha256',sha,flush=True)
if __name__=='__main__':run()

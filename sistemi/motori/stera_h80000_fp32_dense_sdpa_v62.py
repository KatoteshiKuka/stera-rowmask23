#!/usr/bin/env python3
"""FP32 mathematically exact contracted x10 STERA 4.63B clone.
MLX static-shape KV cache and compile whole decode graph; same training H80000
weights, no quantization. ONLY valid for exact Kronecker-replicated F10 model.
Do not equate this speed with arbitrary 4B-independent-weights inference.
"""
import argparse,time,json,statistics,hashlib
from pathlib import Path
import numpy as np
import mlx.core as mx
from stera4b_factored_fp32_v8 import FP32ExactFactored,TOK_PATH
import dense_clone_fp32_v8 as D
ROOT=Path(__file__).resolve().parent
class CompiledStatic(FP32ExactFactored):
 def __init__(self, max_len=128, compile_it=True):
  super().__init__()
  self.max_len=max_len
  self.k=tuple(mx.zeros((max_len,16,48),dtype=mx.float32) for _ in range(6))
  self.v=tuple(mx.zeros((max_len,16,48),dtype=mx.float32) for _ in range(6))
  W=self.m.w;P=self.m.p;E=self.E;POS=self.pos;Q=self.qkv
  norm=self.norms
  position_range=mx.arange(max_len,dtype=mx.int32)
  mask_vocab=mx.arange(50304,dtype=mx.int32)<50257
  def ln(x,name):
   w,b=norm[name]
   mu=mx.mean(x)
   centered=x-mu
   return centered*mx.rsqrt(mx.mean(centered*centered)+1e-5)*w+b
  def stat_step(token,t,old_k,old_v):
   x=mx.take(E,token.reshape((1,)),axis=0)[0]+mx.take(POS,t.reshape((1,)),axis=0)[0]
   new_k=[];new_v=[]
   for l in range(6):
    pre=f'layers.{l}.'
    qkv=(Q[l]@ln(x,pre+'ln1')).reshape(3,16,48)
    q=qkv[0];k=qkv[1];v=qkv[2]
    K=mx.slice_update(old_k[l],k.reshape((1,16,48)),t.reshape((1,)),axes=[0])
    V=mx.slice_update(old_v[l],v.reshape((1,16,48)),t.reshape((1,)),axes=[0])
    new_k.append(K);new_v.append(V)
    valid=(position_range<=t)
    if l%2:valid=valid & (position_range>t-256)
    causal_mask=mx.where(valid,mx.array(0.0,mx.float32),mx.array(-1e9,mx.float32))[None,None,None,:]
    a=mx.fast.scaled_dot_product_attention(q[None,:,None,:],K.transpose(1,0,2)[None],
       V.transpose(1,0,2)[None],scale=1.0,mask=causal_mask)[0,:,0,:].reshape(768)
    x=x + W[pre+'o']@a + P[pre+'o.bias']
    y=W[pre+'fc']@ln(x,pre+'ln2') + P[pre+'fc.bias']
    y=0.5*y*(1+mx.tanh(0.7978845608028654*(y+0.044715*y*y*y)))
    x=x+W[pre+'proj']@y+P[pre+'proj.bias']
   logits=E@ln(x,'final_norm')
   return mx.where(mask_vocab,logits,-1.0e9),tuple(new_k),tuple(new_v)
  self.compiled=mx.compile(stat_step) if compile_it else stat_step
  self.compilation=compile_it
 def step(self,tid,t):
  lg,self.k,self.v=self.compiled(tid,mx.array(t,mx.int32),self.k,self.v)
  return lg
def run(tokens=100,compile_it=True,max_context=128):
 from tokenizers import Tokenizer
 prompt='Once upon a time, there was a little cat named Mimi.'
 tok=Tokenizer.from_file(str(TOK_PATH))
 ids=[50256]+tok.encode(prompt,add_special_tokens=False).ids
 ids=ids[:3]
 assert tokens+len(ids)<=max_context, (tokens,max_context)
 m=CompiledStatic(max_len=max_context,compile_it=compile_it)
 timing=[]
 generated=[]
 nxt=mx.array(ids[0],mx.int32)
 all_tokens=[]
 start=time.perf_counter()
 for t in range(len(ids)+tokens):
  cur=mx.array(ids[t],mx.int32) if t<len(ids) else nxt
  st=time.perf_counter()
  logits=m.step(cur,t)
  nxt=mx.argmax(logits).astype(mx.int32)
  mx.eval(nxt)
  if t>=len(ids)-1:generated.append(nxt)
  timing.append(time.perf_counter()-st)
  if t==2:np.save(ROOT/f'FP32_COMPILED_STATIC_{int(compile_it)}_FIRST_LOGITS.npy',np.array(logits))
  if t%10==0:print('TOKEN',t,'duration_ms',round(timing[-1]*1000,3),'gpu',D.usage(),flush=True)
 t=timing[len(ids):]
 summary={'tok_count':tokens,'elapsed_decode_s':sum(t),'tok_s':tokens/sum(t),
 'median_ms':statistics.median(t)*1000,'p90_ms':sorted(t)[int(.9*len(t))]*1000,
 'first_decoded_top1':int(np.argmax(np.load(ROOT/f'FP32_COMPILED_STATIC_{int(compile_it)}_FIRST_LOGITS.npy'))),
 'full_wall_s':time.perf_counter()-start,'dtype':'fp32','compiled':compile_it,'max_context':max_context,
 'generated_ids':list(map(int,generated[:-1])),
 'model':'F10_Kronecker_repeated_weights_contracted_to_base, exact FP32 structure not independent 4B',
 'kv_cache':f'static{max_context}','memory':D.usage()}
 out=ROOT/f'FACTORED_FP32_COMPILED_STATIC_{int(compile_it)}_{tokens}_CTX{max_context}.json'
 out.write_text(json.dumps(summary,indent=2))
 print('FINAL',json.dumps(summary),flush=True)
if __name__=='__main__':
 pa=argparse.ArgumentParser();pa.add_argument('--tokens',type=int,default=100);pa.add_argument('--no-compile',action='store_true');pa.add_argument('--max-context',type=int,default=128)
 z=pa.parse_args();run(z.tokens,not z.no_compile,z.max_context)

#!/usr/bin/env python3
"""FP32 independent STERA 4.63B dense equivalent, long context KV with 8K global / 256 local.
Uses independently resident RowMask23 tiles initialized deterministically from learned H80000.
Supports storage+compute up to 8192 context tokens; ACTUAL story/C capability still limited
by original 512-token training and 2048 learned absolute positional embeddings.
The code safely rejects positions >=2048; no false claims of validated 8K language skill.
"""
import argparse,json,time,statistics,math
from pathlib import Path
import numpy as np
import mlx.core as mx
import stera4b_wht_sparse_v3 as H
import stera_independent_fp32_variable_width_v23 as M
import stera4b_wht_splitk_transposed_v28 as F
import stera_long_context_attention_fp32_v19 as A
from stera4b_factored_fp32_v8 import TOK_PATH
R=Path(__file__).resolve().parent
class LongCompiled:
 def __init__(self,capacity=8192,kv_precision="fp32",position_mode="strict"):
  assert capacity in (512,2048,4096,8192,16384,32768)
  assert kv_precision in ("fp32","fp16")
  assert position_mode in ("strict","interpolate_experimental")
  self.position_mode=position_mode
  self.kv_precision=kv_precision
  KV_DTYPE=mx.float16 if kv_precision=="fp16" else mx.float32
  self.context=capacity
  self.m=F.repack_model(M.DirectIndependentModel(10))
  target=self.m
  target.matvec=lambda name,x:F.fast_matvec(target,name,x,1 if name=='embedding' else 8 if name.endswith('.proj') else 2 if name.endswith('.fc') else 1)
  self.windows=[capacity if l%2==0 else 256 for l in range(6)]
  self.k=tuple(mx.zeros((w,160,48),dtype=KV_DTYPE) for w in self.windows)
  self.v=tuple(mx.zeros((w,160,48),dtype=KV_DTYPE) for w in self.windows)
  m=self.m;idx_vocab=mx.arange(50304,dtype=mx.int32)<50257
  def ln(x,name):
    w=m.p[name+'.weight'];b=m.p[name+'.bias']
    z=x-mx.mean(x)
    return z*mx.rsqrt(mx.mean(z*z)+1e-5)*w+b
  def linear(name,x):
    return m.matvec(name,x)
  def step(token,t,kold,vold):
    co,sc,gout,gin,U,V=m.w['embedding']
    e=F.KEMB_PACKED(inputs=[co.reshape(-1),sc,gout,gin,m.map,token.reshape(1)],
       grid=(7680,1,1),threadgroup=(256,1,1),output_shapes=[(7680,)],output_dtypes=[mx.float32])[0]
    e=e+mx.take(U,token.reshape(1),axis=0)[0]@V.T
    if position_mode=="interpolate_experimental":
      # Approximate, UNTRAINED extension of learned absolute positional table:
      # map all indices across [0, capacity-1] to [0,2047].
      # Does NOT establish 32K learned semantic context.
      fpos=t.astype(mx.float32)*mx.array(2047.0/(capacity-1),mx.float32)
      lo=mx.floor(fpos).astype(mx.int32)
      hi=mx.minimum(lo+1,2047)
      alpha=fpos-mx.floor(fpos)
      pe=(1-alpha)*mx.take(m.p['position.weight'],lo.reshape(1),axis=0)[0]+alpha*mx.take(m.p['position.weight'],hi.reshape(1),axis=0)[0]
    else:
      pe=mx.take(m.p['position.weight'],mx.minimum(t,2047).reshape(1),axis=0)[0]
    x=e+pe
    kn=[];vn=[]
    for l in range(6):
      pre=f'layers.{l}.'
      z=ln(x,pre+'ln1')
      q=linear(pre+'q',z).reshape((160,48))
      k=linear(pre+'k',z).reshape((160,48))
      v=linear(pre+'v',z).reshape((160,48))
      cap=self.windows[l]
      # Local cache ring: no more than last 256 K/V pairs, order irrelevant to attention.
      where=mx.remainder(t,cap).reshape(1) if l%2 else t.reshape(1)
      K=mx.slice_update(kold[l],k.reshape((1,160,48)).astype(KV_DTYPE),where,axes=[0])
      V=mx.slice_update(vold[l],v.reshape((1,160,48)).astype(KV_DTYPE),where,axes=[0])
      kn.append(K);vn.append(V)
      nvalid=mx.minimum(t+1,cap)
      # MLX metal kernels require shapes static but nvalid can be an array.
      nch=(cap+127)//128
      pp=A.KPART(inputs=[q,K,V,nvalid.reshape(1)],
           grid=(160*nch*256,1,1),threadgroup=(256,1,1),
           output_shapes=[(160*nch,50)],output_dtypes=[mx.float32])[0]
      yy=A.KRED(inputs=[pp],grid=(160*64,1,1),threadgroup=(64,1,1),
           output_shapes=[(160,48)],output_dtypes=[mx.float32])[0]
      x=x+linear(pre+'o',yy.reshape((7680,)))
      y=linear(pre+'fc',ln(x,pre+'ln2'))
      y=.5*y*(1+mx.tanh(.7978845608028654*(y+.044715*y*y*y)))
      x=x+linear(pre+'proj',y)
    logits=linear('embedding',ln(x,'final_norm'))/10
    return mx.where(idx_vocab,logits,-1e9),tuple(kn),tuple(vn)
  self.fn=mx.compile(step)
  self.kv_bytes=sum(k.nbytes+v.nbytes for k,v in zip(self.k,self.v))
  print('LONG_CTX_INIT','capacity',capacity,'KV_bytes',self.kv_bytes,'weights_fp32_bytes',self.m.weight_bytes,'MLX',H.S.get_mem(),flush=True)
 def next(self,token,t):
   if t>=self.context:raise ValueError('Requested context exceeds KV capacity')
   if t>=2048 and self.position_mode=='strict':raise ValueError('Untrained beyond absolute positions 2048; pass explicit interpolate_experimental mode if doing mechanical stress only')
   logits,k,v=self.fn(token,mx.array(t,mx.int32),self.k,self.v)
   self.k,self.v=k,v
   return logits
def bench(capacity=8192,tokens=16):
 from tokenizers import Tokenizer
 tok=Tokenizer.from_file(str(TOK_PATH))
 ids=[50256]+tok.encode('Once upon a time, there was a little cat named Mimi.',add_special_tokens=False).ids
 ids=ids[:3]
 m=LongCompiled(capacity)
 times=[];out=[];next_id=mx.array(ids[0],mx.int32)
 for t in range(len(ids)+tokens):
   x=mx.array(ids[t],mx.int32) if t<len(ids) else next_id
   st=time.perf_counter()
   logits=m.next(x,t)
   next_id=mx.argmax(logits).astype(mx.int32)
   mx.eval(next_id)
   sec=time.perf_counter()-st
   times.append(sec)
   if t>=len(ids)-1:out.append(int(next_id))
   if t==len(ids)-1:np.save(R/f'LONGCTX_FIRST_LOGITS_CAP{capacity}.npy',np.array(logits))
   if t%10==0:print('LONG_CTX_TOKEN',t,'sec',round(sec,4),'memory',H.S.get_mem(),flush=True)
 data={'context_capacity':capacity,'context_trained':512,'positional_embedding_rows':2048,
      'full_attention_layers':3,'local_attention_layers':3,'local_window':256,
      'KV_cache_bytes':m.kv_bytes,'stored_model_fp32_bytes':m.m.weight_bytes,
      'decode_tokens':tokens,'decode_tok_s':tokens/sum(times[3:]),
      'median_ms':1000*statistics.median(times[3:]),'output_token_ids':out[:tokens],
      'precompiled_first_step_seconds':times[0],
      'mem':H.S.get_mem(),
      'important':'Only capacity (not true 8K learned ability) verified; actual prefill was 3 tokens, greedy generate small number to preserve learned position coverage.'}
 (R/f'STERA_LONGCTX_FP32_CAP{capacity}_DECODE{tokens}.json').write_text(json.dumps(data,indent=2))
 print('LONGCTX_SUMMARY',json.dumps({k:v for k,v in data.items() if k!='output_token_ids'}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--capacity',type=int,default=8192);p.add_argument('--tokens',type=int,default=16)
 x=p.parse_args();bench(x.capacity,x.tokens)

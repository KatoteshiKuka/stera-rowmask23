#!/usr/bin/env python3
"""STERA independent RowMask23 FP32 prompt block prefill.
All params are distinct. Matrix-by-matrix 2-token/tile Metal. Causal attention
is exact conceptually; global KV cache and local ring window preserved.
First validate against the original sequential FP32 STERA decoder with SAME
weights. No model re-training or quantization.
"""
import argparse,json,time,statistics
from pathlib import Path
import numpy as np
import mlx.core as mx
from tokenizers import Tokenizer
from stera_independent_fp32_longctx_v47 import LongCompiled,TOK_PATH
import stera4b_wht_splitk_transposed_v28 as K
import stera_batch_rowmask_fp32_v38 as B
R=Path(__file__).resolve().parent
class BatchedPrefill:
 def __init__(self,capacity=2048,factor=10,BLOCK=8,split=2,position_mode="strict",kv_precision="fp32",shared_base=None):
  self.base=shared_base
  assert shared_base is not None and kv_precision=="fp32"
  self.m=self.base.m
  self.batch=BLOCK;self.split=split;self.capacity=capacity;self.position_mode=position_mode
  KVTYPE=mx.float16 if kv_precision=="fp16" else mx.float32
  self.kv_precision=kv_precision
  m=self.m;BS=BLOCK
  # Create packed embedding B-row fast kernel from the SAME code as single-row
  src=K.KEMB_PACKED
  # Independent implementation: multiple separate row embedding calls composed
  # into a compiled graph. This avoids allocating 50304*768 dense embedding.
  def embed(ids):
   co,sc,gout,gin,U,V=m.w['embedding']
   embs=[]
   for i in range(BS):
    token=mx.take(ids,mx.array([i],mx.int32),axis=0)
    el=K.KEMB_PACKED(inputs=[co.reshape(-1),sc,gout,gin,m.map,token],
       grid=(768,1,1),threadgroup=(256,1,1),
       output_shapes=[(768,)],output_dtypes=[mx.float32])[0]
    embs.append(el)
   e=mx.stack(embs,axis=0)
   return e+mx.take(U,ids,axis=0)@V.T
  def norm(x,prefix):
    mu=mx.mean(x,axis=-1,keepdims=True)
    d=x-mu
    return d*mx.rsqrt(mx.mean(d*d,axis=-1,keepdims=True)+1e-5)*m.p[prefix+'.weight'][None,:]+m.p[prefix+'.bias'][None,:]
  split_list=lambda name:1 if name=='embedding' else 1
  def linear(name,x):return B.packed_matmul(m,name,x,split_list(name))
  gpos=mx.arange(capacity,dtype=mx.int32)
  lpos=mx.arange(256,dtype=mx.int32)
  bpos=mx.arange(BS,dtype=mx.int32)
  def step(ids,t,oldK,oldV,last=False):
    ts=t+bpos
    if position_mode=="interpolate_experimental":
      fp=ts.astype(mx.float32)*(2047.0/(capacity-1))
      low=mx.floor(fp).astype(mx.int32);high=mx.minimum(low+1,2047)
      a=(fp-mx.floor(fp))[:,None]
      pos=(1-a)*mx.take(m.p['position.weight'],low,axis=0)+a*mx.take(m.p['position.weight'],high,axis=0)
    else:
      pos=mx.take(m.p['position.weight'],ts,axis=0)
    x=embed(ids)+pos
    nk=[];nv=[]
    for l in range(6):
      n=f'layers.{l}.'
      z=norm(x,n+'ln1')
      q=linear(n+'q',z).reshape((BS,16,48))
      k=linear(n+'k',z).reshape((BS,16,48))
      v=linear(n+'v',z).reshape((BS,16,48))
      ks=k.astype(KVTYPE);vs=v.astype(KVTYPE)
      newK=mx.slice_update(oldK[l],ks,t.reshape((1,)),axes=[0])
      newV=mx.slice_update(oldV[l],vs,t.reshape((1,)),axes=[0])
      allK=newK;allV=newV
      mask=gpos[None,:]<=ts[:,None]
      if l%2:mask=mask&(gpos[None,:]>ts[:,None]-256)
      nk.append(newK);nv.append(newV)
      # Uses MLX SDPA to avoid materializing the huge [heads,B,context] score
      # matrix as separate score/softmax tensors on the Python graph.
      # This path is checked numerically against the sequential reference.
      causal_bias=mx.where(mask, mx.array(0.0,mx.float32),mx.array(-1e9,mx.float32)).astype(KVTYPE)[None,None,:,:]
      attended=mx.fast.scaled_dot_product_attention(q.transpose(1,0,2)[None].astype(KVTYPE),
         allK.transpose(1,0,2)[None],allV.transpose(1,0,2)[None],
         scale=1.0,mask=causal_bias)[0].astype(mx.float32).transpose(1,0,2).reshape((BS,768))
      x=x+linear(n+'o',attended)
      y=linear(n+'fc',norm(x,n+'ln2'))
      y=.5*y*(1+mx.tanh(.7978845608028654*(y+.044715*y*y*y)))
      x=x+linear(n+'proj',y)
    # Only last token is relevant to continue greedy generation; the logits
    # for earlier prefill positions are unnecessary.
    logits=m.matvec('embedding',norm(x,'final_norm')[-1])
    logits=mx.where(mx.arange(50304,dtype=mx.int32)<50257,logits,-1e9)
    return logits,tuple(nk),tuple(nv)
  self.compiled=mx.compile(step)
 def process(self,ids,t):
  assert len(ids)==self.batch and t+self.batch<=self.capacity and (t%256)+self.batch<=256
  assert t+self.batch<=2048 or self.position_mode=='interpolate_experimental'
  out,ks,vs=self.compiled(mx.array(ids,mx.int32),mx.array(t,mx.int32),self.base.k,self.base.v)
  self.base.k=ks;self.base.v=vs
  return out

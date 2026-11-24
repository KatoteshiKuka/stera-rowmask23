#!/usr/bin/env python3
"""Compiles a whole FP32 RowMask23/Walsh-Hadamard STERA decoder step from H80000.
No materialization of dense weights, only learned FP32 coordinates and WHT bases resident.
NO cross-tile sharing. 898,109,280 separately stored FP32 coordinates, initialized from H80000 weights but not independently trained. Dense equivalent 4.63B.
"""
import argparse,time,json,statistics
import numpy as np
from pathlib import Path
import mlx.core as mx
import stera4b_wht_sparse_v3 as H
import stera_independent_fp32_variable_width_v23 as D
import stera4b_wht_splitk_transposed_v28 as KM
from stera4b_factored_fp32_v8 import TOK_PATH
ROOT=Path(__file__).resolve().parent
class CompressedCompiled:
 def __init__(self,context=128,factor=4):
  self.m=KM.repack_model(D.DirectIndependentModel(factor))
  self.factor=factor
  self.width=768*factor
  self.heads=16*factor
  target=self.m
  target.matvec=lambda name,x: KM.fast_matvec(target,name,x,
        1 if name=='embedding' else 8 if name.endswith('.proj') else 2 if name.endswith('.fc') else 1)
  self.context=context
  self.k=tuple(mx.zeros((context,self.heads,48),mx.float32) for _ in range(6))
  self.v=tuple(mx.zeros((context,self.heads,48),mx.float32) for _ in range(6))
  m=self.m
  position_range=mx.arange(context,dtype=mx.int32)
  mask_vocab=mx.arange(50304,dtype=mx.int32)<50257
  def ln(x,name):
   w=m.p[name+'.weight'];b=m.p[name+'.bias']
   mu=mx.mean(x);d=x-mu
   return d*mx.rsqrt(mx.mean(d*d)+1e-5)*w+b
  def matvec(name,x):
   return m.matvec(name,x)
  def st(token,t,oldk,oldv):
   co,sc,gout,gin,U,V=m.w['embedding']
   e=KM.KEMB_PACKED(inputs=[co.reshape(-1),sc,gout,gin,m.map,token.reshape((1,))],
             grid=(self.width,1,1),threadgroup=(256,1,1),
             output_shapes=[(self.width,)],output_dtypes=[mx.float32])[0]
   e=e+mx.take(U,token.reshape((1,)),axis=0)[0]@V.T
   x=e+mx.take(m.p['position.weight'],t.reshape((1,)),axis=0)[0]
   nk=[];nv=[]
   for l in range(6):
    pre=f'layers.{l}.'
    z=ln(x,pre+'ln1')
    q=matvec(pre+'q',z).reshape((self.heads,48))
    k=matvec(pre+'k',z).reshape((self.heads,48))
    v=matvec(pre+'v',z).reshape((self.heads,48))
    K=mx.slice_update(oldk[l],k.reshape((1,self.heads,48)),t.reshape((1,)),axes=[0])
    V=mx.slice_update(oldv[l],v.reshape((1,self.heads,48)),t.reshape((1,)),axes=[0])
    nk.append(K);nv.append(V)
    att=mx.sum(K*q[None],axis=-1)
    good=(position_range<=t)
    if l%2:good=good&(position_range>t-256)
    att=mx.where(good[:,None],att,-1e9)
    a=mx.sum(V*mx.softmax(att,axis=0)[:,:,None],axis=0).reshape((self.width,))
    x=x+matvec(pre+'o',a)
    y=matvec(pre+'fc',ln(x,pre+'ln2'))
    y=.5*y*(1+mx.tanh(.7978845608028654*(y+.044715*y*y*y)))
    x=x+matvec(pre+'proj',y)
   logits=matvec('embedding',ln(x,'final_norm'))/self.factor
   return mx.where(mask_vocab,logits,-1e9),tuple(nk),tuple(nv)
  self.compiled=mx.compile(st)
  print('COMPACT_COORD_FP32_INIT','resident_pre_warmup',H.S.get_mem(),flush=True)
 def step(self,token,t):
  l,k,v=self.compiled(token,mx.array(t,mx.int32),self.k,self.v)
  self.k,self.v=k,v
  return l
def run(count=100,ctx=128,factor=4):
 from tokenizers import Tokenizer
 ids=[50256]+Tokenizer.from_file(str(TOK_PATH)).encode('Once upon a time, there was a little cat named Mimi.',add_special_tokens=False).ids
 ids=ids[:3]
 assert count+len(ids)<=ctx
 m=CompressedCompiled(ctx,factor)
 tms=[];out=[];nxt=mx.array(ids[0],mx.int32)
 for t in range(count+len(ids)):
  inp=mx.array(ids[t],mx.int32) if t<len(ids) else nxt
  st=time.perf_counter()
  logits=m.step(inp,t)
  nxt=mx.argmax(logits).astype(mx.int32)
  mx.eval(nxt)
  if t>=len(ids)-1:out.append(nxt)
  delta=time.perf_counter()-st
  tms.append(delta)
  if t==2:np.save(ROOT/f'FAIR_F{m.factor}_STERA_V28_FIRST_LOGITS.npy',np.array(logits))
  if t%20==0:print('STEP',t,'ms',round(delta*1000,2),'RAM',H.S.get_mem(),flush=True)
 dr=tms[3:]
 j={'tokens':count,'ctx':ctx,'factor':factor,'fp32':True,'live_compressed_coords':True,'independent_coordinate_slots':m.m.unique_coord_count,
    'tps':count/sum(dr),'median_ms':statistics.median(dr)*1000,'p90_ms':sorted(dr)[int(.9*len(dr))]*1000,
    'generated_ids':[int(x) for x in out[:-1]],'memory':H.S.get_mem(),
    'true_dense_equivalent_elements':H.S.theoretical(factor)[0],'limitation':'Coordinate slots differ per tile and originate from learned H80000; NOT trained independently at expanded scale.'}
 (ROOT/f'FAIR_F{factor}_STERA_V28_FP32_{count}_CTX{ctx}.json').write_text(json.dumps(j,indent=2))
 print('FINAL',json.dumps({k:v for k,v in j.items() if k!='generated_ids'}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tokens',type=int,default=100);p.add_argument('--context',type=int,default=128);p.add_argument('--factor',type=int,default=4)
 a=p.parse_args();run(a.tokens,a.context,a.factor)

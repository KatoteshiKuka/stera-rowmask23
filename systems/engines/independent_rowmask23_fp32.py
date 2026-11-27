#!/usr/bin/env python3
"""Independent STERA RowMask23 coordinate slots at 4.63B dense-equivalent scale.
Every one of the expanded 128x128 blocks has a separate FP32 coordinate vector.
Initialize each vector deterministically from PRETRAINED H80000, with a small
tile-and-coordinate-dependent learned-weight perturbation. NO RNG and no repeated
blocks. No training of the expanded slots has occurred: these are independently
stored/trainable slots, not independently learned parameters. Do not claim quality
of trained 4.63B model.
"""
import gc,math,statistics,time,json,hashlib,argparse
from pathlib import Path
import numpy as np, mlx.core as mx
import stera4b_wht_sparse_v3 as H
import stera4b_stream_mlx as S
R=Path(__file__).resolve().parent
class IndependentModel(H.WhtModel):
 def __init__(self, f=10, amplitude=0.009):
  assert f==10
  super().__init__(f)
  self.amplitude=amplitude
  self.sourcehash=hashlib.sha256(S.NPZ.read_bytes()).hexdigest()
  total=0; snapshots={}
  for name,(co,sc,go,gi,U,V) in list(self.w.items()):
   n,c=co.shape
   ti=mx.arange(n,dtype=mx.float32)[:,None]
   ci=mx.arange(c,dtype=mx.float32)[None,:]
   # Learned-template-derived, non-repeated coordinates; no random values.
   # Perturbation has unique tile-and-coordinate phase and is not a block
   # scalar gain which the optimizer might algebraically factor out.
   shift37=mx.roll(co,37,axis=1)
   shift151=mx.roll(co,151,axis=1)
   coef=(0.56*mx.sin(ti*0.61803398875+ci*0.017)
          +0.44*mx.cos(ti*0.41421356237+ci*0.049))
   unique=co+mx.array(amplitude,dtype=mx.float32)*coef*(shift37+0.25*shift151)
   mx.eval(unique)
   if name=='layers.0.q' or name=='layers.0.fc':
     # Tests a base-tile at two different outer-block positions, same
     # legacy template offset. Different values must actually be resident.
     rt,ct=(768//128,768//128) if name.endswith('.q') else (3072//128,768//128)
     ct_big=ct*10
     orig0=0
     orig1=rt*ct_big
     row_a=np.array(unique[orig0],dtype=np.float32)
     row_b=np.array(unique[orig1],dtype=np.float32)
     maxd=float(np.max(np.abs(row_a-row_b)))
     assert maxd>1e-7,(name,maxd)
     snapshots[name]={'block0_versus_block10_max_abs_coordinate_difference':maxd,
                      'n_tiles':int(n),
                      'coordinate_count':int(n*c)}
   self.w[name]=(unique,sc,go,gi,U,V)
   del co,shift37,shift151,coef,ti,ci
   newcow=H.fwht(mx.concatenate([mx.zeros((n,1),mx.float32),unique[:,128:255]],axis=1))
   mx.eval(newcow)
   self.cow[name]=newcow
   total+=int(n*c)
   gc.collect()
   if name=='embedding' or name.endswith('proj'):
    print('UNIQUE_LOADED',name,'tile_count',n,'coord_slots',n*c,'mem',S.get_mem(),flush=True)
  self.unique_coord_count=total
  self.provenance=snapshots
  self.weight_bytes=sum(x.nbytes for a in self.w.values() for x in a)+sum(x.nbytes for x in self.p.values())
  print('INDEPENDENT_COORDINATE_SLOTS',total,'WEIGHT_BYTES',self.weight_bytes,'MEM',S.get_mem(),flush=True)

def benchmark(model,mode='v3',ntokens=8):
  from tokenizers import Tokenizer
  m=model
  ids=[50256]+Tokenizer.from_file(str(S.P/'STERA_FINAL_H80000_LOSSLESS_FROZEN_20261122/tokenizer.json')).encode('Once upon a time, there was a little cat named Mimi.',add_special_tokens=False).ids
  # Short fixed prompt, but true teacher prefill and then greedy autoregressive feedback.
  ids=ids[:3];prefill=len(ids)
  times=[]
  for t in range(prefill+ntokens):
   st=time.perf_counter()
   lg=m.forward_one(ids[t],t)
   mx.eval(lg)
   nxt=int(mx.argmax(lg))
   sec=time.perf_counter()-st
   times.append(sec)
   if t>=prefill-1:ids.append(nxt)
   if t%3==0:print('ONE_TOKEN',t,'seconds',round(sec,3),'gpu',S.get_mem(),flush=True)
  r={'kind':'Independent parameter slots, learned-derived deterministic init, NOT trained at scale',
     'dtype':'float32','mode':mode,'factor':10,'dense_matrix_equiv':S.theoretical(10)[0],
     'independent_coordinate_slots':m.unique_coord_count,
     'separate_learned_template_checkpoint_sha':m.sourcehash,
     'unique_probe':m.provenance,'tokens':ntokens,'decoding_toks_per_second':ntokens/sum(times[prefill:]),
     'timings_sec':times,'generated_ids':ids[prefill:],
     'memory_post':S.get_mem(),'resident_bytes':m.weight_bytes}
  f=R/f'INDEPENDENT_4B_{mode}_{ntokens}.json';f.write_text(json.dumps(r,indent=2))
  print('INDEPENDENT_FINAL',json.dumps({k:v for k,v in r.items() if k not in ['generated_ids','timings_sec']}),flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--tokens',type=int,default=8);a.add_argument('--mode',default='v3')
 args=a.parse_args()
 m=IndependentModel()
 benchmark(m,args.mode,args.tokens)

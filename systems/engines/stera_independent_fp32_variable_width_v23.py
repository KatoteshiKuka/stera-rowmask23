#!/usr/bin/env python3
"""Direct GPU initialization of physically INDEPENDENT STERA coordinate tiles from learned H80000.
No random values, no cloned full 4B intermediate tensor, no cross-block sharing.
Each tile has its own FP32 coordinate vector; deterministic small perturbation from
the pretrained base coordinates and a shifted pretrained coordinate vector.
Coordinates are *not* yet independently trained. No FP16/INT8.
"""
import hashlib,gc,json
import numpy as np
import mlx.core as mx
import stera4b_stream_mlx as S
import stera4b_wht_sparse_v3 as H
from pathlib import Path
KINIT=mx.fast.metal_kernel(
 name='stera_unique_from_trained_tiles_fp32_direct_v16',
 input_names=['base','sizes'],
 output_names=['out'],
 source=r"""
 uint i=thread_position_in_grid.x;
 uint tile=i/3176u;
 uint ci=i%3176u;
 uint original_rt=uint(sizes[0]);
 uint original_ct=uint(sizes[1]);
 uint enlarged_ct=uint(sizes[2]);
 uint scale_f=uint(sizes[3]);
 uint tile_r=tile/enlarged_ct;
 uint tile_c=tile%enlarged_ct;
 uint oldtile=(tile_r%original_rt)*original_ct+(tile_c%original_ct);
 uint srcbase=oldtile*3176u;
 float x=base[srcbase+ci];
 float a=base[srcbase+(ci+37u)%3176u];
 float b=base[srcbase+(ci+151u)%3176u];
 float ph=float(tile)*0.61803398875f+float(ci)*0.017f;
 float ph2=float(tile)*0.41421356237f+float(ci)*0.049f;
 float coef=.56f*sin(ph)+.44f*cos(ph2);
 float z=(x+0.009f*coef*(a+0.25f*b))/float(scale_f);
 out[i]=z;
 """,compile_options={'math_mode':'fast'})
class DirectIndependentModel(S.CoordModel):
 def __init__(self,f=10):
  assert f in (1,2,3,4,5,6,8,10),f
  self.f=f;self.h=768*f;self.ffn=3072*f;self.nhead=16*f
  self.map=S.make_map()
  self.p={};self.w={};self.cow={};self.kv=[None]*6
  with np.load(S.NPZ,allow_pickle=False) as z:
    self.p['position.weight']=mx.array(np.tile(z['position.weight'],(1,f)))
    for n in ['final_norm']+[f'layers.{l}.{k}' for l in range(6) for k in ['ln1','ln2']]:
      for a in ['weight','bias']:
        self.p[n+'.'+a]=mx.array(np.tile(z[n+'.'+a],f))
    for l in range(6):
      for k in ['o','fc','proj']:
        self.p[f'layers.{l}.{k}.bias']=mx.array(np.tile(z[f'layers.{l}.{k}.bias'],f))
    for name,r,c in S.BASE:
      rf=1 if name=='embedding' else f
      rt=r//128;ct=c//128;ct_big=ct*f
      source=mx.array(z[name+'.coords'].reshape(-1,3176))
      target_tile_count=(rt*rf)*(ct*f)
      sizes=mx.array([rt,ct,ct_big,f if rf!=1 else 1],mx.int32)
      newco=KINIT(inputs=[source.reshape(-1),sizes],
         grid=(target_tile_count*3176,1,1),threadgroup=(256,1,1),
         output_shapes=[(target_tile_count,3176)],output_dtypes=[mx.float32])[0]
      mx.eval(newco)
      scales=mx.array(np.tile(z[name+'.scales'].reshape(rt,ct),(rf,f)).reshape(-1))
      gout=mx.array(np.tile(z[name+'.gout'],rf))
      gin=mx.array(np.tile(z[name+'.gin'],f))
      U=mx.array(np.tile(z[name+'.U'],(rf,1)))
      V=mx.array(np.tile(z[name+'.V'],(f,1)))
      if rf!=1:V=V/f
      self.w[name]=(newco,scales,gout,gin,U,V)
      bases=mx.concatenate([mx.zeros((target_tile_count,1),mx.float32),newco[:,128:255]],axis=1)
      self.cow[name]=H.fwht(bases)
      mx.eval(*self.w[name],self.cow[name])
      del source,newco,scales,gout,gin,U,V,bases
      gc.collect()
      if name=='embedding' or name.endswith('.proj'):
        print('DIRECT_MODEL_LOADED',name,'tiles',target_tile_count,'gpu',S.get_mem(),flush=True)
  mx.eval(*self.p.values())
  self.weight_bytes=sum(x.nbytes for parts in self.w.values() for x in parts)+sum(x.nbytes for x in self.p.values())
  self.unique_coord_count=sum(v[0].size for v in self.w.values())
  self.sourcehash=hashlib.sha256(S.NPZ.read_bytes()).hexdigest()
  self.provenance={}
  for name,rt,ct in ([('layers.0.q',6,6),('layers.0.fc',24,6)] if f>1 else []):
    co=self.w[name][0]
    ar=np.array(co[0]);br=np.array(co[rt*ct*f])
    diff=float(np.max(np.abs(ar-br)))
    assert diff>1e-7, (name,diff)
    self.provenance[name]={'block0_versus_block10_max_abs_coordinate_difference':diff,'n_tiles':co.shape[0],'coordinate_count':int(co.size)}
  print('DIRECT_MODEL_COMPLETE','n_coords',self.unique_coord_count,'fp32_weight_bytes',self.weight_bytes,'gpu',S.get_mem(),flush=True)
 def matvec(self,name,x):
    return H.WhtModel.matvec(self,name,x)

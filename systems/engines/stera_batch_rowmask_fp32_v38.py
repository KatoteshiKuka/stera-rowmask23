#!/usr/bin/env python3
"""FP32 RowMask23 matrix-by-matrix, processing B prompt tokens concurrently.
All FP32 coordinate tiles independent. Packed residuals 23x127; same weights as
v28 decoder. Tile-wise kernel amortizes WHT basis + residual reads across 8 tokens
in one Metal threadgroup. No dense materialization and no weight deduplication.
"""
from pathlib import Path
import statistics,time,json
import mlx.core as mx
import stera4b_wht_splitk_transposed_v28 as F
import stera4b_fwht_metal_v28 as MT
K=mx.fast.metal_kernel(
 name="stera_rm23_transposed_batched_8tokens_v38",
 input_names=['coords','scales','vec','vecw','cow','splitk'],
 output_names=['out'],
 source=r"""
 uint lid=thread_position_in_threadgroup.x;
 uint ri=lid%128u;
 uint localtok=lid/128u;
 uint groupx=threadgroup_position_in_grid.x;
 uint groupy=threadgroup_position_in_grid.y;
 uint T=uint(vec_shape[0]), N=uint(vec_shape[1]);
 uint splits=uint(splitk[0]);
 uint ct=N/128u;
 uint rb=groupx/splits, ck=groupx%splits;
 uint tok=groupy*8u+localtok;
 threadgroup float resid[2921];
 threadgroup float shv[1024];
 float accumulated=0.0f;
 for(uint cb=ck;cb<ct;cb+=splits){
    uint tile=rb*ct+cb, off=tile*3176u;
    for(uint j=lid;j<2921u;j+=1024u)resid[j]=coords[off+255u+j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s=0.0f, a=0.0f;
    if(tok<T){
       s=coords[off+ri]*vec[tok*N+cb*128u+ri];
       if(ri>0u){
          uint sh=1u+(7u*ri)%127u;
          for(uint k=0u;k<23u;k++){
             s+=resid[k*127u+ri-1u]*vec[tok*N+cb*128u+(ri^sh)];
             sh+=13u;
             if(sh>127u)sh-=127u;
          }
       }
       a=cow[tile*128u+ri]*vecw[tok*N+cb*128u+ri]*(1.0f/128.0f);
    }
    for(uint h=1u;h<=16u;h<<=1u){
       float b=simd_shuffle_xor(a,ushort(h));
       a=(ri&h)?(b-a):(a+b);
    }
    shv[lid]=a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float b=shv[localtok*128u+(ri^32u)];
    a=(ri&32u)?(b-a):(a+b);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    shv[lid]=a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    b=shv[localtok*128u+(ri^64u)];
    a=(ri&64u)?(b-a):(a+b);
    accumulated+=(s+a)*scales[tile];
    threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(tok<T){
   uint rows=(uint(coords_shape[0])/3176u/ct)*128u;
   out[tok*rows*splits+(rb*128u+ri)*splits+ck]=accumulated;
 }
 """,compile_options={'math_mode':'fast'})
def packed_matmul(m,name,x,split=2):
 assert x.ndim==2
 co,sc,gout,gin,U,V=m.w[name]
 B,N=x.shape
 rows=int(gout.size);ct=N//128
 assert B%8==0 and N%128==0
 vec=x*gin[None,:]
 # Compute all independent input WHT; no aliasing.
 wht=MT.fwht_metal(vec.reshape(-1)).reshape(B,N)
 part=K(inputs=[co.reshape(-1),sc,vec,wht,m.cow[name].reshape(-1),mx.array([split],mx.int32)],
    grid=(rows*split*8,B//8,1),threadgroup=(1024,1,1),
    output_shapes=[(B,rows,split)],output_dtypes=[mx.float32])[0]
 y=mx.sum(part,axis=2)*gout[None,:]+((x@V)@U.T)
 if name+'.bias' in m.p:y=y+m.p[name+'.bias'][None,:]
 return y
def verify(m,B=4):
 for name in ['layers.0.q','layers.0.fc','layers.0.proj','embedding']:
  size=int(m.w[name][3].size)
  x=mx.sin(mx.arange(size*B,dtype=mx.float32).reshape(B,size)*.011)
  mx.eval(x)
  ref=mx.stack([m.matvec(name,x[i]) for i in range(B)],axis=0)
  for split in [1,2,4]:
   z=packed_matmul(m,name,x,split)
   mx.eval(ref,z)
   maxerr=float(mx.max(mx.abs(ref-z)))
   rel=float(mx.sqrt(mx.sum((ref-z)**2)/(mx.sum(ref**2)+1e-30)))
   print('VERIFY',name,'B',B,'split',split,'maxerr',maxerr,'rel',rel,flush=True)
   if not (rel<1e-4 and maxerr<.06):raise ValueError('MATMUL_MISMATCH')
 return True

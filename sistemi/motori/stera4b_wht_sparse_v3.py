#!/usr/bin/env python3
"""Optimized native RowMask23: XOR convolution via Walsh-Hadamard, sparse 23/row residual.
Full matrix still evaluated across all (expanded) tiles. No reuse of repeated tiles.
Measured against earlier exact FP32 coordinate kernel with same weights.
For every tile W_ij = (diag_i*x_i + xorConv(coef_s,x) + sum_23 sparse_i * x_j)*scale.
Precompute WHT(base coefficients) once; WHT(input vector) once per matvec.
Metal each tile: 128 threads, SIMD shuffle first 5 butterfly stages, threadgroup 2 stages.
"""
import argparse,time,statistics,json
import mlx.core as mx
import stera4b_stream_mlx as S
from pathlib import Path
R=Path(__file__).resolve().parent
def fwht(x):
    nt=int(x.shape[0]); assert x.ndim==2 and x.shape[1]==128
    for h in (1,2,4,8,16,32,64):
        p=x.reshape(nt,128//(2*h),2,h)
        a=p[:,:,0,:];b=p[:,:,1,:]
        x=mx.stack((a+b,a-b),axis=2).reshape(nt,128)
    return x
KFW=mx.fast.metal_kernel(
 name="rowmask23_wht_sparse23_gemv_v1",
 input_names=['coords','scales','vec','vecw','cow'],
 output_names=['partial'],
 source=r"""
 uint li=thread_position_in_threadgroup.x;
 uint bi=threadgroup_position_in_grid.x;
 uint ct=uint(vec_shape[0])/128u;
 uint cb=bi%ct;
 uint ri=li;
 uint base=bi*3176u;
 threadgroup float residual[2921];
 for(uint j=li;j<2921u;j+=128u) residual[j]=coords[base+255u+j];
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float sp=coords[base+ri]*vec[cb*128u+ri];
 if(ri>0u){
  for(uint k=0;k<23u;k++){
   uint sh=1u+(13u*k+7u*ri)%127u;
   uint j=ri^sh;
   sp += residual[(ri-1u)*23u+k]*vec[cb*128u+j];
  }
 }
 // WHT^{-1}( WHT(c) * WHT(vec) ), where c_0=0, c_1..127 are shared XOR weights.
 float a=cow[bi*128u+ri]*vecw[cb*128u+ri]*(1.0f/128.0f);
 for(uint h=1u;h<=16u;h<<=1u){
  float b=simd_shuffle_xor(a,ushort(h));
  a=(ri & h)? (b-a) : (a+b);
 }
 threadgroup float shv[128];
 shv[ri]=a;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float b=shv[ri^32u];
 a=(ri&32u)?(b-a):(a+b);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 shv[ri]=a;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 b=shv[ri^64u];
 a=(ri&64u)?(b-a):(a+b);
 partial[(bi/ct)*128u*ct+ri*ct+cb]=(sp+a)*scales[bi];
 """, compile_options={'math_mode':'fast'})
class WhtModel(S.CoordModel):
 def __init__(self,f):
    super().__init__(f)
    self.cow={}
    for name,stuff in self.w.items():
        coords=stuff[0]
        bases=mx.concatenate([mx.zeros((coords.shape[0],1),mx.float32),coords[:,128:255]],axis=1)
        tmp=fwht(bases)
        self.cow[name]=tmp
        mx.eval(tmp)
    self.cow_bytes=sum(c.nbytes for c in self.cow.values())
    print('WHT_READY_BYTES',self.cow_bytes,'RAM',S.get_mem(),flush=True)
 def matvec(self,name,x):
    co,sc,gout,gin,U,V=self.w[name]
    rows=gout.size;cols=gin.size;ct=cols//128
    vec=x*gin
    vecw=fwht(vec.reshape(ct,128))
    partial=KFW(inputs=[co.reshape(-1),sc,vec,vecw.reshape(-1),self.cow[name].reshape(-1)],
       grid=(rows*ct,1,1),threadgroup=(128,1,1),output_shapes=[(rows,ct)],output_dtypes=[mx.float32])[0]
    y=mx.sum(partial,axis=1)*gout + U @ (V.T @ x)
    if name+'.bias' in self.p:y=y+self.p[name+'.bias']
    return y
def run(f,mode,tokens):
    if mode=='only-kernel':
        m=WhtModel(f)
        name='layers.0.fc';co,sc,gout,gin,U,V=m.w[name]
        x=mx.random.normal((gin.size,));mx.eval(x)
        v=x*gin;rows=gout.size;ct=gin.size//128
        expected=S.KROW(inputs=[co.reshape(-1),sc,v,m.map],grid=(rows*ct,1,1),threadgroup=(256,1,1),
             output_shapes=[(rows,ct)],output_dtypes=[mx.float32])[0]
        res=KFW(inputs=[co.reshape(-1),sc,v,fwht(v.reshape(ct,128)).reshape(-1),m.cow[name].reshape(-1)],
             grid=(rows*ct,1,1),threadgroup=(128,1,1),output_shapes=[(rows,ct)],output_dtypes=[mx.float32])[0]
        mx.eval(expected,res)
        err=float(mx.max(mx.abs(expected-res)))
        avg=float(mx.mean(mx.abs(expected-res)))
        print('WHT_PARTIAL_PARITY',f,'max',err,'mean',avg,flush=True)
        if err>0.001:raise ValueError(f'WHT parity failed {err}')
        for label,kernel,th in [('naive',S.KROW,256),('wht',KFW,128)]:
          times=[]
          for i in range(10):
             st=time.perf_counter()
             out=kernel(inputs=([co.reshape(-1),sc,v,m.map] if label=='naive'
                 else [co.reshape(-1),sc,v,fwht(v.reshape(ct,128)).reshape(-1),m.cow[name].reshape(-1)]),
                  grid=(rows*ct,1,1),threadgroup=(th,1,1),
                  output_shapes=[(rows,ct)],output_dtypes=[mx.float32])[0]
             mx.eval(out)
             times.append(time.perf_counter()-st)
          print('KERNEL_TIME',label,'F',f,'median_s',round(statistics.median(times),6),'times',times,flush=True)
    else:
        S.CoordModel=WhtModel
        # Calling run uses self-contained WhtModel, labels speed test exactly F, no shell tee in subprocess.
        S.run(f,tokens)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument('--factor',type=int,default=1)
    ap.add_argument('--tokens',type=int,default=3)
    ap.add_argument('--only-kernel',action='store_true')
    a=ap.parse_args()
    run(a.factor,'only-kernel' if a.only_kernel else 'full',a.tokens)

#!/usr/bin/env python3
"""STERA long-context FP32 exact attention: split tiles of 128 KV positions.
160 independent attention heads, head_dim48, causal decode, no attention score
matrix saved; one partial (m,denom,weighted V) per head and tile. Kernel works
for full global contexts or local ring buffer of 256 tokens. No model weights
or random model parameters are created; this unit benchmark uses deterministic
synthetic K/V activations, not a fake 'trained 8K context' model.
"""
import time,math,statistics,json,argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
ROOT=Path(__file__).resolve().parent
ATTN_TILE=128
KPART=mx.fast.metal_kernel(
 name="stera_attn_f32_partial_128_v19",
 input_names=["q","keys","values","nv"],
 output_names=["partial"],
 source=r"""
 uint li=thread_position_in_threadgroup.x;
 uint wid=li>>5u, lane=li&31u;
 uint bi=threadgroup_position_in_grid.x;
 uint nChunks=(uint(keys_shape[0])+127u)/128u;
 uint h=bi/nChunks, tile=bi%nChunks;
 uint maxvalid=uint(nv[0]), start=tile*128u;
 threadgroup float score[128];
 threadgroup float weights[128];
 threadgroup float meanv;
 threadgroup float normv;
 for(uint p=wid;p<128u;p+=8u){
   uint at=start+p;
   float s=-1.0e30f;
   if(at<maxvalid){
      uint qo=h*48u, ko=(at*160u+h)*48u;
      float a=q[qo+lane]*keys[ko+lane];
      if(lane<16u)a+=q[qo+lane+32u]*keys[ko+lane+32u];
      s=simd_sum(a);
   }
   if(lane==0u)score[p]=s;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(li==0u){
   float m=-1.0e30f;
   for(uint p=0u;p<128u;p++)m=max(m,score[p]);
   float z=0.0f;
   for(uint p=0u;p<128u;p++){
      float v=exp(score[p]-m);
      weights[p]=v;
      z+=v;
   }
   meanv=m;normv=z;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(li==0u){
   partial[bi*50u]=meanv;
   partial[bi*50u+1u]=normv;
 }
 if(li<48u){
   float accum=0;
   for(uint p=0u;p<128u;p++){
     uint at=start+p;
     if(at<maxvalid)accum += weights[p]*values[(at*160u+h)*48u+li];
   }
   partial[bi*50u+2u+li]=accum;
 }
 """,compile_options={"math_mode":"fast"})
KRED=mx.fast.metal_kernel(
 name="stera_attn_f32_merge_128_v19",
 input_names=["partial"],
 output_names=["result"],
 source=r"""
 uint d=thread_position_in_threadgroup.x;
 uint h=threadgroup_position_in_grid.x;
 uint chunks=uint(partial_shape[0])/160u;
 threadgroup float maximum;
 threadgroup float total;
 if(d==0u){
    float m=-1.0e30f;
    for(uint c=0u;c<chunks;c++)m=max(m,partial[(h*chunks+c)*50u]);
    float z=0;
    for(uint c=0u;c<chunks;c++){
      uint at=(h*chunks+c)*50u;
      z+=exp(partial[at]-m)*partial[at+1u];
    }
    maximum=m;total=z;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(d<48u){
    float value=0.0f;
    for(uint c=0;c<chunks;c++){
      uint at=(h*chunks+c)*50u;
      value+=exp(partial[at]-maximum)*partial[at+2u+d];
    }
    result[h*48u+d]=value/total;
 }
 """,compile_options={"math_mode":"fast"})
def attention(q,k,v,nvalid):
    n=k.shape[0]
    nchunk=(n+127)//128
    # Unfilled future rows are masked in kernel using nvalid.
    p=KPART(inputs=[q,k,v,mx.array([nvalid],mx.int32)],
       grid=(160*nchunk*256,1,1),threadgroup=(256,1,1),
       output_shapes=[(160*nchunk,50)],output_dtypes=[mx.float32])[0]
    y=KRED(inputs=[p],grid=(160*64,1,1),threadgroup=(64,1,1),
       output_shapes=[(160,48)],output_dtypes=[mx.float32])[0]
    return y,p
def src(n,head=160,dim=48):
    # Deterministic activation test vectors (NOT MODEL WEIGHTS).
    i=mx.arange(n,dtype=mx.float32).reshape(n,1,1)
    h=mx.arange(head,dtype=mx.float32).reshape(1,head,1)
    d=mx.arange(dim,dtype=mx.float32).reshape(1,1,dim)
    q=mx.sin(mx.arange(head*dim,dtype=mx.float32).reshape(head,dim)*0.017)
    k=mx.sin(i*0.0041+h*0.027+d*.091)*.28
    v=mx.cos(i*.0062+h*.033+d*.059)*.7
    mx.eval(q,k,v)
    return q,k,v
def ref(q,k,v,nvalid):
    kk=k[:nvalid].transpose(1,0,2)
    vv=v[:nvalid].transpose(1,0,2)
    scores=mx.sum(kk*q[:,None,:],axis=-1)
    att=mx.softmax(scores,axis=-1)
    return mx.sum(vv*att[:,:,None],axis=1)
def timed(n,rep=6,fullref=True):
    q,k,v=src(n);r={}
    for seen in [min(n,127),n]:
      pred,p=attention(q,k,v,seen)
      mx.eval(pred)
      if fullref and n<=8192:
        expected=ref(q,k,v,seen);mx.eval(expected)
        maxerr=float(mx.max(mx.abs(pred-expected)))
        rerr=float(mx.sqrt(mx.sum((pred-expected)**2)/(mx.sum(expected**2)+1e-9)))
        print('PARITY','capacity',n,'seen',seen,'max_abs',maxerr,'relative_l2',rerr,flush=True)
        assert rerr<1e-4,(n,seen,maxerr,rerr)
      times=[]
      for j in range(rep):
         st=time.perf_counter()
         out,_=attention(q,k,v,seen)
         mx.eval(out)
         times.append(1000*(time.perf_counter()-st))
      r[str(seen)]={'median_ms':statistics.median(times),'min_ms':min(times),
                    'max_ms':max(times),'partial_bytes':p.nbytes,
                    'kv_bytes':k.nbytes+v.nbytes}
      print('TIMED',n,seen,r[str(seen)],flush=True)
    return r
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--contexts",default="256,2048,8192")
    a=ap.parse_args()
    report={}
    for n in map(int,a.contexts.split(',')):
        print('START',n,flush=True)
        report[str(n)]=timed(n)
    (ROOT/"STERA_LONG_CONTEXT_ATTN_FP32_V19.json").write_text(json.dumps(report,indent=2))
    print('DONE',json.dumps(report),flush=True)

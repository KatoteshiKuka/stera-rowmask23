#!/usr/bin/env python3
"""FP32 native Metal RowMask23 fused split-K tile-loop, all tiles independent.
One independent row-block/split chunk task accumulates multiple unique column
tiles in-register, eliminating giant intermediate partial matrix and its
separate output reduction. Uses precomputed WHT basis separately FOR EVERY TILE,
23 sparse residual per row, plus learned rank96 correction (unchanged).
NO cross-tile sharing, NO Kronecker contraction, no fp16/int8.
"""
import argparse,statistics,time,json
from pathlib import Path
import mlx.core as mx
import stera4b_wht_sparse_v3 as H
import independent_rowmask23_fp32 as I
import gc
import stera4b_fwht_metal_v28 as MT
ROOT=Path(__file__).resolve().parent
KSPLIT=mx.fast.metal_kernel(
 name='stera_rowmask23_wht_independent_splitk_v28_metal_input',
 input_names=['coords','scales','vec','vecw','cow','splitk'],
 output_names=['out'],
 source=r"""
 uint ri=thread_position_in_threadgroup.x;
 uint group=threadgroup_position_in_grid.x;
 uint splits=uint(splitk[0]);
 uint ct=uint(vec_shape[0])/128u;
 uint rb=group/splits;
 uint ck=group%splits;
 threadgroup float shv[128];
 float accumulated=0.0f;
 for(uint cb=ck;cb<ct;cb+=splits){
    uint tile=rb*ct+cb;
    uint off=tile*3176u;
    float s=coords[off+ri]*vec[cb*128u+ri];
    if(ri>0u){
       uint sh=1u+(7u*ri)%127u;
       for(uint k=0u;k<23u;k++){
           s += coords[off+255u+k*127u+(ri-1u)]*vec[cb*128u+(ri^sh)];
           sh+=13u;
           if(sh>127u)sh-=127u;
       }
    }
    float a=cow[tile*128u+ri]*vecw[cb*128u+ri]*(1.0f/128.0f);
    for(uint h=1u;h<=16u;h<<=1u){
       float b=simd_shuffle_xor(a,ushort(h));
       a=(ri & h)? (b-a):(a+b);
    }
    shv[ri]=a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float b=shv[ri^32u];
    a=(ri&32u)?(b-a):(a+b);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    shv[ri]=a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    b=shv[ri^64u];
    a=(ri&64u)?(b-a):(a+b);
    accumulated+=(s+a)*scales[tile];
    threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 out[(rb*128u+ri)*splits+ck]=accumulated;
 """,compile_options={'math_mode':'fast'})
def fast_matvec(m,name,x,split=8):
 co,sc,gout,gin,U,V=m.w[name]
 cols=int(gin.size);rows=int(gout.size)
 assert cols%128==0 and rows%128==0
 vec=x*gin
 ct=cols//128
 xw=MT.fwht_metal(vec)
 split_tensor=mx.array([split],mx.int32)
 part=KSPLIT(inputs=[co.reshape(-1),sc,vec,xw,m.cow[name].reshape(-1),split_tensor],
      grid=(rows*split,1,1),threadgroup=(128,1,1),
      output_shapes=[(rows,split)],output_dtypes=[mx.float32])[0]
 y=mx.sum(part,axis=1)*gout+U@(V.T@x)
 if name+'.bias' in m.p:y=y+m.p[name+'.bias']
 return y

def microbench(m):
 results={};modes=[1,2,4,8,16,32]
 for name in ['layers.0.q','layers.0.fc','layers.0.proj','embedding']:
  gi=m.w[name][3];x=mx.sin(mx.arange(int(gi.size),dtype=mx.float32)*0.013)
  mx.eval(x)
  baseline=H.WhtModel.matvec(m,name,x)
  mx.eval(baseline)
  cur={}
  for mode in modes:
    try:
      y=fast_matvec(m,name,x,mode);mx.eval(y)
      err=float(mx.max(mx.abs(baseline-y)))
      rel=float(mx.sqrt(mx.sum((baseline-y)**2)/(mx.sum(baseline**2)+1e-20)))
      assert err<0.05 and rel<0.0001,('PARITY_FAIL',name,mode,err,rel)
      t=[]
      for i in range(6):
       start=time.perf_counter()
       yy=fast_matvec(m,name,x,mode);mx.eval(yy)
       t.append(1000*(time.perf_counter()-start))
      cur[str(mode)]={'median_ms':statistics.median(t),'max_abs':err,'rel_l2':rel}
      print('MICRO',name,'split',mode,'median_ms',round(statistics.median(t),3),'maxerr',round(err,7),'relative',round(rel,9),flush=True)
    except Exception as e:
      print('MICRO_ERROR',name,'split',mode,str(e),flush=True)
      cur[str(mode)]={'error':str(e)}
  results[name]=cur
 (ROOT/'INDEPENDENT_SPLITK_MICRO_V15.json').write_text(json.dumps(results,indent=2))
 return results

def full(m,splits=8,n=15):
 m.matvec=lambda name,x:fast_matvec(m,name,x,splits)
 I.benchmark(m,f'v15_split_{splits}',n)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['micro','full'],default='micro')
 p.add_argument('--split',type=int,default=8);p.add_argument('--tokens',type=int,default=15)
 a=p.parse_args()
 m=I.IndependentModel()
 if a.mode=='micro':microbench(m)
 else:full(m,a.split,a.tokens)


def repack_model(m):
    """Permute each tile's 2921 FP32 residual coordinates from [row,23] to [23,row].
    No duplicate/reused values or precision conversion; only layout modification.
    Save original WHT base because first 255 coords stay exactly unchanged."""
    for name in list(m.w):
        co,sc,go,gi,U,V=m.w[name]
        n=int(co.shape[0])
        p=mx.concatenate((co[:,:255],co[:,255:].reshape(n,127,23).transpose(0,2,1).reshape(n,2921)),axis=1)
        mx.eval(p)
        m.w[name]=(p,sc,go,gi,U,V)
        del co,p
        gc.collect()
        mx.clear_cache()
    return m


KEMB_PACKED=mx.fast.metal_kernel(
 name='stera_rowmask23_embedding_row_transposed_v25',
 input_names=['coords','scales','gout','gin','rid','tok'],
 output_names=['embedded'],
 source=r"""
 uint j=thread_position_in_grid.x;
 uint row=uint(tok[0]);
 uint ct=uint(gin_shape[0])/128u;
 uint tile=(row>>7u)*ct+(j>>7u);
 uint ri=row&127u, cj=j&127u, sh=ri^cj;
 uint baseidx=(sh==0u)?ri:(127u+sh);
 int old_rx=rid[ri*128u+sh];
 float w=coords[tile*3176u+baseidx];
 if(old_rx>=255){
    uint loc=uint(old_rx-255);
    uint new_rx=255u+(loc%23u)*127u+(loc/23u);
    w+=coords[tile*3176u+new_rx];
 }
 embedded[j]=w*scales[tile]*gout[row]*gin[j];
 """,compile_options={'math_mode':'fast'})

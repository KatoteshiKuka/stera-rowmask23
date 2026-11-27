#!/usr/bin/env python3
"""Real Metal GPU inference over RowMask23 coordinates, expanded deterministically from the frozen STERA H80000 checkpoint.
Widening by F copies is function-preserving with W blocks = W_original/F and repeated learned weights.
Physical 4.63B dense-equivalent elements for F=10. This is NOT a newly trained 4.63B model.
All input/output linear operators process the compressed RowMask23 coordinates with custom
Metal kernels and never allocate full dense matrices; one 128x128 tile reconstructed in registers
per partial matrix-vector result. Low-rank U@V.T performed separately.
"""
import os,sys,json,gc,time,math,hashlib,argparse,statistics
from pathlib import Path
import numpy as np,psutil
import mlx.core as mx
ROOT=Path(__file__).resolve().parent
P=ROOT.parent
# weights to measure: STERA_WEIGHTS (or STERA_PESI) = H122000 (final checkpoint, default) or H80000
PESI=os.environ.get('STERA_WEIGHTS') or os.environ.get('STERA_PESI','H122000')   # PESI = weights tag
NPZ=P/'models'/f'{PESI}_fp32_weights.npz'
BASE=[('embedding',50304,768)]+[(f'layers.{l}.{k}',r,c) for l in range(6) for k,r,c in [('q',768,768),('k',768,768),('v',768,768),('o',768,768),('fc',3072,768),('proj',768,3072)]]
NBASE=sum(r*c for _,r,c in BASE)
def layouts(f):
    return [(n,r*(1 if n=='embedding' else f),c*f) for n,r,c in BASE]
def theoretical(f):
    mat=sum(r*c for _,r,c in layouts(f))
    rest=2048*768*f+13*2*768*f+6*(768+3072+768)*f
    return mat,mat+rest
def get_mem():
    proc=psutil.Process()
    try:gpu=mx.metal.get_active_memory()
    except Exception:gpu=-1
    return {'rss_mib':round(proc.memory_info().rss/2**20,2),'mlx_active_mib':round(gpu/2**20,2),
        'sys_available_mib':round(psutil.virtual_memory().available/2**20,2),
        'swap_used_mib':round(psutil.swap_memory().used/2**20,2)}
def make_map():
    x=np.full((128,128),-1,dtype=np.int32);idx=255
    for i in range(1,128):
        for k in range(23):
            s=1+(13*k+7*i)%127
            assert x[i,s]==-1
            x[i,s]=idx;idx+=1
    assert idx==3176
    return mx.array(x.flatten())
KROW=mx.fast.metal_kernel(
 name='rowmask23_gemv_partial_v1',
 input_names=['coords','scales','vec','rid'],
 output_names=['partial'],
 source=r"""
     uint at = thread_position_in_grid.x;
     uint ct = uint(vec_shape[0]) / 128u;
     uint row = at / ct, cb = at % ct;
     uint ri = row & 127u;
     uint tile = (row >> 7u)*ct+cb;
     uint cbase = tile*3176u;
     float acc=0.0f;
     for (uint j=0;j<128u;j++) {
         uint sh = ri ^ j;
         uint b = (sh == 0u)?ri:(127u+sh);
         int rx = rid[ri*128u+sh];
         float w = coords[cbase+b];
         if(rx>=0) w+=coords[cbase+uint(rx)];
         acc+=w*vec[cb*128u+j];
     }
     partial[at]=acc*scales[tile];
 """)
KEMB=mx.fast.metal_kernel(
 name='rowmask23_embedding_row_v1',
 input_names=['coords','scales','gout','gin','rid','tok'],
 output_names=['embedded'],
 source=r"""
     uint j=thread_position_in_grid.x;
     uint row=uint(tok[0]);
     uint ct=uint(gin_shape[0])/128u;
     uint tile=(row>>7u)*ct+(j>>7u);
     uint ri=row&127u, cj=j&127u, sh=ri^cj;
     uint b=(sh==0u)?ri:(127u+sh);
     int rx=rid[ri*128u+sh];
     float w=coords[tile*3176u+b];
     if(rx>=0)w+=coords[tile*3176u+uint(rx)];
     embedded[j]=w*scales[tile]*gout[row]*gin[j];
 """)
class CoordModel:
    def __init__(self,f):
        self.f=f;self.h=768*f;self.ffn=3072*f;self.nhead=16*f
        self.map=make_map()
        self.p={};self.w={}
        with np.load(NPZ,allow_pickle=False) as z:
            self.p['position.weight']=mx.array(np.tile(z['position.weight'],(1,f)))
            for n in ['final_norm']+[f'layers.{l}.{k}' for l in range(6) for k in ['ln1','ln2']]:
                for attr in ['weight','bias']:
                    self.p[n+'.'+attr]=mx.array(np.tile(z[n+'.'+attr],f))
            for l in range(6):
                for name in ['o','fc','proj']:
                    k=f'layers.{l}.{name}.bias';self.p[k]=mx.array(np.tile(z[k],f))
            for name,r,c in BASE:
                rf=1 if name=='embedding' else f
                cf=f
                rt,ct=r//128,c//128
                co=z[name+'.coords'].reshape(rt,ct,3176)
                sc=z[name+'.scales'].reshape(rt,ct)
                # non-embedding linear maps have EVERY enlarged block W0/F; embedding repeats W0
                expc=np.tile(co,(rf,cf,1)).reshape(-1,3176)
                if rf!=1:expc*=np.float32(1.0/cf)
                exps=np.tile(sc,(rf,cf)).reshape(-1)
                ggout=np.tile(z[name+'.gout'],rf)
                ggin=np.tile(z[name+'.gin'],cf)
                U=np.tile(z[name+'.U'],(rf,1))
                V=np.tile(z[name+'.V'],(cf,1))
                if rf!=1:V*=np.float32(1.0/cf)
                self.w[name]=(mx.array(expc),mx.array(exps),mx.array(ggout),mx.array(ggin),mx.array(U),mx.array(V))
                mx.eval(*self.w[name])
                if name=='embedding' or name.endswith('.proj'):
                    print('LOADED',name,'physical_coords',expc.shape,'gpu',get_mem(),flush=True)
                del expc,exps,ggout,ggin,U,V
        mx.eval(*self.p.values())
        self.kv=[None]*6
        self.weight_bytes=sum(x.nbytes for a in self.w.values() for x in a)+sum(x.nbytes for x in self.p.values())
        print('READY','factor',f,'resident_weight_bytes',self.weight_bytes,'mem',get_mem(),flush=True)
    def matvec(self,name,x):
        co,sc,gout,gin,U,V=self.w[name]
        rows=gout.size;cols=gin.size;ct=cols//128
        assert x.ndim==1 and x.size==cols,(name,x.shape,cols)
        vec=x*gin
        partial=KROW(inputs=[co.reshape(-1),sc,vec,self.map],grid=(rows*ct,1,1),threadgroup=(256,1,1),output_shapes=[(rows,ct)],output_dtypes=[mx.float32])[0]
        y=mx.sum(partial,axis=1)*gout + U @ (V.T @ x)
        key=name+'.bias'
        if key in self.p:y=y+self.p[key]
        return y
    def emb(self,tid,t):
        co,sc,gout,gin,U,V=self.w['embedding']
        tok=mx.array([tid],dtype=mx.int32)
        e=KEMB(inputs=[co.reshape(-1),sc,gout,gin,self.map,tok],grid=(self.h,1,1),threadgroup=(256,1,1),output_shapes=[(self.h,)],output_dtypes=[mx.float32])[0]
        e=e+ (U[tid] @ V.T)
        return e+self.p['position.weight'][t]
    def ln(self,x,n):
        w=self.p[n+'.weight'];b=self.p[n+'.bias']
        return (x-mx.mean(x))*mx.rsqrt(mx.mean(mx.square(x-mx.mean(x)))+1e-5)*w+b
    def forward_one(self,tid,t):
        x=self.emb(tid,t)
        for l in range(6):
            n=f'layers.{l}.'
            z=self.ln(x,n+'ln1')
            q=self.matvec(n+'q',z).reshape(self.nhead,48)
            k=self.matvec(n+'k',z).reshape(self.nhead,48)
            v=self.matvec(n+'v',z).reshape(self.nhead,48)
            if self.kv[l] is None:K=k[None];V=v[None]
            else:
                K=mx.concatenate((self.kv[l][0],k[None]),axis=0)
                V=mx.concatenate((self.kv[l][1],v[None]),axis=0)
            self.kv[l]=(K,V)
            if l%2 and t>=256:K=K[-256:];V=V[-256:]
            logits=mx.sum(K*q[None],axis=-1)
            prob=mx.softmax(logits,axis=0)
            a=mx.sum(V*prob[:,:,None],axis=0).reshape(self.h)
            x=x+self.matvec(n+'o',a)
            y=self.matvec(n+'fc',self.ln(x,n+'ln2'))
            y=0.5*y*(1+mx.tanh(np.float32(0.7978845608028654)*(y+np.float32(0.044715)*y*y*y)))
            x=x+self.matvec(n+'proj',y)
        # Tied output projection embedding; 1/f restores exactly the baseline logits
        logits=self.matvec('embedding',self.ln(x,'final_norm'))/self.f
        return logits[:50257]
def run(f,ntokens):
    t0=time.perf_counter();p=CoordModel(f);loading=time.perf_counter()-t0
    model_info={'f':f,'architecture':{'layers':6,'hidden':768*f,'mlp':3072*f,'heads':16*f,'head_dim':48,'vocab':50257},
      'parameter_counts':{'matrix_dense_elements':theoretical(f)[0],'other_elements_approx':theoretical(f)[1]-theoretical(f)[0],
        'stored_compressed_fp32_bytes':p.weight_bytes,'original_checkpoint_sha256':hashlib.sha256(NPZ.read_bytes()).hexdigest()},
      'memory_resident':get_mem(),'loading_s':loading,'mode':'coords_gpu_streamed_tiled_gemv; 128-column tile never full dense weight',
      'expanded_weights':'deterministic learned checkpoint repetition; all 10x10 blocks W_original/10; tied embedding repeated; output scaled 1/10',
      'limitation':'Functional clone; NOT a 4.63B independent learned parameter model; speed is prototype, Metal kernel unoptimized.'}
    print('MEM_AND_WEIGHTS',json.dumps(model_info),flush=True)
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(P/'STERA_FINAL_H80000_LOSSLESS_FROZEN_20261122/tokenizer.json'))
    prompt='Once upon a time, there was a little cat named Mimi.'
    ids=[50256]+tok.encode(prompt,add_special_tokens=False).ids
    print('PROMPT_IDS',ids,flush=True)
    maxprefill=3;ids=ids[:maxprefill]
    times=[]
    for t in range(maxprefill+ntokens):
        st=time.perf_counter()
        lg=p.forward_one(ids[t],t)
        mx.eval(lg)
        if t==maxprefill-1:
            np.save(ROOT/f'LOGITS_PRE_DECODE_F{f}.npy',np.array(lg))
        nxt=int(mx.argmax(lg))
        dt=time.perf_counter()-st
        times.append(dt)
        if t==maxprefill-1 or t>=maxprefill:
            print('ONE_TOKEN',t,'duration_s',round(dt,3),'next',nxt,'mem',get_mem(),flush=True)
        if t>=maxprefill-1: ids.append(nxt)
    model_info['per_token_wall_seconds']=times
    model_info['measured_tokens_per_sec_decode']=ntokens/sum(times[maxprefill:maxprefill+ntokens])
    model_info['generated_token_ids']=ids[maxprefill:maxprefill+ntokens]
    model_info['post_mem']=get_mem()
    (ROOT/f'BENCH_COORD_MLX_F{f}.json').write_text(json.dumps(model_info,indent=2))
    print('FINISHED_RESULT',json.dumps(model_info),flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--factor',type=int,default=1);a.add_argument('--tokens',type=int,default=1);x=a.parse_args()
    run(x.factor,x.tokens)

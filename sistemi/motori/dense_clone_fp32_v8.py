#!/usr/bin/env python3
"""Dense FP16 resident comparison for the same deterministic width-10 STERA clone.
Full dense FP32 is 18.6GB (>16GiB physical); therefore its resident inference cannot
be safely run on this Mac. Half precision is a separate comparative deployment mode,
NOT an exact FP32 equivalence. Uses same learned checkpoint, tiling W0/10.
Run separate from streamed FP32 experiment; reject if spare RAM is inadequate.
"""
import os,sys,time,json,math,hashlib,gc,psutil,argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
P=Path(__file__).resolve().parent.parent
ROOT=Path(__file__).resolve().parent
NPZ=P/'modelli/H80000_fp32_weights.npz'
BASE=[('embedding',50304,768)]+[(f'layers.{l}.{k}',r,c) for l in range(6) for k,r,c in [('q',768,768),('k',768,768),('v',768,768),('o',768,768),('fc',3072,768),('proj',768,3072)]]
def usage():
 p=psutil.Process()
 return {'rss_mib':round(p.memory_info().rss/2**20,2),'mlx_active_mib':round(mx.get_active_memory()/2**20,2),
         'sys_available_mib':round(psutil.virtual_memory().available/2**20,2),'swap_used_mib':round(psutil.swap_memory().used/2**20,2)}
def build_base_tensor(z,n,r,c,base,res,mask):
    co=mx.array(z[n+'.coords']);sc=mx.array(z[n+'.scales'])
    nt=co.shape[0];rt=r//128;ct=c//128
    tile=(mx.take(co,base,axis=1)+mx.take(co,res,axis=1)*mask).reshape(nt,128,128)*sc
    dense=tile.reshape(rt,ct,128,128).transpose(0,2,1,3).reshape(r,c)
    dense=dense*mx.array(z[n+'.gout'])[:,None]*mx.array(z[n+'.gin'])[None,:]
    dense=dense + mx.array(z[n+'.U']) @ mx.array(z[n+'.V']).T
    return dense
def maps():
    rid=np.full((128,128),-1,np.int32);idx=255
    for i in range(1,128):
        for k in range(23):
            sh=1+(13*k+7*i)%127;rid[i,sh]=idx;idx+=1
    base=np.empty(16384,np.int32);res=np.empty(16384,np.int32);mask=np.empty(16384,np.float32)
    for i in range(128):
        for j in range(128):
            s=i^j;at=i*128+j
            base[at]=i if s==0 else 127+s
            v=rid[i,s];res[at]=max(v,0);mask[at]=float(v>=0)
    return mx.array(base),mx.array(res),mx.array(mask.reshape(1,16384))
class Model:
    def __init__(self,f):
        assert f>=1;self.f=f;self.h=768*f;self.nhead=16*f;self.w={};self.p={};self.kv=[None]*6
        b,r,mask=maps()
        mx.set_cache_limit(150*2**20)
        startmem=usage()
        with np.load(NPZ,allow_pickle=False) as z:
            self.p['position.weight']=mx.array(np.tile(z['position.weight'],(1,f)),dtype=mx.float32)
            for n in ['final_norm']+[f'layers.{l}.{k}' for l in range(6) for k in ['ln1','ln2']]:
                for attr in ('weight','bias'):
                    self.p[n+'.'+attr]=mx.array(np.tile(z[n+'.'+attr],f),dtype=mx.float32)
            for l in range(6):
                for name in ('o','fc','proj'):
                    n=f'layers.{l}.{name}.bias'
                    self.p[n]=mx.array(np.tile(z[n],f),dtype=mx.float32)
            for n,rb,cb in BASE:
                if psutil.virtual_memory().available < 1100*2**20:
                    raise MemoryError('Safety stop: system free memory <1.1GiB')
                frow=1 if n=='embedding' else f
                basew=build_base_tensor(z,n,rb,cb,b,r,mask)
                expanded=mx.tile(basew,(frow,f))
                if frow!=1:expanded=expanded / f
                wf=expanded.astype(mx.float32)
                mx.eval(wf)
                self.w[n]=wf
                del wf,expanded,basew
                mx.clear_cache();gc.collect()
                if n=='embedding' or n.endswith('.proj'):
                    print('DENSE_RESIDENT',n,self.w[n].shape,usage(),flush=True)
        mx.eval(*self.p.values())
        self.weight_bytes=sum(w.nbytes for w in self.w.values())+sum(w.nbytes for w in self.p.values())
        print('DENSE_READY',self.weight_bytes,'fp32',usage(),'start',startmem,flush=True)
    def ln(self,x,n):
        z=x-mx.mean(x);w=self.p[n+'.weight'];b=self.p[n+'.bias']
        return z*mx.rsqrt(mx.mean(z*z)+1e-5)*w+b
    def lin(self,n,x):
        out=self.w[n]@x
        k=n+'.bias'
        if k in self.p:out=out+self.p[k]
        return out
    def forward_one(self,tid,t):
        x=self.w['embedding'][tid]+self.p['position.weight'][t]
        for l in range(6):
            n=f'layers.{l}.'
            z=self.ln(x,n+'ln1')
            q=self.lin(n+'q',z).reshape(self.nhead,48)
            k=self.lin(n+'k',z).reshape(self.nhead,48)
            v=self.lin(n+'v',z).reshape(self.nhead,48)
            if self.kv[l] is None:K=k[None];V=v[None]
            else:
                K=mx.concatenate((self.kv[l][0],k[None]),axis=0)
                V=mx.concatenate((self.kv[l][1],v[None]),axis=0)
            self.kv[l]=(K,V)
            if l%2 and t>=256:K=K[-256:];V=V[-256:]
            attention=mx.sum(K*q[None],axis=-1)
            prob=mx.softmax(attention,axis=0)
            a=mx.sum(V*prob[:,:,None],axis=0).reshape(self.h)
            x=x+self.lin(n+'o',a)
            y=self.lin(n+'fc',self.ln(x,n+'ln2'))
            y=0.5*y*(1+mx.tanh(np.float32(0.7978845608028654)*(y+np.float32(0.044715)*y*y*y)))
            x=x+self.lin(n+'proj',y)
        return ((self.w['embedding']@self.ln(x,'final_norm'))/self.f)[:50257]
def main(f=10,ntokens=3):
    start=time.perf_counter();base_swap=psutil.swap_memory().used
    p=Model(f);load_s=time.perf_counter()-start
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(P/'STERA_FINAL_H80000_LOSSLESS_FROZEN_20261122/tokenizer.json'))
    ids=[50256]+tok.encode('Once upon a time, there was a little cat named Mimi.',add_special_tokens=False).ids
    prefill=3;ids=ids[:prefill];times=[];result={'factor':f,'dtype':'float32','mode':'dense_original_fp32_base; math contraction of F10 trained-coordinate clone','weights_bytes':p.weight_bytes,'load_seconds':load_s,'resident_after_loading':usage()}
    for t in range(prefill+ntokens):
        st=time.perf_counter()
        logits=p.forward_one(ids[t],t)
        mx.eval(logits)
        nxt=int(mx.argmax(logits))
        sec=time.perf_counter()-st
        times.append(sec);ids.append(nxt)
        print('DENSE_TOKEN',t,'seconds',sec,'id',nxt,'mem',usage(),flush=True)
        if psutil.swap_memory().used-base_swap>3*2**30:
            print('SAFETY_STOP swap increased >3GiB',flush=True);break
    result['token_seconds']=times
    result['generated_ids']=ids[prefill:]
    if len(times)>=prefill+ntokens:
        result['decode_tok_s']=ntokens/sum(times[prefill:prefill+ntokens])
    result['final_mem']=usage()
    (ROOT/f'BENCH_DENSE_FP32_F{f}.json').write_text(json.dumps(result,indent=2))
    print('FINAL_DENSE_BENCH',json.dumps(result),flush=True)
if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--factor',type=int,default=10);pa.add_argument('--tokens',type=int,default=3)
    a=pa.parse_args();main(a.factor,a.tokens)

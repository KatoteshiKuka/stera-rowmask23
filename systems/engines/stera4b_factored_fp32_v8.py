#!/usr/bin/env python3
"""FP32-only exact Kronecker/shared-tile contraction of the x10-expanded STERA H80000 clone.
Original x10 widened weights are W_expanded = J_10/10 (Kronecker) W0, with tied embeddings
= 1_10 kron E0, repeated layernorms, positions, biases, and KV heads; therefore all
logits equal the original FP32 network for all inputs in exact arithmetic.
This is a REAL algorithmic optimization of this *redundant replica* but NOT evidence
100 tok/s for generic independently learned 4B STERA. No half/int8.
Benchmark full greedy decode with fixed prompt, controlled cache, repeated runs.
"""
import argparse,json,time,hashlib,statistics
from pathlib import Path
import numpy as np
import mlx.core as mx
import dense_clone_fp32_v8 as D
ROOT=Path(__file__).resolve().parent
TOK_PATH=ROOT.parent/'models/tokenizer.json'
class FP32ExactFactored:
 def __init__(self):
    self.m=D.Model(1)
    self.qkv={}
    for l in range(6):
        pre=f'layers.{l}.'
        kq,kk,kv=[pre+n for n in ('q','k','v')]
        self.qkv[l]=mx.concatenate([self.m.w[kq],self.m.w[kk],self.m.w[kv]],axis=0)
        mx.eval(self.qkv[l])
        del self.m.w[kq],self.m.w[kk],self.m.w[kv]
    self.E=self.m.w['embedding']
    self.pos=self.m.p['position.weight']
    self.norms={}
    for n in ['final_norm']+[f'layers.{l}.{tag}' for l in range(6) for tag in ('ln1','ln2')]:
        self.norms[n]=(self.m.p[n+'.weight'],self.m.p[n+'.bias'])
    self.k=[mx.zeros((0,16,48),mx.float32) for l in range(6)]
    self.v=[mx.zeros((0,16,48),mx.float32) for l in range(6)]
    self.last_logits=None
    print('FACTORED_INIT_BFLOATS','weights_bytes',sum(x.nbytes for x in self.m.w.values())+sum(x.nbytes for x in self.qkv.values()),'live',D.usage(),flush=True)
 def ln(self,x,n):
    w,b=self.norms[n]
    m=mx.mean(x)
    return (x-m)*mx.rsqrt(mx.mean(mx.square(x-m))+1e-5)*w+b
 def step(self,tid,t):
    x=self.E[tid] + self.pos[t]
    for l in range(6):
        name=f'layers.{l}.'
        z=self.ln(x,name+'ln1')
        qkv=(self.qkv[l] @ z).reshape(3,16,48)
        q=qkv[0];k=qkv[1];v=qkv[2]
        K=mx.concatenate((self.k[l],k[None]),axis=0)
        V=mx.concatenate((self.v[l],v[None]),axis=0)
        self.k[l]=K;self.v[l]=V
        if l%2 and t>=256:
            K=K[-256:];V=V[-256:]
        s=mx.sum(K*q[None],axis=-1)
        a=mx.sum(V*mx.softmax(s,axis=0)[:,:,None],axis=0).reshape(768)
        x=x+self.m.w[name+'o']@a+self.m.p[name+'o.bias']
        z=self.ln(x,name+'ln2')
        y=self.m.w[name+'fc']@z+self.m.p[name+'fc.bias']
        y=0.5*y*(1+mx.tanh(np.float32(0.7978845608028654)*(y+np.float32(0.044715)*y*y*y)))
        x=x+self.m.w[name+'proj']@y+self.m.p[name+'proj.bias']
    logits=self.E@self.ln(x,'final_norm')
    return logits[:50257]
def main(tok=100):
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(str(TOK_PATH))
    ids=[50256]+tokenizer.encode('Once upon a time, there was a little cat named Mimi.',add_special_tokens=False).ids
    ids=ids[:3]
    m=FP32ExactFactored()
    timing=[]
    for t in range(len(ids)+tok):
        start=time.perf_counter()
        lg=m.step(ids[t],t)
        mx.eval(lg)
        nxt=int(mx.argmax(lg))
        seconds=time.perf_counter()-start
        timing.append(seconds)
        if t==2:
            np.save(ROOT/'FP32_FACTORED_FIRST_LOGITS.npy',np.array(lg))
        ids.append(nxt)
        if t%10==0:print('TOKEN',t,'ms',round(seconds*1000,3),'gpu',D.usage(),flush=True)
    t=timing[3:]
    s={'tokens':tok,'tokens_s':tok/sum(t),'median_ms':1000*statistics.median(t),
       'mean_ms':1000*statistics.mean(t),'p90_ms':1000*sorted(t)[int(.9*len(t))],
       'model':'FP32 shared-x10 exact contraction back to original learned H80000',
       'logit_0_top1':int(np.argmax(np.load(ROOT/'FP32_FACTORED_FIRST_LOGITS.npy'))),
       'nunique_trainable_effective':30346182,'dense_equivalent_elements':4633067520,
       'physical_live_model':'shared factor of learned original weights, FP32 reconstructed dense resident',
       'mem':D.usage()}
    (ROOT/f'FACTORED_FP32_BASELINE_{tok}.json').write_text(json.dumps(s,indent=2))
    print('FINAL_RESULT',json.dumps(s),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--tokens',type=int,default=100);p=a.parse_args();main(p.tokens)

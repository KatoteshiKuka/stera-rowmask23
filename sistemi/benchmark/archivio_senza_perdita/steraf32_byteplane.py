#!/usr/bin/env python3
"""Exact reversible byte-shuffled + zstd storage for all STERA FP32 NumPy weight tensors.
Preserves every float32 bit; never quantizes.
File format: b'STERAF32BS1' + little-endian uint64(metadata json length) + metadata json + byte-plane-shuffled tensor bytes.
The container stream is compressed with zstd losslessly.
"""
from pathlib import Path
import sys,json,struct,hashlib,subprocess,time,io,os
import numpy as np
MAGIC=b'STERAF32BS1'
def compress(inp,out,level=9):
    inp=Path(inp);out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    metadata=[]
    with np.load(inp,allow_pickle=False) as z:
        for k in z.files:
            a=z[k]
            assert a.dtype==np.dtype('<f4'),(k,a.dtype)
            metadata.append(dict(name=k,shape=list(a.shape),dtype=str(a.dtype),nbytes=a.nbytes))
        info=json.dumps(dict(format='steraf32-byteplanes-v1',tensors=metadata),separators=(',',':')).encode()
        zstd=subprocess.Popen(['/opt/homebrew/bin/zstd','-q',f'-{level}','-T2','-c'],stdin=subprocess.PIPE,stdout=out.open('wb'))
        zstd.stdin.write(MAGIC+struct.pack('<Q',len(info))+info)
        for k in z.files:
            a=z[k]
            view=a.view('u1').reshape(-1,4)
            zstd.stdin.write(view.T.tobytes(order='C'))
        zstd.stdin.close()
        ret=zstd.wait()
        assert ret==0,ret
    print('ENCODED',str(inp),str(out),out.stat().st_size,'bytes',flush=True)
def decompress_verify(inp,orig):
    inp=Path(inp);orig=Path(orig)
    p=subprocess.Popen((['/opt/homebrew/bin/xz','-dc',str(inp)] if inp.open('rb').read(6)==bytes.fromhex('fd377a585a00') else ['/opt/homebrew/bin/zstd','-q','-dc',str(inp)]),stdout=subprocess.PIPE)
    fp=p.stdout
    assert fp.read(len(MAGIC))==MAGIC
    n=struct.unpack('<Q',fp.read(8))[0]
    meta=json.loads(fp.read(n))
    count=0
    with np.load(orig,allow_pickle=False) as z:
        assert set(z.files)==set(m['name'] for m in meta['tensors'])
        for e in meta['tensors']:
            payload=fp.read(e['nbytes'])
            assert len(payload)==e['nbytes']
            # Each of 4 channels of uint8 has numel values, interleave back into FP32 values.
            view=np.frombuffer(payload,dtype=np.uint8).reshape(4,-1).T.copy()
            original=z[e['name']]
            arr=view.reshape(-1).view(np.float32).reshape(e['shape'])
            assert arr.dtype==original.dtype
            assert arr.shape==original.shape
            assert arr.tobytes()==original.tobytes(),e['name']
            count+=1
    assert fp.read(1)==b''
    assert p.wait()==0
    print('LOSSLESS ROUNDTRIP BIT-BY-BIT PASS',count,'tensors',flush=True)
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='compress':compress(sys.argv[2],sys.argv[3],int(sys.argv[4]))
    else:decompress_verify(sys.argv[2],sys.argv[3])

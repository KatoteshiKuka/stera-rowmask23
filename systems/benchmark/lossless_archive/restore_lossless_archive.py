#!/usr/bin/env python3
"""Restore exact original float32 tensors from STERAF32BS1 archives (.xz6/.xz9 or .zst).
Only invoked on trusted local archives. Decoder verifies the 267 entries and can reproduce the model-ready NPZ.
"""
import sys,json,struct,subprocess
from pathlib import Path
from collections import OrderedDict
import numpy as np
MAGIC=b'STERAF32BS1'
def unpack(src):
    src=Path(src)
    cmd=(['/opt/homebrew/bin/xz','-dc',str(src)] if src.open('rb').read(6)==bytes.fromhex('fd377a585a00') else ['/opt/homebrew/bin/zstd','-q','-dc',str(src)])
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE)
    assert p.stdout.read(len(MAGIC))==MAGIC,'bad format'
    count=struct.unpack('<Q',p.stdout.read(8))[0]
    assert count<10**7
    meta=json.loads(p.stdout.read(count))
    assert meta['format']=='steraf32-byteplanes-v1'
    arrays=OrderedDict()
    for d in meta['tensors']:
        assert d['dtype']=='float32'
        nbytes=d['nbytes']
        assert nbytes<100_000_000 and nbytes%4==0
        payload=p.stdout.read(nbytes)
        assert len(payload)==nbytes
        inter=np.frombuffer(payload,dtype=np.uint8).reshape(4,-1).T.copy().reshape(-1).view(np.float32).reshape(d['shape'])
        arrays[d['name']]=inter
    assert p.stdout.read(1)==b''
    assert p.wait()==0
    return arrays
if __name__=='__main__':
    src,out=sys.argv[1:3]
    arrays=unpack(src)
    with open(out,'wb') as f: np.savez(f,**arrays)
    with np.load(out,allow_pickle=False) as z:
        assert all(z[k].tobytes()==v.tobytes() for k,v in arrays.items())
    print('RESTORED',out,len(arrays),'FP32 TENSORS',Path(out).stat().st_size)

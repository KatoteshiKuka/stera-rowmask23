#!/usr/bin/env python3
"""Measure real SSD read volume without allocating a 18.6 GB Dense file on 9.8GiB free Mac.
A raw streaming-I/O lower bound, NOT a Dense inference benchmark.
"""
import mmap,re,subprocess
import os,time,fcntl,json,statistics,hashlib,platform
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=Path.home()/"Desktop/STERA/Experiments/STERA_V3_NEXTGEN_20261115/kaggle_export_20261115/local_extended_v3_100_120_140m/science_diagnosis_20261116/official_gpt4_v2_20261116/TinyStoriesV2-GPT4-train.official.txt"
SIZES={"compact_weight_payload":4002707952,"dense_equivalent_weights":18597089280}
TRIALS=3
F_NOCACHE=48 # macOS fcntl.h
CHUNK=16*1024*1024
def driver_read_bytes():
    tree=subprocess.check_output(["/usr/sbin/ioreg","-r","-c","IOBlockStorageDriver","-l","-w0"],text=True)
    counters=re.findall(r'^  \|   "Statistics" = .*?"Bytes \(Read\)"=(\d+)',tree,re.M)
    if not counters:
        raise RuntimeError("IOBlockStorageDriver read counters unavailable")
    return sum(map(int,counters))

def read_budget(vol):
    before=driver_read_bytes()
    fd=os.open(str(SOURCE),os.O_RDONLY)
    try:
        fcntl.fcntl(fd,F_NOCACHE,1)
        page=mmap.PAGESIZE
        span=SOURCE.stat().st_size//page*page
        read_target=(vol+page-1)//page*page
        buf=mmap.mmap(-1,CHUNK)
        total=0;rewinds=0;started=time.perf_counter()
        while total<read_target:
            n=min(CHUNK,read_target-total,span-os.lseek(fd,0,os.SEEK_CUR))
            if n<=0:
                os.lseek(fd,0,os.SEEK_SET);rewinds+=1;continue
            with memoryview(buf)[:n] as view:
                got=os.readv(fd,[view])
            if got!=n:raise RuntimeError(f"Short read: {got}/{n}")
            total+=got
        elapsed=time.perf_counter()-started
        buf.close()
    finally:os.close(fd)
    return {"driver_read_bytes_delta":driver_read_bytes()-before,"bytes_read":total,"seconds":elapsed,"effective_GB_per_s":total/elapsed/1e9,"file_rewinds":rewinds}
out={"protocol":"STERA_SSD_IO_DIRECT_READ_NO_COMPUTE_V1","date":"2026-11-23","host":platform.platform(),"source_file":str(SOURCE),"source_bytes":SOURCE.stat().st_size,"io_cache_policy":"fcntl F_NOCACHE=48 (macOS), readv() in page-aligned mmap buffer, chunks up to 16 MiB, page-aligned source span and byte budget; sequential; pre-existing real file repeatedly read","trials":TRIALS,"scenarios":{}}
assert SOURCE.exists() and SOURCE.stat().st_size>2_000_000_000
for key,vol in SIZES.items():
    results=[]
    for run in range(TRIALS):
        m=read_budget(vol);results.append(m)
        print("MEASURED",key,"trial",run+1, json.dumps(m),flush=True)
    out["scenarios"][key]={"payload_bytes":vol,"trials":results,
        "median_seconds":statistics.median(x['seconds'] for x in results),
        "median_GB_s":statistics.median(x['effective_GB_per_s'] for x in results)}
out["interpretation"]="I/O-only sequential read time for bytes corresponding to resident compact weights vs Dense-equivalent weights; not per-token model output; repeats same 2.23GB disk file; F_NOCACHE bypasses filesystem data cache; no 18.6GB Dense weights were created or multiplied. SSD data was not transformed into matrices. This is a lower-bound style operational reference, not a measured out-of-core Dense transformer."
out["safety"]="No large files written; only small JSON/Markdown report."
(HERE/"SSD_STREAMING_IO_MAC_M1_20261123.json").write_text(json.dumps(out,indent=2))
a=out["scenarios"];d=a["dense_equivalent_weights"];c=a["compact_weight_payload"]
lines=["# STERA — SSD sequential I/O budget, MacBook Air M1, 23 November 2026","",
"**Important:** this is an SSD I/O measurement, not a Dense model inference measurement. It does not execute model arithmetic or reconstruct the 18.6GB Dense matrix.",
"",
"| Data volume | Bytes | Median sequential SSD read time | Median throughput |",
"|---|---:|---:|---:|"]
for name,rec in a.items(): lines.append(f"| {name} | {rec['payload_bytes']:,} | {rec['median_seconds']:.3f} s | {rec['median_GB_s']:.3f} GB/s |")
lines+=["","The data source was an existing 2.23GB real file repeatedly read through macOS F_NOCACHE, with 16MiB reads. No additional 18.6GB file was written, because free disk space is insufficient. Repeated reads of a source smaller than the nominal weight file are a limitation; cold-read latency, random access and GPU compute are not included.","",
f"If all {SIZES['dense_equivalent_weights']/1e9:.3f} GB had to be fetched from SSD once for **every token**, the **I/O-only** median time would be about **{d['median_seconds']:.2f} s/token**, corresponding to **{1/d['median_seconds']:.3f} tokens/s** before GPU computation and transfers. This is not a measured fully optimized out-of-core Dense model.","",
"Compact F10's previously measured 11–12 token/s is a GPU end-to-end prototype with weights resident, so this comparison is a scenario contrast, not a matched kernel microbenchmark.",
"","To prove a direct out-of-core Dense inference speed, a tiled GPU Dense matvec against real tensor values, parity checked against the RowMask oracle, with measured disk-to-GPU scheduling is still required."]
(HERE/"SSD_STUDY_NOTE_20261123.md").write_text("\n".join(lines)+"\n")
print("REPORT_READY",HERE/"SSD_STUDY_NOTE_20261123.md",flush=True)

import mmap,re,subprocess,tempfile
import os,time,fcntl,json,statistics,hashlib,platform
from pathlib import Path
HERE=Path(__file__).resolve().parent
SIZES={"compact_weight_payload":4002707952,"dense_equivalent_weights":18597089280}
TRIALS=3
F_NOCACHE=48 # macOS fcntl.h
CHUNK=1024*1024
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

def main():
    global SOURCE
    size=256*CHUNK
    samples={key: [] for key in SIZES}
    with tempfile.NamedTemporaryFile(dir=HERE,prefix="ssd_fixture_",buffering=0) as fixture:
        SOURCE=Path(fixture.name)
        fcntl.fcntl(fixture.fileno(),F_NOCACHE,1)
        for _ in range(size//CHUNK):
            if fixture.write(os.urandom(CHUNK))!=CHUNK:
                raise IOError("Short fixture write")
        os.fsync(fixture.fileno())
        stat=os.fstat(fixture.fileno())
        if stat.st_size!=size or stat.st_blocks*512<size:
            raise IOError("Fixture allocation mismatch")
        for rep in range(TRIALS):
            keys=list(SIZES)
            if rep%2:
                keys.reverse()
            for key in keys:
                result=read_budget(SIZES[key])
                result["trial"]=rep+1
                ratio=result["driver_read_bytes_delta"]/result["bytes_read"]
                result["driver_to_logical_ratio"]=ratio
                print(key,json.dumps(result),flush=True)
                if not 0.99<=ratio<=1.02:
                    raise IOError("Device counters did not corroborate byte traffic")
                samples[key].append(result)
    out={"host":platform.platform(),"chunk_bytes":CHUNK,"page_bytes":mmap.PAGESIZE,
         "source_bytes":size,"source_allocated_bytes":stat.st_blocks*512,
         "policy":"Temporary fsynced random binary fixture, F_NOCACHE, aligned readv, device counter corroboration",
         "scenarios":{}}
    for key,results in samples.items():
        out["scenarios"][key]={"payload_bytes":SIZES[key],"samples":results,
          "median_seconds":statistics.median(r["seconds"] for r in results),
          "median_GB_s":statistics.median(r["effective_GB_per_s"] for r in results)}
    (HERE/"ssd_validato.json").write_text(json.dumps(out,indent=2)+"\n")

if __name__=="__main__":
    main()

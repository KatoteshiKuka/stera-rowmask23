import fcntl
import json
import os
import re
import subprocess
import tempfile
import time

BLOCK = 1024 * 1024
SIZE = 256 * BLOCK

def read_counter():
    tree = subprocess.check_output(
        ["/usr/sbin/ioreg", "-r", "-c", "IOBlockStorageDriver", "-l", "-w0"],
        text=True,
    )
    return sum(map(int, re.findall(
        r'^  \|   "Statistics" = .*?"Bytes \(Read\)"=(\d+)', tree, re.M
    )))

with tempfile.NamedTemporaryFile(dir=".", prefix="ssd_probe_", buffering=0) as fixture:
    fcntl.fcntl(fixture.fileno(), 48, 1)
    for _ in range(SIZE // BLOCK):
        fixture.write(os.urandom(BLOCK))
    os.fsync(fixture.fileno())
    stat = os.fstat(fixture.fileno())
    print(json.dumps({"file_bytes": stat.st_size, "allocated_bytes": stat.st_blocks * 512}), flush=True)
    for nocache in (True, True, True, False, False):
        fd = os.open(fixture.name, os.O_RDONLY)
        try:
            if nocache:
                fcntl.fcntl(fd, 48, 1)
            before = read_counter()
            t0 = time.perf_counter()
            received = 0
            while data := os.read(fd, BLOCK):
                received += len(data)
            seconds = time.perf_counter() - t0
            print(json.dumps({"nocache": nocache, "bytes": received, "s": seconds,
                              "device_bytes_delta": read_counter() - before}), flush=True)
        finally:
            os.close(fd)

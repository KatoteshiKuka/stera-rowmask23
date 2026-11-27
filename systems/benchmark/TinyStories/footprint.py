"""Sample macOS physical footprint during one inference request."""

import ctypes
import os
import threading
from contextlib import contextmanager


class RUsageV0(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_ubyte * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups",
            "pageins", "wired_size", "resident_size", "phys_footprint",
            "proc_start_abstime", "proc_exit_abstime",
        )
    ]


@contextmanager
def observe_request(interval=0.01):
    proc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True).proc_pid_rusage
    proc.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(RUsageV0)]
    proc.restype = ctypes.c_int
    pid = os.getpid()

    def footprint():
        usage = RUsageV0()
        if proc(pid, 0, ctypes.byref(usage)) != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        return usage.phys_footprint

    initial = footprint()
    samples = [initial]
    stop = threading.Event()

    def sample_until_stopped():
        while not stop.wait(interval):
            samples.append(footprint())

    sampler = threading.Thread(target=sample_until_stopped)
    sampler.start()
    result = {"start_footprint_bytes": initial, "sampling_interval_s": interval}
    try:
        yield result
    finally:
        stop.set()
        sampler.join()
        samples.append(footprint())
        result.update(peak_footprint_sampled_bytes=max(samples), footprint_samples=len(samples))

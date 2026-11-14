"""Contended A/B of NT stores: N worker processes run a box kernel
simultaneously (barrier-synced), simulating a many-worker server where
per-core DRAM bandwidth is the bottleneck. Rounds alternate WSIDICOM_KERNELS_NT=1/0
(inherited at spawn). Reports per-op medians.

Usage: python bench/bench_contended.py [u8|u16] [factor]
"""

import multiprocessing as mp
import os
import sys
import time

import numpy as np

N_PROCS = 6
OPS = 20
DTYPES = {"u8": (np.uint8, 256), "u16": (np.uint16, 65536)}


def worker(barrier, q, tag, factor):
    from wsidicom_kernels import ResizeFilter, downscale

    rng = np.random.default_rng(os.getpid())
    dt, hi = DTYPES[tag]
    img = rng.integers(0, hi, (8192, 8192)).astype(dt)

    def op():
        downscale(img, factor, ResizeFilter.BOX)

    op()  # warm
    barrier.wait()
    t0 = time.perf_counter()
    for _ in range(OPS):
        op()
    q.put((time.perf_counter() - t0) / OPS * 1e3)


def round_(nt, tag, factor):
    os.environ["WSIDICOM_KERNELS_NT"] = "1" if nt else "0"
    barrier = mp.Barrier(N_PROCS)
    q = mp.Queue()
    procs = [
        mp.Process(target=worker, args=(barrier, q, tag, factor))
        for _ in range(N_PROCS)
    ]
    for p in procs:
        p.start()
    times = [q.get() for _ in procs]
    for p in procs:
        p.join()
    return float(np.median(times))


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "u8"
    factor = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    round_(True, tag, factor)  # warmup round
    a, b = [], []
    for _ in range(4):
        a.append(round_(True, tag, factor))
        b.append(round_(False, tag, factor))
    print(f"{tag} factor {factor}, {N_PROCS} procs, per-op median of {OPS} ops:")
    print(
        f"  NT on : {np.median(a):6.2f} ms   rounds {[f'{x:.1f}' for x in sorted(a)]}"
    )
    print(
        f"  NT off: {np.median(b):6.2f} ms   rounds {[f'{x:.1f}' for x in sorted(b)]}"
    )

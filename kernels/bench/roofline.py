"""Poor-man's roofline: effective throughput (in+out bytes / time) per kernel
vs the machine's measured copy ceiling. Kernels well below the ceiling are
compute-bound (headroom exists); kernels at it are done.
"""

import time

import numpy as np
from wsidicom_kernels import (
    ResizeFilter,
    downscale,
)
from wsidicom_kernels._kernels import set_simd

set_simd(True)
rng = np.random.default_rng(9)


def best(fn, k=5):
    b = float("inf")
    for _ in range(k):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b


# --- ceilings ---------------------------------------------------------------
src = rng.integers(0, 256, 268_435_456, np.uint8)  # 256 MB
dst = np.empty_like(src)
t = best(lambda: np.copyto(dst, src))
ceil_copy = 2 * src.nbytes / t / 1e9
t = best(lambda: src.copy())  # includes fresh allocation each call
ceil_alloc = 2 * src.nbytes / t / 1e9
print(f"copy ceiling (preallocated): {ceil_copy:5.1f} GB/s")
print(
    f"copy ceiling (fresh alloc) : {ceil_alloc:5.1f} GB/s   <- every wsidicom_kernels call allocates"
)

rows = []


def add(name, fn, nbytes):
    t = best(fn)
    rows.append((name, nbytes / t / 1e9, t * 1e3))


g = {}
for dt, tag in ((np.uint8, "u8"), (np.uint16, "u16")):
    hi = np.iinfo(dt).max + 1
    p = rng.integers(0, hi, (8192, 8192)).astype(dt)
    c3 = rng.integers(0, hi, (8192, 8192, 3)).astype(dt)
    e = dt(0).nbytes
    add(
        f"box2 planar {tag}",
        lambda p=p: downscale(p, 2, ResizeFilter.BOX),
        8192 * 8192 * e * 1.25,
    )
    add(
        f"box4 planar {tag}",
        lambda p=p: downscale(p, 4, ResizeFilter.BOX),
        8192 * 8192 * e * 1.0625,
    )
    add(
        f"box2 hwc c3 {tag}",
        lambda c3=c3: downscale(c3, 2, ResizeFilter.BOX),
        8192 * 8192 * 3 * e * 1.25,
    )

print(f"\n{'kernel':<24}{'GB/s':>7}{'ms':>9}{'% of alloc-ceiling':>20}")
for name, gbs, ms in sorted(rows, key=lambda r: r[1]):
    print(f"{name:<24}{gbs:>7.1f}{ms:>9.1f}{gbs / ceil_alloc:>19.0%}")

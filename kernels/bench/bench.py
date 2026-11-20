"""Interleaved A/B benchmark: SIMD dispatch vs forced-scalar, per kernel.

Rules: interleave A/B/A/B and take medians; never block.
The scalar side never changes across builds, so it doubles as a drift control
when comparing two builds' SIMD paths.

Usage: python bench/bench.py [substring filter ...]
"""

import gc
import sys
import time

import numpy as np
from wsidicom_kernels import (
    ResizeFilter,
    downscale,
)
from wsidicom_kernels._kernels import set_simd

REPS = 7


def timed(fn):
    gc.collect()
    t0 = time.perf_counter()
    fn()
    return (time.perf_counter() - t0) * 1e3


def ab(name, fn):
    set_simd(True)
    fn()  # warmup A
    set_simd(False)
    fn()  # warmup B
    simd, scal = [], []
    for _ in range(REPS):
        set_simd(True)
        simd.append(timed(fn))
        set_simd(False)
        scal.append(timed(fn))
    set_simd(True)
    s, c = np.median(simd), np.median(scal)
    ratio = c / s if s > 0 else float("nan")
    print(f"{name:34s} simd {s:8.2f} ms   scalar {c:8.2f} ms   x{ratio:.2f}")


def box_case(dt, shape, hi, factor=2):
    rng = np.random.default_rng(1)
    img = rng.integers(0, hi, shape).astype(dt)
    return lambda: downscale(img, factor, ResizeFilter.BOX)


CASES = []
for dt, tag, hi in (
    (np.uint8, "u8", 256),
    (np.uint16, "u16", 65536),
):
    CASES.append(
        (f"box planar {tag} 8192", lambda dt=dt, hi=hi: box_case(dt, (8192, 8192), hi))
    )
    CASES.append(
        (
            f"box4 planar {tag} 8192",
            lambda dt=dt, hi=hi: box_case(dt, (8192, 8192), hi, factor=4),
        )
    )
    CASES.append(
        (
            f"box hwc {tag} c3 8192",
            lambda dt=dt, hi=hi: box_case(dt, (8192, 8192, 3), hi),
        )
    )


if __name__ == "__main__":
    filters = [a.lower() for a in sys.argv[1:]]
    for name, make in CASES:
        if filters and not any(f in name.lower() for f in filters):
            continue
        ab(name, make())

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
    Interpolation,
    ResizeFilter,
    compose,
    downscale,
    rgb_to_rgbx,
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
    if hi:
        img = rng.integers(0, hi, shape).astype(dt)
    else:
        img = rng.random(shape, np.float32)
    return lambda: downscale(img, factor, ResizeFilter.BOX)


def compose_case(dt, hi, c, filt, n=5120):
    rng = np.random.default_rng(2)
    tshape = (n, n) if c == 1 else (n, n, c)
    cshape = (n + 2, n + 2) if c == 1 else (n + 2, n + 2, c)
    if hi:
        tile = rng.integers(0, hi, tshape).astype(dt)
    else:
        tile = rng.random(tshape, np.float32)
    canvas = np.zeros(cshape, dt)
    position = np.array([[0.3, 0.7]])
    interpolation = Interpolation(filt)
    return lambda: compose(canvas, [tile], position, interpolation)


CASES = []
for dt, tag, hi in (
    (np.uint8, "u8", 256),
    (np.uint16, "u16", 65536),
    (np.float32, "f32", None),
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
    for c in (3, 4):
        CASES.append(
            (
                f"box hwc {tag} c{c} 8192",
                lambda dt=dt, hi=hi, c=c: box_case(dt, (8192, 8192, c), hi),
            )
        )
    for filt in ("bilinear", "bicubic"):
        CASES.append(
            (
                f"{filt} planar {tag} 5120",
                lambda dt=dt, hi=hi, f=filt: compose_case(dt, hi, 1, f),
            )
        )
        CASES.append(
            (
                f"{filt} hwc {tag} c3 5120",
                lambda dt=dt, hi=hi, f=filt: compose_case(dt, hi, 3, f),
            )
        )


def rgbx_case():
    img = np.random.default_rng(4).integers(0, 256, (8192, 8192, 3), np.uint8)
    return lambda: rgb_to_rgbx(img)


CASES.append(("rgb_to_rgbx u8 8192", lambda: rgbx_case()))

if __name__ == "__main__":
    filters = [a.lower() for a in sys.argv[1:]]
    for name, make in CASES:
        if filters and not any(f in name.lower() for f in filters):
            continue
        ab(name, make())

"""Poor-man's roofline: effective throughput (in+out bytes / time) per kernel
vs the machine's measured copy ceiling. Kernels well below the ceiling are
compute-bound (headroom exists); kernels at it are done.
"""

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


POSITION = np.array([[0.3, 0.7]])
g = {}
for dt, tag in ((np.uint8, "u8"), (np.uint16, "u16"), (np.float32, "f32")):
    hi = None if dt == np.float32 else np.iinfo(dt).max + 1
    p = (
        rng.integers(0, hi, (8192, 8192)).astype(dt)
        if hi
        else rng.random((8192, 8192), np.float32)
    )
    c3 = (
        rng.integers(0, hi, (8192, 8192, 3)).astype(dt)
        if hi
        else rng.random((8192, 8192, 3), np.float32)
    )
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
    tile = p[:5120, :5120].copy()
    n = 5120 * 5120 * e
    cv = np.zeros((5122, 5122), dt)
    add(
        f"bilinear planar {tag}",
        lambda cv=cv, tile=tile: compose(cv, [tile], POSITION, Interpolation.BILINEAR),
        2 * n,
    )
    add(
        f"bicubic planar {tag}",
        lambda cv=cv, tile=tile: compose(cv, [tile], POSITION, Interpolation.BICUBIC),
        2 * n,
    )
    tile3 = c3[:5120, :5120].copy()
    cv3 = np.zeros((5122, 5122, 3), dt)
    add(
        f"bilinear hwc c3 {tag}",
        lambda cv3=cv3, tile3=tile3: compose(
            cv3, [tile3], POSITION, Interpolation.BILINEAR
        ),
        6 * n,
    )
    add(
        f"bicubic hwc c3 {tag}",
        lambda cv3=cv3, tile3=tile3: compose(
            cv3, [tile3], POSITION, Interpolation.BICUBIC
        ),
        6 * n,
    )

u16c4 = rng.integers(0, 65536, (8192, 8192, 4)).astype(np.uint16)
add(
    "box2 hwc c4 u16",
    lambda: downscale(u16c4, 2, ResizeFilter.BOX),
    8192 * 8192 * 4 * 2 * 1.25,
)
rgb = rng.integers(0, 256, (8192, 8192, 3), np.uint8)
add("rgb_to_rgbx", lambda: rgb_to_rgbx(rgb), 8192 * 8192 * 7)

print(f"\n{'kernel':<24}{'GB/s':>7}{'ms':>9}{'% of alloc-ceiling':>20}")
for name, gbs, ms in sorted(rows, key=lambda r: r[1]):
    print(f"{name:<24}{gbs:>7.1f}{ms:>9.1f}{gbs / ceil_alloc:>19.0%}")

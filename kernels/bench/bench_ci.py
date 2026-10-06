"""Tight paired benchmark with bootstrap CI: wsidicom_kernels vs cv2 on the contested
2x interleaved-RGB rows (+ planar u8 control). Strictly interleaved A/B pairs
so machine drift cancels in the per-pair ratio; 95% CI on the median ratio
via bootstrap. cv2 single-threaded. ratio = cv2/wsidicom_kernels (>1 = wsidicom_kernels wins).
"""

import gc
import time

import cv2
import numpy as np
from wsidicom_kernels import ResizeFilter, downscale

cv2.setNumThreads(1)
rng = np.random.default_rng(9)
boot_rng = np.random.default_rng(1234)

g8 = rng.integers(0, 256, (4096, 4096), np.uint8)
rgb8 = rng.integers(0, 256, (4096, 4096, 3), np.uint8)
rgb16 = rng.integers(0, 65536, (4096, 4096, 3)).astype(np.uint16)

PAIRS = 100


def sample(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def run_row(name, tess_fn, cv_fn):
    for _ in range(3):  # warmup both sides
        tess_fn()
        cv_fn()
    ts, cs = np.empty(PAIRS), np.empty(PAIRS)
    gc.disable()
    try:
        for i in range(PAIRS):
            ts[i] = sample(tess_fn)
            cs[i] = sample(cv_fn)
    finally:
        gc.enable()
    ratios = cs / ts
    med_r = np.median(ratios)
    idx = boot_rng.integers(0, PAIRS, (10000, PAIRS))
    meds = np.median(ratios[idx], axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    verdict = (
        "wsidicom_kernels faster"
        if lo > 1
        else ("cv2 faster" if hi < 1 else "tie (CI spans 1)")
    )
    print(
        f"{name:<22} tess {np.median(ts) * 1e3:6.2f} ms  cv2 {np.median(cs) * 1e3:6.2f} ms"
        f"   ratio {med_r:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  -> {verdict}"
    )


print(
    f"paired interleaved, n={PAIRS} pairs/row, cv2 1-thread, ratio = cv2/wsidicom_kernels"
)
run_row(
    "box2 planar u8 (ctrl)",
    lambda: downscale(g8, 2, ResizeFilter.BOX),
    lambda: cv2.resize(g8, (2048, 2048), interpolation=cv2.INTER_AREA),
)
run_row(
    "box2 rgb u8",
    lambda: downscale(rgb8, 2, ResizeFilter.BOX),
    lambda: cv2.resize(rgb8, (2048, 2048), interpolation=cv2.INTER_AREA),
)
run_row(
    "box2 rgb u16",
    lambda: downscale(rgb16, 2, ResizeFilter.BOX),
    lambda: cv2.resize(rgb16, (2048, 2048), interpolation=cv2.INTER_AREA),
)

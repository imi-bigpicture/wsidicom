"""GIL-release check: N Python THREADS running kernels concurrently vs the
same total work sequentially. If the extension held the GIL, parallel wall
time would equal sequential (scaling ~1.0x); released, it approaches the
bandwidth-limited process numbers.
"""

import threading
import time

import numpy as np
from wsidicom_kernels import Interpolation, ResizeFilter, compose, downscale

N = 6
OPS = 10


def run_all(imgs, parallel):
    def work(img):
        for _ in range(OPS):
            downscale(img, 2, ResizeFilter.BOX)

    t0 = time.perf_counter()
    if parallel:
        ts = [threading.Thread(target=work, args=(img,)) for img in imgs]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    else:
        for img in imgs:
            work(img)
    return time.perf_counter() - t0


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    imgs = [rng.integers(0, 256, (8192, 8192), np.uint8) for _ in range(N)]
    run_all(imgs, True)  # warm
    seq = min(run_all(imgs, False) for _ in range(3))
    par = min(run_all(imgs, True) for _ in range(3))
    print(f"box u8 8192², {N} workers x {OPS} ops:")
    print(
        f"  sequential {seq:5.2f} s   threaded {par:5.2f} s   scaling x{seq / par:.2f}"
    )

    # compose path too
    tile = rng.integers(0, 256, (5120, 5120), np.uint8)
    canvases = [np.zeros((5122, 5122), np.uint8) for _ in range(N)]
    position = np.array([[0.3, 0.7]])

    def work_c(cv):
        for _ in range(OPS):
            compose(cv, [tile], position, Interpolation.BILINEAR)

    t0 = time.perf_counter()
    for cv in canvases:
        work_c(cv)
    seq = time.perf_counter() - t0
    ts = [threading.Thread(target=work_c, args=(cv,)) for cv in canvases]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    par = time.perf_counter() - t0
    print(f"bilinear compose u8 5120², {N} workers x {OPS} ops:")
    print(
        f"  sequential {seq:5.2f} s   threaded {par:5.2f} s   scaling x{seq / par:.2f}"
    )

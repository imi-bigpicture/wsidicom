"""GIL-release check: N Python THREADS running kernels concurrently vs the
same total work sequentially. If the extension held the GIL, parallel wall
time would equal sequential (scaling ~1.0x); released, it approaches the
bandwidth-limited process numbers.
"""

import threading
import time

import numpy as np
from wsidicom_kernels import ResizeFilter, downscale

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

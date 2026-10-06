"""A tile-mosaic service workload, for profilers (not a timing benchmark).

Each "request" builds a 2048x2048 RGB image from overlapping tiles placed at
fractional offsets, then serves it two common ways:
  - thumbnail: compose to RGB, downscale 2x (box), expand to RGBX for Pillow
  - full size: compose straight into an RGBX canvas (fused expand)
with fresh output buffers per request, as a naive service would allocate
them (page faults on those buffers are about half the request time; see the
README's performance notes — this driver deliberately keeps that cost in).
Tile sizes alternate 256 and 512; filters alternate bilinear and bicubic.

Usage: python bench/tile_workload.py [seconds]
"""

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

REGION = 2048
rng = np.random.default_rng(0)


def tile_set(t):
    """Tiles covering the region with a ~1 px overlap at fractional offsets."""
    n = REGION // (t - 1) + 1
    tiles = [rng.integers(0, 256, (t, t, 3), np.uint8) for _ in range(n * n)]
    jitter = rng.uniform(0.1, 0.9, (n * n, 2))
    grid = np.array(
        [[(i % n) * (t - 1), (i // n) * (t - 1)] for i in range(n * n)], float
    )
    return tiles, grid + jitter - 1.0


SETS = {t: tile_set(t) for t in (256, 512)}


def request(t, filt):
    tiles, pos = SETS[t]
    rgb = np.empty((REGION, REGION, 3), np.uint8)
    compose(rgb, tiles, pos, filt)
    thumb = rgb_to_rgbx(downscale(rgb, 2, ResizeFilter.BOX))
    rgbx = np.empty((REGION, REGION, 4), np.uint8)
    compose(rgbx, tiles, pos, filt)
    return thumb, rgbx


if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < end:
        for t in (256, 512):
            for filt in (Interpolation.BILINEAR, Interpolation.BICUBIC):
                request(t, filt)
                n += 1
    print(f"{n} requests in {seconds:.0f} s")

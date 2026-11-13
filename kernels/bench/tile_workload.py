"""A tile-mosaic service workload, for profilers (not a timing benchmark).

Each "request" builds a 2048x2048 RGB image from tiles overlapping by one
pixel, then serves it two common ways:
  - thumbnail: compose to RGB, downscale 2x (box), expand to RGBX for Pillow
  - full size: compose straight into an RGBX canvas (fused expand)
with fresh output buffers per request, as a naive service would allocate
them (page faults on those buffers are about half the request time; see the
README's performance notes — this driver deliberately keeps that cost in).
Tile sizes alternate 256 and 512.

Usage: python bench/tile_workload.py [seconds]
"""

import sys
import time

import numpy as np
from wsidicom_kernels import (
    ResizeFilter,
    compose,
    downscale,
    rgb_to_rgbx,
)

REGION = 2048
rng = np.random.default_rng(0)


def tile_set(t):
    """Tiles covering the region with a 1 px overlap, the first row and column
    hanging 1 px off the top-left."""
    n = REGION // (t - 1) + 1
    tiles = [rng.integers(0, 256, (t, t, 3), np.uint8) for _ in range(n * n)]
    grid = np.array(
        [[(i % n) * (t - 1), (i // n) * (t - 1)] for i in range(n * n)], np.int64
    )
    return tiles, grid - 1


SETS = {t: tile_set(t) for t in (256, 512)}


def request(t):
    tiles, pos = SETS[t]
    rgb = np.empty((REGION, REGION, 3), np.uint8)
    compose(rgb, tiles, pos)
    thumb = rgb_to_rgbx(downscale(rgb, 2, ResizeFilter.BOX))
    rgbx = np.empty((REGION, REGION, 4), np.uint8)
    compose(rgbx, tiles, pos)
    return thumb, rgbx


if __name__ == "__main__":
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    end = time.perf_counter() + seconds
    n = 0
    while time.perf_counter() < end:
        for t in (256, 512):
            request(t)
            n += 1
    print(f"{n} requests in {seconds:.0f} s")

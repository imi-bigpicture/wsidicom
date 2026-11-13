"""Cross-library benchmark: wsidicom_kernels vs Pillow vs OpenCV vs libvips (pyvips).

Semantics matched per row (all antialiased equivalents). Single-threaded is
the kernel-quality comparison; run with VIPS_CONCURRENCY=1 + cv2 1 thread,
then again with defaults for the deployment view.
4096^2 input, mean +/- stdev in ms over RUNS runs after a warmup run.
Install the compared libraries with `uv sync --group bench`.
"""

import os
import statistics
import time

import cv2
import numpy as np
import pyvips
from PIL import Image
from wsidicom_kernels import (
    ResizeFilter,
    downscale,
    resize,
    rgb_to_rgbx,
)

RUNS = 15
SINGLE_THREADED = os.environ.get("VIPS_CONCURRENCY") == "1"
cv2.setNumThreads(1 if SINGLE_THREADED else cv2.getNumberOfCPUs())

rng = np.random.default_rng(9)
gray = rng.integers(0, 256, (4096, 4096), np.uint8)
rgb = rng.integers(0, 256, (4096, 4096, 3), np.uint8)
rgb16 = rng.integers(0, 65536, (4096, 4096, 3)).astype(np.uint16)
pillow_gray = Image.fromarray(gray, "L")
pillow_rgb = Image.fromarray(rgb, "RGB")


def vips_image(array):
    samples = 1 if array.ndim == 2 else array.shape[2]
    vips_format = "uchar" if array.dtype == np.uint8 else "ushort"
    return pyvips.Image.new_from_memory(
        array.tobytes(), array.shape[1], array.shape[0], samples, vips_format
    )


vips_gray = vips_image(gray)
vips_rgb = vips_image(rgb)
vips_rgb16 = vips_image(rgb16)


def timing(function):
    function()  # warmup
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        function()
        times.append((time.perf_counter() - start) * 1e3)
    return f"{statistics.mean(times):.1f}+/-{statistics.stdev(times):.1f}"


def row(name, kernels, pillow, opencv, vips):
    cells = [
        timing(function) if function else "-"
        for function in (kernels, pillow, opencv, vips)
    ]
    print(f"{name:<26}" + "".join(f"{cell:>18}" for cell in cells))


mode = (
    "single-threaded"
    if SINGLE_THREADED
    else "library-default threads (Pillow/wsidicom_kernels stay 1 thread)"
)
print(f"=== {mode}, ms, mean +/- stdev of {RUNS} runs, 4096^2 ===")
print(f"{'op':<26}{'wsidicom_kernels':>18}{'Pillow':>18}{'cv2':>18}{'libvips':>18}")

row(
    "box2 planar u8",
    lambda: downscale(gray, 2, ResizeFilter.BOX),
    lambda: pillow_gray.reduce(2),
    lambda: cv2.resize(gray, (2048, 2048), interpolation=cv2.INTER_AREA),
    lambda: vips_gray.shrink(2, 2).numpy(),
)
row(
    "box2 rgb u8",
    lambda: downscale(rgb, 2, ResizeFilter.BOX),
    lambda: pillow_rgb.reduce(2),
    lambda: cv2.resize(rgb, (2048, 2048), interpolation=cv2.INTER_AREA),
    lambda: vips_rgb.shrink(2, 2).numpy(),
)
row(
    "box2 rgb u16",
    lambda: downscale(rgb16, 2, ResizeFilter.BOX),
    None,  # Pillow has no 16-bit RGB mode
    lambda: cv2.resize(rgb16, (2048, 2048), interpolation=cv2.INTER_AREA),
    lambda: vips_rgb16.shrink(2, 2).numpy(),
)
row(
    "area resize rgb ->1517",
    lambda: resize(rgb, 1517, 1517, ResizeFilter.BOX),
    lambda: pillow_rgb.resize((1517, 1517), Image.Resampling.BOX),
    lambda: cv2.resize(rgb, (1517, 1517), interpolation=cv2.INTER_AREA),
    lambda: vips_rgb.resize(1517 / 4096, kernel="linear").numpy(),
)
row(
    "lanczos3 rgb ->1517",
    lambda: resize(rgb, 1517, 1517, ResizeFilter.LANCZOS3),
    lambda: pillow_rgb.resize((1517, 1517), Image.Resampling.LANCZOS),
    None,  # cv2 INTER_LANCZOS4 aliases when minifying — not comparable
    lambda: vips_rgb.resize(1517 / 4096, kernel="lanczos3").numpy(),
)


row(
    "rgb->rgbx (Pillow handoff)",
    lambda: rgb_to_rgbx(rgb),
    lambda: pillow_rgb.im.convert("RGBX"),
    None,
    None,
)

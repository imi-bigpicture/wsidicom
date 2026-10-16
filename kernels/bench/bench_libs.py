"""Cross-library benchmark: wsidicom_kernels vs Pillow vs OpenCV vs libvips (pyvips).

Semantics matched per row (all antialiased equivalents). Single-threaded is
the kernel-quality comparison; run with VIPS_CONCURRENCY=1 + cv2 1 thread,
then again with defaults for the deployment view.
u8, 4096^2 input, best-of-7.
"""

import os
import time

import cv2
import numpy as np
import pyvips
from PIL import Image
from wsidicom_kernels import (
    Interpolation,
    ResizeFilter,
    compose,
    downscale,
    resize,
    rgb_to_rgbx,
)

ST = os.environ.get("VIPS_CONCURRENCY") == "1"
cv2.setNumThreads(1 if ST else cv2.getNumberOfCPUs())

rng = np.random.default_rng(9)
gray = rng.integers(0, 256, (4096, 4096), np.uint8)
rgb = rng.integers(0, 256, (4096, 4096, 3), np.uint8)
pil_gray = Image.fromarray(gray, "L")
pil_rgb = Image.fromarray(rgb, "RGB")


def v_img(a):
    if a.ndim == 2:
        return pyvips.Image.new_from_memory(
            a.tobytes(), a.shape[1], a.shape[0], 1, "uchar"
        )
    return pyvips.Image.new_from_memory(
        a.tobytes(), a.shape[1], a.shape[0], a.shape[2], "uchar"
    )


vips_gray = v_img(gray)
vips_rgb = v_img(rgb)


def best(fn, k=7):
    b = float("inf")
    for _ in range(k):
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b * 1e3


def row(name, tess, pil, cv, vips):
    r = f"{name:<26}{best(tess):>9.1f}"
    r += f"{best(pil):>9.1f}" if pil else f"{'-':>9}"
    r += f"{best(cv):>9.1f}" if cv else f"{'-':>9}"
    r += f"{best(vips):>9.1f}" if vips else f"{'-':>9}"
    print(r)


mode = (
    "single-threaded"
    if ST
    else "library-default threads (PIL/wsidicom_kernels stay 1T)"
)
print(f"=== {mode}, ms, best of 7, 4096^2 u8 ===")
print(f"{'op':<26}{'wsidicom_kernels':>9}{'Pillow':>9}{'cv2':>9}{'libvips':>9}")

row(
    "box2 planar",
    lambda: downscale(gray, 2, ResizeFilter.BOX),
    lambda: pil_gray.reduce(2),
    lambda: cv2.resize(gray, (2048, 2048), interpolation=cv2.INTER_AREA),
    lambda: vips_gray.shrink(2, 2).numpy(),
)
row(
    "box2 rgb",
    lambda: downscale(rgb, 2, ResizeFilter.BOX),
    lambda: pil_rgb.reduce(2),
    lambda: cv2.resize(rgb, (2048, 2048), interpolation=cv2.INTER_AREA),
    lambda: vips_rgb.shrink(2, 2).numpy(),
)
row(
    "area resize rgb ->1517",
    lambda: resize(rgb, 1517, 1517, ResizeFilter.BOX),
    lambda: pil_rgb.resize((1517, 1517), Image.Resampling.BOX),
    lambda: cv2.resize(rgb, (1517, 1517), interpolation=cv2.INTER_AREA),
    lambda: vips_rgb.resize(1517 / 4096, kernel="linear").numpy(),
)
row(
    "lanczos3 rgb ->1517",
    lambda: resize(rgb, 1517, 1517, ResizeFilter.LANCZOS3),
    lambda: pil_rgb.resize((1517, 1517), Image.Resampling.LANCZOS),
    None,  # cv2 INTER_LANCZOS4 aliases when minifying — not comparable
    lambda: vips_rgb.resize(1517 / 4096, kernel="lanczos3").numpy(),
)


TRANSLATE_POSITION = np.array([[0.3, 0.7]])


def tess_translate():
    cv = np.zeros((4098, 4098), np.uint8)
    compose(cv, [gray], TRANSLATE_POSITION, Interpolation.BILINEAR)


M = np.float32([[1, 0, 0.3], [0, 1, 0.7]])
row(
    "subpx translate bilinear",
    tess_translate,
    lambda: pil_gray.transform(
        (4098, 4098),
        Image.Transform.AFFINE,
        (1, 0, -0.3, 0, 1, -0.7),
        Image.Resampling.BILINEAR,
    ),
    lambda: cv2.warpAffine(gray, M, (4098, 4098), flags=cv2.INTER_LINEAR),
    lambda: vips_gray.affine(
        [1, 0, 0, 1], odx=0.3, ody=0.7, interpolate=pyvips.Interpolate.new("bilinear")
    ).numpy(),
)
row(
    "rgb->rgbx (PIL handoff)",
    lambda: rgb_to_rgbx(rgb),
    lambda: pil_rgb.im.convert("RGBX"),
    None,
    None,
)

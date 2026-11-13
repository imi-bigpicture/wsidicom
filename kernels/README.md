# wsidicom-kernels

Fast tile compositing, resizing and downscaling for numpy arrays, written in
Rust with SIMD kernels — for building mosaics from tiles, image pyramids and
thumbnails:

- **`compose`** — paste tiles onto a canvas at integer positions, row by row
  (`memcpy`).
- **`stitch`** — lay equally sized tiles out on a grid into a new canvas (a
  convenience over `compose`).
- **`downscale`** — reduce by an integer factor: power-of-2 box (native SIMD,
  ~2× faster than Pillow `reduce`) and `bilinear` / `catmullrom` / `lanczos3`
  (Lanczos ~8× faster than Pillow LANCZOS). The scale is exactly the factor
  and the output `ceil(size / factor)`, as Pillow `reduce` and libvips
  `shrink`: where the factor doesn't divide the size, the last row and column
  cover what remains (with `box`, the mean of those pixels; bit-identical to
  Pillow `reduce(2)`), rather than the image being stretched or cropped.
- **`resize`** — any exact target size (fractional ratios, upscaling),
  antialiased at every ratio.
- All resampling is native, on every CPU: uint8 and uint16 within 1 LSB of
  the exact convolution at any ratio, float32 computed in f32 (see
  [What runs where](#what-runs-where)).

Works on **planar `(H, W)`** or **interleaved RGB `(H, W, 3)`** arrays, dtype
`uint8` / `uint16` / `float32`. Planar is the universal core — the only layout
that expresses subsampled chroma (YCbCr 4:2:0), so compose those per plane.
Overlaps resolve **last-write-wins** (no blending).

## Usage

Every image — tile, canvas or input — is a **numpy array**: planar `(H, W)` or
interleaved RGB `(H, W, 3)`, dtype `uint8`, `uint16` or `float32`.

```python
import numpy as np
from wsidicom_kernels import (
    ResizeFilter,
    compose,
    downscale,
    resize,
    stitch,
)

# 16 RGB tiles of 256 x 256 pixels.
rng = np.random.default_rng(0)
tiles = [rng.integers(0, 256, (256, 256, 3), np.uint8) for _ in range(16)]

# compose: paste tiles into an existing canvas array, in place.
# - canvas: array with the tiles' dtype and channel count.
# - positions: (N, 2) int64 array, one (x, y) per tile = (column, row) of
#   the tile's top-left corner in canvas pixels.
canvas = np.zeros((1024, 1024, 3), np.uint8)
positions = np.array([[(i % 4) * 255, (i // 4) * 255] for i in range(16)], np.int64)
compose(canvas, tiles, positions)

# stitch: equally sized planar tiles on a grid -> a new canvas array.
gray_tiles = [np.ascontiguousarray(t[..., 0]) for t in tiles]  # (256, 256) each
mosaic = stitch(gray_tiles, grid_width=4)  # (1024, 1024)

# downscale: reduce by an integer factor; image array in, new array out.
# Output size is ceil(size / factor): 1023 pixels by 2 gives 512.
half = downscale(canvas, 2, ResizeFilter.BOX)  # (512, 512, 3)
thumb = downscale(canvas, 4, ResizeFilter.LANCZOS3)  # (256, 256, 3)

# resize: resample to an exact (height, width), any ratio, up or down.
sized = resize(canvas, 300, 400, ResizeFilter.CATMULLROM)  # (300, 400, 3)
```

Where tiles overlap, the later one wins. Tiles may hang off the canvas; only
the pixels inside it are written.

Tiles may be slices of larger arrays (`tile[y0:y1, x0:x1]`), read without a
copy. Where one decoded tile packs several pieces placed apart, pass it once
with `crops`, an `(N, 5)` int64 array of `(tile, x, y, width, height)` rows,
one per position:

```python
crops = np.array([[0, 0, 0, 8, 11], [0, 8, 0, 8, 11]], np.int64)
positions = np.array([[10, 4], [18, 5]], np.int64)
compose(canvas, [decoded], positions, crops=crops)
```

## What runs where

wsidicom-kernels's own kernels do the compositing, the power-of-2 box
reductions and every convolution resize, on x86-64 and aarch64 alike; its
convolution follows [fast_image_resize](https://crates.io/crates/fast_image_resize)'s
filters and windows.

| Operation | Computed by |
| --- | --- |
| `compose`, `stitch` | wsidicom-kernels |
| `downscale` / `resize` with `"box"` at an exact power-of-2 factor | wsidicom-kernels |
| `downscale` / `resize` of **uint8**, with `box` / `bilinear` / `catmullrom` / `lanczos3` — any ratio up or down | wsidicom-kernels |
| `nearest`, every dtype (fast_image_resize's sample positions) | wsidicom-kernels |
| `downscale` / `resize` of **uint16 / float32**, with `box` / `bilinear` / `catmullrom` / `lanczos3` — any ratio (computed in f32; uint16 within 1 LSB of exact) | wsidicom-kernels |

The native path computes the same convolution as fast_image_resize (same
filters, windows and edge handling) but within 1 LSB of the exact result at
any ratio. uint8 runs in 16-bit fixed point down to 1/128 per axis, where
each filter tap rounds to the nearest 1/64 LSB; larger uint8 reductions take
the f32 engine of uint16 and float32 (2–3x slower than fixed point would be,
but fixed point's roundings would add up over the long windows). That engine
sums windows of more than 256 taps in chunks added in f64, so even a flat
image reduced by tens of thousands keeps its value. `downscale` with `box`
at a power-of-2 factor stays within 1 LSB at any factor, with the same
result on every CPU (x86-64 and aarch64, SIMD or scalar).
fast_image_resize rounds and clamps an intermediate: with `catmullrom` /
`lanczos3` it is off by up to ~20 LSB near sharp edges that touch 0 or 255
(elsewhere within 1 LSB too). The native results are identical on every CPU
(the AVX2 and NEON kernels and the scalar fallback compute the same
arithmetic), and an interleaved image gives exactly what
resizing each channel as a planar image gives.

Speed against fast_image_resize on x86-64 (Ryzen 7940HS; 512² and 2048²
images, 2x up to 1/8, all four filters; median and slowest case):

| | AVX2 |
| --- | --- |
| uint8 planar | 2.3x (1.1x) |
| uint8 RGB | 1.3x (0.9x) |
| uint16 / float32, every layout | 1.6-2.8x (1.0x) |

Every channel is filtered on its own.

## SIMD

Kernels dispatch at runtime: **AVX2** on x86-64 (via
`is_x86_feature_detected!`), **NEON** on aarch64 (Apple Silicon / ARM), and a
**scalar** fallback everywhere else. The crate builds at baseline x86-64, so one
binary runs on any CPU — full speed where SIMD exists, correct otherwise. No
global `target-cpu` needed.

Environment toggles (read once per process): `WSIDICOM_KERNELS_NT=0` disables non-temporal stores on large box outputs,
and `WSIDICOM_KERNELS_PREFAULT=0` disables the page-prefault helper thread for large
outputs.

The GIL is released around every kernel; parallelize across tiles with Python
threads (there is no internal compute threading).

## Performance notes

- **Reuse large output arrays.** A freshly allocated array pays a page fault
  per 4 KB page on first write — measured at ~0.3 ms per MB here, more than
  copying tiles into it. A service that composes a large image per request
  can keep a pool of canvases by shape instead of calling `np.empty` each
  time.
- **Pass all tiles of an image in one `compose` call**; per-call overhead is
  a few microseconds.
- **Parallelize with threads**, e.g. one output image per thread: the GIL is
  released inside every call.

## Build

Needs Rust, [uv](https://docs.astral.sh/uv/) and
[maturin](https://www.maturin.rs/). Then:

```bash
# macOS / Linux:
uv run --with maturin --with numpy --with pytest maturin develop --release

# Windows: use ./build.ps1 (wraps vcvars64), and install the Windows SDK once:
#   winget install Microsoft.WindowsSDK.10.0.28000
```

## Test

```bash
uv sync --all-packages      # from the repository root: the kernels dev group
uv run --all-packages pytest kernels/tests
```

One test file per operation (`test_compose`, `test_resize`, `test_downscale`,
`test_downscale_grid`), plus cross-cutting ones:

| file | checks |
| --- | --- |
| `test_compose_oracle.py` | compose against an independent numpy reference (bugs shared by SIMD and scalar) |
| `test_fuzz.py` | Hypothesis-generated shapes, positions, factors and filters: SIMD/scalar parity, inputs flush against an unmapped page so any out-of-bounds access crashes |
| `test_purity.py` | batched compose == single-tile calls; no state leaks between calls |
| `test_boundary.py` | saturation, checkerboards, float32 inf/NaN/denormals |
| `test_threads.py` | GIL-released concurrency: identical results, clean errors on contention |
| `test_pillow.py` | bit-identity with Pillow `reduce(2)`, partial edge boxes included |

Most tests compare the SIMD kernels with the forced-scalar reference
(`simd_and_scalar` in `conftest.py`).

A debug build (`maturin develop` without `--release`) additionally checks
every `get_unchecked` precondition at runtime; run the suite against one after
touching kernel bounds.

## Benchmark

`bench/` holds the benchmarks; `bench/bench.py [filter]` is the everyday one
(SIMD vs forced-scalar, interleaved, per kernel). Trust ratios measured within
one run, not absolute times across runs. `bench/bench_libs.py` compares against
Pillow, OpenCV and libvips; install those with `uv sync --group bench`.

# wsidicom-kernels

Fast, accurate resizing and downscaling for numpy arrays, written in Rust with
SIMD kernels — for image pyramids, reads of missing pyramid levels and
thumbnails:

- **`resize`** — any exact target size (fractional ratios, upscaling),
  antialiased at every ratio.
- **`downscale`** — reduce by an integer factor: power-of-2 box (native SIMD,
  ~2× faster than Pillow `reduce`) and `bilinear` / `catmullrom` / `lanczos3`
  (Lanczos ~8× faster than Pillow LANCZOS). The scale is exactly the factor
  and the output `ceil(size / factor)`, as Pillow `reduce` and libvips
  `shrink`: where the factor doesn't divide the size, the last row and column
  cover what remains (with `box`, the mean of those pixels; bit-identical to
  Pillow `reduce(2)`), rather than the image being stretched or cropped.
- **`downscale_grid`** — `downscale` of a grid of tiles stitched row-major,
  such as the four children of a pyramid tile, optionally into a filled frame
  of a given size; with `box` each tile is reduced straight into its place.
- All resampling is native, on every CPU: uint8 and uint16 within 1 LSB of
  the exact convolution at any ratio (see
  [What runs where](#what-runs-where)).

Works on **planar `(H, W)`** or **interleaved RGB `(H, W, 3)`** arrays, dtype
`uint8` / `uint16`. Subsampled chroma (YCbCr 4:2:0) is resized per plane.

## Usage

```python
import numpy as np
from wsidicom_kernels import (
    ResizeFilter,
    downscale,
    downscale_grid,
    resize,
)

rng = np.random.default_rng(0)
image = rng.integers(0, 256, (1023, 1023, 3), np.uint8)

# downscale: reduce by an integer factor; image array in, new array out.
# Output size is ceil(size / factor): 1023 pixels by 2 gives 512.
half = downscale(image, 2, ResizeFilter.BOX)  # (512, 512, 3)
quarter = downscale(image, 4, ResizeFilter.LANCZOS3)  # (256, 256, 3)

# resize: resample to an exact (height, width), any ratio, up or down.
sized = resize(image, 300, 400, ResizeFilter.CATMULLROM)  # (300, 400, 3)

# downscale_grid: four 512 x 512 child tiles (row-major, 2 per row) into one
# parent tile; size and fill frame a partial block at the image's edge.
children = [rng.integers(0, 256, (512, 512, 3), np.uint8) for _ in range(4)]
fill = np.array([255, 255, 255], np.uint8)
parent = downscale_grid(
    children, 2, 2, ResizeFilter.BOX, size=(512, 512), fill=fill
)  # (512, 512, 3)
```

## What runs where

wsidicom-kernels's own kernels do the power-of-2 box reductions and every
convolution resize, on x86-64 and aarch64 alike; its convolution follows [fast_image_resize](https://crates.io/crates/fast_image_resize)'s
filters and windows.

| Operation | Computed by |
| --- | --- |
| `downscale` / `resize` with `"box"` at an exact power-of-2 factor | wsidicom-kernels |
| `downscale` / `resize` of **uint8**, with `box` / `bilinear` / `catmullrom` / `lanczos3` — any ratio up or down | wsidicom-kernels |
| `nearest`, every dtype (fast_image_resize's sample positions) | wsidicom-kernels |
| `downscale` / `resize` of **uint16**, with `box` / `bilinear` / `catmullrom` / `lanczos3` — any ratio (computed in f32; within 1 LSB of exact) | wsidicom-kernels |

The native path computes the same convolution as fast_image_resize (same
filters, windows and edge handling) but within 1 LSB of the exact result at
any ratio. uint8 runs in 16-bit fixed point down to 1/128 per axis, where
each filter tap rounds to the nearest 1/64 LSB; larger uint8 reductions take
the f32 engine of uint16 (2–3x slower than fixed point would be,
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
| uint16, every layout | 1.6-2.8x (1.0x) |

Every channel is filtered on its own.

## SIMD

Kernels dispatch at runtime: **AVX2** on x86-64 (via
`is_x86_feature_detected!`), **NEON** on aarch64 (Apple Silicon / ARM), and a
**scalar** fallback everywhere else. The crate builds at baseline x86-64, so one
binary runs on any CPU — full speed where SIMD exists, correct otherwise. No
global `target-cpu` needed.

Environment toggle (read once per process): `WSIDICOM_KERNELS_NT=0` disables
non-temporal stores on large box outputs.

The GIL is released around every kernel; parallelize across tiles with Python
threads (there is no internal compute threading).

## Performance notes

- **Parallelize with threads**, e.g. one output tile per thread: the GIL is
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

One test file per operation (`test_resize`, `test_downscale`,
`test_downscale_grid`), plus cross-cutting ones:

| file | checks |
| --- | --- |
| `test_fuzz.py` | Hypothesis-generated shapes, factors and filters: resize and downscale SIMD/scalar parity, inputs flush against an unmapped page so any out-of-bounds access crashes |
| `test_state.py` | resize, downscale and downscale_grid keep no state: recycled outputs full of junk, plans cached across SIMD toggles, other calls in between |
| `test_boundary.py` | saturation, checkerboards |
| `test_threads.py` | GIL-released concurrency: identical results on every thread |
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

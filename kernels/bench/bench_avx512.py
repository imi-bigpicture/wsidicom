"""AVX-512 vs AVX2 tier A/B — one kernel per subprocess, ABBA-interleaved, with
a paired bootstrap CI.

The AVX-512 gate (has_avx512) is read once per process via LazyLock, so the
tiers can't be toggled in-process; each measurement is a fresh subprocess with
WSIDICOM_KERNELS_AVX512 unset (512 on) or =0 (drop to AVX2). Both sides are the SAME
build, so machine drift hits them equally and cancels in the per-pair ratio —
but only with discipline:
  - ONE kernel per subprocess: no cross-kernel thermal/ordering contamination
    (a shared-process design let the faster-tier run reach later kernels in a
    different thermal state; the u16 no-512 control exposed it as a phantom 4%).
  - ABBA order per pair: cancels monotonic warmup/turbo drift.
  - Trust the bootstrap CI over >= ~12 pairs, never the median or an eyeball
    pattern (a 6-pair "win" once turned out to be pure drift).

Controls that MUST stay ~1.0 (ratio can't move — no 512 path exists): every
u16_* kernel, and cold_box2x_u8 (memory-bound). If a control drifts off 1.0,
the measurement is contaminated — distrust the whole run.

TWO working-set sizes, because single-core memory bandwidth varies ~2x across
machines (Zen 4 mobile ~40 GB/s vs Cascade Lake ~15-20 GB/s single-thread):
  - "hot"  1024^2: input+output fit in L3, so warm reps never hit DRAM. This
           isolates COMPUTE throughput — the true AVX-512 FMA effect, free of
           the per-machine bandwidth regime.
  - "cold" 5120^2: streams from DRAM, the deployment-realistic picture. A
           memory-bound kernel shows ratio ~1.0 here even where 512 wins hot.
Read the HOT ratio for "does 512 help the math"; the COLD ratio for "does it
help this machine at this size". Run bench/roofline.py first to know the box's
single-core bandwidth ceiling.

Usage: python bench/bench_avx512.py [pairs]        (default 16)
       python bench/bench_avx512.py --measure KEY  (internal: time one kernel, print ms)
"""

import os
import statistics
import subprocess
import sys

import numpy as np

# kernel key -> (size tag, dtype, filter | "box")
KERNELS = {}
for tag in ("hot", "cold"):
    for dt in ("u8", "u16", "f32"):
        for filt in ("bilinear", "bicubic"):
            KERNELS[f"{tag}_{dt}_{filt}"] = (tag, dt, filt)
KERNELS["cold_box2x_u8"] = ("cold", "u8", "box")

NPDT = {"u8": np.uint8, "u16": np.uint16, "f32": np.float32}
HIVAL = {"u8": 256, "u16": 65536, "f32": None}
SIZE = {"hot": 1024, "cold": 5120}


def measure(key):
    """Time one kernel in this (already tier-configured) process; print ms/call."""
    import time

    from wsidicom_kernels import Interpolation, ResizeFilter, compose, downscale

    tag, dt, filt = KERNELS[key]
    rng = np.random.default_rng(3)
    if filt == "box":
        img = rng.integers(0, 256, (8192, 8192), np.uint8)

        def fn():
            return downscale(img, 2, ResizeFilter.BOX)

        reps, inner = 7, 1
    else:
        n = SIZE[tag]
        hi = HIVAL[dt]
        tile = (
            rng.integers(0, hi, (n, n)).astype(NPDT[dt])
            if hi
            else rng.random((n, n), np.float32)
        )
        cv = np.zeros((n + 8, n + 8), NPDT[dt])
        position = np.array([[0.37, 0.61]])
        interpolation = Interpolation(filt)

        def fn():
            return compose(cv, [tile], position, interpolation)

        reps, inner = (11, 40) if tag == "hot" else (7, 1)
    fn()  # warm (pulls the working set into cache for the hot sizes)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(inner):
            fn()
        ts.append((time.perf_counter() - t) / inner)
    ts.sort()
    print(f"{ts[len(ts) // 2] * 1000:.5f}")


def run(key, avx512_on):
    env = dict(os.environ)
    if avx512_on:
        env.pop("WSIDICOM_KERNELS_AVX512", None)
    else:
        env["WSIDICOM_KERNELS_AVX512"] = "0"
    out = subprocess.run(
        [sys.executable, __file__, "--measure", key],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def main():
    pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    boot_rng = np.random.default_rng(0)
    print(
        f"AVX-512 vs AVX2, one kernel/subprocess, {pairs} ABBA pairs (ratio < 1 = 512 faster)\n"
    )
    for key in KERNELS:
        r512, r256 = [], []
        for p in range(pairs):
            if p % 2 == 0:  # ABBA: alternate which tier runs first
                r512.append(run(key, True))
                r256.append(run(key, False))
            else:
                r256.append(run(key, False))
                r512.append(run(key, True))
        ratios = [r512[i] / r256[i] for i in range(pairs)]
        boot = [
            statistics.median(boot_rng.choice(ratios, len(ratios))) for _ in range(5000)
        ]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        tag = (
            "  <-- 512 win"
            if hi < 1.0
            else ("  <-- 512 LOSS" if lo > 1.0 else "  (neutral)")
        )
        is_control = ("u16" in key) or key == "cold_box2x_u8"
        flag = "  [CONTROL: must be ~1.0]" if is_control else ""
        print(
            f"{key:17s} 512 {statistics.median(r512):8.4f}  256 {statistics.median(r256):8.4f}"
            f"  ratio {statistics.median(ratios):.3f}  CI[{lo:.3f},{hi:.3f}]{tag}{flag}"
        )


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--measure":
        measure(sys.argv[2])
    else:
        main()

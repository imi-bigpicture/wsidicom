//! NEON box reduction kernels (aarch64): the 2x2 box, planar and interleaved
//! (RGB and 4-sample pixels, split into channel planes by `vldN`), RGB in
//! with RGBX out, and the fused 4x4 box.
//!
//! Integer kernels sum with `vpaddl`/`vpadal` (widening pairwise adds) and
//! round with `vrshrn` (rounding shift and narrow): `(sum + 2) >> 2` and
//! `(sum + 8) >> 4`, the scalar reference's exact single rounding. Float32
//! kernels add the rows first, then the column pairs: the scalar order.
//!
//! # Safety
//! NEON is baseline on aarch64, gated only by the `set_simd` kill switch.
//! Every `unsafe fn` here requires `src.len() >= width * height * channels`.
//! Loads and stores are justified against that at their loop guards.

use std::arch::aarch64::{
    float32x2_t, float32x2x3_t, float32x2x4_t, float32x4_t, uint8x8_t, uint8x8x3_t, uint8x8x4_t,
    uint8x16_t, uint8x16x3_t, uint16x4_t, uint16x4x3_t, uint16x4x4_t, uint16x8_t, vaddq_f32,
    vaddq_u32, vcombine_f32, vcombine_u8, vcombine_u16, vdup_n_u8, vget_high_f32, vget_low_f32,
    vld1q_f32, vld1q_u8, vld1q_u16, vld3q_f32, vld3q_u8, vld3q_u16, vld4q_f32, vld4q_u8, vld4q_u16,
    vmul_n_f32, vmulq_n_f32, vpadalq_u8, vpadalq_u16, vpadd_f32, vpaddlq_u8, vpaddlq_u16,
    vpaddq_f32, vpaddq_u16, vpaddq_u32, vrshrn_n_u16, vrshrn_n_u32, vst1_u8, vst1q_f32, vst1q_u8,
    vst1q_u16, vst3_f32, vst3_u8, vst3_u16, vst3q_u8, vst4_f32, vst4_u8, vst4_u16,
};

use std::arch::aarch64::{
    uint32x4_t, vaddl_high_u8, vaddl_u8, vaddq_u16, vdup_n_u16, vget_low_u8, vreinterpretq_u16_u64,
    vreinterpretq_u64_u16, vuzp1q_u64, vuzp2q_u64,
};

use super::{BoxSample, box2_mean};
use crate::cpu::{unwritten_output, written_output};
use crate::sample::{RGB, RGBX};

/// Rounded 2x2 means of 16 u8 columns of two rows: 8 outputs.
#[target_feature(enable = "neon")]
#[inline]
fn mean2x2_u8(upper: uint8x16_t, lower: uint8x16_t) -> uint8x8_t {
    vrshrn_n_u16::<2>(vpadalq_u8(vpaddlq_u8(upper), lower))
}

/// Rounded 2x2 means of 32 u8 columns of two rows (each as two registers):
/// 16 outputs.
#[target_feature(enable = "neon")]
#[inline]
fn mean2x2_u8x32(upper: [uint8x16_t; 2], lower: [uint8x16_t; 2]) -> uint8x16_t {
    vcombine_u8(
        mean2x2_u8(upper[0], lower[0]),
        mean2x2_u8(upper[1], lower[1]),
    )
}

/// Rounded 2x2 means of 8 u16 columns of two rows: 4 outputs.
#[target_feature(enable = "neon")]
#[inline]
fn mean2x2_u16(upper: uint16x8_t, lower: uint16x8_t) -> uint16x4_t {
    vrshrn_n_u32::<2>(vaddq_u32(vpaddlq_u16(upper), vpaddlq_u16(lower)))
}

/// 2x2 means of 4 f32 columns of two rows: 2 outputs.
#[target_feature(enable = "neon")]
#[inline]
fn mean2x2_f32(upper: float32x4_t, lower: float32x4_t) -> float32x2_t {
    let column_sums = vaddq_f32(upper, lower);
    vmul_n_f32(
        vpadd_f32(vget_low_f32(column_sums), vget_high_f32(column_sums)),
        0.25,
    )
}

/// Output rows of a 2x box: `(width / 2, height / 2)`.
fn halved(width: usize, height: usize) -> (usize, usize) {
    (width / 2, height / 2)
}

/// NEON uint8 2x2 box: 32 outputs per step; scalar tail.
///
/// # Safety
/// `src` holds `width * height` samples and `dst_ptr` the `height / 2` output
/// rows of `width / 2` samples, `dst_stride` samples apart.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box2x_u8_into_neon(
    src: &[u8],
    width: usize,
    height: usize,
    dst_ptr: *mut u8,
    dst_stride: usize,
) {
    let (ow, oh) = halved(width, height);
    let src_ptr = src.as_ptr();
    for oy in 0..oh {
        let (top, bottom, out_row) = (2 * oy * width, (2 * oy + 1) * width, oy * dst_stride);
        let mut ox = 0usize;
        while ox + 32 <= ow {
            let x = 2 * ox;
            // SAFETY: x + 64 <= 2 ow <= width: 64 samples of both rows; the
            // stores write outputs ox..ox + 32 of the row.
            unsafe {
                let load = |offset: usize| {
                    [
                        vld1q_u8(src_ptr.add(offset)),
                        vld1q_u8(src_ptr.add(offset + 16)),
                    ]
                };
                let low = mean2x2_u8x32(load(top + x), load(bottom + x));
                let high = mean2x2_u8x32(load(top + x + 32), load(bottom + x + 32));
                vst1q_u8(dst_ptr.add(out_row + ox), low);
                vst1q_u8(dst_ptr.add(out_row + ox + 16), high);
            }
            ox += 32;
        }
        for ox in ox..ow {
            // SAFETY: ox < width / 2, oy < height / 2: the 2x2 block is in src.
            unsafe { *dst_ptr.add(out_row + ox) = box2_mean(src_ptr.add(top + 2 * ox), 1, width) };
        }
    }
}

/// NEON uint16 2x2 box: 8 outputs per step; scalar tail.
///
/// # Safety
/// `src` holds `width * height` samples and `dst_ptr` the `height / 2` output
/// rows of `width / 2` samples, `dst_stride` samples apart.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box2x_u16_into_neon(
    src: &[u16],
    width: usize,
    height: usize,
    dst_ptr: *mut u16,
    dst_stride: usize,
) {
    let (ow, oh) = halved(width, height);
    let src_ptr = src.as_ptr();
    for oy in 0..oh {
        let (top, bottom, out_row) = (2 * oy * width, (2 * oy + 1) * width, oy * dst_stride);
        let mut ox = 0usize;
        while ox + 8 <= ow {
            let x = 2 * ox;
            // SAFETY: x + 16 <= 2 ow <= width: 16 samples of both rows; the
            // store writes outputs ox..ox + 8 of the row.
            unsafe {
                let load = |offset: usize| vld1q_u16(src_ptr.add(offset));
                let means = vcombine_u16(
                    mean2x2_u16(load(top + x), load(bottom + x)),
                    mean2x2_u16(load(top + x + 8), load(bottom + x + 8)),
                );
                vst1q_u16(dst_ptr.add(out_row + ox), means);
            }
            ox += 8;
        }
        for ox in ox..ow {
            // SAFETY: ox < width / 2, oy < height / 2: the 2x2 block is in src.
            unsafe { *dst_ptr.add(out_row + ox) = box2_mean(src_ptr.add(top + 2 * ox), 1, width) };
        }
    }
}

/// NEON float32 2x2 box: 4 outputs per step; scalar tail.
///
/// # Safety
/// `src` holds `width * height` samples and `dst_ptr` the `height / 2` output
/// rows of `width / 2` samples, `dst_stride` samples apart.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box2x_f32_into_neon(
    src: &[f32],
    width: usize,
    height: usize,
    dst_ptr: *mut f32,
    dst_stride: usize,
) {
    let (ow, oh) = halved(width, height);
    let src_ptr = src.as_ptr();
    for oy in 0..oh {
        let (top, bottom, out_row) = (2 * oy * width, (2 * oy + 1) * width, oy * dst_stride);
        let mut ox = 0usize;
        while ox + 4 <= ow {
            let x = 2 * ox;
            // SAFETY: x + 8 <= 2 ow <= width: 8 samples of both rows; the
            // store writes outputs ox..ox + 4 of the row.
            unsafe {
                let load = |offset: usize| vld1q_f32(src_ptr.add(offset));
                let means = vcombine_f32(
                    mean2x2_f32(load(top + x), load(bottom + x)),
                    mean2x2_f32(load(top + x + 4), load(bottom + x + 4)),
                );
                vst1q_f32(dst_ptr.add(out_row + ox), means);
            }
            ox += 4;
        }
        for ox in ox..ow {
            // SAFETY: ox < width / 2, oy < height / 2: the 2x2 block is in src.
            unsafe { *dst_ptr.add(out_row + ox) = box2_mean(src_ptr.add(top + 2 * ox), 1, width) };
        }
    }
}

/// A 2x box kernel writing output row `oy` at `dst_ptr + oy * dst_stride`.
type Box2xInto<T> = unsafe fn(&[T], usize, usize, *mut T, usize);

/// `kernel`'s 2x box of `src` into a new image of `channels`-sample pixels.
///
/// # Safety
/// `src` holds `width * height` pixels.
#[target_feature(enable = "neon")]
unsafe fn box2x_allocated<T>(
    kernel: Box2xInto<T>,
    src: &[T],
    (width, height, channels): (usize, usize, usize),
) -> (Vec<T>, usize, usize) {
    let (ow, oh) = halved(width, height);
    let mut out = unwritten_output::<T>(oh * ow * channels);
    let dst_ptr = out.as_mut_ptr().cast::<T>();
    // SAFETY: the caller's bound on src; out holds the oh rows of ow pixels,
    // ow * channels samples apart.
    unsafe { kernel(src, width, height, dst_ptr, ow * channels) };
    // SAFETY: the kernel wrote every output row in full.
    (unsafe { written_output(out) }, ow, oh)
}

/// NEON uint8 2x2 box into a new image.
///
/// # Safety
/// `src` holds `width * height` samples.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box2x_u8_neon(
    src: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, usize, usize) {
    // SAFETY: the caller's bound.
    unsafe { box2x_allocated(box2x_u8_into_neon, src, (width, height, 1)) }
}

/// NEON uint16 2x2 box into a new image.
///
/// # Safety
/// `src` holds `width * height` samples.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box2x_u16_neon(
    src: &[u16],
    width: usize,
    height: usize,
) -> (Vec<u16>, usize, usize) {
    // SAFETY: the caller's bound.
    unsafe { box2x_allocated(box2x_u16_into_neon, src, (width, height, 1)) }
}

/// NEON float32 2x2 box into a new image.
///
/// # Safety
/// `src` holds `width * height` samples.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box2x_f32_neon(
    src: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, usize, usize) {
    // SAFETY: the caller's bound.
    unsafe { box2x_allocated(box2x_f32_into_neon, src, (width, height, 1)) }
}

// Interleaved 2x2 box for one channel count: `$load` (vld3/vld4) splits
// 2 * `$step_px` source pixels into channel planes, `$mean` reduces each
// plane's two rows, and `$store` (vst3/vst4) interleaves the `$step_px`
// output pixels.
macro_rules! gen_box2x_hwc_neon {
    (
        $name:ident, $t:ty, $channels:expr, $step_px:expr,
        $load:ident, $store:ident, $planes:ident, $mean:ident, $($plane:tt),+
    ) => {
        /// # Safety
        /// `src` holds `width * height` pixels and `dst_ptr` the `height / 2`
        /// output rows of `width / 2` pixels, `dst_stride` samples apart.
        #[target_feature(enable = "neon")]
        unsafe fn $name(
            src: &[$t],
            width: usize,
            height: usize,
            dst_ptr: *mut $t,
            dst_stride: usize,
        ) {
            const CHANNELS: usize = $channels;
            let (ow, oh) = halved(width, height);
            let stride = width * CHANNELS;
            let src_ptr = src.as_ptr();
            for oy in 0..oh {
                let (top, bottom) = (2 * oy * stride, (2 * oy + 1) * stride);
                let out_row = oy * dst_stride;
                let mut ox = 0usize;
                while ox + $step_px <= ow {
                    let offset = 2 * ox * CHANNELS;
                    // SAFETY: ox + step <= ow <= width / 2: the loads read
                    // 2 step pixels of both rows, the store writes output
                    // pixels ox..ox + step of the row.
                    unsafe {
                        let upper = $load(src_ptr.add(top + offset));
                        let lower = $load(src_ptr.add(bottom + offset));
                        let means = $planes($($mean(upper.$plane, lower.$plane)),+);
                        $store(dst_ptr.add(out_row + ox * CHANNELS), means);
                    }
                    ox += $step_px;
                }
                for ox in ox..ow {
                    let top_left = top + 2 * ox * CHANNELS;
                    for channel in 0..CHANNELS {
                        // SAFETY: ox < width / 2, oy < height / 2: the 2x2
                        // block of this sample is in src.
                        unsafe {
                            *dst_ptr.add(out_row + ox * CHANNELS + channel) =
                                box2_mean(src_ptr.add(top_left + channel), CHANNELS, stride);
                        }
                    }
                }
            }
        }
    };
}

// The uint8 interleaved 2x box: 16 output pixels per step (two `vldN`
// loads of 16 pixels per row), stored by one full-width `vstN`.
macro_rules! gen_box2x_hwc_u8_neon {
    ($name:ident, $channels:expr, $load:ident, $store:ident, $planes:ident, $($plane:tt),+) => {
        /// # Safety
        /// `src` holds `width * height` pixels and `dst_ptr` the `height / 2`
        /// output rows of `width / 2` pixels, `dst_stride` samples apart.
        #[target_feature(enable = "neon")]
        unsafe fn $name(
            src: &[u8],
            width: usize,
            height: usize,
            dst_ptr: *mut u8,
            dst_stride: usize,
        ) {
            const CHANNELS: usize = $channels;
            const STEP_PX: usize = 16;
            let (ow, oh) = halved(width, height);
            let stride = width * CHANNELS;
            let src_ptr = src.as_ptr();
            for oy in 0..oh {
                let (top, bottom) = (2 * oy * stride, (2 * oy + 1) * stride);
                let out_row = oy * dst_stride;
                // Output pixels ox..ox + 16.
                let step = |ox: usize| {
                    let offset = 2 * ox * CHANNELS;
                    // SAFETY: ox + 16 <= ow <= width / 2 (the callers' loop
                    // guards): the loads read 32 pixels of both rows, the
                    // store writes output pixels ox..ox + 16 of the row.
                    unsafe {
                        let load = |at: usize| {
                            [$load(src_ptr.add(at)), $load(src_ptr.add(at + STEP_PX * CHANNELS))]
                        };
                        let (upper, lower) = (load(top + offset), load(bottom + offset));
                        let means = $planes($(mean2x2_u8x32(
                            [upper[0].$plane, upper[1].$plane],
                            [lower[0].$plane, lower[1].$plane],
                        )),+);
                        $store(dst_ptr.add(out_row + ox * CHANNELS), means);
                    }
                };
                // Two steps per pass while they fit, so more loads are in
                // flight.
                let mut ox = 0usize;
                while ox + 2 * STEP_PX <= ow {
                    step(ox);
                    step(ox + STEP_PX);
                    ox += 2 * STEP_PX;
                }
                if ox + STEP_PX <= ow {
                    step(ox);
                    ox += STEP_PX;
                }
                for ox in ox..ow {
                    let top_left = top + 2 * ox * CHANNELS;
                    for channel in 0..CHANNELS {
                        // SAFETY: ox < width / 2, oy < height / 2: the 2x2
                        // block of this sample is in src.
                        unsafe {
                            *dst_ptr.add(out_row + ox * CHANNELS + channel) =
                                box2_mean(src_ptr.add(top_left + channel), CHANNELS, stride);
                        }
                    }
                }
            }
        }
    };
}

gen_box2x_hwc_u8_neon!(
    box2x_u8_hwc3_neon,
    RGB,
    vld3q_u8,
    vst3q_u8,
    uint8x16x3_t,
    0,
    1,
    2
);
/// Rounded 2x2 means of 4 RGBX pixels of two rows: 2 output pixels. A
/// pixel's 4 widened sums are one 64-bit lane, so `uzp1`/`uzp2` on 64-bit
/// lanes split the even and odd pixels of the pairs.
#[target_feature(enable = "neon")]
#[inline]
fn mean2x2_rgbx(upper: uint8x16_t, lower: uint8x16_t) -> uint8x8_t {
    let front = vreinterpretq_u64_u16(vaddl_u8(vget_low_u8(upper), vget_low_u8(lower)));
    let back = vreinterpretq_u64_u16(vaddl_high_u8(upper, lower));
    let even = vreinterpretq_u16_u64(vuzp1q_u64(front, back));
    let odd = vreinterpretq_u16_u64(vuzp2q_u64(front, back));
    vrshrn_n_u16::<2>(vaddq_u16(even, odd))
}

/// NEON uint8 RGBX 2x2 box without splitting channels: 8 output pixels per
/// step; scalar tail.
///
/// # Safety
/// `src` holds `width * height` pixels and `dst_ptr` the `height / 2` output
/// rows of `width / 2` pixels, `dst_stride` samples apart.
#[target_feature(enable = "neon")]
unsafe fn box2x_u8_hwc4_neon(
    src: &[u8],
    width: usize,
    height: usize,
    dst_ptr: *mut u8,
    dst_stride: usize,
) {
    const STEP_PX: usize = 8;
    let (ow, oh) = halved(width, height);
    let stride = width * RGBX;
    let src_ptr = src.as_ptr();
    for oy in 0..oh {
        let (top, bottom) = (2 * oy * stride, (2 * oy + 1) * stride);
        let out_row = oy * dst_stride;
        let mut ox = 0usize;
        while ox + STEP_PX <= ow {
            let offset = 2 * ox * RGBX;
            // SAFETY: ox + 8 <= ow <= width / 2: the loads read 16 pixels of
            // both rows, the stores write output pixels ox..ox + 8.
            unsafe {
                let load =
                    |row: usize, block: usize| vld1q_u8(src_ptr.add(row + offset + 16 * block));
                let means = |first: usize| {
                    vcombine_u8(
                        mean2x2_rgbx(load(top, first), load(bottom, first)),
                        mean2x2_rgbx(load(top, first + 1), load(bottom, first + 1)),
                    )
                };
                vst1q_u8(dst_ptr.add(out_row + ox * RGBX), means(0));
                vst1q_u8(dst_ptr.add(out_row + ox * RGBX + 16), means(2));
            }
            ox += STEP_PX;
        }
        for ox in ox..ow {
            let top_left = top + 2 * ox * RGBX;
            for channel in 0..RGBX {
                // SAFETY: ox < width / 2, oy < height / 2: the block is in src.
                unsafe {
                    *dst_ptr.add(out_row + ox * RGBX + channel) =
                        box2_mean(src_ptr.add(top_left + channel), RGBX, stride);
                }
            }
        }
    }
}
gen_box2x_hwc_neon!(
    box2x_u16_hwc3_neon,
    u16,
    RGB,
    4,
    vld3q_u16,
    vst3_u16,
    uint16x4x3_t,
    mean2x2_u16,
    0,
    1,
    2
);
gen_box2x_hwc_neon!(
    box2x_u16_hwc4_neon,
    u16,
    RGBX,
    4,
    vld4q_u16,
    vst4_u16,
    uint16x4x4_t,
    mean2x2_u16,
    0,
    1,
    2,
    3
);
gen_box2x_hwc_neon!(
    box2x_f32_hwc3_neon,
    f32,
    RGB,
    2,
    vld3q_f32,
    vst3_f32,
    float32x2x3_t,
    mean2x2_f32,
    0,
    1,
    2
);
gen_box2x_hwc_neon!(
    box2x_f32_hwc4_neon,
    f32,
    RGBX,
    2,
    vld4q_f32,
    vst4_f32,
    float32x2x4_t,
    mean2x2_f32,
    0,
    1,
    2,
    3
);

// The interleaved 2x box of one sample type: 3 or 4 channels, into a new
// image (`$name`) or rows `dst_stride` apart (`$into`).
macro_rules! gen_box2x_hwc_dispatch {
    ($name:ident, $into:ident, $t:ty, $hwc3:ident, $hwc4:ident) => {
        /// # Safety
        /// `src` holds `width * height * channels` samples, `channels` 3 or 4.
        #[target_feature(enable = "neon")]
        pub(crate) unsafe fn $name(
            src: &[$t],
            width: usize,
            height: usize,
            channels: usize,
        ) -> (Vec<$t>, usize, usize) {
            let kernel: Box2xInto<$t> = if channels == RGB { $hwc3 } else { $hwc4 };
            // SAFETY: the caller's bounds, for the channel count's kernel.
            unsafe { box2x_allocated(kernel, src, (width, height, channels)) }
        }

        /// # Safety
        /// `src` holds `width * height * channels` samples, `channels` 3 or
        /// 4, and `dst_ptr` the `height / 2` output rows of `width / 2`
        /// pixels, `dst_stride` samples apart.
        #[target_feature(enable = "neon")]
        pub(crate) unsafe fn $into(
            src: &[$t],
            (width, height, channels): (usize, usize, usize),
            dst_ptr: *mut $t,
            dst_stride: usize,
        ) {
            // SAFETY: the caller's bounds, for the channel count's kernel.
            unsafe {
                if channels == RGB {
                    $hwc3(src, width, height, dst_ptr, dst_stride);
                } else {
                    $hwc4(src, width, height, dst_ptr, dst_stride);
                }
            }
        }
    };
}

gen_box2x_hwc_dispatch!(
    box2x_u8_hwc_neon,
    box2x_u8_hwc_into_neon,
    u8,
    box2x_u8_hwc3_neon,
    box2x_u8_hwc4_neon
);
gen_box2x_hwc_dispatch!(
    box2x_u16_hwc_neon,
    box2x_u16_hwc_into_neon,
    u16,
    box2x_u16_hwc3_neon,
    box2x_u16_hwc4_neon
);
gen_box2x_hwc_dispatch!(
    box2x_f32_hwc_neon,
    box2x_f32_hwc_into_neon,
    f32,
    box2x_f32_hwc3_neon,
    box2x_f32_hwc4_neon
);

/// NEON uint8 2x2 box, RGB in, RGBX out (`pad` in X): the RGB kernel's
/// planes stored by `vst4` with a pad plane. 8 output pixels per step;
/// scalar tail.
///
/// # Safety
/// `src` holds `width * height * 3` samples.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box2x_u8_hwc3to4_neon(
    src: &[u8],
    width: usize,
    height: usize,
    pad: u8,
) -> (Vec<u8>, usize, usize) {
    let (ow, oh) = halved(width, height);
    let (stride, out_stride) = (width * RGB, ow * RGBX);
    let mut out = unwritten_output::<u8>(oh * out_stride);
    let (src_ptr, dst_ptr) = (src.as_ptr(), out.as_mut_ptr().cast::<u8>());
    let pad_plane = vdup_n_u8(pad);
    for oy in 0..oh {
        let (top, bottom) = (2 * oy * stride, (2 * oy + 1) * stride);
        let out_row = oy * out_stride;
        let mut ox = 0usize;
        while ox + 8 <= ow {
            let offset = 2 * ox * RGB;
            // SAFETY: ox + 8 <= ow <= width / 2: the loads read 16 pixels of
            // both rows, the store writes output pixels ox..ox + 8 of the row.
            unsafe {
                let (upper, lower) = (
                    vld3q_u8(src_ptr.add(top + offset)),
                    vld3q_u8(src_ptr.add(bottom + offset)),
                );
                let rgbx = uint8x8x4_t(
                    mean2x2_u8(upper.0, lower.0),
                    mean2x2_u8(upper.1, lower.1),
                    mean2x2_u8(upper.2, lower.2),
                    pad_plane,
                );
                vst4_u8(dst_ptr.add(out_row + ox * RGBX), rgbx);
            }
            ox += 8;
        }
        for ox in ox..ow {
            let top_left = top + 2 * ox * RGB;
            for channel in 0..RGB {
                // SAFETY: ox < width / 2, oy < height / 2: the 2x2 block of
                // this sample is in src; the output pixel is in the row.
                unsafe {
                    *dst_ptr.add(out_row + ox * RGBX + channel) =
                        box2_mean(src_ptr.add(top_left + channel), RGB, stride);
                }
            }
            // SAFETY: as above.
            unsafe { *dst_ptr.add(out_row + ox * RGBX + RGB) = pad };
        }
    }
    // SAFETY: every output row was written in full.
    (unsafe { written_output(out) }, ow, oh)
}

/// Output size of a 4x box: `(width / 4, height / 4)`.
fn quartered(width: usize, height: usize) -> (usize, usize) {
    (width / 4, height / 4)
}

/// The four source rows of 4x output row `oy`, as offsets into src.
fn rows4(oy: usize, width: usize) -> [usize; 4] {
    [0, 1, 2, 3].map(|row| (4 * oy + row) * width)
}

/// NEON uint8 fused 4x4 box: per row, `vpadal` accumulates byte-pair sums of
/// the 4 rows in u16 (at most 2040), `vpaddq` forms the 4x4 sums (at most
/// 4080) and `vrshrn` rounds them once. 16 outputs per step; the scalar
/// fused pass finishes the row.
///
/// # Safety
/// `src` holds `width * height` samples.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box4x_u8_neon(
    src: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, usize, usize) {
    let (ow, oh) = quartered(width, height);
    let mut out = unwritten_output::<u8>(ow * oh);
    let (src_ptr, dst_ptr) = (src.as_ptr(), out.as_mut_ptr().cast::<u8>());
    for oy in 0..oh {
        let rows = rows4(oy, width);
        let out_row = oy * ow;
        // Column-pair sums down the 4 rows of the 16 columns at x.
        let pair_sums = |x: usize| {
            // SAFETY: callers keep x + 16 <= 4 ow <= width.
            let load = |row: usize| unsafe { vld1q_u8(src_ptr.add(row + x)) };
            let mut sums = vpaddlq_u8(load(rows[0]));
            for &row in &rows[1..] {
                sums = vpadalq_u8(sums, load(row));
            }
            sums
        };
        // 8 rounded 4x4 means of the 32 columns at x.
        let means8 = |x: usize| vrshrn_n_u16::<4>(vpaddq_u16(pair_sums(x), pair_sums(x + 16)));
        let mut ox = 0usize;
        while ox + 16 <= ow {
            let x = 4 * ox;
            // SAFETY: x + 64 <= 4 ow <= width; outputs ox..ox + 16 of the row.
            unsafe {
                vst1q_u8(
                    dst_ptr.add(out_row + ox),
                    vcombine_u8(means8(x), means8(x + 32)),
                );
            };
            ox += 16;
        }
        while ox + 8 <= ow {
            // SAFETY: 4 ox + 32 <= 4 ow <= width; outputs ox..ox + 8 of the row.
            unsafe { vst1_u8(dst_ptr.add(out_row + ox), means8(4 * ox)) };
            ox += 8;
        }
        for ox in ox..ow {
            // SAFETY: ox < width / 4, oy < height / 4: the 4x4 block is in src.
            unsafe { *dst_ptr.add(out_row + ox) = box4_mean(src_ptr, rows, 4 * ox) };
        }
    }
    // SAFETY: every output row was written in full.
    (unsafe { written_output(out) }, ow, oh)
}

/// NEON uint16 fused 4x4 box: `vpadal` accumulates u16 pair sums of the 4
/// rows in u32, `vpaddq` forms the 4x4 sums and `vrshrn` rounds them once.
/// 8 outputs per step; scalar tail.
///
/// # Safety
/// `src` holds `width * height` samples.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box4x_u16_neon(
    src: &[u16],
    width: usize,
    height: usize,
) -> (Vec<u16>, usize, usize) {
    let (ow, oh) = quartered(width, height);
    let mut out = unwritten_output::<u16>(ow * oh);
    let (src_ptr, dst_ptr) = (src.as_ptr(), out.as_mut_ptr().cast::<u16>());
    for oy in 0..oh {
        let rows = rows4(oy, width);
        let out_row = oy * ow;
        // Column-pair sums down the 4 rows of the 8 columns at x.
        let pair_sums = |x: usize| {
            // SAFETY: callers keep x + 8 <= 4 ow <= width.
            let load = |row: usize| unsafe { vld1q_u16(src_ptr.add(row + x)) };
            let mut sums = vpaddlq_u16(load(rows[0]));
            for &row in &rows[1..] {
                sums = vpadalq_u16(sums, load(row));
            }
            sums
        };
        // 4 rounded 4x4 means of the 16 columns at x.
        let means4 = |x: usize| vrshrn_n_u32::<4>(vpaddq_u32(pair_sums(x), pair_sums(x + 8)));
        let mut ox = 0usize;
        while ox + 8 <= ow {
            let x = 4 * ox;
            // SAFETY: x + 32 <= 4 ow <= width; outputs ox..ox + 8 of the row.
            unsafe {
                vst1q_u16(
                    dst_ptr.add(out_row + ox),
                    vcombine_u16(means4(x), means4(x + 16)),
                );
            };
            ox += 8;
        }
        for ox in ox..ow {
            // SAFETY: ox < width / 4, oy < height / 4: the 4x4 block is in src.
            unsafe { *dst_ptr.add(out_row + ox) = box4_mean(src_ptr, rows, 4 * ox) };
        }
    }
    // SAFETY: every output row was written in full.
    (unsafe { written_output(out) }, ow, oh)
}

/// NEON float32 fused 4x4 box: column sums down the 4 rows, then two
/// `vpaddq` levels — `(c0 + c1) + (c2 + c3)`, the scalar order — times 1/16.
/// 8 outputs per step; scalar tail.
///
/// # Safety
/// `src` holds `width * height` samples.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box4x_f32_neon(
    src: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, usize, usize) {
    let (ow, oh) = quartered(width, height);
    let mut out = unwritten_output::<f32>(ow * oh);
    let (src_ptr, dst_ptr) = (src.as_ptr(), out.as_mut_ptr().cast::<f32>());
    for oy in 0..oh {
        let rows = rows4(oy, width);
        let out_row = oy * ow;
        // Sums down the 4 rows of the 4 columns at x.
        let column_sums = |x: usize| {
            // SAFETY: callers keep x + 4 <= 4 ow <= width.
            let load = |row: usize| unsafe { vld1q_f32(src_ptr.add(row + x)) };
            let mut sums = load(rows[0]);
            for &row in &rows[1..] {
                sums = vaddq_f32(sums, load(row));
            }
            sums
        };
        // 4 means of the 16 columns at x.
        let means4 = |x: usize| {
            let pairs = |at: usize| vpaddq_f32(column_sums(at), column_sums(at + 4));
            vmulq_n_f32(vpaddq_f32(pairs(x), pairs(x + 8)), 1.0 / 16.0)
        };
        let mut ox = 0usize;
        while ox + 8 <= ow {
            let x = 4 * ox;
            // SAFETY: x + 32 <= 4 ow <= width; outputs ox..ox + 8 of the row.
            unsafe {
                vst1q_f32(dst_ptr.add(out_row + ox), means4(x));
                vst1q_f32(dst_ptr.add(out_row + ox + 4), means4(x + 16));
            }
            ox += 8;
        }
        for ox in ox..ow {
            // SAFETY: ox < width / 4, oy < height / 4: the 4x4 block is in src.
            unsafe { *dst_ptr.add(out_row + ox) = box4_mean(src_ptr, rows, 4 * ox) };
        }
    }
    // SAFETY: every output row was written in full.
    (unsafe { written_output(out) }, ow, oh)
}

/// The scalar fused 4x4 mean of the block at column `x` of `rows`
/// (`BoxSample::box_sums`' arithmetic), for the kernels' row tails.
///
/// # Safety
/// The 4x4 block at `x` of every row is in the buffer at `src`.
unsafe fn box4_mean<T: BoxSample>(src: *const T, rows: [usize; 4], x: usize) -> T {
    let sums = T::box_sums();
    // SAFETY: the caller's contract.
    let sample = |row: usize, column: usize| (sums.widen)(unsafe { *src.add(row + x + column) });
    let column = |column: usize| {
        ((sample(rows[0], column) + sample(rows[1], column)) + sample(rows[2], column))
            + sample(rows[3], column)
    };
    (sums.mean16)((column(0) + column(1)) + (column(2) + column(3)))
}

/// The pyramid's mean of one sample of an interleaved image, exact: the 4x4
/// mean of the block at (`row`, `px`) rounded once, or for an 8x block the
/// rounded mean of its four rounded 4x4 means (the kernels' row tails).
fn hwc_block_mean<T: Copy + Into<u32>>(
    src: &[T],
    (stride, channels): (usize, usize),
    block: usize,
    (row, px): (usize, usize),
    channel: usize,
) -> u32 {
    let mean4 = |row: usize, px: usize| {
        let sum: u32 = (0..4)
            .flat_map(|dy| (0..4).map(move |dx| (dy, dx)))
            .map(|(dy, dx)| src[(row + dy) * stride + (px + dx) * channels + channel].into())
            .sum();
        (sum + 8) >> 4
    };
    if block == 4 {
        mean4(row, px)
    } else {
        (mean4(row, px) + mean4(row, px + 4) + mean4(row + 4, px) + mean4(row + 4, px + 4) + 2) >> 2
    }
}

/// Rounded 4x4 means of 4x4 sums (u16 lanes).
#[target_feature(enable = "neon")]
#[inline]
fn rounded_mean16_u16(sums: uint16x8_t) -> uint8x8_t {
    vrshrn_n_u16::<4>(sums)
}

/// Rounded 4x4 means of 4x4 sums (u32 lanes).
#[target_feature(enable = "neon")]
#[inline]
fn rounded_mean16_u32(sums: uint32x4_t) -> uint16x4_t {
    vrshrn_n_u32::<4>(sums)
}

// The fused interleaved 4x/8x box for one sample type. `$load3`/`$load4`
// split `$lanes` pixels into channel planes; per plane, `$pairs`/`$pairs_acc`
// sum column pairs down 4 rows, `$quads` pairs those into 4x4 sums and
// `$mean16` rounds them once: `$lanes / 2` outputs from 2 `$lanes` pixels.
// 8x takes two such 4x bands and their rounded 2x2 mean (`$mean4`), the
// planar pyramid's passes.
macro_rules! gen_box_levels_hwc_neon {
    (
        $name:ident, $t:ty, $lanes:expr, $plane:ty, $means:ty,
        $load3:ident, $load4:ident, $pairs:ident, $pairs_acc:ident, $quads:ident,
        $mean16:ident, $combine:ident, $mean4:ident,
        $store3:ident, $store4:ident, $splat:ident, $planes3:ident, $planes4:ident
    ) => {
        /// The interleaved box reduced by 4 (`levels` 2) or 8 (`levels` 3),
        /// with each channel equal to the planar pyramid's; `pad` widens RGB
        /// to RGBX output.
        ///
        /// # Safety
        /// `src` holds `width * height * CHANNELS` samples, `CHANNELS` 3 or 4
        /// (3 with `pad`), and `levels` is 2 or 3.
        #[target_feature(enable = "neon")]
        pub(crate) unsafe fn $name<const CHANNELS: usize>(
            src: &[$t],
            (width, height): (usize, usize),
            levels: u32,
            pad: Option<$t>,
        ) -> (Vec<$t>, usize, usize) {
            const LANES: usize = $lanes;
            debug_assert!((2..=3).contains(&levels) && (pad.is_none() || CHANNELS == RGB));
            let block = 1usize << levels;
            let (ow, oh) = (width / block, height / block);
            let out_channels = if pad.is_some() { RGBX } else { CHANNELS };
            let (stride, out_stride) = (width * CHANNELS, ow * out_channels);
            let mut out = unwritten_output::<$t>(oh * out_stride);
            let (src_ptr, dst_ptr) = (src.as_ptr(), out.as_mut_ptr().cast::<$t>());
            let pad_plane = $splat(pad.unwrap_or_default());
            // The channel planes of the LANES pixels at sample `at` (an RGB
            // image repeats its last plane).
            let planes = |at: usize| -> [$plane; 4] {
                // SAFETY: callers keep the LANES pixels at `at` in src.
                unsafe {
                    if CHANNELS == RGB {
                        let split = $load3(src_ptr.add(at));
                        [split.0, split.1, split.2, split.2]
                    } else {
                        let split = $load4(src_ptr.add(at));
                        [split.0, split.1, split.2, split.3]
                    }
                }
            };
            // Rounded 4x4 means of the 2 LANES pixels from `px` in the 4 rows
            // from `top_row`: LANES / 2 per plane.
            let means4 = |top_row: usize, px: usize| -> [$means; 4] {
                let pair_sums = |px: usize| {
                    let at = |row: usize| (top_row + row) * stride + px * CHANNELS;
                    let mut sums = planes(at(0)).map(|plane| $pairs(plane));
                    for row in 1..4 {
                        let next = planes(at(row));
                        for (sum, plane) in sums.iter_mut().zip(next) {
                            *sum = $pairs_acc(*sum, plane);
                        }
                    }
                    sums
                };
                let (left, right) = (pair_sums(px), pair_sums(px + LANES));
                [0, 1, 2, 3].map(|plane| $mean16($quads(left[plane], right[plane])))
            };
            for oy in 0..oh {
                let out_row = oy * out_stride;
                let mut ox = 0usize;
                // block (ox + LANES / 2) <= block ow <= width: the step reads
                // pixels block ox .. block (ox + LANES / 2) of the block's rows.
                while ox + LANES / 2 <= ow {
                    let px = block * ox;
                    let means = if levels == 2 {
                        means4(4 * oy, px)
                    } else {
                        let band = |top_row: usize| {
                            let (left, right) =
                                (means4(top_row, px), means4(top_row, px + 2 * LANES));
                            [0, 1, 2, 3].map(|plane| $combine(left[plane], right[plane]))
                        };
                        let (upper, lower) = (band(8 * oy), band(8 * oy + 4));
                        [0, 1, 2, 3].map(|plane| $mean4(upper[plane], lower[plane]))
                    };
                    // SAFETY: ox + LANES / 2 <= ow: output pixels of this row.
                    unsafe {
                        let dst = dst_ptr.add(out_row + ox * out_channels);
                        match (CHANNELS, pad) {
                            (RGB, None) => $store3(dst, $planes3(means[0], means[1], means[2])),
                            (RGB, Some(_)) => {
                                $store4(dst, $planes4(means[0], means[1], means[2], pad_plane));
                            }
                            _ => $store4(dst, $planes4(means[0], means[1], means[2], means[3])),
                        }
                    }
                    ox += LANES / 2;
                }
                for ox in ox..ow {
                    let dst = out_row + ox * out_channels;
                    for channel in 0..CHANNELS {
                        let mean = hwc_block_mean(
                            src,
                            (stride, CHANNELS),
                            block,
                            (block * oy, block * ox),
                            channel,
                        );
                        // SAFETY: an output sample of this row; the mean of
                        // `$t` samples fits `$t`.
                        unsafe { *dst_ptr.add(dst + channel) = mean as $t };
                    }
                    if let Some(pad) = pad {
                        // SAFETY: the pad sample of this RGBX output pixel.
                        unsafe { *dst_ptr.add(dst + RGB) = pad };
                    }
                }
            }
            // SAFETY: every output row was written in full.
            (unsafe { written_output(out) }, ow, oh)
        }
    };
}

gen_box_levels_hwc_neon!(
    box_levels_u8_hwc_neon,
    u8,
    16,
    uint8x16_t,
    uint8x8_t,
    vld3q_u8,
    vld4q_u8,
    vpaddlq_u8,
    vpadalq_u8,
    vpaddq_u16,
    rounded_mean16_u16,
    vcombine_u8,
    mean2x2_u8,
    vst3_u8,
    vst4_u8,
    vdup_n_u8,
    uint8x8x3_t,
    uint8x8x4_t
);
gen_box_levels_hwc_neon!(
    box_levels_u16_hwc_neon,
    u16,
    8,
    uint16x8_t,
    uint16x4_t,
    vld3q_u16,
    vld4q_u16,
    vpaddlq_u16,
    vpadalq_u16,
    vpaddq_u32,
    rounded_mean16_u32,
    vcombine_u16,
    mean2x2_u16,
    vst3_u16,
    vst4_u16,
    vdup_n_u16,
    uint16x4x3_t,
    uint16x4x4_t
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box_reduce::tests::{
        BoxTestSample, assert_same_image, box_sizes, box4x_fused_reference,
        interleaved_pyramid_reference, test_images,
    };
    use crate::box_reduce::{box2x_interleaved, box2x_scalar};
    use crate::testing::{ImageKernel, Rng, UnsafeImageKernel};

    /// An interleaved NEON kernel taking its channel count.
    type InterleavedKernel<T> = unsafe fn(&[T], usize, usize, usize) -> (Vec<T>, usize, usize);

    fn assert_planar_matches_reference<T: BoxTestSample>(
        kernel: UnsafeImageKernel<T>,
        reference: ImageKernel<T>,
    ) {
        let mut rng = Rng::new(40);
        for (width, height) in box_sizes(1) {
            for src in test_images::<T>(&mut rng, width * height) {
                // SAFETY: NEON is baseline on aarch64; src holds width *
                // height samples.
                let simd = unsafe { kernel(&src, width, height) };
                assert_same_image(
                    &simd,
                    &reference(&src, width, height),
                    &format!("{width} x {height}"),
                );
            }
        }
    }

    fn assert_interleaved_halves_as_scalar<T: BoxTestSample>(kernel: InterleavedKernel<T>) {
        let mut rng = Rng::new(41);
        for channels in [RGB, RGBX] {
            for (width, height) in box_sizes(1) {
                for src in test_images::<T>(&mut rng, width * height * channels) {
                    // SAFETY: NEON is baseline on aarch64; src holds width *
                    // height pixels of `channels` (3 or 4) samples.
                    let simd = unsafe { kernel(&src, width, height, channels) };
                    assert_same_image(
                        &simd,
                        &box2x_interleaved(&src, width, height, channels),
                        &format!("{width} x {height} x {channels}"),
                    );
                }
            }
        }
    }

    #[test]
    fn planar_neon_kernels_equal_scalar() {
        assert_planar_matches_reference::<u8>(box2x_u8_neon, box2x_scalar);
        assert_planar_matches_reference::<u16>(box2x_u16_neon, box2x_scalar);
        assert_planar_matches_reference::<f32>(box2x_f32_neon, box2x_scalar);
    }

    #[test]
    fn interleaved_neon_kernels_equal_scalar() {
        assert_interleaved_halves_as_scalar::<u8>(box2x_u8_hwc_neon);
        assert_interleaved_halves_as_scalar::<u16>(box2x_u16_hwc_neon);
        assert_interleaved_halves_as_scalar::<f32>(box2x_f32_hwc_neon);
    }

    /// A fused interleaved NEON kernel: (src, (width, height), levels, pad).
    type LevelsKernel<T> =
        unsafe fn(&[T], (usize, usize), u32, Option<T>) -> (Vec<T>, usize, usize);

    #[test]
    fn fused_box4x_neon_kernels_equal_the_fused_reference() {
        assert_planar_matches_reference::<u8>(box4x_u8_neon, box4x_fused_reference);
        assert_planar_matches_reference::<u16>(box4x_u16_neon, box4x_fused_reference);
        assert_planar_matches_reference::<f32>(box4x_f32_neon, box4x_fused_reference);
    }

    fn assert_levels_match_the_pyramid<T: BoxTestSample>(
        rgb: LevelsKernel<T>,
        rgbx: LevelsKernel<T>,
    ) {
        let mut rng = Rng::new(42);
        for (channels, kernel) in [(RGB, rgb), (RGBX, rgbx)] {
            for levels in [2, 3] {
                for (width, height) in box_sizes(1) {
                    for src in test_images::<T>(&mut rng, width * height * channels) {
                        // SAFETY: NEON is baseline on aarch64; src holds
                        // width * height pixels of `channels` samples.
                        let simd = unsafe { kernel(&src, (width, height), levels, None) };
                        assert_same_image(
                            &simd,
                            &interleaved_pyramid_reference(&src, (width, height, channels), levels),
                            &format!("{levels} levels, {width} x {height} x {channels}"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fused_interleaved_neon_kernels_equal_the_pyramid() {
        assert_levels_match_the_pyramid::<u8>(
            box_levels_u8_hwc_neon::<RGB>,
            box_levels_u8_hwc_neon::<RGBX>,
        );
        assert_levels_match_the_pyramid::<u16>(
            box_levels_u16_hwc_neon::<RGB>,
            box_levels_u16_hwc_neon::<RGBX>,
        );
    }

    #[test]
    fn rgb_to_rgbx_neon_kernels_equal_the_rgb_results_padded() {
        let mut rng = Rng::new(43);
        let padded = |(rgb, ow, oh): (Vec<u8>, usize, usize)| {
            let rgbx = rgb
                .as_chunks::<RGB>()
                .0
                .iter()
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 7]);
            (rgbx.collect::<Vec<u8>>(), ow, oh)
        };
        for (width, height) in box_sizes(1) {
            for src in test_images::<u8>(&mut rng, width * height * RGB) {
                let context = format!("{width} x {height}");
                // SAFETY: NEON is baseline on aarch64; src holds width *
                // height RGB pixels.
                let halved = unsafe { box2x_u8_hwc3to4_neon(&src, width, height, 7) };
                assert_same_image(
                    &halved,
                    &padded(box2x_interleaved(&src, width, height, RGB)),
                    &context,
                );
                for levels in [2, 3] {
                    // SAFETY: as above.
                    let fused = unsafe {
                        box_levels_u8_hwc_neon::<RGB>(&src, (width, height), levels, Some(7))
                    };
                    let expected =
                        interleaved_pyramid_reference(&src, (width, height, RGB), levels);
                    assert_same_image(
                        &fused,
                        &padded(expected),
                        &format!("{levels} levels, {context}"),
                    );
                }
            }
        }
    }
}

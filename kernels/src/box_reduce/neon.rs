//! NEON box reduction kernels (aarch64): the 2x2 box, planar and interleaved
//! (RGB pixels, split into channel planes by `vld3`), and the fused 4x4
//! box.
//!
//! Integer kernels sum with `vpaddl`/`vpadal` (widening pairwise adds) and
//! round with `vrshrn` (rounding shift and narrow): `(sum + 2) >> 2` and
//! `(sum + 8) >> 4`, the scalar reference's exact single rounding.
//!
//! # Safety
//! NEON is baseline on aarch64, gated only by the `set_simd` kill switch.
//! Every `unsafe fn` here requires `src.len() >= width * height * channels`.
//! Loads and stores are justified against that at their loop guards.

use std::arch::aarch64::{
    uint8x8_t, uint8x8x3_t, uint8x16_t, uint8x16x3_t, uint16x4_t, uint16x4x3_t, uint16x8_t,
    uint32x4_t, vaddq_u32, vcombine_u8, vcombine_u16, vld1q_u8, vld1q_u16, vld3q_u8, vld3q_u16,
    vpadalq_u8, vpadalq_u16, vpaddlq_u8, vpaddlq_u16, vpaddq_u16, vpaddq_u32, vrshrn_n_u16,
    vrshrn_n_u32, vst1_u8, vst1q_u8, vst1q_u16, vst3_u8, vst3_u16, vst3q_u8,
};

use super::{BoxSample, box2_mean};
use crate::cpu::{unwritten_output, written_output};
use crate::sample::{RGB, Sample};

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
    let (out_width, out_height) = halved(width, height);
    let src_ptr = src.as_ptr();
    for out_y in 0..out_height {
        let (top, bottom, out_row) = (
            2 * out_y * width,
            (2 * out_y + 1) * width,
            out_y * dst_stride,
        );
        let mut out_x = 0usize;
        while out_x + 32 <= out_width {
            let x = 2 * out_x;
            // SAFETY: x + 64 <= 2 out_width <= width: 64 samples of both
            // rows; the stores write outputs out_x..out_x + 32 of the row.
            unsafe {
                let load = |offset: usize| {
                    [
                        vld1q_u8(src_ptr.add(offset)),
                        vld1q_u8(src_ptr.add(offset + 16)),
                    ]
                };
                let low = mean2x2_u8x32(load(top + x), load(bottom + x));
                let high = mean2x2_u8x32(load(top + x + 32), load(bottom + x + 32));
                vst1q_u8(dst_ptr.add(out_row + out_x), low);
                vst1q_u8(dst_ptr.add(out_row + out_x + 16), high);
            }
            out_x += 32;
        }
        for out_x in out_x..out_width {
            // SAFETY: out_x < width / 2, out_y < height / 2: the 2x2 block is
            // in src.
            unsafe {
                *dst_ptr.add(out_row + out_x) = box2_mean(src_ptr.add(top + 2 * out_x), 1, width);
            };
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
    let (out_width, out_height) = halved(width, height);
    let src_ptr = src.as_ptr();
    for out_y in 0..out_height {
        let (top, bottom, out_row) = (
            2 * out_y * width,
            (2 * out_y + 1) * width,
            out_y * dst_stride,
        );
        let mut out_x = 0usize;
        while out_x + 8 <= out_width {
            let x = 2 * out_x;
            // SAFETY: x + 16 <= 2 out_width <= width: 16 samples of both
            // rows; the store writes outputs out_x..out_x + 8 of the row.
            unsafe {
                let load = |offset: usize| vld1q_u16(src_ptr.add(offset));
                let means = vcombine_u16(
                    mean2x2_u16(load(top + x), load(bottom + x)),
                    mean2x2_u16(load(top + x + 8), load(bottom + x + 8)),
                );
                vst1q_u16(dst_ptr.add(out_row + out_x), means);
            }
            out_x += 8;
        }
        for out_x in out_x..out_width {
            // SAFETY: out_x < width / 2, out_y < height / 2: the 2x2 block is
            // in src.
            unsafe {
                *dst_ptr.add(out_row + out_x) = box2_mean(src_ptr.add(top + 2 * out_x), 1, width);
            };
        }
    }
}

/// A 2x box kernel writing output row `out_y` at `dst_ptr + out_y * dst_stride`.
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
    let (out_width, out_height) = halved(width, height);
    let out_stride = out_width * channels;
    let mut out = unwritten_output::<T>(out_height * out_stride);
    let dst_ptr = out.as_mut_ptr().cast::<T>();
    // SAFETY: the caller's bound on src; out holds the out_height rows of
    // out_width pixels, out_stride samples apart.
    unsafe { kernel(src, width, height, dst_ptr, out_stride) };
    // SAFETY: the kernel wrote every output row in full.
    (unsafe { written_output(out) }, out_width, out_height)
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

/// The uint16 RGB 2x2 box, 4 output pixels per step: `vld3q` splits 8 source
/// pixels per row into channel planes, `mean2x2_u16` reduces each plane's two
/// rows and `vst3` interleaves the outputs.
///
/// # Safety
/// `src` holds `width * height` pixels and `dst_ptr` the `height / 2` output
/// rows of `width / 2` pixels, `dst_stride` samples apart.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box2x_u16_hwc_into_neon(
    src: &[u16],
    width: usize,
    height: usize,
    dst_ptr: *mut u16,
    dst_stride: usize,
) {
    const STEP_PIXELS: usize = 4;
    let (out_width, out_height) = halved(width, height);
    let stride = width * RGB;
    let src_ptr = src.as_ptr();
    for out_y in 0..out_height {
        let (top, bottom) = (2 * out_y * stride, (2 * out_y + 1) * stride);
        let out_row = out_y * dst_stride;
        let mut out_x = 0usize;
        while out_x + STEP_PIXELS <= out_width {
            let offset = 2 * out_x * RGB;
            // SAFETY: out_x + STEP_PIXELS <= out_width <= width / 2: the
            // loads and the store stay in their rows.
            unsafe {
                let upper = vld3q_u16(src_ptr.add(top + offset));
                let lower = vld3q_u16(src_ptr.add(bottom + offset));
                let means = uint16x4x3_t(
                    mean2x2_u16(upper.0, lower.0),
                    mean2x2_u16(upper.1, lower.1),
                    mean2x2_u16(upper.2, lower.2),
                );
                vst3_u16(dst_ptr.add(out_row + out_x * RGB), means);
            }
            out_x += STEP_PIXELS;
        }
        // SAFETY: the tail pixels out_x < width / 2 of row out_y < height / 2
        // have their 2x2 blocks in src and their outputs in the row.
        unsafe {
            box2x_hwc_tail(
                src_ptr,
                (top, stride),
                dst_ptr.add(out_row),
                out_x..out_width,
            );
        }
    }
}

/// The uint8 RGB 2x2 box, 16 output pixels per step: two `vld3q` loads of 16
/// pixels per row, stored by one full-width `vst3q`.
///
/// # Safety
/// `src` holds `width * height` pixels and `dst_ptr` the `height / 2` output
/// rows of `width / 2` pixels, `dst_stride` samples apart.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box2x_u8_hwc_into_neon(
    src: &[u8],
    width: usize,
    height: usize,
    dst_ptr: *mut u8,
    dst_stride: usize,
) {
    const STEP_PIXELS: usize = 16;
    let (out_width, out_height) = halved(width, height);
    let stride = width * RGB;
    let src_ptr = src.as_ptr();
    for out_y in 0..out_height {
        let (top, bottom) = (2 * out_y * stride, (2 * out_y + 1) * stride);
        let out_row = out_y * dst_stride;
        let step = |out_x: usize| {
            let offset = 2 * out_x * RGB;
            // SAFETY: out_x + STEP_PIXELS <= out_width <= width / 2 (the
            // callers' guards): the loads and the store stay in their rows.
            unsafe {
                let load = |at: usize| {
                    [
                        vld3q_u8(src_ptr.add(at)),
                        vld3q_u8(src_ptr.add(at + STEP_PIXELS * RGB)),
                    ]
                };
                let (upper, lower) = (load(top + offset), load(bottom + offset));
                let means = uint8x16x3_t(
                    mean2x2_u8x32([upper[0].0, upper[1].0], [lower[0].0, lower[1].0]),
                    mean2x2_u8x32([upper[0].1, upper[1].1], [lower[0].1, lower[1].1]),
                    mean2x2_u8x32([upper[0].2, upper[1].2], [lower[0].2, lower[1].2]),
                );
                vst3q_u8(dst_ptr.add(out_row + out_x * RGB), means);
            }
        };
        // Two steps per pass while they fit, so more loads are in flight.
        let mut out_x = 0usize;
        while out_x + 2 * STEP_PIXELS <= out_width {
            step(out_x);
            step(out_x + STEP_PIXELS);
            out_x += 2 * STEP_PIXELS;
        }
        if out_x + STEP_PIXELS <= out_width {
            step(out_x);
            out_x += STEP_PIXELS;
        }
        // SAFETY: the tail pixels out_x < width / 2 of row out_y < height / 2
        // have their 2x2 blocks in src and their outputs in the row.
        unsafe {
            box2x_hwc_tail(
                src_ptr,
                (top, stride),
                dst_ptr.add(out_row),
                out_x..out_width,
            );
        }
    }
}

/// The scalar 2x box of output pixels `out_xs` of an RGB row, its source rows
/// starting at `top` and `top + stride`, written at `out_row`.
///
/// # Safety
/// The 2x2 blocks of those pixels are in the buffer at `src_ptr`, and the
/// output pixels writable at `out_row`.
#[inline(always)]
unsafe fn box2x_hwc_tail<T: Sample>(
    src_ptr: *const T,
    (top, stride): (usize, usize),
    out_row: *mut T,
    out_xs: std::ops::Range<usize>,
) {
    for out_x in out_xs {
        let top_left = top + 2 * out_x * RGB;
        for channel in 0..RGB {
            // SAFETY: the caller's contract.
            unsafe {
                *out_row.add(out_x * RGB + channel) =
                    box2_mean(src_ptr.add(top_left + channel), RGB, stride);
            }
        }
    }
}

/// NEON uint8 RGB 2x2 box into a new image.
///
/// # Safety
/// `src` holds `width * height * 3` samples.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box2x_u8_hwc_neon(
    src: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, usize, usize) {
    // SAFETY: the caller's bound.
    unsafe { box2x_allocated(box2x_u8_hwc_into_neon, src, (width, height, RGB)) }
}

/// NEON uint16 RGB 2x2 box into a new image.
///
/// # Safety
/// `src` holds `width * height * 3` samples.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box2x_u16_hwc_neon(
    src: &[u16],
    width: usize,
    height: usize,
) -> (Vec<u16>, usize, usize) {
    // SAFETY: the caller's bound.
    unsafe { box2x_allocated(box2x_u16_hwc_into_neon, src, (width, height, RGB)) }
}

fn quartered(width: usize, height: usize) -> (usize, usize) {
    (width / 4, height / 4)
}

/// Offsets into the source of the 4 rows of 4x output row `out_y`.
fn source_rows4(out_y: usize, width: usize) -> [usize; 4] {
    [0, 1, 2, 3].map(|row| (4 * out_y + row) * width)
}

/// NEON uint8 fused 4x4 box, 16 outputs per step: `vpadal` accumulates
/// byte-pair sums of the 4 rows in u16 (at most 2040), `vpaddq` forms the
/// 4x4 sums (at most 4080) and `vrshrn` rounds them once.
///
/// # Safety
/// `src` holds `width * height` samples.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box4x_u8_neon(
    src: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, usize, usize) {
    let (out_width, out_height) = quartered(width, height);
    let mut out = unwritten_output::<u8>(out_width * out_height);
    let (src_ptr, dst_ptr) = (src.as_ptr(), out.as_mut_ptr().cast::<u8>());
    for out_y in 0..out_height {
        let rows = source_rows4(out_y, width);
        let out_row = out_y * out_width;
        let column_pair_sums = |x: usize| {
            // SAFETY: callers keep x + 16 <= 4 out_width <= width.
            let load = |row: usize| unsafe { vld1q_u8(src_ptr.add(row + x)) };
            let mut sums = vpaddlq_u8(load(rows[0]));
            for &row in &rows[1..] {
                sums = vpadalq_u8(sums, load(row));
            }
            sums
        };
        let means_of_32_columns =
            |x: usize| vrshrn_n_u16::<4>(vpaddq_u16(column_pair_sums(x), column_pair_sums(x + 16)));
        let mut out_x = 0usize;
        while out_x + 16 <= out_width {
            let x = 4 * out_x;
            // SAFETY: x + 64 <= 4 out_width <= width; outputs out_x..out_x +
            // 16 of the row.
            unsafe {
                vst1q_u8(
                    dst_ptr.add(out_row + out_x),
                    vcombine_u8(means_of_32_columns(x), means_of_32_columns(x + 32)),
                );
            };
            out_x += 16;
        }
        while out_x + 8 <= out_width {
            // SAFETY: 4 out_x + 32 <= 4 out_width <= width; outputs
            // out_x..out_x + 8 of the row.
            unsafe { vst1_u8(dst_ptr.add(out_row + out_x), means_of_32_columns(4 * out_x)) };
            out_x += 8;
        }
        for out_x in out_x..out_width {
            // SAFETY: out_x < width / 4, out_y < height / 4: the 4x4 block is
            // in src.
            unsafe { *dst_ptr.add(out_row + out_x) = box4_mean(src_ptr, rows, 4 * out_x) };
        }
    }
    // SAFETY: every output row was written in full.
    (unsafe { written_output(out) }, out_width, out_height)
}

/// NEON uint16 fused 4x4 box, 8 outputs per step: `vpadal` accumulates u16
/// pair sums of the 4 rows in u32, `vpaddq` forms the 4x4 sums and `vrshrn`
/// rounds them once.
///
/// # Safety
/// `src` holds `width * height` samples.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn box4x_u16_neon(
    src: &[u16],
    width: usize,
    height: usize,
) -> (Vec<u16>, usize, usize) {
    let (out_width, out_height) = quartered(width, height);
    let mut out = unwritten_output::<u16>(out_width * out_height);
    let (src_ptr, dst_ptr) = (src.as_ptr(), out.as_mut_ptr().cast::<u16>());
    for out_y in 0..out_height {
        let rows = source_rows4(out_y, width);
        let out_row = out_y * out_width;
        let column_pair_sums = |x: usize| {
            // SAFETY: callers keep x + 8 <= 4 out_width <= width.
            let load = |row: usize| unsafe { vld1q_u16(src_ptr.add(row + x)) };
            let mut sums = vpaddlq_u16(load(rows[0]));
            for &row in &rows[1..] {
                sums = vpadalq_u16(sums, load(row));
            }
            sums
        };
        let means_of_16_columns =
            |x: usize| vrshrn_n_u32::<4>(vpaddq_u32(column_pair_sums(x), column_pair_sums(x + 8)));
        let mut out_x = 0usize;
        while out_x + 8 <= out_width {
            let x = 4 * out_x;
            // SAFETY: x + 32 <= 4 out_width <= width; outputs out_x..out_x + 8
            // of the row.
            unsafe {
                vst1q_u16(
                    dst_ptr.add(out_row + out_x),
                    vcombine_u16(means_of_16_columns(x), means_of_16_columns(x + 16)),
                );
            };
            out_x += 8;
        }
        for out_x in out_x..out_width {
            // SAFETY: out_x < width / 4, out_y < height / 4: the 4x4 block is
            // in src.
            unsafe { *dst_ptr.add(out_row + out_x) = box4_mean(src_ptr, rows, 4 * out_x) };
        }
    }
    // SAFETY: every output row was written in full.
    (unsafe { written_output(out) }, out_width, out_height)
}

/// The scalar fused 4x4 mean of the block at column `x` of `rows`
/// (`BoxSample::box_sums`' arithmetic), for the kernels' row tails.
///
/// # Safety
/// The 4x4 block at `x` of every row is in the buffer at `src`.
unsafe fn box4_mean<T: BoxSample>(src: *const T, rows: [usize; 4], x: usize) -> T {
    let box_sums = T::box_sums();
    // SAFETY: the caller's contract.
    let sample =
        |row: usize, column: usize| (box_sums.widen)(unsafe { *src.add(row + x + column) });
    let column = |column: usize| {
        ((sample(rows[0], column) + sample(rows[1], column)) + sample(rows[2], column))
            + sample(rows[3], column)
    };
    (box_sums.mean_of_16)((column(0) + column(1)) + (column(2) + column(3)))
}

/// The pyramid's exact mean of one sample of an interleaved image, for the
/// kernels' row tails: the 4x4 mean of the block at (`row`, `pixel`) rounded
/// once, or for an 8x block the rounded mean of its four rounded 4x4 means.
fn hwc_block_mean<T: Copy + Into<u32>>(
    src: &[T],
    (stride, channels): (usize, usize),
    block: usize,
    (row, pixel): (usize, usize),
    channel: usize,
) -> u32 {
    let mean_4x4 = |row: usize, pixel: usize| {
        let sum: u32 = (0..4)
            .flat_map(|block_row| (0..4).map(move |block_column| (block_row, block_column)))
            .map(|(block_row, block_column)| {
                src[(row + block_row) * stride + (pixel + block_column) * channels + channel].into()
            })
            .sum();
        (sum + 8) >> 4
    };
    if block == 4 {
        mean_4x4(row, pixel)
    } else {
        (mean_4x4(row, pixel)
            + mean_4x4(row, pixel + 4)
            + mean_4x4(row + 4, pixel)
            + mean_4x4(row + 4, pixel + 4)
            + 2)
            >> 2
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

// The fused RGB 4x/8x box of one sample type: per channel plane,
// `$pairs`/`$pairs_acc` sum column pairs down 4 rows, `$quads` pairs those
// into 4x4 sums and `$mean16` rounds them once, `$lanes / 2` outputs from
// 2 `$lanes` pixels. 8x takes the rounded 2x2 mean (`$mean4`) of two such
// bands, as the planar pyramid.
macro_rules! gen_box_levels_hwc_neon {
    (
        $name:ident, $t:ty, $lanes:expr, $plane:ty, $means:ty,
        $load3:ident, $pairs:ident, $pairs_acc:ident, $quads:ident,
        $mean16:ident, $combine:ident, $mean4:ident, $store3:ident, $planes3:ident
    ) => {
        /// The RGB box reduced by 4 (`levels` 2) or 8 (`levels` 3), with
        /// each channel equal to the planar pyramid's.
        ///
        /// # Safety
        /// `src` holds `width * height * 3` samples, and `levels` is 2 or 3.
        #[target_feature(enable = "neon")]
        pub(crate) unsafe fn $name(
            src: &[$t],
            (width, height): (usize, usize),
            levels: u32,
        ) -> (Vec<$t>, usize, usize) {
            const LANES: usize = $lanes;
            debug_assert!((2..=3).contains(&levels));
            let block = 1usize << levels;
            let (out_width, out_height) = (width / block, height / block);
            let (stride, out_stride) = (width * RGB, out_width * RGB);
            let mut out = unwritten_output::<$t>(out_height * out_stride);
            let (src_ptr, dst_ptr) = (src.as_ptr(), out.as_mut_ptr().cast::<$t>());
            let channel_planes = |at: usize| -> [$plane; RGB] {
                // SAFETY: callers keep the LANES pixels at `at` in src.
                let split = unsafe { $load3(src_ptr.add(at)) };
                [split.0, split.1, split.2]
            };
            // Rounded 4x4 means of the 2 LANES pixels from `pixel` in the 4
            // rows from `top_row`: LANES / 2 per plane.
            let means4 = |top_row: usize, pixel: usize| -> [$means; RGB] {
                let column_pair_sums = |pixel: usize| {
                    let at = |row: usize| (top_row + row) * stride + pixel * RGB;
                    let mut sums = channel_planes(at(0)).map(|plane| $pairs(plane));
                    for row in 1..4 {
                        let next = channel_planes(at(row));
                        for (sum, plane) in sums.iter_mut().zip(next) {
                            *sum = $pairs_acc(*sum, plane);
                        }
                    }
                    sums
                };
                let (left, right) = (column_pair_sums(pixel), column_pair_sums(pixel + LANES));
                [0, 1, 2].map(|plane| $mean16($quads(left[plane], right[plane])))
            };
            for out_y in 0..out_height {
                let out_row = out_y * out_stride;
                let mut out_x = 0usize;
                // block (out_x + LANES / 2) <= block out_width <= width: the
                // step reads pixels block out_x .. block (out_x + LANES / 2)
                // of the block's rows.
                while out_x + LANES / 2 <= out_width {
                    let pixel = block * out_x;
                    let means = if levels == 2 {
                        means4(4 * out_y, pixel)
                    } else {
                        let band = |top_row: usize| {
                            let (left, right) =
                                (means4(top_row, pixel), means4(top_row, pixel + 2 * LANES));
                            [0, 1, 2].map(|plane| $combine(left[plane], right[plane]))
                        };
                        let (upper, lower) = (band(8 * out_y), band(8 * out_y + 4));
                        [0, 1, 2].map(|plane| $mean4(upper[plane], lower[plane]))
                    };
                    // SAFETY: out_x + LANES / 2 <= out_width: output pixels of
                    // this row.
                    unsafe {
                        let dst = dst_ptr.add(out_row + out_x * RGB);
                        $store3(dst, $planes3(means[0], means[1], means[2]));
                    }
                    out_x += LANES / 2;
                }
                for out_x in out_x..out_width {
                    let dst = out_row + out_x * RGB;
                    for channel in 0..RGB {
                        let mean = hwc_block_mean(
                            src,
                            (stride, RGB),
                            block,
                            (block * out_y, block * out_x),
                            channel,
                        );
                        // SAFETY: an output sample of this row; the mean of
                        // `$t` samples fits `$t`.
                        unsafe { *dst_ptr.add(dst + channel) = mean as $t };
                    }
                }
            }
            // SAFETY: every output row was written in full.
            (unsafe { written_output(out) }, out_width, out_height)
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
    vpaddlq_u8,
    vpadalq_u8,
    vpaddq_u16,
    rounded_mean16_u16,
    vcombine_u8,
    mean2x2_u8,
    vst3_u8,
    uint8x8x3_t
);
gen_box_levels_hwc_neon!(
    box_levels_u16_hwc_neon,
    u16,
    8,
    uint16x8_t,
    uint16x4_t,
    vld3q_u16,
    vpaddlq_u16,
    vpadalq_u16,
    vpaddq_u32,
    rounded_mean16_u32,
    vcombine_u16,
    mean2x2_u16,
    vst3_u16,
    uint16x4x3_t
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box_reduce::tests::{
        BoxTestSample, assert_same_image, box_sizes, box4x_fused_reference, hwc_pyramid_reference,
    };
    use crate::box_reduce::{box2x_hwc_scalar, box2x_scalar};
    use crate::testing::{ImageKernel, Rng, UnsafeImageKernel, test_images};

    fn assert_planar_matches_reference<T: BoxTestSample>(
        kernel: UnsafeImageKernel<T>,
        reference: ImageKernel<T>,
    ) {
        let mut rng = Rng::new(40);
        for (width, height) in box_sizes(1) {
            for (_, src) in test_images(&mut rng, width * height, T::MAX) {
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

    fn assert_hwc_halves_as_scalar<T: BoxTestSample>(kernel: UnsafeImageKernel<T>) {
        let mut rng = Rng::new(41);
        for (width, height) in box_sizes(1) {
            for (_, src) in test_images(&mut rng, width * height * RGB, T::MAX) {
                // SAFETY: NEON is baseline on aarch64; src holds width *
                // height RGB pixels.
                let simd = unsafe { kernel(&src, width, height) };
                assert_same_image(
                    &simd,
                    &box2x_hwc_scalar(&src, width, height),
                    &format!("{width} x {height}"),
                );
            }
        }
    }

    #[test]
    fn planar_neon_kernels_equal_scalar() {
        assert_planar_matches_reference::<u8>(box2x_u8_neon, box2x_scalar);
        assert_planar_matches_reference::<u16>(box2x_u16_neon, box2x_scalar);
    }

    #[test]
    fn hwc_neon_kernels_equal_scalar() {
        assert_hwc_halves_as_scalar::<u8>(box2x_u8_hwc_neon);
        assert_hwc_halves_as_scalar::<u16>(box2x_u16_hwc_neon);
    }

    /// A fused interleaved NEON kernel: (src, (width, height), levels).
    type LevelsKernel<T> = unsafe fn(&[T], (usize, usize), u32) -> (Vec<T>, usize, usize);

    #[test]
    fn fused_box4x_neon_kernels_equal_the_fused_reference() {
        assert_planar_matches_reference::<u8>(box4x_u8_neon, box4x_fused_reference);
        assert_planar_matches_reference::<u16>(box4x_u16_neon, box4x_fused_reference);
    }

    fn assert_levels_match_the_pyramid<T: BoxTestSample>(kernel: LevelsKernel<T>) {
        let mut rng = Rng::new(42);
        for levels in [2, 3] {
            for (width, height) in box_sizes(1) {
                for (_, src) in test_images(&mut rng, width * height * RGB, T::MAX) {
                    // SAFETY: NEON is baseline on aarch64; src holds width *
                    // height RGB pixels.
                    let simd = unsafe { kernel(&src, (width, height), levels) };
                    assert_same_image(
                        &simd,
                        &hwc_pyramid_reference(&src, (width, height, RGB), levels),
                        &format!("{levels} levels, {width} x {height}"),
                    );
                }
            }
        }
    }

    #[test]
    fn fused_hwc_neon_kernels_equal_the_pyramid() {
        assert_levels_match_the_pyramid::<u8>(box_levels_u8_hwc_neon);
        assert_levels_match_the_pyramid::<u16>(box_levels_u16_hwc_neon);
    }
}

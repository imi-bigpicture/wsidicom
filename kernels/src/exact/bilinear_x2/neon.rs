//! The NEON kernel of the exact x2 bilinear reduction.
//!
//! # Safety
//! NEON is baseline on aarch64, gated only by the `set_simd` kill switch;
//! every `unsafe fn` here requires the bounds its doc states.

use std::arch::aarch64::{
    uint8x8_t, uint8x8x3_t, uint8x16_t, uint16x4_t, uint16x4x3_t, uint16x8_t, uint32x4_t,
    vaddl_high_u8, vaddl_high_u16, vaddl_u8, vaddl_u16, vaddq_u16, vaddq_u32, vcombine_u8,
    vcombine_u16, vextq_u16, vextq_u32, vget_low_u8, vget_low_u16, vld1q_u8, vld1q_u16, vld1q_u32,
    vld3q_u8, vld3q_u16, vmlaq_n_u16, vmlaq_n_u32, vrshrn_n_u16, vrshrn_n_u32, vst1_u8, vst1_u16,
    vst1q_u8, vst1q_u16, vst1q_u32, vst3_u8, vst3_u16, vuzp1q_u16, vuzp1q_u32, vuzp2q_u16,
    vuzp2q_u32,
};

use super::{X2Sample, divide_rounded, horizontal_scalar, vertical_sums_scalar, window};
use crate::exact::neon::{keep_even_pixels, keep_even_pixels_u16};
use crate::sample::RGB;
use crate::simd::cover_row;

/// Samples per vertical and horizontal step: two registers of u16 sums.
const BLOCK: usize = 16;

/// Rows with a full window of at least two fused blocks in one pass
/// (`fused_row`); narrower ones as the vertical sums `row0 + row3 + 3 (row1 +
/// row2)` in u16, then the rounded 4-tap sum at every sample (at most 64 x
/// 255, `vrshrn` by 6), of which the even pixels are kept. Clipped columns
/// from the same sums, and clipped or too short rows, by the scalar
/// reference's arithmetic.
///
/// # Safety
/// `src` holds `width * height * channels` samples, `channels` 1 or 3.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bilinear_x2_neon(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    let (out_width, out_height, row_samples) = (width / 2, height / 2, width * channels);
    debug_assert!(src.len() >= width * height * channels);
    let mut out_buf = crate::cpu::unwritten_output::<u8>(out_width * out_height * channels);
    // SAFETY: u8 has no invalid values; every sample is written below before
    // the vector is returned.
    let out = unsafe {
        std::slice::from_raw_parts_mut(
            out_buf.as_mut_ptr().cast::<u8>(),
            out_width * out_height * channels,
        )
    };
    let mut scalar_sums = vec![0u32; row_samples];
    // A zero pixel before the row and room after it for the offset loads
    // (those lanes only feed the edge pixels, done apart).
    let mut sums = vec![0u16; channels + row_samples + 2 * channels];
    // The rounded 4-tap sums at every sample.
    let mut full = vec![0u8; row_samples];
    for y in 0..out_height {
        let (first_row, vertical_weights) = window(y, height);
        let vertical_total: u32 = vertical_weights.iter().sum();
        let dst = &mut out[y * out_width * channels..(y + 1) * out_width * channels];
        if vertical_total != 8 || row_samples < BLOCK || out_width < 3 {
            vertical_sums_scalar(
                &src[first_row * row_samples..],
                row_samples,
                vertical_weights,
                &mut scalar_sums,
            );
            horizontal_scalar(
                &scalar_sums,
                channels,
                width,
                vertical_total,
                0..out_width,
                dst,
            );
            continue;
        }
        let rows: [&[u8]; 4] =
            std::array::from_fn(|tap| &src[(first_row + tap) * row_samples..][..row_samples]);
        if width >= 2 * FUSED_BLOCK_PIXELS {
            // SAFETY: the rows hold `width` pixels of `channels` (1 or 3)
            // samples, at least two blocks; dst holds `out_width` outputs.
            unsafe {
                match channels {
                    1 => fused_row::<1, GrayPlanes>(rows, width, dst),
                    _ => fused_row::<RGB, RgbPlanes>(rows, width, dst),
                }
            }
            continue;
        }
        let sums_ptr = sums[channels..].as_mut_ptr();
        cover_row(row_samples, BLOCK, |x| {
            // SAFETY: x + 16 <= row_samples: 16 samples of each row and of the
            // sums.
            unsafe {
                let [top, upper, lower, bottom] = rows.map(|row| vld1q_u8(row.as_ptr().add(x)));
                let low = vmlaq_n_u16(
                    vaddl_u8(vget_low_u8(top), vget_low_u8(bottom)),
                    vaddl_u8(vget_low_u8(upper), vget_low_u8(lower)),
                    3,
                );
                let high = vmlaq_n_u16(vaddl_high_u8(top, bottom), vaddl_high_u8(upper, lower), 3);
                vst1q_u16(sums_ptr.add(x), low);
                vst1q_u16(sums_ptr.add(x + 8), high);
            }
        });
        let sums_ptr = sums[channels..].as_ptr();
        let full_ptr = full.as_mut_ptr();
        cover_row(row_samples, BLOCK, |x| {
            let tap_sums_at = |half: usize| {
                // SAFETY: x + half + 8 <= row_samples; the offsets reach
                // `channels` samples before the row (the zero lead) and 2
                // `channels` past it (the room after).
                let load = |offset: isize| unsafe {
                    vld1q_u16(sums_ptr.offset((x + half) as isize + offset))
                };
                let (pixel, two_pixels) = (channels as isize, 2 * channels as isize);
                vmlaq_n_u16(
                    vaddq_u16(load(-pixel), load(two_pixels)),
                    vaddq_u16(load(0), load(pixel)),
                    3,
                )
            };
            let rounded = vcombine_u8(
                vrshrn_n_u16::<6>(tap_sums_at(0)),
                vrshrn_n_u16::<6>(tap_sums_at(8)),
            );
            // SAFETY: x + 16 <= row_samples, the length of `full`.
            unsafe { vst1q_u8(full_ptr.add(x), rounded) };
        });
        // SAFETY: full holds 2 * out_width pixels of `channels` samples, dst
        // out_width.
        unsafe { keep_even_pixels(&full, channels, out_width, dst) };
        for clipped_output in [0, out_width - 1] {
            let clipped = clipped_output..clipped_output + 1;
            horizontal_scalar(
                &sums[channels..],
                channels,
                width,
                vertical_total,
                clipped,
                dst,
            );
        }
    }
    // SAFETY: every row wrote all of its samples.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// Samples per uint16 step: two registers of u32 sums.
const BLOCK_U16: usize = 8;

/// `bilinear_x2_neon` for uint16, in u32 lanes (the 4-tap sums reach
/// 64 x 65535).
///
/// # Safety
/// `src` holds `width * height * channels` samples, `channels` 1 or 3.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bilinear_x2_u16_neon(
    src: &[u16],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u16> {
    let (out_width, out_height, row_samples) = (width / 2, height / 2, width * channels);
    debug_assert!(src.len() >= width * height * channels);
    let mut out_buf = crate::cpu::unwritten_output::<u16>(out_width * out_height * channels);
    // SAFETY: u16 has no invalid values; every sample is written below before
    // the vector is returned.
    let out = unsafe {
        std::slice::from_raw_parts_mut(
            out_buf.as_mut_ptr().cast::<u16>(),
            out_width * out_height * channels,
        )
    };
    // As in `bilinear_x2_neon`.
    let mut sums = vec![0u32; channels + row_samples + 2 * channels];
    let mut full = vec![0u16; row_samples];
    for y in 0..out_height {
        let (first_row, vertical_weights) = window(y, height);
        let vertical_total: u32 = vertical_weights.iter().sum();
        let dst = &mut out[y * out_width * channels..(y + 1) * out_width * channels];
        if vertical_total != 8 || row_samples < BLOCK_U16 || out_width < 3 {
            let row_sums = &mut sums[channels..channels + row_samples];
            vertical_sums_scalar(
                &src[first_row * row_samples..],
                row_samples,
                vertical_weights,
                row_sums,
            );
            horizontal_scalar(row_sums, channels, width, vertical_total, 0..out_width, dst);
            continue;
        }
        let rows: [&[u16]; 4] =
            std::array::from_fn(|tap| &src[(first_row + tap) * row_samples..][..row_samples]);
        if width >= 2 * FUSED_BLOCK_PIXELS_U16 {
            // SAFETY: the rows hold `width` pixels of `channels` (1 or 3)
            // samples, at least two blocks; dst holds `out_width` outputs.
            unsafe {
                match channels {
                    1 => fused_row_u16::<1, GrayPlanes>(rows, width, dst),
                    _ => fused_row_u16::<RGB, RgbPlanes>(rows, width, dst),
                }
            }
            continue;
        }
        let sums_ptr = sums[channels..].as_mut_ptr();
        cover_row(row_samples, BLOCK_U16, |x| {
            // SAFETY: x + 8 <= row_samples: 8 samples of each row and of the
            // sums.
            unsafe {
                let [top, upper, lower, bottom] = rows.map(|row| vld1q_u16(row.as_ptr().add(x)));
                let low = vmlaq_n_u32(
                    vaddl_u16(vget_low_u16(top), vget_low_u16(bottom)),
                    vaddl_u16(vget_low_u16(upper), vget_low_u16(lower)),
                    3,
                );
                let high =
                    vmlaq_n_u32(vaddl_high_u16(top, bottom), vaddl_high_u16(upper, lower), 3);
                vst1q_u32(sums_ptr.add(x), low);
                vst1q_u32(sums_ptr.add(x + 4), high);
            }
        });
        let sums_ptr = sums[channels..].as_ptr();
        let full_ptr = full.as_mut_ptr();
        cover_row(row_samples, BLOCK_U16, |x| {
            let tap_sums_at = |half: usize| {
                // SAFETY: x + half + 4 <= row_samples; the offsets reach
                // `channels` samples before the row (the zero lead) and 2
                // `channels` past it (the room after).
                let load = |offset: isize| unsafe {
                    vld1q_u32(sums_ptr.offset((x + half) as isize + offset))
                };
                let (pixel, two_pixels) = (channels as isize, 2 * channels as isize);
                vmlaq_n_u32(
                    vaddq_u32(load(-pixel), load(two_pixels)),
                    vaddq_u32(load(0), load(pixel)),
                    3,
                )
            };
            let rounded = vcombine_u16(
                vrshrn_n_u32::<6>(tap_sums_at(0)),
                vrshrn_n_u32::<6>(tap_sums_at(4)),
            );
            // SAFETY: x + 8 <= row_samples, the length of `full`.
            unsafe { vst1q_u16(full_ptr.add(x), rounded) };
        });
        // SAFETY: full holds 2 * out_width pixels of `channels` samples, dst
        // out_width.
        unsafe { keep_even_pixels_u16(&full, channels, out_width, dst) };
        for clipped_output in [0, out_width - 1] {
            let clipped = clipped_output..clipped_output + 1;
            horizontal_scalar(
                &sums[channels..],
                channels,
                width,
                vertical_total,
                clipped,
                dst,
            );
        }
    }
    // SAFETY: every row wrote all of its samples.
    unsafe { crate::cpu::written_output(out_buf) }
}

const FUSED_BLOCK_PIXELS: usize = 16;

/// One output row with a full vertical window, in one pass over its four
/// rows: per block and channel plane, the vertical sums split by `uzp` into
/// even and odd pixels `E`, `O`; output o is `O[o - 1] + 3 (E[o] + O[o]) +
/// E[o + 1]`, the neighbours taken by `vext` from the previous and next
/// blocks, rounded by `vrshrn` by 6. The last block overlaps its
/// predecessor.
///
/// # Safety
/// Each row holds `width >= 2 * FUSED_BLOCK_PIXELS` pixels of `CHANNELS`
/// samples, `dst` `width / 2` outputs.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fused_row<const CHANNELS: usize, Planes: BlockPlanes<CHANNELS>>(
    rows: [&[u8]; 4],
    width: usize,
    dst: &mut [u8],
) {
    let out_width = width / 2;
    let blocks = width / FUSED_BLOCK_PIXELS;
    let row_ptrs = rows.map(<[u8]>::as_ptr);
    // Per plane: (even, odd) pixels.
    let block_sums = |pixel: usize| -> [(uint16x8_t, uint16x8_t); CHANNELS] {
        // SAFETY: pixel + 16 <= width (every caller): the block lies in
        // every row.
        let [top, upper, lower, bottom] =
            row_ptrs.map(|row| unsafe { Planes::load(row.add(pixel * CHANNELS)) });
        std::array::from_fn(|plane| {
            let (outer, inner) = ((top[plane], bottom[plane]), (upper[plane], lower[plane]));
            let low = vmlaq_n_u16(
                vaddl_u8(vget_low_u8(outer.0), vget_low_u8(outer.1)),
                vaddl_u8(vget_low_u8(inner.0), vget_low_u8(inner.1)),
                3,
            );
            let high = vmlaq_n_u16(
                vaddl_high_u8(outer.0, outer.1),
                vaddl_high_u8(inner.0, inner.1),
                3,
            );
            (vuzp1q_u16(low, high), vuzp2q_u16(low, high))
        })
    };
    let dst_ptr = dst.as_mut_ptr();
    let outputs = |pixel: usize,
                   previous_odd: [uint16x8_t; CHANNELS],
                   current: [(uint16x8_t, uint16x8_t); CHANNELS],
                   next_even: [uint16x8_t; CHANNELS]| {
        let outputs: [uint8x8_t; CHANNELS] = std::array::from_fn(|plane| {
            let (even, odd) = current[plane];
            let outer = vaddq_u16(
                vextq_u16::<7>(previous_odd[plane], odd),
                vextq_u16::<1>(even, next_even[plane]),
            );
            vrshrn_n_u16::<6>(vmlaq_n_u16(outer, vaddq_u16(even, odd), 3))
        });
        // SAFETY: pixel + 16 <= width: outputs pixel / 2 .. pixel / 2 + 8
        // of the row.
        unsafe { Planes::store(dst_ptr.add(pixel / 2 * CHANNELS), outputs) };
    };
    let odd_of = |sums: [(uint16x8_t, uint16x8_t); CHANNELS]| sums.map(|(_, odd)| odd);
    let even_of = |sums: [(uint16x8_t, uint16x8_t); CHANNELS]| sums.map(|(even, _)| even);
    // The first block's previous and the last block's next pixels feed only
    // the clipped outputs, rewritten below; a block's own sums stand in.
    let mut current = block_sums(0);
    let mut previous_odd = odd_of(current);
    for block in 0..blocks {
        let pixel = block * FUSED_BLOCK_PIXELS;
        let next = if block + 1 < blocks {
            block_sums(pixel + FUSED_BLOCK_PIXELS)
        } else {
            current
        };
        outputs(pixel, previous_odd, current, even_of(next));
        previous_odd = odd_of(current);
        current = next;
    }
    if !width.is_multiple_of(FUSED_BLOCK_PIXELS) {
        let pixel = width - FUSED_BLOCK_PIXELS;
        let last = block_sums(pixel);
        outputs(
            pixel,
            odd_of(block_sums(pixel - FUSED_BLOCK_PIXELS)),
            last,
            even_of(last),
        );
    }
    for output in [0, out_width - 1] {
        output_from_rows(rows, (width, CHANNELS), output, dst);
    }
}

/// Output `output` of a row with a full vertical window, by the scalar
/// reference's arithmetic.
fn output_from_rows<T: X2Sample>(
    rows: [&[T]; 4],
    (width, channels): (usize, usize),
    output: usize,
    dst: &mut [T],
) {
    const FULL_VERTICAL_TOTAL: u32 = 8;
    let (first_column, horizontal_weights) = window(output, width);
    let denominator = FULL_VERTICAL_TOTAL * horizontal_weights.iter().sum::<u32>();
    for channel in 0..channels {
        let numerator: u32 = horizontal_weights
            .iter()
            .enumerate()
            .map(|(tap, &weight)| {
                let at = (first_column + tap) * channels + channel;
                let vertical = rows[0][at].into()
                    + 3 * (rows[1][at].into() + rows[2][at].into())
                    + rows[3][at].into();
                weight * vertical
            })
            .sum();
        dst[output * channels + channel] = T::narrow(divide_rounded(numerator, denominator));
    }
}

/// A pixel layout's channel planes: 16 pixels split into them, 8 output
/// pixels interleaved back.
trait BlockPlanes<const CHANNELS: usize> {
    /// # Safety
    /// `src..src + 16 * CHANNELS` must be readable.
    unsafe fn load(src: *const u8) -> [uint8x16_t; CHANNELS];

    /// # Safety
    /// `dst..dst + 8 * CHANNELS` must be writable.
    unsafe fn store(dst: *mut u8, planes: [uint8x8_t; CHANNELS]);
}

struct GrayPlanes;
struct RgbPlanes;

impl BlockPlanes<1> for GrayPlanes {
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load(src: *const u8) -> [uint8x16_t; 1] {
        // SAFETY: the caller's contract.
        [unsafe { vld1q_u8(src) }]
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store(dst: *mut u8, planes: [uint8x8_t; 1]) {
        // SAFETY: the caller's contract.
        unsafe { vst1_u8(dst, planes[0]) };
    }
}

impl BlockPlanes<RGB> for RgbPlanes {
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load(src: *const u8) -> [uint8x16_t; RGB] {
        // SAFETY: the caller's contract.
        let planes = unsafe { vld3q_u8(src) };
        [planes.0, planes.1, planes.2]
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store(dst: *mut u8, planes: [uint8x8_t; RGB]) {
        // SAFETY: the caller's contract.
        unsafe { vst3_u8(dst, uint8x8x3_t(planes[0], planes[1], planes[2])) };
    }
}

const FUSED_BLOCK_PIXELS_U16: usize = 8;

/// `fused_row` for uint16, in u32 lanes.
///
/// # Safety
/// Each row holds `width >= 2 * FUSED_BLOCK_PIXELS_U16` pixels of `CHANNELS`
/// samples, `dst` `width / 2` outputs.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fused_row_u16<const CHANNELS: usize, Planes: BlockPlanesU16<CHANNELS>>(
    rows: [&[u16]; 4],
    width: usize,
    dst: &mut [u16],
) {
    let out_width = width / 2;
    let blocks = width / FUSED_BLOCK_PIXELS_U16;
    let row_ptrs = rows.map(<[u16]>::as_ptr);
    let block_sums = |pixel: usize| -> [(uint32x4_t, uint32x4_t); CHANNELS] {
        // SAFETY: pixel + 8 <= width (every caller): the block lies in every
        // row.
        let [top, upper, lower, bottom] =
            row_ptrs.map(|row| unsafe { Planes::load(row.add(pixel * CHANNELS)) });
        std::array::from_fn(|plane| {
            let (outer, inner) = ((top[plane], bottom[plane]), (upper[plane], lower[plane]));
            let low = vmlaq_n_u32(
                vaddl_u16(vget_low_u16(outer.0), vget_low_u16(outer.1)),
                vaddl_u16(vget_low_u16(inner.0), vget_low_u16(inner.1)),
                3,
            );
            let high = vmlaq_n_u32(
                vaddl_high_u16(outer.0, outer.1),
                vaddl_high_u16(inner.0, inner.1),
                3,
            );
            (vuzp1q_u32(low, high), vuzp2q_u32(low, high))
        })
    };
    let dst_ptr = dst.as_mut_ptr();
    let outputs = |pixel: usize,
                   previous_odd: [uint32x4_t; CHANNELS],
                   current: [(uint32x4_t, uint32x4_t); CHANNELS],
                   next_even: [uint32x4_t; CHANNELS]| {
        let outputs: [uint16x4_t; CHANNELS] = std::array::from_fn(|plane| {
            let (even, odd) = current[plane];
            let outer = vaddq_u32(
                vextq_u32::<3>(previous_odd[plane], odd),
                vextq_u32::<1>(even, next_even[plane]),
            );
            vrshrn_n_u32::<6>(vmlaq_n_u32(outer, vaddq_u32(even, odd), 3))
        });
        // SAFETY: pixel + 8 <= width: outputs pixel / 2 .. pixel / 2 + 4 of
        // the row.
        unsafe { Planes::store(dst_ptr.add(pixel / 2 * CHANNELS), outputs) };
    };
    let odd_of = |sums: [(uint32x4_t, uint32x4_t); CHANNELS]| sums.map(|(_, odd)| odd);
    let even_of = |sums: [(uint32x4_t, uint32x4_t); CHANNELS]| sums.map(|(even, _)| even);
    // As in `fused_row`: stand-ins at the ends feed only the clipped outputs.
    let mut current = block_sums(0);
    let mut previous_odd = odd_of(current);
    for block in 0..blocks {
        let pixel = block * FUSED_BLOCK_PIXELS_U16;
        let next = if block + 1 < blocks {
            block_sums(pixel + FUSED_BLOCK_PIXELS_U16)
        } else {
            current
        };
        outputs(pixel, previous_odd, current, even_of(next));
        previous_odd = odd_of(current);
        current = next;
    }
    if !width.is_multiple_of(FUSED_BLOCK_PIXELS_U16) {
        let pixel = width - FUSED_BLOCK_PIXELS_U16;
        let last = block_sums(pixel);
        outputs(
            pixel,
            odd_of(block_sums(pixel - FUSED_BLOCK_PIXELS_U16)),
            last,
            even_of(last),
        );
    }
    for output in [0, out_width - 1] {
        output_from_rows(rows, (width, CHANNELS), output, dst);
    }
}

/// `BlockPlanes` for uint16: 8 pixels split, 4 output pixels interleaved back.
trait BlockPlanesU16<const CHANNELS: usize> {
    /// # Safety
    /// `src..src + 8 * CHANNELS` must be readable.
    unsafe fn load(src: *const u16) -> [uint16x8_t; CHANNELS];

    /// # Safety
    /// `dst..dst + 4 * CHANNELS` must be writable.
    unsafe fn store(dst: *mut u16, planes: [uint16x4_t; CHANNELS]);
}

impl BlockPlanesU16<1> for GrayPlanes {
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load(src: *const u16) -> [uint16x8_t; 1] {
        // SAFETY: the caller's contract.
        [unsafe { vld1q_u16(src) }]
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store(dst: *mut u16, planes: [uint16x4_t; 1]) {
        // SAFETY: the caller's contract.
        unsafe { vst1_u16(dst, planes[0]) };
    }
}

impl BlockPlanesU16<RGB> for RgbPlanes {
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn load(src: *const u16) -> [uint16x8_t; RGB] {
        // SAFETY: the caller's contract.
        let planes = unsafe { vld3q_u16(src) };
        [planes.0, planes.1, planes.2]
    }

    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn store(dst: *mut u16, planes: [uint16x4_t; RGB]) {
        // SAFETY: the caller's contract.
        unsafe { vst3_u16(dst, uint16x4x3_t(planes[0], planes[1], planes[2])) };
    }
}

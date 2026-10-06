//! The NEON kernel of the exact x2 bilinear reduction.
//!
//! # Safety
//! NEON is baseline on aarch64, gated only by the `set_simd` kill switch.
//! Every `unsafe fn` here requires the bounds its doc states.

use std::arch::aarch64::{
    uint8x8_t, uint8x8x3_t, uint8x16_t, uint16x4_t, uint16x4x3_t, uint16x8_t, uint32x4_t,
    vaddl_high_u8, vaddl_high_u16, vaddl_u8, vaddl_u16, vaddq_u16, vaddq_u32, vcombine_u8,
    vcombine_u16, vextq_u16, vextq_u32, vget_low_u8, vget_low_u16, vld1_u16, vld1q_u8, vld1q_u16,
    vld1q_u32, vld3q_u8, vld3q_u16, vmlaq_n_u16, vmlaq_n_u32, vreinterpretq_u16_u64,
    vreinterpretq_u64_u16, vrshrn_n_u16, vrshrn_n_u32, vst1_u8, vst1_u16, vst1q_u8, vst1q_u16,
    vst1q_u32, vst3_u8, vst3_u16, vuzp1q_u16, vuzp1q_u32, vuzp1q_u64, vuzp2q_u16, vuzp2q_u32,
    vuzp2q_u64,
};

use super::{X2Sample, divide_rounded, horizontal_scalar, vertical_sums_scalar, window};
use crate::exact::neon::{keep_even_pixels, keep_even_pixels_u16};
use crate::sample::{RGB, RGBX};
use crate::simd::cover_row;

/// Samples per vertical and horizontal step (two registers of u16 sums).
const BLOCK: usize = 16;

/// The NEON kernel. Rows with a full window (all but the clipped first and
/// last) of at least 32 pixels in one pass (`fused_row`); narrower ones: the
/// vertical sums `r0 + r3 + 3 (r1 + r2)` in u16, the 4-tap horizontal sum
/// at every sample by loads offset by `channels` (`s[x - c] + s[x + 2c] + 3
/// (s[x] + s[x + c])`, at most 64 x 255), rounded once by `vrshrn` (`(sum +
/// 32) >> 6`), then the even pixels kept. The clipped edge columns from the
/// same sums, and the clipped edge rows (and rows too short for a block),
/// by the scalar reference's arithmetic.
///
/// # Safety
/// `src` holds `width * height * channels` samples, `channels` 1, 3 or 4.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bilinear_x2_neon(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    let (ow, oh, ws) = (width / 2, height / 2, width * channels);
    debug_assert!(src.len() >= width * height * channels);
    let mut out_buf = crate::cpu::unwritten_output::<u8>(ow * oh * channels);
    // SAFETY: u8 has no invalid values; every sample is written below before
    // the vector is returned.
    let out = unsafe {
        std::slice::from_raw_parts_mut(out_buf.as_mut_ptr().cast::<u8>(), ow * oh * channels)
    };
    let mut sums32 = vec![0u32; ws];
    // The vertical sums after `channels` zero samples, with room for the
    // offset loads past the row (lanes that feed only the edge pixels).
    let mut sums = vec![0u16; channels + ws + 2 * channels];
    // The rounded 4-tap results at every sample.
    let mut full = vec![0u8; ws];
    for y in 0..oh {
        let (r0, wv) = window(y, height);
        let zv: u32 = wv.iter().sum();
        let dst = &mut out[y * ow * channels..(y + 1) * ow * channels];
        if zv != 8 || ws < BLOCK || ow < 3 {
            vertical_sums_scalar(&src[r0 * ws..], ws, wv, &mut sums32);
            horizontal_scalar(&sums32, channels, width, zv, 0..ow, dst);
            continue;
        }
        let rows: [&[u8]; 4] = std::array::from_fn(|tap| &src[(r0 + tap) * ws..][..ws]);
        if width >= 2 * FUSED_BLOCK_PX {
            // SAFETY: the rows hold `width` pixels of `channels` (1, 3 or 4)
            // samples, at least two blocks; dst holds `ow` outputs.
            unsafe {
                match channels {
                    1 => fused_row::<1, GrayPlanes>(rows, width, dst),
                    RGB => fused_row::<RGB, RgbPlanes>(rows, width, dst),
                    _ => fused_row_rgbx(rows, width, dst),
                }
            }
            continue;
        }
        let sums_ptr = sums[channels..].as_mut_ptr();
        cover_row(ws, BLOCK, |x| {
            // SAFETY: x + 16 <= ws: 16 samples of each row and of the sums.
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
        cover_row(ws, BLOCK, |x| {
            // The 4-tap sums of the 8 samples at x + `half`.
            let taps = |half: usize| {
                // SAFETY: x + half + 8 <= ws; the offsets reach `channels`
                // samples before the row (the zero lead) and 2 `channels`
                // past it (the room after).
                let load = |offset: isize| unsafe {
                    vld1q_u16(sums_ptr.offset((x + half) as isize + offset))
                };
                let (step, two) = (channels as isize, 2 * channels as isize);
                vmlaq_n_u16(
                    vaddq_u16(load(-step), load(two)),
                    vaddq_u16(load(0), load(step)),
                    3,
                )
            };
            let rounded = vcombine_u8(vrshrn_n_u16::<6>(taps(0)), vrshrn_n_u16::<6>(taps(8)));
            // SAFETY: x + 16 <= ws, the length of `full`.
            unsafe { vst1q_u8(full_ptr.add(x), rounded) };
        });
        // SAFETY: full holds 2 * ow pixels of `channels` samples, dst ow.
        unsafe { keep_even_pixels(&full, channels, ow, dst) };
        // The two clipped edge columns.
        for output in [0, ow - 1] {
            let (x0, wh) = window(output, width);
            let denominator = zv * wh.iter().sum::<u32>();
            for ch in 0..channels {
                let numerator: u32 = wh
                    .iter()
                    .enumerate()
                    .map(|(tap, &weight)| {
                        weight * u32::from(sums[channels + (x0 + tap) * channels + ch])
                    })
                    .sum();
                dst[output * channels + ch] = divide_rounded(numerator, denominator) as u8;
            }
        }
    }
    // SAFETY: every row wrote all of its samples.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// Samples per uint16 step (two registers of u32 sums).
const BLOCK_U16: usize = 8;

/// The uint16 kernel: `bilinear_x2_neon`'s steps in u32 lanes (vertical sums
/// reach 8 x 65535, the 4-tap sums 64 x 65535), rounded once by `vrshrn`
/// (`(sum + 32) >> 6`).
///
/// # Safety
/// `src` holds `width * height * channels` samples, `channels` 1, 3 or 4.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bilinear_x2_u16_neon(
    src: &[u16],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u16> {
    let (ow, oh, ws) = (width / 2, height / 2, width * channels);
    debug_assert!(src.len() >= width * height * channels);
    let mut out_buf = crate::cpu::unwritten_output::<u16>(ow * oh * channels);
    // SAFETY: u16 has no invalid values; every sample is written below before
    // the vector is returned.
    let out = unsafe {
        std::slice::from_raw_parts_mut(out_buf.as_mut_ptr().cast::<u16>(), ow * oh * channels)
    };
    // The vertical sums after `channels` zero samples, with room for the
    // offset loads past the row (lanes that feed only the edge pixels).
    let mut sums = vec![0u32; channels + ws + 2 * channels];
    // The rounded 4-tap results at every sample.
    let mut full = vec![0u16; ws];
    for y in 0..oh {
        let (r0, wv) = window(y, height);
        let zv: u32 = wv.iter().sum();
        let dst = &mut out[y * ow * channels..(y + 1) * ow * channels];
        if zv != 8 || ws < BLOCK_U16 || ow < 3 {
            let row_sums = &mut sums[channels..channels + ws];
            vertical_sums_scalar(&src[r0 * ws..], ws, wv, row_sums);
            horizontal_scalar(row_sums, channels, width, zv, 0..ow, dst);
            continue;
        }
        let rows: [&[u16]; 4] = std::array::from_fn(|tap| &src[(r0 + tap) * ws..][..ws]);
        if width >= 2 * FUSED_BLOCK_PX_U16 {
            // SAFETY: the rows hold `width` pixels of `channels` (1, 3 or 4)
            // samples, at least two blocks; dst holds `ow` outputs.
            unsafe {
                match channels {
                    1 => fused_row_u16::<1, GrayPlanes>(rows, width, dst),
                    RGB => fused_row_u16::<RGB, RgbPlanes>(rows, width, dst),
                    _ => fused_row_u16_rgbx(rows, width, dst),
                }
            }
            continue;
        }
        let sums_ptr = sums[channels..].as_mut_ptr();
        cover_row(ws, BLOCK_U16, |x| {
            // SAFETY: x + 8 <= ws: 8 samples of each row and of the sums.
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
        cover_row(ws, BLOCK_U16, |x| {
            // The 4-tap sums of the 4 samples at x + `half`.
            let taps = |half: usize| {
                // SAFETY: x + half + 4 <= ws; the offsets reach `channels`
                // samples before the row (the zero lead) and 2 `channels`
                // past it (the room after).
                let load = |offset: isize| unsafe {
                    vld1q_u32(sums_ptr.offset((x + half) as isize + offset))
                };
                let (step, two) = (channels as isize, 2 * channels as isize);
                vmlaq_n_u32(
                    vaddq_u32(load(-step), load(two)),
                    vaddq_u32(load(0), load(step)),
                    3,
                )
            };
            let rounded = vcombine_u16(vrshrn_n_u32::<6>(taps(0)), vrshrn_n_u32::<6>(taps(4)));
            // SAFETY: x + 8 <= ws, the length of `full`.
            unsafe { vst1q_u16(full_ptr.add(x), rounded) };
        });
        // SAFETY: full holds 2 * ow pixels of `channels` samples, dst ow.
        unsafe { keep_even_pixels_u16(&full, channels, ow, dst) };
        // The two clipped edge columns.
        for output in [0, ow - 1] {
            let (x0, wh) = window(output, width);
            let denominator = zv * wh.iter().sum::<u32>();
            for ch in 0..channels {
                let numerator: u32 = wh
                    .iter()
                    .enumerate()
                    .map(|(tap, &weight)| weight * sums[channels + (x0 + tap) * channels + ch])
                    .sum();
                dst[output * channels + ch] = divide_rounded(numerator, denominator) as u16;
            }
        }
    }
    // SAFETY: every row wrote all of its samples.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// Input pixels per block of `fused_row` (8 outputs).
const FUSED_BLOCK_PX: usize = 16;

/// One output row with a full vertical window, in one pass over its four
/// rows: per block of 16 input pixels and channel plane, the vertical sums
/// `r0 + r3 + 3 (r1 + r2)` in u16, split by `uzp` into even and odd pixels
/// `E`, `O`; output o is `O[o - 1] + 3 (E[o] + O[o]) + E[o + 1]` (at most
/// 64 x 255), its neighbours taken by `vext` from the previous and next
/// blocks, rounded once by `vrshrn` by 6; the last block overlaps its
/// predecessor. The clipped outputs 0 and ow - 1 go through
/// `output_from_rows`.
///
/// # Safety
/// Each row holds `width >= 2 * FUSED_BLOCK_PX` pixels of `CHANNELS`
/// samples, `dst` `width / 2` outputs.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fused_row<const CHANNELS: usize, Planes: BlockPlanes<CHANNELS>>(
    rows: [&[u8]; 4],
    width: usize,
    dst: &mut [u8],
) {
    let ow = width / 2;
    let blocks = width / FUSED_BLOCK_PX;
    let row_ptrs = rows.map(<[u8]>::as_ptr);
    // The even and odd pixels' vertical sums of the 16 pixels from `pixel`,
    // per plane.
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
    // The 8 outputs from input pixel `pixel`, whose block's sums are
    // `current`, the previous block's odd pixels `previous_odd` and the next
    // block's even ones `next_even`.
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
    // The first block's previous and the last block's next pixels feed
    // only the clipped outputs 0 and ow - 1, rewritten below; a block's own
    // sums stand in for them.
    let mut current = block_sums(0);
    let mut previous_odd = odd_of(current);
    for block in 0..blocks {
        let pixel = block * FUSED_BLOCK_PX;
        let next = if block + 1 < blocks {
            block_sums(pixel + FUSED_BLOCK_PX)
        } else {
            current
        };
        outputs(pixel, previous_odd, current, even_of(next));
        previous_odd = odd_of(current);
        current = next;
    }
    // A last block overlapping its predecessor, for a width that is not a
    // multiple of 16 pixels.
    if !width.is_multiple_of(FUSED_BLOCK_PX) {
        let pixel = width - FUSED_BLOCK_PX;
        let last = block_sums(pixel);
        outputs(
            pixel,
            odd_of(block_sums(pixel - FUSED_BLOCK_PX)),
            last,
            even_of(last),
        );
    }
    for output in [0, ow - 1] {
        output_from_rows(rows, (width, CHANNELS), output, dst);
    }
}

/// `fused_row` for RGBX without splitting channels: a pixel's four widened
/// sums are one 64-bit lane, so per block of 16 pixels the vertical sums of
/// each register of 2 pixels pair up by `uzp1`/`uzp2` on 64-bit lanes into
/// even and odd pixels, 2 of each per pair; output o's neighbours `O[o - 1]`
/// and `E[o + 1]` come by `vext` of one pixel (4 lanes) from the adjacent
/// pairs. Rounded once by `vrshrn` by 6, 8 output bytes (2 pixels) per pair.
///
/// # Safety
/// Each row holds `width >= 2 * FUSED_BLOCK_PX` RGBX pixels, `dst`
/// `width / 2`.
#[target_feature(enable = "neon")]
unsafe fn fused_row_rgbx(rows: [&[u8]; 4], width: usize, dst: &mut [u8]) {
    // Pairs of output pixels per block.
    const PAIRS: usize = FUSED_BLOCK_PX / 4;
    let ow = width / 2;
    let blocks = width / FUSED_BLOCK_PX;
    let row_ptrs = rows.map(<[u8]>::as_ptr);
    // The even and odd pixels' vertical sums of the 16 pixels from `pixel`:
    // per pair `(E, O)`, 2 pixels of 4 lanes each.
    let block_sums = |pixel: usize| -> [(uint16x8_t, uint16x8_t); PAIRS] {
        std::array::from_fn(|pair| {
            // SAFETY: pixel + 16 <= width (every caller): 4 pixels of each
            // row from pixel + 4 pair.
            let [top, upper, lower, bottom] =
                row_ptrs.map(|row| unsafe { vld1q_u8(row.add((pixel + 4 * pair) * RGBX)) });
            let front = vmlaq_n_u16(
                vaddl_u8(vget_low_u8(top), vget_low_u8(bottom)),
                vaddl_u8(vget_low_u8(upper), vget_low_u8(lower)),
                3,
            );
            let back = vmlaq_n_u16(vaddl_high_u8(top, bottom), vaddl_high_u8(upper, lower), 3);
            let (front, back) = (vreinterpretq_u64_u16(front), vreinterpretq_u64_u16(back));
            (
                vreinterpretq_u16_u64(vuzp1q_u64(front, back)),
                vreinterpretq_u16_u64(vuzp2q_u64(front, back)),
            )
        })
    };
    let dst_ptr = dst.as_mut_ptr();
    // The 8 outputs from input pixel `pixel`, from the previous, current and
    // next blocks' sums.
    let outputs =
        |pixel: usize, [previous, current, next]: [[(uint16x8_t, uint16x8_t); PAIRS]; 3]| {
            for pair in 0..PAIRS {
                let (even, odd) = current[pair];
                let previous_odd = if pair == 0 {
                    previous[PAIRS - 1].1
                } else {
                    current[pair - 1].1
                };
                let next_even = if pair + 1 == PAIRS {
                    next[0].0
                } else {
                    current[pair + 1].0
                };
                let outer = vaddq_u16(
                    vextq_u16::<4>(previous_odd, odd),
                    vextq_u16::<4>(even, next_even),
                );
                let rounded = vrshrn_n_u16::<6>(vmlaq_n_u16(outer, vaddq_u16(even, odd), 3));
                // SAFETY: pixel + 16 <= width: output pixels pixel / 2 + 2 pair
                // .. + 2 of the row.
                unsafe { vst1_u8(dst_ptr.add((pixel / 2 + 2 * pair) * RGBX), rounded) };
            }
        };
    // As in `fused_row`: stand-ins at the ends feed only the clipped outputs.
    let mut current = block_sums(0);
    let mut previous = current;
    for block in 0..blocks {
        let pixel = block * FUSED_BLOCK_PX;
        let next = if block + 1 < blocks {
            block_sums(pixel + FUSED_BLOCK_PX)
        } else {
            current
        };
        outputs(pixel, [previous, current, next]);
        previous = current;
        current = next;
    }
    if !width.is_multiple_of(FUSED_BLOCK_PX) {
        let pixel = width - FUSED_BLOCK_PX;
        let last = block_sums(pixel);
        outputs(pixel, [block_sums(pixel - FUSED_BLOCK_PX), last, last]);
    }
    for output in [0, ow - 1] {
        output_from_rows(rows, (width, RGBX), output, dst);
    }
}

/// Output `output` of a row with a full vertical window, from its rows, by
/// the scalar reference's arithmetic.
fn output_from_rows<T: X2Sample>(
    rows: [&[T]; 4],
    (width, channels): (usize, usize),
    output: usize,
    dst: &mut [T],
) {
    const FULL_VERTICAL_TOTAL: u32 = 8;
    let (x0, wh) = window(output, width);
    let denominator = FULL_VERTICAL_TOTAL * wh.iter().sum::<u32>();
    for channel in 0..channels {
        let numerator: u32 = wh
            .iter()
            .enumerate()
            .map(|(tap, &weight)| {
                let at = (x0 + tap) * channels + channel;
                let vertical = rows[0][at].into()
                    + 3 * (rows[1][at].into() + rows[2][at].into())
                    + rows[3][at].into();
                weight * vertical
            })
            .sum();
        dst[output * channels + channel] = T::narrow(divide_rounded(numerator, denominator));
    }
}

/// A pixel layout's channel planes: a block of 16 pixels split into them,
/// and 8 output pixels interleaved back from them.
trait BlockPlanes<const CHANNELS: usize> {
    /// The planes of the 16 pixels at `src`.
    ///
    /// # Safety
    /// `src..src + 16 * CHANNELS` must be readable.
    unsafe fn load(src: *const u8) -> [uint8x16_t; CHANNELS];

    /// 8 output pixels from their planes, at `dst`.
    ///
    /// # Safety
    /// `dst..dst + 8 * CHANNELS` must be writable.
    unsafe fn store(dst: *mut u8, planes: [uint8x8_t; CHANNELS]);
}

/// Gray: one plane.
struct GrayPlanes;
/// RGB: three planes by `vld3`/`vst3`.
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

/// Input pixels per block of `fused_row_u16` (4 outputs).
const FUSED_BLOCK_PX_U16: usize = 8;

/// `fused_row` for uint16: blocks of 8 pixels, the vertical sums (at most
/// 8 x 65535) and outputs (at most 64 x 65535) in u32 lanes, rounded once by
/// `vrshrn` by 6.
///
/// # Safety
/// Each row holds `width >= 2 * FUSED_BLOCK_PX_U16` pixels of `CHANNELS`
/// samples, `dst` `width / 2` outputs.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn fused_row_u16<const CHANNELS: usize, Planes: BlockPlanesU16<CHANNELS>>(
    rows: [&[u16]; 4],
    width: usize,
    dst: &mut [u16],
) {
    let ow = width / 2;
    let blocks = width / FUSED_BLOCK_PX_U16;
    let row_ptrs = rows.map(<[u16]>::as_ptr);
    // The even and odd pixels' vertical sums of the 8 pixels from `pixel`,
    // per plane.
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
    // The 4 outputs from input pixel `pixel` (as `fused_row`'s 8).
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
        let pixel = block * FUSED_BLOCK_PX_U16;
        let next = if block + 1 < blocks {
            block_sums(pixel + FUSED_BLOCK_PX_U16)
        } else {
            current
        };
        outputs(pixel, previous_odd, current, even_of(next));
        previous_odd = odd_of(current);
        current = next;
    }
    if !width.is_multiple_of(FUSED_BLOCK_PX_U16) {
        let pixel = width - FUSED_BLOCK_PX_U16;
        let last = block_sums(pixel);
        outputs(
            pixel,
            odd_of(block_sums(pixel - FUSED_BLOCK_PX_U16)),
            last,
            even_of(last),
        );
    }
    for output in [0, ow - 1] {
        output_from_rows(rows, (width, CHANNELS), output, dst);
    }
}

/// `fused_row_u16` for RGBX without splitting channels: a pixel's four
/// widened sums fill one u32 register, so output o is
/// `O[o - 1] + 3 (E[o] + O[o]) + E[o + 1]` over whole registers, pixel by
/// pixel, rounded once by `vrshrn` by 6.
///
/// # Safety
/// Each row holds `width >= 2 * FUSED_BLOCK_PX_U16` RGBX pixels, `dst`
/// `width / 2`.
#[target_feature(enable = "neon")]
unsafe fn fused_row_u16_rgbx(rows: [&[u16]; 4], width: usize, dst: &mut [u16]) {
    let ow = width / 2;
    let row_ptrs = rows.map(<[u16]>::as_ptr);
    // The vertical sums of pixel `pixel`.
    let pixel_sums = |pixel: usize| {
        // SAFETY: pixel < width (every caller): the pixel lies in every row.
        let [top, upper, lower, bottom] =
            row_ptrs.map(|row| unsafe { vld1_u16(row.add(pixel * RGBX)) });
        vmlaq_n_u32(vaddl_u16(top, bottom), vaddl_u16(upper, lower), 3)
    };
    let dst_ptr = dst.as_mut_ptr();
    // Outputs 1 .. ow - 1 (the clipped ends below), each from pixels
    // 2o - 1 ..= 2o + 2, the last two reused as the next output's first.
    let (mut before, mut even) = (pixel_sums(1), pixel_sums(2));
    for output in 1..ow - 1 {
        let (odd, after) = (pixel_sums(2 * output + 1), pixel_sums(2 * output + 2));
        let sums = vmlaq_n_u32(vaddq_u32(before, after), vaddq_u32(even, odd), 3);
        // SAFETY: output < ow: one output pixel of the row.
        unsafe { vst1_u16(dst_ptr.add(output * RGBX), vrshrn_n_u32::<6>(sums)) };
        (before, even) = (odd, after);
    }
    for output in [0, ow - 1] {
        output_from_rows(rows, (width, RGBX), output, dst);
    }
}

/// `BlockPlanes` for uint16: a block of 8 pixels split into channel planes,
/// and 4 output pixels interleaved back.
trait BlockPlanesU16<const CHANNELS: usize> {
    /// The planes of the 8 pixels at `src`.
    ///
    /// # Safety
    /// `src..src + 8 * CHANNELS` must be readable.
    unsafe fn load(src: *const u16) -> [uint16x8_t; CHANNELS];

    /// 4 output pixels from their planes, at `dst`.
    ///
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

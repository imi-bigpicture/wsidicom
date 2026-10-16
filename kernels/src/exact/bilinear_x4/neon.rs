//! The NEON kernel of the exact x4 bilinear reduction.
//!
//! # Safety
//! NEON is baseline on aarch64, gated only by the `set_simd` kill switch.
//! Every `unsafe fn` here requires the bounds its doc states.

use std::arch::aarch64::{
    uint8x16_t, uint8x16x3_t, uint8x16x4_t, uint16x4_t, uint16x8_t, uint16x8x4_t, vaddq_u16,
    vaddq_u32, vcombine_u8, vcombine_u16, vdup_n_u8, vdupq_n_u16, vdupq_n_u32, vextq_u16,
    vget_high_u8, vget_high_u16, vget_low_u8, vget_low_u16, vgetq_lane_u32, vld1q_u8, vld1q_u16,
    vld3q_u8, vld4q_u8, vld4q_u16, vmlal_high_n_u16, vmlal_n_u16, vmlal_u8, vmovl_u16, vmovn_u16,
    vmull_high_n_u16, vmull_n_u16, vmull_u8, vqtbl1q_u8, vreinterpretq_u32_u8, vrshrn_n_u32,
    vst1_u8, vst1q_u8, vst1q_u16, vst3q_u8, vst4q_u8, vst4q_u16, vuzp1q_u8,
};

use super::{TAPS, divide_rounded, output_scalar, row_scalar, window};
use crate::sample::{RGB, RGBX};
use crate::simd::cover_row;

/// Samples per vertical step (u8 rows, two registers of u16 sums).
const VERTICAL_BLOCK: usize = 16;
/// Samples per horizontal step (two registers of u32 sums).
const HORIZONTAL_BLOCK: usize = 8;
/// Taps before the output's first input of the window: o reads 4o - 2 ..= 4o + 5.
const LEAD: usize = 2;
/// The window total of a full window (and so its square, of a full 2D window).
const FULL_TOTAL: u32 = 32;

/// The NEON kernel. Rows with a full window: the vertical sums of the 8 rows
/// in u16 (at most 32 x 255) by `vmlal_u8`, then the 8-tap horizontal sums
/// in u32 rounded once by `vrshrn` by 10 (`(sum + 512) >> 10`, the full
/// windows' total 1024). Gray and colour rows wide enough take the
/// horizontal pass only at the outputs; narrower rows take it at every
/// sample from loads `channels` apart (`vmlal_n_u16`) and keep every 4th
/// pixel. Columns and rows with clipped windows, and rows too short for a
/// block, by the scalar reference's arithmetic.
///
/// # Safety
/// `src` holds `width * height * channels` samples, `channels` 1, 3 or 4.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bilinear_x4_neon(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    let (ow, oh, ws) = (width / 4, height / 4, width * channels);
    debug_assert!(src.len() >= width * height * channels);
    let mut out_buf = crate::cpu::unwritten_output::<u8>(ow * oh * channels);
    // SAFETY: u8 has no invalid values; every sample is written below before
    // the vector is returned.
    let out = unsafe {
        std::slice::from_raw_parts_mut(out_buf.as_mut_ptr().cast::<u8>(), ow * oh * channels)
    };
    let mut sums32 = vec![0u32; ws];
    // The vertical sums after LEAD zero pixels, with room for the offset
    // loads past the row (lanes that feed only the clipped edge pixels).
    let lead = LEAD * channels;
    let mut sums = vec![0u16; lead + ws + (TAPS.len() - LEAD) * channels];
    // The rounded 8-tap results at every sample.
    let mut full = vec![0u8; ws];
    let weights = TAPS.map(|weight| weight as u8);
    let colour_outputs = channels != 1 && ow >= 2 + COLOUR_OUTPUTS;
    // RGB rows' vertical sums laid out as RGBX, for the colour pass.
    let mut sums_rgbx = vec![
        0u16;
        if colour_outputs && channels == RGB {
            RGBX * width
        } else {
            0
        }
    ];
    // The columns whose windows are clipped (at the row's two ends).
    let clipped_columns: Vec<usize> = (0..ow)
        .filter(|&output| window(output, width).1.len() < TAPS.len())
        .collect();
    for y in 0..oh {
        let (r0, wv) = window(y, height);
        let dst = &mut out[y * ow * channels..(y + 1) * ow * channels];
        if wv.len() < TAPS.len() || ws < VERTICAL_BLOCK {
            row_scalar(src, (width, height, channels), y, &mut sums32, dst);
            continue;
        }
        let rows: [&[u8]; 8] = std::array::from_fn(|tap| &src[(r0 + tap) * ws..][..ws]);
        let vertical_sums = |samples: [uint8x16_t; 8]| {
            let mut low = vmull_u8(vget_low_u8(samples[0]), vdup_n_u8(weights[0]));
            let mut high = vmull_u8(vget_high_u8(samples[0]), vdup_n_u8(weights[0]));
            for (row, &weight) in samples.iter().zip(&weights).skip(1) {
                low = vmlal_u8(low, vget_low_u8(*row), vdup_n_u8(weight));
                high = vmlal_u8(high, vget_high_u8(*row), vdup_n_u8(weight));
            }
            (low, high)
        };
        if colour_outputs && channels == RGB {
            let sums_ptr = sums_rgbx.as_mut_ptr();
            let zero = vdupq_n_u16(0);
            cover_row(width, VERTICAL_BLOCK, |pixel| {
                // SAFETY: pixel + 16 <= width: 16 pixels of each row.
                let planes = rows.map(|row| unsafe { vld3q_u8(row.as_ptr().add(pixel * RGB)) });
                let (red_low, red_high) = vertical_sums(planes.map(|split| split.0));
                let (green_low, green_high) = vertical_sums(planes.map(|split| split.1));
                let (blue_low, blue_high) = vertical_sums(planes.map(|split| split.2));
                // SAFETY: pixel + 16 <= width: inside the RGBX row of sums.
                unsafe {
                    let low = uint16x8x4_t(red_low, green_low, blue_low, zero);
                    vst4q_u16(sums_ptr.add(pixel * RGBX), low);
                    let high = uint16x8x4_t(red_high, green_high, blue_high, zero);
                    vst4q_u16(sums_ptr.add((pixel + 8) * RGBX), high);
                }
            });
        } else {
            let sums_ptr = sums[lead..].as_mut_ptr();
            cover_row(ws, VERTICAL_BLOCK, |x| {
                // SAFETY: x + 16 <= ws: 16 samples of each row.
                let (low, high) =
                    vertical_sums(rows.map(|row| unsafe { vld1q_u8(row.as_ptr().add(x)) }));
                // SAFETY: x + 16 <= ws: inside the row of sums.
                unsafe {
                    vst1q_u16(sums_ptr.add(x), low);
                    vst1q_u16(sums_ptr.add(x + 8), high);
                }
            });
        }
        if colour_outputs {
            let sums = if channels == RGB {
                &sums_rgbx[..]
            } else {
                &sums[lead..][..ws]
            };
            // SAFETY: sums holds the row's `width` = 4 ow RGBX vertical sums,
            // dst ow outputs of `channels` (3 or 4) samples.
            unsafe { horizontal_colour(sums.as_ptr(), channels, ow, dst) };
            for output in [0, ow - 1] {
                clipped_colour_output(sums, (width, channels), output, dst);
            }
            continue;
        }
        let sums_ptr = sums[lead..].as_ptr();
        if channels == 1 && ow >= 2 + PHASED_OUTPUTS {
            // SAFETY: sums holds the row's `width` = 4 ow vertical sums from
            // sums_ptr, dst ow outputs.
            unsafe { horizontal_phased(sums_ptr, ow, dst) };
        } else {
            let full_ptr = full.as_mut_ptr();
            cover_row(ws, HORIZONTAL_BLOCK, |x| {
                let (mut low, mut high) = (vdupq_n_u32(0), vdupq_n_u32(0));
                for (tap, &weight) in TAPS.iter().enumerate() {
                    let offset = x as isize + (tap as isize - LEAD as isize) * channels as isize;
                    // SAFETY: x + 8 <= ws; the offsets reach LEAD pixels before
                    // the row (the zero lead) and 5 past it (the room after).
                    let window_sums = unsafe { vld1q_u16(sums_ptr.offset(offset)) };
                    let weight = weight as u16;
                    low = vmlal_n_u16(low, vget_low_u16(window_sums), weight);
                    high = vmlal_n_u16(high, vget_high_u16(window_sums), weight);
                }
                let rounded = vcombine_u16(vrshrn_n_u32::<10>(low), vrshrn_n_u32::<10>(high));
                // SAFETY: x + 8 <= ws, the length of `full`; the means are bytes.
                unsafe { vst1_u8(full_ptr.add(x), vmovn_u16(rounded)) };
            });
            // SAFETY: full holds 4 * ow pixels of `channels` samples, dst ow.
            unsafe { keep_every_fourth_pixel(&full, channels, ow, dst) };
        }
        for &output in &clipped_columns {
            output_scalar(&sums[lead..], (width, channels), FULL_TOTAL, output, dst);
        }
    }
    // SAFETY: every row wrote all of its samples.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// Output pixels per step of `horizontal_phased`.
const PHASED_OUTPUTS: usize = 8;

/// A gray row's horizontal pass only at the outputs whose windows are full
/// (1 .. ow - 1): `vld4` splits the vertical sums into the 4 pixel phases,
/// so output o's taps 4o - 2 ..= 4o + 5 are phases 2, 3 of o - 1, all four
/// of o, and 0, 1 of o + 1, weighted 1, 3, 5, 7, 7, 5, 3, 1 — summed in
/// symmetric pairs (each at most 2 x 8160, u16), widened to u32, rounded
/// once by `vrshrn` by 10. The clipped outputs 0 and ow - 1 are left to the
/// caller.
///
/// # Safety
/// `sums` points at the row's `4 ow` vertical sums and `dst` holds `ow`
/// outputs, `ow >= 2 + PHASED_OUTPUTS`.
#[target_feature(enable = "neon")]
unsafe fn horizontal_phased(sums: *const u16, ow: usize, dst: &mut [u8]) {
    let dst_ptr = dst.as_mut_ptr();
    cover_row(ow - 2, PHASED_OUTPUTS, |index| {
        let output = 1 + index;
        // SAFETY: 1 <= output and output + 8 <= ow - 1, so the loads read the
        // phases of outputs output - 1 .. output + 9: sums 4 (output - 1) ..
        // 4 (output + 9) <= 4 ow.
        let [before, center, after] =
            [output - 1, output, output + 1].map(|first| unsafe { vld4q_u16(sums.add(4 * first)) });
        let pairs = [
            vaddq_u16(before.2, after.1),
            vaddq_u16(before.3, after.0),
            vaddq_u16(center.0, center.3),
            vaddq_u16(center.1, center.2),
        ];
        let weights = [TAPS[0], TAPS[1], TAPS[2], TAPS[3]].map(|weight| weight as u16);
        let mut low = vmull_n_u16(vget_low_u16(pairs[0]), weights[0]);
        let mut high = vmull_high_n_u16(pairs[0], weights[0]);
        for (pair, &weight) in pairs.iter().zip(&weights).skip(1) {
            low = vmlal_n_u16(low, vget_low_u16(*pair), weight);
            high = vmlal_high_n_u16(high, *pair, weight);
        }
        let rounded = vcombine_u16(vrshrn_n_u32::<10>(low), vrshrn_n_u32::<10>(high));
        // SAFETY: output + 8 <= ow: outputs of the row.
        unsafe { vst1_u8(dst_ptr.add(output), vmovn_u16(rounded)) };
    });
}

/// Output pixels per step of `horizontal_colour`.
const COLOUR_OUTPUTS: usize = 4;

/// One output pixel of a colour row from its RGBX vertical sums: the 8
/// window pixels as 4 registers of 2 pixels, summed in pixel pairs of equal
/// weight (the last two registers' pixels swapped by `vextq`: -2 + 5, -1 + 4
/// weighted 1 and 3; 0 + 3, 1 + 2 weighted 5 and 7; each at most
/// 2 x 8160, u16), widened to u32, rounded once by `vrshrn` by 10.
///
/// # Safety
/// `sums` points at the RGBX vertical sums of pixels `4 output - 2 ..
/// 4 output + 6`.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn colour_output(sums: *const u16) -> uint16x4_t {
    // SAFETY: the window's 8 RGBX pixels (the caller's contract).
    let pairs: [uint16x8_t; 4] = [0, 1, 2, 3].map(|pair| unsafe { vld1q_u16(sums.add(8 * pair)) });
    let outer = vaddq_u16(pairs[0], vextq_u16::<4>(pairs[3], pairs[3]));
    let inner = vaddq_u16(pairs[1], vextq_u16::<4>(pairs[2], pairs[2]));
    let low = vmlal_n_u16(
        vmovl_u16(vget_low_u16(outer)),
        vget_low_u16(inner),
        TAPS[2] as u16,
    );
    let high = vmlal_high_n_u16(
        vmull_high_n_u16(outer, TAPS[1] as u16),
        inner,
        TAPS[3] as u16,
    );
    vrshrn_n_u32::<10>(vaddq_u32(low, high))
}

/// A colour row's horizontal pass only at the outputs whose windows are
/// full (1 .. ow - 1), 4 per step from RGBX sums; RGB keeps 3 of each 4
/// bytes by `vqtbl1q`. The clipped outputs 0 and ow - 1 are left to the
/// caller.
///
/// # Safety
/// `sums` points at the row's `4 ow` RGBX vertical sums and `dst` holds `ow`
/// outputs of `channels` (3 or 4) samples, `ow >= 2 + COLOUR_OUTPUTS`.
#[target_feature(enable = "neon")]
unsafe fn horizontal_colour(sums: *const u16, channels: usize, ow: usize, dst: &mut [u8]) {
    const RGB_BYTES: [u8; 16] = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 16, 16, 16];
    // SAFETY: the 16 bytes of the constant.
    let rgb_bytes = unsafe { vld1q_u8(RGB_BYTES.as_ptr()) };
    let dst_ptr = dst.as_mut_ptr();
    cover_row(ow - 2, COLOUR_OUTPUTS, |index| {
        let first = 1 + index;
        // SAFETY: 1 <= first and first + 4 <= ow - 1, so every window's
        // pixels 4 output - 2 .. 4 output + 6 lie within the row's 4 ow.
        let outputs: [uint16x4_t; 4] = std::array::from_fn(|output| unsafe {
            colour_output(sums.add(RGBX * (4 * (first + output) - LEAD)))
        });
        let rgbx = vcombine_u8(
            vmovn_u16(vcombine_u16(outputs[0], outputs[1])),
            vmovn_u16(vcombine_u16(outputs[2], outputs[3])),
        );
        if channels == RGB {
            let rgb = vqtbl1q_u8(rgbx, rgb_bytes);
            // SAFETY: first + 4 <= ow: 12 bytes of the row's outputs.
            unsafe {
                vst1_u8(dst_ptr.add(first * RGB), vget_low_u8(rgb));
                let last = vgetq_lane_u32::<2>(vreinterpretq_u32_u8(rgb));
                dst_ptr
                    .add(first * RGB + 8)
                    .cast::<u32>()
                    .write_unaligned(last);
            }
        } else {
            // SAFETY: first + 4 <= ow: 16 bytes of the row's outputs.
            unsafe { vst1q_u8(dst_ptr.add(first * RGBX), rgbx) };
        }
    });
}

/// Output `output` (its window clipped) of a colour row from its RGBX
/// vertical sums, by the scalar reference's arithmetic.
fn clipped_colour_output(
    sums: &[u16],
    (width, channels): (usize, usize),
    output: usize,
    dst: &mut [u8],
) {
    let (x0, wh) = window(output, width);
    let denominator = FULL_TOTAL * wh.iter().sum::<u32>();
    for channel in 0..channels {
        let numerator: u32 = wh
            .iter()
            .enumerate()
            .map(|(tap, &weight)| weight * u32::from(sums[(x0 + tap) * RGBX + channel]))
            .sum();
        dst[output * channels + channel] = divide_rounded(numerator, denominator);
    }
}

/// `dst[o]` = pixel `4 o` of `full`, for the `ow` output pixels: 16 per
/// step, by `vld1`/`vld3`/`vld4` (whole channel planes) and two `vuzp1`
/// levels (every 4th lane); scalar tail.
///
/// # Safety
/// `full` holds `4 * ow` pixels of `channels` (1, 3 or 4) samples and `dst`
/// `ow`.
#[target_feature(enable = "neon")]
unsafe fn keep_every_fourth_pixel(full: &[u8], channels: usize, ow: usize, dst: &mut [u8]) {
    const STEP_PX: usize = 16;
    debug_assert!(full.len() >= 4 * ow * channels && dst.len() >= ow * channels);
    let (full_ptr, dst_ptr) = (full.as_ptr(), dst.as_mut_ptr());
    // Lanes 0, 4, 8, .. of four consecutive registers of one plane.
    let fourths = |plane: [uint8x16_t; 4]| {
        vuzp1q_u8(vuzp1q_u8(plane[0], plane[1]), vuzp1q_u8(plane[2], plane[3]))
    };
    let mut output = 0usize;
    while output + STEP_PX <= ow {
        let at = 4 * output * channels;
        // SAFETY: output + 16 <= ow: the loads read pixels 4 output ..
        // 4 output + 64 of full, the store output pixels output .. + 16.
        unsafe {
            match channels {
                1 => {
                    let plane = [0, 1, 2, 3].map(|block| vld1q_u8(full_ptr.add(at + 16 * block)));
                    vst1q_u8(dst_ptr.add(output), fourths(plane));
                }
                RGB => {
                    let blocks = [0, 1, 2, 3].map(|block| vld3q_u8(full_ptr.add(at + 48 * block)));
                    let planes = uint8x16x3_t(
                        fourths(blocks.map(|split| split.0)),
                        fourths(blocks.map(|split| split.1)),
                        fourths(blocks.map(|split| split.2)),
                    );
                    vst3q_u8(dst_ptr.add(output * RGB), planes);
                }
                _ => {
                    let blocks = [0, 1, 2, 3].map(|block| vld4q_u8(full_ptr.add(at + 64 * block)));
                    let planes = uint8x16x4_t(
                        fourths(blocks.map(|split| split.0)),
                        fourths(blocks.map(|split| split.1)),
                        fourths(blocks.map(|split| split.2)),
                        fourths(blocks.map(|split| split.3)),
                    );
                    vst4q_u8(dst_ptr.add(output * RGBX), planes);
                }
            }
        }
        output += STEP_PX;
    }
    for output in output..ow {
        let at = 4 * output * channels;
        dst[output * channels..(output + 1) * channels].copy_from_slice(&full[at..at + channels]);
    }
}

// The rounding shift by 10 divides by a full 2D window's total.
const _: () = assert!(FULL_TOTAL * FULL_TOTAL == 1 << 10);

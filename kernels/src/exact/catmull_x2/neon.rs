//! The NEON kernel of the exact x2 catmullrom reduction.
//!
//! # Safety
//! NEON is baseline on aarch64, gated only by the `set_simd` kill switch;
//! every `unsafe fn` here requires the bounds its doc states.

use std::arch::aarch64::{
    int32x4_t, uint8x8_t, vaddl_u8, vaddq_s32, vaddw_high_s16, vaddw_s16, vcombine_u16, vdup_n_u16,
    vdupq_n_s32, vget_low_s16, vget_low_u16, vld1_u8, vld1q_s32, vld2q_s32, vmlaq_n_s16,
    vmlaq_n_s32, vmovl_high_u16, vmovl_u16, vmulq_n_s16, vmulq_n_u16, vqmovn_u16, vqrshrun_n_s32,
    vreinterpretq_s16_u16, vreinterpretq_s32_u32, vst1_u8, vst1q_s32,
};

use super::{CATMULL_TAPS, catmull_row_scalar, catmull_window, divide_rounded_clamped};
use crate::exact::neon::keep_even_pixels;
use crate::simd::cover_row;

/// Samples per vertical and horizontal step: two registers of i32 sums.
const BLOCK: usize = 8;
const TAP_COUNT: usize = CATMULL_TAPS.len();
/// The weight of the two center rows 3 and 4.
const CENTER_WEIGHT: u16 = CATMULL_TAPS[3] as u16;
/// The weights of the outer row pairs (0, 7), (1, 6), (2, 5).
const OUTER_WEIGHTS: [i16; 3] = [
    CATMULL_TAPS[0] as i16,
    CATMULL_TAPS[1] as i16,
    CATMULL_TAPS[2] as i16,
];
// The vertical pass sums the rows in symmetric pairs, so the taps must be.
const _: () = {
    let mut tap = 0;
    while tap < TAP_COUNT {
        assert!(CATMULL_TAPS[tap] == CATMULL_TAPS[TAP_COUNT - 1 - tap]);
        tap += 1;
    }
};
/// Inputs of a window before the output's own: o reads 2o - 3 ..= 2o + 4.
const LEAD: usize = 3;
/// A full window's weight total along one axis.
const FULL_TOTAL: i64 = 256;

/// In exact i32 arithmetic. Rows with a full window: the vertical sums (at
/// most 280 x 255), then the 8-tap horizontal sum at every sample (|sum| <
/// 2^25), rounded and clamped once — `vqrshrun` by 16 is `(sum + 2^15) >>
/// 16` saturated to [0, 65535], the scalar `floor((2 n + 65536) / 131072)` —
/// then narrowed to u8 saturating, and the even pixels kept (wide enough gray
/// rows take the horizontal pass only at the outputs). Clipped columns, and
/// clipped or too short rows, by the scalar reference's arithmetic.
///
/// # Safety
/// `src` holds `width * height * channels` samples, `channels` 1 or 3,
/// `width` and `height` even (with an odd width the gray horizontal pass
/// would leave output `out_width - 2` unwritten).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn catmull_x2_neon(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    let (out_width, out_height, row_samples) = (width / 2, height / 2, width * channels);
    debug_assert!(src.len() >= width * height * channels);
    debug_assert!(width.is_multiple_of(2) && height.is_multiple_of(2));
    let mut out_buf = crate::cpu::unwritten_output::<u8>(out_width * out_height * channels);
    // SAFETY: u8 has no invalid values; every sample is written below before
    // the vector is returned.
    let out = unsafe {
        std::slice::from_raw_parts_mut(
            out_buf.as_mut_ptr().cast::<u8>(),
            out_width * out_height * channels,
        )
    };
    let mut scalar_sums = vec![0i64; row_samples];
    // LEAD zero pixels before the row and room after it for the offset loads
    // (those lanes only feed the clipped edge pixels).
    let lead = LEAD * channels;
    let mut sums = vec![0i32; lead + row_samples + (TAP_COUNT - LEAD) * channels];
    // The rounded 8-tap sums at every sample.
    let mut full = vec![0u8; row_samples];
    let full_window = |(_, weights): (usize, &[i32])| weights.len() == TAP_COUNT;
    let clipped_columns: Vec<usize> = (0..out_width)
        .filter(|&output| !full_window(catmull_window(output, width)))
        .collect();
    for y in 0..out_height {
        let (first_row, vertical_weights) = catmull_window(y, height);
        let dst = &mut out[y * out_width * channels..(y + 1) * out_width * channels];
        if !full_window((first_row, vertical_weights)) || row_samples < BLOCK {
            // The scalar reference's sums, a row at a time so the loop
            // vectorizes.
            scalar_sums.fill(0);
            for (tap, &weight) in vertical_weights.iter().enumerate() {
                for (sum, &sample) in scalar_sums
                    .iter_mut()
                    .zip(&src[(first_row + tap) * row_samples..][..row_samples])
                {
                    *sum += i64::from(weight) * i64::from(sample);
                }
            }
            let vertical_total: i64 = vertical_weights
                .iter()
                .map(|&weight| i64::from(weight))
                .sum();
            catmull_row_scalar(
                &scalar_sums,
                channels,
                width,
                vertical_total,
                0..out_width,
                dst,
            );
            continue;
        }
        let rows: [&[u8]; TAP_COUNT] =
            std::array::from_fn(|tap| &src[(first_row + tap) * row_samples..][..row_samples]);
        let sums_ptr = sums[lead..].as_mut_ptr();
        cover_row(row_samples, BLOCK, |x| {
            // SAFETY: x + 8 <= row_samples: 8 samples of each row.
            let samples: [uint8x8_t; TAP_COUNT] =
                rows.map(|row| unsafe { vld1_u8(row.as_ptr().add(x)) });
            // Each at most 510.
            let symmetric_pair = |tap: usize| vaddl_u8(samples[tap], samples[TAP_COUNT - 1 - tap]);
            // 111 (row3 + row4) <= 56610 fits u16; the outer pairs' sum
            // 29 (row2 + row5) - 9 (row1 + row6) - 3 (row0 + row7) lies in
            // [-6120, 14790] and fits i16; they meet in i32.
            let center = vmulq_n_u16(symmetric_pair(3), CENTER_WEIGHT);
            let mut outer = vmulq_n_s16(vreinterpretq_s16_u16(symmetric_pair(2)), OUTER_WEIGHTS[2]);
            outer = vmlaq_n_s16(
                outer,
                vreinterpretq_s16_u16(symmetric_pair(1)),
                OUTER_WEIGHTS[1],
            );
            outer = vmlaq_n_s16(
                outer,
                vreinterpretq_s16_u16(symmetric_pair(0)),
                OUTER_WEIGHTS[0],
            );
            let low = vaddw_s16(
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(center))),
                vget_low_s16(outer),
            );
            let high = vaddw_high_s16(vreinterpretq_s32_u32(vmovl_high_u16(center)), outer);
            // SAFETY: x + 8 <= row_samples: inside the row of sums.
            unsafe {
                vst1q_s32(sums_ptr.add(x), low);
                vst1q_s32(sums_ptr.add(x + 4), high);
            }
        });
        let sums_ptr = sums[lead..].as_ptr();
        if channels == 1 && out_width >= 4 + PHASED_OUTPUTS {
            // SAFETY: sums holds the row's `width` = 2 out_width vertical sums
            // from sums_ptr, dst out_width outputs.
            unsafe { horizontal_phased(sums_ptr, out_width, dst) };
        } else {
            let full_ptr = full.as_mut_ptr();
            cover_row(row_samples, BLOCK, |x| {
                let tap_sums_at = |half: usize| -> int32x4_t {
                    let mut acc = vdupq_n_s32(0);
                    for (tap, &weight) in CATMULL_TAPS.iter().enumerate() {
                        let offset = (x + half) as isize
                            + (tap as isize - LEAD as isize) * channels as isize;
                        // SAFETY: x + half + 4 <= row_samples; the offsets
                        // reach LEAD pixels before the row (the zero lead) and
                        // TAP_COUNT - LEAD - 1 past it (the room after).
                        let window_sums = unsafe { vld1q_s32(sums_ptr.offset(offset)) };
                        acc = vmlaq_n_s32(acc, window_sums, weight);
                    }
                    acc
                };
                let rounded = vcombine_u16(
                    vqrshrun_n_s32::<16>(tap_sums_at(0)),
                    vqrshrun_n_s32::<16>(tap_sums_at(4)),
                );
                // SAFETY: x + 8 <= row_samples, the length of `full`.
                unsafe { vst1_u8(full_ptr.add(x), vqmovn_u16(rounded)) };
            });
            // SAFETY: full holds 2 * out_width pixels of `channels` samples,
            // dst out_width.
            unsafe { keep_even_pixels(&full, channels, out_width, dst) };
        }
        for &output in &clipped_columns {
            let (first_column, horizontal_weights) = catmull_window(output, width);
            let denominator = FULL_TOTAL
                * horizontal_weights
                    .iter()
                    .map(|&weight| i64::from(weight))
                    .sum::<i64>();
            for channel in 0..channels {
                let numerator: i64 = horizontal_weights
                    .iter()
                    .enumerate()
                    .map(|(tap, &weight)| {
                        i64::from(weight)
                            * i64::from(sums[lead + (first_column + tap) * channels + channel])
                    })
                    .sum();
                dst[output * channels + channel] = divide_rounded_clamped(numerator, denominator);
            }
        }
    }
    // SAFETY: every row wrote all of its samples.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// Output pixels per step of `horizontal_phased`: two registers of i32 sums.
const PHASED_OUTPUTS: usize = 8;
const I32_LANES: usize = 4;

/// A gray row's horizontal pass at the outputs whose windows are full
/// (`2..out_width - 2`): `vld2` splits the vertical sums into even and odd
/// samples `E`, `O`, so output o's taps 2o - 3 ..= 2o + 4 are `O[o - 2]`,
/// `E[o - 1]`, `O[o - 1]`, `E[o]`, `O[o]`, `E[o + 1]`, `O[o + 1]`,
/// `E[o + 2]`, summed in symmetric pairs, then weighted and rounded as
/// `catmull_x2_neon`.
///
/// # Safety
/// `sums` points at the row's `2 out_width` vertical sums and `dst` holds
/// `out_width` outputs, `out_width >= 4 + PHASED_OUTPUTS`.
#[target_feature(enable = "neon")]
unsafe fn horizontal_phased(sums: *const i32, out_width: usize, dst: &mut [u8]) {
    let dst_ptr = dst.as_mut_ptr();
    let mut rounded = [vdup_n_u16(0); 2];
    cover_row(out_width - 4, PHASED_OUTPUTS, |index| {
        let first = 2 + index;
        for (half, rounded) in rounded.iter_mut().enumerate() {
            let output = first + half * I32_LANES;
            // SAFETY: 2 <= output and output + 4 <= out_width - 2, so the loads
            // read sums 2 (output - 2) .. 2 (output + 6) <= 2 out_width.
            let [two_before, before, center, after, two_after] = [0, 1, 2, 3, 4]
                .map(|phase| unsafe { vld2q_s32(sums.add(2 * (output + phase - 2))) });
            let pairs = [
                vaddq_s32(two_before.1, two_after.0),
                vaddq_s32(before.0, after.1),
                vaddq_s32(before.1, after.0),
                vaddq_s32(center.0, center.1),
            ];
            let mut acc = vdupq_n_s32(0);
            for (pair, &weight) in pairs.iter().zip(&CATMULL_TAPS) {
                acc = vmlaq_n_s32(acc, *pair, weight);
            }
            *rounded = vqrshrun_n_s32::<16>(acc);
        }
        let narrowed = vqmovn_u16(vcombine_u16(rounded[0], rounded[1]));
        // SAFETY: first + 8 <= out_width - 2: outputs of the row.
        unsafe { vst1_u8(dst_ptr.add(first), narrowed) };
    });
}

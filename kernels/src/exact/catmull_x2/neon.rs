//! The NEON kernel of the exact x2 catmullrom reduction.
//!
//! # Safety
//! NEON is baseline on aarch64, gated only by the `set_simd` kill switch.
//! Every `unsafe fn` here requires the bounds its doc states.

use std::arch::aarch64::{
    int32x4_t, uint8x8_t, vaddl_u8, vaddq_s32, vaddw_high_s16, vaddw_s16, vcombine_u16, vdup_n_u16,
    vdupq_n_s32, vget_low_s16, vget_low_u16, vld1_u8, vld1q_s32, vld2q_s32, vmlaq_n_s16,
    vmlaq_n_s32, vmovl_high_u16, vmovl_u16, vmulq_n_s16, vmulq_n_u16, vqmovn_u16, vqrshrun_n_s32,
    vreinterpretq_s16_u16, vreinterpretq_s32_u32, vst1_u8, vst1q_s32,
};

use super::{CATMULL_TAPS, catmull_row_scalar, catmull_window, divide_rounded_clamped};
use crate::exact::neon::keep_even_pixels;
use crate::simd::cover_row;

/// Samples per vertical and horizontal step (two registers of i32 sums).
const BLOCK: usize = 8;
/// Taps of a full window.
const TAPS: usize = CATMULL_TAPS.len();
/// The weight of the two center rows (r3, r4).
const CENTER_WEIGHT: u16 = CATMULL_TAPS[3] as u16;
/// The weights of the outer row pairs (r0, r7), (r1, r6), (r2, r5).
const OUTER_WEIGHTS: [i16; 3] = [
    CATMULL_TAPS[0] as i16,
    CATMULL_TAPS[1] as i16,
    CATMULL_TAPS[2] as i16,
];
// The vertical pass sums the rows in symmetric pairs, so the taps must be.
const _: () = {
    let mut tap = 0;
    while tap < TAPS {
        assert!(CATMULL_TAPS[tap] == CATMULL_TAPS[TAPS - 1 - tap]);
        tap += 1;
    }
};
/// Taps before the output's first input of the window: o reads 2o - 3 ..= 2o + 4.
const LEAD: usize = 3;

/// The NEON kernel, in exact i32 arithmetic. Rows with a full window: the
/// vertical sums (at most 280 x 255) per sample by `vmlal`, the 8-tap
/// horizontal sum at every sample from loads `channels` apart (|sum| < 2^25),
/// rounded and clamped once — `vqrshrun` by 16 is `(sum + 2^15) >> 16`
/// saturated to [0, 65535], the scalar `floor((2 n + 65536) / 131072)` —
/// then narrowed to u8 saturating, and the even pixels kept (gray rows wide
/// enough take the horizontal pass only at the outputs). Columns and rows
/// with clipped windows, and rows too short for a block, by the scalar
/// reference's arithmetic.
///
/// # Safety
/// `src` holds `width * height * channels` samples, `channels` 1, 3 or 4,
/// `width` and `height` even (the exact kernel's condition; with an odd
/// width the gray horizontal pass would leave output `ow - 2` unwritten).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn catmull_x2_neon(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    let (ow, oh, ws) = (width / 2, height / 2, width * channels);
    debug_assert!(src.len() >= width * height * channels);
    debug_assert!(width.is_multiple_of(2) && height.is_multiple_of(2));
    let mut out_buf = crate::cpu::unwritten_output::<u8>(ow * oh * channels);
    // SAFETY: u8 has no invalid values; every sample is written below before
    // the vector is returned.
    let out = unsafe {
        std::slice::from_raw_parts_mut(out_buf.as_mut_ptr().cast::<u8>(), ow * oh * channels)
    };
    let mut sums64 = vec![0i64; ws];
    // The vertical sums after LEAD pixels of zeros, with room for the offset
    // loads past the row (lanes that feed only the clipped edge pixels).
    let lead = LEAD * channels;
    let mut sums = vec![0i32; lead + ws + (TAPS - LEAD) * channels];
    // The rounded 8-tap results at every sample.
    let mut full = vec![0u8; ws];
    let full_window = |(_, weights): (usize, &[i32])| weights.len() == TAPS;
    // The columns whose windows are clipped (at the row's two ends).
    let clipped_columns: Vec<usize> = (0..ow)
        .filter(|&output| !full_window(catmull_window(output, width)))
        .collect();
    for y in 0..oh {
        let (r0, wv) = catmull_window(y, height);
        let dst = &mut out[y * ow * channels..(y + 1) * ow * channels];
        if !full_window((r0, wv)) || ws < BLOCK {
            // The scalar reference's sums, a row at a time (so the loop
            // vectorizes).
            sums64.fill(0);
            for (tap, &weight) in wv.iter().enumerate() {
                for (sum, &sample) in sums64.iter_mut().zip(&src[(r0 + tap) * ws..][..ws]) {
                    *sum += i64::from(weight) * i64::from(sample);
                }
            }
            let zv: i64 = wv.iter().map(|&weight| i64::from(weight)).sum();
            catmull_row_scalar(&sums64, channels, width, zv, 0..ow, dst);
            continue;
        }
        let rows: [&[u8]; TAPS] = std::array::from_fn(|tap| &src[(r0 + tap) * ws..][..ws]);
        let sums_ptr = sums[lead..].as_mut_ptr();
        cover_row(ws, BLOCK, |x| {
            // SAFETY: x + 8 <= ws: 8 samples of each row.
            let samples: [uint8x8_t; TAPS] =
                rows.map(|row| unsafe { vld1_u8(row.as_ptr().add(x)) });
            // The symmetric taps' row pairs, outermost first (each <= 510).
            let pair = |tap: usize| vaddl_u8(samples[tap], samples[TAPS - 1 - tap]);
            // 111 (r3 + r4) <= 56610 fits u16; the outer pairs' sum
            // 29 (r2 + r5) - 9 (r1 + r6) - 3 (r0 + r7) lies in [-6120, 14790]
            // and fits i16; they meet in i32.
            let center = vmulq_n_u16(pair(3), CENTER_WEIGHT);
            let mut outer = vmulq_n_s16(vreinterpretq_s16_u16(pair(2)), OUTER_WEIGHTS[2]);
            outer = vmlaq_n_s16(outer, vreinterpretq_s16_u16(pair(1)), OUTER_WEIGHTS[1]);
            outer = vmlaq_n_s16(outer, vreinterpretq_s16_u16(pair(0)), OUTER_WEIGHTS[0]);
            let low = vaddw_s16(
                vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(center))),
                vget_low_s16(outer),
            );
            let high = vaddw_high_s16(vreinterpretq_s32_u32(vmovl_high_u16(center)), outer);
            // SAFETY: x + 8 <= ws: inside the row of sums.
            unsafe {
                vst1q_s32(sums_ptr.add(x), low);
                vst1q_s32(sums_ptr.add(x + 4), high);
            }
        });
        let sums_ptr = sums[lead..].as_ptr();
        if channels == 1 && ow >= 4 + PHASED_OUTPUTS {
            // SAFETY: sums holds the row's `width` = 2 ow vertical sums from
            // sums_ptr, dst ow outputs.
            unsafe { horizontal_phased(sums_ptr, ow, dst) };
        } else {
            let full_ptr = full.as_mut_ptr();
            cover_row(ws, BLOCK, |x| {
                // The 8-tap sums of the 4 samples at x + `half`.
                let taps = |half: usize| -> int32x4_t {
                    let mut acc = vdupq_n_s32(0);
                    for (tap, &weight) in CATMULL_TAPS.iter().enumerate() {
                        let offset = (x + half) as isize
                            + (tap as isize - LEAD as isize) * channels as isize;
                        // SAFETY: x + half + 4 <= ws; the offsets reach LEAD
                        // pixels before the row (the zero lead) and TAPS -
                        // LEAD - 1 past it (the room after).
                        let window_sums = unsafe { vld1q_s32(sums_ptr.offset(offset)) };
                        acc = vmlaq_n_s32(acc, window_sums, weight);
                    }
                    acc
                };
                let rounded =
                    vcombine_u16(vqrshrun_n_s32::<16>(taps(0)), vqrshrun_n_s32::<16>(taps(4)));
                // SAFETY: x + 8 <= ws, the length of `full`.
                unsafe { vst1_u8(full_ptr.add(x), vqmovn_u16(rounded)) };
            });
            // SAFETY: full holds 2 * ow pixels of `channels` samples, dst ow.
            unsafe { keep_even_pixels(&full, channels, ow, dst) };
        }
        let zv: i64 = 256;
        for &output in &clipped_columns {
            let (x0, wh) = catmull_window(output, width);
            let denominator = zv * wh.iter().map(|&weight| i64::from(weight)).sum::<i64>();
            for ch in 0..channels {
                let numerator: i64 = wh
                    .iter()
                    .enumerate()
                    .map(|(tap, &weight)| {
                        i64::from(weight) * i64::from(sums[lead + (x0 + tap) * channels + ch])
                    })
                    .sum();
                dst[output * channels + ch] = divide_rounded_clamped(numerator, denominator);
            }
        }
    }
    // SAFETY: every row wrote all of its samples.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// Output pixels per step of `horizontal_phased`: two registers of i32 sums.
const PHASED_OUTPUTS: usize = 8;
/// i32 lanes per register.
const LANES: usize = 4;

/// A gray row's horizontal pass only at the outputs whose windows are full
/// (2 .. ow - 2): `vld2` splits the vertical sums into even and odd samples
/// `E`, `O`, so output o's taps 2o - 3 ..= 2o + 4 are `O[o - 2]`, `E[o - 1]`,
/// `O[o - 1]`, `E[o]`, `O[o]`, `E[o + 1]`, `O[o + 1]`, `E[o + 2]`, summed in
/// symmetric pairs, then weighted and rounded as `catmull_x2_neon`. The
/// clipped outputs are left to the caller.
///
/// # Safety
/// `sums` points at the row's `2 ow` vertical sums and `dst` holds `ow`
/// outputs, `ow >= 4 + PHASED_OUTPUTS`.
#[target_feature(enable = "neon")]
unsafe fn horizontal_phased(sums: *const i32, ow: usize, dst: &mut [u8]) {
    let dst_ptr = dst.as_mut_ptr();
    let mut rounded = [vdup_n_u16(0); 2];
    cover_row(ow - 4, PHASED_OUTPUTS, |index| {
        let first = 2 + index;
        for (half, rounded) in rounded.iter_mut().enumerate() {
            let output = first + half * LANES;
            // SAFETY: 2 <= output and output + 4 <= ow - 2, so the loads read
            // sums 2 (output - 2) .. 2 (output + 6) <= 2 ow.
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
        // SAFETY: first + 8 <= ow - 2: outputs of the row.
        unsafe { vst1_u8(dst_ptr.add(first), narrowed) };
    });
}

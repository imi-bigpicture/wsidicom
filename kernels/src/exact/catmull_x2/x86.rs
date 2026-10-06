//! The AVX2 and AVX-512 kernels of the exact x2 catmullrom reduction.
//!
//! # Safety
//! Every `unsafe fn` here requires the CPU features in its `#[target_feature]`
//! (its dispatch checks them at runtime) and the source length, even size and
//! channel count its caller documents, which the kernels read against
//! unchecked. Loads and stores are justified against those bounds and the
//! kernels' own buffers at their loop guards.

use super::{catmull_window, catmull_x2_scalar};
use crate::exact::x86::{cover_steps, prefetch_next_rows};

/// The AVX-512 catmullrom kernel for RGB (`catmull_x2_avx512_split` covers
/// 1 and 4 channels). Vertical: the symmetric
/// row pairs added in
/// u16, two pairs per `madd` with their weights into i32 sums, split by one
/// permute each into the even and the odd pixels' sums (which also undoes
/// the unpacks' lane order); a clipped row of an edge row's window reads a
/// zero row. Horizontal: 8 unit-stride loads at the output resolution from
/// the split sums, zero-padded around the row so the clipped edge columns
/// take the same step; `(sum + 2^15) >> 16` where both windows are whole,
/// else an exact per-lane division (float64: operands below 2^53, a
/// non-integer quotient off an integer by at least 1 / (2d), far above its
/// rounding error).
///
/// # Safety
/// AVX-512 F/BW/VL must be available; `src` holds `width * height *
/// channels` samples, `width` and `height` even.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(crate) unsafe fn catmull_x2_avx512(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    const LANES: usize = 16; // i32 lanes
    let (ow, oh, ws) = (width / 2, height / 2, width * channels);
    debug_assert!(src.len() >= width * height * channels);
    if channels > LANES || ow < LANES / channels + 2 {
        return catmull_x2_scalar(src, width, height, channels);
    }
    let mut out_buf = crate::cpu::unwritten_output::<u8>(ow * oh * channels);
    // SAFETY: u8 has no invalid values; every sample is written below.
    let out = unsafe {
        std::slice::from_raw_parts_mut(out_buf.as_mut_ptr().cast::<u8>(), ow * oh * channels)
    };
    // Output pixels per step (their samples fill most of 16 i32 lanes), from
    // twice as many input pixels (at most 32 u8 samples).
    let step = LANES / channels;
    let (step_lanes, in_lanes) = (step * channels, 2 * step * channels);
    // Sample `sample` of the 32 the unpacks spread: 128-bit lane L = sample
    // / 8 holds samples 8L..; the low unpack's madd gives samples 8L..8L+4 at
    // i32 positions 4L.., the high one's 8L+4.. (index + 16, the second
    // operand).
    let position = |sample: usize| {
        (if sample % 8 < 4 {
            4 * (sample / 8) + sample % 8
        } else {
            16 + 4 * (sample / 8) + sample % 8 - 4
        }) as i32
    };
    let mut even_idx = [0i32; LANES];
    let mut odd_idx = [0i32; LANES];
    for lane in 0..step_lanes {
        let sample = 2 * channels * (lane / channels) + lane % channels;
        even_idx[lane] = position(sample);
        odd_idx[lane] = position(sample + channels);
    }
    // SAFETY: each load reads its array's 16 i32.
    let (even_idx, odd_idx) = unsafe {
        (
            _mm512_loadu_si512(even_idx.as_ptr() as *const __m512i),
            _mm512_loadu_si512(odd_idx.as_ptr() as *const __m512i),
        )
    };
    let in_mask = u32::MAX >> (32 - in_lanes);
    let store_mask = u16::MAX >> (LANES - step_lanes);
    // The even and odd pixels' sums with two zero pixels before and room
    // after: the clipped edge taps read zeros.
    let pad = 2 * channels;
    let mut even = vec![0i32; pad + ow * channels + 2 * LANES + pad];
    let mut odd = vec![0i32; pad + ow * channels + 2 * LANES + pad];
    // SAFETY: both vectors hold more than `pad` values.
    let (ep, op) = unsafe { (even.as_mut_ptr().add(pad), odd.as_mut_ptr().add(pad)) };
    // (111, 29) for the inner pairs, (-9, -3) for the outer ones, as madd's
    // i16 pairs.
    let inner_w = _mm512_set1_epi32((29 << 16) | 111);
    let outer_w = _mm512_set1_epi32((i32::from(-3i32 as u16) << 16) | i32::from(-9i32 as u16));
    let (w111, w29, w9, w3) = (
        _mm512_set1_epi32(111),
        _mm512_set1_epi32(29),
        _mm512_set1_epi32(9),
        _mm512_set1_epi32(3),
    );
    let half = _mm512_set1_epi32(1 << 15);
    let zero = _mm512_setzero_si512();
    let zeros = vec![0u8; ws + 64];
    // Each output's column-window total (256 but for the clipped edges).
    let col_totals: Vec<i32> = (0..ow)
        .map(|output| catmull_window(output, width).1.iter().sum())
        .collect();
    for y in 0..oh {
        let (_, wv) = catmull_window(y, height);
        let zv: i32 = wv.iter().sum();
        let row = |row_index: isize| {
            if (0..height as isize).contains(&row_index) {
                // SAFETY: row_index < height, so the row starts within src
                // (width * height * channels samples, the caller's contract).
                unsafe { src.as_ptr().add(row_index as usize * ws) }
            } else {
                zeros.as_ptr()
            }
        };
        let first = 2 * y as isize - 3;
        let rows: [*const u8; 8] = std::array::from_fn(|tap| row(first + tap as isize));
        let vstep = |k0: usize| {
            let at = 2 * k0 * channels;
            // Plain loads where 32 bytes fit (the bytes past the step land in
            // lanes the stores leave out); masked at the row end.
            prefetch_next_rows(src, 2 * y + 5, ws, at);
            let fits = at + 32 <= ws;
            let ld = |tap: usize| {
                // SAFETY: the plain load is taken where its 32 bytes end
                // within the row's ws (`fits`); the masked one reads
                // in_lanes = 2 step channels bytes, ending within
                // 2 ow channels <= ws as k0 + step <= ow (cover_steps). Each
                // row is a src row (the caller's contract) or `zeros`
                // (ws + 64).
                let bytes = unsafe {
                    let tap_ptr = rows[tap].add(at);
                    if fits {
                        _mm256_loadu_si256(tap_ptr as *const __m256i)
                    } else {
                        _mm256_maskz_loadu_epi8(in_mask, tap_ptr as *const i8)
                    }
                };
                _mm512_cvtepu8_epi16(bytes)
            };
            let pair =
                |upper_tap: usize, lower_tap: usize| _mm512_add_epi16(ld(upper_tap), ld(lower_tap));
            let (inner, mid) = (pair(3, 4), pair(2, 5));
            let (outer, far) = (pair(1, 6), pair(0, 7));
            let madd2 = |low_words: __m512i, high_words: __m512i, weights: __m512i| {
                (
                    _mm512_madd_epi16(_mm512_unpacklo_epi16(low_words, high_words), weights),
                    _mm512_madd_epi16(_mm512_unpackhi_epi16(low_words, high_words), weights),
                )
            };
            let (in_lo, in_hi) = madd2(inner, mid, inner_w);
            let (out_lo, out_hi) = madd2(outer, far, outer_w);
            let (lo, hi) = (
                _mm512_add_epi32(in_lo, out_lo),
                _mm512_add_epi32(in_hi, out_hi),
            );
            // Masked: the lanes past the step would land in the zero padding
            // the clipped edge taps read.
            // SAFETY: k0 + step <= ow (cover_steps): the step channels values
            // stored from k0 channels lie within even's and odd's ow channels.
            unsafe {
                _mm512_mask_storeu_epi32(
                    ep.add(k0 * channels),
                    store_mask,
                    _mm512_permutex2var_epi32(lo, even_idx, hi),
                );
                _mm512_mask_storeu_epi32(
                    op.add(k0 * channels),
                    store_mask,
                    _mm512_permutex2var_epi32(lo, odd_idx, hi),
                );
            }
        };
        cover_steps(ow, step, &vstep);
        // SAFETY: y < oh: the row starts within the ow * oh * channels outputs.
        let dp = unsafe { out.as_mut_ptr().add(y * ow * channels) };
        let hstep = |o0: usize| {
            let ld = |parity: *const i32, offset: isize| {
                // SAFETY: o0 + step <= ow (cover_steps) and offset is -2..=2,
                // so the load's 16 values lie within even's and odd's two
                // zero pixels before, ow channels values and 2 LANES + pad
                // after.
                unsafe {
                    _mm512_loadu_si512(
                        parity.offset((o0 as isize + offset) * channels as isize) as *const __m512i
                    )
                }
            };
            let (ep, op) = (ep as *const i32, op as *const i32);
            let pair = |left: __m512i, right: __m512i, weight: __m512i| {
                _mm512_mullo_epi32(_mm512_add_epi32(left, right), weight)
            };
            let positive = _mm512_add_epi32(
                pair(ld(ep, 0), ld(op, 0), w111),
                pair(ld(op, -1), ld(ep, 1), w29),
            );
            let negative = _mm512_add_epi32(
                pair(ld(ep, -1), ld(op, 1), w9),
                pair(ld(op, -2), ld(ep, 2), w3),
            );
            let sums = _mm512_sub_epi32(positive, negative);
            let whole = zv == 256 && o0 >= 2 && o0 + step + 2 <= ow;
            let quotient = if whole {
                _mm512_srai_epi32::<16>(_mm512_add_epi32(sums, half))
            } else {
                // floor((2 sum + d) / (2 d)) per lane, d = zv * the column total.
                let denominators: [f64; LANES] = std::array::from_fn(|lane| {
                    f64::from(zv * col_totals[(o0 + lane / channels).min(ow - 1)])
                });
                let divide = |sums_half: __m256i, denominators: &[f64]| {
                    // SAFETY: both calls pass 8 values, which the load reads.
                    let divisors = unsafe { _mm512_loadu_pd(denominators.as_ptr()) };
                    let x = _mm512_cvtepi32_pd(sums_half);
                    let quotient = _mm512_div_pd(
                        _mm512_add_pd(_mm512_add_pd(x, x), divisors),
                        _mm512_add_pd(divisors, divisors),
                    );
                    _mm512_cvtpd_epi32(_mm512_roundscale_pd::<
                        { _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC },
                    >(quotient))
                };
                let lo = divide(_mm512_castsi512_si256(sums), &denominators[..8]);
                let hi = divide(_mm512_extracti64x4_epi64::<1>(sums), &denominators[8..]);
                _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
            };
            // SAFETY: o0 + step <= ow (cover_steps): the masked store's
            // step channels bytes from o0 channels lie within the row.
            unsafe {
                _mm_mask_storeu_epi8(
                    dp.add(o0 * channels) as *mut i8,
                    store_mask,
                    _mm512_cvtusepi32_epi8(_mm512_max_epi32(quotient, zero)),
                );
            }
        };
        cover_steps(ow, step, &hstep);
    }
    // SAFETY: every row's steps covered all of its outputs.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// The AVX-512 catmullrom kernel for 1 and 4 channels: the AVX2 kernel
/// (`catmull_x2_avx2`) at twice the width, 64 input samples per vertical
/// step. The even and odd pixels' sums are gathered from the four vertical
/// result registers by two `permutex2var` each (after the AVX2 kernel's
/// shuffle for 1 channel); the index tables
/// come from running the same steps on registers of sample labels.
///
/// # Safety
/// AVX-512 F/BW/VL and FMA must be available; `src` holds `width * height *
/// channels` samples, `width` and `height` even; `channels` is 1 or 4.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,fma")]
pub(crate) unsafe fn catmull_x2_avx512_split(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    const OFFSET: i32 = 24010;
    const LANES: usize = 16; // i32 lanes
    debug_assert!(matches!(channels, 1 | 4));
    let (ow, oh, ws) = (width / 2, height / 2, width * channels);
    let row_samples = ow * channels;
    debug_assert!(src.len() >= width * height * channels);
    // Outputs per vertical step: 64 input samples, 32 of each parity.
    let vstep_outputs = 32 / channels;
    if ow < vstep_outputs || row_samples < LANES + 4 * channels {
        return catmull_x2_scalar(src, width, height, channels);
    }
    let mut out_buf = crate::cpu::unwritten_output::<u8>(ow * oh * channels);
    // SAFETY: u8 has no invalid values; every sample is written below.
    let out = unsafe {
        std::slice::from_raw_parts_mut(out_buf.as_mut_ptr().cast::<u8>(), ow * oh * channels)
    };
    // The even and odd pixels' sums with two zero pixels each side: the
    // clipped edge taps read zeros.
    let pad = 2 * channels;
    let mut even = vec![0f32; pad + row_samples + 2 * LANES + pad];
    let mut odd = vec![0f32; pad + row_samples + 2 * LANES + pad];
    // SAFETY: both vectors hold more than `pad` values.
    let (ep, op) = unsafe { (even.as_mut_ptr().add(pad), odd.as_mut_ptr().add(pad)) };
    let zeros = vec![0u8; ws + 64];
    // As in the AVX2 split kernel: pmaddubsw pairs, the near pairs' sum less
    // OFFSET, the horizontal pass in exact f32.
    let near_w = _mm512_set1_epi16((i16::from(-9i8 as u8) << 8) | 111);
    let far_w = _mm512_set1_epi16((i16::from(-3i8 as u8) << 8) | 29);
    let offset = _mm512_set1_epi16(OFFSET as i16);
    let ones = _mm512_set1_epi16(1);
    let (w111, w29, w9, w3) = (
        _mm512_set1_ps(111.0),
        _mm512_set1_ps(29.0),
        _mm512_set1_ps(-9.0),
        _mm512_set1_ps(-3.0),
    );
    let half = _mm512_set1_epi32((1 << 15) + OFFSET * 256);
    let zero = _mm512_setzero_si512();
    // The even (odd) pixels of the four vertical results as two registers.
    let combine = |results: [__m512i; 4], odd: bool| -> (__m512i, __m512i) {
        let shuffle = |x: __m512i, y: __m512i| {
            let (x, y) = (_mm512_castsi512_ps(x), _mm512_castsi512_ps(y));
            _mm512_castps_si512(if odd {
                _mm512_shuffle_ps::<0b11_01_11_01>(x, y)
            } else {
                _mm512_shuffle_ps::<0b10_00_10_00>(x, y)
            })
        };
        match channels {
            1 => (
                shuffle(results[0], results[1]),
                shuffle(results[2], results[3]),
            ),
            _ => {
                if odd {
                    (results[1], results[3])
                } else {
                    (results[0], results[2])
                }
            }
        }
    };
    // Result register `register` holds samples 16L + 4 register .. 16L +
    // 4 register + 4 in 128-bit lane L. Each label is the sample's index in its parity's run of 32,
    // plus 64 if odd.
    let label = |sample: usize| -> i32 {
        let (pixel, ch) = (sample / channels, sample % channels);
        ((pixel / 2 * channels + ch) + if pixel % 2 == 1 { 64 } else { 0 }) as i32
    };
    let labels: [__m512i; 4] = std::array::from_fn(|register| {
        let register_labels: [i32; LANES] =
            std::array::from_fn(|lane| label(16 * (lane / 4) + 4 * register + lane % 4));
        // SAFETY: the load reads the 16 values of `register_labels`.
        unsafe { _mm512_loadu_si512(register_labels.as_ptr() as *const __m512i) }
    });
    // The two permutex2var index tables per parity: lane `lane` of table
    // `table` takes the parity's sample 16 table + lane.
    let tables = |odd: bool| -> [__m512i; 2] {
        let (first_source, second_source) = combine(labels, odd);
        let mut held = [0i32; 2 * LANES];
        // SAFETY: the stores write the two halves of `held`, LANES values each.
        unsafe {
            _mm512_storeu_si512(held.as_mut_ptr() as *mut __m512i, first_source);
            _mm512_storeu_si512(held.as_mut_ptr().add(LANES) as *mut __m512i, second_source);
        }
        let want = if odd { 64 } else { 0 };
        std::array::from_fn(|table| {
            let idx: [i32; LANES] = std::array::from_fn(|lane| {
                let sample = (16 * table + lane) as i32 + want;
                #[expect(
                    clippy::expect_used,
                    reason = "the two sources hold every sample of the parity by construction"
                )]
                let position = held
                    .iter()
                    .position(|&held_label| held_label == sample)
                    .expect("every sample is held");
                position as i32
            });
            // SAFETY: the load reads the 16 values of `idx`.
            unsafe { _mm512_loadu_si512(idx.as_ptr() as *const __m512i) }
        })
    };
    let (even_tables, odd_tables) = (tables(false), tables(true));
    let col_totals: Vec<i32> = (0..ow)
        .map(|output| catmull_window(output, width).1.iter().sum())
        .collect();
    for y in 0..oh {
        let (_, wv) = catmull_window(y, height);
        let zv: i32 = wv.iter().sum();
        let row = |row_index: isize| {
            if (0..height as isize).contains(&row_index) {
                // SAFETY: row_index < height, so the row starts within src
                // (width * height * channels samples, the caller's contract).
                unsafe { src.as_ptr().add(row_index as usize * ws) }
            } else {
                zeros.as_ptr()
            }
        };
        let first = 2 * y as isize - 3;
        let rows: [*const u8; 8] = std::array::from_fn(|tap| row(first + tap as isize));
        let vstep = |k0: usize| {
            prefetch_next_rows(src, 2 * y + 5, ws, 2 * k0 * channels);
            let loaded: [__m512i; 8] = std::array::from_fn(|tap| {
                // SAFETY: k0 + 32 / channels <= ow (cover_steps): the load's 64
                // bytes from 2 k0 channels end within 2 ow channels <= ws of a
                // src row (the caller's contract) or `zeros` (ws + 64).
                unsafe { _mm512_loadu_si512(rows[tap].add(2 * k0 * channels) as *const __m512i) }
            });
            let near = |lo: bool| {
                let pair = |low_tap: usize, high_tap: usize| {
                    if lo {
                        _mm512_unpacklo_epi8(loaded[low_tap], loaded[high_tap])
                    } else {
                        _mm512_unpackhi_epi8(loaded[low_tap], loaded[high_tap])
                    }
                };
                let near_sums = _mm512_add_epi16(
                    _mm512_maddubs_epi16(pair(3, 1), near_w),
                    _mm512_maddubs_epi16(pair(4, 6), near_w),
                );
                let far_sums = _mm512_add_epi16(
                    _mm512_maddubs_epi16(pair(2, 0), far_w),
                    _mm512_maddubs_epi16(pair(5, 7), far_w),
                );
                let near_sums = _mm512_sub_epi16(near_sums, offset);
                (
                    _mm512_madd_epi16(_mm512_unpacklo_epi16(near_sums, far_sums), ones),
                    _mm512_madd_epi16(_mm512_unpackhi_epi16(near_sums, far_sums), ones),
                )
            };
            let ((q0, q1), (q2, q3)) = (near(true), near(false));
            let results =
                [q0, q1, q2, q3].map(|result| _mm512_castps_si512(_mm512_cvtepi32_ps(result)));
            for (parity, odd, [lo, hi]) in [(ep, false, even_tables), (op, true, odd_tables)] {
                let (first_source, second_source) = combine(results, odd);
                // SAFETY: k0 + 32 / channels <= ow (cover_steps): the two
                // stores' 2 LANES = 32 values from k0 channels end within the
                // parity's row_samples.
                unsafe {
                    let dst_ptr = parity.add(k0 * channels) as *mut i32;
                    _mm512_storeu_si512(
                        dst_ptr as *mut __m512i,
                        _mm512_permutex2var_epi32(first_source, lo, second_source),
                    );
                    _mm512_storeu_si512(
                        dst_ptr.add(LANES) as *mut __m512i,
                        _mm512_permutex2var_epi32(first_source, hi, second_source),
                    );
                }
            }
        };
        cover_steps(ow, vstep_outputs, &vstep);
        // SAFETY: y < oh: the row starts within the ow * oh * channels outputs.
        let dp = unsafe { out.as_mut_ptr().add(y * row_samples) };
        let hstep = |s0: usize| {
            let ld = |parity: *const f32, offset: isize| {
                // SAFETY: s0 + LANES <= row_samples (cover_steps) and offset
                // is -2..=2, so the load's 16 values lie within the parity's
                // two zero pixels before, row_samples and 2 LANES + pad after.
                unsafe { _mm512_loadu_ps(parity.offset(s0 as isize + offset * channels as isize)) }
            };
            let (ep, op) = (ep as *const f32, op as *const f32);
            let pair = |left: __m512, right: __m512| _mm512_add_ps(left, right);
            let near = _mm512_fmadd_ps(
                pair(ld(op, -1), ld(ep, 1)),
                w29,
                _mm512_mul_ps(pair(ld(ep, 0), ld(op, 0)), w111),
            );
            let far = _mm512_fmadd_ps(
                pair(ld(op, -2), ld(ep, 2)),
                w3,
                _mm512_mul_ps(pair(ld(ep, -1), ld(op, 1)), w9),
            );
            let sums = _mm512_cvtps_epi32(_mm512_add_ps(near, far));
            let whole = zv == 256 && s0 >= 2 * channels && s0 + LANES + 2 * channels <= row_samples;
            let quotient = if whole {
                _mm512_srai_epi32::<16>(_mm512_add_epi32(sums, half))
            } else {
                // floor((2 sum + d) / (2 d)) per lane, d = zv * the column
                // total, the sum's offset (OFFSET per weight unit) added back.
                let zh = |lane: usize| col_totals[((s0 + lane) / channels).min(ow - 1)];
                let denominators: [f64; LANES] =
                    std::array::from_fn(|lane| f64::from(zv * zh(lane)));
                let back: [f64; LANES] = std::array::from_fn(|lane| f64::from(OFFSET * zh(lane)));
                let divide = |sums_half: __m256i, denominators: &[f64], back: &[f64]| {
                    // SAFETY: both calls pass 8 values in each slice, which
                    // the loads read.
                    let (divisors, back) = unsafe {
                        (
                            _mm512_loadu_pd(denominators.as_ptr()),
                            _mm512_loadu_pd(back.as_ptr()),
                        )
                    };
                    let x = _mm512_add_pd(_mm512_cvtepi32_pd(sums_half), back);
                    let quotient = _mm512_div_pd(
                        _mm512_add_pd(_mm512_add_pd(x, x), divisors),
                        _mm512_add_pd(divisors, divisors),
                    );
                    _mm512_cvtpd_epi32(_mm512_roundscale_pd::<
                        { _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC },
                    >(quotient))
                };
                let lo = divide(_mm512_castsi512_si256(sums), &denominators[..8], &back[..8]);
                let hi = divide(
                    _mm512_extracti64x4_epi64::<1>(sums),
                    &denominators[8..],
                    &back[8..],
                );
                _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
            };
            // SAFETY: s0 + LANES <= row_samples (cover_steps): the store's 16
            // bytes lie within the row.
            unsafe {
                _mm_storeu_si128(
                    dp.add(s0) as *mut __m128i,
                    _mm512_cvtusepi32_epi8(_mm512_max_epi32(quotient, zero)),
                );
            }
        };
        cover_steps(row_samples, LANES, &hstep);
    }
    // SAFETY: every row's steps covered all of its outputs.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// The AVX2 catmullrom kernel. Vertical: u8 row pairs
/// weighted by `pmaddubsw` ((111, -9) and (29, -3) per pair), the two near
/// pairs' sum less an offset to fit i16, then `madd` by ones adds the far
/// pairs' sum into i32; 32 input samples per step. The unpacks leave samples
/// 0..4 | 16..20, 4..8 | 20..24, 8..12 | 24..28 and 12..16 | 28..32 in four
/// registers, which is already the even and odd pixels for 4 channels and
/// one shuffle away for 1; RGB rows are shuffled to
/// even | odd pixels per 4-pixel group as they load, and the 6-sample halves
/// stored overlapping. Horizontal: 8 samples per step from the zero-padded
/// even and odd sums, 8 unit-stride loads, in f32 (exact, see below).
///
/// # Safety
/// AVX2 and FMA must be available; `src` holds `width * height * channels`
/// samples, `width` and `height` even; `channels` is 1, 3 or 4.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn catmull_x2_avx2(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    // The near pairs' two sums span -4590..=56610: less the offset they fit
    // i16, and madd by (1, 1) adds the far pairs' sums into i32. The vertical
    // sums are stored less the offset, which the rounding adds back.
    const OFFSET: i32 = 24010;
    const LANES: usize = 8; // i32 lanes
    debug_assert!([1, 3, 4].contains(&channels));
    let (ow, oh, ws) = (width / 2, height / 2, width * channels);
    let row_samples = ow * channels;
    debug_assert!(src.len() >= width * height * channels);
    // Outputs per vertical step: 32 input samples (8 pixels for RGB).
    let vstep_outputs = if channels == 3 {
        4
    } else {
        2 * LANES / channels
    };
    if ow < vstep_outputs || row_samples < LANES + 4 * channels {
        return catmull_x2_scalar(src, width, height, channels);
    }
    let mut out_buf = crate::cpu::unwritten_output::<u8>(ow * oh * channels);
    // SAFETY: u8 has no invalid values; every sample is written below.
    let out = unsafe {
        std::slice::from_raw_parts_mut(out_buf.as_mut_ptr().cast::<u8>(), ow * oh * channels)
    };
    // The even and odd pixels' sums with two zero pixels each side: the
    // clipped edge taps read zeros.
    let pad = 2 * channels;
    let mut even = vec![0i32; pad + row_samples + 2 * LANES + pad];
    let mut odd = vec![0i32; pad + row_samples + 2 * LANES + pad];
    // SAFETY: both vectors hold more than `pad` values.
    let (ep, op) = unsafe { (even.as_mut_ptr().add(pad), odd.as_mut_ptr().add(pad)) };
    let zeros = vec![0u8; ws + 32];
    // pmaddubsw weight pairs: (111, -9) for rows (3, 1) and (4, 6), (29, -3)
    // for (2, 0) and (5, 7); each pair's sum stays within i16 (at most
    // 255 * 111), so nothing saturates.
    let near_w = _mm256_set1_epi16((i16::from(-9i8 as u8) << 8) | 111);
    let far_w = _mm256_set1_epi16((i16::from(-3i8 as u8) << 8) | 29);
    let offset = _mm256_set1_epi16(OFFSET as i16);
    let ones = _mm256_set1_epi16(1);
    // The horizontal pass runs in f32, exactly: the stored sums are within
    // +-47390, so every partial sum stays below 304 * 47390 < 2^24.
    let (w111, w29, w9, w3) = (
        _mm256_set1_ps(111.0),
        _mm256_set1_ps(29.0),
        _mm256_set1_ps(-9.0),
        _mm256_set1_ps(-3.0),
    );
    // The rounding half, plus the offset over a whole window (256 weight units).
    let half = _mm256_set1_epi32((1 << 15) + OFFSET * 256);
    // RGB: 4 pixels' 12 bytes per lane (dwords 0..3 and 3..6), shuffled to the
    // even pixels' 6 bytes | the odd pixels' 6 bytes, 2 zero bytes after each.
    let rgb_bytes = _mm_setr_epi32(-1, -1, -1, 0);
    let rgb_halves = _mm256_setr_epi32(0, 1, 2, 0, 3, 4, 5, 0);
    let even_odd = _mm_setr_epi8(0, 1, 2, 6, 7, 8, -1, -1, 3, 4, 5, 9, 10, 11, -1, -1);
    let even_odd_ymm = _mm256_set_m128i(even_odd, even_odd);
    // The low byte of each lane's two dwords: q0..q3 | q4..q7 as 8 bytes.
    let low_dwords = _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0);
    let col_totals: Vec<i32> = (0..ow)
        .map(|output| catmull_window(output, width).1.iter().sum())
        .collect();
    for y in 0..oh {
        let (_, wv) = catmull_window(y, height);
        let zv: i32 = wv.iter().sum();
        let row = |row_index: isize| {
            if (0..height as isize).contains(&row_index) {
                // SAFETY: row_index < height, so the row starts within src
                // (width * height * channels samples, the caller's contract).
                unsafe { src.as_ptr().add(row_index as usize * ws) }
            } else {
                zeros.as_ptr()
            }
        };
        let first = 2 * y as isize - 3;
        let rows: [*const u8; 8] = std::array::from_fn(|tap| row(first + tap as isize));
        // 32 bytes from pixel p0 of tap row `tap` (24 for RGB, then shuffled).
        let load = |tap: usize, p0: usize| {
            // SAFETY: p0 = 2 k0 with k0 + vstep_outputs <= ow (cover_steps),
            // so the plain load's 32 bytes from p0 channels end within
            // 2 ow channels <= ws (vstep_outputs channels = 16 but for RGB),
            // the RGB load is taken where its 32 bytes end within ws (the
            // guard), and the masked RGB loads read the 12 bytes of pixels p0
            // and p0 + 4, ending within 3 (p0 + 8) <= ws; each row is a src
            // row (the caller's contract) or `zeros` (ws + 32).
            unsafe {
                if channels != 3 {
                    _mm256_loadu_si256(rows[tap].add(p0 * channels) as *const __m256i)
                } else if 3 * p0 + 32 <= ws {
                    let bytes = _mm256_loadu_si256(rows[tap].add(3 * p0) as *const __m256i);
                    _mm256_shuffle_epi8(
                        _mm256_permutevar8x32_epi32(bytes, rgb_halves),
                        even_odd_ymm,
                    )
                } else {
                    let group = |pixel: usize| {
                        _mm_shuffle_epi8(
                            _mm_maskload_epi32(rows[tap].add(3 * pixel) as *const i32, rgb_bytes),
                            even_odd,
                        )
                    };
                    _mm256_set_m128i(group(p0 + 4), group(p0))
                }
            }
        };
        let vstep = |k0: usize| {
            prefetch_next_rows(src, 2 * y + 5, ws, 2 * k0 * channels);
            let loaded: [__m256i; 8] = std::array::from_fn(|tap| load(tap, 2 * k0));
            // Per 128-bit lane, the low unpacks hold samples 0..8, the high
            // ones 8..16 (of 16 per lane).
            let near = |lo: bool| {
                let pair = |low_tap: usize, high_tap: usize| {
                    if lo {
                        _mm256_unpacklo_epi8(loaded[low_tap], loaded[high_tap])
                    } else {
                        _mm256_unpackhi_epi8(loaded[low_tap], loaded[high_tap])
                    }
                };
                let near_sums = _mm256_add_epi16(
                    _mm256_maddubs_epi16(pair(3, 1), near_w),
                    _mm256_maddubs_epi16(pair(4, 6), near_w),
                );
                let far_sums = _mm256_add_epi16(
                    _mm256_maddubs_epi16(pair(2, 0), far_w),
                    _mm256_maddubs_epi16(pair(5, 7), far_w),
                );
                let near_sums = _mm256_sub_epi16(near_sums, offset);
                (
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(near_sums, far_sums), ones),
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(near_sums, far_sums), ones),
                )
            };
            // q0: samples 0..4 | 16..20, q1: 4..8 | 20..24, q2: 8..12 | 24..28,
            // q3: 12..16 | 28..32.
            let ((q0, q1), (q2, q3)) = (near(true), near(false));
            let [q0, q1, q2, q3] =
                [q0, q1, q2, q3].map(|result| _mm256_castps_si256(_mm256_cvtepi32_ps(result)));
            // The even (odd) pixels of q0, q1 and of q2, q3, lane by lane.
            let split = |x: __m256i, y: __m256i, odd: bool| {
                if odd {
                    _mm256_castps_si256(_mm256_shuffle_ps::<0b11_01_11_01>(
                        _mm256_castsi256_ps(x),
                        _mm256_castsi256_ps(y),
                    ))
                } else {
                    _mm256_castps_si256(_mm256_shuffle_ps::<0b10_00_10_00>(
                        _mm256_castsi256_ps(x),
                        _mm256_castsi256_ps(y),
                    ))
                }
            };
            for (parity, odd) in [(ep, false), (op, true)] {
                let (first_source, second_source) = match channels {
                    4 => (if odd { q1 } else { q0 }, if odd { q3 } else { q2 }),
                    3 => (if odd { q2 } else { q0 }, if odd { q3 } else { q1 }),
                    _ => (split(q0, q1, odd), split(q2, q3, odd)),
                };
                // RGB: 6 samples and 2 junk lanes per store, the next store
                // overwriting the junk.
                let lane_samples = if channels == 3 { 6 } else { LANES };
                // SAFETY: k0 + vstep_outputs <= ow (cover_steps): the stores
                // end at k0 channels + lane_samples + LANES, within the
                // parity's row_samples (16 values per step but for RGB) or,
                // for RGB, row_samples + 2 of its 2 LANES + pad after.
                unsafe {
                    _mm256_storeu_si256(
                        parity.add(k0 * channels) as *mut __m256i,
                        _mm256_permute2x128_si256::<0x20>(first_source, second_source),
                    );
                    _mm256_storeu_si256(
                        parity.add(k0 * channels + lane_samples) as *mut __m256i,
                        _mm256_permute2x128_si256::<0x31>(first_source, second_source),
                    );
                }
            }
        };
        cover_steps(ow, vstep_outputs, &vstep);
        if channels == 3 {
            // The last step's junk lanes, back to the zero padding.
            // SAFETY: the stores' LANES values from row_samples lie within the
            // 2 LANES + pad after even's and odd's row.
            unsafe {
                _mm256_storeu_si256(ep.add(row_samples) as *mut __m256i, _mm256_setzero_si256());
                _mm256_storeu_si256(op.add(row_samples) as *mut __m256i, _mm256_setzero_si256());
            }
        }
        // SAFETY: y < oh: the row starts within the ow * oh * channels outputs.
        let dp = unsafe { out.as_mut_ptr().add(y * row_samples) };
        let hstep = |s0: usize| {
            let ld = |parity: *const i32, offset: isize| {
                // SAFETY: s0 + LANES <= row_samples (cover_steps) and offset
                // is -2..=2, so the load's 8 values lie within the parity's
                // two zero pixels before, row_samples and 2 LANES + pad after.
                unsafe {
                    _mm256_loadu_ps(
                        parity.offset(s0 as isize + offset * channels as isize) as *const f32
                    )
                }
            };
            let (ep, op) = (ep as *const i32, op as *const i32);
            let pair = |left: __m256, right: __m256| _mm256_add_ps(left, right);
            let near = _mm256_fmadd_ps(
                pair(ld(op, -1), ld(ep, 1)),
                w29,
                _mm256_mul_ps(pair(ld(ep, 0), ld(op, 0)), w111),
            );
            let far = _mm256_fmadd_ps(
                pair(ld(op, -2), ld(ep, 2)),
                w3,
                _mm256_mul_ps(pair(ld(ep, -1), ld(op, 1)), w9),
            );
            let sums = _mm256_cvtps_epi32(_mm256_add_ps(near, far));
            let whole = zv == 256 && s0 >= 2 * channels && s0 + LANES + 2 * channels <= row_samples;
            let quotient = if whole {
                _mm256_srai_epi32::<16>(_mm256_add_epi32(sums, half))
            } else {
                // floor((2 sum + d) / (2 d)) per lane, d = zv * the column
                // total, the sum's offset (OFFSET per weight unit) added back.
                let zh = |lane: usize| col_totals[((s0 + lane) / channels).min(ow - 1)];
                let denominators: [f64; LANES] =
                    std::array::from_fn(|lane| f64::from(zv * zh(lane)));
                let back: [f64; LANES] = std::array::from_fn(|lane| f64::from(OFFSET * zh(lane)));
                let divide = |sums_half: __m128i, denominators: &[f64], back: &[f64]| {
                    // SAFETY: both calls pass 4 values in each slice, which
                    // the loads read.
                    let (divisors, back) = unsafe {
                        (
                            _mm256_loadu_pd(denominators.as_ptr()),
                            _mm256_loadu_pd(back.as_ptr()),
                        )
                    };
                    let x = _mm256_add_pd(_mm256_cvtepi32_pd(sums_half), back);
                    let quotient = _mm256_div_pd(
                        _mm256_add_pd(_mm256_add_pd(x, x), divisors),
                        _mm256_add_pd(divisors, divisors),
                    );
                    _mm256_cvtpd_epi32(_mm256_floor_pd(quotient))
                };
                let lo = divide(_mm256_castsi256_si128(sums), &denominators[..4], &back[..4]);
                let hi = divide(
                    _mm256_extracti128_si256::<1>(sums),
                    &denominators[4..],
                    &back[4..],
                );
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
            };
            // i32 -> i16 (saturating) -> u8 (clamped to 0..=255): q0..3 in the
            // low lane's first dword, q4..7 in the high lane's.
            let words = _mm256_packs_epi32(quotient, quotient);
            let bytes = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(words, words), low_dwords);
            // SAFETY: s0 + LANES <= row_samples (cover_steps): the store's 8
            // bytes lie within the row.
            unsafe { _mm_storel_epi64(dp.add(s0) as *mut __m128i, _mm256_castsi256_si128(bytes)) };
        };
        cover_steps(row_samples, LANES, &hstep);
    }
    // SAFETY: every row's steps covered all of its outputs.
    unsafe { crate::cpu::written_output(out_buf) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::catmull_x2::catmull_x2_scalar;
    use crate::exact::catmull_x2::tests::{HEIGHTS, WIDTHS};
    use crate::exact::tests::{assert_equals_reference, size_grid};
    use crate::testing::CHANNELS;

    fn has_avx512() -> bool {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
    }

    #[test]
    fn avx2_equals_scalar() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let kernel = |src: &[u8], width, height, channels| {
            // SAFETY: AVX2 and FMA are available; src holds width * height *
            // channels samples, width and height even (WIDTHS and HEIGHTS),
            // channels 1, 3 or 4.
            unsafe { catmull_x2_avx2(src, width, height, channels) }
        };
        assert_equals_reference(
            size_grid(&WIDTHS, &HEIGHTS),
            &CHANNELS,
            u8::MAX,
            &kernel,
            catmull_x2_scalar,
        );
    }

    #[test]
    fn avx512_equals_scalar() {
        if !has_avx512() {
            return;
        }
        let kernel = |src: &[u8], width, height, channels| {
            // SAFETY: AVX-512 F/BW/VL are available; src holds width * height
            // * channels samples, width and height even (WIDTHS and HEIGHTS).
            unsafe { catmull_x2_avx512(src, width, height, channels) }
        };
        assert_equals_reference(
            size_grid(&WIDTHS, &HEIGHTS),
            &CHANNELS,
            u8::MAX,
            &kernel,
            catmull_x2_scalar,
        );
    }

    #[test]
    fn avx512_split_equals_scalar() {
        if !(has_avx512() && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let kernel = |src: &[u8], width, height, channels| {
            // SAFETY: AVX-512 F/BW/VL and FMA are available; src holds width
            // * height * channels samples, width and height even (WIDTHS and
            // HEIGHTS), channels 1 or 4.
            unsafe { catmull_x2_avx512_split(src, width, height, channels) }
        };
        assert_equals_reference(
            size_grid(&WIDTHS, &HEIGHTS),
            &[1, 4],
            u8::MAX,
            &kernel,
            catmull_x2_scalar,
        );
    }
}

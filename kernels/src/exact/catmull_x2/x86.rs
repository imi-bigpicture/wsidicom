//! The AVX2 kernel of the exact x2 catmullrom reduction.
//!
//! # Safety
//! The kernel requires AVX2, FMA and the bounds its doc states; loads and
//! stores are justified against those and its own buffers at their loop
//! guards.

use super::{catmull_window, catmull_x2_scalar};
use crate::exact::x86::cover_steps;
use crate::x86::prefetch_rows_after_window;

/// Vertical, 32 input samples per step: u8 row pairs weighted by `pmaddubsw`
/// ((111, -9) and (29, -3) per pair), the two near pairs' sum less an offset
/// to fit i16, then `madd` by ones adds the far pairs' sum into i32. The
/// unpacks leave the samples in four registers `q0..q3`, one shuffle away from
/// the even and odd pixels for gray; RGB rows are shuffled to even | odd
/// pixels per 4-pixel group as they load, and the 6-sample halves stored
/// overlapping. Horizontal, 8 samples per step: unit-stride loads of the
/// zero-padded even and odd sums, in exact f32.
///
/// # Safety
/// AVX2 and FMA must be available; `src` holds `width * height * channels`
/// samples, `width` and `height` even; `channels` is 1 or 3.
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
    const LANES: usize = 8;
    debug_assert!([1, 3].contains(&channels));
    let (out_width, out_height, row_samples) = (width / 2, height / 2, width * channels);
    let out_row_samples = out_width * channels;
    debug_assert!(src.len() >= width * height * channels);
    let vertical_step_outputs = if channels == 3 {
        4
    } else {
        2 * LANES / channels
    };
    if out_width < vertical_step_outputs || out_row_samples < LANES + 4 * channels {
        return catmull_x2_scalar(src, width, height, channels);
    }
    let mut out_buf = crate::cpu::unwritten_output::<u8>(out_width * out_height * channels);
    // SAFETY: u8 has no invalid values; every sample is written below.
    let out = unsafe {
        std::slice::from_raw_parts_mut(
            out_buf.as_mut_ptr().cast::<u8>(),
            out_width * out_height * channels,
        )
    };
    // Two zero pixels each side: the clipped edge taps read zeros.
    let pad = 2 * channels;
    let mut even = vec![0i32; pad + out_row_samples + 2 * LANES + pad];
    let mut odd = vec![0i32; pad + out_row_samples + 2 * LANES + pad];
    // SAFETY: both vectors hold more than `pad` values.
    let (even_ptr, odd_ptr) = unsafe { (even.as_mut_ptr().add(pad), odd.as_mut_ptr().add(pad)) };
    let zeros = vec![0u8; row_samples + 32];
    // pmaddubsw weight pairs: (111, -9) for rows (3, 1) and (4, 6), (29, -3)
    // for (2, 0) and (5, 7); each pair's sum stays within i16 (at most
    // 255 * 111), so nothing saturates.
    let near_weights = _mm256_set1_epi16((i16::from(-9i8 as u8) << 8) | 111);
    let far_weights = _mm256_set1_epi16((i16::from(-3i8 as u8) << 8) | 29);
    let offset = _mm256_set1_epi16(OFFSET as i16);
    let ones = _mm256_set1_epi16(1);
    // The horizontal pass runs in f32, exactly: the stored sums are within
    // +-47390, so every partial sum stays below 304 * 47390 < 2^24.
    let (weight_111, weight_29, weight_minus_9, weight_minus_3) = (
        _mm256_set1_ps(111.0),
        _mm256_set1_ps(29.0),
        _mm256_set1_ps(-9.0),
        _mm256_set1_ps(-3.0),
    );
    let half_plus_window_offset = _mm256_set1_epi32((1 << 15) + OFFSET * 256);
    // RGB: 4 pixels' 12 bytes per lane (dwords 0..3 and 3..6), shuffled to the
    // even pixels' 6 bytes | the odd pixels' 6 bytes, 2 zero bytes after each.
    let rgb_bytes = _mm_setr_epi32(-1, -1, -1, 0);
    let rgb_halves = _mm256_setr_epi32(0, 1, 2, 0, 3, 4, 5, 0);
    let even_odd = _mm_setr_epi8(0, 1, 2, 6, 7, 8, -1, -1, 3, 4, 5, 9, 10, 11, -1, -1);
    let even_odd_ymm = _mm256_set_m128i(even_odd, even_odd);
    // The low byte of each lane's two dwords: q0..q3 | q4..q7 as 8 bytes.
    let low_dwords = _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0);
    let col_totals: Vec<i32> = (0..out_width)
        .map(|output| catmull_window(output, width).1.iter().sum())
        .collect();
    for y in 0..out_height {
        let (_, vertical_weights) = catmull_window(y, height);
        let vertical_total: i32 = vertical_weights.iter().sum();
        let row = |row_index: isize| {
            if (0..height as isize).contains(&row_index) {
                // SAFETY: row_index < height, so the row starts within src
                // (width * height * channels samples, the caller's contract).
                unsafe { src.as_ptr().add(row_index as usize * row_samples) }
            } else {
                zeros.as_ptr()
            }
        };
        let first = 2 * y as isize - 3;
        let rows: [*const u8; 8] = std::array::from_fn(|tap| row(first + tap as isize));
        // 32 bytes from pixel `first_pixel` of tap row `tap` (24 for RGB,
        // then shuffled).
        let load = |tap: usize, first_pixel: usize| {
            // SAFETY: first_pixel = 2 first_output with first_output +
            // vertical_step_outputs <= out_width (cover_steps), so the gray
            // load's 32 bytes end within 2 out_width = row_samples, the RGB
            // load is taken where its 32 bytes end within row_samples (the
            // guard), and the masked RGB loads read the 12 bytes of pixels
            // first_pixel and first_pixel + 4, ending within 3 (first_pixel +
            // 8) <= row_samples; each row is a src row (the caller's contract)
            // or `zeros` (row_samples + 32).
            unsafe {
                if channels != 3 {
                    _mm256_loadu_si256(rows[tap].add(first_pixel * channels) as *const __m256i)
                } else if 3 * first_pixel + 32 <= row_samples {
                    let bytes =
                        _mm256_loadu_si256(rows[tap].add(3 * first_pixel) as *const __m256i);
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
                    _mm256_set_m128i(group(first_pixel + 4), group(first_pixel))
                }
            }
        };
        let vertical_step = |first_output: usize| {
            prefetch_rows_after_window(
                src.as_ptr(),
                2 * y + 5,
                row_samples,
                2 * first_output * channels,
            );
            let loaded: [__m256i; 8] = std::array::from_fn(|tap| load(tap, 2 * first_output));
            // Per 128-bit lane, the low unpacks hold samples 0..8, the high
            // ones 8..16.
            let window_sums = |low_half: bool| {
                let pair = |low_tap: usize, high_tap: usize| {
                    if low_half {
                        _mm256_unpacklo_epi8(loaded[low_tap], loaded[high_tap])
                    } else {
                        _mm256_unpackhi_epi8(loaded[low_tap], loaded[high_tap])
                    }
                };
                let near_sums = _mm256_add_epi16(
                    _mm256_maddubs_epi16(pair(3, 1), near_weights),
                    _mm256_maddubs_epi16(pair(4, 6), near_weights),
                );
                let far_sums = _mm256_add_epi16(
                    _mm256_maddubs_epi16(pair(2, 0), far_weights),
                    _mm256_maddubs_epi16(pair(5, 7), far_weights),
                );
                let near_sums = _mm256_sub_epi16(near_sums, offset);
                (
                    _mm256_madd_epi16(_mm256_unpacklo_epi16(near_sums, far_sums), ones),
                    _mm256_madd_epi16(_mm256_unpackhi_epi16(near_sums, far_sums), ones),
                )
            };
            // q0: samples 0..4 | 16..20, q1: 4..8 | 20..24, q2: 8..12 | 24..28,
            // q3: 12..16 | 28..32.
            let ((q0, q1), (q2, q3)) = (window_sums(true), window_sums(false));
            let [q0, q1, q2, q3] =
                [q0, q1, q2, q3].map(|result| _mm256_castps_si256(_mm256_cvtepi32_ps(result)));
            // The even (odd) pixels of q0, q1 and of q2, q3, lane by lane.
            let split = |first: __m256i, second: __m256i, odd: bool| {
                if odd {
                    _mm256_castps_si256(_mm256_shuffle_ps::<0b11_01_11_01>(
                        _mm256_castsi256_ps(first),
                        _mm256_castsi256_ps(second),
                    ))
                } else {
                    _mm256_castps_si256(_mm256_shuffle_ps::<0b10_00_10_00>(
                        _mm256_castsi256_ps(first),
                        _mm256_castsi256_ps(second),
                    ))
                }
            };
            for (parity, odd) in [(even_ptr, false), (odd_ptr, true)] {
                let (first_source, second_source) = if channels == 3 {
                    (if odd { q2 } else { q0 }, if odd { q3 } else { q1 })
                } else {
                    (split(q0, q1, odd), split(q2, q3, odd))
                };
                // RGB: 6 samples and 2 junk lanes per store, the next store
                // overwriting the junk.
                let lane_samples = if channels == 3 { 6 } else { LANES };
                // SAFETY: first_output + vertical_step_outputs <= out_width
                // (cover_steps): the stores end at first_output channels +
                // lane_samples + LANES, within the parity's out_row_samples (16
                // values per step but for RGB) or, for RGB, out_row_samples + 2
                // of its 2 LANES + pad after.
                unsafe {
                    _mm256_storeu_si256(
                        parity.add(first_output * channels) as *mut __m256i,
                        _mm256_permute2x128_si256::<0x20>(first_source, second_source),
                    );
                    _mm256_storeu_si256(
                        parity.add(first_output * channels + lane_samples) as *mut __m256i,
                        _mm256_permute2x128_si256::<0x31>(first_source, second_source),
                    );
                }
            }
        };
        cover_steps(out_width, vertical_step_outputs, &vertical_step);
        if channels == 3 {
            // The last step's junk lanes, back to the zero padding.
            // SAFETY: the stores' LANES values from out_row_samples lie within
            // the 2 LANES + pad after even's and odd's row.
            unsafe {
                _mm256_storeu_si256(
                    even_ptr.add(out_row_samples) as *mut __m256i,
                    _mm256_setzero_si256(),
                );
                _mm256_storeu_si256(
                    odd_ptr.add(out_row_samples) as *mut __m256i,
                    _mm256_setzero_si256(),
                );
            }
        }
        // SAFETY: y < out_height: the row starts within the out_width *
        // out_height * channels outputs.
        let dst_ptr = unsafe { out.as_mut_ptr().add(y * out_row_samples) };
        let horizontal_step = |first_sample: usize| {
            let load = |parity: *const i32, offset: isize| {
                // SAFETY: first_sample + LANES <= out_row_samples (cover_steps)
                // and offset is -2..=2, so the load's 8 values lie within the
                // parity's two zero pixels before, out_row_samples and 2 LANES
                // + pad after.
                unsafe {
                    _mm256_loadu_ps(
                        parity.offset(first_sample as isize + offset * channels as isize)
                            as *const f32,
                    )
                }
            };
            let (even, odd) = (even_ptr as *const i32, odd_ptr as *const i32);
            let pair = |left: __m256, right: __m256| _mm256_add_ps(left, right);
            let near = _mm256_fmadd_ps(
                pair(load(odd, -1), load(even, 1)),
                weight_29,
                _mm256_mul_ps(pair(load(even, 0), load(odd, 0)), weight_111),
            );
            let far = _mm256_fmadd_ps(
                pair(load(odd, -2), load(even, 2)),
                weight_minus_3,
                _mm256_mul_ps(pair(load(even, -1), load(odd, 1)), weight_minus_9),
            );
            let sums = _mm256_cvtps_epi32(_mm256_add_ps(near, far));
            let whole = vertical_total == 256
                && first_sample >= 2 * channels
                && first_sample + LANES + 2 * channels <= out_row_samples;
            let quotient = if whole {
                _mm256_srai_epi32::<16>(_mm256_add_epi32(sums, half_plus_window_offset))
            } else {
                // floor((2 sum + d) / (2 d)) per lane, d = vertical_total *
                // the column total, the sum's offset (OFFSET per weight unit)
                // added back.
                let column_total =
                    |lane: usize| col_totals[((first_sample + lane) / channels).min(out_width - 1)];
                let denominators: [f64; LANES] =
                    std::array::from_fn(|lane| f64::from(vertical_total * column_total(lane)));
                let offsets: [f64; LANES] =
                    std::array::from_fn(|lane| f64::from(OFFSET * column_total(lane)));
                let divide = |sums_half: __m128i, denominators: &[f64], offsets: &[f64]| {
                    // SAFETY: both calls pass 4 values in each slice, which
                    // the loads read.
                    let (divisors, offsets) = unsafe {
                        (
                            _mm256_loadu_pd(denominators.as_ptr()),
                            _mm256_loadu_pd(offsets.as_ptr()),
                        )
                    };
                    let sums = _mm256_add_pd(_mm256_cvtepi32_pd(sums_half), offsets);
                    let quotient = _mm256_div_pd(
                        _mm256_add_pd(_mm256_add_pd(sums, sums), divisors),
                        _mm256_add_pd(divisors, divisors),
                    );
                    _mm256_cvtpd_epi32(_mm256_floor_pd(quotient))
                };
                let low = divide(
                    _mm256_castsi256_si128(sums),
                    &denominators[..4],
                    &offsets[..4],
                );
                let high = divide(
                    _mm256_extracti128_si256::<1>(sums),
                    &denominators[4..],
                    &offsets[4..],
                );
                _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(low), high)
            };
            // i32 -> i16 (saturating) -> u8 (clamped to 0..=255): q0..3 in the
            // low lane's first dword, q4..7 in the high lane's.
            let words = _mm256_packs_epi32(quotient, quotient);
            let bytes = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(words, words), low_dwords);
            // SAFETY: first_sample + LANES <= out_row_samples (cover_steps):
            // the store's 8 bytes lie within the row.
            unsafe {
                _mm_storel_epi64(
                    dst_ptr.add(first_sample) as *mut __m128i,
                    _mm256_castsi256_si128(bytes),
                );
            }
        };
        cover_steps(out_row_samples, LANES, &horizontal_step);
    }
    // SAFETY: every row's steps covered all of its outputs.
    unsafe { crate::cpu::written_output(out_buf) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::catmull_x2::catmull_x2_scalar;
    use crate::exact::catmull_x2::tests::{HEIGHTS, WIDTHS};
    use crate::exact::tests::assert_equals_reference;
    use crate::testing::{CHANNELS, size_grid};

    #[test]
    fn avx2_equals_scalar() {
        if !crate::testing::avx2_fma_detected() {
            return;
        }
        let kernel = |src: &[u8], width, height, channels| {
            // SAFETY: AVX2 and FMA are available; src holds width * height *
            // channels samples, width and height even (WIDTHS and HEIGHTS),
            // channels 1 or 3.
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
}

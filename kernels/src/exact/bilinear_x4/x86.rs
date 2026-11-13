//! The AVX2 kernel of the exact x4 bilinear reduction.
//!
//! # Safety
//! Every `unsafe fn` here requires AVX2 and the bounds its doc states; loads
//! and stores are justified against those and the frame's buffers at their
//! loop guards.

use super::{bilinear_x4_scalar, window};
use crate::exact::x86::cover_steps;
use crate::x86::prefetch_rows_after_window;

/// Samples per pixel of the RGBX layout RGB rows take in the phase arrays.
const RGBX: usize = 4;

/// Samples per pixel in the phase arrays: RGB is read as RGBX.
const fn phase_channels(is_rgb: bool) -> usize {
    if is_rgb { RGBX } else { 1 }
}

/// The kernel's frame. Per output row y, `vertical(rows, y, phases)` splits
/// the weighted sum of the window's 8 `rows` (a zero row for a clipped one)
/// by pixel phase (x mod 4) into the four phase arrays, which have a zero
/// pixel before and after; `horizontal(phases, vertical_total, col_totals,
/// dst)` writes the row's outputs from them. RGB rows are written as RGBX and
/// packed back. Inlined, so both closures compile with the caller's target
/// features.
///
/// # Safety
/// `src` holds `width * height * channels` samples (3 for `IS_RGB`, else 1),
/// `width` and `height` multiples of 4; AVX2 is available for `IS_RGB`;
/// `vertical` writes the row's phase samples only, and `horizontal` the row's
/// `width / 4 * phase_channels(IS_RGB)` bytes.
#[inline(always)]
pub(crate) unsafe fn x4_frame<const IS_RGB: bool>(
    src: &[u8],
    (width, height): (usize, usize),
    vertical: impl Fn(&[*const u8; 8], usize, &[*mut u16; 4]),
    horizontal: impl Fn(&[*mut u16; 4], u32, &[u32], *mut u8),
) -> Vec<u8> {
    let phase_channels = phase_channels(IS_RGB);
    let channels = if IS_RGB { 3 } else { 1 };
    let (out_width, out_height, row_samples) = (width / 4, height / 4, width * channels);
    debug_assert!(src.len() >= width * height * channels);
    let phase_row_samples = out_width * phase_channels;
    let mut out_buf = crate::cpu::unwritten_output::<u8>(out_width * out_height * channels);
    let dst_ptr = out_buf.as_mut_ptr().cast::<u8>();
    let phase_len = phase_channels + phase_row_samples + phase_channels;
    let mut phases = vec![0u16; 4 * phase_len];
    let phase_ptrs: [*mut u16; 4] = std::array::from_fn(|phase| {
        // SAFETY: phase < 4: the array starts within the 4 phase_len values.
        unsafe { phases.as_mut_ptr().add(phase * phase_len + phase_channels) }
    });
    let mut rgbx_row = vec![0u8; if IS_RGB { phase_row_samples } else { 0 }];
    let zeros = vec![0u8; row_samples + 64];
    let col_totals: Vec<u32> = (0..out_width)
        .map(|output| window(output, width).1.iter().sum())
        .collect();
    for y in 0..out_height {
        let (_, vertical_weights) = window(y, height);
        let row = |row_index: isize| {
            if (0..height as isize).contains(&row_index) {
                // SAFETY: row_index < height, so the row starts within src
                // (width * height * channels samples, the caller's contract).
                unsafe { src.as_ptr().add(row_index as usize * row_samples) }
            } else {
                zeros.as_ptr()
            }
        };
        let first = 4 * y as isize - 2;
        let rows: [*const u8; 8] = std::array::from_fn(|tap| row(first + tap as isize));
        vertical(&rows, y, &phase_ptrs);
        let dst = if IS_RGB {
            rgbx_row.as_mut_ptr()
        } else {
            // SAFETY: y < out_height: the row starts within the out_width *
            // out_height outputs.
            unsafe { dst_ptr.add(y * phase_row_samples) }
        };
        horizontal(&phase_ptrs, vertical_weights.iter().sum(), &col_totals, dst);
        if IS_RGB {
            // SAFETY: y < out_height: the row's out_width * 3 bytes lie within
            // the output (u8 has no invalid values; the pack only writes
            // them); AVX2 is available for RGB (the caller's contract), and
            // `rgbx_row` holds the row's out_width * 4 bytes.
            unsafe {
                let out_row =
                    std::slice::from_raw_parts_mut(dst_ptr.add(y * out_width * 3), out_width * 3);
                crate::convolve::x86::pack_from_rgbx_avx2(&rgbx_row, out_row);
            }
        }
    }
    // SAFETY: every row's steps covered all of its outputs.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// The AVX2 kernel (see `x4_frame`).
///
/// Vertical, 32 input samples per step: `pmaddubsw` on the row pairs (3, 2)
/// and (4, 5) at (7, 5) and (1, 0) and (6, 7) at (3, 1), so the window's sum
/// (at most 32 * 255) is u16. Per 128-bit lane the low unpacks hold samples
/// 0..8 and the high ones 8..16, already one pixel per phase for RGBX and a
/// word shuffle and an unpack away for gray; one 64-bit permute then joins
/// the lanes.
///
/// Horizontal, 16 samples per step, unit-stride at the output resolution:
/// `7 (p1 + p2) + 5 (p0 + p3) + 3 (p3[o - 1] + p0[o + 1]) + (p2[o - 1] +
/// p1[o + 1])`, the pair sums i16 and two `madd`s into i32;
/// `(sum + 2^9) >> 10` where both windows are whole, else an exact per-lane
/// division in f32 (the sum is below 2^24, and a non-integer quotient lies
/// at least 1 / (2d) >= 2^-11 from an integer, far above f32's rounding
/// error at 256).
///
/// # Safety
/// AVX2 must be available; as `x4_frame`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bilinear_x4_avx2<const IS_RGB: bool>(
    src: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    let phase_channels = phase_channels(IS_RGB);
    let channels = if IS_RGB { 3 } else { 1 };
    let (out_width, row_samples) = (width / 4, width * channels);
    // A vertical step writes 8 samples per phase, a horizontal one 16.
    let (vertical_step_outputs, horizontal_step_outputs) =
        (8 / phase_channels, 16 / phase_channels);
    if out_width < horizontal_step_outputs {
        return bilinear_x4_scalar(src, width, height, channels);
    }
    let near_weights = _mm256_set1_epi16((5 << 8) | 7);
    let far_weights = _mm256_set1_epi16((1 << 8) | 3);
    let (madd_near, madd_far) = (
        _mm256_set1_epi32((5 << 16) | 7),
        _mm256_set1_epi32((1 << 16) | 3),
    );
    let half = _mm256_set1_epi32(1 << 9);
    // Gray: a lane's words 0..8 as (0, 4), (1, 5), (2, 6), (3, 7).
    let pair_phases = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
    ));
    // RGB: 4 pixels' 12 bytes per lane (dwords 0..3 and 3..6), widened to RGBX.
    let rgb_bytes = _mm_setr_epi32(-1, -1, -1, 0);
    let rgb_halves = _mm256_setr_epi32(0, 1, 2, 0, 3, 4, 5, 0);
    let to_rgbx = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
    let to_rgbx_ymm = _mm256_broadcastsi128_si256(to_rgbx);
    let vertical = |rows: &[*const u8; 8], y: usize, phase_ptrs: &[*mut u16; 4]| {
        // 32 phase-layout bytes from pixel `first_pixel` of tap row `tap`.
        let load = |tap: usize, first_pixel: usize| {
            // SAFETY: first_pixel = 4 first_output with first_output +
            // vertical_step_outputs <= out_width (cover_steps), so the gray
            // load's 32 bytes end within 4 out_width = row_samples, the RGB
            // load is taken where its 32 bytes end within row_samples (the
            // guard), and the masked RGB loads read the 12 bytes of pixels
            // first_pixel and first_pixel + 4, ending within 3 (first_pixel +
            // 8) <= row_samples; each row is a src row (the frame's contract)
            // or `zeros` (row_samples + 64).
            unsafe {
                if !IS_RGB {
                    _mm256_loadu_si256(rows[tap].add(first_pixel) as *const __m256i)
                } else if 3 * first_pixel + 32 <= row_samples {
                    let bytes =
                        _mm256_loadu_si256(rows[tap].add(3 * first_pixel) as *const __m256i);
                    _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(bytes, rgb_halves), to_rgbx_ymm)
                } else {
                    let rgbx = |pixel: usize| {
                        _mm_shuffle_epi8(
                            _mm_maskload_epi32(rows[tap].add(3 * pixel) as *const i32, rgb_bytes),
                            to_rgbx,
                        )
                    };
                    _mm256_set_m128i(rgbx(first_pixel + 4), rgbx(first_pixel))
                }
            }
        };
        let vertical_step = |first_output: usize| {
            let first_pixel = 4 * first_output;
            prefetch_next_rows(src, 4 * y + 6, row_samples, first_pixel * channels);
            prefetch_next_rows(src, 4 * y + 8, row_samples, first_pixel * channels);
            let loaded: [__m256i; 8] = std::array::from_fn(|tap| load(tap, first_pixel));
            let window_sums = |low_half: bool| {
                let pair = |low_tap: usize, high_tap: usize| {
                    if low_half {
                        _mm256_unpacklo_epi8(loaded[low_tap], loaded[high_tap])
                    } else {
                        _mm256_unpackhi_epi8(loaded[low_tap], loaded[high_tap])
                    }
                };
                let near = _mm256_add_epi16(
                    _mm256_maddubs_epi16(pair(3, 2), near_weights),
                    _mm256_maddubs_epi16(pair(4, 5), near_weights),
                );
                let far = _mm256_add_epi16(
                    _mm256_maddubs_epi16(pair(1, 0), far_weights),
                    _mm256_maddubs_epi16(pair(6, 7), far_weights),
                );
                _mm256_add_epi16(near, far)
            };
            let (low, high) = (window_sums(true), window_sums(false));
            // Per lane: phases 0 and 1, then 2 and 3, 4 words each.
            let (phases01, phases23) = if IS_RGB {
                (low, high)
            } else {
                let (low_shuffled, high_shuffled) = (
                    _mm256_shuffle_epi8(low, pair_phases),
                    _mm256_shuffle_epi8(high, pair_phases),
                );
                (
                    _mm256_unpacklo_epi32(low_shuffled, high_shuffled),
                    _mm256_unpackhi_epi32(low_shuffled, high_shuffled),
                )
            };
            let (phases01, phases23) = (
                _mm256_permute4x64_epi64::<0b11_01_10_00>(phases01),
                _mm256_permute4x64_epi64::<0b11_01_10_00>(phases23),
            );
            let at = first_output * phase_channels;
            // SAFETY: first_output + 8 / phase_channels <= out_width
            // (cover_steps): each store's 8 words from at end within the phase
            // array's out_width phase_channels (the frame's `phase_ptrs`).
            unsafe {
                _mm_storeu_si128(
                    phase_ptrs[0].add(at) as *mut __m128i,
                    _mm256_castsi256_si128(phases01),
                );
                _mm_storeu_si128(
                    phase_ptrs[1].add(at) as *mut __m128i,
                    _mm256_extracti128_si256::<1>(phases01),
                );
                _mm_storeu_si128(
                    phase_ptrs[2].add(at) as *mut __m128i,
                    _mm256_castsi256_si128(phases23),
                );
                _mm_storeu_si128(
                    phase_ptrs[3].add(at) as *mut __m128i,
                    _mm256_extracti128_si256::<1>(phases23),
                );
            }
        };
        cover_steps(out_width, vertical_step_outputs, &vertical_step);
    };
    let horizontal =
        |phase_ptrs: &[*mut u16; 4], vertical_total: u32, col_totals: &[u32], dst: *mut u8| {
            let horizontal_step = |first_output: usize| {
                let at = first_output * phase_channels;
                let load = |phase: usize, offset: isize| {
                    // SAFETY: first_output + 16 / phase_channels <= out_width
                    // (cover_steps) and offset is -1..=1, so the load's 16
                    // words lie within the phase array's zero pixel before,
                    // out_width phase_channels words and zero pixel after.
                    unsafe {
                        _mm256_loadu_si256(
                            phase_ptrs[phase].offset(at as isize + offset * phase_channels as isize)
                                as *const __m256i,
                        )
                    }
                };
                let pair = |left: __m256i, right: __m256i| _mm256_add_epi16(left, right);
                let (inner, outer) = (pair(load(1, 0), load(2, 0)), pair(load(0, 0), load(3, 0)));
                let (next, last) = (pair(load(3, -1), load(0, 1)), pair(load(2, -1), load(1, 1)));
                let weighted_sums = |low_half: bool| {
                    let unpack = |low_words: __m256i, high_words: __m256i| {
                        if low_half {
                            _mm256_unpacklo_epi16(low_words, high_words)
                        } else {
                            _mm256_unpackhi_epi16(low_words, high_words)
                        }
                    };
                    _mm256_add_epi32(
                        _mm256_madd_epi16(unpack(inner, outer), madd_near),
                        _mm256_madd_epi16(unpack(next, last), madd_far),
                    )
                };
                // Samples 0..4 | 8..12 and 4..8 | 12..16.
                let (low, high) = (weighted_sums(true), weighted_sums(false));
                let whole = vertical_total == 32
                    && first_output >= 1
                    && first_output + horizontal_step_outputs < out_width;
                let round = |sums: __m256i, first: usize| {
                    if whole {
                        return _mm256_srli_epi32::<10>(_mm256_add_epi32(sums, half));
                    }
                    // floor((2 sum + d) / (2 d)) per lane, d = vertical_total *
                    // the column total.
                    let denominators: [f32; 8] = std::array::from_fn(|lane| {
                        let sample = first + lane % 4 + 8 * (lane / 4);
                        (vertical_total
                            * col_totals
                                [(first_output + sample / phase_channels).min(out_width - 1)])
                            as f32
                    });
                    // SAFETY: the load reads the 8 values of `denominators`.
                    let divisors = unsafe { _mm256_loadu_ps(denominators.as_ptr()) };
                    let sums = _mm256_cvtepi32_ps(sums);
                    let quotient = _mm256_div_ps(
                        _mm256_add_ps(_mm256_add_ps(sums, sums), divisors),
                        _mm256_add_ps(divisors, divisors),
                    );
                    _mm256_cvttps_epi32(_mm256_floor_ps(quotient))
                };
                let words = _mm256_packs_epi32(round(low, 0), round(high, 4));
                let bytes =
                    _mm256_permute4x64_epi64::<0b00_00_10_00>(_mm256_packus_epi16(words, words));
                // SAFETY: first_output + 16 / phase_channels <= out_width
                // (cover_steps): the store's 16 bytes from at end within the
                // row's out_width phase_channels bytes (the frame's `dst`).
                unsafe {
                    _mm_storeu_si128(dst.add(at) as *mut __m128i, _mm256_castsi256_si128(bytes));
                }
            };
            cover_steps(out_width, horizontal_step_outputs, &horizontal_step);
        };
    // SAFETY: this kernel's contract is the frame's, with AVX2 enabled here;
    // `vertical` and `horizontal` write within the bounds argued at their
    // stores.
    unsafe { x4_frame::<IS_RGB>(src, (width, height), vertical, horizontal) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::bilinear_x4::tests::{HEIGHTS, WIDTHS};
    use crate::exact::tests::{assert_equals_reference, size_grid};
    use crate::testing::CHANNELS;

    #[test]
    fn avx2_equals_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let kernel = |src: &[u8], width, height, channels| {
            // SAFETY: AVX2 is available; src holds width * height * channels
            // samples, width and height multiples of 4 (WIDTHS and HEIGHTS).
            unsafe {
                match channels {
                    1 => bilinear_x4_avx2::<false>(src, width, height),
                    _ => bilinear_x4_avx2::<true>(src, width, height),
                }
            }
        };
        assert_equals_reference(
            size_grid(&WIDTHS, &HEIGHTS),
            &CHANNELS,
            u8::MAX,
            &kernel,
            bilinear_x4_scalar,
        );
    }
}

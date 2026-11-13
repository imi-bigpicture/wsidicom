//! The AVX2 kernel of the exact x2 bilinear reduction.
//!
//! # Safety
//! The kernel requires AVX2 and the source length its doc states; loads and
//! stores are justified against those and its own buffers at their loop
//! guards.

use super::{horizontal_scalar, vertical_sums_scalar, window};
use crate::exact::x86::keep_even_pixels;
use crate::x86::prefetch_rows_after_window;

/// The vertical sums `row0 + row3 + 3 (row1 + row2)` in u16, then the
/// rounded 4-tap sums at every sample as bytes, of which the even pixels are
/// kept.
///
/// # Safety
/// AVX2 must be available; `src` holds `width * height * channels` samples.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bilinear_x2_avx2(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    const LANES: usize = 16;
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
    let mut sums = vec![0u16; channels + row_samples + 2 * LANES + 3 * channels];
    // The rounded 4-tap sums at every sample.
    let mut full = vec![0u8; row_samples + 2 * LANES];
    let zeros = vec![0u8; row_samples + LANES];
    let triple = |sum: __m256i| _mm256_add_epi16(sum, _mm256_slli_epi16::<1>(sum));
    let half = _mm256_set1_epi16(32);
    // A clipped vertical window (total 7): (2 sum + 56) * 37450 >> 22 is
    // (2 sum + 56) / 112, the rounded sum / 56.
    let (edge_bias, divide_by_112) = (_mm256_set1_epi16(56), _mm256_set1_epi16(37450u16 as i16));
    for y in 0..out_height {
        let (first_row, vertical_weights) = window(y, height);
        let dst = &mut out[y * out_width * channels..(y + 1) * out_width * channels];
        if vertical_weights.len() < 3 || row_samples < 2 * LANES || out_width < 3 {
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
                vertical_weights.iter().sum(),
                0..out_width,
                dst,
            );
            continue;
        }
        let row = |row_index: isize| {
            if (0..height as isize).contains(&row_index) {
                // SAFETY: row_index < height, so the row starts within src
                // (width * height * channels samples, the caller's contract).
                unsafe { src.as_ptr().add(row_index as usize * row_samples) }
            } else {
                zeros.as_ptr()
            }
        };
        let first = 2 * y as isize - 1;
        let rows: [*const u8; 4] = std::array::from_fn(|tap| row(first + tap as isize));
        let vertical_total: u32 = vertical_weights.iter().sum();
        // SAFETY: `sums` holds more than `channels` words.
        let sums_ptr = unsafe { sums.as_mut_ptr().add(channels) };
        let vertical_block = |x: usize| {
            if x.is_multiple_of(64) {
                prefetch_rows_after_window(src.as_ptr(), 2 * y + 3, row_samples, x);
            }
            let load = |tap: usize| {
                // SAFETY: x <= row_samples - LANES at both calls, so the load's
                // 16 bytes lie within a src row of row_samples samples (the
                // caller's contract) or `zeros` (row_samples + LANES).
                let bytes = unsafe { _mm_loadu_si128(rows[tap].add(x) as *const __m128i) };
                _mm256_cvtepu8_epi16(bytes)
            };
            let window_sums = _mm256_add_epi16(
                _mm256_add_epi16(load(0), load(3)),
                triple(_mm256_add_epi16(load(1), load(2))),
            );
            // SAFETY: x + LANES <= row_samples: the store's 16 words lie within
            // `sums` past its leading `channels`.
            unsafe { _mm256_storeu_si256(sums_ptr.add(x) as *mut __m256i, window_sums) };
        };
        let mut x = 0;
        while x + LANES <= row_samples {
            vertical_block(x);
            x += LANES;
        }
        if x < row_samples {
            vertical_block(row_samples - LANES);
        }
        let full_ptr = full.as_mut_ptr();
        let round = |tap_sums: __m256i| {
            if vertical_total == 8 {
                _mm256_srli_epi16::<6>(_mm256_add_epi16(tap_sums, half))
            } else {
                _mm256_srli_epi16::<6>(_mm256_mulhi_epu16(
                    _mm256_add_epi16(_mm256_add_epi16(tap_sums, tap_sums), edge_bias),
                    divide_by_112,
                ))
            }
        };
        let horizontal_block = |start: usize| {
            let taps = |block: usize| {
                // SAFETY: start <= row_samples - 2 LANES at both calls, so the
                // loads at -channels..=2 channels around sample start + block
                // LANES lie within `sums` (channels before the row, 2 LANES + 3
                // channels after it).
                let load = |offset: isize| unsafe {
                    let block_ptr = sums_ptr.add(start + block * LANES) as *const i16;
                    _mm256_loadu_si256(block_ptr.offset(offset) as *const __m256i)
                };
                let (pixel, two_pixels) = (channels as isize, 2 * channels as isize);
                round(_mm256_add_epi16(
                    _mm256_add_epi16(load(-pixel), load(two_pixels)),
                    triple(_mm256_add_epi16(load(0), load(pixel))),
                ))
            };
            let bytes =
                _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(taps(0), taps(1)));
            // SAFETY: start + 2 LANES <= row_samples: the store lies within
            // `full`.
            unsafe { _mm256_storeu_si256(full_ptr.add(start) as *mut __m256i, bytes) };
        };
        let mut start = 0;
        while start + 2 * LANES <= row_samples {
            horizontal_block(start);
            start += 2 * LANES;
        }
        if start < row_samples {
            horizontal_block(row_samples - 2 * LANES);
        }
        // SAFETY: avx2 is enabled here; full[..row_samples] holds 2 dst.len()
        // samples.
        unsafe { keep_even_pixels(&full[..row_samples], channels, dst) };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::bilinear_x2::bilinear_x2_scalar;
    use crate::exact::bilinear_x2::tests::{HEIGHTS, WIDTHS};
    use crate::exact::tests::assert_equals_reference;
    use crate::testing::{CHANNELS, size_grid};

    #[test]
    fn avx2_equals_scalar() {
        if !crate::testing::avx2_detected() {
            return;
        }
        let kernel = |src: &[u8], width, height, channels| {
            // SAFETY: AVX2 is available; src holds width * height * channels
            // samples.
            unsafe { bilinear_x2_avx2(src, width, height, channels) }
        };
        assert_equals_reference(
            size_grid(&WIDTHS, &HEIGHTS),
            &CHANNELS,
            u8::MAX,
            &kernel,
            bilinear_x2_scalar,
        );
    }
}

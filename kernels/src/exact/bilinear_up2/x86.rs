//! The AVX2 kernel of the exact x2 bilinear enlargement.
//!
//! # Safety
//! Every `unsafe fn` here requires AVX2 and the bounds its doc states; loads
//! and stores are justified against those at their loop guards.

use super::{bilinear_up2_scalar, far, horizontal_scalar};
use crate::sample::RGB;

/// The horizontal steps' byte shuffle: after the pack a lane's even outputs
/// are bytes 0..8 and its odd ones 8..16; this interleaves them pixel by
/// pixel.
pub(crate) fn interleave_bytes(channels: usize) -> [i8; 16] {
    std::array::from_fn(|lane_byte| {
        let pixel_pair = 2 * channels;
        let (pixel, byte_in_pair) = (lane_byte / pixel_pair, lane_byte % pixel_pair);
        let byte = if byte_in_pair < channels {
            pixel * channels + byte_in_pair
        } else {
            8 + pixel * channels + byte_in_pair - channels
        };
        byte.min(15) as i8
    })
}

/// The kernel's frame, per output row: `vertical(near, far, sums)` writes
/// `3 near + far` for the row's samples into a u16 row with the edge pixel
/// repeated on each side (and 32 more readable sums); `horizontal(sums, row)`
/// writes the row's outputs from them and returns the samples covered (any
/// rest is scalar). Inlined, so both closures compile with the caller's target
/// features.
///
/// # Safety
/// `src` holds `width * height * channels` samples, `width * channels >= 32`;
/// `vertical` writes the row's sums only, `horizontal` the row's
/// `2 width channels` bytes only.
#[inline(always)]
pub(crate) unsafe fn up2_rows(
    src: &[u8],
    (width, height, channels): (usize, usize, usize),
    vertical: impl Fn(*const u8, *const u8, *mut u16),
    horizontal: impl Fn(*const u16, *mut u8) -> usize,
) -> Vec<u8> {
    let (out_width, out_height, row_samples) = (2 * width, 2 * height, width * channels);
    let out_row_samples = out_width * channels;
    let mut out_buf = crate::cpu::unwritten_output::<u8>(out_row_samples * out_height);
    let dst_ptr = out_buf.as_mut_ptr().cast::<u8>();
    let write_rows = |start: usize, end: usize| {
        // The edge pixel repeated each side, and room for a row-end step's
        // loads past its outputs.
        let mut sums = vec![0u16; channels + row_samples + channels + 32];
        for y in start..end {
            // SAFETY: y / 2 and far(y, height) are below height, so both rows
            // start within src (width * height * channels samples, the
            // caller's contract).
            let (near, far) = unsafe {
                (
                    src.as_ptr().add(y / 2 * row_samples),
                    src.as_ptr().add(far(y, height) * row_samples),
                )
            };
            // SAFETY: `sums` holds more than `channels` words.
            vertical(near, far, unsafe { sums.as_mut_ptr().add(channels) });
            sums.copy_within(channels..2 * channels, 0);
            sums.copy_within(row_samples..row_samples + channels, row_samples + channels);
            // SAFETY: y < end <= out_height (with_prefault's rows): the row
            // starts within the out_row_samples * out_height outputs.
            let row = unsafe { dst_ptr.add(y * out_row_samples) };
            // SAFETY: `sums` holds more than `channels` words.
            let covered = horizontal(unsafe { sums.as_ptr().add(channels) }, row);
            // SAFETY: u8 has no invalid values; the scalar tail only writes,
            // within the row's out_row_samples bytes.
            let tail = unsafe { std::slice::from_raw_parts_mut(row, out_row_samples) };
            horizontal_scalar(
                &sums[channels..],
                channels,
                width,
                covered / channels * 2..out_width,
                tail,
            );
        }
    };
    // SAFETY: dst_ptr holds out_height rows of out_row_samples bytes;
    // write_rows(start, end) writes rows start..end only (the horizontal steps'
    // last store ends within the row).
    unsafe {
        crate::cpu::with_prefault(
            dst_ptr,
            out_row_samples * out_height,
            out_row_samples,
            out_height,
            write_rows,
        );
    }
    // SAFETY: every row's steps and scalar tail covered all of its outputs.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// The vertical sums `3 near + far`, then `horizontal_steps_avx2`.
///
/// # Safety
/// AVX2 must be available; `src` holds `width * height * channels` samples,
/// `channels` 1 or 3.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bilinear_up2_avx2(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    let row_samples = width * channels;
    debug_assert!(matches!(channels, 1 | RGB));
    // Rows of at least 32 samples: one vertical step of 16, one horizontal
    // step of two lanes and a neighbour pixel.
    if row_samples < 32 {
        return bilinear_up2_scalar(src, width, height, channels);
    }
    let interleave = interleave_bytes(channels);
    // SAFETY: the load reads the 16 bytes of `interleave`.
    let interleave = _mm256_broadcastsi128_si256(unsafe {
        _mm_loadu_si128(interleave.as_ptr() as *const __m128i)
    });
    let vertical = |near: *const u8, far: *const u8, sums: *mut u16| {
        let vertical_step = |x: usize| {
            let load = |row: *const u8| {
                // SAFETY: x <= row_samples - 16 at both calls: the load's 16
                // bytes lie within a src row of row_samples samples (up2_rows
                // passes src rows).
                let bytes = unsafe { _mm_loadu_si128(row.add(x) as *const __m128i) };
                _mm256_cvtepu8_epi16(bytes)
            };
            let near_samples = load(near);
            let window_sums = _mm256_add_epi16(
                _mm256_add_epi16(near_samples, near_samples),
                _mm256_add_epi16(near_samples, load(far)),
            );
            // SAFETY: x + 16 <= row_samples: the store lies within the row's
            // row_samples sums (up2_rows' `sums` past its leading pixel).
            unsafe { _mm256_storeu_si256(sums.add(x) as *mut __m256i, window_sums) };
        };
        let mut x = 0;
        while x + 16 <= row_samples {
            vertical_step(x);
            x += 16;
        }
        if x < row_samples {
            vertical_step(row_samples - 16);
        }
    };
    let horizontal = |sums: *const u16, row: *mut u8| {
        // SAFETY: avx2 is enabled here; up2_rows passes `sums` with a pixel
        // before and 32 sums after its row_samples, and a row of 2 row_samples
        // bytes; row_samples >= 32 (checked above).
        unsafe {
            if channels == RGB {
                horizontal_steps_avx2::<true>(sums, row_samples, channels, row, interleave)
            } else {
                horizontal_steps_avx2::<false>(sums, row_samples, channels, row, interleave)
            }
        }
    };
    // SAFETY: src holds width * height * channels samples (this kernel's
    // contract), row_samples >= 32 (checked above); `vertical` writes the row's
    // row_samples sums and `horizontal` the row's 2 row_samples bytes only.
    unsafe { up2_rows(src, (width, height, channels), vertical, horizontal) }
}

/// The even and odd outputs `(3 s[p] + s[p - 1] + 8) >> 4` and `(3 s[p] +
/// s[p + 1] + 8) >> 4` of a row of vertical sums (`sums` past its leading
/// edge pixel), packed to bytes and interleaved by one byte shuffle. Per
/// 128-bit lane 8 contiguous sums; for RGB 6 (two pixels), the lanes loaded 6
/// sums apart and stored as 16 bytes for their 12, the next store overwriting
/// the rest. The last step is shifted back to end at the row's end (RGB: its
/// second lane stored as 3 masked dwords). Returns `row_samples`, the samples
/// covered.
///
/// # Safety
/// AVX2 must be available; `row_samples >= 32`; `sums` has a pixel of sums
/// before it and 32 sums after its `row_samples`; `row` holds
/// `2 row_samples` bytes.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn horizontal_steps_avx2<const IS_RGB: bool>(
    sums: *const u16,
    row_samples: usize,
    channels: usize,
    row: *mut u8,
    interleave: std::arch::x86_64::__m256i,
) -> usize {
    use std::arch::x86_64::*;
    let lane_sums = if IS_RGB { 6 } else { 8 };
    let rounding = _mm256_set1_epi16(8);
    // Signed: the even outputs' far taps start a pixel before `sums`.
    let load = |at: isize| {
        // SAFETY: at is start or start -+ channels with start <= row_samples -
        // cover, so the 16 sums read (RGB: 8 at at and at + 6) lie within the
        // pixel before `sums` and the 32 sums after its row_samples (the
        // caller's contract).
        unsafe {
            if IS_RGB {
                _mm256_loadu2_m128i(
                    sums.offset(at + lane_sums as isize) as *const __m128i,
                    sums.offset(at) as *const __m128i,
                )
            } else {
                _mm256_loadu_si256(sums.offset(at) as *const __m256i)
            }
        }
    };
    let cover = 2 * lane_sums;
    let lane_dwords = _mm_setr_epi32(-1, -1, -1, 0);
    let mut next = 0;
    while next < row_samples {
        // The last step shifts back to end at the row's end. RGB: a step's
        // stores end 4 bytes past its 2 cover output bytes, so it is last
        // once those would pass the row's end; the rest is then a multiple
        // of 3 samples up to 12, which the shifted step covers.
        let last = IS_RGB && 2 * next + 2 * lane_sums + 16 > 2 * row_samples;
        let start = if IS_RGB {
            if last { row_samples - cover } else { next }
        } else {
            next.min(row_samples - cover)
        };
        let (start_at, pixel) = (start as isize, channels as isize);
        let center = load(start_at);
        let triple = _mm256_add_epi16(
            _mm256_add_epi16(center, center),
            _mm256_add_epi16(center, rounding),
        );
        let even = _mm256_srli_epi16::<4>(_mm256_add_epi16(triple, load(start_at - pixel)));
        let odd = _mm256_srli_epi16::<4>(_mm256_add_epi16(triple, load(start_at + pixel)));
        let bytes = _mm256_shuffle_epi8(_mm256_packus_epi16(even, odd), interleave);
        // SAFETY: `row` holds 2 row_samples bytes (the caller's contract). RGB:
        // a step that is not last stores 2 start..2 start + 2 lane_sums + 16 <=
        // 2 row_samples (`last`'s guard); the last, from start = row_samples -
        // cover, stores 16 bytes and 12 masked ones, ending at 2 row_samples.
        // Else start <= row_samples - cover and the 32 bytes end within 2
        // row_samples.
        unsafe {
            if IS_RGB {
                let high = row.add(2 * (start + lane_sums));
                _mm_storeu_si128(
                    row.add(2 * start) as *mut __m128i,
                    _mm256_castsi256_si128(bytes),
                );
                if last {
                    _mm_maskstore_epi32(
                        high as *mut i32,
                        lane_dwords,
                        _mm256_extracti128_si256::<1>(bytes),
                    );
                } else {
                    _mm_storeu_si128(high as *mut __m128i, _mm256_extracti128_si256::<1>(bytes));
                }
            } else {
                _mm256_storeu_si256(row.add(2 * start) as *mut __m256i, bytes);
            }
        }
        next = start + cover;
    }
    row_samples
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::bilinear_up2::tests::{HEIGHTS, WIDTHS};
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
            unsafe { bilinear_up2_avx2(src, width, height, channels) }
        };
        assert_equals_reference(
            size_grid(&WIDTHS, &HEIGHTS),
            &CHANNELS,
            u8::MAX,
            &kernel,
            bilinear_up2_scalar,
        );
    }

    #[test]
    fn interleave_bytes_reads_each_pixel_pair_in_order() {
        assert_eq!(
            interleave_bytes(1),
            [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]
        );
        assert_eq!(
            interleave_bytes(4),
            [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]
        );
    }
}

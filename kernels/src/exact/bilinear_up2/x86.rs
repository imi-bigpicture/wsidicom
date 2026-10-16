//! The AVX2 and AVX-512 kernels of the exact x2 bilinear enlargement.
//!
//! # Safety
//! Every `unsafe fn` here requires the CPU features in its `#[target_feature]`
//! (its dispatch checks them at runtime) and the bounds its caller documents
//! (the source length, or the rows behind raw pointers), which it reads and
//! writes against unchecked. Loads and stores are justified against those
//! bounds at their loop guards.

use super::{bilinear_up2_scalar, far, horizontal_scalar};

/// The horizontal steps' byte shuffle: lane byte `lane_byte` from output
/// pixel `lane_byte / (2 channels)` of the lane, the even one (bytes 0..8
/// after the pack) for its first `channels` bytes, the odd one (8..16) next.
pub(crate) fn interleave_bytes(channels: usize) -> [i8; 16] {
    std::array::from_fn(|lane_byte| {
        let (pixel, at) = (lane_byte / (2 * channels), lane_byte % (2 * channels));
        let byte = if at < channels {
            pixel * channels + at
        } else {
            8 + pixel * channels + at - channels
        };
        byte.min(15) as i8
    })
}

/// The SIMD kernels' frame, per output row: `vertical(near, far, sums)`
/// writes `3 near + far` for the row's `width * channels` samples into a u16
/// row with the edge pixel repeated on each side (and 32 more readable sums);
/// `horizontal(sums, row)` writes the row's outputs from them and returns
/// the samples covered (any rest is scalar). Inlined, so the two steps compile
/// with the caller's target features.
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
    let (ow, oh, ws) = (2 * width, 2 * height, width * channels);
    let ows = ow * channels;
    let mut out_buf = crate::cpu::unwritten_output::<u8>(ows * oh);
    let dp = out_buf.as_mut_ptr().cast::<u8>();
    let rows = |lo: usize, hi: usize| {
        // The vertical sums with one pixel each side repeated from the edge,
        // and room for a row-end step's loads past its outputs.
        let mut sums = vec![0u16; channels + ws + channels + 32];
        for y in lo..hi {
            // SAFETY: y / 2 and far(y, height) are below height, so both rows
            // start within src (width * height * channels samples, the
            // caller's contract).
            let (near, far) = unsafe {
                (
                    src.as_ptr().add(y / 2 * ws),
                    src.as_ptr().add(far(y, height) * ws),
                )
            };
            // SAFETY: `sums` holds more than `channels` words.
            vertical(near, far, unsafe { sums.as_mut_ptr().add(channels) });
            sums.copy_within(channels..2 * channels, 0);
            sums.copy_within(ws..ws + channels, ws + channels);
            // SAFETY: y < hi <= oh (with_prefault's rows): the row starts
            // within the ows * oh outputs.
            let row = unsafe { dp.add(y * ows) };
            // SAFETY: `sums` holds more than `channels` words.
            let covered = horizontal(unsafe { sums.as_ptr().add(channels) }, row);
            // SAFETY: u8 has no invalid values; the scalar tail only writes,
            // within the row's ows bytes.
            let tail = unsafe { std::slice::from_raw_parts_mut(row, ows) };
            horizontal_scalar(
                &sums[channels..],
                channels,
                width,
                covered / channels * 2..ow,
                tail,
            );
        }
    };
    // SAFETY: dp holds oh rows of ows bytes; rows(lo, hi) writes rows lo..hi
    // only (the horizontal steps' last store ends within the row).
    unsafe { crate::cpu::with_prefault(dp, ows * oh, ows, oh, rows) };
    // SAFETY: every row's steps and scalar tail covered all of its outputs.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// The AVX2 kernel. Vertical: `3 near + far`, 16 samples per step.
/// Horizontal (`horizontal_steps_avx2`): `3 v[p] + v[p - 1]` and
/// `3 v[p] + v[p + 1]` (the even and odd outputs), rounded, packed to bytes
/// and interleaved by pixel with one byte shuffle.
///
/// # Safety
/// AVX2 must be available; `src` holds `width * height * channels` samples.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bilinear_up2_avx2(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    let ws = width * channels;
    // Rows of at least 32 samples: one vertical step of 16, one horizontal
    // step of two lanes and a neighbor pixel.
    if !(1..=4).contains(&channels) || ws < 32 {
        return bilinear_up2_scalar(src, width, height, channels);
    }
    let interleave = interleave_bytes(channels);
    // SAFETY: the load reads the 16 bytes of `interleave`.
    let interleave = _mm256_broadcastsi128_si256(unsafe {
        _mm_loadu_si128(interleave.as_ptr() as *const __m128i)
    });
    let vertical = |near: *const u8, far: *const u8, sums: *mut u16| {
        let vstep = |x: usize| {
            let ld = |row: *const u8| {
                // SAFETY: x <= ws - 16 at both calls: the load's 16 bytes lie
                // within a src row of ws samples (up2_rows passes src rows).
                let bytes = unsafe { _mm_loadu_si128(row.add(x) as *const __m128i) };
                _mm256_cvtepu8_epi16(bytes)
            };
            let near_samples = ld(near);
            let window_sums = _mm256_add_epi16(
                _mm256_add_epi16(near_samples, near_samples),
                _mm256_add_epi16(near_samples, ld(far)),
            );
            // SAFETY: x + 16 <= ws: the store lies within the row's ws sums
            // (up2_rows' `sums` past its leading pixel).
            unsafe { _mm256_storeu_si256(sums.add(x) as *mut __m256i, window_sums) };
        };
        let mut x = 0;
        while x + 16 <= ws {
            vstep(x);
            x += 16;
        }
        if x < ws {
            vstep(ws - 16);
        }
    };
    let horizontal = |sums: *const u16, row: *mut u8| {
        // SAFETY: avx2 is enabled here; up2_rows passes `sums` with a pixel
        // before and 32 sums after its ws, and a row of 2 ws bytes;
        // ws >= 32 (checked above).
        unsafe {
            if channels == 3 {
                horizontal_steps_avx2::<true>(sums, ws, channels, row, interleave)
            } else {
                horizontal_steps_avx2::<false>(sums, ws, channels, row, interleave)
            }
        }
    };
    // SAFETY: src holds width * height * channels samples (this kernel's
    // contract), ws >= 32 (checked above); `vertical` writes the row's ws
    // sums and `horizontal` the row's 2 ws bytes only.
    unsafe { up2_rows(src, (width, height, channels), vertical, horizontal) }
}
/// The horizontal steps of `bilinear_up2_avx2` over a row of `ws` vertical
/// sums (`sums` past its leading edge pixel) into `row`: per 128-bit lane 8
/// sums, contiguous; for `RGB` 6 (two pixels), the lanes loaded 6 sums apart
/// and stored as 16 bytes for their 12, the next store overwriting the rest.
/// The last step is shifted back to end at the row's end (RGB: its second
/// lane stored as 3 masked dwords). Returns `ws`, the samples covered.
///
/// # Safety
/// AVX2 must be available; `sums` has a pixel of sums before it and 32
/// sums after its `ws`; `row` holds `2 ws` bytes; `ws >= 32`.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn horizontal_steps_avx2<const RGB: bool>(
    sums: *const u16,
    ws: usize,
    channels: usize,
    row: *mut u8,
    interleave: std::arch::x86_64::__m256i,
) -> usize {
    use std::arch::x86_64::*;
    let lane_sums = if RGB { 6 } else { 8 };
    let rounding = _mm256_set1_epi16(8);
    // Offsets are signed: the even outputs' far taps start a pixel before
    // `sums`.
    let ld = |at: isize| {
        // SAFETY: at is start or start -+ channels with start <= ws - cover,
        // so the 16 sums read (RGB: 8 at at and at + 6) lie within the pixel
        // before `sums` and the 32 sums after its ws (the caller's contract).
        unsafe {
            if RGB {
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
    while next < ws {
        // The last step shifts back to end at the row's end. RGB: a step's
        // stores end 4 bytes past its 2 cover output bytes, so it is last
        // once those would pass the row's end; the rest is then a multiple
        // of 3 samples up to 12, which the shifted step covers.
        let last = RGB && 2 * next + 2 * lane_sums + 16 > 2 * ws;
        let start = if RGB {
            if last { ws - cover } else { next }
        } else {
            next.min(ws - cover)
        };
        let (start_at, pixel) = (start as isize, channels as isize);
        let center = ld(start_at);
        let triple = _mm256_add_epi16(
            _mm256_add_epi16(center, center),
            _mm256_add_epi16(center, rounding),
        );
        let even = _mm256_srli_epi16::<4>(_mm256_add_epi16(triple, ld(start_at - pixel)));
        let odd = _mm256_srli_epi16::<4>(_mm256_add_epi16(triple, ld(start_at + pixel)));
        let bytes = _mm256_shuffle_epi8(_mm256_packus_epi16(even, odd), interleave);
        // SAFETY: `row` holds 2 ws bytes (the caller's contract). RGB: a step
        // that is not last stores 2 start..2 start + 2 lane_sums + 16 <= 2 ws
        // (`last`'s guard); the last, from start = ws - cover, stores 16 bytes
        // and 12 masked ones, ending at 2 ws. Else start <= ws - cover and
        // the 32 bytes end within 2 ws.
        unsafe {
            if RGB {
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
    ws
}

/// The AVX-512 kernel: the AVX2 kernel at twice the width, 32 vertical sums
/// per step and 64 output bytes per horizontal step (`horizontal_steps_avx512`).
///
/// # Safety
/// AVX-512 F/BW must be available; `src` holds `width * height * channels` samples.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bilinear_up2_avx512(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    let ws = width * channels;
    // Rows of at least 32 samples: one vertical step, one horizontal step.
    if !(1..=4).contains(&channels) || ws < 32 {
        return bilinear_up2_scalar(src, width, height, channels);
    }
    let interleave = interleave_bytes(channels);
    // SAFETY: the load reads the 16 bytes of `interleave`.
    let interleave =
        _mm512_broadcast_i32x4(unsafe { _mm_loadu_si128(interleave.as_ptr() as *const __m128i) });
    let vertical = |near: *const u8, far: *const u8, sums: *mut u16| {
        let vstep = |x: usize| {
            let ld = |row: *const u8| {
                // SAFETY: x <= ws - 32 at both calls: the load's 32 bytes lie
                // within a src row of ws samples (up2_rows passes src rows).
                let bytes = unsafe { _mm256_loadu_si256(row.add(x) as *const __m256i) };
                _mm512_cvtepu8_epi16(bytes)
            };
            let near_samples = ld(near);
            let window_sums = _mm512_add_epi16(
                _mm512_add_epi16(near_samples, near_samples),
                _mm512_add_epi16(near_samples, ld(far)),
            );
            // SAFETY: x + 32 <= ws: the store lies within the row's ws sums
            // (up2_rows' `sums` past its leading pixel).
            unsafe { _mm512_storeu_si512(sums.add(x) as *mut __m512i, window_sums) };
        };
        let mut x = 0;
        while x + 32 <= ws {
            vstep(x);
            x += 32;
        }
        if x < ws {
            vstep(ws - 32);
        }
    };
    let horizontal = |sums: *const u16, row: *mut u8| {
        // SAFETY: avx512f+bw are enabled here; up2_rows passes `sums` with a
        // pixel before and 32 sums after its ws, and a row of 2 ws bytes;
        // ws >= 32 (checked above).
        unsafe {
            if channels == 3 {
                horizontal_steps_avx512::<true>(sums, ws, channels, row, interleave)
            } else {
                horizontal_steps_avx512::<false>(sums, ws, channels, row, interleave)
            }
        }
    };
    // SAFETY: src holds width * height * channels samples (this kernel's
    // contract), ws >= 32 (checked above); `vertical` writes the row's ws
    // sums and `horizontal` the row's 2 ws bytes only.
    unsafe { up2_rows(src, (width, height, channels), vertical, horizontal) }
}

/// The horizontal steps of `bilinear_up2_avx512`: as `horizontal_steps_avx2`
/// over four 128-bit lanes. For `RGB` the lanes take 6 sums each, 6 apart
/// (one word permute of 32 contiguous sums per load), and their 12 output
/// bytes are joined by a dword permute into 48, stored as 64 (the next step
/// overwriting the rest). The last step is shifted back to end at the row's
/// end (RGB: a masked store of its 48 bytes). Returns `ws`, the samples
/// covered.
///
/// # Safety
/// AVX-512 F/BW must be available; `sums` has a pixel of sums before it and
/// 32 sums after its `ws`; `row` holds `2 ws` bytes; `ws >= 32`.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
pub(crate) unsafe fn horizontal_steps_avx512<const RGB: bool>(
    sums: *const u16,
    ws: usize,
    channels: usize,
    row: *mut u8,
    interleave: std::arch::x86_64::__m512i,
) -> usize {
    use std::arch::x86_64::*;
    let lane_sums = if RGB { 6 } else { 8 };
    let rounding = _mm512_set1_epi16(8);
    let lane_words: [u16; 32] =
        std::array::from_fn(|word| (lane_sums * (word / 8) + word % 8) as u16);
    // SAFETY: the load reads the 32 words of `lane_words`.
    let lane_words = unsafe { _mm512_loadu_si512(lane_words.as_ptr() as *const __m512i) };
    let lane_bytes = _mm512_setr_epi32(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0, 0, 0, 0);
    // Offsets are signed: the even outputs' far taps start a pixel before
    // `sums`.
    let ld = |at: isize| {
        // SAFETY: at is start or start -+ channels, where start + 32 <= ws
        // (a full-width step) or start <= ws - cover, so the 32 sums read lie
        // within the pixel before `sums` and the 32 sums after its ws (the
        // caller's contract).
        let words = unsafe { _mm512_loadu_si512(sums.offset(at) as *const __m512i) };
        if RGB {
            _mm512_permutexvar_epi16(lane_words, words)
        } else {
            words
        }
    };
    let cover = 4 * lane_sums;
    let mut next = 0;
    while next < ws {
        // A step's 64-byte store starts at 2 start. The last step shifts back to
        // end at the row's end. RGB: where the store would pass the row's
        // end it stores its 2 cover bytes masked, and shifts back only once
        // its outputs reach the row's end (so it starts within the previous
        // step's outputs).
        let start = if RGB {
            if 2 * next + 64 > 2 * ws && next + cover >= ws {
                ws - cover
            } else {
                next
            }
        } else {
            next.min(ws - cover)
        };
        let fits = !RGB || 2 * start + 64 <= 2 * ws;
        let (start_at, pixel) = (start as isize, channels as isize);
        let center = ld(start_at);
        let triple = _mm512_add_epi16(
            _mm512_add_epi16(center, center),
            _mm512_add_epi16(center, rounding),
        );
        let even = _mm512_srli_epi16::<4>(_mm512_add_epi16(triple, ld(start_at - pixel)));
        let odd = _mm512_srli_epi16::<4>(_mm512_add_epi16(triple, ld(start_at + pixel)));
        let bytes = _mm512_shuffle_epi8(_mm512_packus_epi16(even, odd), interleave);
        let bytes = if RGB {
            _mm512_permutexvar_epi32(lane_bytes, bytes)
        } else {
            bytes
        };
        // SAFETY: `row` holds 2 ws bytes (the caller's contract): a full
        // store is taken where its 64 bytes end within 2 ws (`fits`, and
        // start <= ws - cover otherwise); the masked one writes 2 cover = 48
        // bytes from a start at most ws - cover.
        unsafe {
            if fits {
                _mm512_storeu_si512(row.add(2 * start) as *mut __m512i, bytes);
            } else {
                _mm512_mask_storeu_epi8(row.add(2 * start) as *mut i8, (1 << 48) - 1, bytes);
            }
        }
        next = start + cover;
    }
    ws
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::bilinear_up2::tests::{HEIGHTS, WIDTHS};
    use crate::exact::tests::{assert_equals_reference, size_grid};
    use crate::testing::CHANNELS;

    #[test]
    fn avx2_equals_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
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
    fn avx512_equals_scalar() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw"))
        {
            return;
        }
        let kernel = |src: &[u8], width, height, channels| {
            // SAFETY: AVX-512 F/BW are available; src holds width * height *
            // channels samples.
            unsafe { bilinear_up2_avx512(src, width, height, channels) }
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
        // After the pack, bytes 0..8 hold a lane's even outputs and 8..16 its
        // odd ones: the shuffle alternates them pixel by pixel.
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

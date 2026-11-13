//! The AVX2 kernel of the exact x2 bilinear reduction.
//!
//! # Safety
//! Every `unsafe fn` here requires the CPU features in its `#[target_feature]`
//! (its dispatch checks them at runtime) and the source length its caller
//! documents, which the kernels read against unchecked. Loads and stores are
//! justified against that length and their own buffers at their loop guards.

use super::{divide_rounded, horizontal_scalar, vertical_sums_scalar, window};
use crate::exact::x86::keep_even_pixels;
use crate::x86::prefetch_rows_after_window;

/// The AVX2 kernel: the vertical sums `r0 + r3 + 3 (r1 + r2)` in u16 (16
/// samples per register), then the 4-tap sums and their rounding at every sample, narrowed to bytes,
/// of which the even pixels are kept by byte shuffles (RGB: 6 of every 12
/// pixels per register, by two 16-byte loads starting at pixel pairs).
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
    let (ow, oh, ws) = (width / 2, height / 2, width * channels);
    debug_assert!(src.len() >= width * height * channels);
    let mut out_buf = crate::cpu::unwritten_output::<u8>(ow * oh * channels);
    // SAFETY: u8 has no invalid values; every sample is written below before
    // the vector is returned.
    let out = unsafe {
        std::slice::from_raw_parts_mut(out_buf.as_mut_ptr().cast::<u8>(), ow * oh * channels)
    };
    let mut sums32 = vec![0u32; ws];
    // The vertical sums with `channels` zero samples before and room after for the
    // offset loads (those lanes only feed the edge pixels, done apart).
    let mut sums = vec![0u16; channels + ws + 2 * LANES + 3 * channels];
    // The rounded 4-tap results at every sample, as bytes.
    let mut full = vec![0u8; ws + 2 * LANES];
    let zeros = vec![0u8; ws + LANES];
    let triple = |sum: __m256i| _mm256_add_epi16(sum, _mm256_slli_epi16::<1>(sum));
    let half = _mm256_set1_epi16(32);
    let (edge_bias, edge_magic) = (_mm256_set1_epi16(56), _mm256_set1_epi16(37450u16 as i16));
    for y in 0..oh {
        let (r0, wv) = window(y, height);
        let dst = &mut out[y * ow * channels..(y + 1) * ow * channels];
        if wv.len() < 3 || ws < 2 * LANES || ow < 3 {
            vertical_sums_scalar(&src[r0 * ws..], ws, wv, &mut sums32);
            horizontal_scalar(&sums32, channels, width, wv.iter().sum(), 0..ow, dst);
            continue;
        }
        let row = |row_index: isize| {
            if (0..height as isize).contains(&row_index) {
                // SAFETY: row_index < height, so the row starts within src
                // (width * height * channels samples, the caller's contract).
                unsafe { src.as_ptr().add(row_index as usize * ws) }
            } else {
                zeros.as_ptr()
            }
        };
        let first = 2 * y as isize - 1;
        let rows: [*const u8; 4] = std::array::from_fn(|tap| row(first + tap as isize));
        let zv: u32 = wv.iter().sum();
        // SAFETY: `sums` holds more than `channels` words.
        let sp = unsafe { sums.as_mut_ptr().add(channels) };
        let vblock = |x: usize| {
            if x.is_multiple_of(64) {
                prefetch_rows_after_window(src.as_ptr(), 2 * y + 3, ws, x);
            }
            let ld = |tap: usize| {
                // SAFETY: x <= ws - LANES at both calls, so the load's 16 bytes
                // lie within a src row of ws samples (the caller's contract)
                // or `zeros` (ws + LANES).
                let bytes = unsafe { _mm_loadu_si128(rows[tap].add(x) as *const __m128i) };
                _mm256_cvtepu8_epi16(bytes)
            };
            let window_sums = _mm256_add_epi16(
                _mm256_add_epi16(ld(0), ld(3)),
                triple(_mm256_add_epi16(ld(1), ld(2))),
            );
            // SAFETY: x + LANES <= ws: the store's 16 words lie within
            // `sums` past its leading `channels`.
            unsafe { _mm256_storeu_si256(sp.add(x) as *mut __m256i, window_sums) };
        };
        let mut x = 0;
        while x + LANES <= ws {
            vblock(x);
            x += LANES;
        }
        if x < ws {
            vblock(ws - LANES);
        }
        // The 4-tap sums at every sample, rounded, as bytes.
        let fp = full.as_mut_ptr();
        let round = |tap_sums: __m256i| {
            if zv == 8 {
                _mm256_srli_epi16::<6>(_mm256_add_epi16(tap_sums, half))
            } else {
                _mm256_srli_epi16::<6>(_mm256_mulhi_epu16(
                    _mm256_add_epi16(_mm256_add_epi16(tap_sums, tap_sums), edge_bias),
                    edge_magic,
                ))
            }
        };
        let hblock = |start: usize| {
            let taps = |block: usize| {
                // SAFETY: start <= ws - 2 LANES at both calls, so the loads at
                // -channels..=2 channels around sample start + block LANES lie
                // within `sums` (channels before the row, 2 LANES + 3 channels
                // after it).
                let ld = |off: isize| unsafe {
                    let block_ptr = sp.add(start + block * LANES) as *const i16;
                    _mm256_loadu_si256(block_ptr.offset(off) as *const __m256i)
                };
                let (cc, c2) = (channels as isize, 2 * channels as isize);
                round(_mm256_add_epi16(
                    _mm256_add_epi16(ld(-cc), ld(c2)),
                    triple(_mm256_add_epi16(ld(0), ld(cc))),
                ))
            };
            let bytes =
                _mm256_permute4x64_epi64::<0b11_01_10_00>(_mm256_packus_epi16(taps(0), taps(1)));
            // SAFETY: start + 2 LANES <= ws: the store lies within `full`.
            unsafe { _mm256_storeu_si256(fp.add(start) as *mut __m256i, bytes) };
        };
        let mut start = 0;
        while start + 2 * LANES <= ws {
            hblock(start);
            start += 2 * LANES;
        }
        if start < ws {
            hblock(ws - 2 * LANES);
        }
        // SAFETY: avx2 is enabled here; full[..ws] holds width channels >= 2 ow
        // channels = 2 dst.len() samples.
        unsafe { keep_even_pixels(&full[..ws], channels, dst) };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::bilinear_x2::bilinear_x2_scalar;
    use crate::exact::bilinear_x2::tests::{HEIGHTS, WIDTHS};
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

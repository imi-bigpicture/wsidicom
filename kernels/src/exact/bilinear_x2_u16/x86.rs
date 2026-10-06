//! The AVX2 kernels of the exact uint16 x2 bilinear reduction.
//!
//! # Safety
//! The kernels (`unsafe fn`) require AVX2 (their dispatch checks it at
//! runtime) and the source length and even size their caller documents, which
//! they read against unchecked; `store_pixel_quad` requires a row the caller
//! bounds. Loads and stores are justified against those bounds and the
//! kernels' own buffers at their loop guards. The helpers that only read
//! their own arguments are safe.

use crate::exact::bilinear_x2::{bilinear_x2_scalar, window};
use crate::exact::x86::{cover_steps, prefetch_next_rows};

/// The AVX2 kernel for one channel. A source pixel pair is one dword, so a
/// row's even and odd pixels come from each load by a mask and a shift. The
/// vertical sums `r0 + r3 + 3 (r1 + r2)` (a zero row for a clipped one) go
/// into even and odd rows with a zero pixel each side; the horizontal pass is
/// unit-stride at the output resolution: `odd[o - 1] + even[o + 1] +
/// 3 (even[o] + odd[o])`, then `(sum + 32) >> 6` where both windows are
/// whole, else an exact per-lane division (float64) by the windows' totals.
///
/// # Safety
/// AVX2 must be available; `src` holds `width * height` samples, `width` and
/// `height` even.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bilinear_x2_u16_gray_avx2(
    src: &[u16],
    width: usize,
    height: usize,
) -> Vec<u16> {
    use std::arch::x86_64::*;
    const LANES: usize = 8; // i32 lanes, output pixels per step
    let (ow, oh) = (width / 2, height / 2);
    if ow < LANES + 2 {
        return bilinear_x2_scalar(src, width, height, 1);
    }
    let mut out_buf = crate::cpu::unwritten_output::<u16>(ow * oh);
    let dp = out_buf.as_mut_ptr().cast::<u16>();
    // The even and odd pixels' vertical sums, a zero pixel each side.
    let mut even = vec![0i32; 1 + ow + 1];
    let mut odd = vec![0i32; 1 + ow + 1];
    // SAFETY: both vectors hold 1 + ow + 1 values.
    let (ep, op) = unsafe { (even.as_mut_ptr().add(1), odd.as_mut_ptr().add(1)) };
    let zeros = vec![0u16; width];
    let low_word = _mm256_set1_epi32(0xFFFF);
    let half = _mm256_set1_epi32(32);
    let triple = |sum: __m256i| _mm256_add_epi32(sum, _mm256_add_epi32(sum, sum));
    let col_totals: Vec<u32> = (0..ow)
        .map(|output| window(output, width).1.iter().sum())
        .collect();
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    for y in 0..oh {
        let (_, wv) = window(y, height);
        let zv: u32 = wv.iter().sum();
        let row = |row_index: isize| {
            if (0..height as isize).contains(&row_index) {
                // SAFETY: row_index < height, so the row starts within src
                // (width * height samples, the caller's contract).
                unsafe { src.as_ptr().add(row_index as usize * width) }
            } else {
                zeros.as_ptr()
            }
        };
        let first = 2 * y as isize - 1;
        let rows: [*const u16; 4] = std::array::from_fn(|tap| row(first + tap as isize));
        let vstep = |k0: usize| {
            prefetch_next_rows(src_bytes, 2 * y + 3, 2 * width, 4 * k0);
            let pairs = |tap: usize| {
                // SAFETY: k0 + LANES <= ow (cover_steps), so the load's 16
                // samples from 2 k0 lie within a src row of width = 2 ow
                // samples (the caller's contract) or `zeros` (width).
                let dwords = unsafe { _mm256_loadu_si256(rows[tap].add(2 * k0) as *const __m256i) };
                (
                    _mm256_and_si256(dwords, low_word),
                    _mm256_srli_epi32::<16>(dwords),
                )
            };
            let [(e0, o0), (e1, o1), (e2, o2), (e3, o3)] = std::array::from_fn(pairs);
            let sums = |row0: __m256i, row1: __m256i, row2: __m256i, row3: __m256i| {
                _mm256_add_epi32(
                    _mm256_add_epi32(row0, row3),
                    triple(_mm256_add_epi32(row1, row2)),
                )
            };
            // SAFETY: k0 + LANES <= ow (cover_steps): the stores lie within
            // even's and odd's ow values past their leading zero.
            unsafe {
                _mm256_storeu_si256(ep.add(k0) as *mut __m256i, sums(e0, e1, e2, e3));
                _mm256_storeu_si256(op.add(k0) as *mut __m256i, sums(o0, o1, o2, o3));
            }
        };
        cover_steps(ow, LANES, &vstep);
        // SAFETY: y < oh: the row starts within the ow * oh outputs.
        let row_out = unsafe { dp.add(y * ow) };
        let hstep = |o0: usize| {
            let ld = |parity: *const i32, offset: isize| {
                // SAFETY: o0 + LANES <= ow (cover_steps) and offset is -1..=1,
                // so the load's 8 values lie within even's and odd's zero
                // pixel before, ow values and zero pixel after.
                unsafe { _mm256_loadu_si256(parity.offset(o0 as isize + offset) as *const __m256i) }
            };
            let (ep, op) = (ep as *const i32, op as *const i32);
            let sums = _mm256_add_epi32(
                _mm256_add_epi32(ld(op, -1), ld(ep, 1)),
                triple(_mm256_add_epi32(ld(ep, 0), ld(op, 0))),
            );
            let whole = zv == 8 && o0 >= 1 && o0 + LANES < ow;
            let means = if whole {
                _mm256_srli_epi32::<6>(_mm256_add_epi32(sums, half))
            } else {
                divide_lanes(
                    sums,
                    std::array::from_fn(|lane| f64::from(zv * col_totals[o0 + lane])),
                )
            };
            // Each mean is at most 65535: the unsigned pack keeps it.
            let words =
                _mm256_permute4x64_epi64::<0b00_00_10_00>(_mm256_packus_epi32(means, means));
            // SAFETY: o0 + LANES <= ow (cover_steps): the store's 8 samples lie
            // within the row's ow outputs.
            unsafe {
                _mm_storeu_si128(
                    row_out.add(o0) as *mut __m128i,
                    _mm256_castsi256_si128(words),
                );
            }
        };
        cover_steps(ow, LANES, &hstep);
    }
    // SAFETY: every row's steps covered all of its outputs.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// The AVX2 kernel for `CHANNELS` (3 or 4) interleaved channels: as the
/// one-channel kernel, with each pixel in 4 i32 lanes (`CHANNELS` used) of
/// the even and odd rows. A pixel pair (at most 16 bytes) loads into a
/// 128-bit lane, one byte shuffle puts its even and odd pixels' samples into
/// words 0.. and 4.. (the rest zero), and unpacking against zero widens them;
/// the horizontal pass's means lose the unused lanes by one byte shuffle,
/// each 128-bit lane of 2 pixels stored as 16 bytes, the next store
/// overwriting the rest (the row's last as `CHANNELS` masked dwords).
///
/// # Safety
/// AVX2 must be available; `src` holds `width * height * CHANNELS` samples,
/// `width` and `height` even.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bilinear_x2_u16_hwc_avx2<const CHANNELS: usize>(
    src: &[u16],
    width: usize,
    height: usize,
) -> Vec<u16> {
    use std::arch::x86_64::*;
    const PIXEL: usize = 4; // i32 lanes per pixel in the even and odd rows
    const STEP: usize = 4; // output pixels per step
    let (ow, oh, ws) = (width / 2, height / 2, width * CHANNELS);
    if ow < STEP + 2 {
        return bilinear_x2_scalar(src, width, height, CHANNELS);
    }
    let mut out_buf = crate::cpu::unwritten_output::<u16>(ow * oh * CHANNELS);
    let dp = out_buf.as_mut_ptr().cast::<u16>();
    let mut even = vec![0i32; PIXEL + ow * PIXEL + PIXEL];
    let mut odd = vec![0i32; PIXEL + ow * PIXEL + PIXEL];
    // SAFETY: both vectors hold PIXEL + ow PIXEL + PIXEL values.
    let (ep, op) = unsafe { (even.as_mut_ptr().add(PIXEL), odd.as_mut_ptr().add(PIXEL)) };
    // Room for a 16-byte load at any pixel pair.
    let zeros = vec![0u16; ws + 8];
    let (split_pair, drop_unused, pair_dwords) = pixel_pair_shuffles::<CHANNELS>();
    let half = _mm256_set1_epi32(32);
    let zero = _mm256_setzero_si256();
    let triple = |sum: __m256i| _mm256_add_epi32(sum, _mm256_add_epi32(sum, sum));
    let col_totals: Vec<u32> = (0..ow)
        .map(|output| window(output, width).1.iter().sum())
        .collect();
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    for y in 0..oh {
        let (_, wv) = window(y, height);
        let zv: u32 = wv.iter().sum();
        let row = |row_index: isize| {
            if (0..height as isize).contains(&row_index) {
                // SAFETY: row_index < height, so the row starts within src
                // (width * height * CHANNELS samples, the caller's contract).
                unsafe { src.as_ptr().add(row_index as usize * ws) }
            } else {
                zeros.as_ptr()
            }
        };
        let first = 2 * y as isize - 1;
        let rows: [*const u16; 4] = std::array::from_fn(|tap| row(first + tap as isize));
        // Pixel pair `pair_index` of tap row `tap`, in a 128-bit lane: masked
        // where a 16-byte load would pass the row's end.
        let pair = |tap: usize, pair_index: usize| {
            // SAFETY: pair_index < ow (vstep's pairs from k0 <= ow - STEP), so
            // the pair starts within the row; the 16-byte load is taken where
            // its 8 samples end within the row's ws (the guard) or within
            // `zeros` (ws + 8), else the masked load reads the pair's
            // 2 CHANNELS samples only, which end within 2 ow CHANNELS = ws. A
            // src row is within src by the caller's contract.
            unsafe {
                let at = rows[tap].add(2 * pair_index * CHANNELS);
                if (2 * pair_index + 2) * CHANNELS + 8 - 2 * CHANNELS <= ws
                    || rows[tap] == zeros.as_ptr()
                {
                    _mm_loadu_si128(at as *const __m128i)
                } else {
                    _mm_maskload_epi32(at as *const i32, pair_dwords)
                }
            }
        };
        let vstep = |k0: usize| {
            prefetch_next_rows(src_bytes, 2 * y + 3, 2 * ws, 4 * k0 * CHANNELS);
            for pair_index in [k0, k0 + 2] {
                let widened = |tap: usize| {
                    let split_words = _mm256_shuffle_epi8(
                        _mm256_set_m128i(pair(tap, pair_index + 1), pair(tap, pair_index)),
                        split_pair,
                    );
                    (
                        _mm256_unpacklo_epi16(split_words, zero),
                        _mm256_unpackhi_epi16(split_words, zero),
                    )
                };
                let [(e0, o0), (e1, o1), (e2, o2), (e3, o3)] = std::array::from_fn(widened);
                let sums = |row0: __m256i, row1: __m256i, row2: __m256i, row3: __m256i| {
                    _mm256_add_epi32(
                        _mm256_add_epi32(row0, row3),
                        triple(_mm256_add_epi32(row1, row2)),
                    )
                };
                // SAFETY: pair_index + 2 <= ow (k0 + STEP <= ow, cover_steps):
                // the stores' 2 PIXEL values lie within even's and odd's
                // ow PIXEL past their leading zero pixel.
                unsafe {
                    _mm256_storeu_si256(
                        ep.add(pair_index * PIXEL) as *mut __m256i,
                        sums(e0, e1, e2, e3),
                    );
                    _mm256_storeu_si256(
                        op.add(pair_index * PIXEL) as *mut __m256i,
                        sums(o0, o1, o2, o3),
                    );
                }
            }
        };
        cover_steps(ow, STEP, &vstep);
        // SAFETY: y < oh: the row starts within the ow * oh * CHANNELS outputs.
        let row_out = unsafe { dp.add(y * ow * CHANNELS) };
        let hstep = |o0: usize| {
            let whole = zv == 8 && o0 >= 1 && o0 + STEP < ow;
            let means = |output: usize| {
                let ld = |parity: *const i32, at: isize| {
                    // SAFETY: output is o0 or o0 + 2 with o0 + STEP <= ow
                    // (cover_steps) and at is -1..=1, so the load's 2 pixels
                    // lie within even's and odd's zero pixel before, ow pixels
                    // and zero pixel after.
                    unsafe {
                        _mm256_loadu_si256(parity.offset((output as isize + at) * PIXEL as isize)
                            as *const __m256i)
                    }
                };
                let (ep, op) = (ep as *const i32, op as *const i32);
                let sums = _mm256_add_epi32(
                    _mm256_add_epi32(ld(op, -1), ld(ep, 1)),
                    triple(_mm256_add_epi32(ld(ep, 0), ld(op, 0))),
                );
                if whole {
                    return _mm256_srli_epi32::<6>(_mm256_add_epi32(sums, half));
                }
                divide_lanes(
                    sums,
                    std::array::from_fn(|lane| f64::from(zv * col_totals[output + lane / PIXEL])),
                )
            };
            // SAFETY: avx2 is enabled here; row_out holds the row's
            // ow * CHANNELS outputs and o0 + 4 <= ow (cover_steps).
            unsafe {
                store_pixel_quad::<CHANNELS>(
                    row_out,
                    o0,
                    ow,
                    means(o0),
                    means(o0 + 2),
                    drop_unused,
                    pair_dwords,
                );
            }
        };
        cover_steps(ow, STEP, &hstep);
    }
    // SAFETY: every row's steps covered all of its outputs.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// For `CHANNELS` (3 or 4) interleaved channels: the byte shuffle putting a
/// pixel pair's 2 CHANNELS words (in a 128-bit lane) at [first's CHANNELS,
/// 0.., second's CHANNELS, 0..]; the one taking two pixels' 4-word groups
/// back to their 2 CHANNELS samples; and the mask of a pixel pair's
/// CHANNELS dwords (for loads and stores at a row's end).
#[target_feature(enable = "avx2")]
pub(crate) fn pixel_pair_shuffles<const CHANNELS: usize>() -> (
    std::arch::x86_64::__m256i,
    std::arch::x86_64::__m256i,
    std::arch::x86_64::__m128i,
) {
    use std::arch::x86_64::*;
    const PIXEL: usize = 4;
    let byte_pick = |word_of: &dyn Fn(usize) -> Option<usize>| -> __m256i {
        let bytes: [i8; 16] = std::array::from_fn(|byte| {
            word_of(byte / 2).map_or(-1, |word| (2 * word + byte % 2) as i8)
        });
        // SAFETY: the load reads the 16 bytes of `bytes`.
        _mm256_broadcastsi128_si256(unsafe { _mm_loadu_si128(bytes.as_ptr() as *const __m128i) })
    };
    let split_pair = byte_pick(&|word| match word {
        word if word < CHANNELS => Some(word),
        word if (PIXEL..PIXEL + CHANNELS).contains(&word) => Some(CHANNELS + word - PIXEL),
        _ => None,
    });
    let drop_unused = byte_pick(&|word| match word {
        word if word < CHANNELS => Some(word),
        word if (CHANNELS..2 * CHANNELS).contains(&word) => Some(PIXEL + word - CHANNELS),
        _ => None,
    });
    let pair_dwords = _mm_setr_epi32(-1, -1, -1, if CHANNELS > 3 { -1 } else { 0 });
    (split_pair, drop_unused, pair_dwords)
}

/// Stores output pixels o0..o0 + 4 of a `CHANNELS`-channel row from their means
/// (pixels o0 | o0 + 1 in `first`, o0 + 2 | o0 + 3 in `second`, 4 lanes
/// each): packed per lane as [o0, o0 + 2 | o0 + 1, o0 + 3], put back in
/// order, the unused lanes dropped; each 128-bit lane stored as 16 bytes, the
/// next store overwriting the rest, the row's last (o0 + 4 = `ow`) as
/// `CHANNELS` masked dwords.
///
/// # Safety
/// AVX2 must be available; `CHANNELS` is 3 or 4, `row_out` is valid for
/// writes of `ow * CHANNELS` samples and `o0 + 4 <= ow`.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn store_pixel_quad<const CHANNELS: usize>(
    row_out: *mut u16,
    o0: usize,
    ow: usize,
    first: std::arch::x86_64::__m256i,
    second: std::arch::x86_64::__m256i,
    drop_unused: std::arch::x86_64::__m256i,
    pair_dwords: std::arch::x86_64::__m128i,
) {
    use std::arch::x86_64::*;
    let packed = _mm256_packus_epi32(first, second);
    let samples = _mm256_shuffle_epi8(
        _mm256_permute4x64_epi64::<0b11_01_10_00>(packed),
        drop_unused,
    );
    // SAFETY: o0 + 4 <= ow (the caller's contract): the first store's 8
    // samples from o0 CHANNELS end within (o0 + 4) CHANNELS <= ow CHANNELS; the
    // second, from (o0 + 2) CHANNELS, writes the row's last 2 CHANNELS samples
    // masked when o0 + 4 = ow, else 8 samples ending within (o0 + 5) CHANNELS.
    unsafe {
        _mm_storeu_si128(
            row_out.add(o0 * CHANNELS) as *mut __m128i,
            _mm256_castsi256_si128(samples),
        );
        let at = row_out.add((o0 + 2) * CHANNELS);
        if o0 + 4 == ow {
            _mm_maskstore_epi32(
                at as *mut i32,
                pair_dwords,
                _mm256_extracti128_si256::<1>(samples),
            );
        } else {
            _mm_storeu_si128(at as *mut __m128i, _mm256_extracti128_si256::<1>(samples));
        }
    }
}

/// floor((2 sum + d) / (2 d)) per lane of `sums`, `d = denominators[lane]`
/// the lane's denominator: exact in float64 (sums below 2^23).
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) fn divide_lanes(
    sums: std::arch::x86_64::__m256i,
    denominators: [f64; 8],
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let divide = |sums_half: __m128i, denominators: &[f64]| {
        // SAFETY: both calls pass 4 values, which the load reads.
        let divisors = unsafe { _mm256_loadu_pd(denominators.as_ptr()) };
        let x = _mm256_cvtepi32_pd(sums_half);
        let quotient = _mm256_div_pd(
            _mm256_add_pd(_mm256_add_pd(x, x), divisors),
            _mm256_add_pd(divisors, divisors),
        );
        _mm256_cvtpd_epi32(_mm256_floor_pd(quotient))
    };
    let lo = divide(_mm256_castsi256_si128(sums), &denominators[..4]);
    let hi = divide(_mm256_extracti128_si256::<1>(sums), &denominators[4..]);
    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::bilinear_x2_u16::tests::{HEIGHTS, WIDTHS};
    use crate::exact::tests::{assert_equals_reference, size_grid};
    use crate::testing::CHANNELS;

    #[test]
    fn avx2_equals_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let kernel = |src: &[u16], width, height, channels| {
            // SAFETY: AVX2 is available; src holds width * height * channels
            // samples, width and height even (WIDTHS and HEIGHTS).
            unsafe {
                match channels {
                    1 => bilinear_x2_u16_gray_avx2(src, width, height),
                    3 => bilinear_x2_u16_hwc_avx2::<3>(src, width, height),
                    _ => bilinear_x2_u16_hwc_avx2::<4>(src, width, height),
                }
            }
        };
        assert_equals_reference(
            size_grid(&WIDTHS, &HEIGHTS),
            &CHANNELS,
            u16::MAX,
            &kernel,
            bilinear_x2_scalar,
        );
    }
}

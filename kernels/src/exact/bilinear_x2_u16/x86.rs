//! The AVX2 kernels of the exact uint16 x2 bilinear reduction.
//!
//! # Safety
//! Every `unsafe fn` here requires AVX2 and the bounds its doc states; loads
//! and stores are justified against those and the kernels' own buffers at
//! their loop guards.

use crate::exact::bilinear_x2::{bilinear_x2_scalar, window};
use crate::exact::x86::cover_steps;
use crate::sample::RGB;
use crate::x86::prefetch_rows_after_window;

/// The gray kernel. A source pixel pair is one dword, so each load splits
/// into even and odd pixels by a mask and a shift. Their vertical sums
/// `row0 + row3 + 3 (row1 + row2)` (a zero row for a clipped one) go into
/// even and odd rows; output o is `odd[o - 1] + even[o + 1] + 3 (even[o] +
/// odd[o])`, `(sum + 32) >> 6` where both windows are whole, else divided by
/// `divide_lanes`.
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
    const LANES: usize = 8;
    let (out_width, out_height) = (width / 2, height / 2);
    if out_width < LANES + 2 {
        return bilinear_x2_scalar(src, width, height, 1);
    }
    let mut out_buf = crate::cpu::unwritten_output::<u16>(out_width * out_height);
    let dst_ptr = out_buf.as_mut_ptr().cast::<u16>();
    // A zero pixel each side.
    let mut even = vec![0i32; 1 + out_width + 1];
    let mut odd = vec![0i32; 1 + out_width + 1];
    // SAFETY: both vectors hold 1 + out_width + 1 values.
    let (even_ptr, odd_ptr) = unsafe { (even.as_mut_ptr().add(1), odd.as_mut_ptr().add(1)) };
    let zeros = vec![0u16; width];
    let low_word = _mm256_set1_epi32(0xFFFF);
    let half = _mm256_set1_epi32(32);
    let triple = |sum: __m256i| _mm256_add_epi32(sum, _mm256_add_epi32(sum, sum));
    let col_totals: Vec<u32> = (0..out_width)
        .map(|output| window(output, width).1.iter().sum())
        .collect();
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    for y in 0..out_height {
        let (_, vertical_weights) = window(y, height);
        let vertical_total: u32 = vertical_weights.iter().sum();
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
        let vertical_step = |first_output: usize| {
            prefetch_rows_after_window(src_bytes.as_ptr(), 2 * y + 3, 2 * width, 4 * first_output);
            let pairs = |tap: usize| {
                // SAFETY: first_output + LANES <= out_width (cover_steps), so
                // the load's 16 samples from 2 first_output lie within a src
                // row of width = 2 out_width samples (the caller's contract) or
                // `zeros` (width).
                let dwords = unsafe {
                    _mm256_loadu_si256(rows[tap].add(2 * first_output) as *const __m256i)
                };
                (
                    _mm256_and_si256(dwords, low_word),
                    _mm256_srli_epi32::<16>(dwords),
                )
            };
            let [(even0, odd0), (even1, odd1), (even2, odd2), (even3, odd3)] =
                std::array::from_fn(pairs);
            let sums = |row0: __m256i, row1: __m256i, row2: __m256i, row3: __m256i| {
                _mm256_add_epi32(
                    _mm256_add_epi32(row0, row3),
                    triple(_mm256_add_epi32(row1, row2)),
                )
            };
            // SAFETY: first_output + LANES <= out_width (cover_steps): the
            // stores lie within even's and odd's out_width values past their
            // leading zero.
            unsafe {
                _mm256_storeu_si256(
                    even_ptr.add(first_output) as *mut __m256i,
                    sums(even0, even1, even2, even3),
                );
                _mm256_storeu_si256(
                    odd_ptr.add(first_output) as *mut __m256i,
                    sums(odd0, odd1, odd2, odd3),
                );
            }
        };
        cover_steps(out_width, LANES, &vertical_step);
        // SAFETY: y < out_height: the row starts within the out_width *
        // out_height outputs.
        let row_out = unsafe { dst_ptr.add(y * out_width) };
        let horizontal_step = |first_output: usize| {
            let load = |parity: *const i32, offset: isize| {
                // SAFETY: first_output + LANES <= out_width (cover_steps) and
                // offset is -1..=1, so the load's 8 values lie within even's
                // and odd's zero pixel before, out_width values and zero pixel
                // after.
                unsafe {
                    _mm256_loadu_si256(
                        parity.offset(first_output as isize + offset) as *const __m256i
                    )
                }
            };
            let (even, odd) = (even_ptr as *const i32, odd_ptr as *const i32);
            let sums = _mm256_add_epi32(
                _mm256_add_epi32(load(odd, -1), load(even, 1)),
                triple(_mm256_add_epi32(load(even, 0), load(odd, 0))),
            );
            let whole =
                vertical_total == 8 && first_output >= 1 && first_output + LANES < out_width;
            let means = if whole {
                _mm256_srli_epi32::<6>(_mm256_add_epi32(sums, half))
            } else {
                divide_lanes(
                    sums,
                    std::array::from_fn(|lane| {
                        f64::from(vertical_total * col_totals[first_output + lane])
                    }),
                )
            };
            let words =
                _mm256_permute4x64_epi64::<0b00_00_10_00>(_mm256_packus_epi32(means, means));
            // SAFETY: first_output + LANES <= out_width (cover_steps): the
            // store's 8 samples lie within the row's out_width outputs.
            unsafe {
                _mm_storeu_si128(
                    row_out.add(first_output) as *mut __m128i,
                    _mm256_castsi256_si128(words),
                );
            }
        };
        cover_steps(out_width, LANES, &horizontal_step);
    }
    // SAFETY: every row's steps covered all of its outputs.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// The RGB kernel: as the gray kernel, with each pixel in 4 i32 lanes (3
/// used) of the even and odd rows. A pixel pair loads into a 128-bit lane, a
/// byte shuffle puts its even and odd pixels' samples into words 0.. and 4..
/// (the rest zero), and unpacking against zero widens them.
///
/// # Safety
/// AVX2 must be available; `src` holds `width * height * 3` samples,
/// `width` and `height` even.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bilinear_x2_u16_hwc_avx2(
    src: &[u16],
    width: usize,
    height: usize,
) -> Vec<u16> {
    use std::arch::x86_64::*;
    const CHANNELS: usize = RGB;
    const PIXEL: usize = RGBX_PIXEL_LANES;
    const STEP: usize = 4;
    let (out_width, out_height, row_samples) = (width / 2, height / 2, width * CHANNELS);
    if out_width < STEP + 2 {
        return bilinear_x2_scalar(src, width, height, CHANNELS);
    }
    let mut out_buf = crate::cpu::unwritten_output::<u16>(out_width * out_height * CHANNELS);
    let dst_ptr = out_buf.as_mut_ptr().cast::<u16>();
    let mut even = vec![0i32; PIXEL + out_width * PIXEL + PIXEL];
    let mut odd = vec![0i32; PIXEL + out_width * PIXEL + PIXEL];
    // SAFETY: both vectors hold PIXEL + out_width PIXEL + PIXEL values.
    let (even_ptr, odd_ptr) =
        unsafe { (even.as_mut_ptr().add(PIXEL), odd.as_mut_ptr().add(PIXEL)) };
    // Room for a 16-byte load at any pixel pair.
    let zeros = vec![0u16; row_samples + 8];
    let (split_pair, drop_unused, pair_dwords) = pixel_pair_shuffles();
    let half = _mm256_set1_epi32(32);
    let zero = _mm256_setzero_si256();
    let triple = |sum: __m256i| _mm256_add_epi32(sum, _mm256_add_epi32(sum, sum));
    let col_totals: Vec<u32> = (0..out_width)
        .map(|output| window(output, width).1.iter().sum())
        .collect();
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    for y in 0..out_height {
        let (_, vertical_weights) = window(y, height);
        let vertical_total: u32 = vertical_weights.iter().sum();
        let row = |row_index: isize| {
            if (0..height as isize).contains(&row_index) {
                // SAFETY: row_index < height, so the row starts within src
                // (width * height * CHANNELS samples, the caller's contract).
                unsafe { src.as_ptr().add(row_index as usize * row_samples) }
            } else {
                zeros.as_ptr()
            }
        };
        let first = 2 * y as isize - 1;
        let rows: [*const u16; 4] = std::array::from_fn(|tap| row(first + tap as isize));
        // Masked where a 16-byte load would pass the row's end.
        let pair = |tap: usize, pair_index: usize| {
            // SAFETY: pair_index < out_width (vertical_step's pairs from
            // first_output <= out_width - STEP), so the pair starts within the
            // row; the 16-byte load is taken where its 8 samples end within the
            // row's row_samples (the guard) or within `zeros` (row_samples +
            // 8), else the masked load reads the pair's 2 CHANNELS samples
            // only, which end within 2 out_width CHANNELS = row_samples. A src
            // row is within src by the caller's contract.
            unsafe {
                let at = rows[tap].add(2 * pair_index * CHANNELS);
                if (2 * pair_index + 2) * CHANNELS + 8 - 2 * CHANNELS <= row_samples
                    || rows[tap] == zeros.as_ptr()
                {
                    _mm_loadu_si128(at as *const __m128i)
                } else {
                    _mm_maskload_epi32(at as *const i32, pair_dwords)
                }
            }
        };
        let vertical_step = |first_output: usize| {
            prefetch_rows_after_window(
                src_bytes.as_ptr(),
                2 * y + 3,
                2 * row_samples,
                4 * first_output * CHANNELS,
            );
            for pair_index in [first_output, first_output + 2] {
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
                let [(even0, odd0), (even1, odd1), (even2, odd2), (even3, odd3)] =
                    std::array::from_fn(widened);
                let sums = |row0: __m256i, row1: __m256i, row2: __m256i, row3: __m256i| {
                    _mm256_add_epi32(
                        _mm256_add_epi32(row0, row3),
                        triple(_mm256_add_epi32(row1, row2)),
                    )
                };
                // SAFETY: pair_index + 2 <= out_width (first_output + STEP <=
                // out_width, cover_steps): the stores' 2 PIXEL values lie
                // within even's and odd's out_width PIXEL past their leading
                // zero pixel.
                unsafe {
                    _mm256_storeu_si256(
                        even_ptr.add(pair_index * PIXEL) as *mut __m256i,
                        sums(even0, even1, even2, even3),
                    );
                    _mm256_storeu_si256(
                        odd_ptr.add(pair_index * PIXEL) as *mut __m256i,
                        sums(odd0, odd1, odd2, odd3),
                    );
                }
            }
        };
        cover_steps(out_width, STEP, &vertical_step);
        // SAFETY: y < out_height: the row starts within the out_width *
        // out_height * CHANNELS outputs.
        let row_out = unsafe { dst_ptr.add(y * out_width * CHANNELS) };
        let horizontal_step = |first_output: usize| {
            let whole = vertical_total == 8 && first_output >= 1 && first_output + STEP < out_width;
            let means = |output: usize| {
                let load = |parity: *const i32, offset: isize| {
                    // SAFETY: output is first_output or first_output + 2 with
                    // first_output + STEP <= out_width (cover_steps) and offset
                    // is -1..=1, so the load's 2 pixels lie within even's and
                    // odd's zero pixel before, out_width pixels and zero pixel
                    // after.
                    unsafe {
                        _mm256_loadu_si256(
                            parity.offset((output as isize + offset) * PIXEL as isize)
                                as *const __m256i,
                        )
                    }
                };
                let (even, odd) = (even_ptr as *const i32, odd_ptr as *const i32);
                let sums = _mm256_add_epi32(
                    _mm256_add_epi32(load(odd, -1), load(even, 1)),
                    triple(_mm256_add_epi32(load(even, 0), load(odd, 0))),
                );
                if whole {
                    return _mm256_srli_epi32::<6>(_mm256_add_epi32(sums, half));
                }
                divide_lanes(
                    sums,
                    std::array::from_fn(|lane| {
                        f64::from(vertical_total * col_totals[output + lane / PIXEL])
                    }),
                )
            };
            // SAFETY: avx2 is enabled here; row_out holds the row's out_width *
            // CHANNELS outputs and first_output + 4 <= out_width (cover_steps).
            unsafe {
                store_pixel_quad(
                    row_out,
                    first_output,
                    out_width,
                    means(first_output),
                    means(first_output + 2),
                    drop_unused,
                    pair_dwords,
                );
            }
        };
        cover_steps(out_width, STEP, &horizontal_step);
    }
    // SAFETY: every row's steps covered all of its outputs.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// i32 lanes per RGB pixel in the even and odd rows.
const RGBX_PIXEL_LANES: usize = 4;

/// Per 128-bit lane: the byte shuffle putting a pixel pair's 6 words at
/// [first's 3, 0, second's 3, 0]; the one taking two pixels' 4-word groups
/// back to their 6 samples; and the mask of a pixel pair's 3 dwords.
#[target_feature(enable = "avx2")]
pub(crate) fn pixel_pair_shuffles() -> (
    std::arch::x86_64::__m256i,
    std::arch::x86_64::__m256i,
    std::arch::x86_64::__m128i,
) {
    use std::arch::x86_64::*;
    const CHANNELS: usize = RGB;
    const PIXEL: usize = RGBX_PIXEL_LANES;
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
    let pair_dwords = _mm_setr_epi32(-1, -1, -1, 0);
    (split_pair, drop_unused, pair_dwords)
}

/// Stores output pixels `start..start + 4` of an RGB row from their means
/// (pixels start | start + 1 in `first`, start + 2 | start + 3 in `second`):
/// the pack leaves [start, start + 2 | start + 1, start + 3], the permute puts
/// them in order, the shuffle drops the unused lanes. Each 128-bit lane is
/// stored as 16 bytes, the next store overwriting the rest; the row's last as
/// 3 masked dwords.
///
/// # Safety
/// AVX2 must be available, `row_out` is valid for writes of `out_width * 3`
/// samples and `start + 4 <= out_width`.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn store_pixel_quad(
    row_out: *mut u16,
    start: usize,
    out_width: usize,
    first: std::arch::x86_64::__m256i,
    second: std::arch::x86_64::__m256i,
    drop_unused: std::arch::x86_64::__m256i,
    pair_dwords: std::arch::x86_64::__m128i,
) {
    use std::arch::x86_64::*;
    const CHANNELS: usize = RGB;
    let packed = _mm256_packus_epi32(first, second);
    let samples = _mm256_shuffle_epi8(
        _mm256_permute4x64_epi64::<0b11_01_10_00>(packed),
        drop_unused,
    );
    // SAFETY: start + 4 <= out_width (the caller's contract): the first store's
    // 8 samples from start CHANNELS end within (start + 4) CHANNELS <=
    // out_width CHANNELS; the second, from (start + 2) CHANNELS, writes the
    // row's last 2 CHANNELS samples masked when start + 4 = out_width, else 8
    // samples ending within (start + 5) CHANNELS.
    unsafe {
        _mm_storeu_si128(
            row_out.add(start * CHANNELS) as *mut __m128i,
            _mm256_castsi256_si128(samples),
        );
        let at = row_out.add((start + 2) * CHANNELS);
        if start + 4 == out_width {
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

/// `floor((2 sum + d) / (2 d))` per lane, `d` the lane's denominator: exact
/// in float64 (sums below 2^23).
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
        let sums = _mm256_cvtepi32_pd(sums_half);
        let quotient = _mm256_div_pd(
            _mm256_add_pd(_mm256_add_pd(sums, sums), divisors),
            _mm256_add_pd(divisors, divisors),
        );
        _mm256_cvtpd_epi32(_mm256_floor_pd(quotient))
    };
    let low = divide(_mm256_castsi256_si128(sums), &denominators[..4]);
    let high = divide(_mm256_extracti128_si256::<1>(sums), &denominators[4..]);
    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(low), high)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::bilinear_x2_u16::tests::{HEIGHTS, WIDTHS};
    use crate::exact::tests::assert_equals_reference;
    use crate::testing::{CHANNELS, size_grid};

    #[test]
    fn avx2_equals_scalar() {
        if !crate::testing::avx2_detected() {
            return;
        }
        let kernel = |src: &[u16], width, height, channels| {
            // SAFETY: AVX2 is available; src holds width * height * channels
            // samples, width and height even (WIDTHS and HEIGHTS).
            unsafe {
                match channels {
                    1 => bilinear_x2_u16_gray_avx2(src, width, height),
                    _ => bilinear_x2_u16_hwc_avx2(src, width, height),
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

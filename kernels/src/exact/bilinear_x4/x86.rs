//! The AVX2 and AVX-512 kernels of the exact x4 bilinear reduction.
//!
//! # Safety
//! Every `unsafe fn` here requires the CPU features in its `#[target_feature]`
//! (its dispatch checks them at runtime) and the source length and size its
//! caller documents, which the kernels read against unchecked. Loads and
//! stores are justified against those bounds and the frame's buffers at their
//! loop guards.

use super::{bilinear_x4_scalar, window};
use crate::exact::x86::{cover_steps, prefetch_next_rows};

/// The SIMD kernels' frame, `PC` samples per pixel in the phase arrays (1,
/// 2 or 4; RGB is read as RGBX, `PC` 4, and packed back per output row). Per
/// output row y: `vertical(rows, y, phases)` splits the window's weighted sum
/// of its 8 `rows` (a zero row for a clipped one) by pixel phase (x mod 4)
/// into the four phase arrays, which have a zero pixel before and after;
/// `horizontal(phases, zv, col_totals, dst)` writes the row's outputs from
/// them (`zv` the row's window total, `col_totals` each column's). Inlined,
/// so both compile with the caller's target features.
///
/// # Safety
/// `src` holds `width * height * channels` samples (`channels` = 3 for `RGB`,
/// else `PC`), `width` and `height` multiples of 4; AVX2 is available for
/// `RGB`; `vertical` writes the row's phase samples only, and `horizontal`
/// the row's `width / 4 * PC` bytes.
#[inline(always)]
pub(crate) unsafe fn x4_frame<const PC: usize, const RGB: bool>(
    src: &[u8],
    (width, height): (usize, usize),
    vertical: impl Fn(&[*const u8; 8], usize, &[*mut u16; 4]),
    horizontal: impl Fn(&[*mut u16; 4], u32, &[u32], *mut u8),
) -> Vec<u8> {
    let channels = if RGB { 3 } else { PC };
    let (ow, oh, ws) = (width / 4, height / 4, width * channels);
    debug_assert!(src.len() >= width * height * channels);
    let row_samples = ow * PC;
    let mut out_buf = crate::cpu::unwritten_output::<u8>(ow * oh * channels);
    let dp = out_buf.as_mut_ptr().cast::<u8>();
    let phase_len = PC + row_samples + PC;
    let mut phases = vec![0u16; 4 * phase_len];
    let pp: [*mut u16; 4] = std::array::from_fn(|phase| {
        // SAFETY: phase < 4: the array starts within the 4 phase_len values.
        unsafe { phases.as_mut_ptr().add(phase * phase_len + PC) }
    });
    // RGB: an output row as RGBX before it is packed.
    let mut rgbx_row = vec![0u8; if RGB { row_samples } else { 0 }];
    let zeros = vec![0u8; ws + 64];
    let col_totals: Vec<u32> = (0..ow)
        .map(|output| window(output, width).1.iter().sum())
        .collect();
    for y in 0..oh {
        let (_, wv) = window(y, height);
        let row = |row_index: isize| {
            if (0..height as isize).contains(&row_index) {
                // SAFETY: row_index < height, so the row starts within src
                // (width * height * channels samples, the caller's contract).
                unsafe { src.as_ptr().add(row_index as usize * ws) }
            } else {
                zeros.as_ptr()
            }
        };
        let first = 4 * y as isize - 2;
        let rows: [*const u8; 8] = std::array::from_fn(|tap| row(first + tap as isize));
        vertical(&rows, y, &pp);
        let dst = if RGB {
            rgbx_row.as_mut_ptr()
        } else {
            // SAFETY: y < oh: the row starts within the ow * oh * PC outputs.
            unsafe { dp.add(y * row_samples) }
        };
        horizontal(&pp, wv.iter().sum(), &col_totals, dst);
        if RGB {
            // SAFETY: y < oh: the row's ow * 3 bytes lie within the output
            // (u8 has no invalid values; the pack only writes them); AVX2 is
            // available for RGB (the caller's contract), and `rgbx_row`
            // holds the row's ow * 4 bytes.
            unsafe {
                let out_row = std::slice::from_raw_parts_mut(dp.add(y * ow * 3), ow * 3);
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
/// 0..8 and the high ones 8..16, already one pixel per phase for 4 channels
/// and a word shuffle and an unpack away for 1; one 64-bit permute then
/// joins the lanes.
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
pub(crate) unsafe fn bilinear_x4_avx2<const PC: usize, const RGB: bool>(
    src: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    let channels = if RGB { 3 } else { PC };
    let (ow, ws) = (width / 4, width * channels);
    // Outputs per vertical step (32 phase-layout samples, 8 per phase) and
    // per horizontal step (16).
    let (vstep_outputs, hstep_outputs) = (8 / PC, 16 / PC);
    if ow < hstep_outputs {
        return bilinear_x4_scalar(src, width, height, channels);
    }
    let near_w = _mm256_set1_epi16((5 << 8) | 7);
    let far_w = _mm256_set1_epi16((1 << 8) | 3);
    let (madd_near, madd_far) = (
        _mm256_set1_epi32((5 << 16) | 7),
        _mm256_set1_epi32((1 << 16) | 3),
    );
    let half = _mm256_set1_epi32(1 << 9);
    // 1 channel: a lane's words 0..8 as (0, 4), (1, 5), (2, 6), (3, 7).
    let pair_phases = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
    ));
    // RGB: 4 pixels' 12 bytes per lane (dwords 0..3 and 3..6), widened to RGBX.
    let rgb_bytes = _mm_setr_epi32(-1, -1, -1, 0);
    let rgb_halves = _mm256_setr_epi32(0, 1, 2, 0, 3, 4, 5, 0);
    let to_rgbx = _mm_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1);
    let to_rgbx_ymm = _mm256_broadcastsi128_si256(to_rgbx);
    let vertical = |rows: &[*const u8; 8], y: usize, pp: &[*mut u16; 4]| {
        // 32 phase-layout bytes from pixel p0 of tap row `tap`.
        let load = |tap: usize, p0: usize| {
            // SAFETY: p0 = 4 k0 with k0 + 8 / PC <= ow (cover_steps), so the
            // plain load's 32 bytes from p0 PC end within 4 ow PC = ws, the
            // RGB load is taken where its 32 bytes end within ws (the guard),
            // and the masked RGB loads read the 12 bytes of pixels p0 and
            // p0 + 4, ending within 3 (p0 + 8) <= ws; each row is a src row
            // (the frame's contract) or `zeros` (ws + 64).
            unsafe {
                if !RGB {
                    _mm256_loadu_si256(rows[tap].add(p0 * PC) as *const __m256i)
                } else if 3 * p0 + 32 <= ws {
                    let bytes = _mm256_loadu_si256(rows[tap].add(3 * p0) as *const __m256i);
                    _mm256_shuffle_epi8(_mm256_permutevar8x32_epi32(bytes, rgb_halves), to_rgbx_ymm)
                } else {
                    let rgbx = |pixel: usize| {
                        _mm_shuffle_epi8(
                            _mm_maskload_epi32(rows[tap].add(3 * pixel) as *const i32, rgb_bytes),
                            to_rgbx,
                        )
                    };
                    _mm256_set_m128i(rgbx(p0 + 4), rgbx(p0))
                }
            }
        };
        let vstep = |k0: usize| {
            let p0 = 4 * k0;
            prefetch_next_rows(src, 4 * y + 6, ws, p0 * channels);
            prefetch_next_rows(src, 4 * y + 8, ws, p0 * channels);
            let loaded: [__m256i; 8] = std::array::from_fn(|tap| load(tap, p0));
            let sums = |lo: bool| {
                let pair = |low_tap: usize, high_tap: usize| {
                    if lo {
                        _mm256_unpacklo_epi8(loaded[low_tap], loaded[high_tap])
                    } else {
                        _mm256_unpackhi_epi8(loaded[low_tap], loaded[high_tap])
                    }
                };
                let near = _mm256_add_epi16(
                    _mm256_maddubs_epi16(pair(3, 2), near_w),
                    _mm256_maddubs_epi16(pair(4, 5), near_w),
                );
                let far = _mm256_add_epi16(
                    _mm256_maddubs_epi16(pair(1, 0), far_w),
                    _mm256_maddubs_epi16(pair(6, 7), far_w),
                );
                _mm256_add_epi16(near, far)
            };
            let (low, high) = (sums(true), sums(false));
            // Per lane: phases 0 and 1 (x), 2 and 3 (y), 4 words each.
            let (x, y) = if PC == 4 {
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
            let (x, y) = (
                _mm256_permute4x64_epi64::<0b11_01_10_00>(x),
                _mm256_permute4x64_epi64::<0b11_01_10_00>(y),
            );
            let at = k0 * PC;
            // SAFETY: k0 + 8 / PC <= ow (cover_steps): each store's 8 words
            // from k0 PC end within the phase array's ow PC (the frame's
            // `pp`).
            unsafe {
                _mm_storeu_si128(pp[0].add(at) as *mut __m128i, _mm256_castsi256_si128(x));
                _mm_storeu_si128(
                    pp[1].add(at) as *mut __m128i,
                    _mm256_extracti128_si256::<1>(x),
                );
                _mm_storeu_si128(pp[2].add(at) as *mut __m128i, _mm256_castsi256_si128(y));
                _mm_storeu_si128(
                    pp[3].add(at) as *mut __m128i,
                    _mm256_extracti128_si256::<1>(y),
                );
            }
        };
        cover_steps(ow, vstep_outputs, &vstep);
    };
    let horizontal = |pp: &[*mut u16; 4], zv: u32, col_totals: &[u32], dst: *mut u8| {
        let hstep = |o0: usize| {
            let at = o0 * PC;
            let ld = |phase: usize, offset: isize| {
                // SAFETY: o0 + 16 / PC <= ow (cover_steps) and offset is
                // -1..=1, so the load's 16 words lie within the phase array's
                // zero pixel before, ow PC words and zero pixel after.
                unsafe {
                    _mm256_loadu_si256(
                        pp[phase].offset(at as isize + offset * PC as isize) as *const __m256i
                    )
                }
            };
            let pair = |left: __m256i, right: __m256i| _mm256_add_epi16(left, right);
            let (inner, outer) = (pair(ld(1, 0), ld(2, 0)), pair(ld(0, 0), ld(3, 0)));
            let (next, last) = (pair(ld(3, -1), ld(0, 1)), pair(ld(2, -1), ld(1, 1)));
            let madd2 = |lo: bool| {
                let unpack = |low_words: __m256i, high_words: __m256i| {
                    if lo {
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
            let (lo, hi) = (madd2(true), madd2(false));
            let whole = zv == 32 && o0 >= 1 && o0 + hstep_outputs < ow;
            let round = |sums: __m256i, first: usize| {
                if whole {
                    return _mm256_srli_epi32::<10>(_mm256_add_epi32(sums, half));
                }
                // floor((2 sum + d) / (2 d)) per lane, d = zv * the column total.
                let denominators: [f32; 8] = std::array::from_fn(|lane| {
                    let sample = first + lane % 4 + 8 * (lane / 4);
                    (zv * col_totals[(o0 + sample / PC).min(ow - 1)]) as f32
                });
                // SAFETY: the load reads the 8 values of `denominators`.
                let divisors = unsafe { _mm256_loadu_ps(denominators.as_ptr()) };
                let x = _mm256_cvtepi32_ps(sums);
                let quotient = _mm256_div_ps(
                    _mm256_add_ps(_mm256_add_ps(x, x), divisors),
                    _mm256_add_ps(divisors, divisors),
                );
                _mm256_cvttps_epi32(_mm256_floor_ps(quotient))
            };
            let words = _mm256_packs_epi32(round(lo, 0), round(hi, 4));
            let bytes =
                _mm256_permute4x64_epi64::<0b00_00_10_00>(_mm256_packus_epi16(words, words));
            // SAFETY: o0 + 16 / PC <= ow (cover_steps): the store's 16 bytes
            // from o0 PC end within the row's ow PC bytes (the frame's `dst`).
            unsafe {
                _mm_storeu_si128(dst.add(at) as *mut __m128i, _mm256_castsi256_si128(bytes));
            }
        };
        cover_steps(ow, hstep_outputs, &hstep);
    };
    // SAFETY: this kernel's contract is the frame's, with AVX2 enabled here;
    // `vertical` and `horizontal` write within the bounds argued at their
    // stores.
    unsafe { x4_frame::<PC, RGB>(src, (width, height), vertical, horizontal) }
}

/// The AVX-512 kernel: the AVX2 kernel's steps over four 128-bit lanes, 64
/// input samples per vertical step (the phases joined across lanes by one
/// 64-bit permute) and 32 samples per horizontal step (narrowed to bytes by
/// `vpmovwb`). Narrower images take the AVX2 kernel.
///
/// # Safety
/// AVX-512 F/BW must be available; as `x4_frame`.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bilinear_x4_avx512<const PC: usize, const RGB: bool>(
    src: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    let channels = if RGB { 3 } else { PC };
    let (ow, ws) = (width / 4, width * channels);
    let (vstep_outputs, hstep_outputs) = (16 / PC, 32 / PC);
    if ow < hstep_outputs {
        // SAFETY: AVX-512 F implies AVX2; the same source contract.
        return unsafe { bilinear_x4_avx2::<PC, RGB>(src, width, height) };
    }
    let near_w = _mm512_set1_epi16((5 << 8) | 7);
    let far_w = _mm512_set1_epi16((1 << 8) | 3);
    let (madd_near, madd_far) = (
        _mm512_set1_epi32((5 << 16) | 7),
        _mm512_set1_epi32((1 << 16) | 3),
    );
    let half = _mm512_set1_epi32(1 << 9);
    let pair_phases = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
    ));
    // Each lane's two 64-bit phase halves, joined across the lanes.
    let join_lanes = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    // RGB: 4 pixels' 12 bytes per lane, widened to RGBX; masked at a row's end.
    let rgb_lanes = _mm512_setr_epi32(0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 9, 10, 11, 0);
    let to_rgbx = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1,
    ));
    let rgb_bytes: __mmask64 = (1 << 48) - 1;
    let vertical = |rows: &[*const u8; 8], y: usize, pp: &[*mut u16; 4]| {
        // 64 phase-layout bytes from pixel p0 of tap row `tap`.
        let load = |tap: usize, p0: usize| {
            // SAFETY: p0 = 4 k0 with k0 + 16 / PC <= ow (cover_steps), so the
            // plain load's 64 bytes from p0 PC end within 4 ow PC = ws, the
            // RGB load is taken where its 64 bytes end within ws (the guard),
            // and the masked RGB load reads 48 bytes, ending within
            // 3 (p0 + 16) <= ws; each row is a src row (the frame's contract)
            // or `zeros` (ws + 64).
            let bytes = unsafe {
                if !RGB {
                    return _mm512_loadu_si512(rows[tap].add(p0 * PC) as *const __m512i);
                }
                let at = rows[tap].add(3 * p0);
                if 3 * p0 + 64 <= ws {
                    _mm512_loadu_si512(at as *const __m512i)
                } else {
                    _mm512_maskz_loadu_epi8(rgb_bytes, at as *const i8)
                }
            };
            _mm512_shuffle_epi8(_mm512_permutexvar_epi32(rgb_lanes, bytes), to_rgbx)
        };
        let vstep = |k0: usize| {
            let p0 = 4 * k0;
            prefetch_next_rows(src, 4 * y + 6, ws, p0 * channels);
            prefetch_next_rows(src, 4 * y + 8, ws, p0 * channels);
            let loaded: [__m512i; 8] = std::array::from_fn(|tap| load(tap, p0));
            let sums = |lo: bool| {
                let pair = |low_tap: usize, high_tap: usize| {
                    if lo {
                        _mm512_unpacklo_epi8(loaded[low_tap], loaded[high_tap])
                    } else {
                        _mm512_unpackhi_epi8(loaded[low_tap], loaded[high_tap])
                    }
                };
                let near = _mm512_add_epi16(
                    _mm512_maddubs_epi16(pair(3, 2), near_w),
                    _mm512_maddubs_epi16(pair(4, 5), near_w),
                );
                let far = _mm512_add_epi16(
                    _mm512_maddubs_epi16(pair(1, 0), far_w),
                    _mm512_maddubs_epi16(pair(6, 7), far_w),
                );
                _mm512_add_epi16(near, far)
            };
            let (low, high) = (sums(true), sums(false));
            let (x, y) = if PC == 4 {
                (low, high)
            } else {
                let (low_shuffled, high_shuffled) = (
                    _mm512_shuffle_epi8(low, pair_phases),
                    _mm512_shuffle_epi8(high, pair_phases),
                );
                (
                    _mm512_unpacklo_epi32(low_shuffled, high_shuffled),
                    _mm512_unpackhi_epi32(low_shuffled, high_shuffled),
                )
            };
            let (x, y) = (
                _mm512_permutexvar_epi64(join_lanes, x),
                _mm512_permutexvar_epi64(join_lanes, y),
            );
            let at = k0 * PC;
            // SAFETY: k0 + 16 / PC <= ow (cover_steps): each store's 16 words
            // from k0 PC end within the phase array's ow PC (the frame's
            // `pp`).
            unsafe {
                _mm256_storeu_si256(pp[0].add(at) as *mut __m256i, _mm512_castsi512_si256(x));
                _mm256_storeu_si256(
                    pp[1].add(at) as *mut __m256i,
                    _mm512_extracti64x4_epi64::<1>(x),
                );
                _mm256_storeu_si256(pp[2].add(at) as *mut __m256i, _mm512_castsi512_si256(y));
                _mm256_storeu_si256(
                    pp[3].add(at) as *mut __m256i,
                    _mm512_extracti64x4_epi64::<1>(y),
                );
            }
        };
        cover_steps(ow, vstep_outputs, &vstep);
    };
    let horizontal = |pp: &[*mut u16; 4], zv: u32, col_totals: &[u32], dst: *mut u8| {
        let hstep = |o0: usize| {
            let at = o0 * PC;
            let ld = |phase: usize, offset: isize| {
                // SAFETY: o0 + 32 / PC <= ow (cover_steps) and offset is
                // -1..=1, so the load's 32 words lie within the phase array's
                // zero pixel before, ow PC words and zero pixel after.
                unsafe {
                    _mm512_loadu_si512(
                        pp[phase].offset(at as isize + offset * PC as isize) as *const __m512i
                    )
                }
            };
            let pair = |left: __m512i, right: __m512i| _mm512_add_epi16(left, right);
            let (inner, outer) = (pair(ld(1, 0), ld(2, 0)), pair(ld(0, 0), ld(3, 0)));
            let (next, last) = (pair(ld(3, -1), ld(0, 1)), pair(ld(2, -1), ld(1, 1)));
            let madd2 = |lo: bool| {
                let unpack = |low_words: __m512i, high_words: __m512i| {
                    if lo {
                        _mm512_unpacklo_epi16(low_words, high_words)
                    } else {
                        _mm512_unpackhi_epi16(low_words, high_words)
                    }
                };
                _mm512_add_epi32(
                    _mm512_madd_epi16(unpack(inner, outer), madd_near),
                    _mm512_madd_epi16(unpack(next, last), madd_far),
                )
            };
            // Per lane L: samples 8L..8L + 4 and 8L + 4..8L + 8.
            let (lo, hi) = (madd2(true), madd2(false));
            let whole = zv == 32 && o0 >= 1 && o0 + hstep_outputs < ow;
            let round = |sums: __m512i, first: usize| {
                if whole {
                    return _mm512_srli_epi32::<10>(_mm512_add_epi32(sums, half));
                }
                // floor((2 sum + d) / (2 d)) per lane, d = zv * the column total.
                let denominators: [f32; 16] = std::array::from_fn(|lane| {
                    let sample = first + lane % 4 + 8 * (lane / 4);
                    (zv * col_totals[(o0 + sample / PC).min(ow - 1)]) as f32
                });
                // SAFETY: the load reads the 16 values of `denominators`.
                let divisors = unsafe { _mm512_loadu_ps(denominators.as_ptr()) };
                let x = _mm512_cvtepi32_ps(sums);
                let quotient = _mm512_div_ps(
                    _mm512_add_ps(_mm512_add_ps(x, x), divisors),
                    _mm512_add_ps(divisors, divisors),
                );
                _mm512_cvttps_epi32(_mm512_roundscale_ps::<
                    { _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC },
                >(quotient))
            };
            // Samples in order; each at most 255.
            let words = _mm512_packs_epi32(round(lo, 0), round(hi, 4));
            // SAFETY: o0 + 32 / PC <= ow (cover_steps): the store's 32 bytes
            // from o0 PC end within the row's ow PC bytes (the frame's `dst`).
            unsafe {
                _mm256_storeu_si256(dst.add(at) as *mut __m256i, _mm512_cvtepi16_epi8(words));
            }
        };
        cover_steps(ow, hstep_outputs, &hstep);
    };
    // SAFETY: this kernel's contract is the frame's, with AVX2 implied by
    // AVX-512 F; `vertical` and `horizontal` write within the bounds argued
    // at their stores.
    unsafe { x4_frame::<PC, RGB>(src, (width, height), vertical, horizontal) }
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
                    1 => bilinear_x4_avx2::<1, false>(src, width, height),
                    3 => bilinear_x4_avx2::<4, true>(src, width, height),
                    _ => bilinear_x4_avx2::<4, false>(src, width, height),
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

    #[test]
    fn avx512_equals_scalar() {
        if !(std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw"))
        {
            return;
        }
        let kernel = |src: &[u8], width, height, channels| {
            // SAFETY: AVX-512 F/BW (and so AVX2, for RGB) are available; src
            // holds width * height * channels samples, width and height
            // multiples of 4 (WIDTHS and HEIGHTS).
            unsafe {
                match channels {
                    1 => bilinear_x4_avx512::<1, false>(src, width, height),
                    3 => bilinear_x4_avx512::<4, true>(src, width, height),
                    _ => bilinear_x4_avx512::<4, false>(src, width, height),
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

//! Separable bicubic passes: horizontal 4-tap (hconv) into cached rows,
//! vertical 4-tap (vconv) out of them.
//!
//! # Safety
//! Every kernel here is an `unsafe fn`: it requires the CPU features in its
//! `#[target_feature]` (its dispatch checks them at runtime) and the bounds
//! of the `ComposeSample` row method it implements, which its caller upholds.
//! Its loads and stores are justified against those bounds where they occur.

use super::rows::vconv4_u8_px;
use crate::sample::{BICUBIC_U8_FRAC_BITS, Sample, mulhrs, q15};
use crate::x86::*;

// ---- Separable bicubic passes (AVX2) ----------------------------------------
// hconv: horizontal 4-tap of one source row -> f32 (reused by 4 output rows).
// vconv: 4 cached f32 rows -> output dtype. Instantiated for u16 only: u8 uses
// the fixed-point (Q6 i16) kernels below, f32 the direct kernel (see
// ComposeSample::SEP_BICUBIC).
macro_rules! gen_hconv4 {
    ($name:ident, $t:ty, $load:expr_2021) => {
        /// AVX2 f32 `ComposeSample::hconv4_row`.
        ///
        /// # Safety
        /// AVX2 and FMA are available, and `ComposeSample::hconv4_row`'s
        /// contract holds.
        #[target_feature(enable = "avx2,fma")]
        pub(crate) unsafe fn $name(
            src: &[$t],
            dst: &mut [f32],
            x0s: usize,
            len: usize,
            step: usize,
            wx: [f32; 4],
        ) {
            use std::arch::x86_64::*;
            let tw = src.len();
            let vwx = [
                _mm256_set1_ps(wx[0]),
                _mm256_set1_ps(wx[1]),
                _mm256_set1_ps(wx[2]),
                _mm256_set1_ps(wx[3]),
            ];
            let (sp, dp) = (src.as_ptr(), dst.as_mut_ptr());
            let load = $load;
            let mut element = 0usize;
            // 8-wide while the x0+2*step 8-element load stays in bounds.
            while element + 8 <= len && x0s + element + 8 + 2 * step <= tw {
                let x0 = x0s + element;
                // SAFETY: x0 >= x0s >= step and element + 8 <= len, so the 8
                // elements at each tap x0 - step ..= x0 + 2 * step end at or
                // before the furthest tap, inside src, and the 8 at `element`
                // inside dst (the caller's contract).
                unsafe {
                    let mut sum = _mm256_mul_ps(load(sp.add(x0 - step)), vwx[0]);
                    sum = _mm256_fmadd_ps(load(sp.add(x0)), vwx[1], sum);
                    sum = _mm256_fmadd_ps(load(sp.add(x0 + step)), vwx[2], sum);
                    sum = _mm256_fmadd_ps(load(sp.add(x0 + 2 * step)), vwx[3], sum);
                    _mm256_storeu_ps(dp.add(element), sum);
                }
                element += 8;
            }
            while element < len {
                let x0 = x0s + element;
                // SAFETY: element < len: x0 - step >= 0, x0 + 2 * step is at
                // most the furthest tap, inside src, and element inside dst
                // (the caller's contract).
                unsafe {
                    *dp.add(element) = wx[0] * src.get_unchecked(x0 - step).to_f32()
                        + wx[1] * src.get_unchecked(x0).to_f32()
                        + wx[2] * src.get_unchecked(x0 + step).to_f32()
                        + wx[3] * src.get_unchecked(x0 + 2 * step).to_f32();
                }
                element += 1;
            }
        }
    };
}

/// Fixed-point u8 bicubic horizontal pass: pixels widened to Q6 (u8 << 6, max
/// 16320 — Q7 would overflow i16 under Catmull-Rom's 1.14x overshoot), taps
/// via mulhrs with Q15 weights, summed in i16 (range ~[-2305, 18700], safe).
/// Output rows are i16 Q6 — half the ring-buffer traffic of the f32 layout.
/// 16 elements/iter, overlapping last block (see `cover_row`); rows under 16
/// elements take a scalar loop that reproduces the mulhrs rounding exactly.
///
/// # Safety
/// AVX2 is available, and `ComposeSample::hconv4_row`'s contract holds.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn hconv4_row_u8_fx_avx2(
    src: &[u8],
    dst: &mut [i16],
    x0s: usize,
    len: usize,
    step: usize,
    wx: [f32; 4],
) {
    use std::arch::x86_64::*;
    const BLOCK: usize = 16;
    let qw = wx.map(q15);
    if len < BLOCK {
        for (element, out) in dst[..len].iter_mut().enumerate() {
            let x_left = x0s + element - step;
            let acc: i32 = (0..4)
                .map(|tap| {
                    mulhrs(
                        i32::from(src[x_left + tap * step]) << BICUBIC_U8_FRAC_BITS,
                        qw[tap],
                    )
                })
                .sum();
            // i16 wraparound of the sum == the vector kernel's per-term i16 adds.
            *out = acc as i16;
        }
        return;
    }
    debug_assert!(x0s >= step && x0s + len + 2 * step <= src.len() && len <= dst.len());
    let vw = qw.map(|weight| _mm256_set1_epi16(weight));
    let (sp, dp) = (src.as_ptr(), dst.as_mut_ptr());
    let ld = |src_ptr: *const u8| {
        // SAFETY: `block` passes pointers to 16 bytes inside src.
        let bytes = unsafe { _mm_loadu_si128(src_ptr as *const __m128i) };
        _mm256_slli_epi16::<{ BICUBIC_FRAC_BITS }>(_mm256_cvtepu8_epi16(bytes))
    };
    let block = |element: usize| {
        let x0 = x0s + element;
        // SAFETY: x0 >= x0s >= step and `cover_row` passes element + 16 <= len,
        // so the 16 bytes at each tap x0 - step ..= x0 + 2 * step end at or
        // before the furthest tap, inside src (the caller's contract).
        let sum = unsafe {
            _mm256_add_epi16(
                _mm256_add_epi16(
                    _mm256_mulhrs_epi16(ld(sp.add(x0 - step)), vw[0]),
                    _mm256_mulhrs_epi16(ld(sp.add(x0)), vw[1]),
                ),
                _mm256_add_epi16(
                    _mm256_mulhrs_epi16(ld(sp.add(x0 + step)), vw[2]),
                    _mm256_mulhrs_epi16(ld(sp.add(x0 + 2 * step)), vw[3]),
                ),
            )
        };
        // SAFETY: element + 16 <= len <= dst.len() (the caller's contract).
        unsafe { _mm256_storeu_si256(dp.add(element) as *mut __m256i, sum) };
    };
    cover_row(len, BLOCK, block);
}

/// Fixed-point u8 bicubic vertical pass over four Q6 i16 rows: mulhrs with
/// Q15 wy, i16 sums (bounded ~[-2700, 21300]), (sum + 32) >> 6 arithmetic
/// shift, packus clamps negatives/overshoot to [0, 255]. 16 px/iter,
/// overlapping last block (see `cover_row`); rows under 16 px go scalar.
///
/// # Safety
/// AVX2 is available, and `ComposeSample::vconv4_row`'s contract holds.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vconv4_row_u8_fx_avx2(
    rows: [&[i16]; 4],
    dst: &mut [u8],
    len: usize,
    wy: [f32; 4],
) {
    use std::arch::x86_64::*;
    const BLOCK: usize = 16;
    let qw = wy.map(q15);
    if len < BLOCK {
        for (element, out) in dst[..len].iter_mut().enumerate() {
            *out = vconv4_u8_px(rows, element, qw);
        }
        return;
    }
    debug_assert!(rows.iter().all(|row| row.len() >= len) && len <= dst.len());
    let vw = qw.map(|weight| _mm256_set1_epi16(weight));
    let round = _mm256_set1_epi16(half_ulp(BICUBIC_FRAC_BITS));
    let ptrs = rows.map(<[i16]>::as_ptr);
    let dp = dst.as_mut_ptr();
    let block = |element: usize| {
        // SAFETY: `cover_row` passes element + 16 <= len, and len bounds every
        // row (the caller's contract).
        let ld =
            |tap: usize| unsafe { _mm256_loadu_si256(ptrs[tap].add(element) as *const __m256i) };
        let sum = _mm256_add_epi16(
            _mm256_add_epi16(
                _mm256_mulhrs_epi16(ld(0), vw[0]),
                _mm256_mulhrs_epi16(ld(1), vw[1]),
            ),
            _mm256_add_epi16(
                _mm256_mulhrs_epi16(ld(2), vw[2]),
                _mm256_mulhrs_epi16(ld(3), vw[3]),
            ),
        );
        let rounded = _mm256_srai_epi16::<{ BICUBIC_FRAC_BITS }>(_mm256_add_epi16(sum, round));
        let p8 = _mm_packus_epi16(
            _mm256_castsi256_si128(rounded),
            _mm256_extracti128_si256(rounded, 1),
        );
        // SAFETY: element + 16 <= len <= dst.len() (the caller's contract).
        unsafe { _mm_storeu_si128(dp.add(element) as *mut __m128i, p8) };
    };
    cover_row(len, BLOCK, block);
}

/// AVX-512 variant of the fixed-point u8 bicubic horizontal pass: 32 el/iter,
/// same Q6 arithmetic as the AVX2 kernel; overlapping last block (see
/// `cover_row`), rows under 32 elements go to the AVX2 kernel.
///
/// # Safety
/// AVX-512 F and BW are available, and `ComposeSample::hconv4_row`'s
/// contract holds.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn hconv4_row_u8_fx_avx512(
    src: &[u8],
    dst: &mut [i16],
    x0s: usize,
    len: usize,
    step: usize,
    wx: [f32; 4],
) {
    use std::arch::x86_64::*;
    const BLOCK: usize = 32;
    if len < BLOCK {
        // SAFETY: AVX-512 F implies AVX2; the contract is the same.
        return unsafe { hconv4_row_u8_fx_avx2(src, dst, x0s, len, step, wx) };
    }
    debug_assert!(x0s >= step && x0s + len + 2 * step <= src.len() && len <= dst.len());
    let vw = wx.map(|weight| _mm512_set1_epi16(q15(weight)));
    let (sp, dp) = (src.as_ptr(), dst.as_mut_ptr());
    let ld = |src_ptr: *const u8| {
        // SAFETY: `block` passes pointers to 32 bytes inside src.
        let bytes = unsafe { _mm256_loadu_si256(src_ptr as *const __m256i) };
        _mm512_slli_epi16::<{ BICUBIC_FRAC_BITS as u32 }>(_mm512_cvtepu8_epi16(bytes))
    };
    let block = |element: usize| {
        let x0 = x0s + element;
        // SAFETY: x0 >= x0s >= step and `cover_row` passes element + 32 <= len,
        // so the 32 bytes at each tap x0 - step ..= x0 + 2 * step end at or
        // before the furthest tap, inside src (the caller's contract).
        let sum = unsafe {
            _mm512_add_epi16(
                _mm512_add_epi16(
                    _mm512_mulhrs_epi16(ld(sp.add(x0 - step)), vw[0]),
                    _mm512_mulhrs_epi16(ld(sp.add(x0)), vw[1]),
                ),
                _mm512_add_epi16(
                    _mm512_mulhrs_epi16(ld(sp.add(x0 + step)), vw[2]),
                    _mm512_mulhrs_epi16(ld(sp.add(x0 + 2 * step)), vw[3]),
                ),
            )
        };
        // SAFETY: element + 32 <= len <= dst.len() (the caller's contract).
        unsafe { _mm512_storeu_si512(dp.add(element) as *mut __m512i, sum) };
    };
    cover_row(len, BLOCK, block);
}

/// AVX-512 variant of the fixed-point u8 bicubic vertical pass: 32 px/iter.
/// `max(rounded, 0)` before the unsigned-saturating narrow (cvtusepi16 would wrap
/// negative overshoot to 255 otherwise); overlapping last block (see
/// `cover_row`), rows under 32 px go to the AVX2 kernel.
///
/// # Safety
/// AVX-512 F and BW are available, and `ComposeSample::vconv4_row`'s
/// contract holds.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn vconv4_row_u8_fx_avx512(
    rows: [&[i16]; 4],
    dst: &mut [u8],
    len: usize,
    wy: [f32; 4],
) {
    use std::arch::x86_64::*;
    const BLOCK: usize = 32;
    if len < BLOCK {
        // SAFETY: AVX-512 F implies AVX2; the contract is the same.
        return unsafe { vconv4_row_u8_fx_avx2(rows, dst, len, wy) };
    }
    debug_assert!(rows.iter().all(|row| row.len() >= len) && len <= dst.len());
    let vw = wy.map(|weight| _mm512_set1_epi16(q15(weight)));
    let round = _mm512_set1_epi16(half_ulp(BICUBIC_FRAC_BITS));
    let zero = _mm512_setzero_si512();
    let ptrs = rows.map(<[i16]>::as_ptr);
    let dp = dst.as_mut_ptr();
    let block = |element: usize| {
        // SAFETY: `cover_row` passes element + 32 <= len, and len bounds every
        // row (the caller's contract).
        let ld =
            |tap: usize| unsafe { _mm512_loadu_si512(ptrs[tap].add(element) as *const __m512i) };
        let sum = _mm512_add_epi16(
            _mm512_add_epi16(
                _mm512_mulhrs_epi16(ld(0), vw[0]),
                _mm512_mulhrs_epi16(ld(1), vw[1]),
            ),
            _mm512_add_epi16(
                _mm512_mulhrs_epi16(ld(2), vw[2]),
                _mm512_mulhrs_epi16(ld(3), vw[3]),
            ),
        );
        let rounded =
            _mm512_srai_epi16::<{ BICUBIC_FRAC_BITS as u32 }>(_mm512_add_epi16(sum, round));
        let rounded = _mm512_max_epi16(rounded, zero);
        // SAFETY: element + 32 <= len <= dst.len() (the caller's contract).
        unsafe {
            _mm256_storeu_si256(
                dp.add(element) as *mut __m256i,
                _mm512_cvtusepi16_epi8(rounded),
            );
        }
    };
    cover_row(len, BLOCK, block);
}

/// AVX2 separable-bicubic vertical pass, RGB (3 samples) in -> RGBX (4) out.
///
/// The horizontal pass (`hconv4_row`) produces the 3-sample
/// Q6 i16 intermediate. Only this final store widens, by the same trick as
/// the bilinear kernel -- 24 elements = 8 pixels = one 32-byte RGBX store --
/// so bicubic gets the fused expand for free too. `packus` still clamps, which
/// is what keeps bicubic's ringing overshoot in range. Overlapping last block
/// (see `cover_row`); rows under 8 px go scalar.
///
/// # Safety
/// AVX2 is available, every row holds at least `npix * 3` elements and `dst`
/// at least `npix * 4`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vconv4_row_rgb_to_rgbx_avx2(
    rows: [&[i16]; 4],
    dst: &mut [u8],
    npix: usize,
    pad: u8,
    wy: [f32; 4],
) {
    use std::arch::x86_64::*;
    const BLOCK_PX: usize = 8;
    let qw = wy.map(q15);
    if npix < BLOCK_PX {
        for (px, out) in dst[..npix * 4]
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .enumerate()
        {
            for (ch, sample) in out[..3].iter_mut().enumerate() {
                *sample = vconv4_u8_px(rows, px * 3 + ch, qw);
            }
            out[3] = pad;
        }
        return;
    }
    debug_assert!(rows.iter().all(|row| row.len() >= npix * 3) && npix * 4 <= dst.len());
    let vwy = qw.map(|weight| _mm_set1_epi16(weight));
    let round = _mm_set1_epi16(half_ulp(BICUBIC_FRAC_BITS));
    let ptrs = rows.map(<[i16]>::as_ptr);
    let dp = dst.as_mut_ptr();
    let (mask, padv) = (rgb_to_rgbx_shuffle(), rgbx_pad_lanes(pad));

    let conv8 = |element: usize| -> __m128i {
        // SAFETY: `block` passes element + 8 <= npix * 3, which bounds every
        // row (the caller's contract).
        let ld = |tap: usize| unsafe { _mm_loadu_si128(ptrs[tap].add(element) as *const __m128i) };
        let sum = _mm_add_epi16(
            _mm_add_epi16(
                _mm_mulhrs_epi16(ld(0), vwy[0]),
                _mm_mulhrs_epi16(ld(1), vwy[1]),
            ),
            _mm_add_epi16(
                _mm_mulhrs_epi16(ld(2), vwy[2]),
                _mm_mulhrs_epi16(ld(3), vwy[3]),
            ),
        );
        let rounded = _mm_srai_epi16::<{ BICUBIC_FRAC_BITS }>(_mm_add_epi16(sum, round));
        _mm_packus_epi16(rounded, rounded)
    };

    // Output pixels px..px+8 = row elements 3 px .. 3 px + 24.
    let block = |px: usize| {
        let element = px * 3;
        let (b0, b1, b2) = (conv8(element), conv8(element + 8), conv8(element + 16));
        let c01 = _mm_unpacklo_epi64(b0, b1);
        let hi = _mm_alignr_epi8(b2, c01, 12);
        let full = _mm256_inserti128_si256(_mm256_castsi128_si256(c01), hi, 1);
        let expanded = _mm256_or_si256(_mm256_shuffle_epi8(full, mask), padv);
        // SAFETY: `cover_row` passes px + 8 <= npix, so the 32 bytes at
        // px * 4 are inside dst (npix * 4 elements, the caller's contract).
        unsafe { _mm256_storeu_si256(dp.add(px * 4) as *mut __m256i, expanded) };
    };
    cover_row(npix, BLOCK_PX, block);
}

/// AVX-512 variant of `vconv4_row_rgb_to_rgbx_avx2`: 32 px/iter as three
/// 32-element vertical passes (the `vconv4_row_u8_fx_avx512` arithmetic,
/// bit-identical to the AVX2 kernel) expanded by `store_rgbx_32px_avx512`.
/// Overlapping last block; rows under 32 px go to the AVX2 kernel.
///
/// # Safety
/// AVX-512 F and BW are available, and `vconv4_row_rgb_to_rgbx_avx2`'s
/// bounds hold.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn vconv4_row_rgb_to_rgbx_avx512(
    rows: [&[i16]; 4],
    dst: &mut [u8],
    npix: usize,
    pad: u8,
    wy: [f32; 4],
) {
    use std::arch::x86_64::*;
    const BLOCK_PX: usize = 32;
    if npix < BLOCK_PX {
        // SAFETY: AVX-512 F implies AVX2; the bounds are this function's.
        return unsafe { vconv4_row_rgb_to_rgbx_avx2(rows, dst, npix, pad, wy) };
    }
    debug_assert!(rows.iter().all(|row| row.len() >= npix * 3) && npix * 4 <= dst.len());
    let vw = wy.map(|weight| _mm512_set1_epi16(q15(weight)));
    let round = _mm512_set1_epi16(half_ulp(BICUBIC_FRAC_BITS));
    let zero = _mm512_setzero_si512();
    let ptrs = rows.map(<[i16]>::as_ptr);
    let dp = dst.as_mut_ptr();
    // 32 output samples starting at row `element`, as bytes.
    let conv32 = |element: usize| -> __m256i {
        // SAFETY: `block` passes element + 32 <= npix * 3, which bounds every
        // row (the caller's contract).
        let ld =
            |tap: usize| unsafe { _mm512_loadu_si512(ptrs[tap].add(element) as *const __m512i) };
        let sum = _mm512_add_epi16(
            _mm512_add_epi16(
                _mm512_mulhrs_epi16(ld(0), vw[0]),
                _mm512_mulhrs_epi16(ld(1), vw[1]),
            ),
            _mm512_add_epi16(
                _mm512_mulhrs_epi16(ld(2), vw[2]),
                _mm512_mulhrs_epi16(ld(3), vw[3]),
            ),
        );
        let rounded =
            _mm512_srai_epi16::<{ BICUBIC_FRAC_BITS as u32 }>(_mm512_add_epi16(sum, round));
        _mm512_cvtusepi16_epi8(_mm512_max_epi16(rounded, zero))
    };
    // Output pixels px..px+32 = row elements 3 px .. 3 px + 96.
    let block = |px: usize| {
        let element = px * 3;
        let rgb = [conv32(element), conv32(element + 32), conv32(element + 64)];
        // SAFETY: AVX-512 F+BW are enabled here; `cover_row` passes
        // px + 32 <= npix, so the 128 bytes at px * 4 are inside dst (npix * 4
        // elements, the caller's contract).
        unsafe { store_rgbx_32px_avx512(dp.add(px * 4), rgb, pad) };
    };
    cover_row(npix, BLOCK_PX, block);
}

macro_rules! gen_vconv4 {
    ($name:ident, $t:ty, $store:expr_2021) => {
        /// AVX2 f32-row `ComposeSample::vconv4_row`.
        ///
        /// # Safety
        /// AVX2 and FMA are available, and `ComposeSample::vconv4_row`'s
        /// contract holds.
        #[target_feature(enable = "avx2,fma")]
        pub(crate) unsafe fn $name(rows: [&[f32]; 4], dst: &mut [$t], len: usize, wy: [f32; 4]) {
            use std::arch::x86_64::*;
            let vwy = [
                _mm256_set1_ps(wy[0]),
                _mm256_set1_ps(wy[1]),
                _mm256_set1_ps(wy[2]),
                _mm256_set1_ps(wy[3]),
            ];
            let ptrs = [
                rows[0].as_ptr(),
                rows[1].as_ptr(),
                rows[2].as_ptr(),
                rows[3].as_ptr(),
            ];
            let dp = dst.as_mut_ptr();
            let store = $store;
            let mut element = 0usize;
            while element + 8 <= len {
                // SAFETY: element + 8 <= len, which bounds every row and dst
                // (the caller's contract).
                unsafe {
                    let mut acc = _mm256_mul_ps(_mm256_loadu_ps(ptrs[0].add(element)), vwy[0]);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(ptrs[1].add(element)), vwy[1], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(ptrs[2].add(element)), vwy[2], acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(ptrs[3].add(element)), vwy[3], acc);
                    store(dp.add(element), acc);
                }
                element += 8;
            }
            while element < len {
                // SAFETY: element < len, which bounds every row and dst (the
                // caller's contract).
                unsafe {
                    let acc = wy[0] * rows[0].get_unchecked(element)
                        + wy[1] * rows[1].get_unchecked(element)
                        + wy[2] * rows[2].get_unchecked(element)
                        + wy[3] * rows[3].get_unchecked(element);
                    *dp.add(element) = <$t as Sample>::from_f32(acc);
                }
                element += 1;
            }
        }
    };
}

gen_hconv4!(hconv4_row_u16_avx2, u16, |src_ptr: *const u16| {
    // SAFETY: the kernel passes pointers to 8 elements inside src.
    let samples = unsafe { _mm_loadu_si128(src_ptr as *const __m128i) };
    _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(samples))
});

gen_vconv4!(
    vconv4_row_u16_avx2,
    u16,
    |dst_ptr: *mut u16, value: __m256| {
        let vi = _mm256_cvttps_epi32(_mm256_add_ps(value, _mm256_set1_ps(0.5)));
        let u16s = _mm_packus_epi32(_mm256_castsi256_si128(vi), _mm256_extracti128_si256(vi, 1));
        // SAFETY: the kernel passes pointers to 8 elements inside dst.
        unsafe { _mm_storeu_si128(dst_ptr as *mut __m128i, u16s) };
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::rows::tests::{
        BILINEAR_ROUNDING_STEPS, HconvU16, TestSample, VconvCase, VconvU16, assert_close,
        bicubic_cases, f32_rows, guarded_output, l1, q6_hconv_reference, q6_rows,
        q6_vconv_reference, q6_vconv_rgbx_reference, row_lengths,
    };
    use crate::composite::rows::{scalar_hconv4_row, scalar_vconv4_row};
    use crate::sample::RGB;

    fn avx2_fma_detected() -> bool {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }

    /// The AVX-512 kernels' features, plus the AVX2 of the kernels their
    /// short rows fall back to.
    fn avx512_detected() -> bool {
        avx2_fma_detected()
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
    }

    type HconvU8 = unsafe fn(&[u8], &mut [i16], usize, usize, usize, [f32; 4]);
    type VconvU8 = unsafe fn([&[i16]; 4], &mut [u8], usize, [f32; 4]);
    type VconvRgbx = unsafe fn([&[i16]; 4], &mut [u8], usize, u8, [f32; 4]);

    /// Asserts `kernel` equals the Q6 mirror on every bicubic case.
    ///
    /// # Safety
    /// The CPU has the features `kernel` needs.
    unsafe fn assert_hconv_u8_equals_mirror(kernel: HconvU8) {
        for case in bicubic_cases::<u8>(31) {
            let [src, ..] = case.rows();
            let simd = guarded_output(case.len, i16::MIN, |dst| {
                // SAFETY: x0s >= step and src ends at the furthest tap; dst
                // holds len elements; the features are the caller's.
                unsafe { kernel(src, dst, case.x0s, case.len, case.step, case.wx) };
            });
            let expected = q6_hconv_reference(src, case.x0s, case.len, case.step, case.wx);
            assert_eq!(simd, expected, "{}", case.describe());
        }
    }

    /// Asserts `kernel` equals the Q6 mirror on rows of every test length.
    ///
    /// # Safety
    /// The CPU has the features `kernel` needs.
    unsafe fn assert_vconv_u8_equals_mirror(kernel: VconvU8) {
        for case in q6_rows(32, row_lengths()) {
            let simd = guarded_output(case.len, u8::SENTINEL, |dst| {
                // SAFETY: every row and dst hold len elements; the features
                // are the caller's.
                unsafe { kernel(case.rows(), dst, case.len, case.wy) };
            });
            let expected = q6_vconv_reference(case.rows(), case.len, case.wy);
            assert_eq!(simd, expected, "len {} wy {:?}", case.len, case.wy);
        }
    }

    /// Asserts `kernel` equals the scalar RGB -> RGBX mirror for every pixel
    /// count of `row_lengths`.
    ///
    /// # Safety
    /// The CPU has the features `kernel` needs.
    unsafe fn assert_vconv_rgbx_equals_mirror(kernel: VconvRgbx) {
        let cases: Vec<VconvCase<i16>> = q6_rows(33, row_lengths().map(|npix| npix * RGB));
        for case in cases {
            let npix = case.len / RGB;
            let simd = guarded_output(npix * 4, u8::SENTINEL, |dst| {
                // SAFETY: every row holds npix * 3 elements, dst npix * 4; the
                // features are the caller's.
                unsafe { kernel(case.rows(), dst, npix, 0xC3, case.wy) };
            });
            let expected = q6_vconv_rgbx_reference(case.rows(), npix, 0xC3, case.wy);
            assert_eq!(simd, expected, "{npix} px wy {:?}", case.wy);
        }
    }

    #[test]
    fn hconv_u8_fixed_point_equals_mirror() {
        if avx2_fma_detected() {
            // SAFETY: AVX2 is available.
            unsafe { assert_hconv_u8_equals_mirror(hconv4_row_u8_fx_avx2) };
        }
        if avx512_detected() {
            // SAFETY: AVX-512 F+BW and AVX2 are available.
            unsafe { assert_hconv_u8_equals_mirror(hconv4_row_u8_fx_avx512) };
        }
    }

    #[test]
    fn vconv_u8_fixed_point_equals_mirror() {
        if avx2_fma_detected() {
            // SAFETY: AVX2 is available.
            unsafe { assert_vconv_u8_equals_mirror(vconv4_row_u8_fx_avx2) };
        }
        if avx512_detected() {
            // SAFETY: AVX-512 F+BW and AVX2 are available.
            unsafe { assert_vconv_u8_equals_mirror(vconv4_row_u8_fx_avx512) };
        }
    }

    #[test]
    fn vconv_rgb_to_rgbx_equals_mirror() {
        if avx2_fma_detected() {
            // SAFETY: AVX2 is available.
            unsafe { assert_vconv_rgbx_equals_mirror(vconv4_row_rgb_to_rgbx_avx2) };
        }
        if avx512_detected() {
            // SAFETY: AVX-512 F+BW and AVX2 are available.
            unsafe { assert_vconv_rgbx_equals_mirror(vconv4_row_rgb_to_rgbx_avx512) };
        }
    }

    #[test]
    fn hconv_u16_avx2_matches_scalar() {
        if !avx2_fma_detected() {
            return;
        }
        for case in bicubic_cases::<u16>(34) {
            let [src, ..] = case.rows();
            let run = |kernel: HconvU16| {
                guarded_output(case.len, f32::SENTINEL, |dst| {
                    // SAFETY: x0s >= step and src ends at the furthest tap;
                    // dst holds len elements; AVX2 and FMA are available.
                    unsafe { kernel(src, dst, case.x0s, case.len, case.step, case.wx) };
                })
            };
            let (simd, expected) = (run(hconv4_row_u16_avx2), run(scalar_hconv4_row::<u16>));
            // A 4-tap sum, rounded like a bilinear blend.
            let tolerance =
                f32::tolerance(BILINEAR_ROUNDING_STEPS, u16::MAX.to_f32() * l1(case.wx));
            assert_close(&simd, &expected, tolerance, &case.describe());
        }
    }

    #[test]
    fn vconv_u16_avx2_within_one_lsb_of_scalar() {
        if !avx2_fma_detected() {
            return;
        }
        for case in f32_rows(35, row_lengths()) {
            let run = |kernel: VconvU16| {
                guarded_output(case.len, u16::SENTINEL, |dst| {
                    // SAFETY: every row and dst hold len elements; AVX2 and
                    // FMA are available.
                    unsafe { kernel(case.rows(), dst, case.len, case.wy) };
                })
            };
            let (simd, expected) = (run(vconv4_row_u16_avx2), run(scalar_vconv4_row::<u16>));
            assert_close(&simd, &expected, 1.0, &format!("len {}", case.len));
        }
    }
}

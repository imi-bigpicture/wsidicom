//! Translate-resample row kernels: bilinear and direct bicubic, AVX2 and
//! AVX-512, f32 FMA and u8 fixed-point.
//!
//! # Safety
//! Every kernel here is an `unsafe fn`: it requires the CPU features in its
//! `#[target_feature]` (its dispatch checks them at runtime) and the bounds
//! of the `ComposeSample` row method it implements, which its caller upholds.
//! Its loads and stores are justified against those bounds where they occur.

use super::rows::scalar_bilinear_row_rgb_to_rgbx;
use crate::sample::{Sample, q15};
use crate::x86::*;

// ---- AVX2 resample kernels -------------------------------------------------
// One bilinear body + one bicubic body, specialized per dtype by `load` (widen
// to f32x8) and `store` (narrow/round back) closures, each called only with a
// pointer to 8 in-bounds elements. The FMA math and the element-based bounds
// guards are identical across dtypes.
macro_rules! gen_bilinear {
    ($name:ident, $src:ty => $t:ty, $load:expr_2021, $store:expr_2021) => {
        /// AVX2 `ComposeSample::bilinear_row` (rows of `$src` samples).
        ///
        /// # Safety
        /// AVX2 and FMA are available, and `ComposeSample::bilinear_row`'s
        /// contract holds.
        #[target_feature(enable = "avx2,fma")]
        pub(crate) unsafe fn $name(
            row0: &[$src],
            row1: &[$src],
            dst: &mut [$t],
            x0s: usize,
            len: usize,
            step: usize,
            weights: [f32; 4],
        ) {
            use std::arch::x86_64::*;
            let tw = row0.len();
            let (vw0, vw1, vw2, vw3) = (
                _mm256_set1_ps(weights[0]),
                _mm256_set1_ps(weights[1]),
                _mm256_set1_ps(weights[2]),
                _mm256_set1_ps(weights[3]),
            );
            let (p0, p1, dp) = (row0.as_ptr(), row1.as_ptr(), dst.as_mut_ptr());
            let load = $load;
            let store = $store;
            let mut element = 0usize;
            // 8-wide while the +step-offset 8-element load stays in bounds.
            while element + 8 <= len && x0s + element + 8 + step <= tw {
                let x0 = x0s + element;
                // SAFETY: element + 8 <= len, so the 8 elements at x0 and at
                // x0 + step end at or before the furthest tap, inside both
                // rows, and the 8 at `element` inside dst (the caller's contract).
                unsafe {
                    let mut blend = _mm256_mul_ps(load(p0.add(x0)), vw0);
                    blend = _mm256_fmadd_ps(load(p0.add(x0 + step)), vw1, blend);
                    blend = _mm256_fmadd_ps(load(p1.add(x0)), vw2, blend);
                    blend = _mm256_fmadd_ps(load(p1.add(x0 + step)), vw3, blend);
                    store(dp.add(element), blend);
                }
                element += 8;
            }
            while element < len {
                let x0 = x0s + element;
                // SAFETY: element < len: x0 + step is at most the furthest
                // tap, inside both rows, and element inside dst (the caller's
                // contract).
                unsafe {
                    let blend = weights[0] * row0.get_unchecked(x0).to_f32()
                        + weights[1] * row0.get_unchecked(x0 + step).to_f32()
                        + weights[2] * row1.get_unchecked(x0).to_f32()
                        + weights[3] * row1.get_unchecked(x0 + step).to_f32();
                    *dp.add(element) = <$t as Sample>::from_f32(blend);
                }
                element += 1;
            }
        }
    };
}

macro_rules! gen_bicubic {
    ($name:ident, $t:ty, $load:expr_2021, $store:expr_2021) => {
        /// AVX2 `ComposeSample::bicubic_row`.
        ///
        /// # Safety
        /// AVX2 and FMA are available, and `ComposeSample::bicubic_row`'s
        /// contract holds.
        #[target_feature(enable = "avx2,fma")]
        pub(crate) unsafe fn $name(
            rows: [&[$t]; 4],
            dst: &mut [$t],
            x0s: usize,
            len: usize,
            step: usize,
            wx: [f32; 4],
            wy: [f32; 4],
        ) {
            use std::arch::x86_64::*;
            let tw = rows[0].len();
            let vwx = [
                _mm256_set1_ps(wx[0]),
                _mm256_set1_ps(wx[1]),
                _mm256_set1_ps(wx[2]),
                _mm256_set1_ps(wx[3]),
            ];
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
            let load = $load;
            let store = $store;
            let mut element = 0usize;
            // 8-wide while the x0+2*step 8-element load stays in bounds.
            while element + 8 <= len && x0s + element + 8 + 2 * step <= tw {
                let x0 = x0s + element;
                let mut acc = _mm256_setzero_ps();
                for tap in 0..4 {
                    let row_ptr = ptrs[tap];
                    // SAFETY: x0 >= x0s >= step and element + 8 <= len, so the
                    // 8 elements at each tap x0 - step ..= x0 + 2 * step end at
                    // or before the furthest tap, inside every row (the
                    // caller's contract).
                    unsafe {
                        let mut sum = _mm256_mul_ps(load(row_ptr.add(x0 - step)), vwx[0]);
                        sum = _mm256_fmadd_ps(load(row_ptr.add(x0)), vwx[1], sum);
                        sum = _mm256_fmadd_ps(load(row_ptr.add(x0 + step)), vwx[2], sum);
                        sum = _mm256_fmadd_ps(load(row_ptr.add(x0 + 2 * step)), vwx[3], sum);
                        acc = _mm256_fmadd_ps(sum, vwy[tap], acc);
                    }
                }
                // SAFETY: element + 8 <= len <= dst.len() (the caller's contract).
                unsafe { store(dp.add(element), acc) };
                element += 8;
            }
            while element < len {
                let x0 = x0s + element;
                let mut acc = 0.0f32;
                for tap in 0..4 {
                    let row = rows[tap];
                    // SAFETY: element < len: x0 - step >= 0 and x0 + 2 * step
                    // is at most the furthest tap, inside every row (the
                    // caller's contract).
                    unsafe {
                        acc += wy[tap]
                            * (wx[0] * row.get_unchecked(x0 - step).to_f32()
                                + wx[1] * row.get_unchecked(x0).to_f32()
                                + wx[2] * row.get_unchecked(x0 + step).to_f32()
                                + wx[3] * row.get_unchecked(x0 + 2 * step).to_f32());
                    }
                }
                // SAFETY: element < len <= dst.len() (the caller's contract).
                unsafe { *dp.add(element) = <$t as Sample>::from_f32(acc) };
                element += 1;
            }
        }
    };
}

/// Fixed-point u8 bilinear row: pixels widened to Q7 (u8 << 7, max 32640),
/// weights Q15, terms via `mulhrs` (rounded), sum <= 32643 fits i16, final
/// (sum + 64) >> 7. 16 px/iter. Error <= term rounding (1/256 px * 4) + weight
/// quantization (2^-15) => <= 1 LSB vs the f32/scalar path (the documented
/// resample latitude). The row tail is one overlapping block ending at `len`
/// (see `cover_row`); rows shorter than a block use the f32 kernel.
///
/// # Safety
/// AVX2 and FMA are available (FMA for the f32 kernel of short rows), and
/// `ComposeSample::bilinear_row`'s contract holds.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn bilinear_row_u8_fx_avx2(
    row0: &[u8],
    row1: &[u8],
    dst: &mut [u8],
    x0s: usize,
    len: usize,
    step: usize,
    weights: [f32; 4],
) {
    use std::arch::x86_64::*;
    const BLOCK: usize = 16;
    if len < BLOCK {
        // SAFETY: AVX2 and FMA are available, and the contract is the same
        // (this function's).
        return unsafe { bilinear_row_u8_avx2(row0, row1, dst, x0s, len, step, weights) };
    }
    debug_assert!(x0s + len + step <= row0.len() && len <= dst.len());
    let vw = weights.map(|weight| _mm256_set1_epi16(q15(weight)));
    let round = _mm256_set1_epi16(half_ulp(BILINEAR_FRAC_BITS));
    let (p0, p1, dp) = (row0.as_ptr(), row1.as_ptr(), dst.as_mut_ptr());
    let ld = |src_ptr: *const u8| {
        // SAFETY: `block` passes pointers to 16 bytes inside a row.
        let bytes = unsafe { _mm_loadu_si128(src_ptr as *const __m128i) };
        _mm256_slli_epi16::<{ BILINEAR_FRAC_BITS }>(_mm256_cvtepu8_epi16(bytes))
    };
    // Outputs element..element+16; reads x0 .. x0+16+step, in the row by bilinear_row's contract.
    let block = |element: usize| {
        let x0 = x0s + element;
        // SAFETY: `cover_row` passes element + 16 <= len, so the 16 bytes at
        // x0 and at x0 + step end at or before the furthest tap, inside both
        // rows (the caller's contract).
        let sum = unsafe {
            _mm256_add_epi16(
                _mm256_add_epi16(
                    _mm256_mulhrs_epi16(ld(p0.add(x0)), vw[0]),
                    _mm256_mulhrs_epi16(ld(p0.add(x0 + step)), vw[1]),
                ),
                _mm256_add_epi16(
                    _mm256_mulhrs_epi16(ld(p1.add(x0)), vw[2]),
                    _mm256_mulhrs_epi16(ld(p1.add(x0 + step)), vw[3]),
                ),
            )
        };
        let rounded = _mm256_srli_epi16::<{ BILINEAR_FRAC_BITS }>(_mm256_add_epi16(sum, round));
        let p8 = _mm_packus_epi16(
            _mm256_castsi256_si128(rounded),
            _mm256_extracti128_si256(rounded, 1),
        );
        // SAFETY: element + 16 <= len <= dst.len() (the caller's contract).
        unsafe { _mm_storeu_si128(dp.add(element) as *mut __m128i, p8) };
    };
    cover_row(len, BLOCK, block);
}

/// Fused fixed-point bilinear, RGB (3 samples) in -> RGBX (4 samples) out:
/// 16 output px per iteration = three 16-element fx blends (48 RGB bytes)
/// assembled by alignr into two 24-byte lane-pairs, expanded to 2 x 32-byte
/// RGBX stores in-register — no scratch row, no second pass. Same Q7 x Q15
/// mulhrs arithmetic as `bilinear_row_u8_fx_avx2`; overlapping last block (see
/// `cover_row`), scalar f32 only for rows under 16 px.
///
/// # Safety
/// AVX2 is available; `ComposeSample::bilinear_row`'s contract holds for
/// `len = npix * 3`, except that `dst` holds `npix * 4` elements.
#[target_feature(enable = "avx2")]
#[expect(
    clippy::too_many_arguments,
    reason = "bilinear_row's contract plus `pad`"
)]
pub(crate) unsafe fn bilinear_row_rgb_to_rgbx_fx_avx2(
    row0: &[u8],
    row1: &[u8],
    dst: &mut [u8],
    x0s: usize,
    npix: usize,
    step: usize,
    pad: u8,
    weights: [f32; 4],
) {
    use std::arch::x86_64::*;
    const BLOCK_PX: usize = 16;
    if npix < BLOCK_PX {
        return scalar_bilinear_row_rgb_to_rgbx(row0, row1, dst, x0s, npix, step, pad, weights);
    }
    debug_assert!(x0s + npix * 3 + step <= row0.len() && npix * 4 <= dst.len());
    let vw = weights.map(|weight| _mm256_set1_epi16(q15(weight)));
    let round = _mm256_set1_epi16(half_ulp(BILINEAR_FRAC_BITS));
    let (mask, padv) = (rgb_to_rgbx_shuffle(), rgbx_pad_lanes(pad));
    let (p0, p1, dp) = (row0.as_ptr(), row1.as_ptr(), dst.as_mut_ptr());

    // 16 blended RGB elements at element offset `off` (one xmm of bytes).
    let blend16 = |off: usize| -> __m128i {
        let ld = |src_ptr: *const u8| {
            // SAFETY: the calls below pass pointers to 16 bytes inside a row.
            let bytes = unsafe { _mm_loadu_si128(src_ptr as *const __m128i) };
            _mm256_slli_epi16::<{ BILINEAR_FRAC_BITS }>(_mm256_cvtepu8_epi16(bytes))
        };
        // SAFETY: `block` passes off <= x0s + npix * 3 - 16, so the 16 bytes
        // at off and at off + step end at or before the furthest tap, inside
        // both rows (the caller's contract).
        let sum = unsafe {
            _mm256_add_epi16(
                _mm256_add_epi16(
                    _mm256_mulhrs_epi16(ld(p0.add(off)), vw[0]),
                    _mm256_mulhrs_epi16(ld(p0.add(off + step)), vw[1]),
                ),
                _mm256_add_epi16(
                    _mm256_mulhrs_epi16(ld(p1.add(off)), vw[2]),
                    _mm256_mulhrs_epi16(ld(p1.add(off + step)), vw[3]),
                ),
            )
        };
        let rounded = _mm256_srli_epi16::<{ BILINEAR_FRAC_BITS }>(_mm256_add_epi16(sum, round));
        _mm_packus_epi16(
            _mm256_castsi256_si128(rounded),
            _mm256_extracti128_si256(rounded, 1),
        )
    };

    // Output pixels px..px+16; reads 48 + step bytes from x0s + 3 px.
    let block = |px: usize| {
        let off = x0s + px * 3;
        let (b0, b1, b2) = (blend16(off), blend16(off + 16), blend16(off + 32));
        // bytes 0..24 -> lanes [b0 | b0[12..] ++ b1[..8]]
        let full0 =
            _mm256_inserti128_si256(_mm256_castsi128_si256(b0), _mm_alignr_epi8::<12>(b1, b0), 1);
        // bytes 24..48 -> lanes [b1[8..] ++ b2[..8] | b2[4..]]
        let full1 = _mm256_inserti128_si256(
            _mm256_castsi128_si256(_mm_alignr_epi8::<8>(b2, b1)),
            _mm_alignr_epi8::<4>(b2, b2),
            1,
        );
        let e0 = _mm256_or_si256(_mm256_shuffle_epi8(full0, mask), padv);
        let e1 = _mm256_or_si256(_mm256_shuffle_epi8(full1, mask), padv);
        // SAFETY: `cover_row` passes px + 16 <= npix, so the 64 bytes at
        // px * 4 are inside dst (npix * 4 elements, the caller's contract).
        unsafe {
            _mm256_storeu_si256(dp.add(px * 4) as *mut __m256i, e0);
            _mm256_storeu_si256(dp.add(px * 4 + 32) as *mut __m256i, e1);
        }
    };
    cover_row(npix, BLOCK_PX, block);
}

/// AVX-512 variant of `bilinear_row_rgb_to_rgbx_fx_avx2`: 32 px/iter as three
/// 32-element fx blends (the `bilinear_row_u8_fx_avx512` arithmetic, so
/// bit-identical) expanded by `store_rgbx_32px_avx512`. Overlapping last
/// block; rows under 32 px go to the AVX2 kernel.
///
/// # Safety
/// AVX-512 F and BW are available, and `bilinear_row_rgb_to_rgbx_fx_avx2`'s
/// bounds hold.
#[target_feature(enable = "avx512f,avx512bw")]
#[expect(
    clippy::too_many_arguments,
    reason = "bilinear_row's contract plus `pad`"
)]
pub(crate) unsafe fn bilinear_row_rgb_to_rgbx_fx_avx512(
    row0: &[u8],
    row1: &[u8],
    dst: &mut [u8],
    x0s: usize,
    npix: usize,
    step: usize,
    pad: u8,
    weights: [f32; 4],
) {
    use std::arch::x86_64::*;
    const BLOCK_PX: usize = 32;
    if npix < BLOCK_PX {
        // SAFETY: AVX-512 F implies AVX2; the bounds are this function's.
        unsafe {
            bilinear_row_rgb_to_rgbx_fx_avx2(row0, row1, dst, x0s, npix, step, pad, weights);
        }
        return;
    }
    debug_assert!(x0s + npix * 3 + step <= row0.len() && npix * 4 <= dst.len());
    let vw = weights.map(|weight| _mm512_set1_epi16(q15(weight)));
    let round = _mm512_set1_epi16(half_ulp(BILINEAR_FRAC_BITS));
    let (p0, p1, dp) = (row0.as_ptr(), row1.as_ptr(), dst.as_mut_ptr());
    let ld = |src_ptr: *const u8| {
        // SAFETY: `blend32` passes pointers to 32 bytes inside a row.
        let bytes = unsafe { _mm256_loadu_si256(src_ptr as *const __m256i) };
        _mm512_slli_epi16::<{ BILINEAR_FRAC_BITS as u32 }>(_mm512_cvtepu8_epi16(bytes))
    };
    // 32 blended RGB elements at element offset `off`, as bytes.
    let blend32 = |off: usize| -> __m256i {
        // SAFETY: `block` passes off <= x0s + npix * 3 - 32, so the 32 bytes
        // at off and at off + step end at or before the furthest tap, inside
        // both rows (the caller's contract).
        let sum = unsafe {
            _mm512_add_epi16(
                _mm512_add_epi16(
                    _mm512_mulhrs_epi16(ld(p0.add(off)), vw[0]),
                    _mm512_mulhrs_epi16(ld(p0.add(off + step)), vw[1]),
                ),
                _mm512_add_epi16(
                    _mm512_mulhrs_epi16(ld(p1.add(off)), vw[2]),
                    _mm512_mulhrs_epi16(ld(p1.add(off + step)), vw[3]),
                ),
            )
        };
        _mm512_cvtusepi16_epi8(_mm512_srli_epi16::<{ BILINEAR_FRAC_BITS as u32 }>(
            _mm512_add_epi16(sum, round),
        ))
    };
    // Output pixels px..px+32; reads 96 + step bytes from x0s + 3 px.
    let block = |px: usize| {
        let off = x0s + px * 3;
        let rgb = [blend32(off), blend32(off + 32), blend32(off + 64)];
        // SAFETY: AVX-512 F+BW are enabled here; `cover_row` passes
        // px + 32 <= npix, so the 128 bytes at px * 4 are inside dst (npix * 4
        // elements, the caller's contract).
        unsafe { store_rgbx_32px_avx512(dp.add(px * 4), rgb, pad) };
    };
    cover_row(npix, BLOCK_PX, block);
}

/// AVX-512 variant of the fixed-point u8 bilinear: 32 px/iter, and the 512
/// widen/narrow converts are sequence-preserving so the pack + lane fix-up
/// disappears. Same Q7 x Q15 mulhrs arithmetic as the AVX2 kernel (bit-
/// identical results); overlapping last block (see `cover_row`), rows under
/// 32 elements go to the AVX2 kernel.
///
/// # Safety
/// AVX-512 F and BW are available, and `ComposeSample::bilinear_row`'s
/// contract holds.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn bilinear_row_u8_fx_avx512(
    row0: &[u8],
    row1: &[u8],
    dst: &mut [u8],
    x0s: usize,
    len: usize,
    step: usize,
    weights: [f32; 4],
) {
    use std::arch::x86_64::*;
    const BLOCK: usize = 32;
    if len < BLOCK {
        // SAFETY: AVX-512 F implies AVX2 and FMA; the contract is the same.
        return unsafe { bilinear_row_u8_fx_avx2(row0, row1, dst, x0s, len, step, weights) };
    }
    debug_assert!(x0s + len + step <= row0.len() && len <= dst.len());
    let vw = weights.map(|weight| _mm512_set1_epi16(q15(weight)));
    let round = _mm512_set1_epi16(half_ulp(BILINEAR_FRAC_BITS));
    let (p0, p1, dp) = (row0.as_ptr(), row1.as_ptr(), dst.as_mut_ptr());
    let ld = |src_ptr: *const u8| {
        // SAFETY: `block` passes pointers to 32 bytes inside a row.
        let bytes = unsafe { _mm256_loadu_si256(src_ptr as *const __m256i) };
        _mm512_slli_epi16::<{ BILINEAR_FRAC_BITS as u32 }>(_mm512_cvtepu8_epi16(bytes))
    };
    let block = |element: usize| {
        let x0 = x0s + element;
        // SAFETY: `cover_row` passes element + 32 <= len, so the 32 bytes at
        // x0 and at x0 + step end at or before the furthest tap, inside both
        // rows (the caller's contract).
        let sum = unsafe {
            _mm512_add_epi16(
                _mm512_add_epi16(
                    _mm512_mulhrs_epi16(ld(p0.add(x0)), vw[0]),
                    _mm512_mulhrs_epi16(ld(p0.add(x0 + step)), vw[1]),
                ),
                _mm512_add_epi16(
                    _mm512_mulhrs_epi16(ld(p1.add(x0)), vw[2]),
                    _mm512_mulhrs_epi16(ld(p1.add(x0 + step)), vw[3]),
                ),
            )
        };
        let rounded =
            _mm512_srli_epi16::<{ BILINEAR_FRAC_BITS as u32 }>(_mm512_add_epi16(sum, round));
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

// load = widen 8 elements to f32x8; store = round (+0.5, truncate) and narrow.
// Integer stores clamp via packus (handles bicubic ringing under/overshoot).
gen_bilinear!(
    bilinear_row_u8_avx2,
    u8 => u8,
    |src_ptr: *const u8| {
        // SAFETY: the kernel passes pointers to 8 elements inside a row.
        let bytes = unsafe { _mm_loadl_epi64(src_ptr as *const __m128i) };
        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes))
    },
    |dst_ptr: *mut u8, value: __m256| {
        let vi = _mm256_cvttps_epi32(_mm256_add_ps(value, _mm256_set1_ps(0.5)));
        let u16s = _mm_packus_epi32(_mm256_castsi256_si128(vi), _mm256_extracti128_si256(vi, 1));
        // SAFETY: the kernel passes pointers to 8 elements inside dst.
        unsafe { _mm_storel_epi64(dst_ptr as *mut __m128i, _mm_packus_epi16(u16s, u16s)) };
    }
);
/// 8 f32 rounded (+0.5, truncate) and stored as u16, clamped by packus.
///
/// # Safety
/// `dst_ptr` is valid for 8 u16 writes.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn store_u16x8_rounded(dst_ptr: *mut u16, value: std::arch::x86_64::__m256) {
    use std::arch::x86_64::*;
    let vi = _mm256_cvttps_epi32(_mm256_add_ps(value, _mm256_set1_ps(0.5)));
    let u16s = _mm_packus_epi32(_mm256_castsi256_si128(vi), _mm256_extracti128_si256(vi, 1));
    // SAFETY: 8 u16 writes at dst_ptr (the caller's contract).
    unsafe { _mm_storeu_si128(dst_ptr as *mut __m128i, u16s) };
}

/// u16 samples widened to f32 (exactly): `dst[x] = src[x]`, 8 per step;
/// scalar tail.
#[target_feature(enable = "avx2")]
pub(crate) fn widen_u16_row_avx2(src: &[u16], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let len = src.len().min(dst.len());
    let (src_ptr, dst_ptr) = (src.as_ptr(), dst.as_mut_ptr());
    let mut x = 0;
    while x + 8 <= len {
        // SAFETY: x + 8 <= len: 8 samples of both rows.
        unsafe {
            let samples = _mm_loadu_si128(src_ptr.add(x) as *const __m128i);
            _mm256_storeu_ps(
                dst_ptr.add(x),
                _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(samples)),
            );
        }
        x += 8;
    }
    for (sample, widened) in src[x..len].iter().zip(&mut dst[x..len]) {
        *widened = f32::from(*sample);
    }
}

gen_bilinear!(
    bilinear_row_u16_avx2,
    u16 => u16,
    |src_ptr: *const u16| {
        // SAFETY: the kernel passes pointers to 8 elements inside a row.
        let samples = unsafe { _mm_loadu_si128(src_ptr as *const __m128i) };
        _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(samples))
    },
    store_u16x8_rounded
);
// `bilinear_row_u16_avx2` on rows already widened to f32 (exactly, by
// `widen_u16_row_avx2`): the same FMA chain and rounding, so the same
// results, without widening each sample once per tap.
gen_bilinear!(
    bilinear_row_widened_u16_avx2,
    f32 => u16,
    // SAFETY: the kernel passes pointers to 8 elements inside a row.
    |src_ptr: *const f32| unsafe { _mm256_loadu_ps(src_ptr) },
    store_u16x8_rounded
);
gen_bilinear!(
    bilinear_row_f32_avx2,
    f32 => f32,
    // SAFETY: the kernel passes pointers to 8 elements inside a row.
    |src_ptr: *const f32| unsafe { _mm256_loadu_ps(src_ptr) },
    // SAFETY: the kernel passes pointers to 8 elements inside dst.
    |dst_ptr: *mut f32, value: __m256| unsafe { _mm256_storeu_ps(dst_ptr, value) }
);

/// AVX-512 f32 bilinear: 16-wide FMA, same association as the AVX2 kernel
/// (bit-identical results); tail delegates to it.
///
/// # Safety
/// AVX-512 F is available, and `ComposeSample::bilinear_row`'s contract
/// holds.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn bilinear_row_f32_avx512(
    row0: &[f32],
    row1: &[f32],
    dst: &mut [f32],
    x0s: usize,
    len: usize,
    step: usize,
    weights: [f32; 4],
) {
    use std::arch::x86_64::*;
    let tw = row0.len();
    let (vw0, vw1, vw2, vw3) = (
        _mm512_set1_ps(weights[0]),
        _mm512_set1_ps(weights[1]),
        _mm512_set1_ps(weights[2]),
        _mm512_set1_ps(weights[3]),
    );
    let (p0, p1, dp) = (row0.as_ptr(), row1.as_ptr(), dst.as_mut_ptr());
    let mut element = 0usize;
    while element + 16 <= len && x0s + element + 16 + step <= tw {
        let x0 = x0s + element;
        // SAFETY: element + 16 <= len, so the 16 elements at x0 and at
        // x0 + step end at or before the furthest tap, inside both rows, and
        // the 16 at `element` inside dst (the caller's contract).
        unsafe {
            let mut blend = _mm512_mul_ps(_mm512_loadu_ps(p0.add(x0)), vw0);
            blend = _mm512_fmadd_ps(_mm512_loadu_ps(p0.add(x0 + step)), vw1, blend);
            blend = _mm512_fmadd_ps(_mm512_loadu_ps(p1.add(x0)), vw2, blend);
            blend = _mm512_fmadd_ps(_mm512_loadu_ps(p1.add(x0 + step)), vw3, blend);
            _mm512_storeu_ps(dp.add(element), blend);
        }
        element += 16;
    }
    if element < len {
        // SAFETY: AVX-512 F implies AVX2 and FMA; the remaining outputs keep
        // the contract (same taps, a dst suffix).
        unsafe {
            bilinear_row_f32_avx2(
                row0,
                row1,
                &mut dst[element..],
                x0s + element,
                len - element,
                step,
                weights,
            );
        }
    }
}

// f32 only: on AVX2 CPUs u8/u16 always take the separable bicubic instead
// (see ComposeSample::SEP_BICUBIC).
gen_bicubic!(
    bicubic_row_f32_avx2,
    f32,
    // SAFETY: the kernel passes pointers to 8 elements inside a row.
    |src_ptr: *const f32| unsafe { _mm256_loadu_ps(src_ptr) },
    // SAFETY: the kernel passes pointers to 8 elements inside dst.
    |dst_ptr: *mut f32, value: __m256| unsafe { _mm256_storeu_ps(dst_ptr, value) }
);

/// AVX-512 f32 direct bicubic: 16-wide FMA, same association as the AVX2
/// kernel (bit-identical results); tail delegates to it.
///
/// # Safety
/// AVX-512 F is available, and `ComposeSample::bicubic_row`'s contract holds.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn bicubic_row_f32_avx512(
    rows: [&[f32]; 4],
    dst: &mut [f32],
    x0s: usize,
    len: usize,
    step: usize,
    wx: [f32; 4],
    wy: [f32; 4],
) {
    use std::arch::x86_64::*;
    let tw = rows[0].len();
    let vwx = [
        _mm512_set1_ps(wx[0]),
        _mm512_set1_ps(wx[1]),
        _mm512_set1_ps(wx[2]),
        _mm512_set1_ps(wx[3]),
    ];
    let vwy = [
        _mm512_set1_ps(wy[0]),
        _mm512_set1_ps(wy[1]),
        _mm512_set1_ps(wy[2]),
        _mm512_set1_ps(wy[3]),
    ];
    let ptrs = [
        rows[0].as_ptr(),
        rows[1].as_ptr(),
        rows[2].as_ptr(),
        rows[3].as_ptr(),
    ];
    let dp = dst.as_mut_ptr();
    let mut element = 0usize;
    while element + 16 <= len && x0s + element + 16 + 2 * step <= tw {
        let x0 = x0s + element;
        let mut acc = _mm512_setzero_ps();
        for tap in 0..4 {
            let row_ptr = ptrs[tap];
            // SAFETY: x0 >= x0s >= step and element + 16 <= len, so the 16
            // elements at each tap x0 - step ..= x0 + 2 * step end at or
            // before the furthest tap, inside every row (the caller's contract).
            unsafe {
                let mut sum = _mm512_mul_ps(_mm512_loadu_ps(row_ptr.add(x0 - step)), vwx[0]);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(row_ptr.add(x0)), vwx[1], sum);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(row_ptr.add(x0 + step)), vwx[2], sum);
                sum = _mm512_fmadd_ps(_mm512_loadu_ps(row_ptr.add(x0 + 2 * step)), vwx[3], sum);
                acc = _mm512_fmadd_ps(sum, vwy[tap], acc);
            }
        }
        // SAFETY: element + 16 <= len <= dst.len() (the caller's contract).
        unsafe { _mm512_storeu_ps(dp.add(element), acc) };
        element += 16;
    }
    if element < len {
        // SAFETY: AVX-512 F implies AVX2 and FMA; the remaining outputs keep
        // the contract (same taps, a dst suffix).
        unsafe {
            bicubic_row_f32_avx2(
                rows,
                &mut dst[element..],
                x0s + element,
                len - element,
                step,
                wx,
                wy,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::rows::tests::{
        BICUBIC_ROUNDING_STEPS, BILINEAR_ROUNDING_STEPS, BilinearCase, TestSample,
        assert_bicubic_rows_match, assert_bilinear_rows_match, assert_close, bilinear_cases,
        guarded_output, row_lengths, run_bilinear,
    };
    use crate::composite::rows::{scalar_bicubic_row, scalar_bilinear_row};

    fn avx2_fma_detected() -> bool {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }

    /// The AVX-512 kernels' features, plus the AVX2 and FMA of the kernels
    /// their short rows fall back to.
    fn avx512_detected() -> bool {
        avx2_fma_detected()
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
    }

    #[test]
    fn bilinear_avx2_matches_scalar() {
        if !avx2_fma_detected() {
            return;
        }
        // SAFETY: AVX2 and FMA are available.
        unsafe {
            assert_bilinear_rows_match::<u8>(
                bilinear_row_u8_avx2,
                scalar_bilinear_row::<u8>,
                Some(BILINEAR_ROUNDING_STEPS),
            );
            assert_bilinear_rows_match::<u16>(
                bilinear_row_u16_avx2,
                scalar_bilinear_row::<u16>,
                Some(BILINEAR_ROUNDING_STEPS),
            );
            assert_bilinear_rows_match::<f32>(
                bilinear_row_f32_avx2,
                scalar_bilinear_row::<f32>,
                Some(BILINEAR_ROUNDING_STEPS),
            );
        }
    }

    #[test]
    fn bilinear_u8_fixed_point_avx2_within_one_lsb_of_scalar() {
        if !avx2_fma_detected() {
            return;
        }
        // SAFETY: AVX2 and FMA are available.
        unsafe {
            assert_bilinear_rows_match::<u8>(
                bilinear_row_u8_fx_avx2,
                scalar_bilinear_row::<u8>,
                Some(BILINEAR_ROUNDING_STEPS),
            );
        }
    }

    #[test]
    fn bilinear_avx512_equals_avx2() {
        if !avx512_detected() {
            return;
        }
        // SAFETY: AVX-512 F+BW, AVX2 and FMA are available.
        unsafe {
            assert_bilinear_rows_match::<u8>(
                bilinear_row_u8_fx_avx512,
                bilinear_row_u8_fx_avx2,
                None,
            );
            assert_bilinear_rows_match::<f32>(bilinear_row_f32_avx512, bilinear_row_f32_avx2, None);
        }
    }

    #[test]
    fn bicubic_f32_avx2_matches_scalar() {
        if !avx2_fma_detected() {
            return;
        }
        // SAFETY: AVX2 and FMA are available.
        unsafe {
            assert_bicubic_rows_match::<f32>(
                bicubic_row_f32_avx2,
                scalar_bicubic_row::<f32>,
                Some(BICUBIC_ROUNDING_STEPS),
            );
        }
    }

    #[test]
    fn bicubic_f32_avx512_equals_avx2() {
        if !avx512_detected() {
            return;
        }
        // SAFETY: AVX-512 F, AVX2 and FMA are available.
        unsafe {
            assert_bicubic_rows_match::<f32>(bicubic_row_f32_avx512, bicubic_row_f32_avx2, None);
        }
    }

    const PAD: u8 = 0x5A;

    /// An RGB -> RGBX bilinear kernel.
    type RgbxBilinearRow = unsafe fn(&[u8], &[u8], &mut [u8], usize, usize, usize, u8, [f32; 4]);

    /// RGB rows of every pixel count of `row_lengths`, taps 3 elements apart
    /// or (integer-aligned) 0.
    fn rgb_cases() -> Vec<BilinearCase<u8>> {
        bilinear_cases(21, &[0, 3], row_lengths().map(|npix| npix * 3))
    }

    /// Runs `kernel` on `case` (`len / 3` pixels) into a guarded RGBX output.
    ///
    /// # Safety
    /// The CPU has the features `kernel` needs.
    unsafe fn run_rgbx(kernel: RgbxBilinearRow, case: &BilinearCase<u8>) -> Vec<u8> {
        let npix = case.len / 3;
        guarded_output(npix * 4, u8::SENTINEL, |dst| {
            // SAFETY: the rows end at the furthest tap of npix * 3 outputs and
            // dst holds npix * 4 elements; the features are the caller's.
            unsafe {
                kernel(
                    case.row(0),
                    case.row(1),
                    dst,
                    case.x0s,
                    npix,
                    case.step,
                    PAD,
                    case.weights,
                );
            }
        })
    }

    /// The scalar RGB bilinear row expanded to RGBX.
    fn rgbx_reference(case: &BilinearCase<u8>) -> Vec<u8> {
        // SAFETY: the scalar kernel needs no CPU features.
        let rgb = unsafe { run_bilinear(scalar_bilinear_row::<u8>, case) };
        rgb.as_chunks::<3>()
            .0
            .iter()
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], PAD])
            .collect()
    }

    #[test]
    fn bilinear_rgb_to_rgbx_avx2_within_one_lsb_of_scalar() {
        if !avx2_fma_detected() {
            return;
        }
        for case in rgb_cases() {
            // SAFETY: AVX2 and FMA are available.
            let simd = unsafe { run_rgbx(bilinear_row_rgb_to_rgbx_fx_avx2, &case) };
            let expected = rgbx_reference(&case);
            // Rows under one 16-pixel block take the f32 scalar loop: exact.
            let tolerance = if case.len < 16 * 3 { 0.0 } else { 1.0 };
            assert_close(&simd, &expected, tolerance, &case.describe());
            assert!(
                simd.as_chunks::<4>().0.iter().all(|pixel| pixel[3] == PAD),
                "{}",
                case.describe()
            );
        }
    }

    #[test]
    fn bilinear_rgb_to_rgbx_avx512_equals_avx2() {
        if !avx512_detected() {
            return;
        }
        for case in rgb_cases() {
            // SAFETY: AVX-512 F+BW, AVX2 and FMA are available.
            let (avx512, avx2) = unsafe {
                (
                    run_rgbx(bilinear_row_rgb_to_rgbx_fx_avx512, &case),
                    run_rgbx(bilinear_row_rgb_to_rgbx_fx_avx2, &case),
                )
            };
            assert_eq!(avx512, avx2, "{}", case.describe());
        }
    }

    #[test]
    fn bilinear_rgb_to_rgbx_equals_planar_fixed_point_expanded() {
        // Same arithmetic per element as the u8 fixed-point row once both
        // take vector blocks (16 pixels = 48 elements on either tier).
        if !avx2_fma_detected() {
            return;
        }
        for case in rgb_cases().iter().filter(|case| case.len >= 16 * 3) {
            // SAFETY: AVX2 and FMA are available.
            let (rgbx, rgb) = unsafe {
                (
                    run_rgbx(bilinear_row_rgb_to_rgbx_fx_avx2, case),
                    run_bilinear(bilinear_row_u8_fx_avx2, case),
                )
            };
            let expanded: Vec<u8> = rgb
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], PAD])
                .collect();
            assert_eq!(rgbx, expanded, "{}", case.describe());
        }
    }
}

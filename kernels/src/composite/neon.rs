//! NEON compose row kernels (aarch64): bilinear (f32 for uint16/float32, Q15
//! fixed point for uint8), the two passes of the separable bicubic, and the
//! RGB -> RGBX variants of the uint8 kernels.
//!
//! The uint8 kernels use `vqrdmulh`, which is x86's `mulhrs`
//! (`(value * weight + 2^14) >> 15`, saturating only at -32768 * -32768,
//! unreachable for weights above -1), and `vqrshrun`, the rounding shift and
//! unsigned saturating narrow in one — so they compute what the x86
//! fixed-point kernels and the scalar mirrors in `rows` compute, bit for bit.
//!
//! # Safety
//! NEON is baseline on aarch64, gated only by the `set_simd` kill switch.
//! Every `unsafe fn` here requires the bounds its caller documents. Loads and
//! stores are justified against those bounds at their loop guards.

use crate::cpu::has_rdm;
use std::arch::aarch64::{
    float32x4_t, int16x8_t, uint8x8x4_t, uint8x16_t, uint8x16x4_t, uint32x4_t, vaddq_f32,
    vaddq_s16, vcombine_u8, vcombine_u16, vcvtq_f32_u32, vcvtq_u32_f32, vdup_n_u8, vdupq_n_f32,
    vdupq_n_s16, vdupq_n_u8, vfmaq_n_f32, vget_low_u8, vget_low_u16, vld1_u8, vld1q_f32, vld1q_s16,
    vld1q_u8, vld1q_u16, vld3q_s16, vld3q_u8, vmovl_high_u16, vmovl_u8, vmovl_u16, vmulq_n_f32,
    vqmovn_u16, vqmovn_u32, vqrdmlahq_s16, vqrdmulhq_n_s16, vqrshrun_n_s16, vreinterpretq_s16_u16,
    vshll_high_n_u8, vshll_n_u8, vst1_u8, vst1q_f32, vst1q_s16, vst1q_u8, vst1q_u16, vst4_u8,
    vst4q_u8,
};

use super::rows::{
    scalar_bilinear_row, scalar_bilinear_row_rgb_to_rgbx, scalar_hconv4_row,
    scalar_hconv4_row_u8_fx, scalar_vconv4_row, scalar_vconv4_row_rgb_to_rgbx,
    scalar_vconv4_row_u8_fx,
};
use crate::sample::{BICUBIC_U8_FRAC_BITS, BILINEAR_U8_FRAC_BITS, RGB, RGBX, Sample, q15};
use crate::simd::cover_row;

const BILINEAR_Q: i32 = BILINEAR_U8_FRAC_BITS as i32;
const BICUBIC_Q: i32 = BICUBIC_U8_FRAC_BITS as i32;

/// Eight samples as two f32x4 halves.
type F32x8 = (float32x4_t, float32x4_t);

/// Eight u8 at `src` as f32.
///
/// # Safety
/// `src..src + 8` must be readable.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_u8x8_f32(src: *const u8) -> F32x8 {
    // SAFETY: the caller's contract.
    let samples = vmovl_u8(unsafe { vld1_u8(src) });
    (
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(samples))),
        vcvtq_f32_u32(vmovl_high_u16(samples)),
    )
}

/// Eight u16 at `src` as f32.
///
/// # Safety
/// `src..src + 8` must be readable.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_u16x8_f32(src: *const u16) -> F32x8 {
    // SAFETY: the caller's contract.
    let samples = unsafe { vld1q_u16(src) };
    (
        vcvtq_f32_u32(vmovl_u16(vget_low_u16(samples))),
        vcvtq_f32_u32(vmovl_high_u16(samples)),
    )
}

/// Eight f32 at `src`.
///
/// # Safety
/// `src..src + 8` must be readable.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_f32x8(src: *const f32) -> F32x8 {
    // SAFETY: the caller's contract.
    unsafe { (vld1q_f32(src), vld1q_f32(src.add(4))) }
}

/// Eight f32 rounded and saturated to u32 as the scalar `(value + 0.5) as uN`:
/// `vcvtq_u32` truncates and clamps negatives to 0.
#[target_feature(enable = "neon")]
#[inline]
fn round_to_u32(values: F32x8) -> (uint32x4_t, uint32x4_t) {
    let half = vdupq_n_f32(0.5);
    (
        vcvtq_u32_f32(vaddq_f32(values.0, half)),
        vcvtq_u32_f32(vaddq_f32(values.1, half)),
    )
}

/// Store eight f32 as u8, rounded and saturated.
///
/// # Safety
/// `dst..dst + 8` must be writable.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn store_u8x8_f32(dst: *mut u8, values: F32x8) {
    let (low, high) = round_to_u32(values);
    let narrowed = vqmovn_u16(vcombine_u16(vqmovn_u32(low), vqmovn_u32(high)));
    // SAFETY: the caller's contract.
    unsafe { vst1_u8(dst, narrowed) };
}

/// Store eight f32 as u16, rounded and saturated.
///
/// # Safety
/// `dst..dst + 8` must be writable.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn store_u16x8_f32(dst: *mut u16, values: F32x8) {
    let (low, high) = round_to_u32(values);
    // SAFETY: the caller's contract.
    unsafe { vst1q_u16(dst, vcombine_u16(vqmovn_u32(low), vqmovn_u32(high))) };
}

/// Store eight f32.
///
/// # Safety
/// `dst..dst + 8` must be writable.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn store_f32x8(dst: *mut f32, values: F32x8) {
    // SAFETY: the caller's contract.
    unsafe {
        vst1q_f32(dst, values.0);
        vst1q_f32(dst.add(4), values.1);
    }
}

/// `taps[0] * weights[0] + ... + taps[3] * weights[3]`, multiplied then
/// fused-multiply-added in tap order (the x86 FMA kernels' order).
#[target_feature(enable = "neon")]
#[inline]
fn dot4_f32x8(taps: [F32x8; 4], weights: [f32; 4]) -> F32x8 {
    let mut low = vmulq_n_f32(taps[0].0, weights[0]);
    let mut high = vmulq_n_f32(taps[0].1, weights[0]);
    for tap in 1..4 {
        low = vfmaq_n_f32(low, taps[tap].0, weights[tap]);
        high = vfmaq_n_f32(high, taps[tap].1, weights[tap]);
    }
    (low, high)
}

// One f32 bilinear body, specialized per dtype by `$load` (widen 8 elements)
// and `$store` (round and narrow 8).
macro_rules! gen_bilinear_neon {
    ($name:ident, $t:ty, $load:ident, $store:ident) => {
        /// f32 bilinear row (`ComposeSample::bilinear_row`), 8 elements per
        /// step; scalar tail.
        ///
        /// # Safety
        /// The bounds of `ComposeSample::bilinear_row`.
        #[target_feature(enable = "neon")]
        pub(crate) unsafe fn $name(
            row0: &[$t],
            row1: &[$t],
            dst: &mut [$t],
            x0s: usize,
            len: usize,
            step: usize,
            weights: [f32; 4],
        ) {
            let (top, bottom, dst_ptr) = (row0.as_ptr(), row1.as_ptr(), dst.as_mut_ptr());
            let mut element = 0usize;
            // 8-wide while the +step-offset 8-element load stays in the row.
            while element + 8 <= len && x0s + element + 8 + step <= row0.len() {
                let x0 = x0s + element;
                // SAFETY: x0 + step + 8 <= row length (loop guard), both rows
                // have that length, and element + 8 <= len <= dst length.
                unsafe {
                    let taps = [
                        $load(top.add(x0)),
                        $load(top.add(x0 + step)),
                        $load(bottom.add(x0)),
                        $load(bottom.add(x0 + step)),
                    ];
                    $store(dst_ptr.add(element), dot4_f32x8(taps, weights));
                }
                element += 8;
            }
            if element < len {
                // SAFETY: the rest of the row, under the caller's contract.
                unsafe {
                    scalar_bilinear_row::<$t>(
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
    };
}

gen_bilinear_neon!(bilinear_row_u8_f32_neon, u8, load_u8x8_f32, store_u8x8_f32);
gen_bilinear_neon!(bilinear_row_u16_neon, u16, load_u16x8_f32, store_u16x8_f32);
gen_bilinear_neon!(bilinear_row_f32_neon, f32, load_f32x8, store_f32x8);

/// Sum of the four taps times their Q15 weights (`vqrdmulh`, the `mulhrs`
/// twin), in wrapping i16 — the fixed-point kernels' 4-tap dot.
#[target_feature(enable = "neon")]
#[inline]
fn dot4_q15(taps: [int16x8_t; 4], weights: [i16; 4]) -> int16x8_t {
    vaddq_s16(
        vaddq_s16(
            vqrdmulhq_n_s16(taps[0], weights[0]),
            vqrdmulhq_n_s16(taps[1], weights[1]),
        ),
        vaddq_s16(
            vqrdmulhq_n_s16(taps[2], weights[2]),
            vqrdmulhq_n_s16(taps[3], weights[3]),
        ),
    )
}

/// `dot4_q15` as a chain of `vqrdmlah` (ARMv8.1 RDM), which saturates where
/// it wraps: the same, since the fixed-point kernels' partial sums stay in
/// i16 (a convex bilinear blend within [0, 32643]; catmull-rom's positive
/// weights sum to at most 1.125, so the bicubic passes stay below 21 000).
#[target_feature(enable = "neon,rdm")]
#[inline]
fn dot4_q15_rdm(taps: [int16x8_t; 4], weights: [i16; 4]) -> int16x8_t {
    let mut sums = vqrdmulhq_n_s16(taps[0], weights[0]);
    for tap in 1..4 {
        sums = vqrdmlahq_s16(sums, taps[tap], vdupq_n_s16(weights[tap]));
    }
    sums
}

/// The low and high halves of 16 u8, widened to i16 with `FRAC_BITS`
/// fraction bits.
#[target_feature(enable = "neon")]
#[inline]
fn widen_q<const FRAC_BITS: i32>(samples: uint8x16_t) -> (int16x8_t, int16x8_t) {
    (
        vreinterpretq_s16_u16(vshll_n_u8::<FRAC_BITS>(vget_low_u8(samples))),
        vreinterpretq_s16_u16(vshll_high_n_u8::<FRAC_BITS>(samples)),
    )
}

/// u16 samples widened to f32 (exactly): `dst[x] = src[x]`, 8 per step;
/// scalar tail.
#[target_feature(enable = "neon")]
pub(crate) fn widen_u16_row_neon(src: &[u16], dst: &mut [f32]) {
    let len = src.len().min(dst.len());
    let (src_ptr, dst_ptr) = (src.as_ptr(), dst.as_mut_ptr());
    let mut x = 0;
    while x + 8 <= len {
        // SAFETY: x + 8 <= len: 8 samples of both rows.
        unsafe { store_f32x8(dst_ptr.add(x), load_u16x8_f32(src_ptr.add(x))) };
        x += 8;
    }
    for (sample, widened) in src[x..len].iter().zip(&mut dst[x..len]) {
        *widened = sample.to_f32();
    }
}

/// `bilinear_row_u16_neon` on rows already widened to f32: the same taps,
/// FMA chain and rounding, so the same results, without widening each
/// sample once per tap. Scalar tail as the scalar reference (on the same
/// f32 values).
///
/// # Safety
/// The bounds of `ComposeSample::bilinear_row`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bilinear_row_widened_u16_neon(
    row0: &[f32],
    row1: &[f32],
    dst: &mut [u16],
    x0s: usize,
    len: usize,
    step: usize,
    weights: [f32; 4],
) {
    let (top, bottom, dst_ptr) = (row0.as_ptr(), row1.as_ptr(), dst.as_mut_ptr());
    let mut element = 0usize;
    // 8-wide while the +step-offset 8-element load stays in the row.
    while element + 8 <= len && x0s + element + 8 + step <= row0.len() {
        let x0 = x0s + element;
        // SAFETY: x0 + step + 8 <= row length (loop guard), both rows have
        // that length, and element + 8 <= len <= dst length.
        unsafe {
            let taps = [
                load_f32x8(top.add(x0)),
                load_f32x8(top.add(x0 + step)),
                load_f32x8(bottom.add(x0)),
                load_f32x8(bottom.add(x0 + step)),
            ];
            store_u16x8_f32(dst_ptr.add(element), dot4_f32x8(taps, weights));
        }
        element += 8;
    }
    for (element, out) in dst.iter_mut().enumerate().take(len).skip(element) {
        let x0 = x0s + element;
        let value = weights[0] * row0[x0]
            + weights[1] * row0[x0 + step]
            + weights[2] * row1[x0]
            + weights[3] * row1[x0 + step];
        *out = u16::from_f32(value);
    }
}

// The fixed-point u8 kernels, compiled for the target features `$features`
// with `$dot` the 4-tap dot: once plain, once with RDM's `vqrdmlah`.
macro_rules! gen_fixed_point_kernels {
    ($features:literal, $dot:ident) => {
        /// 16 fixed-point bilinear blends (Q7 samples, Q15 weights) of the four
        /// taps' u8 vectors, rounded back to u8. A convex blend keeps the sum in
        /// [0, 32643]: no wraparound, and the narrow saturates nothing.
        #[target_feature(enable = $features)]
        #[inline]
        fn bilinear16_fx(taps: [uint8x16_t; 4], weights: [i16; 4]) -> uint8x16_t {
            let widened = taps.map(|tap| widen_q::<BILINEAR_Q>(tap));
            vcombine_u8(
                vqrshrun_n_s16::<BILINEAR_Q>($dot(widened.map(|halves| halves.0), weights)),
                vqrshrun_n_s16::<BILINEAR_Q>($dot(widened.map(|halves| halves.1), weights)),
            )
        }

        /// Fixed-point u8 bilinear row (`ComposeSample::bilinear_row`): 16 elements
        /// per block, the last block overlapping its predecessor (`cover_row`); rows
        /// under one block take the f32 kernel.
        ///
        /// # Safety
        /// The bounds of `ComposeSample::bilinear_row`.
        #[target_feature(enable = $features)]
        pub(super) unsafe fn bilinear_row_u8_fx_neon(
            row0: &[u8],
            row1: &[u8],
            dst: &mut [u8],
            x0s: usize,
            len: usize,
            step: usize,
            weights: [f32; 4],
        ) {
            const BLOCK: usize = 16;
            if len < BLOCK {
                // SAFETY: the caller's bounds, which the f32 kernel shares.
                return unsafe {
                    bilinear_row_u8_f32_neon(row0, row1, dst, x0s, len, step, weights)
                };
            }
            debug_assert!(x0s + len + step <= row0.len() && len <= dst.len());
            let qweights = weights.map(q15);
            let (top, bottom, dst_ptr) = (row0.as_ptr(), row1.as_ptr(), dst.as_mut_ptr());
            cover_row(len, BLOCK, |start| {
                let x0 = x0s + start;
                // SAFETY: start + 16 <= len, so the taps read at most x0s + len + step
                // elements of each row and the store stays in dst (caller's bounds).
                unsafe {
                    let taps = [
                        vld1q_u8(top.add(x0)),
                        vld1q_u8(top.add(x0 + step)),
                        vld1q_u8(bottom.add(x0)),
                        vld1q_u8(bottom.add(x0 + step)),
                    ];
                    vst1q_u8(dst_ptr.add(start), bilinear16_fx(taps, qweights));
                }
            });
        }

        /// Fused fixed-point bilinear, RGB rows in, RGBX out (`pad` in X): `vld3`
        /// splits 16 RGB pixels per tap into channel planes (the +step tap is whole
        /// pixels away, so it splits the same way), each plane blends as in
        /// `bilinear_row_u8_fx_neon`, and `vst4` interleaves them with the pad
        /// plane. 16 pixels per block, overlapping last block; rows under one block
        /// take the scalar f32 formula.
        ///
        /// # Safety
        /// The bounds of `ComposeSample::bilinear_row` for `pixels * 3` elements, and
        /// `dst` holds `pixels * 4` samples.
        #[target_feature(enable = $features)]
        #[expect(
            clippy::too_many_arguments,
            reason = "bilinear_row's contract plus pad"
        )]
        pub(super) unsafe fn bilinear_row_rgb_to_rgbx_fx_neon(
            row0: &[u8],
            row1: &[u8],
            dst: &mut [u8],
            x0s: usize,
            pixels: usize,
            step: usize,
            pad: u8,
            weights: [f32; 4],
        ) {
            const BLOCK_PX: usize = 16;
            if pixels < BLOCK_PX {
                return scalar_bilinear_row_rgb_to_rgbx(
                    row0, row1, dst, x0s, pixels, step, pad, weights,
                );
            }
            debug_assert!(x0s + pixels * RGB + step <= row0.len() && pixels * RGBX <= dst.len());
            let qweights = weights.map(q15);
            let pad_plane = vdupq_n_u8(pad);
            let (top, bottom, dst_ptr) = (row0.as_ptr(), row1.as_ptr(), dst.as_mut_ptr());
            cover_row(pixels, BLOCK_PX, |start| {
                let offset = x0s + start * RGB;
                // SAFETY: start + 16 <= pixels, so each 48-byte tap load ends by
                // x0s + 3 pixels + step and the 64-byte store by 4 pixels (caller's
                // bounds).
                unsafe {
                    let [left0, right0, left1, right1] = [
                        vld3q_u8(top.add(offset)),
                        vld3q_u8(top.add(offset + step)),
                        vld3q_u8(bottom.add(offset)),
                        vld3q_u8(bottom.add(offset + step)),
                    ];
                    let rgbx = uint8x16x4_t(
                        bilinear16_fx([left0.0, right0.0, left1.0, right1.0], qweights),
                        bilinear16_fx([left0.1, right0.1, left1.1, right1.1], qweights),
                        bilinear16_fx([left0.2, right0.2, left1.2, right1.2], qweights),
                        pad_plane,
                    );
                    vst4q_u8(dst_ptr.add(start * RGBX), rgbx);
                }
            });
        }

        /// Fixed-point u8 bicubic horizontal pass (`ComposeSample::hconv4_row`): Q6
        /// samples, Q15 weights, wrapping i16 sums (the scalar mirror's arithmetic).
        /// 16 elements per block, overlapping last block; rows under one block go
        /// scalar.
        ///
        /// # Safety
        /// The bounds of `ComposeSample::hconv4_row`.
        #[target_feature(enable = $features)]
        pub(super) unsafe fn hconv4_row_u8_fx_neon(
            src: &[u8],
            dst: &mut [i16],
            x0s: usize,
            len: usize,
            step: usize,
            wx: [f32; 4],
        ) {
            const BLOCK: usize = 16;
            if len < BLOCK {
                return scalar_hconv4_row_u8_fx(src, dst, x0s, len, step, wx);
            }
            debug_assert!(x0s >= step && x0s + len + 2 * step <= src.len() && len <= dst.len());
            let qweights = wx.map(q15);
            let (src_ptr, dst_ptr) = (src.as_ptr(), dst.as_mut_ptr());
            cover_row(len, BLOCK, |start| {
                let left = x0s + start - step;
                // SAFETY: start + 16 <= len, so the taps read src[x0s - step ..
                // x0s + len + 2 step] and the stores write dst[..len] (caller's bounds).
                unsafe {
                    let taps = [0, 1, 2, 3]
                        .map(|tap| widen_q::<BICUBIC_Q>(vld1q_u8(src_ptr.add(left + tap * step))));
                    vst1q_s16(
                        dst_ptr.add(start),
                        $dot(taps.map(|halves| halves.0), qweights),
                    );
                    vst1q_s16(
                        dst_ptr.add(start + 8),
                        $dot(taps.map(|halves| halves.1), qweights),
                    );
                }
            });
        }

        /// Fixed-point u8 bicubic vertical pass over four Q6 i16 rows
        /// (`ComposeSample::vconv4_row`): `vqrshrun` = `(sum + 32) >> 6` clamped to
        /// [0, 255]. 16 elements per block, overlapping last block; rows under one
        /// block go scalar.
        ///
        /// # Safety
        /// The bounds of `ComposeSample::vconv4_row`.
        #[target_feature(enable = $features)]
        pub(super) unsafe fn vconv4_row_u8_fx_neon(
            rows: [&[i16]; 4],
            dst: &mut [u8],
            len: usize,
            wy: [f32; 4],
        ) {
            const BLOCK: usize = 16;
            if len < BLOCK {
                return scalar_vconv4_row_u8_fx(rows, dst, len, wy);
            }
            debug_assert!(rows.iter().all(|row| row.len() >= len) && len <= dst.len());
            let qweights = wy.map(q15);
            let row_ptrs = rows.map(<[i16]>::as_ptr);
            let dst_ptr = dst.as_mut_ptr();
            cover_row(len, BLOCK, |start| {
                // SAFETY: start + 16 <= len <= every row's and dst's length.
                unsafe {
                    let low = $dot(row_ptrs.map(|row| vld1q_s16(row.add(start))), qweights);
                    let high = $dot(row_ptrs.map(|row| vld1q_s16(row.add(start + 8))), qweights);
                    let narrowed = vcombine_u8(
                        vqrshrun_n_s16::<BICUBIC_Q>(low),
                        vqrshrun_n_s16::<BICUBIC_Q>(high),
                    );
                    vst1q_u8(dst_ptr.add(start), narrowed);
                }
            });
        }

        /// Separable-bicubic vertical pass, Q6 RGB rows in, RGBX out: `vld3` splits 8
        /// pixels of each row into channel planes, each plane runs the
        /// `vconv4_row_u8_fx_neon` arithmetic, and `vst4` interleaves them with
        /// `pad`. 8 pixels per block, overlapping last block; rows under one block
        /// go scalar.
        ///
        /// # Safety
        /// Every row holds `pixels * 3` elements and `dst` `pixels * 4`.
        #[target_feature(enable = $features)]
        pub(super) unsafe fn vconv4_row_rgb_to_rgbx_neon(
            rows: [&[i16]; 4],
            dst: &mut [u8],
            pixels: usize,
            pad: u8,
            wy: [f32; 4],
        ) {
            const BLOCK_PX: usize = 8;
            if pixels < BLOCK_PX {
                return scalar_vconv4_row_rgb_to_rgbx(rows, dst, pixels, pad, wy);
            }
            debug_assert!(
                rows.iter().all(|row| row.len() >= pixels * RGB) && pixels * RGBX <= dst.len()
            );
            let qweights = wy.map(q15);
            let row_ptrs = rows.map(<[i16]>::as_ptr);
            let pad_plane = vdup_n_u8(pad);
            let dst_ptr = dst.as_mut_ptr();
            let channel = |taps: [int16x8_t; 4]| vqrshrun_n_s16::<BICUBIC_Q>($dot(taps, qweights));
            cover_row(pixels, BLOCK_PX, |start| {
                // SAFETY: start + 8 <= pixels: the loads read row elements
                // [3 start, 3 start + 24) and the store writes [4 start, 4 start + 32).
                unsafe {
                    let [row0, row1, row2, row3] =
                        row_ptrs.map(|row| vld3q_s16(row.add(start * RGB)));
                    let rgbx = uint8x8x4_t(
                        channel([row0.0, row1.0, row2.0, row3.0]),
                        channel([row0.1, row1.1, row2.1, row3.1]),
                        channel([row0.2, row1.2, row2.2, row3.2]),
                        pad_plane,
                    );
                    vst4_u8(dst_ptr.add(start * RGBX), rgbx);
                }
            });
        }
    };
}

/// The fixed-point kernels with `dot4_q15`.
mod plain {
    use super::*;
    gen_fixed_point_kernels!("neon", dot4_q15);
}

/// The fixed-point kernels with `dot4_q15_rdm`.
mod rdm {
    use super::*;
    gen_fixed_point_kernels!("neon,rdm", dot4_q15_rdm);
}

/// The kernel of `plain`, or of `rdm` where the CPU has RDM (the same
/// results).
///
/// # Safety
/// As the kernels.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bilinear_row_u8_fx_neon(
    row0: &[u8],
    row1: &[u8],
    dst: &mut [u8],
    x0s: usize,
    len: usize,
    step: usize,
    weights: [f32; 4],
) {
    // SAFETY: the caller's bounds; RDM is detected for its kernel.
    unsafe {
        if has_rdm() {
            rdm::bilinear_row_u8_fx_neon(row0, row1, dst, x0s, len, step, weights);
        } else {
            plain::bilinear_row_u8_fx_neon(row0, row1, dst, x0s, len, step, weights);
        }
    }
}

/// The kernel of `plain`, or of `rdm` where the CPU has RDM (the same
/// results).
///
/// # Safety
/// As the kernels.
#[target_feature(enable = "neon")]
#[expect(
    clippy::too_many_arguments,
    reason = "bilinear_row's contract plus pad"
)]
pub(crate) unsafe fn bilinear_row_rgb_to_rgbx_fx_neon(
    row0: &[u8],
    row1: &[u8],
    dst: &mut [u8],
    x0s: usize,
    pixels: usize,
    step: usize,
    pad: u8,
    weights: [f32; 4],
) {
    // SAFETY: the caller's bounds; RDM is detected for its kernel.
    unsafe {
        if has_rdm() {
            rdm::bilinear_row_rgb_to_rgbx_fx_neon(row0, row1, dst, x0s, pixels, step, pad, weights);
        } else {
            plain::bilinear_row_rgb_to_rgbx_fx_neon(
                row0, row1, dst, x0s, pixels, step, pad, weights,
            );
        }
    }
}

/// The kernel of `plain`, or of `rdm` where the CPU has RDM (the same
/// results).
///
/// # Safety
/// As the kernels.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn hconv4_row_u8_fx_neon(
    src: &[u8],
    dst: &mut [i16],
    x0s: usize,
    len: usize,
    step: usize,
    wx: [f32; 4],
) {
    // SAFETY: the caller's bounds; RDM is detected for its kernel.
    unsafe {
        if has_rdm() {
            rdm::hconv4_row_u8_fx_neon(src, dst, x0s, len, step, wx);
        } else {
            plain::hconv4_row_u8_fx_neon(src, dst, x0s, len, step, wx);
        }
    }
}

/// The kernel of `plain`, or of `rdm` where the CPU has RDM (the same
/// results).
///
/// # Safety
/// As the kernels.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vconv4_row_u8_fx_neon(
    rows: [&[i16]; 4],
    dst: &mut [u8],
    len: usize,
    wy: [f32; 4],
) {
    // SAFETY: the caller's bounds; RDM is detected for its kernel.
    unsafe {
        if has_rdm() {
            rdm::vconv4_row_u8_fx_neon(rows, dst, len, wy);
        } else {
            plain::vconv4_row_u8_fx_neon(rows, dst, len, wy);
        }
    }
}

/// The kernel of `plain`, or of `rdm` where the CPU has RDM (the same
/// results).
///
/// # Safety
/// As the kernels.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vconv4_row_rgb_to_rgbx_neon(
    rows: [&[i16]; 4],
    dst: &mut [u8],
    pixels: usize,
    pad: u8,
    wy: [f32; 4],
) {
    // SAFETY: the caller's bounds; RDM is detected for its kernel.
    unsafe {
        if has_rdm() {
            rdm::vconv4_row_rgb_to_rgbx_neon(rows, dst, pixels, pad, wy);
        } else {
            plain::vconv4_row_rgb_to_rgbx_neon(rows, dst, pixels, pad, wy);
        }
    }
}

// The f32 passes of the separable bicubic, specialized per dtype: `$load`
// widens 8 source elements, `$store` rounds and narrows 8 results.
macro_rules! gen_separable_neon {
    ($hconv:ident, $vconv:ident, $t:ty, $load:ident, $store:ident) => {
        /// Bicubic horizontal pass into f32 (`ComposeSample::hconv4_row`), 8
        /// elements per step in tap order; scalar tail.
        ///
        /// # Safety
        /// The bounds of `ComposeSample::hconv4_row`.
        #[target_feature(enable = "neon")]
        pub(crate) unsafe fn $hconv(
            src: &[$t],
            dst: &mut [f32],
            x0s: usize,
            len: usize,
            step: usize,
            wx: [f32; 4],
        ) {
            let (src_ptr, dst_ptr) = (src.as_ptr(), dst.as_mut_ptr());
            let mut element = 0usize;
            // 8-wide while the x0 + 2 step 8-element load stays in the row.
            while element + 8 <= len && x0s + element + 8 + 2 * step <= src.len() {
                let left = x0s + element - step;
                // SAFETY: x0s >= step, left + 3 step + 8 <= src length (loop
                // guard), and element + 8 <= len <= dst length.
                unsafe {
                    let taps = [0, 1, 2, 3].map(|tap| $load(src_ptr.add(left + tap * step)));
                    store_f32x8(dst_ptr.add(element), dot4_f32x8(taps, wx));
                }
                element += 8;
            }
            if element < len {
                // SAFETY: the rest of the row, under the caller's contract.
                unsafe {
                    scalar_hconv4_row::<$t>(
                        src,
                        &mut dst[element..],
                        x0s + element,
                        len - element,
                        step,
                        wx,
                    );
                }
            }
        }

        /// Bicubic vertical pass from four f32 rows
        /// (`ComposeSample::vconv4_row`), 8 elements per step in tap order;
        /// scalar tail.
        ///
        /// # Safety
        /// The bounds of `ComposeSample::vconv4_row`.
        #[target_feature(enable = "neon")]
        pub(crate) unsafe fn $vconv(rows: [&[f32]; 4], dst: &mut [$t], len: usize, wy: [f32; 4]) {
            let row_ptrs = rows.map(<[f32]>::as_ptr);
            let dst_ptr = dst.as_mut_ptr();
            let mut element = 0usize;
            while element + 8 <= len {
                // SAFETY: element + 8 <= len <= every row's and dst's length.
                unsafe {
                    let taps = row_ptrs.map(|row| load_f32x8(row.add(element)));
                    $store(dst_ptr.add(element), dot4_f32x8(taps, wy));
                }
                element += 8;
            }
            if element < len {
                // SAFETY: the rest of the rows, under the caller's contract.
                unsafe {
                    scalar_vconv4_row::<$t>(
                        rows.map(|row| &row[element..]),
                        &mut dst[element..],
                        len - element,
                        wy,
                    );
                }
            }
        }
    };
}

gen_separable_neon!(
    hconv4_row_u16_neon,
    vconv4_row_u16_neon,
    u16,
    load_u16x8_f32,
    store_u16x8_f32
);
gen_separable_neon!(
    hconv4_row_f32_neon,
    vconv4_row_f32_neon,
    f32,
    load_f32x8,
    store_f32x8
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::rows::tests::{
        BILINEAR_ROUNDING_STEPS, BilinearCase, HconvU16, TestSample, VconvCase, VconvU16,
        assert_bilinear_rows_match, assert_close, bicubic_cases, bilinear_cases, f32_rows,
        guarded_output, l1, q6_hconv_reference, q6_rows, q6_vconv_reference,
        q6_vconv_rgbx_reference, row_lengths, run_bilinear,
    };
    use crate::sample::Sample;

    /// A float32 separable horizontal pass.
    type HconvF32 = unsafe fn(&[f32], &mut [f32], usize, usize, usize, [f32; 4]);
    /// A float32 separable vertical pass.
    type VconvF32 = unsafe fn([&[f32]; 4], &mut [f32], usize, [f32; 4]);
    /// A uint8 fixed-point horizontal pass.
    type HconvU8 = unsafe fn(&[u8], &mut [i16], usize, usize, usize, [f32; 4]);
    /// A uint8 fixed-point vertical pass.
    type VconvU8 = unsafe fn([&[i16]; 4], &mut [u8], usize, [f32; 4]);
    /// A uint8 fixed-point vertical pass, RGB rows in, RGBX out.
    type VconvRgbx = unsafe fn([&[i16]; 4], &mut [u8], usize, u8, [f32; 4]);
    /// A uint8 fixed-point bilinear row, RGB in, RGBX out.
    type BilinearRgbx = unsafe fn(&[u8], &[u8], &mut [u8], usize, usize, usize, u8, [f32; 4]);

    #[test]
    fn bilinear_neon_matches_scalar() {
        // SAFETY: NEON is baseline on aarch64.
        unsafe {
            assert_bilinear_rows_match::<u8>(
                bilinear_row_u8_f32_neon,
                scalar_bilinear_row::<u8>,
                Some(BILINEAR_ROUNDING_STEPS),
            );
            assert_bilinear_rows_match::<u16>(
                bilinear_row_u16_neon,
                scalar_bilinear_row::<u16>,
                Some(BILINEAR_ROUNDING_STEPS),
            );
            assert_bilinear_rows_match::<f32>(
                bilinear_row_f32_neon,
                scalar_bilinear_row::<f32>,
                Some(BILINEAR_ROUNDING_STEPS),
            );
        }
    }

    #[test]
    fn bilinear_u8_fixed_point_within_one_lsb_of_scalar() {
        // SAFETY: NEON is baseline on aarch64.
        unsafe {
            assert_bilinear_rows_match::<u8>(
                bilinear_row_u8_fx_neon,
                scalar_bilinear_row::<u8>,
                Some(BILINEAR_ROUNDING_STEPS),
            );
        }
    }

    /// The plain fixed-point kernels (for CPUs without RDM) give the RDM
    /// kernels' results, which the tests around this one check against the
    /// scalar formulas and mirrors.
    #[test]
    fn plain_and_rdm_fixed_point_kernels_agree() {
        if !has_rdm() {
            return;
        }
        // SAFETY: NEON is baseline on aarch64.
        unsafe {
            assert_bilinear_rows_match::<u8>(
                plain::bilinear_row_u8_fx_neon,
                scalar_bilinear_row::<u8>,
                Some(BILINEAR_ROUNDING_STEPS),
            );
        }
        for case in bicubic_cases::<u8>(34) {
            let [src, ..] = case.rows();
            let run = |kernel: HconvU8| {
                guarded_output(case.len, i16::MIN, |dst| {
                    // SAFETY: NEON and RDM detected; x0s >= step and src ends
                    // at the furthest tap; dst holds len elements.
                    unsafe { kernel(src, dst, case.x0s, case.len, case.step, case.wx) };
                })
            };
            let (plain, rdm) = (
                run(plain::hconv4_row_u8_fx_neon),
                run(rdm::hconv4_row_u8_fx_neon),
            );
            assert_eq!(plain, rdm, "{}", case.describe());
        }
        for case in q6_rows(35, row_lengths()) {
            let run = |kernel: VconvU8| {
                guarded_output(case.len, u8::SENTINEL, |dst| {
                    // SAFETY: NEON and RDM detected; every row and dst hold
                    // len elements.
                    unsafe { kernel(case.rows(), dst, case.len, case.wy) };
                })
            };
            let (plain, rdm) = (
                run(plain::vconv4_row_u8_fx_neon),
                run(rdm::vconv4_row_u8_fx_neon),
            );
            assert_eq!(plain, rdm, "len {} wy {:?}", case.len, case.wy);
        }
        let cases: Vec<VconvCase<i16>> = q6_rows(36, row_lengths().map(|npix| npix * RGB));
        for case in cases {
            let npix = case.len / RGB;
            let run = |kernel: VconvRgbx| {
                guarded_output(npix * RGBX, u8::SENTINEL, |dst| {
                    // SAFETY: NEON and RDM detected; every row holds npix * 3
                    // elements, dst npix * 4.
                    unsafe { kernel(case.rows(), dst, npix, 0xC3, case.wy) };
                })
            };
            let (plain, rdm) = (
                run(plain::vconv4_row_rgb_to_rgbx_neon),
                run(rdm::vconv4_row_rgb_to_rgbx_neon),
            );
            assert_eq!(plain, rdm, "{npix} px wy {:?}", case.wy);
        }
        for case in rgb_cases() {
            let npix = case.len / RGB;
            let run = |kernel: BilinearRgbx| {
                guarded_output(npix * RGBX, u8::SENTINEL, |dst| {
                    // SAFETY: NEON and RDM detected; the rows end at the
                    // furthest tap of npix * 3 outputs and dst holds npix * 4.
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
            };
            let (plain, rdm) = (
                run(plain::bilinear_row_rgb_to_rgbx_fx_neon),
                run(rdm::bilinear_row_rgb_to_rgbx_fx_neon),
            );
            assert_eq!(plain, rdm, "{npix} px");
        }
    }

    #[test]
    fn hconv_u8_fixed_point_equals_mirror() {
        for case in bicubic_cases::<u8>(31) {
            let [src, ..] = case.rows();
            let simd = guarded_output(case.len, i16::MIN, |dst| {
                // SAFETY: NEON is baseline; x0s >= step and src ends at the
                // furthest tap; dst holds len elements.
                unsafe { hconv4_row_u8_fx_neon(src, dst, case.x0s, case.len, case.step, case.wx) };
            });
            let expected = q6_hconv_reference(src, case.x0s, case.len, case.step, case.wx);
            assert_eq!(simd, expected, "{}", case.describe());
        }
    }

    #[test]
    fn vconv_u8_fixed_point_equals_mirror() {
        for case in q6_rows(32, row_lengths()) {
            let simd = guarded_output(case.len, u8::SENTINEL, |dst| {
                // SAFETY: NEON is baseline; every row and dst hold len elements.
                unsafe { vconv4_row_u8_fx_neon(case.rows(), dst, case.len, case.wy) };
            });
            let expected = q6_vconv_reference(case.rows(), case.len, case.wy);
            assert_eq!(simd, expected, "len {} wy {:?}", case.len, case.wy);
        }
    }

    #[test]
    fn vconv_rgb_to_rgbx_equals_mirror() {
        let cases: Vec<VconvCase<i16>> = q6_rows(33, row_lengths().map(|npix| npix * RGB));
        for case in cases {
            let npix = case.len / RGB;
            let simd = guarded_output(npix * RGBX, u8::SENTINEL, |dst| {
                // SAFETY: NEON is baseline; every row holds npix * 3
                // elements, dst npix * 4.
                unsafe { vconv4_row_rgb_to_rgbx_neon(case.rows(), dst, npix, 0xC3, case.wy) };
            });
            let expected = q6_vconv_rgbx_reference(case.rows(), npix, 0xC3, case.wy);
            assert_eq!(simd, expected, "{npix} px wy {:?}", case.wy);
        }
    }

    #[test]
    fn hconv_f32_passes_match_scalar() {
        for case in bicubic_cases::<u16>(34) {
            let [src, ..] = case.rows();
            let run = |kernel: HconvU16| {
                guarded_output(case.len, f32::SENTINEL, |dst| {
                    // SAFETY: x0s >= step and src ends at the furthest tap;
                    // dst holds len elements; NEON is baseline.
                    unsafe { kernel(src, dst, case.x0s, case.len, case.step, case.wx) };
                })
            };
            let (simd, expected) = (run(hconv4_row_u16_neon), run(scalar_hconv4_row::<u16>));
            // A 4-tap sum, rounded like a bilinear blend.
            let tolerance =
                f32::tolerance(BILINEAR_ROUNDING_STEPS, u16::MAX.to_f32() * l1(case.wx));
            assert_close(&simd, &expected, tolerance, &case.describe());
        }
        for case in bicubic_cases::<f32>(36) {
            let [src, ..] = case.rows();
            let run = |kernel: HconvF32| {
                guarded_output(case.len, f32::SENTINEL, |dst| {
                    // SAFETY: as above.
                    unsafe { kernel(src, dst, case.x0s, case.len, case.step, case.wx) };
                })
            };
            let (simd, expected) = (run(hconv4_row_f32_neon), run(scalar_hconv4_row::<f32>));
            let tolerance = f32::tolerance(BILINEAR_ROUNDING_STEPS, l1(case.wx));
            assert_close(&simd, &expected, tolerance, &case.describe());
        }
    }

    #[test]
    fn vconv_f32_passes_match_scalar() {
        for case in f32_rows(35, row_lengths()) {
            let run = |kernel: VconvU16| {
                guarded_output(case.len, u16::SENTINEL, |dst| {
                    // SAFETY: every row and dst hold len elements; NEON is
                    // baseline.
                    unsafe { kernel(case.rows(), dst, case.len, case.wy) };
                })
            };
            let (simd, expected) = (run(vconv4_row_u16_neon), run(scalar_vconv4_row::<u16>));
            assert_close(&simd, &expected, 1.0, &format!("len {}", case.len));
            let run_f32 = |kernel: VconvF32| {
                guarded_output(case.len, f32::SENTINEL, |dst| {
                    // SAFETY: as above.
                    unsafe { kernel(case.rows(), dst, case.len, case.wy) };
                })
            };
            let (simd, expected) = (
                run_f32(vconv4_row_f32_neon),
                run_f32(scalar_vconv4_row::<f32>),
            );
            let magnitude = case
                .rows()
                .iter()
                .flat_map(|row| row.iter())
                .fold(0f32, |max, &value| max.max(value.abs()));
            let tolerance = f32::tolerance(BILINEAR_ROUNDING_STEPS, magnitude * l1(case.wy));
            assert_close(&simd, &expected, tolerance, &format!("len {}", case.len));
        }
    }

    const PAD: u8 = 0x5A;

    /// RGB rows of every pixel count of `row_lengths`, taps 3 elements apart
    /// or (integer-aligned) 0.
    fn rgb_cases() -> Vec<BilinearCase<u8>> {
        bilinear_cases(21, &[0, 3], row_lengths().map(|npix| npix * RGB))
    }

    #[test]
    fn bilinear_rgb_to_rgbx_within_one_lsb_of_scalar() {
        for case in rgb_cases() {
            let npix = case.len / RGB;
            let simd = guarded_output(npix * RGBX, u8::SENTINEL, |dst| {
                // SAFETY: NEON is baseline; the rows end at the furthest tap
                // of npix * 3 outputs and dst holds npix * 4 elements.
                unsafe {
                    bilinear_row_rgb_to_rgbx_fx_neon(
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
            });
            // SAFETY: the scalar kernel needs no CPU features.
            let rgb = unsafe { run_bilinear(scalar_bilinear_row::<u8>, &case) };
            let expected: Vec<u8> = rgb
                .as_chunks::<RGB>()
                .0
                .iter()
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], PAD])
                .collect();
            // Rows under one 16-pixel block take the f32 scalar loop: exact.
            let tolerance = if npix < 16 { 0.0 } else { 1.0 };
            assert_close(&simd, &expected, tolerance, &case.describe());
            assert!(
                simd.as_chunks::<RGBX>()
                    .0
                    .iter()
                    .all(|pixel| pixel[3] == PAD),
                "{}",
                case.describe()
            );
        }
    }
}

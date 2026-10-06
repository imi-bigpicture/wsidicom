//! The per-row kernels of `compose`: bilinear, direct bicubic and the two
//! passes of the separable bicubic, dispatched per sample type.

use super::geometry::{Canvas, Placement, Tile};
#[cfg(target_arch = "aarch64")]
use super::neon::*;
#[cfg(target_arch = "x86_64")]
use super::{x86_rows::*, x86_separable::*};
#[cfg(target_arch = "aarch64")]
use crate::cpu::use_neon;
#[cfg(target_arch = "x86_64")]
use crate::cpu::{has_avx2_fma, has_avx512};
use crate::sample::{BICUBIC_U8_FRAC_BITS, RGB, RGBX, Sample, mulhrs, q15};

/// One output pixel of the u8 separable bicubic's vertical pass: the Q15
/// weighted sum of four Q6 rows at `element`, rounded back to u8.
#[inline]
pub(crate) fn vconv4_u8_px(rows: [&[i16]; 4], element: usize, qwy: [i16; 4]) -> u8 {
    let acc: i32 = (0..4)
        .map(|tap| mulhrs(i32::from(rows[tap][element]), qwy[tap]))
        .sum();
    let half = 1 << (BICUBIC_U8_FRAC_BITS - 1);
    ((acc + half) >> BICUBIC_U8_FRAC_BITS).clamp(0, 255) as u8
}

/// Whether the separable (two-pass) bicubic kernels exist on this CPU (AVX2
/// or NEON); forced-scalar runs the direct kernel.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn separable_bicubic() -> bool {
    has_avx2_fma()
}
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn separable_bicubic() -> bool {
    use_neon()
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
pub(crate) fn separable_bicubic() -> bool {
    false
}

/// The compose row kernels per sample type: the scalar reference,
/// overridden with AVX2/AVX-512/NEON kernels where they exist.
pub(crate) trait ComposeSample: Sample {
    /// Whether the separable two-pass bicubic beats the direct kernel: true
    /// for u8/u16 (the cached horizontal pass amortizes the widening 4x); for
    /// f32 it depends on the instruction set (see the f32 impl).
    const SEP_BICUBIC: bool = true;

    /// Element type of the separable bicubic's cached horizontal rows:
    /// f32 for u16/f32; i16 (Q6 fixed point) for u8, which halves the ring
    /// traffic and enables the 16-wide mulhrs vertical pass.
    type HAcc: Copy + Default + Send + Sync + 'static;

    /// The bilinear blend of a whole tile, where the type has a faster
    /// path than row by row through `bilinear_row` with the same results:
    /// true if it composed the tile, false (nothing written) if not.
    fn compose_bilinear_tile(
        _canvas: &mut Canvas<Self>,
        _tile: &Tile<Self>,
        _placement: &Placement,
    ) -> bool
    where
        Self: Sized,
    {
        false
    }

    /// Fill dst[0..len] with the bilinear blend of two source rows in element
    /// units, output `element` reading source elements x0s+element and
    /// x0s+element+step.
    /// Planar rows use step=1; interleaved HWC rows use step=channels (the x-tap of
    /// the same channel is channels elements away — the weights are per-position, so
    /// consecutive elements of different channels share one weight vector).
    /// Weights [w00, w10, w01, w11]. Default scalar; overridden with AVX2/NEON.
    ///
    /// # Safety
    /// `len <= dst.len()`, and when `len > 0` the furthest tap,
    /// `x0s + len - 1 + step`, is inside both `row0` and `row1`.
    unsafe fn bilinear_row(
        row0: &[Self],
        row1: &[Self],
        dst: &mut [Self],
        x0s: usize,
        len: usize,
        step: usize,
        weights: [f32; 4],
    ) where
        Self: Sized,
    {
        // SAFETY: this method's contract is scalar_bilinear_row's.
        unsafe { scalar_bilinear_row::<Self>(row0, row1, dst, x0s, len, step, weights) };
    }

    /// Fill dst[0..len] with the interior bicubic blend of four source rows
    /// (offsets y0-1..y0+2) in element units, x-taps at x0-step..x0+2*step
    /// (`x0s >= step`; step = 1 planar, channels interleaved — see `bilinear_row`).
    /// 4x4 constant weights wx/wy. Default scalar; overridden with AVX2/NEON.
    ///
    /// # Safety
    /// `len <= dst.len()`, `x0s >= step`, and when `len > 0` the furthest tap,
    /// `x0s + len - 1 + 2 * step`, is inside every row of `rows`.
    unsafe fn bicubic_row(
        rows: [&[Self]; 4],
        dst: &mut [Self],
        x0s: usize,
        len: usize,
        step: usize,
        wx: [f32; 4],
        wy: [f32; 4],
    ) where
        Self: Sized,
    {
        // SAFETY: this method's contract is scalar_bicubic_row's.
        unsafe { scalar_bicubic_row::<Self>(rows, dst, x0s, len, step, wx, wy) };
    }

    /// Horizontal 4-tap convolution of one source row into `HAcc` (the first pass
    /// of the separable bicubic):
    /// `dst[element] = wx . src[x0s + element + {-step, 0, step, 2 step}]`.
    /// Element units, same contract as `bicubic_row`.
    ///
    /// # Safety
    /// `len <= dst.len()`, `x0s >= step`, and when `len > 0` the furthest tap,
    /// `x0s + len - 1 + 2 * step`, is inside `src`.
    unsafe fn hconv4_row(
        src: &[Self],
        dst: &mut [Self::HAcc],
        x0s: usize,
        len: usize,
        step: usize,
        wx: [f32; 4],
    ) where
        Self: Sized;

    /// Vertical 4-tap blend of four hconv'd `HAcc` rows back to Self (the second
    /// pass): `dst[element] = wy . rows[*][element]`.
    ///
    /// # Safety
    /// `len <= dst.len()` and `len <=` every row's length.
    unsafe fn vconv4_row(rows: [&[Self::HAcc]; 4], dst: &mut [Self], len: usize, wy: [f32; 4])
    where
        Self: Sized;
}

/// The scalar `ComposeSample::bicubic_row`.
///
/// # Safety
/// `ComposeSample::bicubic_row`'s contract.
pub(crate) unsafe fn scalar_bicubic_row<T: Sample>(
    rows: [&[T]; 4],
    dst: &mut [T],
    x0s: usize,
    len: usize,
    step: usize,
    wx: [f32; 4],
    wy: [f32; 4],
) {
    for element in 0..len {
        let x0 = x0s + element;
        let mut acc = 0.0f32;
        // SAFETY: x0 - step >= 0, x0 + 2 * step is inside every row and
        // element < dst.len() (the caller's contract, element < len).
        unsafe {
            for tap in 0..4 {
                let row = rows[tap];
                let sum = wx[0] * row.get_unchecked(x0 - step).to_f32()
                    + wx[1] * row.get_unchecked(x0).to_f32()
                    + wx[2] * row.get_unchecked(x0 + step).to_f32()
                    + wx[3] * row.get_unchecked(x0 + 2 * step).to_f32();
                acc += wy[tap] * sum;
            }
            *dst.get_unchecked_mut(element) = T::from_f32(acc);
        }
    }
}

/// The scalar f32 `ComposeSample::hconv4_row`.
///
/// # Safety
/// `ComposeSample::hconv4_row`'s contract.
pub(crate) unsafe fn scalar_hconv4_row<T: Sample>(
    src: &[T],
    dst: &mut [f32],
    x0s: usize,
    len: usize,
    step: usize,
    wx: [f32; 4],
) {
    for element in 0..len {
        let x0 = x0s + element;
        // SAFETY: x0 - step >= 0, x0 + 2 * step < src.len() and
        // element < dst.len() (the caller's contract, element < len).
        unsafe {
            let sum = wx[0] * src.get_unchecked(x0 - step).to_f32()
                + wx[1] * src.get_unchecked(x0).to_f32()
                + wx[2] * src.get_unchecked(x0 + step).to_f32()
                + wx[3] * src.get_unchecked(x0 + 2 * step).to_f32();
            *dst.get_unchecked_mut(element) = sum;
        }
    }
}

/// The scalar f32 `ComposeSample::vconv4_row`.
///
/// # Safety
/// `ComposeSample::vconv4_row`'s contract.
pub(crate) unsafe fn scalar_vconv4_row<T: Sample>(
    rows: [&[f32]; 4],
    dst: &mut [T],
    len: usize,
    wy: [f32; 4],
) {
    for element in 0..len {
        // SAFETY: element < len, which bounds every row and dst (the caller's
        // contract).
        unsafe {
            let acc = wy[0] * rows[0].get_unchecked(element)
                + wy[1] * rows[1].get_unchecked(element)
                + wy[2] * rows[2].get_unchecked(element)
                + wy[3] * rows[3].get_unchecked(element);
            *dst.get_unchecked_mut(element) = T::from_f32(acc);
        }
    }
}

/// Scalar mirror of the fixed-point u8 bicubic horizontal pass: Q6 samples,
/// Q15 weights, the sum wrapped to i16 as the SIMD kernels' per-term i16 adds.
pub(crate) fn scalar_hconv4_row_u8_fx(
    src: &[u8],
    dst: &mut [i16],
    x0s: usize,
    len: usize,
    step: usize,
    wx: [f32; 4],
) {
    let qwx = wx.map(q15);
    for (element, out) in dst[..len].iter_mut().enumerate() {
        let x_left = x0s + element - step;
        let acc: i32 = (0..4)
            .map(|tap| {
                mulhrs(
                    i32::from(src[x_left + tap * step]) << BICUBIC_U8_FRAC_BITS,
                    qwx[tap],
                )
            })
            .sum();
        *out = acc as i16;
    }
}

/// Scalar mirror of the fixed-point u8 bicubic vertical pass.
pub(crate) fn scalar_vconv4_row_u8_fx(rows: [&[i16]; 4], dst: &mut [u8], len: usize, wy: [f32; 4]) {
    let qwy = wy.map(q15);
    for (element, dst_sample) in dst[..len].iter_mut().enumerate() {
        *dst_sample = vconv4_u8_px(rows, element, qwy);
    }
}

/// Scalar mirror of the fixed-point vertical pass, Q6 RGB rows in, RGBX out.
pub(crate) fn scalar_vconv4_row_rgb_to_rgbx(
    rows: [&[i16]; 4],
    dst: &mut [u8],
    npix: usize,
    pad: u8,
    wy: [f32; 4],
) {
    let qwy = wy.map(q15);
    for (pixel, rgbx) in dst[..npix * RGBX]
        .as_chunks_mut::<RGBX>()
        .0
        .iter_mut()
        .enumerate()
    {
        for (ch, sample) in rgbx[..RGB].iter_mut().enumerate() {
            *sample = vconv4_u8_px(rows, pixel * RGB + ch, qwy);
        }
        rgbx[RGB] = pad;
    }
}

/// `scalar_bilinear_row` of RGB rows into RGBX (`pad` in X), `npix` pixels.
#[expect(
    clippy::too_many_arguments,
    reason = "bilinear_row's contract plus pad"
)]
pub(crate) fn scalar_bilinear_row_rgb_to_rgbx(
    row0: &[u8],
    row1: &[u8],
    dst: &mut [u8],
    x0s: usize,
    npix: usize,
    step: usize,
    pad: u8,
    weights: [f32; 4],
) {
    for (pixel, rgbx) in dst[..npix * RGBX]
        .as_chunks_mut::<RGBX>()
        .0
        .iter_mut()
        .enumerate()
    {
        let offset = x0s + pixel * RGB;
        for (ch, sample) in rgbx[..RGB].iter_mut().enumerate() {
            let value = weights[0] * f32::from(row0[offset + ch])
                + weights[1] * f32::from(row0[offset + step + ch])
                + weights[2] * f32::from(row1[offset + ch])
                + weights[3] * f32::from(row1[offset + step + ch]);
            *sample = u8::from_f32(value);
        }
        rgbx[RGB] = pad;
    }
}

/// The scalar `ComposeSample::bilinear_row`.
///
/// # Safety
/// `ComposeSample::bilinear_row`'s contract.
pub(crate) unsafe fn scalar_bilinear_row<T: Sample>(
    row0: &[T],
    row1: &[T],
    dst: &mut [T],
    x0s: usize,
    len: usize,
    step: usize,
    weights: [f32; 4],
) {
    for element in 0..len {
        let x0 = x0s + element;
        // SAFETY: x0 + step is inside both rows and element < dst.len() (the
        // caller's contract, element < len).
        unsafe {
            let value = weights[0] * row0.get_unchecked(x0).to_f32()
                + weights[1] * row0.get_unchecked(x0 + step).to_f32()
                + weights[2] * row1.get_unchecked(x0).to_f32()
                + weights[3] * row1.get_unchecked(x0 + step).to_f32();
            *dst.get_unchecked_mut(element) = T::from_f32(value);
        }
    }
}

impl ComposeSample for u8 {
    unsafe fn bilinear_row(
        row0: &[u8],
        row1: &[u8],
        dst: &mut [u8],
        x0s: usize,
        len: usize,
        step: usize,
        weights: [f32; 4],
    ) {
        #[cfg(target_arch = "x86_64")]
        if has_avx512() {
            // SAFETY: avx512f+bw detected at runtime; this method's contract is the kernel's.
            return unsafe { bilinear_row_u8_fx_avx512(row0, row1, dst, x0s, len, step, weights) };
        }
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            // SAFETY: avx2+fma detected at runtime; this method's contract is the kernel's.
            return unsafe { bilinear_row_u8_fx_avx2(row0, row1, dst, x0s, len, step, weights) };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: NEON is baseline on aarch64; this method's contract is the kernel's.
            return unsafe { bilinear_row_u8_fx_neon(row0, row1, dst, x0s, len, step, weights) };
        }
        // SAFETY: this method's contract is the scalar kernel's.
        unsafe { scalar_bilinear_row::<u8>(row0, row1, dst, x0s, len, step, weights) };
    }
    // No SIMD arm: AVX2 and NEON CPUs take the separable path (hconv4/vconv4).
    unsafe fn bicubic_row(
        rows: [&[u8]; 4],
        dst: &mut [u8],
        x0s: usize,
        len: usize,
        step: usize,
        wx: [f32; 4],
        wy: [f32; 4],
    ) {
        // SAFETY: this method's contract is the scalar kernel's.
        unsafe { scalar_bicubic_row::<u8>(rows, dst, x0s, len, step, wx, wy) };
    }
    type HAcc = i16;
    unsafe fn hconv4_row(
        src: &[u8],
        dst: &mut [i16],
        x0s: usize,
        len: usize,
        step: usize,
        wx: [f32; 4],
    ) {
        #[cfg(target_arch = "x86_64")]
        if has_avx512() {
            // SAFETY: avx512f+bw detected at runtime; this method's contract is the kernel's.
            return unsafe { hconv4_row_u8_fx_avx512(src, dst, x0s, len, step, wx) };
        }
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            // SAFETY: avx2+fma detected at runtime; this method's contract is the kernel's.
            return unsafe { hconv4_row_u8_fx_avx2(src, dst, x0s, len, step, wx) };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: NEON is baseline on aarch64; this method's contract is the kernel's.
            return unsafe { hconv4_row_u8_fx_neon(src, dst, x0s, len, step, wx) };
        }
        // Kept so the trait contract holds: separable_bicubic() keeps CPUs
        // without SIMD off this path.
        scalar_hconv4_row_u8_fx(src, dst, x0s, len, step, wx);
    }
    unsafe fn vconv4_row(rows: [&[i16]; 4], dst: &mut [u8], len: usize, wy: [f32; 4]) {
        #[cfg(target_arch = "x86_64")]
        if has_avx512() {
            // SAFETY: avx512f+bw detected at runtime; this method's contract is the kernel's.
            return unsafe { vconv4_row_u8_fx_avx512(rows, dst, len, wy) };
        }
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            // SAFETY: avx2+fma detected at runtime; this method's contract is the kernel's.
            return unsafe { vconv4_row_u8_fx_avx2(rows, dst, len, wy) };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: NEON is baseline on aarch64; this method's contract is the kernel's.
            return unsafe { vconv4_row_u8_fx_neon(rows, dst, len, wy) };
        }
        scalar_vconv4_row_u8_fx(rows, dst, len, wy);
    }
}

impl ComposeSample for u16 {
    /// On NEON, each tile row widened to f32 once (`compose_bilinear_widened`).
    #[cfg(target_arch = "aarch64")]
    fn compose_bilinear_tile(
        canvas: &mut Canvas<u16>,
        tile: &Tile<u16>,
        placement: &Placement,
    ) -> bool {
        if !use_neon() {
            return false;
        }
        // SAFETY: NEON is baseline on aarch64.
        unsafe { crate::composite::compose_bilinear_widened(canvas, tile, placement) };
        true
    }
    /// On AVX2 + FMA, each tile row widened to f32 once
    /// (`compose_bilinear_widened`).
    #[cfg(target_arch = "x86_64")]
    fn compose_bilinear_tile(
        canvas: &mut Canvas<u16>,
        tile: &Tile<u16>,
        placement: &Placement,
    ) -> bool {
        if !has_avx2_fma() {
            return false;
        }
        // SAFETY: avx2+fma detected at runtime.
        unsafe { crate::composite::compose_bilinear_widened(canvas, tile, placement) };
        true
    }

    unsafe fn bilinear_row(
        row0: &[u16],
        row1: &[u16],
        dst: &mut [u16],
        x0s: usize,
        len: usize,
        step: usize,
        weights: [f32; 4],
    ) {
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            // SAFETY: avx2+fma detected at runtime; this method's contract is the kernel's.
            return unsafe { bilinear_row_u16_avx2(row0, row1, dst, x0s, len, step, weights) };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: NEON is baseline on aarch64; this method's contract is the kernel's.
            return unsafe { bilinear_row_u16_neon(row0, row1, dst, x0s, len, step, weights) };
        }
        // SAFETY: this method's contract is the scalar kernel's.
        unsafe { scalar_bilinear_row::<u16>(row0, row1, dst, x0s, len, step, weights) };
    }
    // No SIMD arm — see the u8 impl.
    unsafe fn bicubic_row(
        rows: [&[u16]; 4],
        dst: &mut [u16],
        x0s: usize,
        len: usize,
        step: usize,
        wx: [f32; 4],
        wy: [f32; 4],
    ) {
        // SAFETY: this method's contract is the scalar kernel's.
        unsafe { scalar_bicubic_row::<u16>(rows, dst, x0s, len, step, wx, wy) };
    }
    type HAcc = f32;
    unsafe fn hconv4_row(
        src: &[u16],
        dst: &mut [f32],
        x0s: usize,
        len: usize,
        step: usize,
        wx: [f32; 4],
    ) {
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            // SAFETY: avx2+fma detected at runtime; this method's contract is the kernel's.
            return unsafe { hconv4_row_u16_avx2(src, dst, x0s, len, step, wx) };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: NEON is baseline on aarch64; this method's contract is the kernel's.
            return unsafe { hconv4_row_u16_neon(src, dst, x0s, len, step, wx) };
        }
        // SAFETY: this method's contract is the scalar kernel's.
        unsafe { scalar_hconv4_row::<u16>(src, dst, x0s, len, step, wx) };
    }
    unsafe fn vconv4_row(rows: [&[f32]; 4], dst: &mut [u16], len: usize, wy: [f32; 4]) {
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            // SAFETY: avx2+fma detected at runtime; this method's contract is the kernel's.
            return unsafe { vconv4_row_u16_avx2(rows, dst, len, wy) };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: NEON is baseline on aarch64; this method's contract is the kernel's.
            return unsafe { vconv4_row_u16_neon(rows, dst, len, wy) };
        }
        // SAFETY: this method's contract is the scalar kernel's.
        unsafe { scalar_vconv4_row::<u16>(rows, dst, len, wy) };
    }
}

impl ComposeSample for f32 {
    // Per instruction set: with AVX2 the ring traffic outweighs the saved
    // multiplies; NEON's separable passes beat its direct kernel.
    const SEP_BICUBIC: bool = cfg!(target_arch = "aarch64");
    unsafe fn bilinear_row(
        row0: &[f32],
        row1: &[f32],
        dst: &mut [f32],
        x0s: usize,
        len: usize,
        step: usize,
        weights: [f32; 4],
    ) {
        #[cfg(target_arch = "x86_64")]
        if has_avx512() {
            // SAFETY: avx512f detected at runtime; this method's contract is the kernel's.
            return unsafe { bilinear_row_f32_avx512(row0, row1, dst, x0s, len, step, weights) };
        }
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            // SAFETY: avx2+fma detected at runtime; this method's contract is the kernel's.
            return unsafe { bilinear_row_f32_avx2(row0, row1, dst, x0s, len, step, weights) };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: NEON is baseline on aarch64; this method's contract is the kernel's.
            return unsafe { bilinear_row_f32_neon(row0, row1, dst, x0s, len, step, weights) };
        }
        // SAFETY: this method's contract is the scalar kernel's.
        unsafe { scalar_bilinear_row::<f32>(row0, row1, dst, x0s, len, step, weights) };
    }
    unsafe fn bicubic_row(
        rows: [&[f32]; 4],
        dst: &mut [f32],
        x0s: usize,
        len: usize,
        step: usize,
        wx: [f32; 4],
        wy: [f32; 4],
    ) {
        #[cfg(target_arch = "x86_64")]
        if has_avx512() {
            // SAFETY: avx512f detected at runtime; this method's contract is the kernel's.
            return unsafe { bicubic_row_f32_avx512(rows, dst, x0s, len, step, wx, wy) };
        }
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            // SAFETY: avx2+fma detected at runtime; this method's contract is the kernel's.
            return unsafe { bicubic_row_f32_avx2(rows, dst, x0s, len, step, wx, wy) };
        }
        // SAFETY: this method's contract is the scalar kernel's.
        unsafe { scalar_bicubic_row::<f32>(rows, dst, x0s, len, step, wx, wy) };
    }
    // Reached on aarch64 only (SEP_BICUBIC); x86 takes the direct kernel.
    type HAcc = f32;
    unsafe fn hconv4_row(
        src: &[f32],
        dst: &mut [f32],
        x0s: usize,
        len: usize,
        step: usize,
        wx: [f32; 4],
    ) {
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: NEON is baseline on aarch64; this method's contract is the kernel's.
            return unsafe { hconv4_row_f32_neon(src, dst, x0s, len, step, wx) };
        }
        // SAFETY: this method's contract is the scalar kernel's.
        unsafe { scalar_hconv4_row::<f32>(src, dst, x0s, len, step, wx) };
    }
    unsafe fn vconv4_row(rows: [&[f32]; 4], dst: &mut [f32], len: usize, wy: [f32; 4]) {
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: NEON is baseline on aarch64; this method's contract is the kernel's.
            return unsafe { vconv4_row_f32_neon(rows, dst, len, wy) };
        }
        // SAFETY: this method's contract is the scalar kernel's.
        unsafe { scalar_vconv4_row::<f32>(rows, dst, len, wy) };
    }
}

/// Dispatching separable-bicubic vertical pass, Q6 RGB rows -> RGBX.
///
/// # Safety
/// Every row holds at least `npix * 3` elements and `dst` at least `npix * 4`.
#[inline]
pub(crate) unsafe fn vconv4_rgbx(
    rows: [&[i16]; 4],
    dst: &mut [u8],
    npix: usize,
    pad: u8,
    wy: [f32; 4],
) {
    #[cfg(target_arch = "x86_64")]
    if has_avx512() {
        // SAFETY: avx512f+bw detected at runtime; this function's contract is the kernel's.
        return unsafe { vconv4_row_rgb_to_rgbx_avx512(rows, dst, npix, pad, wy) };
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() {
        // SAFETY: avx2+fma detected at runtime; this function's contract is the kernel's.
        return unsafe { vconv4_row_rgb_to_rgbx_avx2(rows, dst, npix, pad, wy) };
    }
    #[cfg(target_arch = "aarch64")]
    if use_neon() {
        // SAFETY: NEON is baseline on aarch64; this function's contract is the kernel's.
        return unsafe { vconv4_row_rgb_to_rgbx_neon(rows, dst, npix, pad, wy) };
    }
    scalar_vconv4_row_rgb_to_rgbx(rows, dst, npix, pad, wy);
}

#[cfg(test)]
pub(crate) mod tests {
    //! The row-kernel test harness (shared by the per-tier tests of
    //! `x86_rows`, `x86_separable` and `neon`) and the dispatching kernels
    //! checked against their scalar references.

    use super::*;
    use crate::composite::geometry::cubic_weights;
    use crate::testing::{RandomSample, Rng, random_samples};

    /// Samples past a row's or output's end: enough to catch a vector
    /// overrun of one block.
    const PAST_END: usize = 64;

    /// What the row tests need of a sample type beyond `RandomSample`.
    pub(crate) trait TestSample: RandomSample + ComposeSample {
        /// The largest test sample (1.0 for float32: test data lies in [0, 1]).
        const MAX: Self;
        /// Placed past an output's end, to catch stores beyond it.
        const SENTINEL: Self;
        /// Placed past a row's end: a sample that changes any output that
        /// reads it (NaN for float32).
        fn past_row(rng: &mut Rng) -> Self;
        /// How far two results may differ when each rounds `rounding_steps`
        /// times in f32 on terms summing to `magnitude` in absolute value:
        /// 1 LSB for integers (both round to nearest from nearly equal
        /// values), `rounding_steps` units of f32 epsilon of `magnitude` for
        /// float32.
        fn tolerance(rounding_steps: f32, magnitude: f32) -> f32;
        fn distance(self, other: Self) -> f32;
    }

    impl TestSample for u8 {
        const MAX: Self = u8::MAX;
        const SENTINEL: Self = 0xA5;
        fn past_row(rng: &mut Rng) -> Self {
            Self::random(rng)
        }
        fn tolerance(_rounding_steps: f32, _magnitude: f32) -> f32 {
            1.0
        }
        fn distance(self, other: Self) -> f32 {
            f32::from(self.abs_diff(other))
        }
    }

    impl TestSample for u16 {
        const MAX: Self = u16::MAX;
        const SENTINEL: Self = 0xA5A5;
        fn past_row(rng: &mut Rng) -> Self {
            Self::random(rng)
        }
        fn tolerance(_rounding_steps: f32, _magnitude: f32) -> f32 {
            1.0
        }
        fn distance(self, other: Self) -> f32 {
            f32::from(self.abs_diff(other))
        }
    }

    impl TestSample for f32 {
        const MAX: Self = 1.0;
        const SENTINEL: Self = -7.0;
        fn past_row(_rng: &mut Rng) -> Self {
            f32::NAN
        }
        fn tolerance(rounding_steps: f32, magnitude: f32) -> f32 {
            rounding_steps * f32::EPSILON * magnitude
        }
        fn distance(self, other: Self) -> f32 {
            (self - other).abs()
        }
    }

    /// Bilinear blends round four products and three sums (or fuse them):
    /// each side is within 2 epsilon of the magnitude, so two are within 4.
    pub(crate) const BILINEAR_ROUNDING_STEPS: f32 = 4.0;
    /// The 4x4 bicubic rounds twice per tap row and twice per vertical tap
    /// (nine rounding errors of half an epsilon on each side).
    pub(crate) const BICUBIC_ROUNDING_STEPS: f32 = 9.0;

    /// Element steps between a pixel's taps: integer-aligned bilinear (0, the
    /// left tap re-read), planar (1) and interleaved RGB and 4-sample rows.
    pub(crate) const BILINEAR_STEPS: [usize; 4] = [0, 1, 3, 4];
    /// Bicubic steps: planar and interleaved (the taps straddle x0, so never 0).
    pub(crate) const BICUBIC_STEPS: [usize; 3] = [1, 3, 4];

    /// Output lengths: every length through two 32-wide blocks (the AVX-512
    /// u8 block), so the short-row fallbacks, vector bodies, overlapping last
    /// blocks and scalar tails all run, plus a few longer rows.
    pub(crate) fn row_lengths() -> impl Iterator<Item = usize> + Clone {
        (0..=70).chain([95, 96, 97, 127, 128, 129, 200])
    }

    /// Sub-pixel phases: aligned, half, quarter, nearly one and one random.
    fn phase(rng: &mut Rng, case_index: usize) -> f32 {
        [0.0, 0.5, 0.25, 0.999][..]
            .get(case_index % 5)
            .copied()
            .unwrap_or_else(|| f32::random(rng))
    }

    /// The bilinear weights of `Placement::bilinear_weights` for phase (fx, fy).
    pub(crate) fn bilinear_weights(fx: f32, fy: f32) -> [f32; 4] {
        [
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        ]
    }

    pub(crate) fn l1(weights: [f32; 4]) -> f32 {
        weights.iter().map(|weight| weight.abs()).sum()
    }

    /// `len` test samples, random or (every third case) only 0 and the
    /// maximum, which drives bicubic overshoot to its extremes; followed by
    /// `PAST_END` samples that no correct output reads.
    pub(crate) fn test_row<T: TestSample>(rng: &mut Rng, len: usize, extremes: bool) -> Vec<T> {
        let mut row: Vec<T> = if extremes {
            (0..len)
                .map(|_| {
                    if rng.next_u64() & 1 == 0 {
                        T::from_f32(0.0)
                    } else {
                        T::MAX
                    }
                })
                .collect()
        } else {
            random_samples(rng, len)
        };
        row.extend((0..PAST_END).map(|_| T::past_row(rng)));
        row
    }

    /// Runs `kernel` on an output of `len` elements followed by sentinels,
    /// asserts that it stored nothing past the end, and returns the output.
    pub(crate) fn guarded_output<A: Copy + PartialEq + std::fmt::Debug>(
        len: usize,
        sentinel: A,
        kernel: impl FnOnce(&mut [A]),
    ) -> Vec<A> {
        let mut out = vec![sentinel; len + PAST_END];
        kernel(&mut out[..len]);
        assert!(
            out[len..].iter().all(|&sample| sample == sentinel),
            "stored past the output's end"
        );
        out.truncate(len);
        out
    }

    /// Asserts `simd` and `reference` differ by at most `tolerance` per
    /// element.
    pub(crate) fn assert_close<T: TestSample>(
        simd: &[T],
        reference: &[T],
        tolerance: f32,
        context: &str,
    ) {
        assert_eq!(simd.len(), reference.len(), "{context}");
        for (index, (&simd, &reference)) in simd.iter().zip(reference).enumerate() {
            assert!(
                simd.distance(reference) <= tolerance,
                "{context}: element {index}: {simd:?} vs {reference:?} (tolerance {tolerance})"
            );
        }
    }

    /// One bilinear row call whose rows end exactly at the furthest tap
    /// (`x0s + len - 1 + step`), so a read past it leaves the row.
    pub(crate) struct BilinearCase<T> {
        rows: [Vec<T>; 2],
        row_len: usize,
        pub(crate) x0s: usize,
        pub(crate) len: usize,
        pub(crate) step: usize,
        pub(crate) weights: [f32; 4],
    }

    impl<T> BilinearCase<T> {
        pub(crate) fn row(&self, index: usize) -> &[T] {
            &self.rows[index][..self.row_len]
        }

        pub(crate) fn describe(&self) -> String {
            format!(
                "len {} step {} x0s {} weights {:?}",
                self.len, self.step, self.x0s, self.weights
            )
        }
    }

    /// Bilinear cases for every step of `steps`, length of `lengths` and a
    /// few leading offsets, cycling through the phases and data patterns.
    pub(crate) fn bilinear_cases<T: TestSample>(
        seed: u64,
        steps: &[usize],
        lengths: impl Iterator<Item = usize> + Clone,
    ) -> Vec<BilinearCase<T>> {
        let mut rng = Rng::new(seed);
        let mut cases = Vec::new();
        for &step in steps {
            for len in lengths.clone() {
                for lead in [0, 1, 5] {
                    let case_index = cases.len();
                    let row_len = lead + len + step;
                    let extremes = case_index % 3 == 2;
                    let (fx, fy) = (phase(&mut rng, case_index), phase(&mut rng, case_index / 5));
                    cases.push(BilinearCase {
                        rows: [
                            test_row(&mut rng, row_len, extremes),
                            test_row(&mut rng, row_len, extremes),
                        ],
                        row_len,
                        x0s: lead,
                        len,
                        step,
                        weights: bilinear_weights(fx, fy),
                    });
                }
            }
        }
        cases
    }

    /// A `ComposeSample::bilinear_row` implementation.
    pub(crate) type BilinearRow<T> = unsafe fn(&[T], &[T], &mut [T], usize, usize, usize, [f32; 4]);

    /// Runs `kernel` on `case` into a guarded output.
    ///
    /// # Safety
    /// The CPU has the features `kernel` needs.
    pub(crate) unsafe fn run_bilinear<T: TestSample>(
        kernel: BilinearRow<T>,
        case: &BilinearCase<T>,
    ) -> Vec<T> {
        guarded_output(case.len, T::SENTINEL, |dst| {
            // SAFETY: the case's rows end at the furthest tap and `dst` holds
            // `len` elements (bilinear_row's contract); the CPU features are
            // the caller's.
            unsafe {
                kernel(
                    case.row(0),
                    case.row(1),
                    dst,
                    case.x0s,
                    case.len,
                    case.step,
                    case.weights,
                );
            }
        })
    }

    /// Asserts `kernel` matches `reference` on every bilinear case: exactly
    /// when `rounding_steps` is `None`, else within `TestSample::tolerance`.
    ///
    /// # Safety
    /// The CPU has the features `kernel` and `reference` need.
    pub(crate) unsafe fn assert_bilinear_rows_match<T: TestSample>(
        kernel: BilinearRow<T>,
        reference: BilinearRow<T>,
        rounding_steps: Option<f32>,
    ) {
        for case in bilinear_cases::<T>(11, &BILINEAR_STEPS, row_lengths()) {
            let tolerance = rounding_steps.map_or(0.0, |steps| {
                T::tolerance(steps, T::MAX.to_f32() * l1(case.weights))
            });
            // SAFETY: the features are the caller's.
            let (simd, expected) =
                unsafe { (run_bilinear(kernel, &case), run_bilinear(reference, &case)) };
            assert_close(&simd, &expected, tolerance, &case.describe());
        }
    }

    /// One 4-tap row call: `x0s >= step` and every row ends exactly at the
    /// furthest tap (`x0s + len - 1 + 2 * step`).
    pub(crate) struct BicubicCase<T> {
        rows: [Vec<T>; 4],
        row_len: usize,
        pub(crate) x0s: usize,
        pub(crate) len: usize,
        pub(crate) step: usize,
        pub(crate) wx: [f32; 4],
        pub(crate) wy: [f32; 4],
    }

    impl<T> BicubicCase<T> {
        pub(crate) fn rows(&self) -> [&[T]; 4] {
            self.rows.each_ref().map(|row| &row[..self.row_len])
        }

        /// The magnitude bound of a 4x4 blend of samples up to `max`.
        pub(crate) fn magnitude(&self, max: f32) -> f32 {
            max * l1(self.wx) * l1(self.wy)
        }

        pub(crate) fn describe(&self) -> String {
            format!(
                "len {} step {} x0s {} wx {:?} wy {:?}",
                self.len, self.step, self.x0s, self.wx, self.wy
            )
        }
    }

    /// Bicubic cases for every step, length and leading offset from the
    /// least valid one (`x0s == step`), cycling phases and data patterns.
    pub(crate) fn bicubic_cases<T: TestSample>(seed: u64) -> Vec<BicubicCase<T>> {
        let mut rng = Rng::new(seed);
        let mut cases = Vec::new();
        for step in BICUBIC_STEPS {
            for len in row_lengths() {
                for lead in [0, 1, 5] {
                    let case_index = cases.len();
                    let x0s = step + lead;
                    let row_len = x0s + len + 2 * step;
                    let extremes = case_index % 3 == 2;
                    let (fx, fy) = (phase(&mut rng, case_index), phase(&mut rng, case_index / 5));
                    cases.push(BicubicCase {
                        rows: std::array::from_fn(|_| test_row(&mut rng, row_len, extremes)),
                        row_len,
                        x0s,
                        len,
                        step,
                        wx: cubic_weights(fx),
                        wy: cubic_weights(fy),
                    });
                }
            }
        }
        cases
    }

    /// A `ComposeSample::hconv4_row` implementation for u16.
    pub(crate) type HconvU16 = unsafe fn(&[u16], &mut [f32], usize, usize, usize, [f32; 4]);
    /// A `ComposeSample::vconv4_row` implementation for u16.
    pub(crate) type VconvU16 = unsafe fn([&[f32]; 4], &mut [u16], usize, [f32; 4]);

    /// A `ComposeSample::bicubic_row` implementation.
    pub(crate) type BicubicRow<T> =
        unsafe fn([&[T]; 4], &mut [T], usize, usize, usize, [f32; 4], [f32; 4]);

    /// Runs `kernel` on `case` into a guarded output.
    ///
    /// # Safety
    /// The CPU has the features `kernel` needs.
    pub(crate) unsafe fn run_bicubic<T: TestSample>(
        kernel: BicubicRow<T>,
        case: &BicubicCase<T>,
    ) -> Vec<T> {
        guarded_output(case.len, T::SENTINEL, |dst| {
            // SAFETY: x0s >= step, the case's rows end at the furthest tap
            // and `dst` holds `len` elements (bicubic_row's contract); the
            // CPU features are the caller's.
            unsafe {
                kernel(
                    case.rows(),
                    dst,
                    case.x0s,
                    case.len,
                    case.step,
                    case.wx,
                    case.wy,
                );
            }
        })
    }

    /// Asserts `kernel` matches `reference` on every bicubic case: exactly
    /// when `rounding_steps` is `None`, else within `TestSample::tolerance`.
    ///
    /// # Safety
    /// The CPU has the features `kernel` and `reference` need.
    pub(crate) unsafe fn assert_bicubic_rows_match<T: TestSample>(
        kernel: BicubicRow<T>,
        reference: BicubicRow<T>,
        rounding_steps: Option<f32>,
    ) {
        for case in bicubic_cases::<T>(12) {
            let tolerance = rounding_steps.map_or(0.0, |steps| {
                T::tolerance(steps, case.magnitude(T::MAX.to_f32()))
            });
            // SAFETY: the features are the caller's.
            let (simd, expected) =
                unsafe { (run_bicubic(kernel, &case), run_bicubic(reference, &case)) };
            assert_close(&simd, &expected, tolerance, &case.describe());
        }
    }

    /// The u8 horizontal pass in Q6, as the scalar mirror in `ComposeSample
    /// for u8` computes it: each tap `mulhrs(sample << 6, q15(weight))`,
    /// summed and wrapped to i16 like the vector kernels' i16 adds.
    pub(crate) fn q6_hconv_reference(
        src: &[u8],
        x0s: usize,
        len: usize,
        step: usize,
        wx: [f32; 4],
    ) -> Vec<i16> {
        let qwx = wx.map(q15);
        (0..len)
            .map(|element| {
                let x_left = x0s + element - step;
                let acc: i32 = (0..4)
                    .map(|tap| {
                        mulhrs(
                            i32::from(src[x_left + tap * step]) << BICUBIC_U8_FRAC_BITS,
                            qwx[tap],
                        )
                    })
                    .sum();
                acc as i16
            })
            .collect()
    }

    /// The u8 vertical pass, as the scalar mirror computes it.
    pub(crate) fn q6_vconv_reference(rows: [&[i16]; 4], len: usize, wy: [f32; 4]) -> Vec<u8> {
        let qwy = wy.map(q15);
        (0..len)
            .map(|element| vconv4_u8_px(rows, element, qwy))
            .collect()
    }

    /// The RGB -> RGBX vertical pass, as `vconv4_rgbx`'s scalar path computes it.
    pub(crate) fn q6_vconv_rgbx_reference(
        rows: [&[i16]; 4],
        npix: usize,
        pad: u8,
        wy: [f32; 4],
    ) -> Vec<u8> {
        let rgb = q6_vconv_reference(rows, npix * RGB, wy);
        rgb.as_chunks::<RGB>()
            .0
            .iter()
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], pad])
            .collect()
    }

    /// Four rows for a vertical pass, each `len` elements followed by
    /// samples no correct output reads, and the vertical weights.
    pub(crate) struct VconvCase<A> {
        rows: [Vec<A>; 4],
        pub(crate) len: usize,
        pub(crate) wy: [f32; 4],
    }

    impl<A> VconvCase<A> {
        pub(crate) fn rows(&self) -> [&[A]; 4] {
            self.rows.each_ref().map(|row| &row[..self.len])
        }
    }

    /// Vertical-pass cases of every length in `lengths`: rows of the
    /// horizontal pass `hconv(src, wx)` run over test rows, so they hold
    /// what the vertical pass sees in compose (overshoot included), followed
    /// by `past_end` samples.
    pub(crate) fn vconv_cases<A: Copy>(
        seed: u64,
        lengths: impl Iterator<Item = usize>,
        past_end: A,
        mut hconv: impl FnMut(&[u16], usize, [f32; 4]) -> Vec<A>,
    ) -> Vec<VconvCase<A>> {
        let mut rng = Rng::new(seed);
        let mut cases = Vec::new();
        for len in lengths {
            let case_index = cases.len();
            let wx = cubic_weights(phase(&mut rng, case_index + 1));
            let wy = cubic_weights(phase(&mut rng, case_index));
            let rows = std::array::from_fn(|_| {
                // One pixel of left tap, two of right taps around `len` outputs.
                let src: Vec<u16> = test_row(&mut rng, len + 3, case_index % 3 == 2);
                let mut row = hconv(&src[..len + 3], len, wx);
                row.extend([past_end; PAST_END]);
                row
            });
            cases.push(VconvCase { rows, len, wy });
        }
        cases
    }

    /// The Q6 rows the u8 horizontal pass makes of `len` + 3 u16 test samples
    /// narrowed to u8.
    pub(crate) fn q6_rows(seed: u64, lengths: impl Iterator<Item = usize>) -> Vec<VconvCase<i16>> {
        vconv_cases(seed, lengths, i16::MAX, |src, len, wx| {
            let bytes: Vec<u8> = src.iter().map(|&sample| (sample >> 8) as u8).collect();
            q6_hconv_reference(&bytes, 1, len, 1, wx)
        })
    }

    /// The f32 rows the u16 horizontal pass makes of `len` + 3 u16 samples.
    pub(crate) fn f32_rows(seed: u64, lengths: impl Iterator<Item = usize>) -> Vec<VconvCase<f32>> {
        vconv_cases(seed, lengths, f32::NAN, |src, len, wx| {
            let mut row = vec![0.0; len];
            // SAFETY: x0s = 1 = step, and src holds len + 3 samples, so the
            // furthest tap 1 + len - 1 + 2 is its last.
            unsafe { scalar_hconv4_row::<u16>(src, &mut row, 1, len, 1, wx) };
            row
        })
    }

    #[test]
    fn bilinear_row_dispatch_matches_scalar() {
        // SAFETY: the dispatching kernels and the scalar reference check
        // their CPU features themselves.
        unsafe {
            assert_bilinear_rows_match::<u8>(
                <u8 as ComposeSample>::bilinear_row,
                scalar_bilinear_row::<u8>,
                Some(BILINEAR_ROUNDING_STEPS),
            );
            assert_bilinear_rows_match::<u16>(
                <u16 as ComposeSample>::bilinear_row,
                scalar_bilinear_row::<u16>,
                Some(BILINEAR_ROUNDING_STEPS),
            );
            assert_bilinear_rows_match::<f32>(
                <f32 as ComposeSample>::bilinear_row,
                scalar_bilinear_row::<f32>,
                Some(BILINEAR_ROUNDING_STEPS),
            );
        }
    }

    #[test]
    fn bicubic_row_dispatch_matches_scalar() {
        // SAFETY: the dispatching kernels and the scalar reference check
        // their CPU features themselves.
        unsafe {
            assert_bicubic_rows_match::<u8>(
                <u8 as ComposeSample>::bicubic_row,
                scalar_bicubic_row::<u8>,
                Some(BICUBIC_ROUNDING_STEPS),
            );
            assert_bicubic_rows_match::<u16>(
                <u16 as ComposeSample>::bicubic_row,
                scalar_bicubic_row::<u16>,
                Some(BICUBIC_ROUNDING_STEPS),
            );
            assert_bicubic_rows_match::<f32>(
                <f32 as ComposeSample>::bicubic_row,
                scalar_bicubic_row::<f32>,
                Some(BICUBIC_ROUNDING_STEPS),
            );
        }
    }

    #[test]
    fn hconv4_row_u8_dispatch_equals_fixed_point_mirror() {
        for case in bicubic_cases::<u8>(13) {
            let [src, ..] = case.rows();
            let simd = guarded_output(case.len, i16::MIN, |dst| {
                // SAFETY: x0s >= step and src ends at the furthest tap; dst
                // holds len elements.
                unsafe {
                    <u8 as ComposeSample>::hconv4_row(
                        src, dst, case.x0s, case.len, case.step, case.wx,
                    );
                }
            });
            let expected = q6_hconv_reference(src, case.x0s, case.len, case.step, case.wx);
            assert_eq!(simd, expected, "{}", case.describe());
        }
    }

    #[test]
    fn hconv4_row_u16_dispatch_matches_scalar() {
        for case in bicubic_cases::<u16>(14) {
            let [src, ..] = case.rows();
            let run = |kernel: HconvU16| {
                guarded_output(case.len, f32::SENTINEL, |dst| {
                    // SAFETY: x0s >= step and src ends at the furthest tap;
                    // dst holds len elements; both kernels check their CPU
                    // features themselves.
                    unsafe { kernel(src, dst, case.x0s, case.len, case.step, case.wx) };
                })
            };
            let simd = run(<u16 as ComposeSample>::hconv4_row);
            let expected = run(scalar_hconv4_row::<u16>);
            let tolerance =
                f32::tolerance(BILINEAR_ROUNDING_STEPS, u16::MAX.to_f32() * l1(case.wx));
            assert_close(&simd, &expected, tolerance, &case.describe());
        }
    }

    #[test]
    fn vconv4_row_u8_dispatch_equals_fixed_point_mirror() {
        for case in q6_rows(15, row_lengths()) {
            let simd = guarded_output(case.len, u8::SENTINEL, |dst| {
                // SAFETY: every row and dst hold len elements.
                unsafe { <u8 as ComposeSample>::vconv4_row(case.rows(), dst, case.len, case.wy) };
            });
            assert_eq!(
                simd,
                q6_vconv_reference(case.rows(), case.len, case.wy),
                "len {}",
                case.len
            );
        }
    }

    #[test]
    fn vconv4_row_u16_dispatch_matches_scalar() {
        for case in f32_rows(16, row_lengths()) {
            let run = |kernel: VconvU16| {
                guarded_output(case.len, u16::SENTINEL, |dst| {
                    // SAFETY: every row and dst hold len elements; both
                    // kernels check their CPU features themselves.
                    unsafe { kernel(case.rows(), dst, case.len, case.wy) };
                })
            };
            let simd = run(<u16 as ComposeSample>::vconv4_row);
            let expected = run(scalar_vconv4_row::<u16>);
            assert_close(&simd, &expected, 1.0, &format!("len {}", case.len));
        }
    }

    #[test]
    fn vconv4_rgbx_dispatch_equals_fixed_point_mirror() {
        for case in q6_rows(17, row_lengths().map(|npix| npix * RGB)) {
            let npix = case.len / RGB;
            let simd = guarded_output(npix * RGBX, u8::SENTINEL, |dst| {
                // SAFETY: every row holds npix * 3 elements and dst npix * 4.
                unsafe { vconv4_rgbx(case.rows(), dst, npix, 0x5A, case.wy) };
            });
            let expected = q6_vconv_rgbx_reference(case.rows(), npix, 0x5A, case.wy);
            assert_eq!(simd, expected, "{npix} px");
        }
    }

    #[test]
    fn u8_fixed_point_separable_within_one_lsb_of_float_bicubic() {
        // The fixed-point passes round each Q6 tap (4 x 1/128 px per pass)
        // and quantize weights to Q15: well under half an LSB in all, so the
        // result is within 1 LSB of the f32 direct kernel.
        for case in bicubic_cases::<u8>(18) {
            let hconv_rows = case
                .rows()
                .map(|row| q6_hconv_reference(row, case.x0s, case.len, case.step, case.wx));
            let separable =
                q6_vconv_reference(hconv_rows.each_ref().map(Vec::as_slice), case.len, case.wy);
            // SAFETY: the scalar kernel needs no CPU features.
            let direct = unsafe { run_bicubic(scalar_bicubic_row::<u8>, &case) };
            assert_close(&separable, &direct, 1.0, &case.describe());
        }
    }
}

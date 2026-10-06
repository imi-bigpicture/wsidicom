//! Box reduction: each output sample is the mean of a factor x factor block.

#[cfg(target_arch = "aarch64")]
mod neon;
pub(crate) mod pyramid;
#[cfg(target_arch = "x86_64")]
mod x86_interleaved;
#[cfg(target_arch = "x86_64")]
mod x86_output;
#[cfg(target_arch = "x86_64")]
mod x86_planar;

#[cfg(target_arch = "aarch64")]
use crate::cpu::use_neon;
#[cfg(target_arch = "x86_64")]
use crate::cpu::{has_avx2, has_avx512};
use crate::rgbx::expand_image;
use crate::sample::{RGB, RGBX, Sample, assert_image, assert_interleaved};
#[cfg(target_arch = "aarch64")]
use neon::*;
#[cfg(target_arch = "x86_64")]
use x86_interleaved::*;
#[cfg(target_arch = "x86_64")]
use x86_output::BoxRows;
#[cfg(target_arch = "x86_64")]
use x86_planar::*;

/// The rounded mean of the 2x2 block at `top_left` (`right` elements to the
/// partner column, `below` to the partner row) — the scalar reference every
/// box kernel matches bit for bit. Integer sums are exact in f32 (< 2^24), so
/// for u8/u16 this is `(sum + 2) >> 2`.
///
/// # Safety
/// `top_left`, `top_left + right`, `top_left + below` and
/// `top_left + below + right` must be readable.
#[inline(always)]
pub(crate) unsafe fn box2_mean<T: Sample>(top_left: *const T, right: usize, below: usize) -> T {
    // The SIMD kernels' order (column sums, then the pair), so float32 tails
    // match their bodies bit for bit.
    // SAFETY: the four samples are readable (the caller's contract).
    let (left_col, right_col) = unsafe {
        (
            (*top_left).to_f32() + (*top_left.add(below)).to_f32(),
            (*top_left.add(right)).to_f32() + (*top_left.add(below + right)).to_f32(),
        )
    };
    T::from_f32((left_col + right_col) * 0.25)
}

/// How a box pass sums samples: widened to an accumulator that holds the
/// sums exactly (u32 for uint16) or in the planar kernels' order (f32), and
/// the sum of a 2x2 or 4x4 block finished to a sample.
pub(crate) struct BoxSums<T, Acc> {
    pub(crate) widen: fn(T) -> Acc,
    pub(crate) mean4: fn(Acc) -> T,
    pub(crate) mean16: fn(Acc) -> T,
}

/// Sums in f32, in the planar kernels' order (float32, and the interleaved
/// 2x of any dtype).
fn f32_box_sums<T: Sample>() -> BoxSums<T, f32> {
    BoxSums {
        widen: T::to_f32,
        mean4: |sum| T::from_f32(sum * 0.25),
        mean16: |sum| T::from_f32(sum * (1.0 / 16.0)),
    }
}

/// The scalar fused 4x box: a 4x4 mean rounded once (the SIMD kernels'
/// arithmetic, so every tier gives the same result).
fn box4x_scalar<T: BoxSample>(src: &[T], width: usize, height: usize) -> (Vec<T>, usize, usize) {
    box_levels_fused(src, (width, height, 1), 2, &T::box_sums())
}

/// The fused 4x box of a row band (4 rows of `channels`-sample
/// pixels, `sums.len()` samples used) into `out`: per sample, the columns
/// summed down the rows, then the 4 columns as (c0 + c1) + (c2 + c3) — the
/// planar `box4x` kernels' order. Two contiguous passes, which vectorize.
#[inline(always)]
fn box4_band<T: Copy, Acc: Copy + std::ops::Add<Output = Acc>>(
    rows: [&[T]; 4],
    channels: usize,
    sums: &mut [Acc],
    out: &mut Vec<T>,
    box_sums: &BoxSums<T, Acc>,
) {
    let widen = box_sums.widen;
    for (index, sum) in sums.iter_mut().enumerate() {
        *sum = ((widen(rows[0][index]) + widen(rows[1][index])) + widen(rows[2][index]))
            + widen(rows[3][index]);
    }
    match channels {
        1 => box4_quads::<T, Acc, 1>(sums, out, box_sums.mean16),
        RGB => box4_quads::<T, Acc, RGB>(sums, out, box_sums.mean16),
        RGBX => box4_quads::<T, Acc, RGBX>(sums, out, box_sums.mean16),
        _ => unreachable!("{channels} channels"),
    }
}

/// `box4_band`'s pixel loop for a compile-time channel count, so it
/// vectorizes too.
#[inline(always)]
fn box4_quads<T: Copy, Acc: Copy + std::ops::Add<Output = Acc>, const CHANNELS: usize>(
    sums: &[Acc],
    out: &mut Vec<T>,
    mean16: fn(Acc) -> T,
) {
    let quads = sums.chunks_exact(4 * CHANNELS);
    out.reserve(quads.len() * CHANNELS);
    for quad in quads {
        let px: [T; CHANNELS] = std::array::from_fn(|ch| {
            mean16(
                (quad[ch] + quad[CHANNELS + ch])
                    + (quad[2 * CHANNELS + ch] + quad[3 * CHANNELS + ch]),
            )
        });
        out.extend_from_slice(&px);
    }
}

/// `BoxSample::box_levels_hwc` with `box4_band`: the 4x pass, and for 8x a 2x
/// pass of each pair of its rows while they are in cache (rather than
/// halving a whole 4x image afterwards, which writes and re-reads it) — the
/// planar pyramid's passes.
// Inlined (with the band and pixel loops below) so `box_sums`' function
// pointers are constants at the call site: called through, they cost a call
// per sample and block vectorization.
#[inline(always)]
fn box_levels_fused<T: Copy, Acc: Copy + Default + std::ops::Add<Output = Acc>>(
    src: &[T],
    (width, height, channels): (usize, usize, usize),
    levels: u32,
    box_sums: &BoxSums<T, Acc>,
) -> (Vec<T>, usize, usize) {
    let stride = width * channels;
    let used = 4 * (width / 4) * channels;
    let mut sums = vec![Acc::default(); used];
    let band = |y: usize, out: &mut Vec<T>, sums: &mut [Acc]| {
        let rows = [0, 1, 2, 3].map(|row| &src[(y + row) * stride..][..used]);
        box4_band(rows, channels, sums, out, box_sums);
    };
    if levels == 2 {
        let (ow, oh) = (width / 4, height / 4);
        let mut out = Vec::with_capacity(ow * oh * channels);
        for oy in 0..oh {
            band(4 * oy, &mut out, &mut sums);
        }
        return (out, ow, oh);
    }
    let (ow, oh) = (width / 8, height / 8);
    let mut out = Vec::with_capacity(ow * oh * channels);
    let (mut top, mut bottom) = (
        Vec::with_capacity(width / 4 * channels),
        Vec::with_capacity(width / 4 * channels),
    );
    let mut pair_sums = vec![Acc::default(); 2 * ow * channels];
    for oy in 0..oh {
        top.clear();
        bottom.clear();
        band(8 * oy, &mut top, &mut sums);
        band(8 * oy + 4, &mut bottom, &mut sums);
        box2_band(&top, &bottom, channels, &mut pair_sums, &mut out, box_sums);
    }
    (out, ow, oh)
}

/// Box reduction per sample type: the scalar reference, overridden with
/// AVX2/AVX-512/NEON kernels where they exist.
///
/// Every method takes `src` holding the `width * height` (times `channels`)
/// samples its dimensions describe — an array's shape at the API — which the
/// kernels read without checking.
pub(crate) trait BoxSample: Sample + pyramid::BlockMean {
    /// The accumulator the fused passes sum in.
    type BoxAcc: Copy + Default + std::ops::Add<Output = Self::BoxAcc>;

    /// How the fused passes sum and round: exactly for the integer types (a
    /// pass rounds once), in the planar SIMD kernels' order for float32.
    fn box_sums() -> BoxSums<Self, Self::BoxAcc>
    where
        Self: Sized;

    /// Halve `src` (width x height) by 2x2 box average -> (out, width/2,
    /// height/2). Default scalar; every dtype overrides with AVX2/NEON kernels.
    fn box2x(src: &[Self], width: usize, height: usize) -> (Vec<Self>, usize, usize)
    where
        Self: Sized,
    {
        assert_image(src, width, height, 1);
        box2x_scalar::<Self>(src, width, height)
    }

    /// Quarter `src` (width x height) by 4x4 box average -> (out, width/4,
    /// height/4), rounded once (closer to the true mean than two rounded 2x
    /// passes) on every tier. Default scalar; overridden with AVX2/NEON.
    fn box4x(src: &[Self], width: usize, height: usize) -> (Vec<Self>, usize, usize)
    where
        Self: Sized,
    {
        assert_image(src, width, height, 1);
        box4x_scalar(src, width, height)
    }

    /// `box2x` (`channels` 1) or `box2x_hwc` written to `dst_ptr`, the
    /// output rows `dst_stride` samples apart, where this CPU has such a
    /// kernel: true if it did, false (nothing written) if not.
    ///
    /// # Safety
    /// `dst_ptr` holds the `height / 2` output rows of `width / 2` pixels,
    /// `dst_stride` samples apart.
    unsafe fn box2x_into(
        _src: &[Self],
        (_height, _width, _channels): (usize, usize, usize),
        _dst_ptr: *mut Self,
        _dst_stride: usize,
    ) -> bool
    where
        Self: Sized,
    {
        false
    }

    /// Halve an interleaved HWC array (`channels` 3 or 4). Default scalar;
    /// every dtype overrides with exact AVX2 and NEON (vldN) kernels.
    fn box2x_hwc(
        src: &[Self],
        width: usize,
        height: usize,
        channels: usize,
    ) -> (Vec<Self>, usize, usize)
    where
        Self: Sized,
    {
        assert_image(src, width, height, channels);
        assert_interleaved(channels);
        box2x_interleaved::<Self>(src, width, height, channels)
    }

    /// Reduce an interleaved HWC array by 4 (`levels` 2) or 8 (`levels` 3)
    /// in one pass, as the planar pyramid: a 4x mean, for 8x followed by a 2x
    /// mean. Default scalar.
    fn box_levels_hwc(
        src: &[Self],
        width: usize,
        height: usize,
        channels: usize,
        levels: u32,
    ) -> (Vec<Self>, usize, usize)
    where
        Self: Sized,
    {
        assert_image(src, width, height, channels);
        assert_interleaved(channels);
        box_levels_fused(src, (width, height, channels), levels, &Self::box_sums())
    }
}

/// Scalar 2x2 box halving, planar.
pub(crate) fn box2x_scalar<T: Sample>(
    src: &[T],
    width: usize,
    height: usize,
) -> (Vec<T>, usize, usize) {
    assert_image(src, width, height, 1);
    let (ow, oh) = (width / 2, height / 2);
    let mut out = Vec::with_capacity(ow * oh);
    for oy in 0..oh {
        let row = 2 * oy * width;
        for ox in 0..ow {
            // SAFETY: ox < width/2 and oy < height/2 keep the whole 2x2 block inside src.
            out.push(unsafe { box2_mean(src.as_ptr().add(row + 2 * ox), 1, width) });
        }
    }
    (out, ow, oh)
}

/// Scalar 2x2 box halving, interleaved HWC with `channels` samples per pixel
/// (each channel averaged on its own).
pub(crate) fn box2x_interleaved<T: Sample>(
    src: &[T],
    width: usize,
    height: usize,
    channels: usize,
) -> (Vec<T>, usize, usize) {
    assert_image(src, width, height, channels);
    let (ow, oh) = (width / 2, height / 2);
    let (stride, used) = (width * channels, 2 * ow * channels);
    let mut out = Vec::with_capacity(ow * oh * channels);
    let mut sums = vec![0f32; used];
    let box_sums = f32_box_sums::<T>();
    for oy in 0..oh {
        box2_band(
            &src[2 * oy * stride..][..used],
            &src[(2 * oy + 1) * stride..][..used],
            channels,
            &mut sums,
            &mut out,
            &box_sums,
        );
    }
    (out, ow, oh)
}

/// The 2x box of two interleaved rows (`sums.len()` samples each) into
/// `out`: the two rows' sums, then each output pixel's two adjacent pixels —
/// both passes over contiguous samples, which vectorize (unlike one sample
/// at a time with strided loads).
#[inline(always)]
fn box2_band<T: Copy, Acc: Copy + std::ops::Add<Output = Acc>>(
    top: &[T],
    bottom: &[T],
    channels: usize,
    sums: &mut [Acc],
    out: &mut Vec<T>,
    box_sums: &BoxSums<T, Acc>,
) {
    let widen = box_sums.widen;
    for ((sum, &top_sample), &bottom_sample) in sums.iter_mut().zip(top).zip(bottom) {
        *sum = widen(top_sample) + widen(bottom_sample);
    }
    // A compile-time channel count, so the pixel loop vectorizes too.
    match channels {
        1 => box2_pairs::<T, Acc, 1>(sums, out, box_sums.mean4),
        RGB => box2_pairs::<T, Acc, RGB>(sums, out, box_sums.mean4),
        RGBX => box2_pairs::<T, Acc, RGBX>(sums, out, box_sums.mean4),
        _ => unreachable!("{channels} channels"),
    }
}

/// Averages of horizontally adjacent pixel pairs of `sums` (vertical pair
/// sums of `CHANNELS`-sample pixels), appended to `out`.
#[inline(always)]
fn box2_pairs<T: Copy, Acc: Copy + std::ops::Add<Output = Acc>, const CHANNELS: usize>(
    sums: &[Acc],
    out: &mut Vec<T>,
    mean4: fn(Acc) -> T,
) {
    let pairs = sums.chunks_exact(2 * CHANNELS);
    out.reserve(pairs.len() * CHANNELS);
    for pair in pairs {
        let px: [T; CHANNELS] = std::array::from_fn(|ch| mean4(pair[ch] + pair[CHANNELS + ch]));
        out.extend_from_slice(&px);
    }
}

impl BoxSample for u8 {
    #[cfg(target_arch = "aarch64")]
    unsafe fn box2x_into(
        src: &[u8],
        (height, width, channels): (usize, usize, usize),
        dst_ptr: *mut u8,
        dst_stride: usize,
    ) -> bool {
        assert_image(src, width, height, channels);
        if !use_neon() {
            return false;
        }
        // SAFETY: `src` matches its dimensions (checked); `dst_ptr` holds the
        // output rows (the caller's contract).
        unsafe {
            if channels == 1 {
                box2x_u8_into_neon(src, width, height, dst_ptr, dst_stride);
            } else {
                assert_interleaved(channels);
                box2x_u8_hwc_into_neon(src, (width, height, channels), dst_ptr, dst_stride);
            }
        }
        true
    }
    #[cfg(target_arch = "x86_64")]
    unsafe fn box2x_into(
        src: &[u8],
        (height, width, channels): (usize, usize, usize),
        dst_ptr: *mut u8,
        dst_stride: usize,
    ) -> bool {
        assert_image(src, width, height, channels);
        // SAFETY: `dst_ptr` holds the output rows (the caller's contract).
        let output =
            unsafe { BoxRows::in_place(dst_ptr, dst_stride, height / 2, width / 2 * channels) };
        // SAFETY: each kernel's features are detected at runtime; `src`
        // matches its dimensions (checked) and `output` holds its rows. The
        // tiers are box2x's and box2x_hwc's.
        unsafe {
            match channels {
                1 if has_avx2() => box2x_u8_avx2_to(src, width, height, output),
                RGB if has_avx512() => box2x_u8_hwc3_avx512_to(src, width, height, output),
                RGBX if has_avx512() => box2x_u8_hwc4_avx512_to(src, width, height, output),
                RGB if has_avx2() => box2x_u8_hwc3_avx2_to(src, width, height, output),
                RGBX if has_avx2() => box2x_u8_hwc4_avx2_to(src, width, height, output),
                _ => return false,
            };
        }
        true
    }
    // A 4x4 block sums to at most 4080.
    type BoxAcc = u16;
    fn box_sums() -> BoxSums<u8, u16> {
        BoxSums {
            widen: u16::from,
            mean4: |sum| ((sum + 2) >> 2) as u8,
            mean16: |sum| ((sum + 8) >> 4) as u8,
        }
    }
    fn box2x(src: &[u8], width: usize, height: usize) -> (Vec<u8>, usize, usize) {
        assert_image(src, width, height, 1);
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            // SAFETY: avx2 detected at runtime; `src` matches its dimensions (BoxSample).
            return unsafe { box2x_u8_avx2(src, width, height) };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: `src` matches its dimensions (BoxSample).
            return unsafe { box2x_u8_neon(src, width, height) };
        }
        box2x_scalar::<u8>(src, width, height)
    }
    fn box4x(src: &[u8], width: usize, height: usize) -> (Vec<u8>, usize, usize) {
        assert_image(src, width, height, 1);
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            // SAFETY: avx2 detected at runtime; `src` matches its dimensions (BoxSample).
            return unsafe { box4x_u8_avx2(src, width, height) };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: `src` matches its dimensions (BoxSample).
            return unsafe { box4x_u8_neon(src, width, height) };
        }
        box4x_scalar(src, width, height)
    }
    fn box2x_hwc(
        src: &[u8],
        width: usize,
        height: usize,
        channels: usize,
    ) -> (Vec<u8>, usize, usize) {
        assert_image(src, width, height, channels);
        assert_interleaved(channels);
        #[cfg(target_arch = "x86_64")]
        if has_avx512() && (channels == RGB || channels == RGBX) {
            // SAFETY: avx512f+bw+vl detected at runtime; `src` matches its dimensions (BoxSample).
            return unsafe {
                if channels == RGB {
                    box2x_u8_hwc3_avx512(src, width, height)
                } else {
                    box2x_u8_hwc4_avx512(src, width, height)
                }
            };
        }
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            // SAFETY: avx2 detected at runtime; `src` matches its dimensions (BoxSample).
            return unsafe {
                if channels == RGB {
                    box2x_u8_hwc3_avx2(src, width, height)
                } else {
                    box2x_u8_hwc4_avx2(src, width, height)
                }
            };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: `src` matches its dimensions (BoxSample).
            return unsafe { box2x_u8_hwc_neon(src, width, height, channels) };
        }
        box2x_interleaved::<u8>(src, width, height, channels)
    }
    fn box_levels_hwc(
        src: &[u8],
        width: usize,
        height: usize,
        channels: usize,
        levels: u32,
    ) -> (Vec<u8>, usize, usize) {
        assert_image(src, width, height, channels);
        assert_interleaved(channels);
        assert!((2..=3).contains(&levels));
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            // SAFETY: avx2 detected at runtime; `src` matches its dimensions (BoxSample); channels is 3 or 4.
            return unsafe { box_reduce_u8_hwc_avx2(src, (width, height), channels, levels, None) };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: `src` matches its dimensions (BoxSample); channels is 3 or 4.
            return unsafe {
                if channels == RGB {
                    box_levels_u8_hwc_neon::<RGB>(src, (width, height), levels, None)
                } else {
                    box_levels_u8_hwc_neon::<RGBX>(src, (width, height), levels, None)
                }
            };
        }
        box_levels_fused(src, (width, height, channels), levels, &Self::box_sums())
    }
}

impl BoxSample for u16 {
    #[cfg(target_arch = "aarch64")]
    unsafe fn box2x_into(
        src: &[u16],
        (height, width, channels): (usize, usize, usize),
        dst_ptr: *mut u16,
        dst_stride: usize,
    ) -> bool {
        assert_image(src, width, height, channels);
        if !use_neon() {
            return false;
        }
        // SAFETY: `src` matches its dimensions (checked); `dst_ptr` holds the
        // output rows (the caller's contract).
        unsafe {
            if channels == 1 {
                box2x_u16_into_neon(src, width, height, dst_ptr, dst_stride);
            } else {
                assert_interleaved(channels);
                box2x_u16_hwc_into_neon(src, (width, height, channels), dst_ptr, dst_stride);
            }
        }
        true
    }
    #[cfg(target_arch = "x86_64")]
    unsafe fn box2x_into(
        src: &[u16],
        (height, width, channels): (usize, usize, usize),
        dst_ptr: *mut u16,
        dst_stride: usize,
    ) -> bool {
        assert_image(src, width, height, channels);
        if !has_avx2() {
            return false;
        }
        // SAFETY: `dst_ptr` holds the output rows (the caller's contract).
        let output =
            unsafe { BoxRows::in_place(dst_ptr, dst_stride, height / 2, width / 2 * channels) };
        // SAFETY: avx2 detected at runtime; `src` matches its dimensions
        // (checked) and `output` holds its rows. The kernels are box2x's and
        // box2x_hwc's.
        unsafe {
            match channels {
                1 => box2x_u16_avx2_to(src, width, height, output),
                RGB => box2x_u16_hwc3_avx2_to(src, width, height, output),
                _ => box2x_u16_hwc4_avx2_to(src, width, height, output),
            };
        }
        true
    }
    // A 4x4 block sums to at most 16 * 65535.
    type BoxAcc = u32;
    fn box_sums() -> BoxSums<u16, u32> {
        BoxSums {
            widen: u32::from,
            mean4: |sum| ((sum + 2) >> 2) as u16,
            mean16: |sum| ((sum + 8) >> 4) as u16,
        }
    }
    fn box2x(src: &[u16], width: usize, height: usize) -> (Vec<u16>, usize, usize) {
        assert_image(src, width, height, 1);
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            // SAFETY: avx2 detected at runtime; `src` matches its dimensions (BoxSample).
            return unsafe { box2x_u16_avx2(src, width, height) };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: `src` matches its dimensions (BoxSample).
            return unsafe { box2x_u16_neon(src, width, height) };
        }
        box2x_scalar::<u16>(src, width, height)
    }
    fn box4x(src: &[u16], width: usize, height: usize) -> (Vec<u16>, usize, usize) {
        assert_image(src, width, height, 1);
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            // SAFETY: avx2 detected at runtime; `src` matches its dimensions (BoxSample).
            return unsafe { box4x_u16_avx2(src, width, height) };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: `src` matches its dimensions (BoxSample).
            return unsafe { box4x_u16_neon(src, width, height) };
        }
        box4x_scalar(src, width, height)
    }
    fn box_levels_hwc(
        src: &[u16],
        width: usize,
        height: usize,
        channels: usize,
        levels: u32,
    ) -> (Vec<u16>, usize, usize) {
        assert_image(src, width, height, channels);
        assert_interleaved(channels);
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: `src` matches its dimensions (BoxSample); channels is 3 or 4.
            return unsafe {
                if channels == RGB {
                    box_levels_u16_hwc_neon::<RGB>(src, (width, height), levels, None)
                } else {
                    box_levels_u16_hwc_neon::<RGBX>(src, (width, height), levels, None)
                }
            };
        }
        box_levels_fused(src, (width, height, channels), levels, &Self::box_sums())
    }
    fn box2x_hwc(
        src: &[u16],
        width: usize,
        height: usize,
        channels: usize,
    ) -> (Vec<u16>, usize, usize) {
        assert_image(src, width, height, channels);
        assert_interleaved(channels);
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            // SAFETY: avx2 detected at runtime; `src` matches its dimensions (BoxSample).
            return unsafe {
                if channels == RGB {
                    box2x_u16_hwc3_avx2(src, width, height)
                } else {
                    box2x_u16_hwc4_avx2(src, width, height)
                }
            };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: `src` matches its dimensions (BoxSample).
            return unsafe { box2x_u16_hwc_neon(src, width, height, channels) };
        }
        box2x_interleaved::<u16>(src, width, height, channels)
    }
}

impl BoxSample for f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe fn box2x_into(
        src: &[f32],
        (height, width, channels): (usize, usize, usize),
        dst_ptr: *mut f32,
        dst_stride: usize,
    ) -> bool {
        assert_image(src, width, height, channels);
        if !use_neon() {
            return false;
        }
        // SAFETY: `src` matches its dimensions (checked); `dst_ptr` holds the
        // output rows (the caller's contract).
        unsafe {
            if channels == 1 {
                box2x_f32_into_neon(src, width, height, dst_ptr, dst_stride);
            } else {
                assert_interleaved(channels);
                box2x_f32_hwc_into_neon(src, (width, height, channels), dst_ptr, dst_stride);
            }
        }
        true
    }
    #[cfg(target_arch = "x86_64")]
    unsafe fn box2x_into(
        src: &[f32],
        (height, width, channels): (usize, usize, usize),
        dst_ptr: *mut f32,
        dst_stride: usize,
    ) -> bool {
        assert_image(src, width, height, channels);
        if !has_avx2() {
            return false;
        }
        // SAFETY: `dst_ptr` holds the output rows (the caller's contract).
        let output =
            unsafe { BoxRows::in_place(dst_ptr, dst_stride, height / 2, width / 2 * channels) };
        // SAFETY: avx2 detected at runtime; `src` matches its dimensions
        // (checked) and `output` holds its rows. The kernels are box2x's and
        // box2x_hwc's.
        unsafe {
            match channels {
                1 => box2x_f32_avx2_to(src, width, height, output),
                RGB => box2x_f32_hwc3_avx2_to(src, width, height, output),
                _ => box2x_f32_hwc4_avx2_to(src, width, height, output),
            };
        }
        true
    }
    type BoxAcc = f32;
    fn box_sums() -> BoxSums<f32, f32> {
        f32_box_sums()
    }
    fn box2x(src: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
        assert_image(src, width, height, 1);
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            // SAFETY: avx2 detected at runtime; `src` matches its dimensions (BoxSample).
            return unsafe { box2x_f32_avx2(src, width, height) };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: `src` matches its dimensions (BoxSample).
            return unsafe { box2x_f32_neon(src, width, height) };
        }
        box2x_scalar::<f32>(src, width, height)
    }
    fn box4x(src: &[f32], width: usize, height: usize) -> (Vec<f32>, usize, usize) {
        assert_image(src, width, height, 1);
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            // SAFETY: avx2 detected at runtime; `src` matches its dimensions (BoxSample).
            return unsafe { box4x_f32_avx2(src, width, height) };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: `src` matches its dimensions (BoxSample).
            return unsafe { box4x_f32_neon(src, width, height) };
        }
        box4x_scalar(src, width, height)
    }
    fn box2x_hwc(
        src: &[f32],
        width: usize,
        height: usize,
        channels: usize,
    ) -> (Vec<f32>, usize, usize) {
        assert_image(src, width, height, channels);
        assert_interleaved(channels);
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            // SAFETY: avx2 detected at runtime; `src` matches its dimensions (BoxSample).
            return unsafe {
                if channels == RGB {
                    box2x_f32_hwc3_avx2(src, width, height)
                } else {
                    box2x_f32_hwc4_avx2(src, width, height)
                }
            };
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: `src` matches its dimensions (BoxSample).
            return unsafe { box2x_f32_hwc_neon(src, width, height, channels) };
        }
        box2x_interleaved::<f32>(src, width, height, channels)
    }
}

/// `box_levels_hwc` of an RGB image with RGBX output (`pad` in X): fused into
/// the SIMD kernels, else the RGB result expanded.
pub(crate) fn box_levels_rgb_to_rgbx(
    src: &[u8],
    width: usize,
    height: usize,
    levels: u32,
    pad: u8,
) -> (Vec<u8>, usize, usize) {
    assert_image(src, width, height, RGB);
    assert!((2..=3).contains(&levels));
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: avx2 detected at runtime; `src` holds the image.
        return unsafe { box_reduce_u8_hwc_avx2(src, (width, height), RGB, levels, Some(pad)) };
    }
    #[cfg(target_arch = "aarch64")]
    if use_neon() {
        // SAFETY: `src` holds the image.
        return unsafe { box_levels_u8_hwc_neon::<RGB>(src, (width, height), levels, Some(pad)) };
    }
    let (rgb, ow, oh) = u8::box_levels_hwc(src, width, height, RGB, levels);
    (expand_image(&rgb, oh, ow, pad), ow, oh)
}

/// Dispatching 2x2 box halving, RGB in -> RGBX out (`pad` fills the 4th sample).
/// `src` holds `width * height` RGB pixels (unchecked).
pub(crate) fn box2x_rgb_to_rgbx(
    src: &[u8],
    width: usize,
    height: usize,
    pad: u8,
) -> (Vec<u8>, usize, usize) {
    assert_image(src, width, height, RGB);
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: avx2 detected at runtime; `src` holds the image.
        return unsafe { box2x_u8_hwc3to4_avx2(src, width, height, pad) };
    }
    #[cfg(target_arch = "aarch64")]
    if use_neon() {
        // SAFETY: `src` holds the image.
        return unsafe { box2x_u8_hwc3to4_neon(src, width, height, pad) };
    }
    let (ow, oh) = (width / 2, height / 2);
    let mut out = vec![pad; ow * oh * RGBX];
    for oy in 0..oh {
        for ox in 0..ow {
            for ch in 0..RGB {
                let top_left = (2 * oy * width + 2 * ox) * RGB + ch;
                // SAFETY: ox < width/2 and oy < height/2 keep the whole 2x2 block inside src.
                out[(oy * ow + ox) * RGBX + ch] =
                    unsafe { box2_mean(src.as_ptr().add(top_left), RGB, width * RGB) };
            }
        }
    }
    (out, ow, oh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{CHANNELS, RandomSample, Rng, SIZES, random_samples, sizes_from};

    /// A box sample type with the kernels' fused 4x4 mean and its extremes
    /// (the saturation edges of the integer kernels).
    pub(crate) trait BoxTestSample: RandomSample + BoxSample {
        const MAX: Self;
        const ZERO: Self;

        /// The fused 4x4 mean of `block` (4 rows of 4 columns): integers
        /// rounded once, `(sum + 8) >> 4`; float32 in the planar kernels'
        /// order, columns down the rows, then `(c0 + c1) + (c2 + c3)`.
        fn mean16(block: [[Self; 4]; 4]) -> Self;
    }

    fn integer_mean16<T: Copy + Into<u32>>(block: [[T; 4]; 4]) -> u32 {
        let sum: u32 = block.iter().flatten().map(|&sample| sample.into()).sum();
        (sum + 8) >> 4
    }

    impl BoxTestSample for u8 {
        const MAX: Self = u8::MAX;
        const ZERO: Self = 0;
        fn mean16(block: [[Self; 4]; 4]) -> Self {
            u8::try_from(integer_mean16(block)).expect("a mean of u8 fits u8")
        }
    }

    impl BoxTestSample for u16 {
        const MAX: Self = u16::MAX;
        const ZERO: Self = 0;
        fn mean16(block: [[Self; 4]; 4]) -> Self {
            u16::try_from(integer_mean16(block)).expect("a mean of u16 fits u16")
        }
    }

    impl BoxTestSample for f32 {
        const MAX: Self = 1.0;
        const ZERO: Self = 0.0;
        fn mean16(block: [[Self; 4]; 4]) -> Self {
            let column =
                |col: usize| ((block[0][col] + block[1][col]) + block[2][col]) + block[3][col];
            ((column(0) + column(1)) + (column(2) + column(3))) * (1.0 / 16.0)
        }
    }

    /// The fused 4x box of a planar image, one block at a time.
    pub(crate) fn box4x_fused_reference<T: BoxTestSample>(
        src: &[T],
        width: usize,
        height: usize,
    ) -> (Vec<T>, usize, usize) {
        let (ow, oh) = (width / 4, height / 4);
        let mut out = Vec::with_capacity(ow * oh);
        for oy in 0..oh {
            for ox in 0..ow {
                let block = std::array::from_fn(|row| {
                    std::array::from_fn(|col| src[(4 * oy + row) * width + 4 * ox + col])
                });
                out.push(T::mean16(block));
            }
        }
        (out, ow, oh)
    }

    /// Two scalar 2x halvings: the chained (twice rounded) 4x box.
    pub(crate) fn box4x_chained_reference<T: Sample>(
        src: &[T],
        width: usize,
        height: usize,
    ) -> (Vec<T>, usize, usize) {
        let (half, half_width, half_height) = box2x_scalar(src, width, height);
        box2x_scalar(&half, half_width, half_height)
    }

    /// The planar pyramid's `times` halvings: a fused 4x per two levels, a
    /// scalar 2x for an odd last one.
    pub(crate) fn planar_pyramid_reference<T: BoxTestSample>(
        src: &[T],
        width: usize,
        height: usize,
        times: u32,
    ) -> (Vec<T>, usize, usize) {
        let mut level = (src.to_vec(), width, height);
        let mut left = times;
        while left > 0 {
            level = if left >= 2 {
                left -= 2;
                box4x_fused_reference(&level.0, level.1, level.2)
            } else {
                left -= 1;
                box2x_scalar(&level.0, level.1, level.2)
            };
        }
        level
    }

    /// `planar` applied to every channel of an interleaved image on its own.
    pub(crate) fn per_channel<T: Copy>(
        src: &[T],
        (width, height, channels): (usize, usize, usize),
        planar: impl Fn(&[T], usize, usize) -> (Vec<T>, usize, usize),
    ) -> (Vec<T>, usize, usize) {
        let pixels = width * height;
        let reduced: Vec<(Vec<T>, usize, usize)> = (0..channels)
            .map(|ch| {
                let plane: Vec<T> = (0..pixels)
                    .map(|pixel| src[pixel * channels + ch])
                    .collect();
                planar(&plane, width, height)
            })
            .collect();
        let (ow, oh) = (reduced[0].1, reduced[0].2);
        let out = (0..ow * oh)
            .flat_map(|pixel| reduced.iter().map(move |plane| plane.0[pixel]))
            .collect();
        (out, ow, oh)
    }

    /// The planar pyramid reference of every channel: what an interleaved
    /// pyramid (or a fused `levels` pass) must equal.
    pub(crate) fn interleaved_pyramid_reference<T: BoxTestSample>(
        src: &[T],
        dims: (usize, usize, usize),
        times: u32,
    ) -> (Vec<T>, usize, usize) {
        per_channel(src, dims, |plane, width, height| {
            planar_pyramid_reference(plane, width, height, times)
        })
    }

    /// RGB pixels widened to RGBX with `pad` in X.
    pub(crate) fn rgb_to_rgbx(rgb: &[u8], pad: u8) -> Vec<u8> {
        rgb.as_chunks::<RGB>()
            .0
            .iter()
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], pad])
            .collect()
    }

    /// The images every box kernel is checked on: uniform random samples,
    /// and samples at the extremes (zero or the largest value), where the
    /// integer kernels' sums and packs would overflow or saturate.
    pub(crate) fn test_images<T: BoxTestSample>(rng: &mut Rng, len: usize) -> [Vec<T>; 2] {
        let uniform = random_samples(rng, len);
        let extremes = random_samples::<u8>(rng, len)
            .into_iter()
            .map(|byte| if byte & 1 == 0 { T::ZERO } else { T::MAX })
            .collect();
        [uniform, extremes]
    }

    /// Widths past `SIZES` with several vector iterations per row, for the
    /// kernels with wide steps (32 to 64 output pixels, 8x reductions).
    const WIDE_WIDTHS: [usize; 4] = [190, 256, 263, 520];

    /// `sizes_from(minimum)`, and the wide widths with a few short heights.
    pub(crate) fn box_sizes(minimum: usize) -> Vec<(usize, usize)> {
        let short_heights = SIZES
            .into_iter()
            .filter(|&height| height >= minimum && height <= 17);
        sizes_from(minimum)
            .chain(
                WIDE_WIDTHS
                    .into_iter()
                    .flat_map(|width| short_heights.clone().map(move |height| (width, height))),
            )
            .collect()
    }

    /// The largest difference between two integer images and how many of
    /// their samples differ.
    pub(crate) fn integer_differences<T: Copy + Into<u32>>(
        first: &[T],
        second: &[T],
    ) -> (u32, usize) {
        assert_eq!(first.len(), second.len());
        let differences: Vec<u32> = first
            .iter()
            .zip(second)
            .map(|(&first, &second)| first.into().abs_diff(second.into()))
            .collect();
        let largest = differences.iter().copied().max().unwrap_or(0);
        (
            largest,
            differences.iter().filter(|&&step| step > 0).count(),
        )
    }

    /// Asserts two reduced images are the same, naming the first differing
    /// sample (rather than printing whole images).
    #[track_caller]
    pub(crate) fn assert_same_image<T: PartialEq + std::fmt::Debug>(
        actual: &(Vec<T>, usize, usize),
        expected: &(Vec<T>, usize, usize),
        context: &str,
    ) {
        assert_eq!(
            (actual.1, actual.2, actual.0.len()),
            (expected.1, expected.2, expected.0.len()),
            "{context}: output size (width, height, samples)"
        );
        let first_difference = actual
            .0
            .iter()
            .zip(&expected.0)
            .position(|(actual, expected)| actual != expected);
        if let Some(index) = first_difference {
            panic!(
                "{context}: sample {index} is {:?}, expected {:?}",
                actual.0[index], expected.0[index]
            );
        }
    }

    #[test]
    fn box2_mean_of_integers_rounds_the_exact_sum_half_up() {
        for sum in 0..=4 * u32::from(u8::MAX) {
            let samples: [u8; 4] = std::array::from_fn(|index| {
                u8::try_from((sum + 3 - index as u32) / 4).expect("a quarter of the sum fits u8")
            });
            // SAFETY: the 2x2 block of a 2 x 2 image.
            let mean = unsafe { box2_mean(samples.as_ptr(), 1, 2) };
            assert_eq!(u32::from(mean), (sum + 2) >> 2, "sum {sum}");
        }
        let mut rng = Rng::new(10);
        for samples in test_images::<u16>(&mut rng, 4 * 4096) {
            for block in samples.as_chunks::<4>().0 {
                let sum: u32 = block.iter().copied().map(u32::from).sum();
                // SAFETY: the 2x2 block of a 2 x 2 image.
                let mean = unsafe { box2_mean(block.as_ptr(), 1, 2) };
                assert_eq!(u32::from(mean), (sum + 2) >> 2, "{block:?}");
            }
        }
    }

    fn assert_interleaved_scalar_equals_planar<T: BoxTestSample>() {
        let mut rng = Rng::new(11);
        for channels in [RGB, RGBX] {
            for (width, height) in sizes_from(1) {
                for src in test_images::<T>(&mut rng, width * height * channels) {
                    assert_same_image(
                        &box2x_interleaved(&src, width, height, channels),
                        &per_channel(&src, (width, height, channels), box2x_scalar),
                        &format!("{width} x {height} x {channels}"),
                    );
                }
            }
        }
    }

    #[test]
    fn box2x_interleaved_equals_each_channel_halved_as_planar() {
        assert_interleaved_scalar_equals_planar::<u8>();
        assert_interleaved_scalar_equals_planar::<u16>();
        assert_interleaved_scalar_equals_planar::<f32>();
    }

    /// Every `BoxSample` method of `T` through its runtime dispatch (the
    /// best tier this CPU has) against the scalar references.
    fn assert_dispatch_matches_references<T: BoxTestSample>() {
        let mut rng = Rng::new(12);
        for (width, height) in box_sizes(1) {
            for src in test_images::<T>(&mut rng, width * height) {
                let context = format!("{width} x {height}");
                assert_same_image(
                    &T::box2x(&src, width, height),
                    &box2x_scalar(&src, width, height),
                    &format!("box2x {context}"),
                );
                assert_same_image(
                    &T::box4x(&src, width, height),
                    &box4x_fused_reference(&src, width, height),
                    &format!("box4x {context}"),
                );
            }
        }
        for channels in [RGB, RGBX] {
            for (width, height) in box_sizes(1) {
                let dims = (width, height, channels);
                for src in test_images::<T>(&mut rng, width * height * channels) {
                    let context = format!("{width} x {height} x {channels}");
                    assert_same_image(
                        &T::box2x_hwc(&src, width, height, channels),
                        &box2x_interleaved(&src, width, height, channels),
                        &format!("box2x_hwc {context}"),
                    );
                    for levels in [2, 3] {
                        assert_same_image(
                            &T::box_levels_hwc(&src, width, height, channels, levels),
                            &interleaved_pyramid_reference(&src, dims, levels),
                            &format!("box_levels_hwc {levels} {context}"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn dispatched_box_kernels_match_the_scalar_references() {
        assert_dispatch_matches_references::<u8>();
        assert_dispatch_matches_references::<u16>();
        assert_dispatch_matches_references::<f32>();
    }

    #[test]
    fn fused_box4x_rounds_once_where_chained_halvings_round_twice() {
        // The documented difference: at most one step, and real on ordinary
        // data.
        let mut rng = Rng::new(13);
        let (width, height) = (64, 32);
        let src: Vec<u8> = random_samples(&mut rng, width * height);
        let (largest, count) = integer_differences(
            &box4x_fused_reference(&src, width, height).0,
            &box4x_chained_reference(&src, width, height).0,
        );
        assert!(largest == 1 && count > 0, "u8: {largest}, {count}");
        let src: Vec<u16> = random_samples(&mut rng, width * height);
        let (largest, count) = integer_differences(
            &box4x_fused_reference(&src, width, height).0,
            &box4x_chained_reference(&src, width, height).0,
        );
        assert!(largest == 1 && count > 0, "u16: {largest}, {count}");
        // A block of mean 1.5 rounds up once, to 2; its 2x2 sums 5, 5, 5, 9
        // first round to 1, 1, 1, 2, whose mean 1.25 rounds to 1.
        let block: [u8; 16] = [2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 3, 2, 1, 1, 2, 2];
        assert_eq!(box4x_fused_reference(&block, 4, 4).0, [2]);
        assert_eq!(box4x_chained_reference(&block, 4, 4).0, [1]);
        // Every tier rounds once.
        assert_eq!(u8::box4x(&block, 4, 4).0, [2]);
    }

    #[test]
    fn fused_and_chained_float32_agree_where_the_sums_are_exact() {
        // Small integer samples keep every float32 sum and quarter exact, so
        // both orders give the true means.
        let mut rng = Rng::new(14);
        for (width, height) in sizes_from(4) {
            let src: Vec<f32> = random_samples::<u8>(&mut rng, width * height)
                .into_iter()
                .map(f32::from)
                .collect();
            assert_same_image(
                &f32::box4x(&src, width, height),
                &box4x_chained_reference(&src, width, height),
                &format!("{width} x {height}"),
            );
        }
    }

    #[test]
    fn box2x_rgb_to_rgbx_equals_the_rgb_halving_padded() {
        let mut rng = Rng::new(15);
        for (width, height) in box_sizes(1) {
            for src in test_images::<u8>(&mut rng, width * height * RGB) {
                let (rgb, ow, oh) = box2x_interleaved(&src, width, height, RGB);
                assert_same_image(
                    &box2x_rgb_to_rgbx(&src, width, height, 0xA5),
                    &(rgb_to_rgbx(&rgb, 0xA5), ow, oh),
                    &format!("{width} x {height}"),
                );
            }
        }
    }

    #[test]
    fn box_levels_rgb_to_rgbx_equals_the_planar_pyramid_padded() {
        let mut rng = Rng::new(16);
        for levels in [2, 3] {
            for (width, height) in box_sizes(1) {
                for src in test_images::<u8>(&mut rng, width * height * RGB) {
                    let (rgb, ow, oh) =
                        interleaved_pyramid_reference(&src, (width, height, RGB), levels);
                    assert_same_image(
                        &box_levels_rgb_to_rgbx(&src, width, height, levels, 7),
                        &(rgb_to_rgbx(&rgb, 7), ow, oh),
                        &format!("{levels} levels, {width} x {height}"),
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "fewer than width * height * channels")]
    fn box2x_hwc_rejects_a_short_image() {
        let src = vec![0u8; 4 * 4 * RGB - 1];
        let _ = u8::box2x_hwc(&src, 4, 4, RGB);
    }

    #[test]
    #[should_panic(expected = "interleaved images have 3 or 4 channels")]
    fn box2x_hwc_rejects_a_planar_image() {
        let src = vec![0u16; 16];
        let _ = u16::box2x_hwc(&src, 4, 4, 1);
    }

    #[test]
    #[should_panic(expected = "fewer than width * height * channels")]
    fn box4x_rejects_overflowing_dimensions() {
        let _ = f32::box4x(&[0.0; 16], usize::MAX, 2);
    }

    #[test]
    fn the_test_layouts_are_gray_rgb_and_rgbx() {
        assert_eq!(CHANNELS, [1, RGB, RGBX]);
    }

    /// Whether this CPU's tier writes 2x boxes in place (`box2x_into`).
    #[cfg(target_arch = "x86_64")]
    fn writes_in_place() -> bool {
        has_avx2()
    }
    #[cfg(target_arch = "aarch64")]
    fn writes_in_place() -> bool {
        use_neon()
    }

    /// A 2x box written in place equals the image the kernel allocates: its
    /// rows land `dst_stride` samples apart, and the samples between them
    /// keep the sentinel (no store spills past a row, as stores into a new
    /// image may).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn assert_into_matches_allocated<T: BoxTestSample>(sentinel: T) {
        const ROW_GAP: usize = 5;
        let mut rng = Rng::new(42);
        for channels in CHANNELS {
            for (width, height) in box_sizes(1) {
                let src = random_samples::<T>(&mut rng, width * height * channels);
                let (expected, ow, oh) = if channels == 1 {
                    box2x_scalar(&src, width, height)
                } else {
                    box2x_interleaved(&src, width, height, channels)
                };
                let dst_stride = ow * channels + ROW_GAP;
                let mut dst = vec![sentinel; oh * dst_stride];
                // SAFETY: src holds width x height pixels of `channels`
                // samples; dst holds oh rows dst_stride apart.
                let in_place = unsafe {
                    T::box2x_into(
                        &src,
                        (height, width, channels),
                        dst.as_mut_ptr(),
                        dst_stride,
                    )
                };
                assert_eq!(in_place, writes_in_place());
                if !in_place {
                    continue;
                }
                let context = format!("{width} x {height} x {channels}");
                let row_len = ow * channels;
                for oy in 0..oh {
                    let row = &dst[oy * dst_stride..][..dst_stride];
                    let expected_row = &expected[oy * row_len..][..row_len];
                    assert_eq!(&row[..row_len], expected_row, "{context}, row {oy}");
                    let gap = &row[row_len..];
                    assert!(gap.iter().all(|&sample| sample == sentinel), "{context}");
                }
            }
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn in_place_kernels_equal_the_allocated_images() {
        assert_into_matches_allocated::<u8>(0xA5);
        assert_into_matches_allocated::<u16>(0xA5A5);
        assert_into_matches_allocated::<f32>(-1.5);
    }
}

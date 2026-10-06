//! Native convolution resize of uint16 and float32 images (planar, RGB or
//! 4-sample pixels), with `fast_image_resize`'s filters and windows.
//!
//! Computed in f32: every output sample is a chain of fused multiply-adds
//! over its taps in tap order, vertical pass first (per row, into an f32
//! intermediate row), then the horizontal pass. The SIMD tiers run the
//! horizontal pass on transposed batches of rows — one register holds a
//! sample of every row of the batch, so each tap is a unit-stride load times
//! a broadcast weight, full lanes at any ratio and channel count — and keep
//! the per-lane tap order, so results are identical to the scalar reference
//! on every CPU. uint8 and uint16 convert exactly on load and round to
//! nearest (ties to even), clamped, on store. Windows longer than
//! `CHUNK_TAPS` (large reductions) sum in chunks of that many taps, the
//! chunks' sums added in f64: one chain of thousands of taps loses accuracy
//! (every term rounds the same way on flat input).

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

use std::cell::RefCell;
use std::sync::Arc;

use crate::convolve::{Geometry, Kernel, PlanCache, Windows};

/// Source bytes a batch's column strip may span (within L2).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const BATCH_SOURCE_BYTES: usize = 512 << 10;
/// Window bytes (taps x row) from which the vertical pass goes in strips:
/// one output row's window alone overflows L2.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const STRIPED_WINDOW_BYTES: usize = 768 << 10;
/// Strip widths are whole vertical-pass blocks on either tier.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const STRIP_ALIGN_SAMPLES: usize = 64;
/// Scratch kept per thread up to this size, so a very wide image doesn't pin
/// its buffers to the thread.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const SCRATCH_KEPT_BYTES: usize = 8 << 20;

/// Taps per FMA chain: longer windows sum in chunks of this many taps.
pub(crate) const CHUNK_TAPS: usize = 256;

/// A sample type computed in f32.
pub(crate) trait FloatSample: Copy + Default + Send + Sync {
    fn to_f32(self) -> f32;
    fn from_f32(value: f32) -> Self;
}

impl FloatSample for f32 {
    #[inline]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline]
    fn from_f32(value: f32) -> f32 {
        value
    }
}

impl FloatSample for u8 {
    #[inline]
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
    #[inline]
    fn from_f32(value: f32) -> u8 {
        value.round_ties_even().clamp(0.0, 255.0) as u8
    }
}

impl FloatSample for u16 {
    #[inline]
    fn to_f32(self) -> f32 {
        f32::from(self)
    }
    #[inline]
    fn from_f32(value: f32) -> u16 {
        value.round_ties_even().clamp(0.0, 65535.0) as u16
    }
}

/// Everything precomputed for one resize (the weights are dtype-independent).
pub(crate) struct FloatPlan {
    pub(crate) rows: Windows,
    pub(crate) cols: Windows,
    /// `cols` in chunks of `CHUNK_TAPS` taps where a window is longer.
    pub(crate) col_chunks: Option<Vec<Windows>>,
    pub(crate) channels: usize,
}

thread_local! {
    static PLANS: RefCell<PlanCache<FloatPlan>> = const { RefCell::new(PlanCache::new()) };
}

impl FloatPlan {
    pub(crate) fn new(rows: Windows, cols: Windows, channels: usize) -> FloatPlan {
        FloatPlan {
            rows,
            col_chunks: cols.chunks(CHUNK_TAPS),
            cols,
            channels,
        }
    }

    pub(crate) fn cached(geometry: Geometry, kernel: Kernel, channels: usize) -> Arc<FloatPlan> {
        let build = || {
            let Geometry {
                src: (width, height),
                extent: (extent_width, extent_height),
                dst: (ow, oh),
            } = geometry;
            FloatPlan::new(
                Windows::over(height, extent_height, oh, kernel),
                Windows::over(width, extent_width, ow, kernel),
                channels,
            )
        };
        PLANS.with_borrow_mut(|plans| plans.get_or_build((geometry, kernel, channels), build))
    }
}

/// Resize `src` (width x height pixels of `plan.channels` samples) by `plan`.
pub(crate) fn resize_float<T: FloatResample>(
    src: &[T],
    (width, height): (usize, usize),
    plan: &FloatPlan,
) -> Vec<T> {
    debug_assert!(src.len() >= width * height * plan.channels);
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_avx2_fma() {
        return T::resize_transposed(src, width, plan);
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::use_neon() {
        return T::resize_transposed(src, width, plan);
    }
    resize_rows(src, width, plan)
}

/// The scalar reference: per row, the vertical then the horizontal pass.
pub(crate) fn resize_rows<T: FloatSample>(src: &[T], width: usize, plan: &FloatPlan) -> Vec<T> {
    let channels = plan.channels;
    let (row_samples, out_row_samples) = (width * channels, plan.cols.len() * channels);
    let mut out = vec![T::default(); out_row_samples * plan.rows.len()];
    let mut mid = vec![0f32; row_samples];
    let mut chunk_sums = ChunkSums::default();
    let cols: &[Windows] = plan
        .col_chunks
        .as_deref()
        .unwrap_or(std::slice::from_ref(&plan.cols));
    for (win, dst) in plan.rows.iter().zip(out.chunks_exact_mut(out_row_samples)) {
        vertical_in_chunks(
            vertical,
            &src[win.start * row_samples..win.end() * row_samples],
            row_samples,
            win.f32,
            &mut mid,
            &mut chunk_sums,
        );
        for (output, sample) in dst.iter_mut().enumerate() {
            let (col, ch) = (output / channels, output % channels);
            let sum = cols.iter().fold(0f64, |sum, chunk| {
                let window = chunk.get(col);
                let taps = (window.start..window.end()).map(|x| mid[x * channels + ch]);
                sum + f64::from(fma_chain(taps, window.f32))
            });
            *sample = T::from_f32(sum as f32);
        }
    }
    out
}

/// Scratch of the chunked sums: one chunk's f32 sums and the f64 total.
#[derive(Default)]
pub(crate) struct ChunkSums {
    part: Vec<f32>,
    total: Vec<f64>,
}

impl ChunkSums {
    /// The sums of `chunks` chunks (`chunk(index, part)` writes chunk
    /// `index`'s `out.len()` sums to `part`) added in f64, rounded to f32 into
    /// `out`. A single chunk writes straight to `out`.
    fn sum_into(
        &mut self,
        chunks: usize,
        out: &mut [f32],
        mut chunk: impl FnMut(usize, &mut [f32]),
    ) {
        if chunks == 1 {
            return chunk(0, out);
        }
        self.part.resize(out.len(), 0.0);
        self.total.clear();
        self.total.resize(out.len(), 0.0);
        for index in 0..chunks {
            chunk(index, &mut self.part[..out.len()]);
            for (total, &part) in self.total.iter_mut().zip(&self.part) {
                *total += f64::from(part);
            }
        }
        for (out, &total) in out.iter_mut().zip(&self.total) {
            *out = total as f32;
        }
    }
}

/// `vertical` (any tier's pass) over `weights` in chunks of `CHUNK_TAPS`
/// taps, summed by `ChunkSums::sum_into`.
pub(crate) fn vertical_in_chunks<T>(
    vertical: impl Fn(&[T], usize, &[f32], &mut [f32]),
    rows: &[T],
    stride: usize,
    weights: &[f32],
    mid: &mut [f32],
    sums: &mut ChunkSums,
) {
    sums.sum_into(weights.len().div_ceil(CHUNK_TAPS), mid, |chunk, part| {
        let taps = chunk * CHUNK_TAPS..((chunk + 1) * CHUNK_TAPS).min(weights.len());
        vertical(&rows[taps.start * stride..], stride, &weights[taps], part);
    });
}

/// `mid[x] = sum_t weights[t] * rows[t][x]`, as an FMA chain in tap order.
pub(crate) fn vertical<T: FloatSample>(
    rows: &[T],
    stride: usize,
    weights: &[f32],
    mid: &mut [f32],
) {
    for (x, sum) in mid.iter_mut().enumerate() {
        *sum = fma_chain(
            (0..weights.len()).map(|tap| rows[tap * stride + x].to_f32()),
            weights,
        );
    }
}

/// `sum_t weights[t] * samples[t]` as `acc = sample * weight + acc` from 0,
/// in tap order.
#[inline]
pub(crate) fn fma_chain(samples: impl Iterator<Item = f32>, weights: &[f32]) -> f32 {
    samples
        .zip(weights)
        .fold(0.0, |acc, (sample, &weight)| sample.mul_add(weight, acc))
}

/// A sample type `resize_float` takes: with its SIMD path where one exists.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) trait FloatResample: TransposedTier {}
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl<T: TransposedTier> FloatResample for T {}
/// A sample type `resize_float` takes.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) trait FloatResample: FloatSample {}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
impl<T: FloatSample> FloatResample for T {}

/// The SIMD path per sample type (the kernels load and store per dtype).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) trait TransposedTier: FloatSample {
    fn resize_transposed(src: &[Self], width: usize, plan: &FloatPlan) -> Vec<Self>;
}

#[cfg(target_arch = "x86_64")]
macro_rules! transposed_tier {
    (
        $t:ty,
        $vertical_avx512:ident,
        $store_avx512:ident,
        $vertical_avx2:ident,
        $store_avx2:ident
    ) => {
        impl TransposedTier for $t {
            fn resize_transposed(src: &[$t], width: usize, plan: &FloatPlan) -> Vec<$t> {
                use crate::convolve_float::x86;
                if crate::cpu::has_avx512() {
                    // SAFETY: avx512f detected; the driver sizes every buffer.
                    unsafe {
                        transposed::<$t, 16>(
                            src,
                            width,
                            plan,
                            |rows, stride, weights, mid| {
                                x86::$vertical_avx512(rows, stride, weights, mid)
                            },
                            x86::transpose_f32_rows_avx512,
                            x86::horizontal_f32_transposed_avx512,
                            x86::$store_avx512,
                        )
                    }
                } else {
                    // SAFETY: avx2+fma detected; the driver sizes every buffer.
                    unsafe {
                        transposed::<$t, 8>(
                            src,
                            width,
                            plan,
                            |rows, stride, weights, mid| {
                                x86::$vertical_avx2(rows, stride, weights, mid)
                            },
                            x86::transpose_f32_rows_avx2,
                            x86::horizontal_f32_transposed_avx2,
                            x86::$store_avx2,
                        )
                    }
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
transposed_tier!(
    f32,
    vertical_f32_avx512,
    untranspose_store_f32_avx512,
    vertical_f32_avx2,
    untranspose_store_f32_avx2
);
#[cfg(target_arch = "x86_64")]
transposed_tier!(
    u8,
    vertical_u8_avx512,
    untranspose_store_u8_avx512,
    vertical_u8_avx2,
    untranspose_store_u8_avx2
);
#[cfg(target_arch = "x86_64")]
transposed_tier!(
    u16,
    vertical_u16_avx512,
    untranspose_store_u16_avx512,
    vertical_u16_avx2,
    untranspose_store_u16_avx2
);

// NEON: 8 rows per batch.
#[cfg(target_arch = "aarch64")]
macro_rules! neon_transposed_tier {
    ($t:ty, $vertical:ident, $store:ident) => {
        impl TransposedTier for $t {
            fn resize_transposed(src: &[$t], width: usize, plan: &FloatPlan) -> Vec<$t> {
                // SAFETY: NEON is baseline on aarch64; the driver sizes every
                // buffer.
                unsafe {
                    transposed::<$t, { neon::ROWS }>(
                        src,
                        width,
                        plan,
                        |rows, stride, weights, mid| neon::$vertical(rows, stride, weights, mid),
                        neon::transpose_f32_rows_neon,
                        neon::horizontal_f32_transposed_neon,
                        neon::$store,
                    )
                }
            }
        }
    };
}
#[cfg(target_arch = "aarch64")]
neon_transposed_tier!(f32, vertical_f32_neon, untranspose_store_f32_neon);
#[cfg(target_arch = "aarch64")]
neon_transposed_tier!(u16, vertical_u16_neon, untranspose_store_u16_neon);
#[cfg(target_arch = "aarch64")]
neon_transposed_tier!(u8, vertical_u8_neon, untranspose_store_u8_neon);

/// The SIMD driver for `ROWS` rows per batch: each batch's vertical passes, a
/// transpose (sample x of row r at `mid_t[x * ROWS + r]`), the horizontal pass
/// over all `ROWS` rows at once, and the transpose back with the dtype's store.
///
/// # Safety
/// `transpose`, `horizontal` and `store` are the `ROWS`-row kernels of
/// `convolve_float::x86` or `convolve_float::neon` (or take the same bounds),
/// and their target features are available. The driver sizes every buffer
/// they are given.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
unsafe fn transposed<T: FloatSample, const ROWS: usize>(
    src: &[T],
    width: usize,
    plan: &FloatPlan,
    vertical: impl Fn(&[T], usize, &[f32], &mut [f32]),
    transpose: unsafe fn(&[f32], usize, &mut [f32]),
    horizontal: unsafe fn(&[f32], &Windows, usize, &mut [f32]),
    store: unsafe fn(&[f32], usize, *mut T, usize, usize),
) -> Vec<T> {
    let channels = plan.channels;
    let (ow, oh) = (plan.cols.len(), plan.rows.len());
    let (row_samples, out_row_samples) = (width * channels, ow * channels);
    let sample_bytes = std::mem::size_of::<T>();
    // mid's row stride: whole ROWS-sample tiles for the transposes, plus one
    // more, so a power-of-2 row width doesn't put the rows a power of 2
    // apart (they would share L1 sets).
    let mid_stride = row_samples.next_multiple_of(ROWS) + ROWS;
    let out_t_samples = out_row_samples.next_multiple_of(ROWS);
    let mut out_buf = crate::cpu::unwritten_output::<T>(out_row_samples * oh);
    // Scratch kept per thread between calls, so repeated small resizes skip
    // the buffers' first-touch page faults. A call writes everything it
    // uses (the row padding passes through the transposes unused).
    let TransposeScratch {
        mut mid,
        mut mid_t,
        mut out_t,
    } = SCRATCH.with_borrow_mut(std::mem::take);
    let (mut vertical_sums, mut horizontal_sums) = (ChunkSums::default(), ChunkSums::default());
    let cols: &[Windows] = plan
        .col_chunks
        .as_deref()
        .unwrap_or(std::slice::from_ref(&plan.cols));
    mid.resize(ROWS * mid_stride, 0.0);
    mid_t.resize(mid_stride * ROWS, 0.0);
    out_t.resize(out_t_samples * ROWS, 0.0);
    let out_ptr = out_buf.as_mut_ptr().cast::<T>();
    let batches = |lo: usize, hi: usize| {
        for batch in (lo..hi).step_by(ROWS) {
            let rows = ROWS.min(hi - batch);
            // Short batch: repeat its last row (those outputs are dropped).
            let win = |row: usize| plan.rows.get((batch + row).min(hi - 1));
            // Column strips narrow enough that the batch's source rows stay in
            // L2 while all ROWS rows read them — only where one output row's
            // window alone overflows L2, since elsewhere the strips' overhead
            // outweighs the reuse. At least one aligned strip: a batch whose
            // rows alone pass the budget still takes a strip of samples.
            let batch_rows = win(ROWS - 1).end() - win(0).start;
            let strip = if win(0).f32.len() * row_samples * sample_bytes >= STRIPED_WINDOW_BYTES {
                (BATCH_SOURCE_BYTES / (batch_rows * sample_bytes))
                    .max(1)
                    .next_multiple_of(STRIP_ALIGN_SAMPLES)
                    .min(row_samples)
            } else {
                row_samples
            };
            for x0 in (0..row_samples).step_by(strip) {
                let strip_width = strip.min(row_samples - x0);
                for row in 0..ROWS {
                    let window = win(row);
                    let src_rows = &src[window.start * row_samples + x0
                        ..(window.end() - 1) * row_samples + x0 + strip_width];
                    let mid_row = row * mid_stride + x0;
                    vertical_in_chunks(
                        &vertical,
                        src_rows,
                        row_samples,
                        window.f32,
                        &mut mid[mid_row..mid_row + strip_width],
                        &mut vertical_sums,
                    );
                }
            }
            // SAFETY: the kernels' features are available (contract); mid and
            // mid_t hold ROWS rows of mid_stride, a multiple of ROWS.
            unsafe { transpose(&mid, mid_stride, &mut mid_t) };
            horizontal_sums.sum_into(cols.len(), &mut out_t, |chunk, part| {
                // SAFETY: as above; every window (and chunk of one) ends by
                // the row's width < mid_stride, and `part` holds ROWS x
                // out_t_samples, whole tiles of the output row.
                unsafe { horizontal(&mid_t, &cols[chunk], channels, part) };
            });
            // SAFETY: as above; rows [batch, batch + rows) of the output,
            // which only this closure call writes (with_prefault's contract).
            unsafe {
                store(
                    &out_t,
                    out_row_samples,
                    out_ptr.add(batch * out_row_samples),
                    out_row_samples,
                    rows,
                );
            }
        }
    };
    let out_bytes = out_buf.len() * sample_bytes;
    // SAFETY: `out_buf` is this function's, and `batches(lo, hi)` writes only
    // rows [lo, hi) (whole rows, no slop).
    unsafe {
        crate::cpu::with_prefault(
            out_ptr as *mut u8,
            out_bytes,
            out_row_samples * sample_bytes,
            oh,
            batches,
        );
    }
    // SAFETY: every batch stores all of its rows.
    let out = unsafe { crate::cpu::written_output(out_buf) };
    if (mid.capacity() + mid_t.capacity() + out_t.capacity()) * std::mem::size_of::<f32>()
        <= SCRATCH_KEPT_BYTES
    {
        SCRATCH.with_borrow_mut(|scratch| *scratch = TransposeScratch { mid, mid_t, out_t });
    }
    out
}

/// The transposed driver's buffers: `ROWS` intermediate rows, the same
/// transposed, and the transposed output sums.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[derive(Default)]
struct TransposeScratch {
    mid: Vec<f32>,
    mid_t: Vec<f32>,
    out_t: Vec<f32>,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
thread_local! {
    static SCRATCH: RefCell<TransposeScratch> =
        const {
            RefCell::new(TransposeScratch {
                mid: Vec::new(),
                mid_t: Vec::new(),
                out_t: Vec::new(),
            })
        };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convolve::RGB;
    use crate::testing::{CHANNELS, RandomSample, Rng, random_samples};

    const KERNELS: [Kernel; 4] = [
        Kernel::Box,
        Kernel::Bilinear,
        Kernel::CatmullRom,
        Kernel::Lanczos3,
    ];

    /// Column ratios (input width, output width): the smallest images,
    /// ratios around the batch and register widths, upscales, and large
    /// reductions.
    const COLUMN_CASES: [(usize, usize); 14] = [
        (1, 1),
        (1, 3),
        (3, 1),
        (7, 16),
        (16, 7),
        (17, 33),
        (31, 15),
        (33, 64),
        (64, 32),
        (130, 65),
        (130, 37),
        (130, 200),
        (67, 201),
        (640, 10),
    ];

    /// Row ratios: batches of 1 to 16 rows (a full AVX-512 batch, two AVX2
    /// ones, short last batches), upscales, and windows long enough for the
    /// grouped vertical taps.
    const ROW_CASES: [(usize, usize); 7] = [
        (1, 1),
        (3, 7),
        (17, 9),
        (33, 16),
        (40, 5),
        (40, 37),
        (130, 2),
    ];

    fn kernel_name(kernel: Kernel) -> &'static str {
        match kernel {
            Kernel::Box => "box",
            Kernel::Bilinear => "bilinear",
            Kernel::CatmullRom => "catmullrom",
            Kernel::Lanczos3 => "lanczos3",
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn has_avx2_fma() -> bool {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }

    #[cfg(target_arch = "x86_64")]
    fn has_avx512() -> bool {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
    }

    /// A sample type's dtype-specific kernels of both tiers.
    #[cfg(target_arch = "x86_64")]
    trait TierKernels: TransposedTier + RandomSample {
        const VERTICAL_AVX512: unsafe fn(&[Self], usize, &[f32], &mut [f32]);
        const STORE_AVX512: unsafe fn(&[f32], usize, *mut Self, usize, usize);
        const VERTICAL_AVX2: unsafe fn(&[Self], usize, &[f32], &mut [f32]);
        const STORE_AVX2: unsafe fn(&[f32], usize, *mut Self, usize, usize);
    }

    #[cfg(target_arch = "x86_64")]
    impl TierKernels for f32 {
        const VERTICAL_AVX512: unsafe fn(&[f32], usize, &[f32], &mut [f32]) =
            x86::vertical_f32_avx512;
        const STORE_AVX512: unsafe fn(&[f32], usize, *mut f32, usize, usize) =
            x86::untranspose_store_f32_avx512;
        const VERTICAL_AVX2: unsafe fn(&[f32], usize, &[f32], &mut [f32]) = x86::vertical_f32_avx2;
        const STORE_AVX2: unsafe fn(&[f32], usize, *mut f32, usize, usize) =
            x86::untranspose_store_f32_avx2;
    }

    #[cfg(target_arch = "x86_64")]
    impl TierKernels for u8 {
        const VERTICAL_AVX512: unsafe fn(&[u8], usize, &[f32], &mut [f32]) =
            x86::vertical_u8_avx512;
        const STORE_AVX512: unsafe fn(&[f32], usize, *mut u8, usize, usize) =
            x86::untranspose_store_u8_avx512;
        const VERTICAL_AVX2: unsafe fn(&[u8], usize, &[f32], &mut [f32]) = x86::vertical_u8_avx2;
        const STORE_AVX2: unsafe fn(&[f32], usize, *mut u8, usize, usize) =
            x86::untranspose_store_u8_avx2;
    }

    #[cfg(target_arch = "x86_64")]
    impl TierKernels for u16 {
        const VERTICAL_AVX512: unsafe fn(&[u16], usize, &[f32], &mut [f32]) =
            x86::vertical_u16_avx512;
        const STORE_AVX512: unsafe fn(&[f32], usize, *mut u16, usize, usize) =
            x86::untranspose_store_u16_avx512;
        const VERTICAL_AVX2: unsafe fn(&[u16], usize, &[f32], &mut [f32]) = x86::vertical_u16_avx2;
        const STORE_AVX2: unsafe fn(&[f32], usize, *mut u16, usize, usize) =
            x86::untranspose_store_u16_avx2;
    }

    /// The transposed driver on the AVX-512 tier (16-row batches).
    #[cfg(target_arch = "x86_64")]
    fn resize_avx512<T: TierKernels>(src: &[T], width: usize, plan: &FloatPlan) -> Vec<T> {
        let vertical = |rows: &[T], stride: usize, weights: &[f32], mid: &mut [f32]| {
            // SAFETY: the caller detected AVX-512F; the driver passes each
            // window's rows of the strip.
            unsafe { (T::VERTICAL_AVX512)(rows, stride, weights, mid) }
        };
        // SAFETY: the caller detected AVX-512F; these are the 16-row kernels.
        unsafe {
            transposed::<T, 16>(
                src,
                width,
                plan,
                vertical,
                x86::transpose_f32_rows_avx512,
                x86::horizontal_f32_transposed_avx512,
                T::STORE_AVX512,
            )
        }
    }

    /// The transposed driver on the AVX2 tier (8-row batches).
    #[cfg(target_arch = "x86_64")]
    fn resize_avx2<T: TierKernels>(src: &[T], width: usize, plan: &FloatPlan) -> Vec<T> {
        let vertical = |rows: &[T], stride: usize, weights: &[f32], mid: &mut [f32]| {
            // SAFETY: the caller detected AVX2 and FMA; the driver passes each
            // window's rows of the strip.
            unsafe { (T::VERTICAL_AVX2)(rows, stride, weights, mid) }
        };
        // SAFETY: the caller detected AVX2 and FMA; these are the 8-row kernels.
        unsafe {
            transposed::<T, 8>(
                src,
                width,
                plan,
                vertical,
                x86::transpose_f32_rows_avx2,
                x86::horizontal_f32_transposed_avx2,
                T::STORE_AVX2,
            )
        }
    }

    /// A sample type the tests resize: with its x86 tiers' kernels there.
    #[cfg(target_arch = "x86_64")]
    trait TestFloat: TierKernels {}
    #[cfg(target_arch = "x86_64")]
    impl<T: TierKernels> TestFloat for T {}
    /// A sample type the tests resize.
    #[cfg(not(target_arch = "x86_64"))]
    trait TestFloat: FloatResample + RandomSample {}
    #[cfg(not(target_arch = "x86_64"))]
    impl<T: FloatResample + RandomSample> TestFloat for T {}

    fn assert_same<T: PartialEq + std::fmt::Debug>(actual: &[T], expected: &[T], context: &str) {
        assert_eq!(actual.len(), expected.len(), "{context}: length");
        if let Some(index) = (0..actual.len()).find(|&index| actual[index] != expected[index]) {
            panic!(
                "{context}: sample {index} is {:?}, expected {:?}",
                actual[index], expected[index]
            );
        }
    }

    /// The entry point and each tier's driver equal the scalar reference bit
    /// for bit (the same FMA chains in the same order).
    fn assert_tiers_match_reference<T: TestFloat>(
        src: &[T],
        (width, height): (usize, usize),
        plan: &FloatPlan,
        context: &str,
    ) {
        let expected = resize_rows(src, width, plan);
        assert_same(
            &resize_float(src, (width, height), plan),
            &expected,
            &format!("{context}, dispatched"),
        );
        #[cfg(target_arch = "x86_64")]
        {
            if has_avx512() {
                assert_same(
                    &resize_avx512(src, width, plan),
                    &expected,
                    &format!("{context}, avx512"),
                );
            }
            if has_avx2_fma() {
                assert_same(
                    &resize_avx2(src, width, plan),
                    &expected,
                    &format!("{context}, avx2"),
                );
            }
        }
    }

    fn assert_resize_matches_reference<T: TestFloat>(seed: u64) {
        let mut rng = Rng::new(seed);
        for kernel in KERNELS {
            for channels in CHANNELS {
                for (width, ow) in COLUMN_CASES {
                    for (height, oh) in ROW_CASES {
                        let src: Vec<T> = random_samples(&mut rng, width * height * channels);
                        let plan = FloatPlan::new(
                            Windows::new(height, oh, kernel),
                            Windows::new(width, ow, kernel),
                            channels,
                        );
                        let context = format!(
                            "{} {channels} channels {width} x {height} -> {ow} x {oh}",
                            kernel_name(kernel)
                        );
                        assert_tiers_match_reference(&src, (width, height), &plan, &context);
                    }
                }
            }
        }
    }

    #[test]
    fn float32_resize_equals_reference() {
        assert_resize_matches_reference::<f32>(1);
    }

    /// Random uint16 samples over the whole range, so the filters' overshoot
    /// clamps at both ends.
    #[test]
    fn uint16_resize_equals_reference() {
        assert_resize_matches_reference::<u16>(2);
    }

    /// uint8 takes the engine for large reductions; any ratio must agree.
    #[test]
    fn uint8_resize_equals_reference() {
        assert_resize_matches_reference::<u8>(4);
    }

    /// Windows longer than `CHUNK_TAPS` sum in chunks, in both passes, the
    /// same on every tier: reductions whose windows take one, two and more
    /// chunks along either axis.
    fn assert_chunked_matches_reference<T: TestFloat>(seed: u64) {
        let mut rng = Rng::new(seed);
        let cases = [
            ((1000, 3), (40, 3)),
            ((40, 3), (1000, 3)),
            ((2600, 2), (600, 1)),
            ((300, 1), (2600, 2)),
        ];
        for kernel in [Kernel::Box, Kernel::Lanczos3] {
            for channels in CHANNELS {
                for ((width, ow), (height, oh)) in cases {
                    let src: Vec<T> = random_samples(&mut rng, width * height * channels);
                    let plan = FloatPlan::new(
                        Windows::new(height, oh, kernel),
                        Windows::new(width, ow, kernel),
                        channels,
                    );
                    let context = format!(
                        "chunked {} {channels} channels {width} x {height} -> {ow} x {oh}",
                        kernel_name(kernel)
                    );
                    assert_tiers_match_reference(&src, (width, height), &plan, &context);
                }
            }
        }
    }

    #[test]
    fn chunked_resize_equals_reference() {
        assert_chunked_matches_reference::<u8>(5);
        assert_chunked_matches_reference::<u16>(6);
        assert_chunked_matches_reference::<f32>(7);
    }

    /// A flat image reduced to one pixel through tens of thousands of taps
    /// keeps its value: the chunked sums don't drift.
    #[test]
    fn long_windows_keep_flat_values() {
        for kernel in KERNELS {
            let (width, height) = (40000, 3);
            let plan = FloatPlan::new(
                Windows::new(height, 1, kernel),
                Windows::new(width, 1, kernel),
                1,
            );
            let flat = vec![40000u16; width * height];
            let out = resize_float(&flat, (width, height), &plan);
            assert_eq!(out, [40000], "{}", kernel_name(kernel));
            let white = vec![255u8; width * height];
            let out = resize_float(&white, (width, height), &plan);
            assert_eq!(out, [255], "{}", kernel_name(kernel));
        }
    }

    /// A window whose rows overflow L2 runs the vertical pass in column
    /// strips.
    fn assert_striped_matches_reference<T: TestFloat>(width: usize, channels: usize) {
        let (height, oh, ow) = (1024, 4, width / 2 + 1);
        let plan = FloatPlan::new(
            Windows::new(height, oh, Kernel::Lanczos3),
            Windows::new(width, ow, Kernel::Lanczos3),
            channels,
        );
        let window_bytes = plan.rows.get(0).f32.len() * width * channels * size_of::<T>();
        assert!(window_bytes >= STRIPED_WINDOW_BYTES, "not striped");
        let mut rng = Rng::new(3);
        let src: Vec<T> = random_samples(&mut rng, width * height * channels);
        let context = format!("striped {width} x {height} x {channels}");
        assert_tiers_match_reference(&src, (width, height), &plan, &context);
    }

    #[test]
    fn striped_vertical_pass_equals_reference() {
        assert_striped_matches_reference::<f32>(64, 4);
        assert_striped_matches_reference::<u16>(171, RGB);
    }

    /// A vertical kernel, whether this host runs it, and its tier's name.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    type VerticalTier<T> = (
        bool,
        unsafe fn(&[T], usize, &[f32], &mut [f32]),
        &'static str,
    );

    /// The vertical kernels of every tier: rows narrower than a register
    /// (the scalar fallback), widths around the registers and blocks, strides
    /// past the row (column strips), and the strides and window lengths where
    /// the taps run in groups (4 KB aliasing, 2 KB on AVX2, 40+ taps).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn assert_vertical_matches_reference<T: FloatSample + RandomSample>(
        rng: &mut Rng,
        tiers: &[VerticalTier<T>],
    ) {
        let widths = [1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 100, 130];
        let sample_bytes = size_of::<T>();
        let aliased = [
            (4096 / sample_bytes, 12),
            (4096 / sample_bytes, 16),
            (2048 / sample_bytes, 16),
        ];
        let cases = widths
            .into_iter()
            .flat_map(|width| [1, 2, 5, 16, 41].map(|taps| (width, taps)))
            .chain(aliased);
        for (width, taps) in cases {
            let weights: Vec<f32> = random_samples::<f32>(rng, taps)
                .into_iter()
                .map(|weight| 2.0 * weight - 0.5)
                .collect();
            for stride in [width, width + 13] {
                let rows: Vec<T> = random_samples(rng, (taps - 1) * stride + width);
                let mut expected = vec![0.0; width];
                vertical(&rows, stride, &weights, &mut expected);
                let context = format!("width {width}, stride {stride}, {taps} taps");
                for &(available, kernel, name) in tiers {
                    if available {
                        let mut mid = vec![f32::NAN; width];
                        // SAFETY: the kernel's features detected; weights is
                        // not empty and rows holds taps rows of width, stride
                        // apart.
                        unsafe { kernel(&rows, stride, &weights, &mut mid) };
                        let (actual, wanted): (Vec<u32>, Vec<u32>) = (
                            mid.iter().map(|sum| sum.to_bits()).collect(),
                            expected.iter().map(|sum| sum.to_bits()).collect(),
                        );
                        assert_same(&actual, &wanted, &format!("{context}, {name}"));
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn x86_vertical_tiers<T: TierKernels>() -> [VerticalTier<T>; 2] {
        [
            (has_avx512(), T::VERTICAL_AVX512, "avx512"),
            (has_avx2_fma(), T::VERTICAL_AVX2, "avx2"),
        ]
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vertical_kernels_equal_reference() {
        let mut rng = Rng::new(4);
        assert_vertical_matches_reference::<f32>(&mut rng, &x86_vertical_tiers());
        assert_vertical_matches_reference::<u16>(&mut rng, &x86_vertical_tiers());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn vertical_kernels_equal_reference() {
        let mut rng = Rng::new(4);
        let neon = crate::cpu::use_neon();
        assert_vertical_matches_reference::<f32>(
            &mut rng,
            &[(neon, neon::vertical_f32_neon, "neon")],
        );
        assert_vertical_matches_reference::<u16>(
            &mut rng,
            &[(neon, neon::vertical_u16_neon, "neon")],
        );
    }
}

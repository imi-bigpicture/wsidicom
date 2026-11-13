//! Convolution resize computed in f32: uint16 images, and uint8 images
//! reduced past the fixed-point engine's range, with `fast_image_resize`'s
//! filters and windows.
//!
//! Every output sample is an FMA chain over its taps in tap order: the
//! vertical pass into an f32 intermediate row (`mid`), then the horizontal
//! pass. The SIMD tiers run the horizontal pass on transposed batches of rows
//! (a register holds one sample of every row in the batch) and keep each
//! lane's tap order, so every CPU matches the scalar reference bit for bit.
//! Windows longer than `CHUNK_TAPS` sum in chunks whose sums add in f64: one
//! chain of thousands of taps drifts, as every term rounds the same way on
//! flat input.

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

use std::cell::RefCell;
use std::sync::Arc;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::convolve::TransposeScratch;
use crate::convolve::{Geometry, Kernel, PlanCache, Windows};

/// Source bytes a batch's column strip may span (within L2).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const BATCH_SOURCE_BYTES: usize = 512 << 10;
/// Window bytes (taps x row) from which the vertical pass goes in strips, as
/// one output row's window alone overflows L2.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const STRIPED_WINDOW_BYTES: usize = 768 << 10;
/// Strip widths are whole vertical-pass blocks on either tier.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const STRIP_ALIGN_SAMPLES: usize = 64;

/// Taps per FMA chain: longer windows sum in chunks of this many taps.
pub(crate) const CHUNK_TAPS: usize = 256;

/// A sample type computed in f32: exact on load, rounded to nearest (ties to
/// even) and saturated on store.
pub(crate) trait FloatSample: Copy + Default + Send + Sync {
    fn to_f32(self) -> f32;
    fn from_f32(value: f32) -> Self;
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
    /// `cols` split into windows of at most `CHUNK_TAPS` taps, where any is
    /// longer.
    pub(crate) chunked_cols: Option<Vec<Windows>>,
    pub(crate) channels: usize,
}

thread_local! {
    static PLANS: RefCell<PlanCache<FloatPlan>> = const { RefCell::new(PlanCache::new()) };
}

impl FloatPlan {
    pub(crate) fn new(rows: Windows, cols: Windows, channels: usize) -> FloatPlan {
        FloatPlan {
            rows,
            chunked_cols: cols.chunks(CHUNK_TAPS),
            cols,
            channels,
        }
    }

    pub(crate) fn cached(geometry: Geometry, kernel: Kernel, channels: usize) -> Arc<FloatPlan> {
        let build = || {
            let Geometry {
                src: (width, height),
                extent: (extent_width, extent_height),
                dst: (out_width, out_height),
            } = geometry;
            FloatPlan::new(
                Windows::over(height, extent_height, out_height, kernel),
                Windows::over(width, extent_width, out_width, kernel),
                channels,
            )
        };
        PLANS.with_borrow_mut(|plans| plans.get_or_build((geometry, kernel, channels), build))
    }

    /// The column windows as the horizontal pass sums them, one set per
    /// chunk of `CHUNK_TAPS` taps.
    fn col_chunks(&self) -> &[Windows] {
        self.chunked_cols
            .as_deref()
            .unwrap_or(std::slice::from_ref(&self.cols))
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
    let col_chunks = plan.col_chunks();
    for (row_window, dst) in plan.rows.iter().zip(out.chunks_exact_mut(out_row_samples)) {
        vertical_in_chunks(
            vertical,
            &src[row_window.start * row_samples..row_window.end() * row_samples],
            row_samples,
            row_window.f32,
            &mut mid,
            &mut chunk_sums,
        );
        for (output, sample) in dst.iter_mut().enumerate() {
            let (col, channel) = (output / channels, output % channels);
            let sum = col_chunks.iter().fold(0f64, |sum, chunk| {
                let window = chunk.get(col);
                let taps = (window.start..window.end()).map(|x| mid[x * channels + channel]);
                sum + f64::from(fma_chain(taps, window.f32))
            });
            *sample = T::from_f32(sum as f32);
        }
    }
    out
}

/// Scratch of the chunked sums: one chunk's f32 sums and their f64 total.
#[derive(Default)]
pub(crate) struct ChunkSums {
    chunk: Vec<f32>,
    total: Vec<f64>,
}

impl ChunkSums {
    /// Adds in f64 the `out.len()` sums `sum_chunk(index, sums)` writes for
    /// each chunk index below `chunk_count`, rounded to f32 into `out`. A
    /// single chunk sums straight into `out`.
    fn sum_into(
        &mut self,
        chunk_count: usize,
        out: &mut [f32],
        mut sum_chunk: impl FnMut(usize, &mut [f32]),
    ) {
        if chunk_count == 1 {
            return sum_chunk(0, out);
        }
        self.chunk.resize(out.len(), 0.0);
        self.total.clear();
        self.total.resize(out.len(), 0.0);
        for index in 0..chunk_count {
            sum_chunk(index, &mut self.chunk[..out.len()]);
            for (total, &chunk_sum) in self.total.iter_mut().zip(&self.chunk) {
                *total += f64::from(chunk_sum);
            }
        }
        for (out, &total) in out.iter_mut().zip(&self.total) {
            *out = total as f32;
        }
    }
}

/// `vertical` (any tier's pass) over `weights` in chunks of `CHUNK_TAPS`
/// taps.
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

/// `sum_t weights[t] * samples[t]` as `acc = sample * weight + acc` from 0.
#[inline]
pub(crate) fn fma_chain(samples: impl Iterator<Item = f32>, weights: &[f32]) -> f32 {
    samples
        .zip(weights)
        .fold(0.0, |acc, (sample, &weight)| sample.mul_add(weight, acc))
}

/// A sample type `resize_float` takes, with its SIMD path.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) trait FloatResample: TransposedTier {}
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl<T: TransposedTier> FloatResample for T {}
/// A sample type `resize_float` takes.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub(crate) trait FloatResample: FloatSample {}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
impl<T: FloatSample> FloatResample for T {}

/// The SIMD path of a sample type (its kernels load and store that dtype).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) trait TransposedTier: FloatSample {
    fn resize_transposed(src: &[Self], width: usize, plan: &FloatPlan) -> Vec<Self>;
}

/// `TransposedTier` for `$t` by the `$tier` module's `$rows`-row kernels.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
macro_rules! transposed_tier {
    (
        $t:ty,
        $tier:ident,
        $rows:expr,
        $vertical:ident,
        $transpose:ident,
        $horizontal:ident,
        $store:ident
    ) => {
        impl TransposedTier for $t {
            fn resize_transposed(src: &[$t], width: usize, plan: &FloatPlan) -> Vec<$t> {
                // SAFETY: the tier's features are detected (avx2+fma) or
                // baseline (NEON on aarch64); the driver sizes every buffer.
                unsafe {
                    transposed::<$t, { $rows }>(
                        src,
                        width,
                        plan,
                        |rows, stride, weights, mid| $tier::$vertical(rows, stride, weights, mid),
                        $tier::$transpose,
                        $tier::$horizontal,
                        $tier::$store,
                    )
                }
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
transposed_tier!(
    u8,
    x86,
    8,
    vertical_u8_avx2,
    transpose_f32_rows_avx2,
    horizontal_f32_transposed_avx2,
    untranspose_store_u8_avx2
);
#[cfg(target_arch = "x86_64")]
transposed_tier!(
    u16,
    x86,
    8,
    vertical_u16_avx2,
    transpose_f32_rows_avx2,
    horizontal_f32_transposed_avx2,
    untranspose_store_u16_avx2
);
#[cfg(target_arch = "aarch64")]
transposed_tier!(
    u16,
    neon,
    neon::ROWS,
    vertical_u16_neon,
    transpose_f32_rows_neon,
    horizontal_f32_transposed_neon,
    untranspose_store_u16_neon
);
#[cfg(target_arch = "aarch64")]
transposed_tier!(
    u8,
    neon,
    neon::ROWS,
    vertical_u8_neon,
    transpose_f32_rows_neon,
    horizontal_f32_transposed_neon,
    untranspose_store_u8_neon
);

/// The SIMD driver for batches of `ROWS` rows: per batch the vertical passes,
/// a transpose (sample x of row r to `mid_t[x * ROWS + r]`), the horizontal
/// pass over all rows at once, and the transpose back with the dtype's store.
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
    let (out_width, out_height) = (plan.cols.len(), plan.rows.len());
    let (row_samples, out_row_samples) = (width * channels, out_width * channels);
    let sample_bytes = std::mem::size_of::<T>();
    // Whole ROWS-sample tiles for the transposes, plus one more tile: rows a
    // power of 2 apart would share L1 sets.
    let mid_stride = row_samples.next_multiple_of(ROWS) + ROWS;
    let out_t_samples = out_row_samples.next_multiple_of(ROWS);
    let mut out_buf = crate::cpu::unwritten_output::<T>(out_row_samples * out_height);
    // Kept per thread so repeated small resizes skip first-touch page faults;
    // a call writes everything it reads (row padding passes through unused).
    let TransposeScratch {
        mut mid,
        mut mid_t,
        mut out_t,
    } = SCRATCH.with_borrow_mut(std::mem::take);
    let (mut vertical_sums, mut horizontal_sums) = (ChunkSums::default(), ChunkSums::default());
    let col_chunks = plan.col_chunks();
    mid.resize(ROWS * mid_stride, 0.0);
    mid_t.resize(mid_stride * ROWS, 0.0);
    out_t.resize(out_t_samples * ROWS, 0.0);
    let out_ptr = out_buf.as_mut_ptr().cast::<T>();
    let batches = |lo: usize, hi: usize| {
        for batch in (lo..hi).step_by(ROWS) {
            let rows = ROWS.min(hi - batch);
            // A short batch repeats its last row; those outputs are dropped.
            let window_of = |row: usize| plan.rows.get((batch + row).min(hi - 1));
            let batch_source_rows = window_of(ROWS - 1).end() - window_of(0).start;
            let strip = vertical_strip_samples(
                window_of(0).f32.len(),
                batch_source_rows,
                row_samples,
                sample_bytes,
            );
            for strip_start in (0..row_samples).step_by(strip) {
                let strip_width = strip.min(row_samples - strip_start);
                for row in 0..ROWS {
                    let window = window_of(row);
                    let src_rows = &src[window.start * row_samples + strip_start
                        ..(window.end() - 1) * row_samples + strip_start + strip_width];
                    let mid_row = row * mid_stride + strip_start;
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
            horizontal_sums.sum_into(col_chunks.len(), &mut out_t, |chunk, sums| {
                // SAFETY: as above; every window (and chunk of one) ends by
                // the row's width < mid_stride, and `sums` holds ROWS x
                // out_t_samples, whole tiles of the output row.
                unsafe { horizontal(&mid_t, &col_chunks[chunk], channels, sums) };
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
            out_height,
            batches,
        );
    }
    // SAFETY: every batch stores all of its rows.
    let out = unsafe { crate::cpu::written_output(out_buf) };
    TransposeScratch { mid, mid_t, out_t }.keep_in(&SCRATCH);
    out
}

/// Samples per column strip of a batch's vertical pass: strips narrow enough
/// that the batch's `batch_source_rows` stay in L2 while all its rows read
/// them, where one output row's window alone overflows L2 (elsewhere the
/// strips cost more than the reuse saves); else the whole row.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline]
fn vertical_strip_samples(
    window_taps: usize,
    batch_source_rows: usize,
    row_samples: usize,
    sample_bytes: usize,
) -> usize {
    if window_taps * row_samples * sample_bytes < STRIPED_WINDOW_BYTES {
        return row_samples;
    }
    (BATCH_SOURCE_BYTES / (batch_source_rows * sample_bytes))
        .max(1)
        .next_multiple_of(STRIP_ALIGN_SAMPLES)
        .min(row_samples)
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
thread_local! {
    static SCRATCH: RefCell<TransposeScratch<f32>> = const { RefCell::new(TransposeScratch::new()) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::RGB;
    #[cfg(target_arch = "x86_64")]
    use crate::testing::avx2_fma_detected;
    use crate::testing::{CHANNELS, RandomSample, Rng, assert_same, kernel_name, random_samples};

    const KERNELS: [Kernel; 4] = [
        Kernel::Box,
        Kernel::Bilinear,
        Kernel::CatmullRom,
        Kernel::Lanczos3,
    ];

    /// `(width, out_width)`: the smallest images, ratios around the batch and
    /// register widths, upscales, and large reductions.
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

    /// `(height, out_height)`: batches of 1 to 16 rows, short last batches,
    /// upscales, and windows long enough for the grouped vertical taps.
    const ROW_CASES: [(usize, usize); 7] = [
        (1, 1),
        (3, 7),
        (17, 9),
        (33, 16),
        (40, 5),
        (40, 37),
        (130, 2),
    ];

    /// A sample type's dtype-specific AVX2 kernels.
    #[cfg(target_arch = "x86_64")]
    trait TierKernels: TransposedTier + RandomSample {
        const VERTICAL_AVX2: unsafe fn(&[Self], usize, &[f32], &mut [f32]);
    }

    #[cfg(target_arch = "x86_64")]
    impl TierKernels for u8 {
        const VERTICAL_AVX2: unsafe fn(&[u8], usize, &[f32], &mut [f32]) = x86::vertical_u8_avx2;
    }

    #[cfg(target_arch = "x86_64")]
    impl TierKernels for u16 {
        const VERTICAL_AVX2: unsafe fn(&[u16], usize, &[f32], &mut [f32]) = x86::vertical_u16_avx2;
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

    /// The entry point and each tier's driver equal the scalar path bit
    /// for bit (the same FMA chains in the same order).
    fn assert_tiers_match_scalar<T: TestFloat>(
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
    }

    fn assert_resize_matches_scalar<T: TestFloat>(seed: u64) {
        let mut rng = Rng::new(seed);
        for kernel in KERNELS {
            for channels in CHANNELS {
                for (width, out_width) in COLUMN_CASES {
                    for (height, out_height) in ROW_CASES {
                        let src: Vec<T> = random_samples(&mut rng, width * height * channels);
                        let plan = FloatPlan::new(
                            Windows::new(height, out_height, kernel),
                            Windows::new(width, out_width, kernel),
                            channels,
                        );
                        let context = format!(
                            "{} {channels} channels {width} x {height} -> {out_width} x {out_height}",
                            kernel_name(kernel)
                        );
                        assert_tiers_match_scalar(&src, (width, height), &plan, &context);
                    }
                }
            }
        }
    }

    /// Random uint16 samples over the whole range, so the filters' overshoot
    /// clamps at both ends.
    #[test]
    fn uint16_resize_equals_scalar() {
        assert_resize_matches_scalar::<u16>(2);
    }

    /// uint8 takes the engine for large reductions; any ratio must agree.
    #[test]
    fn uint8_resize_equals_scalar() {
        assert_resize_matches_scalar::<u8>(4);
    }

    /// Windows longer than `CHUNK_TAPS` sum in chunks, in both passes, the
    /// same on every tier: reductions whose windows take one, two and more
    /// chunks along either axis.
    fn assert_chunked_matches_scalar<T: TestFloat>(seed: u64) {
        let mut rng = Rng::new(seed);
        let cases = [
            ((1000, 3), (40, 3)),
            ((40, 3), (1000, 3)),
            ((2600, 2), (600, 1)),
            ((300, 1), (2600, 2)),
        ];
        for kernel in [Kernel::Box, Kernel::Lanczos3] {
            for channels in CHANNELS {
                for ((width, out_width), (height, out_height)) in cases {
                    let src: Vec<T> = random_samples(&mut rng, width * height * channels);
                    let plan = FloatPlan::new(
                        Windows::new(height, out_height, kernel),
                        Windows::new(width, out_width, kernel),
                        channels,
                    );
                    let context = format!(
                        "chunked {} {channels} channels {width} x {height} -> {out_width} x {out_height}",
                        kernel_name(kernel)
                    );
                    assert_tiers_match_scalar(&src, (width, height), &plan, &context);
                }
            }
        }
    }

    #[test]
    fn chunked_resize_equals_scalar() {
        assert_chunked_matches_scalar::<u8>(5);
        assert_chunked_matches_scalar::<u16>(6);
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
    fn assert_striped_matches_scalar<T: TestFloat>(width: usize, channels: usize) {
        let (height, out_height, out_width) = (1024, 4, width / 2 + 1);
        let plan = FloatPlan::new(
            Windows::new(height, out_height, Kernel::Lanczos3),
            Windows::new(width, out_width, Kernel::Lanczos3),
            channels,
        );
        let window_bytes = plan.rows.get(0).f32.len() * width * channels * size_of::<T>();
        assert!(window_bytes >= STRIPED_WINDOW_BYTES, "not striped");
        let mut rng = Rng::new(3);
        let src: Vec<T> = random_samples(&mut rng, width * height * channels);
        let context = format!("striped {width} x {height} x {channels}");
        assert_tiers_match_scalar(&src, (width, height), &plan, &context);
    }

    #[test]
    fn striped_vertical_pass_equals_scalar() {
        assert_striped_matches_scalar::<u8>(1026, 1);
        assert_striped_matches_scalar::<u16>(171, RGB);
    }

    /// A vertical kernel, whether this host runs it, and its tier's name.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    type VerticalTier<T> = (
        bool,
        unsafe fn(&[T], usize, &[f32], &mut [f32]),
        &'static str,
    );

    /// The vertical kernels of every tier: rows narrower than a register,
    /// widths around the registers and blocks, strides past the row (column
    /// strips), and the strides and window lengths where the taps run in
    /// groups.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn assert_vertical_matches_scalar<T: FloatSample + RandomSample>(
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
            // Weights in [-0.5, 1.5), exactly representable.
            let weights: Vec<f32> = random_samples::<u16>(rng, taps)
                .into_iter()
                .map(|weight| 2.0 * f32::from(weight) / 65536.0 - 0.5)
                .collect();
            for stride in [width, width + 13] {
                let rows: Vec<T> = random_samples(rng, (taps - 1) * stride + width);
                let mut expected = vec![0.0; width];
                vertical(&rows, stride, &weights, &mut expected);
                let context = format!("width {width}, stride {stride}, {taps} taps");
                for &(available, kernel, name) in tiers {
                    if available {
                        let mut mid = vec![f32::NAN; width];
                        // SAFETY: the kernel's features detected; rows holds
                        // taps (> 0) rows of width, stride apart.
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
    fn x86_vertical_tiers<T: TierKernels>() -> [VerticalTier<T>; 1] {
        [(avx2_fma_detected(), T::VERTICAL_AVX2, "avx2")]
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vertical_kernels_equal_scalar() {
        let mut rng = Rng::new(4);
        assert_vertical_matches_scalar::<u8>(&mut rng, &x86_vertical_tiers());
        assert_vertical_matches_scalar::<u16>(&mut rng, &x86_vertical_tiers());
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn vertical_kernels_equal_scalar() {
        let mut rng = Rng::new(4);
        let neon = crate::cpu::use_neon();
        assert_vertical_matches_scalar::<u8>(&mut rng, &[(neon, neon::vertical_u8_neon, "neon")]);
        assert_vertical_matches_scalar::<u16>(&mut rng, &[(neon, neon::vertical_u16_neon, "neon")]);
    }
}

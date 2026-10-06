//! AVX-512 and AVX2 kernels of the uint16/float32 convolution resize (see
//! `crate::convolve_float`): FMA chains in tap order, horizontal pass on
//! transposed batches of 16 (AVX-512) or 8 (AVX2) rows.
//!
//! # Safety
//! Every kernel requires the CPU features in its `#[target_feature]` (its
//! dispatch checks them at runtime). The register-only transposes are safe
//! `fn`s; the kernels that load and store are `unsafe fn`s whose `# Safety`
//! section gives the bounds the caller upholds, and each load and store is
//! justified against those bounds at its loop guard.

use std::ops::Range;

use crate::convolve::Windows;
use crate::convolve_float::FloatSample;
use crate::x86::{grouped_taps, groups_taps, prefetch_rows_after_window};

/// Registers (of 16 or 8 f32) per block of the vertical pass: independent
/// FMA chains that hide the FMA latency.
const REGISTERS_PER_BLOCK: usize = 4;

/// The kernels both tiers share, in the tier's register type `$vector` of
/// `$lanes` f32 (= rows per transposed batch): the vertical pass, the
/// transposes into and out of a batch, and the transposed horizontal pass.
macro_rules! float_tier {
    (
        feature: $feature:literal,
        avx512_tier: $avx512_tier:literal,
        lanes: $lanes:literal,
        vector: $vector:ident,
        zero: $zero:ident,
        load: $load:ident,
        store: $store:ident,
        set1: $set1:ident,
        fmadd: $fmadd:ident,
        transpose_tile: $transpose_tile:ident,
        vertical_with_load: $vertical_with_load:ident,
        vertical_f32: $vertical_f32:ident,
        transpose_rows: $transpose_rows:ident,
        horizontal: $horizontal:ident,
        untranspose: $untranspose:ident $(,)?
    ) => {
        /// The vertical pass, `REGISTERS_PER_BLOCK` FMA chains per block,
        /// with a `load(row pointer)` of one register of samples as f32; the
        /// taps run in groups where `groups_taps` says so. `load` is only
        /// given the start of a register of samples within `rows`.
        ///
        /// # Safety
        /// `weights` is not empty and `rows` holds `weights.len()` rows of
        /// `mid.len()` samples `stride` apart (`rows.len() >=
        /// (weights.len() - 1) * stride + mid.len()`).
        #[target_feature(enable = $feature)]
        #[inline]
        unsafe fn $vertical_with_load<T: FloatSample>(
            rows: &[T],
            stride: usize,
            weights: &[f32],
            mid: &mut [f32],
            load: impl Fn(*const T) -> std::arch::x86_64::$vector,
        ) {
            use std::arch::x86_64::*;
            const LANES: usize = $lanes;
            let width = mid.len();
            debug_assert!(rows.len() >= (weights.len() - 1) * stride + width);
            if width < LANES {
                return crate::convolve_float::vertical(rows, stride, weights, mid);
            }
            let broadcast: Vec<$vector> = weights.iter().map(|&weight| $set1(weight)).collect();
            let (src_ptr, mid_ptr) = (rows.as_ptr(), mid.as_mut_ptr());
            // Accumulates taps `taps` into `registers` registers at x; a later tap
            // group resumes from the sums stored in mid.
            let block = |x: usize, registers: usize, taps: Range<usize>| {
                // The next window's rows ahead of the pass, for uint16 only:
                // float32 blocks span twice the lines, and with few taps
                // their prefetches outweigh the misses they hide.
                if taps.start == 0 && std::mem::size_of::<T>() == 2 {
                    let (sample, block_bytes) = (
                        std::mem::size_of::<T>(),
                        registers * LANES * std::mem::size_of::<T>(),
                    );
                    for line in (0..block_bytes).step_by(64) {
                        prefetch_rows_after_window(
                            src_ptr as *const u8,
                            weights.len(),
                            stride * sample,
                            x * sample + line,
                        );
                    }
                }
                let mut acc = [$zero(); REGISTERS_PER_BLOCK];
                if taps.start > 0 {
                    for (register, acc) in acc.iter_mut().enumerate().take(registers) {
                        // SAFETY: every block ends by the row's end (x +
                        // registers * LANES <= width = mid.len()).
                        *acc = unsafe { $load(mid_ptr.add(x + register * LANES)) };
                    }
                }
                for tap in taps {
                    for (register, acc) in acc.iter_mut().enumerate().take(registers) {
                        let at = tap * stride + x + register * LANES;
                        // SAFETY: tap < weights.len() and the block ends by
                        // `width`: the register lies in the tap's row of
                        // `rows` (contract).
                        let register_ptr = unsafe { src_ptr.add(at) };
                        *acc = $fmadd(load(register_ptr), broadcast[tap], *acc);
                    }
                }
                for (register, acc) in acc.iter().enumerate().take(registers) {
                    // SAFETY: x + registers * LANES <= width = mid.len().
                    unsafe { $store(mid_ptr.add(x + register * LANES), *acc) };
                }
            };
            let all = 0..weights.len();
            let block_len = REGISTERS_PER_BLOCK * LANES;
            let blocks = 0..width / block_len * block_len;
            if groups_taps(
                stride * std::mem::size_of::<T>(),
                weights.len(),
                $avx512_tier,
            ) {
                let sample_bytes = std::mem::size_of::<f32>();
                grouped_taps(
                    blocks.clone(),
                    block_len,
                    sample_bytes,
                    weights.len(),
                    |x, taps| block(x, REGISTERS_PER_BLOCK, taps),
                );
            } else {
                blocks
                    .clone()
                    .step_by(block_len)
                    .for_each(|x| block(x, REGISTERS_PER_BLOCK, all.clone()));
            }
            let mut x = blocks.end;
            while x + LANES <= width {
                block(x, 1, all.clone());
                x += LANES;
            }
            if x < width {
                block(width - LANES, 1, all);
            }
        }

        /// The vertical pass of float32 rows.
        ///
        /// # Safety
        /// `weights` is not empty and `rows.len() >= (weights.len() - 1) *
        /// stride + mid.len()`.
        #[target_feature(enable = $feature)]
        pub(crate) unsafe fn $vertical_f32(
            rows: &[f32],
            stride: usize,
            weights: &[f32],
            mid: &mut [f32],
        ) {
            use std::arch::x86_64::*;
            // SAFETY: the bounds are the caller's (contract); the load is
            // only given the start of a register within `rows`.
            unsafe { $vertical_with_load(rows, stride, weights, mid, |ptr| $load(ptr)) };
        }

        /// A batch of rows of `mid` (stride `stride`, a multiple of the
        /// batch) into `mid_t[x * LANES + r]`.
        ///
        /// # Safety
        /// `stride` is a multiple of the batch's `LANES` rows, and `mid` and
        /// `mid_t` hold `LANES * stride` samples each.
        #[target_feature(enable = $feature)]
        pub(crate) unsafe fn $transpose_rows(mid: &[f32], stride: usize, mid_t: &mut [f32]) {
            use std::arch::x86_64::*;
            const LANES: usize = $lanes;
            debug_assert!(
                stride.is_multiple_of(LANES)
                    && mid.len() >= LANES * stride
                    && mid_t.len() >= LANES * stride
            );
            for x0 in (0..stride).step_by(LANES) {
                let mut tile = [$zero(); LANES];
                for (row, register) in tile.iter_mut().enumerate() {
                    // SAFETY: row < LANES and x0 + LANES <= stride: within
                    // the LANES * stride samples of `mid` (contract).
                    *register = unsafe { $load(mid.as_ptr().add(row * stride + x0)) };
                }
                $transpose_tile(&mut tile);
                for (column, register) in tile.iter().enumerate() {
                    // SAFETY: x0 + column < stride: within the stride *
                    // LANES samples of `mid_t` (contract).
                    unsafe { $store(mid_t.as_mut_ptr().add((x0 + column) * LANES), *register) };
                }
            }
        }

        /// The horizontal pass of a transposed batch: per output sample, an
        /// FMA chain over its taps (each a unit-stride load of the batch's
        /// rows).
        ///
        /// # Safety
        /// For every output o, `col.end() * channels * LANES <= mid_t.len()`
        /// and `(o + 1) * channels * LANES <= out_t.len()`.
        #[target_feature(enable = $feature)]
        pub(crate) unsafe fn $horizontal(
            mid_t: &[f32],
            cols: &Windows,
            channels: usize,
            out_t: &mut [f32],
        ) {
            use std::arch::x86_64::*;
            const LANES: usize = $lanes;
            for (output, col) in cols.iter().enumerate() {
                debug_assert!(
                    col.end() * channels * LANES <= mid_t.len()
                        && (output + 1) * channels * LANES <= out_t.len()
                );
                for channel in 0..channels {
                    let mut acc = $zero();
                    for (tap, &weight) in col.f32.iter().enumerate() {
                        // SAFETY: start + tap < col.end() and channel <
                        // channels: the LANES samples lie in `mid_t`
                        // (contract).
                        let samples = unsafe {
                            $load(
                                mid_t
                                    .as_ptr()
                                    .add(((col.start + tap) * channels + channel) * LANES),
                            )
                        };
                        acc = $fmadd(samples, $set1(weight), acc);
                    }
                    // SAFETY: channel < channels: the LANES sums lie in
                    // `out_t` (contract).
                    unsafe {
                        $store(
                            out_t
                                .as_mut_ptr()
                                .add((output * channels + channel) * LANES),
                            acc,
                        );
                    }
                }
            }
        }

        /// `out_t[o * LANES + r]` back to rows `dst[r * stride + o]` (o <
        /// `ows`, r < `rows`) through `store(row, o, LANES samples, count)`.
        ///
        /// # Safety
        /// `out_t` holds `ows.next_multiple_of(LANES) * LANES` samples.
        #[target_feature(enable = $feature)]
        #[inline]
        unsafe fn $untranspose(
            out_t: &[f32],
            ows: usize,
            rows: usize,
            mut store: impl FnMut(usize, usize, std::arch::x86_64::$vector, usize),
        ) {
            use std::arch::x86_64::*;
            const LANES: usize = $lanes;
            for o0 in (0..ows).step_by(LANES) {
                debug_assert!((o0 + LANES) * LANES <= out_t.len());
                let mut tile = [$zero(); LANES];
                for (output, register) in tile.iter_mut().enumerate() {
                    // SAFETY: o0 < ows, a multiple of LANES: outputs o0..o0 +
                    // LANES lie in `out_t` (contract).
                    *register = unsafe { $load(out_t.as_ptr().add((o0 + output) * LANES)) };
                }
                $transpose_tile(&mut tile);
                for (row, register) in tile.iter().enumerate().take(rows) {
                    store(row, o0, *register, LANES.min(ows - o0));
                }
            }
        }
    };
}

/// A store mask of the low `lanes` (1..=16) lanes.
fn low_lanes_mask16(lanes: usize) -> u16 {
    debug_assert!((1..=16).contains(&lanes));
    u16::MAX >> (16 - lanes)
}

/// The first `count` samples of `staged` to `at`: the partial store of a
/// row's last register.
///
/// # Safety
/// `count <= staged.len()`, and `at` is valid for `count` writes of `T`
/// that do not overlap `staged`.
unsafe fn store_prefix<T: Copy>(staged: &[T], at: *mut T, count: usize) {
    // SAFETY: the caller's bounds (contract).
    unsafe { std::ptr::copy_nonoverlapping(staged.as_ptr(), at, count) };
}

// ---- AVX-512: 16 rows per batch ----------------------------------------------

float_tier! {
    feature: "avx512f",
    avx512_tier: true,
    lanes: 16,
    vector: __m512,
    zero: _mm512_setzero_ps,
    load: _mm512_loadu_ps,
    store: _mm512_storeu_ps,
    set1: _mm512_set1_ps,
    fmadd: _mm512_fmadd_ps,
    transpose_tile: transpose_16x16_f32,
    vertical_with_load: vertical_with_load_avx512,
    vertical_f32: vertical_f32_avx512,
    transpose_rows: transpose_f32_rows_avx512,
    horizontal: horizontal_f32_transposed_avx512,
    untranspose: untranspose_avx512,
}

/// Transpose a 16x16 tile of f32: `rows[i]` = row i on entry, column i on
/// exit.
#[target_feature(enable = "avx512f")]
#[inline]
fn transpose_16x16_f32(rows: &mut [std::arch::x86_64::__m512; 16]) {
    use std::arch::x86_64::*;
    let mut pairs = [_mm512_setzero_ps(); 16];
    for pair in 0..8 {
        pairs[2 * pair] = _mm512_unpacklo_ps(rows[2 * pair], rows[2 * pair + 1]);
        pairs[2 * pair + 1] = _mm512_unpackhi_ps(rows[2 * pair], rows[2 * pair + 1]);
    }
    // quads[4 group + column], per 128-bit lane L: column 4L + column of rows
    // 4 group..4 group + 4.
    let mut quads = [_mm512_setzero_ps(); 16];
    for group in 0..4 {
        quads[4 * group] = _mm512_shuffle_ps::<0x44>(pairs[4 * group], pairs[4 * group + 2]);
        quads[4 * group + 1] = _mm512_shuffle_ps::<0xEE>(pairs[4 * group], pairs[4 * group + 2]);
        quads[4 * group + 2] =
            _mm512_shuffle_ps::<0x44>(pairs[4 * group + 1], pairs[4 * group + 3]);
        quads[4 * group + 3] =
            _mm512_shuffle_ps::<0xEE>(pairs[4 * group + 1], pairs[4 * group + 3]);
    }
    // Per column: a 4x4 transpose of 128-bit lanes across the four row groups.
    for column in 0..4 {
        let b0 = _mm512_shuffle_f32x4::<0x44>(quads[column], quads[4 + column]);
        let b1 = _mm512_shuffle_f32x4::<0xEE>(quads[column], quads[4 + column]);
        let b2 = _mm512_shuffle_f32x4::<0x44>(quads[8 + column], quads[12 + column]);
        let b3 = _mm512_shuffle_f32x4::<0xEE>(quads[8 + column], quads[12 + column]);
        rows[column] = _mm512_shuffle_f32x4::<0x88>(b0, b2);
        rows[4 + column] = _mm512_shuffle_f32x4::<0xDD>(b0, b2);
        rows[8 + column] = _mm512_shuffle_f32x4::<0x88>(b1, b3);
        rows[12 + column] = _mm512_shuffle_f32x4::<0xDD>(b1, b3);
    }
}

/// Untranspose a batch to float32 rows.
///
/// # Safety
/// `rows <= 16`, `out_t` holds `ows.next_multiple_of(16) * 16` samples, and
/// `dst` is valid for writes of `rows` rows of `ows` samples, `stride` apart.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn untranspose_store_f32_avx512(
    out_t: &[f32],
    ows: usize,
    dst: *mut f32,
    stride: usize,
    rows: usize,
) {
    use std::arch::x86_64::*;
    let store = |row: usize, output: usize, register: __m512, count: usize| {
        // SAFETY: row < rows, and the mask keeps the `count` samples
        // from `output` that lie in the row (contract).
        unsafe {
            _mm512_mask_storeu_ps(
                dst.add(row * stride + output),
                low_lanes_mask16(count),
                register,
            );
        }
    };
    // SAFETY: out_t's size is the caller's (contract), and `store` writes
    // only within the rows of `dst` (contract).
    unsafe { untranspose_avx512(out_t, ows, rows, store) };
}

/// Untranspose a batch to uint16 rows.
///
/// # Safety
/// `untranspose_store_f32_avx512`'s bounds.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(crate) unsafe fn untranspose_store_u16_avx512(
    out_t: &[f32],
    ows: usize,
    dst: *mut u16,
    stride: usize,
    rows: usize,
) {
    use std::arch::x86_64::*;
    let zero = _mm512_setzero_si512();
    let store = |row: usize, output: usize, register: __m512, count: usize| {
        // Round to nearest even (the default MXCSR mode), clamp to 0..65535.
        let samples = _mm512_cvtusepi32_epi16(_mm512_max_epi32(_mm512_cvtps_epi32(register), zero));
        // SAFETY: row < rows, and the mask keeps the `count` samples
        // from `output` that lie in the row (contract).
        unsafe {
            _mm256_mask_storeu_epi16(
                dst.add(row * stride + output) as *mut i16,
                low_lanes_mask16(count),
                samples,
            );
        }
    };
    // SAFETY: out_t's size is the caller's (contract), and `store` writes
    // only within the rows of `dst` (contract).
    unsafe { untranspose_avx512(out_t, ows, rows, store) };
}

/// The vertical pass of uint16 rows.
///
/// # Safety
/// `weights` is not empty and `rows.len() >= (weights.len() - 1) * stride +
/// mid.len()`.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn vertical_u16_avx512(
    rows: &[u16],
    stride: usize,
    weights: &[f32],
    mid: &mut [f32],
) {
    use std::arch::x86_64::*;
    // SAFETY: the bounds are the caller's (contract); the load is only given
    // the start of a register within `rows`.
    unsafe {
        vertical_with_load_avx512(rows, stride, weights, mid, |ptr| {
            _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
                ptr as *const __m256i,
            )))
        });
    }
}

// ---- AVX2: 8 rows per batch --------------------------------------------------

float_tier! {
    feature: "avx2,fma",
    avx512_tier: false,
    lanes: 8,
    vector: __m256,
    zero: _mm256_setzero_ps,
    load: _mm256_loadu_ps,
    store: _mm256_storeu_ps,
    set1: _mm256_set1_ps,
    fmadd: _mm256_fmadd_ps,
    transpose_tile: transpose_8x8_f32,
    vertical_with_load: vertical_with_load_avx2,
    vertical_f32: vertical_f32_avx2,
    transpose_rows: transpose_f32_rows_avx2,
    horizontal: horizontal_f32_transposed_avx2,
    untranspose: untranspose_avx2,
}

/// Transpose an 8x8 tile of f32: `rows[i]` = row i on entry, column i on
/// exit.
#[target_feature(enable = "avx2,fma")]
#[inline]
fn transpose_8x8_f32(rows: &mut [std::arch::x86_64::__m256; 8]) {
    use std::arch::x86_64::*;
    let mut pairs = [_mm256_setzero_ps(); 8];
    for pair in 0..4 {
        pairs[2 * pair] = _mm256_unpacklo_ps(rows[2 * pair], rows[2 * pair + 1]);
        pairs[2 * pair + 1] = _mm256_unpackhi_ps(rows[2 * pair], rows[2 * pair + 1]);
    }
    // quads[4 group + column], per 128-bit lane L: column 4L + column of rows
    // 4 group..4 group + 4.
    let mut quads = [_mm256_setzero_ps(); 8];
    for group in 0..2 {
        quads[4 * group] = _mm256_shuffle_ps::<0x44>(pairs[4 * group], pairs[4 * group + 2]);
        quads[4 * group + 1] = _mm256_shuffle_ps::<0xEE>(pairs[4 * group], pairs[4 * group + 2]);
        quads[4 * group + 2] =
            _mm256_shuffle_ps::<0x44>(pairs[4 * group + 1], pairs[4 * group + 3]);
        quads[4 * group + 3] =
            _mm256_shuffle_ps::<0xEE>(pairs[4 * group + 1], pairs[4 * group + 3]);
    }
    // Per column: rows 0..4's lane L paired with rows 4..8's (low lanes, then
    // high).
    for column in 0..4 {
        rows[column] = _mm256_permute2f128_ps::<0x20>(quads[column], quads[4 + column]);
        rows[4 + column] = _mm256_permute2f128_ps::<0x31>(quads[column], quads[4 + column]);
    }
}

/// Untranspose a batch to float32 rows.
///
/// # Safety
/// `rows <= 8`, `out_t` holds `ows.next_multiple_of(8) * 8` samples, and
/// `dst` is valid for writes of `rows` rows of `ows` samples, `stride` apart.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn untranspose_store_f32_avx2(
    out_t: &[f32],
    ows: usize,
    dst: *mut f32,
    stride: usize,
    rows: usize,
) {
    use std::arch::x86_64::*;
    let store = |row: usize, output: usize, register: __m256, count: usize| {
        // SAFETY: row < rows, and the `count` samples from `output` lie
        // in the row (contract); `staged` is the 8 samples stored.
        unsafe {
            let at = dst.add(row * stride + output);
            if count == 8 {
                _mm256_storeu_ps(at, register);
            } else {
                let mut staged = [0f32; 8];
                _mm256_storeu_ps(staged.as_mut_ptr(), register);
                store_prefix(&staged, at, count);
            }
        }
    };
    // SAFETY: out_t's size is the caller's (contract), and `store` writes
    // only within the rows of `dst` (contract).
    unsafe { untranspose_avx2(out_t, ows, rows, store) };
}

/// Untranspose a batch to uint16 rows.
///
/// # Safety
/// `untranspose_store_f32_avx2`'s bounds.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn untranspose_store_u16_avx2(
    out_t: &[f32],
    ows: usize,
    dst: *mut u16,
    stride: usize,
    rows: usize,
) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let store = |row: usize, output: usize, register: __m256, count: usize| {
        // Round to nearest even (the default MXCSR mode), clamp to 0..65535.
        let ints = _mm256_max_epi32(_mm256_cvtps_epi32(register), zero);
        let samples = _mm256_castsi256_si128(_mm256_permute4x64_epi64::<0b10_00>(
            _mm256_packus_epi32(ints, ints),
        ));
        // SAFETY: row < rows, and the `count` samples from `output` lie
        // in the row (contract); `staged` is the 8 samples stored.
        unsafe {
            let at = dst.add(row * stride + output);
            if count == 8 {
                _mm_storeu_si128(at as *mut __m128i, samples);
            } else {
                let mut staged = [0u16; 8];
                _mm_storeu_si128(staged.as_mut_ptr() as *mut __m128i, samples);
                store_prefix(&staged, at, count);
            }
        }
    };
    // SAFETY: out_t's size is the caller's (contract), and `store` writes
    // only within the rows of `dst` (contract).
    unsafe { untranspose_avx2(out_t, ows, rows, store) };
}

/// The vertical pass of uint16 rows.
///
/// # Safety
/// `weights` is not empty and `rows.len() >= (weights.len() - 1) * stride +
/// mid.len()`.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vertical_u16_avx2(
    rows: &[u16],
    stride: usize,
    weights: &[f32],
    mid: &mut [f32],
) {
    use std::arch::x86_64::*;
    // SAFETY: the bounds are the caller's (contract); the load is only given
    // the start of a register within `rows`.
    unsafe {
        vertical_with_load_avx2(rows, stride, weights, mid, |ptr| {
            _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(
                ptr as *const __m128i,
            )))
        });
    }
}

/// Untranspose a batch to uint8 rows.
///
/// # Safety
/// `untranspose_store_f32_avx512`'s bounds.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(crate) unsafe fn untranspose_store_u8_avx512(
    out_t: &[f32],
    ows: usize,
    dst: *mut u8,
    stride: usize,
    rows: usize,
) {
    use std::arch::x86_64::*;
    let zero = _mm512_setzero_si512();
    let store = |row: usize, output: usize, register: __m512, count: usize| {
        // Round to nearest even (the default MXCSR mode), clamp to 0..255.
        let samples = _mm512_cvtusepi32_epi8(_mm512_max_epi32(_mm512_cvtps_epi32(register), zero));
        // SAFETY: row < rows, and the mask keeps the `count` samples
        // from `output` that lie in the row (contract).
        unsafe {
            _mm_mask_storeu_epi8(
                dst.add(row * stride + output) as *mut i8,
                low_lanes_mask16(count),
                samples,
            );
        }
    };
    // SAFETY: out_t's size is the caller's (contract), and `store` writes
    // only within the rows of `dst` (contract).
    unsafe { untranspose_avx512(out_t, ows, rows, store) };
}

/// The vertical pass of uint8 rows.
///
/// # Safety
/// `weights` is not empty and `rows.len() >= (weights.len() - 1) * stride +
/// mid.len()`.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn vertical_u8_avx512(
    rows: &[u8],
    stride: usize,
    weights: &[f32],
    mid: &mut [f32],
) {
    use std::arch::x86_64::*;
    // SAFETY: the bounds are the caller's (contract); the load is only given
    // the start of a register (16 samples) within `rows`.
    unsafe {
        vertical_with_load_avx512(rows, stride, weights, mid, |ptr| {
            _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(ptr as *const __m128i)))
        });
    }
}

/// Untranspose a batch to uint8 rows.
///
/// # Safety
/// `untranspose_store_f32_avx2`'s bounds.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn untranspose_store_u8_avx2(
    out_t: &[f32],
    ows: usize,
    dst: *mut u8,
    stride: usize,
    rows: usize,
) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let store = |row: usize, output: usize, register: __m256, count: usize| {
        // Round to nearest even (the default MXCSR mode), clamp to 0..255:
        // the 8 bytes end up in dwords 0 (lane 0) and 4 (lane 1).
        let ints = _mm256_max_epi32(_mm256_cvtps_epi32(register), zero);
        let words = _mm256_packus_epi32(ints, ints);
        let bytes = _mm256_packus_epi16(words, words);
        let samples = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(
            bytes,
            _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4),
        ));
        // SAFETY: row < rows, and the `count` samples from `output` lie
        // in the row (contract); `staged` is the 8 samples stored.
        unsafe {
            let at = dst.add(row * stride + output);
            if count == 8 {
                _mm_storel_epi64(at as *mut __m128i, samples);
            } else {
                let mut staged = [0u8; 16];
                _mm_storeu_si128(staged.as_mut_ptr() as *mut __m128i, samples);
                store_prefix(&staged, at, count);
            }
        }
    };
    // SAFETY: out_t's size is the caller's (contract), and `store` writes
    // only within the rows of `dst` (contract).
    unsafe { untranspose_avx2(out_t, ows, rows, store) };
}

/// The vertical pass of uint8 rows.
///
/// # Safety
/// `weights` is not empty and `rows.len() >= (weights.len() - 1) * stride +
/// mid.len()`.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn vertical_u8_avx2(
    rows: &[u8],
    stride: usize,
    weights: &[f32],
    mid: &mut [f32],
) {
    use std::arch::x86_64::*;
    // SAFETY: the bounds are the caller's (contract); the load is only given
    // the start of a register (8 samples) within `rows`.
    unsafe {
        vertical_with_load_avx2(rows, stride, weights, mid, |ptr| {
            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(ptr as *const __m128i)))
        });
    }
}

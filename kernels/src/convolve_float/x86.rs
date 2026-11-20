//! AVX2 kernels of the float32 convolution engine for uint8 and uint16 (see
//! `crate::convolve_float`): FMA chains in tap order, horizontal pass on
//! transposed batches of 8 rows.
//!
//! # Safety
//! Every kernel requires the CPU features in its `#[target_feature]`, which
//! its dispatch checks at runtime. The kernels that load and store are
//! `unsafe fn`s whose `# Safety` section gives the caller's bounds.

use std::ops::Range;

use crate::convolve::Windows;
use crate::convolve_float::FloatSample;
use crate::x86::{grouped_taps, groups_taps, prefetch_rows_after_window};

/// Registers (of 8 f32) per block of the vertical pass: independent FMA
/// chains that hide the FMA latency.
const REGISTERS_PER_BLOCK: usize = 4;
/// Bytes per cache line.
const CACHE_LINE_BYTES: usize = 64;

/// The vertical pass, `REGISTERS_PER_BLOCK` FMA chains per block, `load`
/// widening the register of samples at a pointer into `rows` to f32; the taps
/// run in groups where `groups_taps` says so.
///
/// # Safety
/// `weights` is not empty and `rows` holds `weights.len()` rows of
/// `mid.len()` samples `stride` apart (`rows.len() >=
/// (weights.len() - 1) * stride + mid.len()`).
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn vertical_with_load_avx2<T: FloatSample>(
    rows: &[T],
    stride: usize,
    weights: &[f32],
    mid: &mut [f32],
    load: impl Fn(*const T) -> std::arch::x86_64::__m256,
) {
    use std::arch::x86_64::*;
    const LANES: usize = 8;
    let width = mid.len();
    debug_assert!(rows.len() >= (weights.len() - 1) * stride + width);
    if width < LANES {
        return crate::convolve_float::vertical(rows, stride, weights, mid);
    }
    let broadcast: Vec<__m256> = weights
        .iter()
        .map(|&weight| _mm256_set1_ps(weight))
        .collect();
    let (src_ptr, mid_ptr) = (rows.as_ptr(), mid.as_mut_ptr());
    let sample_bytes = std::mem::size_of::<T>();
    // A tap group after the first resumes from the sums stored in mid.
    let accumulate_block = |x: usize, registers: usize, taps: Range<usize>| {
        let prefetches_next_window = taps.start == 0 && sample_bytes == 2;
        if prefetches_next_window {
            let block_bytes = registers * LANES * sample_bytes;
            for line in (0..block_bytes).step_by(CACHE_LINE_BYTES) {
                prefetch_rows_after_window(
                    src_ptr as *const u8,
                    weights.len(),
                    stride * sample_bytes,
                    x * sample_bytes + line,
                );
            }
        }
        let mut acc = [_mm256_setzero_ps(); REGISTERS_PER_BLOCK];
        if taps.start > 0 {
            for (register, acc) in acc.iter_mut().enumerate().take(registers) {
                // SAFETY: every block ends by the row's end (x +
                // registers * LANES <= width = mid.len()).
                *acc = unsafe { _mm256_loadu_ps(mid_ptr.add(x + register * LANES)) };
            }
        }
        for tap in taps {
            for (register, acc) in acc.iter_mut().enumerate().take(registers) {
                let at = tap * stride + x + register * LANES;
                // SAFETY: tap < weights.len() and the block ends by
                // `width`: the register lies in the tap's row of
                // `rows` (contract).
                let register_ptr = unsafe { src_ptr.add(at) };
                *acc = _mm256_fmadd_ps(load(register_ptr), broadcast[tap], *acc);
            }
        }
        for (register, acc) in acc.iter().enumerate().take(registers) {
            // SAFETY: x + registers * LANES <= width = mid.len().
            unsafe { _mm256_storeu_ps(mid_ptr.add(x + register * LANES), *acc) };
        }
    };
    let all_taps = 0..weights.len();
    let block_len = REGISTERS_PER_BLOCK * LANES;
    let blocks = 0..width / block_len * block_len;
    if groups_taps(stride * sample_bytes, weights.len()) {
        grouped_taps(
            blocks.clone(),
            block_len,
            std::mem::size_of::<f32>(),
            weights.len(),
            |x, taps| accumulate_block(x, REGISTERS_PER_BLOCK, taps),
        );
    } else {
        blocks
            .clone()
            .step_by(block_len)
            .for_each(|x| accumulate_block(x, REGISTERS_PER_BLOCK, all_taps.clone()));
    }
    let mut x = blocks.end;
    while x + LANES <= width {
        accumulate_block(x, 1, all_taps.clone());
        x += LANES;
    }
    if x < width {
        accumulate_block(width - LANES, 1, all_taps);
    }
}

/// The batch's rows of `mid`, `stride` apart, into `mid_t[x * LANES + r]`.
///
/// # Safety
/// `stride` is a multiple of the batch's `LANES` rows, and `mid` and
/// `mid_t` hold `LANES * stride` samples each.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn transpose_f32_rows_avx2(mid: &[f32], stride: usize, mid_t: &mut [f32]) {
    use std::arch::x86_64::*;
    const LANES: usize = 8;
    debug_assert!(
        stride.is_multiple_of(LANES)
            && mid.len() >= LANES * stride
            && mid_t.len() >= LANES * stride
    );
    for tile_x in (0..stride).step_by(LANES) {
        let mut tile = [_mm256_setzero_ps(); LANES];
        for (row, register) in tile.iter_mut().enumerate() {
            // SAFETY: row < LANES and tile_x + LANES <= stride: within the
            // LANES * stride samples of `mid` (contract).
            *register = unsafe { _mm256_loadu_ps(mid.as_ptr().add(row * stride + tile_x)) };
        }
        transpose_8x8_f32(&mut tile);
        for (column, register) in tile.iter().enumerate() {
            let at = (tile_x + column) * LANES;
            // SAFETY: tile_x + column < stride: within the stride * LANES
            // samples of `mid_t` (contract).
            unsafe { _mm256_storeu_ps(mid_t.as_mut_ptr().add(at), *register) };
        }
    }
}

/// The horizontal pass of a transposed batch: per output sample, an FMA chain
/// over its taps, each a unit-stride load of the batch's rows.
///
/// # Safety
/// For every output o, `col.end() * channels * LANES <= mid_t.len()` and
/// `(o + 1) * channels * LANES <= out_t.len()`.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn horizontal_f32_transposed_avx2(
    mid_t: &[f32],
    cols: &Windows,
    channels: usize,
    out_t: &mut [f32],
) {
    use std::arch::x86_64::*;
    const LANES: usize = 8;
    for (output, col) in cols.iter().enumerate() {
        debug_assert!(
            col.end() * channels * LANES <= mid_t.len()
                && (output + 1) * channels * LANES <= out_t.len()
        );
        for channel in 0..channels {
            let mut acc = _mm256_setzero_ps();
            for (tap, &weight) in col.f32.iter().enumerate() {
                // SAFETY: start + tap < col.end() and channel < channels: the
                // LANES samples lie in `mid_t` (contract).
                let samples = unsafe {
                    _mm256_loadu_ps(
                        mid_t
                            .as_ptr()
                            .add(((col.start + tap) * channels + channel) * LANES),
                    )
                };
                acc = _mm256_fmadd_ps(samples, _mm256_set1_ps(weight), acc);
            }
            // SAFETY: channel < channels: the LANES sums lie in `out_t`
            // (contract).
            unsafe {
                _mm256_storeu_ps(
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
/// `out_row_samples`, r < `rows`) through `store(row, o, LANES samples,
/// count)`.
///
/// # Safety
/// `out_t` holds `out_row_samples.next_multiple_of(LANES) * LANES` samples.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn untranspose_avx2(
    out_t: &[f32],
    out_row_samples: usize,
    rows: usize,
    mut store: impl FnMut(usize, usize, std::arch::x86_64::__m256, usize),
) {
    use std::arch::x86_64::*;
    const LANES: usize = 8;
    for tile_output in (0..out_row_samples).step_by(LANES) {
        debug_assert!((tile_output + LANES) * LANES <= out_t.len());
        let mut tile = [_mm256_setzero_ps(); LANES];
        for (output, register) in tile.iter_mut().enumerate() {
            let at = (tile_output + output) * LANES;
            // SAFETY: tile_output < out_row_samples, a multiple of LANES: its
            // LANES outputs lie in `out_t` (contract).
            *register = unsafe { _mm256_loadu_ps(out_t.as_ptr().add(at)) };
        }
        transpose_8x8_f32(&mut tile);
        let count = LANES.min(out_row_samples - tile_output);
        for (row, register) in tile.iter().enumerate().take(rows) {
            store(row, tile_output, *register, count);
        }
    }
}

/// The partial store of a row's last register.
///
/// # Safety
/// `count <= staged.len()`, and `at` is valid for `count` writes of `T`
/// that do not overlap `staged`.
unsafe fn store_prefix<T: Copy>(staged: &[T], at: *mut T, count: usize) {
    // SAFETY: the caller's bounds (contract).
    unsafe { std::ptr::copy_nonoverlapping(staged.as_ptr(), at, count) };
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
    // quads[4 * group + column], per 128-bit lane L: column 4L + column of
    // rows 4 * group..4 * group + 4.
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

/// Untranspose a batch to uint16 rows.
///
/// # Safety
/// `rows <= 8`, `out_t` holds `out_row_samples.next_multiple_of(8) * 8`
/// samples, and `dst` is valid for writes of `rows` rows of `out_row_samples`
/// samples, `stride` apart.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn untranspose_store_u16_avx2(
    out_t: &[f32],
    out_row_samples: usize,
    dst: *mut u16,
    stride: usize,
    rows: usize,
) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let store = |row: usize, output: usize, register: __m256, count: usize| {
        // Rounds to nearest even (the default MXCSR mode); saturates.
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
    unsafe { untranspose_avx2(out_t, out_row_samples, rows, store) };
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
/// `untranspose_store_u16_avx2`'s bounds.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn untranspose_store_u8_avx2(
    out_t: &[f32],
    out_row_samples: usize,
    dst: *mut u8,
    stride: usize,
    rows: usize,
) {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let store = |row: usize, output: usize, register: __m256, count: usize| {
        // Rounds to nearest even (the default MXCSR mode); saturates. The 8
        // bytes end up in dwords 0 (lane 0) and 4 (lane 1).
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
    unsafe { untranspose_avx2(out_t, out_row_samples, rows, store) };
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
    // the start of a register within `rows`.
    unsafe {
        vertical_with_load_avx2(rows, stride, weights, mid, |ptr| {
            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(ptr as *const __m128i)))
        });
    }
}

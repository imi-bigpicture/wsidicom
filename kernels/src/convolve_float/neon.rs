//! NEON kernels of the uint16/float32 convolution resize (aarch64): 8 rows
//! per transposed batch, two f32x4 registers per column. Every sum is the
//! scalar reference's FMA chain (`vfmaq` from zero, in tap order) and uint16
//! rounds to nearest even and saturates (`vcvtnq_u32` + `vqmovn`), so the
//! results equal the scalar reference's bit for bit.
//!
//! # Safety
//! NEON is baseline on aarch64, gated only by the `set_simd` kill switch.
//! Every `unsafe fn` here requires the buffer sizes its doc states.

use std::arch::aarch64::{
    float32x4_t, uint8x8_t, uint16x4_t, vcombine_u16, vcreate_u8, vcvtnq_u32_f32, vcvtq_f32_u32,
    vdupq_n_f32, vfmaq_n_f32, vget_low_u16, vld1_u16, vld1q_f32, vmovl_u8, vmovl_u16, vqmovn_u16,
    vqmovn_u32, vreinterpretq_f32_f64, vreinterpretq_f64_f32, vst1_u8, vst1_u16, vst1q_f32,
    vtrn1q_f32, vtrn1q_f64, vtrn2q_f32, vtrn2q_f64,
};

use super::Windows;
use crate::convolve::Window;
use crate::simd::cover_row;

/// f32 lanes per register.
const LANES: usize = 4;
/// Rows per transposed batch: two registers.
pub(crate) const ROWS: usize = 2 * LANES;

/// Transpose 4x4 f32: `rows[r][c]` -> `columns[c][r]`.
#[target_feature(enable = "neon")]
#[inline]
fn transpose_4x4_f32(rows: [float32x4_t; 4]) -> [float32x4_t; 4] {
    let pairs = [
        vreinterpretq_f64_f32(vtrn1q_f32(rows[0], rows[1])),
        vreinterpretq_f64_f32(vtrn2q_f32(rows[0], rows[1])),
        vreinterpretq_f64_f32(vtrn1q_f32(rows[2], rows[3])),
        vreinterpretq_f64_f32(vtrn2q_f32(rows[2], rows[3])),
    ];
    [
        vreinterpretq_f32_f64(vtrn1q_f64(pairs[0], pairs[2])),
        vreinterpretq_f32_f64(vtrn1q_f64(pairs[1], pairs[3])),
        vreinterpretq_f32_f64(vtrn2q_f64(pairs[0], pairs[2])),
        vreinterpretq_f32_f64(vtrn2q_f64(pairs[1], pairs[3])),
    ]
}

/// Four f32 at `src`.
///
/// # Safety
/// `src..src + 4` must be readable.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_f32x4(src: *const f32) -> float32x4_t {
    // SAFETY: the caller's contract.
    unsafe { vld1q_f32(src) }
}

/// Four u16 at `src` as f32.
///
/// # Safety
/// `src..src + 4` must be readable.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_u16x4_f32(src: *const u16) -> float32x4_t {
    // SAFETY: the caller's contract.
    vcvtq_f32_u32(vmovl_u16(unsafe { vld1_u16(src) }))
}

/// 4 uint8 samples as f32.
///
/// # Safety
/// `src` is valid for reads of 4 bytes.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_u8x4_f32(src: *const u8) -> float32x4_t {
    // SAFETY: the caller's contract (4 bytes, read unaligned).
    let word = unsafe { (src as *const u32).read_unaligned() };
    let bytes = vcreate_u8(u64::from(word));
    vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(bytes))))
}

/// Bytes per cache line (Apple silicon's).
const LINE: usize = 128;
/// Taps from which a vertical block is a whole line.
const LONG_WINDOW_TAPS: usize = 16;

// The vertical pass for one sample type: `$load` widens 4 samples to f32.
macro_rules! gen_vertical_neon {
    ($name:ident, $blocks:ident, $t:ty, $load:ident) => {
        /// `mid[x] = sum_t weights[t] * rows[t * stride + x]`, an FMA chain
        /// in tap order, in blocks of several registers per tap (independent
        /// chains), the last block overlapping its predecessor: a whole
        /// 128-byte cache line of each row for long windows, whose rows
        /// otherwise evict each other's lines from shared L1 sets before the
        /// next block uses them, half a line for shorter ones; narrower rows
        /// in 8-sample blocks, and rows shorter than that by the scalar
        /// reference.
        ///
        /// # Safety
        /// `rows` holds `weights.len()` rows `stride` apart of `mid.len()`
        /// samples.
        #[target_feature(enable = "neon")]
        pub(crate) unsafe fn $name(rows: &[$t], stride: usize, weights: &[f32], mid: &mut [f32]) {
            // uint8 blocks as uint16's: a line of uint8 would take more
            // registers than there are.
            const SAMPLE: usize = if size_of::<$t>() < 2 {
                2
            } else {
                size_of::<$t>()
            };
            const LINE_REGISTERS: usize = LINE / (LANES * SAMPLE);
            let len = mid.len();
            // SAFETY: the contract, passed on; each arm's rows hold a block.
            unsafe {
                if weights.len() >= LONG_WINDOW_TAPS && len >= LINE_REGISTERS * LANES {
                    $blocks::<LINE_REGISTERS>(rows, stride, weights, mid);
                } else if len >= LINE_REGISTERS / 2 * LANES {
                    $blocks::<{ LINE_REGISTERS / 2 }>(rows, stride, weights, mid);
                } else if len >= 2 * LANES {
                    $blocks::<2>(rows, stride, weights, mid);
                } else {
                    super::vertical(rows, stride, weights, mid);
                }
            }
        }

        /// The vertical pass in blocks of `REGISTERS` x 4 samples.
        ///
        /// # Safety
        /// As the pass, with `mid.len() >= REGISTERS * 4`.
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn $blocks<const REGISTERS: usize>(
            rows: &[$t],
            stride: usize,
            weights: &[f32],
            mid: &mut [f32],
        ) {
            let (src_ptr, mid_ptr) = (rows.as_ptr(), mid.as_mut_ptr());
            cover_row(mid.len(), REGISTERS * LANES, |x| {
                let mut sums = [vdupq_n_f32(0.0); REGISTERS];
                for (tap, &weight) in weights.iter().enumerate() {
                    for (register, sum) in sums.iter_mut().enumerate() {
                        let at = tap * stride + x + register * LANES;
                        // SAFETY: x + REGISTERS * 4 <= mid.len(): 4 samples
                        // of row `tap`.
                        let samples = unsafe { $load(src_ptr.add(at)) };
                        *sum = vfmaq_n_f32(*sum, samples, weight);
                    }
                }
                for (register, sum) in sums.into_iter().enumerate() {
                    // SAFETY: x + REGISTERS * 4 <= mid.len().
                    unsafe { vst1q_f32(mid_ptr.add(x + register * LANES), sum) };
                }
            });
        }
    };
}

gen_vertical_neon!(vertical_f32_neon, vertical_f32_blocks, f32, load_f32x4);
gen_vertical_neon!(vertical_u16_neon, vertical_u16_blocks, u16, load_u16x4_f32);
gen_vertical_neon!(vertical_u8_neon, vertical_u8_blocks, u8, load_u8x4_f32);

/// `ROWS` (= 8) rows of `mid` (`stride` apart, a multiple of 4) into
/// column-major `mid_t[x * 8 + r]`.
///
/// # Safety
/// `mid` and `mid_t` hold at least `8 * stride` samples.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn transpose_f32_rows_neon(mid: &[f32], stride: usize, mid_t: &mut [f32]) {
    debug_assert!(stride.is_multiple_of(LANES) && mid.len() >= ROWS * stride);
    debug_assert!(mid_t.len() >= ROWS * stride);
    let (mid_ptr, mid_t_ptr) = (mid.as_ptr(), mid_t.as_mut_ptr());
    for x0 in (0..stride).step_by(LANES) {
        for half in 0..ROWS / LANES {
            // SAFETY: x0 + 4 <= stride: 4 samples of each row.
            let rows = std::array::from_fn(|row| unsafe {
                vld1q_f32(mid_ptr.add((half * LANES + row) * stride + x0))
            });
            for (column, samples) in transpose_4x4_f32(rows).into_iter().enumerate() {
                // SAFETY: (x0 + column) * 8 + 8 <= stride * 8.
                unsafe { vst1q_f32(mid_t_ptr.add((x0 + column) * ROWS + half * LANES), samples) };
            }
        }
    }
}

/// The horizontal pass of a transposed batch: per output sample, an FMA
/// chain over its taps, each a unit-stride load of the batch's 8 rows.
///
/// # Safety
/// `mid_t` holds every window's samples (`col.end() * channels * 8`) and
/// `out_t` every output's (`cols.len() * channels * 8`).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn horizontal_f32_transposed_neon(
    mid_t: &[f32],
    cols: &Windows,
    channels: usize,
    out_t: &mut [f32],
) {
    for (output, col) in cols.iter().enumerate() {
        debug_assert!(
            col.end() * channels * ROWS <= mid_t.len()
                && (output + 1) * channels * ROWS <= out_t.len()
        );
        // SAFETY: (every arm) the caller's contract, for this output.
        unsafe {
            match col.f32.len() {
                1 => output_fixed_taps::<1>(mid_t, col, channels, output, out_t),
                2 => output_fixed_taps::<2>(mid_t, col, channels, output, out_t),
                3 => output_fixed_taps::<3>(mid_t, col, channels, output, out_t),
                4 => output_fixed_taps::<4>(mid_t, col, channels, output, out_t),
                5 => output_fixed_taps::<5>(mid_t, col, channels, output, out_t),
                6 => output_fixed_taps::<6>(mid_t, col, channels, output, out_t),
                7 => output_fixed_taps::<7>(mid_t, col, channels, output, out_t),
                8 => output_fixed_taps::<8>(mid_t, col, channels, output, out_t),
                9 => output_fixed_taps::<9>(mid_t, col, channels, output, out_t),
                10 => output_fixed_taps::<10>(mid_t, col, channels, output, out_t),
                11 => output_fixed_taps::<11>(mid_t, col, channels, output, out_t),
                12 => output_fixed_taps::<12>(mid_t, col, channels, output, out_t),
                _ => output_taps(mid_t, (col.start, col.f32), channels, output, out_t),
            }
        }
    }
}

/// One output of `horizontal_f32_transposed_neon` with a window of `TAPS`
/// taps: the tap loop unrolled (per-output loop setup outweighs short
/// windows' taps).
///
/// # Safety
/// As `horizontal_f32_transposed_neon`, for `output`, whose window `col`
/// has `TAPS` taps.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn output_fixed_taps<const TAPS: usize>(
    mid_t: &[f32],
    col: Window<'_>,
    channels: usize,
    output: usize,
    out_t: &mut [f32],
) {
    let weights: [f32; TAPS] = std::array::from_fn(|tap| col.f32[tap]);
    // SAFETY: the caller's contract.
    unsafe { output_taps(mid_t, (col.start, &weights), channels, output, out_t) };
}

/// One output of `horizontal_f32_transposed_neon`: per channel, an FMA chain
/// over the window's taps, each a unit-stride load of the batch's 8 rows.
///
/// # Safety
/// As `horizontal_f32_transposed_neon`, for `output` with the window
/// `(start, weights)`.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn output_taps(
    mid_t: &[f32],
    (start, weights): (usize, &[f32]),
    channels: usize,
    output: usize,
    out_t: &mut [f32],
) {
    let (mid_t_ptr, out_t_ptr) = (mid_t.as_ptr(), out_t.as_mut_ptr());
    for channel in 0..channels {
        let (mut low, mut high) = (vdupq_n_f32(0.0), vdupq_n_f32(0.0));
        for (tap, &weight) in weights.iter().enumerate() {
            let at = ((start + tap) * channels + channel) * ROWS;
            // SAFETY: inside the window's samples (the caller's contract).
            let (samples_low, samples_high) = unsafe {
                (
                    vld1q_f32(mid_t_ptr.add(at)),
                    vld1q_f32(mid_t_ptr.add(at + LANES)),
                )
            };
            low = vfmaq_n_f32(low, samples_low, weight);
            high = vfmaq_n_f32(high, samples_high, weight);
        }
        let at = (output * channels + channel) * ROWS;
        // SAFETY: inside the outputs' sums (the caller's contract).
        unsafe {
            vst1q_f32(out_t_ptr.add(at), low);
            vst1q_f32(out_t_ptr.add(at + LANES), high);
        }
    }
}

/// `out_t[o * 8 + r]` back to rows `dst[r * stride + o]` (o < `ows`, r <
/// `rows`), 4x4 transposes, each row's 4 samples through `store(row, o,
/// samples, count)` (`count` < 4 at the row end).
///
/// # Safety
/// `out_t` holds `ows.next_multiple_of(8) * 8` sums; `rows <= 8`.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn untranspose(
    out_t: &[f32],
    ows: usize,
    rows: usize,
    mut store: impl FnMut(usize, usize, float32x4_t, usize),
) {
    debug_assert!(out_t.len() >= ows.next_multiple_of(ROWS) * ROWS && rows <= ROWS);
    let out_t_ptr = out_t.as_ptr();
    for o0 in (0..ows).step_by(LANES) {
        for half in 0..ROWS / LANES {
            // SAFETY: o0 < ows, a multiple of 4, so o0 + 4 <=
            // ows.next_multiple_of(8): 4 outputs' sums of 4 rows.
            let sums = std::array::from_fn(|output| unsafe {
                vld1q_f32(out_t_ptr.add((o0 + output) * ROWS + half * LANES))
            });
            let by_row = transpose_4x4_f32(sums);
            for row in (half * LANES..(half + 1) * LANES).take_while(|&row| row < rows) {
                store(row, o0, by_row[row - half * LANES], LANES.min(ows - o0));
            }
        }
    }
}

/// Untranspose a batch to float32 rows.
///
/// # Safety
/// As `untranspose`, and `dst` holds `rows` rows `stride` apart of `ows`
/// samples.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn untranspose_store_f32_neon(
    out_t: &[f32],
    ows: usize,
    dst: *mut f32,
    stride: usize,
    rows: usize,
) {
    // SAFETY: the caller's contract; each store writes `count` samples of
    // row `row` from output `output`.
    unsafe {
        untranspose(out_t, ows, rows, |row, output, samples, count| {
            let at = dst.add(row * stride + output);
            if count == LANES {
                vst1q_f32(at, samples);
            } else {
                let mut staged = [0f32; LANES];
                vst1q_f32(staged.as_mut_ptr(), samples);
                std::ptr::copy_nonoverlapping(staged.as_ptr(), at, count);
            }
        });
    }
}

/// Untranspose a batch to uint16 rows: rounded to nearest even, saturated.
///
/// # Safety
/// As `untranspose_store_f32_neon`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn untranspose_store_u16_neon(
    out_t: &[f32],
    ows: usize,
    dst: *mut u16,
    stride: usize,
    rows: usize,
) {
    let narrow = |samples: float32x4_t| -> uint16x4_t { vqmovn_u32(vcvtnq_u32_f32(samples)) };
    // SAFETY: the caller's contract; each store writes `count` samples of
    // row `row` from output `output`.
    unsafe {
        untranspose(out_t, ows, rows, |row, output, samples, count| {
            let at = dst.add(row * stride + output);
            if count == LANES {
                vst1_u16(at, narrow(samples));
            } else {
                let mut staged = [0u16; LANES];
                vst1_u16(staged.as_mut_ptr(), narrow(samples));
                std::ptr::copy_nonoverlapping(staged.as_ptr(), at, count);
            }
        });
    }
}

/// Untranspose a batch to uint8 rows: rounded to nearest even, saturated.
///
/// # Safety
/// As `untranspose_store_f32_neon`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn untranspose_store_u8_neon(
    out_t: &[f32],
    ows: usize,
    dst: *mut u8,
    stride: usize,
    rows: usize,
) {
    let narrow = |samples: float32x4_t| -> uint8x8_t {
        let words = vqmovn_u32(vcvtnq_u32_f32(samples));
        vqmovn_u16(vcombine_u16(words, words))
    };
    // SAFETY: the caller's contract; each store writes `count` samples of
    // row `row` from output `output`.
    unsafe {
        untranspose(out_t, ows, rows, |row, output, samples, count| {
            let mut staged = [0u8; 8];
            vst1_u8(staged.as_mut_ptr(), narrow(samples));
            std::ptr::copy_nonoverlapping(staged.as_ptr(), dst.add(row * stride + output), count);
        });
    }
}

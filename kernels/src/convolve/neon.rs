//! NEON kernels of the uint8 convolution resize (aarch64): the Q6 vertical
//! pass and the transposed path's horizontal pass, transposes and narrowing.
//!
//! `vqrdmulh` is x86's `mulhrs` (`(value * weight + 2^14) >> 15`; it
//! saturates only at -32768 * -32768, which Q15 weights above -1 never
//! reach) and the sums wrap in i16 as the x86 kernels' do, so the results
//! equal the scalar reference's bit for bit. Where the CPU has ARMv8.1 RDM,
//! each multiply and add is one `vqrdmlah`, which saturates the sum instead
//! of wrapping it: the same, since the plans give these kernels only
//! windows whose partial sums stay within i16 (`q15_sums_fit`).
//!
//! # Safety
//! NEON is baseline on aarch64, gated only by the `set_simd` kill switch.
//! Every `unsafe fn` here requires the buffer sizes its doc states.

use std::arch::aarch64::{
    int16x8_t, vaddq_s16, vdupq_n_s16, vget_low_u8, vld1q_s16, vld1q_u8, vqrdmlahq_s16,
    vqrdmulhq_s16, vqrshrun_n_s16, vreinterpretq_s16_s64, vreinterpretq_s16_u16,
    vreinterpretq_s32_s16, vreinterpretq_s64_s32, vshll_high_n_u8, vshll_n_u8, vst1_u8, vst1q_s16,
    vtrn1q_s16, vtrn1q_s32, vtrn1q_s64, vtrn2q_s16, vtrn2q_s32, vtrn2q_s64,
};

use super::{MAX_PADDED_TAPS, PaddedWindows, TRANSPOSE_ROWS, Windows};
use crate::cpu::has_rdm;
use crate::sample::BICUBIC_U8_FRAC_BITS;
use crate::simd::cover_row;

const Q6: i32 = BICUBIC_U8_FRAC_BITS as i32;
/// i16 lanes per register.
const LANES: usize = 8;

/// `mid[x] = sum_t mulhrs(rows[t][x] << 6, weights[t])` (Q6, wrapping i16
/// sums) in blocks of several registers per tap, the last block overlapping
/// its predecessor: a whole 128-byte cache line for long windows, whose
/// rows otherwise evict each other's lines from shared L1 sets before the
/// next block uses them, half a line for shorter ones; narrower rows in
/// 16-sample blocks.
///
/// # Safety
/// `rows` holds `weights.len()` rows of `width` samples, `mid` at least
/// `width`, and `width >= 16`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn vertical_q6_neon(rows: &[u8], width: usize, weights: &[i16], mid: &mut [i16]) {
    debug_assert!(rows.len() >= weights.len() * width && mid.len() >= width && width >= 16);
    let line = weights.len() >= LONG_WINDOW_TAPS && width >= LINE;
    let half_line = width >= LINE / 2;
    // SAFETY: the contract, passed on; RDM is detected for its kernels.
    unsafe {
        match (has_rdm(), line, half_line) {
            (true, true, _) => vertical_blocks_rdm::<{ LINE / 16 }>(rows, width, weights, mid),
            (true, false, true) => vertical_blocks_rdm::<{ LINE / 32 }>(rows, width, weights, mid),
            (true, false, false) => vertical_blocks_rdm::<1>(rows, width, weights, mid),
            (false, true, _) => vertical_blocks::<{ LINE / 16 }>(rows, width, weights, mid),
            (false, false, true) => vertical_blocks::<{ LINE / 32 }>(rows, width, weights, mid),
            (false, false, false) => vertical_blocks::<1>(rows, width, weights, mid),
        }
    }
}

/// Bytes per cache line (Apple silicon's; two lines of 64 bytes elsewhere).
const LINE: usize = 128;
/// Taps from which a vertical block is a whole line.
const LONG_WINDOW_TAPS: usize = 16;

/// `sums + mulhrs(samples, weight)` per i16 lane, wrapping.
#[target_feature(enable = "neon")]
#[inline]
fn add_mulhrs(sums: int16x8_t, samples: int16x8_t, weight: int16x8_t) -> int16x8_t {
    vaddq_s16(sums, vqrdmulhq_s16(samples, weight))
}

/// `add_mulhrs` in one instruction, saturating instead of wrapping.
#[target_feature(enable = "neon,rdm")]
#[inline]
fn add_mulhrs_rdm(sums: int16x8_t, samples: int16x8_t, weight: int16x8_t) -> int16x8_t {
    vqrdmlahq_s16(sums, samples, weight)
}

// The vertical pass in blocks of `LOADS` x 16 samples, with `$add` the
// multiply-accumulate of the target features `$features`.
macro_rules! gen_vertical_blocks {
    ($name:ident, $features:literal, $add:ident) => {
        /// `vertical_q6_neon` in blocks of `LOADS` x 16 samples.
        ///
        /// # Safety
        /// As `vertical_q6_neon`, with `width >= LOADS * 16`.
        #[target_feature(enable = $features)]
        #[inline]
        unsafe fn $name<const LOADS: usize>(
            rows: &[u8],
            width: usize,
            weights: &[i16],
            mid: &mut [i16],
        ) {
            let (src_ptr, mid_ptr) = (rows.as_ptr(), mid.as_mut_ptr());
            cover_row(width, LOADS * 16, |x| {
                let mut sums = [[vdupq_n_s16(0); 2]; LOADS];
                for (tap, &weight) in weights.iter().enumerate() {
                    let weight = vdupq_n_s16(weight);
                    for (load, [low, high]) in sums.iter_mut().enumerate() {
                        // SAFETY: x + LOADS * 16 <= width: 16 samples of row `tap`.
                        let samples = unsafe { vld1q_u8(src_ptr.add(tap * width + x + 16 * load)) };
                        let widened_low =
                            vreinterpretq_s16_u16(vshll_n_u8::<Q6>(vget_low_u8(samples)));
                        let widened_high = vreinterpretq_s16_u16(vshll_high_n_u8::<Q6>(samples));
                        *low = $add(*low, widened_low, weight);
                        *high = $add(*high, widened_high, weight);
                    }
                }
                for (load, [low, high]) in sums.into_iter().enumerate() {
                    // SAFETY: x + LOADS * 16 <= width <= mid's length.
                    unsafe {
                        vst1q_s16(mid_ptr.add(x + 16 * load), low);
                        vst1q_s16(mid_ptr.add(x + 16 * load + LANES), high);
                    }
                }
            });
        }
    };
}

gen_vertical_blocks!(vertical_blocks, "neon", add_mulhrs);
gen_vertical_blocks!(vertical_blocks_rdm, "neon,rdm", add_mulhrs_rdm);

/// Transpose 8x8 i16: `rows[r][c]` -> `columns[c][r]`, by 16-, 32- and
/// 64-bit lane-pair swaps.
#[target_feature(enable = "neon")]
#[inline]
fn transpose_8x8_i16(rows: [int16x8_t; 8]) -> [int16x8_t; 8] {
    let pairs16 = [0, 2, 4, 6].map(|row| {
        (
            vreinterpretq_s32_s16(vtrn1q_s16(rows[row], rows[row + 1])),
            vreinterpretq_s32_s16(vtrn2q_s16(rows[row], rows[row + 1])),
        )
    });
    // quads[q][k]: 32-bit pairs of rows 4q .. 4q + 4, column offset k.
    let quads = [0, 2].map(|first| {
        let (upper, lower) = (pairs16[first], pairs16[first + 1]);
        [
            vreinterpretq_s64_s32(vtrn1q_s32(upper.0, lower.0)),
            vreinterpretq_s64_s32(vtrn1q_s32(upper.1, lower.1)),
            vreinterpretq_s64_s32(vtrn2q_s32(upper.0, lower.0)),
            vreinterpretq_s64_s32(vtrn2q_s32(upper.1, lower.1)),
        ]
    });
    let [top, bottom] = quads;
    std::array::from_fn(|column| {
        let (top, bottom) = (top[column % 4], bottom[column % 4]);
        vreinterpretq_s16_s64(if column < 4 {
            vtrn1q_s64(top, bottom)
        } else {
            vtrn2q_s64(top, bottom)
        })
    })
}

/// `TRANSPOSE_ROWS` (= 16) rows of `mid` (stride `wp`, a multiple of 8)
/// into column-major `mid_t[x * 16 + r]`.
///
/// # Safety
/// `mid` and `mid_t` hold at least `16 * wp` samples.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn transpose_rows_i16_neon(mid: &[i16], wp: usize, mid_t: &mut [i16]) {
    const ROWS: usize = TRANSPOSE_ROWS;
    debug_assert!(wp.is_multiple_of(LANES) && mid.len() >= ROWS * wp && mid_t.len() >= ROWS * wp);
    let (mid_ptr, mid_t_ptr) = (mid.as_ptr(), mid_t.as_mut_ptr());
    for x0 in (0..wp).step_by(LANES) {
        for half in 0..ROWS / LANES {
            // SAFETY: x0 + 8 <= wp: 8 samples of each of the 16 rows.
            let rows = std::array::from_fn(|row| unsafe {
                vld1q_s16(mid_ptr.add((half * LANES + row) * wp + x0))
            });
            for (column, samples) in transpose_8x8_i16(rows).into_iter().enumerate() {
                // SAFETY: (x0 + column) * 16 + 16 <= wp * 16 <= mid_t's length.
                unsafe { vst1q_s16(mid_t_ptr.add((x0 + column) * ROWS + half * LANES), samples) };
            }
        }
    }
}

/// The horizontal pass on transposed rows, as `horizontal_transposed_avx2`:
/// for every output whose weights fit Q15 and each of its `channels`
/// samples, the Q6 sums of all 16 rows — a unit-stride load per tap.
///
/// # Safety
/// `mid_t` holds every window's samples (`col.end() * channels * 16`) and
/// `out_t` every output's (`cols.len() * channels * 16`).
#[target_feature(enable = "neon")]
pub(crate) unsafe fn horizontal_transposed_neon(
    mid_t: &[i16],
    cols: &Windows,
    channels: usize,
    out_t: &mut [i16],
) {
    // SAFETY: the contract, passed on; RDM is detected for its kernel.
    unsafe {
        if has_rdm() {
            horizontal_transposed_rdm(mid_t, cols, channels, out_t);
        } else {
            horizontal_transposed_plain(mid_t, cols, channels, out_t);
        }
    }
}

// The transposed horizontal pass with `$add` the multiply-accumulate of the
// target features `$features`.
macro_rules! gen_horizontal_transposed {
    ($name:ident, $features:literal, $add:ident) => {
        /// `horizontal_transposed_neon` with one multiply-accumulate.
        ///
        /// # Safety
        /// As `horizontal_transposed_neon`.
        #[target_feature(enable = $features)]
        unsafe fn $name(mid_t: &[i16], cols: &Windows, channels: usize, out_t: &mut [i16]) {
            const ROWS: usize = TRANSPOSE_ROWS;
            let (mid_t_ptr, out_t_ptr) = (mid_t.as_ptr(), out_t.as_mut_ptr());
            for (output, col) in cols.iter().enumerate() {
                if !col.fits_q15() {
                    continue;
                }
                debug_assert!(
                    col.end() * channels * ROWS <= mid_t.len()
                        && (output + 1) * channels * ROWS <= out_t.len()
                );
                for channel in 0..channels {
                    let (mut low, mut high) = (vdupq_n_s16(0), vdupq_n_s16(0));
                    for (tap, &weight) in col.q15.iter().enumerate() {
                        let weight = vdupq_n_s16(weight);
                        let at = ((col.start + tap) * channels + channel) * ROWS;
                        // SAFETY: inside the window's samples (the caller's contract).
                        let (samples_low, samples_high) = unsafe {
                            (
                                vld1q_s16(mid_t_ptr.add(at)),
                                vld1q_s16(mid_t_ptr.add(at + LANES)),
                            )
                        };
                        low = $add(low, samples_low, weight);
                        high = $add(high, samples_high, weight);
                    }
                    let at = (output * channels + channel) * ROWS;
                    // SAFETY: inside the outputs' sums (the caller's contract).
                    unsafe {
                        vst1q_s16(out_t_ptr.add(at), low);
                        vst1q_s16(out_t_ptr.add(at + LANES), high);
                    }
                }
            }
        }
    };
}

gen_horizontal_transposed!(horizontal_transposed_plain, "neon", add_mulhrs);
gen_horizontal_transposed!(horizontal_transposed_rdm, "neon,rdm", add_mulhrs_rdm);

/// `horizontal_transposed_neon` over padded windows.
///
/// # Safety
/// `mid_t` holds every padded window's samples, `(start + taps) *
/// channels * 16`, and `out_t` every output's, `outputs * channels * 16`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn horizontal_padded_neon(
    mid_t: &[i16],
    padded: &PaddedWindows,
    channels: usize,
    out_t: &mut [i16],
) {
    let (starts, weights) = (&padded.starts[..], &padded.weights[..]);
    // SAFETY: the contract, passed on; RDM is detected for its kernels.
    unsafe {
        match (has_rdm(), padded.taps) {
            (true, 1) => horizontal_padded_rdm::<1>(mid_t, starts, weights, channels, out_t),
            (true, 2) => horizontal_padded_rdm::<2>(mid_t, starts, weights, channels, out_t),
            (true, 3) => horizontal_padded_rdm::<3>(mid_t, starts, weights, channels, out_t),
            (true, 4) => horizontal_padded_rdm::<4>(mid_t, starts, weights, channels, out_t),
            (true, 5) => horizontal_padded_rdm::<5>(mid_t, starts, weights, channels, out_t),
            (true, 6) => horizontal_padded_rdm::<6>(mid_t, starts, weights, channels, out_t),
            (true, 7) => horizontal_padded_rdm::<7>(mid_t, starts, weights, channels, out_t),
            (true, 8) => horizontal_padded_rdm::<8>(mid_t, starts, weights, channels, out_t),
            (true, 9) => horizontal_padded_rdm::<9>(mid_t, starts, weights, channels, out_t),
            (true, 10) => horizontal_padded_rdm::<10>(mid_t, starts, weights, channels, out_t),
            (true, 11) => horizontal_padded_rdm::<11>(mid_t, starts, weights, channels, out_t),
            (true, 12) => horizontal_padded_rdm::<12>(mid_t, starts, weights, channels, out_t),
            (true, 13) => horizontal_padded_rdm::<13>(mid_t, starts, weights, channels, out_t),
            (true, 14) => horizontal_padded_rdm::<14>(mid_t, starts, weights, channels, out_t),
            (true, 15) => horizontal_padded_rdm::<15>(mid_t, starts, weights, channels, out_t),
            (true, 16) => horizontal_padded_rdm::<16>(mid_t, starts, weights, channels, out_t),
            (true, 17) => horizontal_padded_rdm::<17>(mid_t, starts, weights, channels, out_t),
            (true, 18) => horizontal_padded_rdm::<18>(mid_t, starts, weights, channels, out_t),
            (true, 19) => horizontal_padded_rdm::<19>(mid_t, starts, weights, channels, out_t),
            (true, _) => horizontal_padded_rdm::<{ MAX_PADDED_TAPS }>(
                mid_t, starts, weights, channels, out_t,
            ),
            (false, 1) => horizontal_padded_plain::<1>(mid_t, starts, weights, channels, out_t),
            (false, 2) => horizontal_padded_plain::<2>(mid_t, starts, weights, channels, out_t),
            (false, 3) => horizontal_padded_plain::<3>(mid_t, starts, weights, channels, out_t),
            (false, 4) => horizontal_padded_plain::<4>(mid_t, starts, weights, channels, out_t),
            (false, 5) => horizontal_padded_plain::<5>(mid_t, starts, weights, channels, out_t),
            (false, 6) => horizontal_padded_plain::<6>(mid_t, starts, weights, channels, out_t),
            (false, 7) => horizontal_padded_plain::<7>(mid_t, starts, weights, channels, out_t),
            (false, 8) => horizontal_padded_plain::<8>(mid_t, starts, weights, channels, out_t),
            (false, 9) => horizontal_padded_plain::<9>(mid_t, starts, weights, channels, out_t),
            (false, 10) => horizontal_padded_plain::<10>(mid_t, starts, weights, channels, out_t),
            (false, 11) => horizontal_padded_plain::<11>(mid_t, starts, weights, channels, out_t),
            (false, 12) => horizontal_padded_plain::<12>(mid_t, starts, weights, channels, out_t),
            (false, 13) => horizontal_padded_plain::<13>(mid_t, starts, weights, channels, out_t),
            (false, 14) => horizontal_padded_plain::<14>(mid_t, starts, weights, channels, out_t),
            (false, 15) => horizontal_padded_plain::<15>(mid_t, starts, weights, channels, out_t),
            (false, 16) => horizontal_padded_plain::<16>(mid_t, starts, weights, channels, out_t),
            (false, 17) => horizontal_padded_plain::<17>(mid_t, starts, weights, channels, out_t),
            (false, 18) => horizontal_padded_plain::<18>(mid_t, starts, weights, channels, out_t),
            (false, 19) => horizontal_padded_plain::<19>(mid_t, starts, weights, channels, out_t),
            (false, _) => horizontal_padded_plain::<{ MAX_PADDED_TAPS }>(
                mid_t, starts, weights, channels, out_t,
            ),
        }
    }
}

// The padded horizontal pass with `$add` the multiply-accumulate of the
// target features `$features`.
macro_rules! gen_horizontal_padded {
    ($name:ident, $features:literal, $add:ident) => {
        /// `horizontal_padded_neon` for `TAPS` taps per window.
        ///
        /// # Safety
        /// As `horizontal_padded_neon`, `weights` holding `TAPS` per start.
        #[target_feature(enable = $features)]
        unsafe fn $name<const TAPS: usize>(
            mid_t: &[i16],
            starts: &[usize],
            weights: &[i16],
            channels: usize,
            out_t: &mut [i16],
        ) {
            const ROWS: usize = TRANSPOSE_ROWS;
            let (mid_t_ptr, out_t_ptr) = (mid_t.as_ptr(), out_t.as_mut_ptr());
            let windows = starts.iter().zip(weights.chunks_exact(TAPS));
            for (output, (&start, weights)) in windows.enumerate() {
                debug_assert!(
                    (start + TAPS) * channels * ROWS <= mid_t.len()
                        && (output + 1) * channels * ROWS <= out_t.len()
                );
                let weights: [int16x8_t; TAPS] =
                    std::array::from_fn(|tap| vdupq_n_s16(weights[tap]));
                for channel in 0..channels {
                    let (mut low, mut high) = (vdupq_n_s16(0), vdupq_n_s16(0));
                    for (tap, &weight) in weights.iter().enumerate() {
                        let at = ((start + tap) * channels + channel) * ROWS;
                        // SAFETY: inside the padded window's samples (the
                        // caller's contract).
                        let (samples_low, samples_high) = unsafe {
                            (
                                vld1q_s16(mid_t_ptr.add(at)),
                                vld1q_s16(mid_t_ptr.add(at + LANES)),
                            )
                        };
                        low = $add(low, samples_low, weight);
                        high = $add(high, samples_high, weight);
                    }
                    let at = (output * channels + channel) * ROWS;
                    // SAFETY: inside the outputs' sums (the caller's contract).
                    unsafe {
                        vst1q_s16(out_t_ptr.add(at), low);
                        vst1q_s16(out_t_ptr.add(at + LANES), high);
                    }
                }
            }
        }
    };
}

gen_horizontal_padded!(horizontal_padded_plain, "neon", add_mulhrs);
gen_horizontal_padded!(horizontal_padded_rdm, "neon,rdm", add_mulhrs_rdm);

/// Q6 sums `out_t[o * 16 + r]` back to row-major u8 `out[r * ow + o]` for the
/// first `rows` rows: 8x8 transposes, then `vqrshrun` (rounded, clamped).
/// Each row's output is staged a cache line at a time: 8-byte pieces of 16
/// rows at once would, at power-of-2 row strides (which share L1 sets),
/// evict each line before it is complete.
///
/// # Safety
/// `out_t` holds `ow.next_multiple_of(16) * 16` sums and `out` `rows * ow`
/// samples, `rows <= 16`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn untranspose_narrow_neon(
    out_t: &[i16],
    ow: usize,
    out: &mut [u8],
    rows: usize,
) {
    const LINE: usize = 64;
    const ROWS: usize = TRANSPOSE_ROWS;
    debug_assert!(out_t.len() >= ow.next_multiple_of(ROWS) * ROWS && out.len() >= rows * ow);
    let out_t_ptr = out_t.as_ptr();
    let mut staged = [[0u8; LINE]; ROWS];
    for line_start in (0..ow).step_by(LINE) {
        for o0 in (line_start..(line_start + LINE).min(ow)).step_by(LANES) {
            for half in 0..ROWS / LANES {
                // SAFETY: o0 < ow with o0 a multiple of 8, so o0 + 8 <=
                // ow.next_multiple_of(16): 8 outputs' sums of 8 rows.
                let sums = std::array::from_fn(|output| unsafe {
                    vld1q_s16(out_t_ptr.add((o0 + output) * ROWS + half * LANES))
                });
                let by_row = transpose_8x8_i16(sums);
                for (row, staged) in staged.iter_mut().enumerate().skip(half * LANES).take(LANES) {
                    let narrowed = vqrshrun_n_s16::<Q6>(by_row[row - half * LANES]);
                    // SAFETY: o0 - line_start + 8 <= 64, inside the staged line.
                    unsafe { vst1_u8(staged.as_mut_ptr().add(o0 - line_start), narrowed) };
                }
            }
        }
        let len = LINE.min(ow - line_start);
        for (row, line) in staged.iter().enumerate().take(rows) {
            out[row * ow + line_start..][..len].copy_from_slice(&line[..len]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convolve::{Kernel, PADDED_READS_PAST_ROW};
    use crate::testing::{Rng, random_samples};

    /// A vertical block kernel.
    type VerticalBlocks = unsafe fn(&[u8], usize, &[i16], &mut [i16]);
    /// A padded horizontal pass.
    type PaddedPass = unsafe fn(&[i16], &[usize], &[i16], usize, &mut [i16]);

    /// The plain transposed horizontal passes (for CPUs without RDM), over
    /// variable and padded windows, give the RDM ones' results, which the
    /// convolve tests check against the reference.
    #[test]
    fn rdm_and_plain_horizontal_kernels_agree() {
        const ROWS: usize = TRANSPOSE_ROWS;
        // The padded pass of `taps` taps, plain or RDM.
        macro_rules! padded_pass {
            ($rdm:expr, $taps:expr, $($tap_count:literal)+) => {
                match $taps {
                    $($tap_count => if $rdm {
                        horizontal_padded_rdm::<$tap_count> as PaddedPass
                    } else {
                        horizontal_padded_plain::<$tap_count> as PaddedPass
                    },)+
                    taps => unreachable!("{taps} padded taps"),
                }
            };
        }
        if !has_rdm() {
            return;
        }
        let mut rng = Rng::new(32);
        for kernel in [Kernel::Bilinear, Kernel::CatmullRom, Kernel::Lanczos3] {
            for channels in [1, 3, 4] {
                for (width, ow) in [
                    (17, 33),
                    (64, 32),
                    (130, 37),
                    (130, 200),
                    (201, 67),
                    (640, 10),
                ] {
                    let cols = Windows::new(width, ow, kernel);
                    let mid_samples =
                        ((width * channels).next_multiple_of(64) + 32 + PADDED_READS_PAST_ROW)
                            * ROWS;
                    let mid_t: Vec<i16> = random_samples::<u8>(&mut rng, mid_samples)
                        .into_iter()
                        .map(|sample| (i16::from(sample) - 16) << Q6)
                        .collect();
                    let out_samples = (ow * channels).next_multiple_of(ROWS) * ROWS;
                    let run = |pass: &dyn Fn(&mut [i16])| {
                        let mut out_t = vec![i16::MIN; out_samples];
                        pass(&mut out_t);
                        out_t
                    };
                    let context = format!("{width} -> {ow} x {channels}");
                    let plain = run(&|out_t| {
                        // SAFETY: NEON detected; mid_t is padded as
                        // resize_transposed pads it, out_t holds every output.
                        unsafe { horizontal_transposed_plain(&mid_t, &cols, channels, out_t) };
                    });
                    let rdm = run(&|out_t| {
                        // SAFETY: as above, and RDM detected.
                        unsafe { horizontal_transposed_rdm(&mid_t, &cols, channels, out_t) };
                    });
                    assert_eq!(plain, rdm, "transposed {context}");
                    if let Some(padded) = PaddedWindows::new(&cols) {
                        let padded_run = |rdm: bool| {
                            let pass = padded_pass!(rdm, padded.taps, 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20);
                            run(&|out_t| {
                                let (starts, weights) = (&padded.starts, &padded.weights);
                                // SAFETY: NEON (and for the RDM pass RDM)
                                // detected; mid_t is padded as
                                // resize_transposed pads it, out_t holds
                                // every output.
                                unsafe { pass(&mid_t, starts, weights, channels, out_t) };
                            })
                        };
                        assert_eq!(padded_run(false), padded_run(true), "padded {context}");
                    }
                }
            }
        }
    }

    /// The RDM kernels' saturating sums equal the plain kernels' wrapping
    /// ones (and the scalar definition) for windows whose partial sums stay
    /// within i16, in every block size.
    #[test]
    fn rdm_and_plain_vertical_kernels_agree() {
        // Absolute weights summing to at most this keep the sums of
        // mulhrs(16320, w) within i16.
        const WEIGHT_TOTAL: i32 = 60_000;
        if !has_rdm() {
            return;
        }
        let blocks: [(VerticalBlocks, usize, &str); 6] = [
            (vertical_blocks::<1>, 1, "plain"),
            (vertical_blocks_rdm::<1>, 1, "rdm"),
            (vertical_blocks::<{ LINE / 32 }>, LINE / 32, "plain"),
            (vertical_blocks_rdm::<{ LINE / 32 }>, LINE / 32, "rdm"),
            (vertical_blocks::<{ LINE / 16 }>, LINE / 16, "plain"),
            (vertical_blocks_rdm::<{ LINE / 16 }>, LINE / 16, "rdm"),
        ];
        let mut rng = Rng::new(31);
        for (width, taps) in [(16, 1), (40, 3), (64, 6), (100, 17), (130, 18), (300, 41)] {
            let bound = (WEIGHT_TOTAL / i32::try_from(taps).unwrap_or(i32::MAX)).min(32_767);
            let weights: Vec<i16> = random_samples::<u16>(&mut rng, taps)
                .into_iter()
                .map(|weight| ((i32::from(weight) - 32_768) * bound / 32_768) as i16)
                .collect();
            let rows: Vec<u8> = random_samples(&mut rng, taps * width);
            let expected: Vec<i16> = (0..width)
                .map(|x| {
                    let terms = weights.iter().enumerate().map(|(tap, &weight)| {
                        let sample = i32::from(rows[tap * width + x]) << Q6;
                        ((sample * i32::from(weight) + (1 << 14)) >> 15) as i16
                    });
                    terms.fold(0i16, i16::wrapping_add)
                })
                .collect();
            for (kernel, loads, name) in blocks {
                if width < loads * 16 {
                    continue;
                }
                let mut mid = vec![i16::MIN; width];
                // SAFETY: NEON and RDM detected; rows holds taps rows of
                // width >= loads * 16 samples, mid width.
                unsafe { kernel(&rows, width, &weights, &mut mid) };
                assert_eq!(
                    mid, expected,
                    "{name} x {loads}, width {width}, {taps} taps"
                );
            }
        }
    }
}

//! Native convolution resize of u8 images, planar or interleaved RGB, any
//! ratio, with `fast_image_resize`'s filters and window
//! rules (clipped at the image edge, renormalized) — the same function, but
//! more exactly: the vertical pass keeps a Q6 i16 intermediate instead of
//! rounding and clamping to u8, which keeps every output within 1 LSB of the
//! exact convolution for reductions down to 1/128 per axis. Each tap's
//! product rounds to the Q6 grid, so the error grows with the tap count:
//! beyond 1/128 it could exceed 1 LSB, so larger reductions take the f32
//! engine (`crate::convolve_float`) instead.
//!
//! Vertical pass first, one output row at a time (contiguous, SIMD across the
//! row, no intermediate image), then the horizontal pass on that cached row.
//! The SIMD horizontal pass runs on transposed 16-row batches
//! (`resize_transposed`); on AVX2, some RGB reductions run per row instead
//! (`TapRuns`, on rows widened to the RGBX layout and packed back).
//! Everything else runs the scalar reference per row. The SIMD kernels mirror the scalar arithmetic bit for
//! bit, so results are identical on every CPU.
//!
//! Interleaved images run the same vertical pass over cw samples and the
//! same weights per channel, so they equal resizing each channel as a planar
//! image.

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

use std::cell::RefCell;
use std::sync::Arc;

use crate::sample::{CONVOLVE_U8_FRAC_BITS, mulhrs, q15};

/// Fraction bits of the intermediate row (u8 << 6 leaves i16 headroom for
/// the filters' overshoot).
const MID_FRAC_BITS: u32 = CONVOLVE_U8_FRAC_BITS;
/// The weights' fixed point: Q15, as `mulhrs` multiplies by.
const Q15_BITS: u32 = 15;
const Q15_ONE: f64 = (1 << Q15_BITS) as f64;
/// Unnormalized weights this small are zeros that rounding left behind.
const ZERO_WEIGHT_EPS: f64 = 1e-9;
/// i16 lanes per AVX2 register: the narrowest rows its vertical kernels
/// take.
const AVX2_I16_LANES: usize = 16;
/// Output rows batched per transposed horizontal pass (one AVX2 register of
/// i16 lanes, two NEON ones).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) const TRANSPOSE_ROWS: usize = 16;
/// i16 lanes of a run's weight group (`Runs`): `PIXEL_OUTPUTS` taps of 4
/// lanes.
pub(crate) const RUN_GROUP_LANES: usize = 32;
/// Samples per pixel of the interleaved layout, RGB, and of the RGBX layout
/// its rows are widened to for the per-row kernels.
pub(crate) const RGB: usize = 3;
pub(crate) const RGBX: usize = 4;
/// Output pixels per pixel block at most, and pixels per tap-run group.
pub(crate) const PIXEL_OUTPUTS: usize = 8;
/// Pixels a pixel block's first taps may span.
const PIXEL_SPAN: usize = 16;
/// Pixel blocks this short (median outputs per block) leave tap runs ahead.
const SPARSE_RUN_GROUP_LANES: usize = 5;
/// Longer pixel blocks still lose to tap runs on windows of
/// `TAP_RUN_WINDOW_TAPS`+ taps.
const SHORT_RUN_GROUP_LANES: usize = 7;
const TAP_RUN_WINDOW_TAPS: usize = 12;
/// RGB windows from which the transposed path beats widening to RGBX.
#[cfg(target_arch = "x86_64")]
const RGB_TRANSPOSED_TAPS: usize = 6;

/// The convolution filters with a native path (`fast_image_resize`'s
/// definitions and supports).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Kernel {
    Box,
    Bilinear,
    CatmullRom,
    Lanczos3,
}

impl Kernel {
    pub(crate) fn parse(name: &str) -> Option<Kernel> {
        match name {
            "box" => Some(Kernel::Box),
            "bilinear" => Some(Kernel::Bilinear),
            "catmullrom" => Some(Kernel::CatmullRom),
            "lanczos3" => Some(Kernel::Lanczos3),
            _ => None,
        }
    }

    fn support(self) -> f64 {
        match self {
            Kernel::Box => 0.5,
            Kernel::Bilinear => 1.0,
            Kernel::CatmullRom => 2.0,
            Kernel::Lanczos3 => 3.0,
        }
    }

    fn weight(self, x: f64) -> f64 {
        match self {
            Kernel::Box => f64::from(u8::from(x > -0.5 && x <= 0.5)),
            Kernel::Bilinear => (1.0 - x.abs()).max(0.0),
            Kernel::CatmullRom => {
                let (keys_a, x) = (-0.5, x.abs());
                if x < 1.0 {
                    ((keys_a + 2.0) * x - (keys_a + 3.0)) * x * x + 1.0
                } else if x < 2.0 {
                    (((x - 5.0) * x + 8.0) * x - 4.0) * keys_a
                } else {
                    0.0
                }
            }
            Kernel::Lanczos3 if x.abs() < 3.0 => sinc(x) * sinc(x / 3.0),
            Kernel::Lanczos3 => 0.0,
        }
    }
}

/// sin and cos of pi*spacing and pi*spacing/3: the per-tap rotations of the
/// two sines in lanczos3 for taps `spacing` apart.
struct Rotation {
    spacing: f64,
    full: (f64, f64),
    third: (f64, f64),
}

impl Rotation {
    fn new(spacing: f64) -> Rotation {
        let angle = spacing * std::f64::consts::PI;
        Rotation {
            spacing,
            full: angle.sin_cos(),
            third: (angle / 3.0).sin_cos(),
        }
    }
}

/// lanczos3 at `x0, x0 + spacing, ...` (`taps` taps): sin(pi x) and
/// sin(pi x / 3) advance by angle addition, so a window costs two `sin_cos`
/// calls instead of two sines per tap.
fn lanczos3_weights(x0: f64, step: &Rotation, taps: usize, out: &mut Vec<f64>) {
    let pi = std::f64::consts::PI;
    let rotate = |(sine, cosine): (f64, f64), (sine_step, cosine_step): (f64, f64)| {
        (
            sine * cosine_step + cosine * sine_step,
            cosine * cosine_step - sine * sine_step,
        )
    };
    let (mut full, mut third) = ((x0 * pi).sin_cos(), (x0 * pi / 3.0).sin_cos());
    for tap in 0..taps {
        let x = x0 + tap as f64 * step.spacing;
        out.push(if x.abs() >= 3.0 {
            0.0
        } else if x.abs() < ZERO_WEIGHT_EPS {
            1.0 // sin(pi x) / (pi x) at its removable singularity
        } else {
            3.0 * full.0 * third.0 / (pi * pi * x * x)
        });
        full = rotate(full, step.full);
        third = rotate(third, step.third);
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let angle = x * std::f64::consts::PI;
        angle.sin() / angle
    }
}

/// One output's taps: first input index and normalized weights, as Q15 for
/// the SIMD kernels, as unclamped Q15-in-i32 for the scalar path, and as
/// f32 for the float32 engine (`convolve_float`). Renormalizing a clipped
/// edge window can push a weight past 1.0 (1.13 for lanczos3 when
/// upscaling), which i16 Q15 cannot hold.
#[derive(Clone, Copy)]
pub(crate) struct Window<'a> {
    pub(crate) start: usize,
    pub(crate) q15: &'a [i16],
    q32: &'a [i32],
    pub(crate) f32: &'a [f32],
    fits_q15: bool,
}

impl Window<'_> {
    /// Whether the SIMD kernels may use `q15` (a weight of exactly 1.0 is
    /// stored as 32767: at most 1/64 LSB off).
    pub(crate) fn fits_q15(&self) -> bool {
        self.fits_q15
    }

    pub(crate) fn end(&self) -> usize {
        self.start + self.q15.len()
    }

    /// Tap sum over `taps` (the window's inputs) with the unclamped weights,
    /// each product rounded as `mulhrs` rounds.
    fn dot(&self, taps: impl Iterator<Item = i32>) -> i32 {
        taps.zip(self.q32)
            .map(|(sample, &weight)| (sample * weight + (1 << (Q15_BITS - 1))) >> Q15_BITS)
            .sum()
    }

    /// The window's inputs in `samples`, input x at `x * step + offset`.
    fn inputs<'s>(
        &self,
        samples: &'s [i16],
        step: usize,
        offset: usize,
    ) -> impl Iterator<Item = i32> + 's + use<'s> {
        (self.start..self.end()).map(move |x| i32::from(samples[x * step + offset]))
    }

    /// `dot` over the window's inputs in `samples` (see `inputs`).
    fn dot_at(&self, samples: &[i16], step: usize, offset: usize) -> i32 {
        self.dot(self.inputs(samples, step, offset))
    }

    /// The SIMD kernels' sum (Q15 weights, `mulhrs`, i16 wraparound adds)
    /// over the window's inputs in `samples` (see `inputs`).
    fn mulhrs_dot_at(&self, samples: &[i16], step: usize, offset: usize) -> i32 {
        self.inputs(samples, step, offset)
            .zip(self.q15)
            .map(|(sample, &weight)| mulhrs(sample, weight))
            .sum()
    }
}

/// Every output's window along one axis, the weights of all of them in
/// shared buffers (one allocation per buffer, not per output).
pub(crate) struct Windows {
    starts: Vec<usize>,
    /// Output `output`'s weights are `q15[offsets[output]..offsets[output + 1]]`.
    offsets: Vec<usize>,
    q15: Vec<i16>,
    q32: Vec<i32>,
    f32: Vec<f32>,
    fits_q15: Vec<bool>,
    /// Whether every window passes `q15_sums_fit` (false only for windows of
    /// tens of thousands of taps).
    sums_fit: bool,
}

/// The range the vertical pass's Q6 sums of a SIMD window stay in (and the
/// scalar path's exact sums of the other windows, whose filters overshoot
/// less than this), which the horizontal pass then reads.
const MID_RANGE: (i32, i32) = (-6144, 22528);

/// The most negative and most positive partial sums of `mulhrs(sample, w)`
/// over `weights` in tap order, for samples in `low..=high`.
fn partial_sum_extremes(weights: &[i16], (low, high): (i32, i32)) -> (i32, i32) {
    let mulhrs_wide = |sample: i32, weight: i16| (sample * i32::from(weight) + (1 << 14)) >> 15;
    let (mut least, mut most) = (0, 0);
    let (mut lowest, mut highest) = (0, 0);
    for &weight in weights {
        let (from_low, from_high) = (mulhrs_wide(low, weight), mulhrs_wide(high, weight));
        least += from_low.min(from_high);
        most += from_low.max(from_high);
        (lowest, highest) = (lowest.min(least), highest.max(most));
    }
    (lowest, highest)
}

/// Whether the SIMD kernels' i16 sums of this window's Q15 terms stay in
/// range in both passes: the vertical pass's (of Q6 u8 samples) within
/// `MID_RANGE`, the horizontal pass's (of samples in `MID_RANGE`) within
/// i16. So their wrapping (x86, NEON) and saturating (NEON RDM) sums agree,
/// and neither wraps. Only windows of thousands of taps fail (each term's
/// rounding adds up); those take the scalar path's exact sums.
fn q15_sums_fit(weights: &[i16]) -> bool {
    let top = i32::from(u8::MAX) << MID_FRAC_BITS;
    let vertical = partial_sum_extremes(weights, (0, top));
    let horizontal = partial_sum_extremes(weights, MID_RANGE);
    vertical.0 >= MID_RANGE.0
        && vertical.1 <= MID_RANGE.1
        && horizontal.0 >= i32::from(i16::MIN)
        && horizontal.1 <= i32::from(i16::MAX)
}

impl Windows {
    /// `fast_image_resize`'s windows for `n_in -> n_out`.
    #[cfg(test)]
    pub(crate) fn new(n_in: usize, n_out: usize, kernel: Kernel) -> Windows {
        Windows::over(n_in, n_in, n_out, kernel)
    }

    /// The windows of `extent -> n_out` (the extent sets the scale), clipped
    /// to the `n_in` samples there are: past them a window keeps the taps
    /// inside, renormalized, as at any edge.
    pub(crate) fn over(n_in: usize, extent: usize, n_out: usize, kernel: Kernel) -> Windows {
        let scale = extent as f64 / n_out as f64;
        let kernel_scale = scale.max(1.0);
        let radius = kernel.support() * kernel_scale;
        let mut all = Windows {
            starts: Vec::with_capacity(n_out),
            offsets: vec![0],
            q15: Vec::new(),
            q32: Vec::new(),
            f32: Vec::new(),
            fits_q15: Vec::with_capacity(n_out),
            sums_fit: true,
        };
        let mut weights = Vec::new();
        let step = Rotation::new(1.0 / kernel_scale);
        for output in 0..n_out {
            let in_center = (output as f64 + 0.5) * scale;
            let lo = (in_center - radius).floor().max(0.0) as usize;
            let hi = ((in_center + radius).ceil() as usize).min(n_in);
            weights.clear();
            let x0 = (lo as f64 - (in_center - 0.5)) / kernel_scale;
            match kernel {
                Kernel::Lanczos3 => lanczos3_weights(x0, &step, hi - lo, &mut weights),
                _ => weights.extend(
                    (lo..hi).map(|x| kernel.weight((x as f64 - (in_center - 0.5)) / kernel_scale)),
                ),
            }
            // Lanczos3's sine recurrence leaves ~1e-17 where its zeros fall on
            // a sample: exact zeros, and zero taps at either end dropped (they
            // add nothing but work).
            for weight in weights
                .iter_mut()
                .filter(|weight| weight.abs() < ZERO_WEIGHT_EPS)
            {
                *weight = 0.0;
            }
            let first = weights
                .iter()
                .position(|&weight| weight != 0.0)
                .unwrap_or(0);
            let last = weights
                .iter()
                .rposition(|&weight| weight != 0.0)
                .map_or(weights.len(), |index| index + 1);
            let lo = lo + first;
            weights.truncate(last);
            weights.drain(..first);
            let sum: f64 = weights.iter().sum();
            let first = all.q32.len();
            for weight in &weights {
                all.q15.push(q15((weight / sum) as f32));
                all.q32.push((weight / sum * Q15_ONE).round() as i32);
                all.f32.push((weight / sum) as f32);
            }
            all.starts.push(lo);
            all.offsets.push(all.q15.len());
            let sums_fit = q15_sums_fit(&all.q15[first..]);
            all.sums_fit &= sums_fit;
            all.fits_q15.push(
                all.q32[first..]
                    .iter()
                    .all(|weight| (i32::from(i16::MIN)..=-i32::from(i16::MIN)).contains(weight))
                    && sums_fit,
            );
        }
        all
    }

    pub(crate) fn len(&self) -> usize {
        self.starts.len()
    }

    /// The windows cut into chunks of at most `taps` taps: chunk k holds taps
    /// `k * taps..(k + 1) * taps` of every window (none where a window is
    /// shorter), so the chunks' sums add up to the window's. None if no window
    /// is longer than `taps`.
    pub(crate) fn chunks(&self, taps: usize) -> Option<Vec<Windows>> {
        let widest = self.iter().map(|window| window.q15.len()).max()?;
        if widest <= taps {
            return None;
        }
        let chunk = |index: usize| {
            let mut part = Windows {
                starts: Vec::with_capacity(self.len()),
                offsets: vec![0],
                q15: Vec::new(),
                q32: Vec::new(),
                f32: Vec::new(),
                fits_q15: vec![true; self.len()],
                sums_fit: true,
            };
            for window in self.iter() {
                let len = window.q15.len();
                let span = (index * taps).min(len)..((index + 1) * taps).min(len);
                part.starts.push(window.start + span.start);
                part.q15.extend_from_slice(&window.q15[span.clone()]);
                part.q32.extend_from_slice(&window.q32[span.clone()]);
                part.f32.extend_from_slice(&window.f32[span]);
                part.offsets.push(part.q15.len());
            }
            part
        };
        Some((0..widest.div_ceil(taps)).map(chunk).collect())
    }

    /// Whether every window's SIMD sums stay in range (`q15_sums_fit`).
    pub(crate) fn sums_fit(&self) -> bool {
        self.sums_fit
    }

    pub(crate) fn get(&self, output: usize) -> Window<'_> {
        let taps = self.offsets[output]..self.offsets[output + 1];
        Window {
            start: self.starts[output],
            q15: &self.q15[taps.clone()],
            q32: &self.q32[taps.clone()],
            f32: &self.f32[taps],
            fits_q15: self.fits_q15[output],
        }
    }

    /// Every window in order: zipped over the buffers rather than built by
    /// `get` per output, since the transposed kernels walk all of them per
    /// batch and short windows leave little else to amortize the lookups.
    pub(crate) fn iter(&self) -> impl Iterator<Item = Window<'_>> {
        let spans = self.offsets.windows(2).map(|span| span[0]..span[1]);
        self.starts
            .iter()
            .zip(spans)
            .zip(&self.fits_q15)
            .map(|((&start, taps), &fits_q15)| Window {
                start,
                q15: &self.q15[taps.clone()],
                q32: &self.q32[taps.clone()],
                f32: &self.f32[taps],
                fits_q15,
            })
    }
}

/// The widest windows `PaddedWindows` pads.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const MAX_PADDED_TAPS: usize = 20;
/// Samples past a row the padded windows' zero-weight taps read: a window
/// starts at most at the row's last pixel, of at most 3 samples.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const PADDED_READS_PAST_ROW: usize = (MAX_PADDED_TAPS - 1) * RGB;

/// Column windows padded with zero-weight taps to the widest one that fits
/// Q15 (at most `MAX_PADDED_TAPS`), so the transposed horizontal pass has a
/// constant trip count per output: a zero weight adds exactly 0. Windows
/// past Q15 get zero weights (the scalar path writes those outputs).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) struct PaddedWindows {
    taps: usize,
    starts: Vec<usize>,
    /// `taps` weights per output.
    weights: Vec<i16>,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl PaddedWindows {
    /// The padded windows of `cols`, or None where one is wider than
    /// `MAX_PADDED_TAPS`.
    pub(crate) fn new(cols: &Windows) -> Option<PaddedWindows> {
        let widest = cols
            .iter()
            .filter(|col| col.fits_q15())
            .map(|col| col.q15.len())
            .max()
            .unwrap_or(0);
        if widest > MAX_PADDED_TAPS {
            return None;
        }
        let taps = widest.max(1);
        let mut padded = PaddedWindows {
            taps,
            starts: Vec::with_capacity(cols.len()),
            weights: vec![0; taps * cols.len()],
        };
        for (col, weights) in cols.iter().zip(padded.weights.chunks_exact_mut(taps)) {
            padded.starts.push(col.start);
            if col.fits_q15() {
                weights[..col.q15.len()].copy_from_slice(col.q15);
            }
        }
        Some(padded)
    }
}

/// Per-output runs over the taps (large reductions): per output, its
/// window's inputs a group of `PIXEL_OUTPUTS` pixels (4 lanes each) at a
/// time times its weights, the lanes then summed to one.
#[cfg_attr(
    not(target_arch = "x86_64"),
    expect(
        dead_code,
        reason = "read by the x86 kernels; the scalar path has no use"
    )
)]
pub(crate) struct Runs {
    /// The outputs handled (every output whose weights fit Q15), ascending.
    pub(crate) outputs: Vec<usize>,
    /// Each output's first input.
    pub(crate) starts: Vec<usize>,
    /// Groups of taps per output.
    pub(crate) groups: usize,
    /// Per output, per group: tap t's weight in the group's lanes for t (0
    /// past the window).
    pub(crate) weights: Vec<[i16; RUN_GROUP_LANES]>,
    /// Inputs (samples or pixels, as `width`) a group may read past the row end.
    overreach: usize,
}

impl Runs {
    fn new(cols: &Windows, width: usize) -> Runs {
        let group_taps = PIXEL_OUTPUTS;
        let lanes = RUN_GROUP_LANES / group_taps;
        let outputs: Vec<usize> = (0..cols.len())
            .filter(|&output| cols.get(output).fits_q15())
            .collect();
        let taps = outputs
            .iter()
            .map(|&output| cols.get(output).q15.len())
            .max()
            .unwrap_or(0);
        let groups = taps.div_ceil(group_taps).max(1);
        let mut weights = vec![[0; RUN_GROUP_LANES]; outputs.len() * groups];
        for (index, &output) in outputs.iter().enumerate() {
            for (tap, &weight) in cols.get(output).q15.iter().enumerate() {
                let slot = tap % group_taps;
                weights[index * groups + tap / group_taps][lanes * slot..lanes * (slot + 1)]
                    .fill(weight);
            }
        }
        let starts: Vec<usize> = outputs
            .iter()
            .map(|&output| cols.get(output).start)
            .collect();
        let read_end = starts
            .iter()
            .map(|&start| start + groups * group_taps)
            .max()
            .unwrap_or(0);
        Runs {
            outputs,
            starts,
            groups,
            weights,
            overreach: read_end.saturating_sub(width),
        }
    }
}

/// Outputs per pixel block: runs of up to `PIXEL_OUTPUTS` outputs whose
/// first taps lie within `PIXEL_SPAN` pixels, the blocks a per-row pixel
/// kernel would gather (outputs whose weights exceed Q15 belong to none).
/// How densely the windows pack picks tap runs (`TapRuns::pay_off`).
fn pixel_block_outputs(cols: &Windows) -> Vec<usize> {
    let mut counts = Vec::new();
    let mut output = 0;
    while output < cols.len() {
        if !cols.get(output).fits_q15() {
            output += 1;
            continue;
        }
        let base = cols.get(output).start;
        // A window starting before `base` (trimmed zero taps can start a
        // window after the next one's) ends the block.
        let fits = |slot: usize| {
            output + slot < cols.len()
                && cols.get(output + slot).fits_q15()
                && cols
                    .get(output + slot)
                    .start
                    .checked_sub(base)
                    .is_some_and(|offset| offset < PIXEL_SPAN)
        };
        let count = (0..PIXEL_OUTPUTS).take_while(|&slot| fits(slot)).count();
        counts.push(count);
        output += count;
    }
    counts
}

/// The RGB horizontal pass for large reductions, vectorized over each
/// output's taps instead of across outputs: `Runs` of `PIXEL_OUTPUTS`
/// pixels per group. Pixel blocks fill their lanes only while several
/// outputs' taps fit `PIXEL_SPAN` pixels; this costs the same per tap at any
/// ratio. Used where blocks run short of outputs (`pay_off`). It gives the
/// scalar reference's result.
#[cfg_attr(
    not(target_arch = "x86_64"),
    expect(
        dead_code,
        reason = "read by the x86 kernels; the scalar path has no use"
    )
)]
pub(crate) struct TapRuns {
    pub(crate) runs: Runs,
    /// Pixel blocks would hold fewer than `SPARSE_RUN_GROUP_LANES` outputs.
    pub(crate) sparse_blocks: bool,
}

impl TapRuns {
    /// Whether tap runs pay off for these windows, whose middle pixel block
    /// holds `median` outputs (blocks are uniform away from the edges).
    fn pay_off(cols: &Windows, median: Option<usize>) -> bool {
        let Some(median) = median else {
            return false;
        };
        let taps = cols.get(cols.len() / 2).q15.len();
        median < SPARSE_RUN_GROUP_LANES
            || (median < SHORT_RUN_GROUP_LANES && taps >= TAP_RUN_WINDOW_TAPS)
    }

    fn new(cols: &Windows, width: usize, median: Option<usize>) -> TapRuns {
        let sparse_blocks = median.is_some_and(|median| median < SPARSE_RUN_GROUP_LANES);
        TapRuns {
            runs: Runs::new(cols, width),
            sparse_blocks,
        }
    }
}

/// Everything precomputed for one resize.
#[cfg_attr(
    not(target_arch = "x86_64"),
    expect(
        dead_code,
        reason = "read by the x86 kernels; the scalar path has no use"
    )
)]
pub(crate) struct Plan {
    rows: Windows,
    cols: Windows,
    /// Windows too wide for consecutive outputs to share a short span of
    /// inputs (`is_large_reduction`).
    large_reduction: bool,
    /// Samples per pixel: 1 or RGB.
    channels: usize,
    /// Some for RGB reductions where tap runs pay off.
    tap_runs: Option<TapRuns>,
    /// Outputs whose weights exceed Q15, done by the scalar path.
    scalar_cols: Vec<usize>,
    /// The most taps of any column window.
    widest_col: usize,
    /// The column windows padded for the transposed horizontal pass, where
    /// none is wider than it pads to.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    padded_cols: Option<PaddedWindows>,
    n_in: usize,
}

/// A resize's sizes, each (width, height): the image's, the extent that
/// sets the scale, and the output's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Geometry {
    pub(crate) src: (usize, usize),
    pub(crate) extent: (usize, usize),
    pub(crate) dst: (usize, usize),
}

impl Geometry {
    /// `src` resampled to `dst`: the image is the extent.
    pub(crate) const fn resize(src: (usize, usize), dst: (usize, usize)) -> Geometry {
        Geometry {
            src,
            extent: src,
            dst,
        }
    }

    /// `src` reduced by exactly `factor` per axis: `ceil(n / factor)`
    /// outputs, the last one covering what remains of the image when the
    /// factor doesn't divide it.
    pub(crate) const fn downscale((width, height): (usize, usize), factor: usize) -> Geometry {
        let dst = (width.div_ceil(factor), height.div_ceil(factor));
        Geometry {
            src: (width, height),
            extent: (dst.0 * factor, dst.1 * factor),
            dst,
        }
    }

    /// Whether the scale is the image's own (no partial last output).
    pub(crate) fn fits_image(&self) -> bool {
        self.extent == self.src
    }

    /// Whether either axis is reduced by more than `ratio` (input pixels per
    /// output pixel).
    pub(crate) fn reduces_past(&self, ratio: usize) -> bool {
        self.extent.0 > ratio * self.dst.0 || self.extent.1 > ratio * self.dst.1
    }
}

impl Plan {
    /// The native plan for `(width, height) -> (ow, oh)` of a
    /// `channels`-sample image: the windows, and which per-row kernels the
    /// tier has for them.
    #[cfg(test)]
    pub(crate) fn new(
        src: (usize, usize),
        dst: (usize, usize),
        kernel: Kernel,
        channels: usize,
    ) -> Plan {
        Plan::over(Geometry::resize(src, dst), kernel, channels)
    }

    /// `Plan::new` for any `geometry`.
    pub(crate) fn over(geometry: Geometry, kernel: Kernel, channels: usize) -> Plan {
        debug_assert!([1, RGB].contains(&channels));
        let Geometry {
            src: (width, height),
            extent: (extent_width, extent_height),
            dst: (ow, oh),
        } = geometry;
        let (cols, rows) = (
            Windows::over(width, extent_width, ow, kernel),
            Windows::over(height, extent_height, oh, kernel),
        );
        let scalar_cols: Vec<usize> = (0..ow)
            .filter(|&output| !cols.get(output).fits_q15())
            .collect();
        let widest_col = cols.iter().map(|col| col.q15.len()).max().unwrap_or(0);
        let large_reduction = is_large_reduction(extent_width, ow, kernel);
        let tap_runs = (channels > 1)
            .then(|| {
                let block_outputs = pixel_block_outputs(&cols);
                block_outputs.get(block_outputs.len() / 2).copied()
            })
            .filter(|&median| TapRuns::pay_off(&cols, median))
            .map(|median| TapRuns::new(&cols, width, median));
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let padded_cols = PaddedWindows::new(&cols);
        Plan {
            rows,
            cols,
            large_reduction,
            channels,
            tap_runs,
            scalar_cols,
            widest_col,
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            padded_cols,
            n_in: width,
        }
    }
}

/// A resize's geometry, filter and channels.
pub(crate) type PlanKey = (Geometry, Kernel, usize);

/// The last few plans built on this thread, most recent first: building one
/// evaluates the kernel per weight, a large share of a small resize, and
/// tile workloads repeat the same few geometries.
pub(crate) struct PlanCache<P> {
    recent: Vec<(PlanKey, Arc<P>)>,
}

impl<P> PlanCache<P> {
    const CAPACITY: usize = 8;

    pub(crate) const fn new() -> Self {
        PlanCache { recent: Vec::new() }
    }

    pub(crate) fn get_or_build(&mut self, key: PlanKey, build: impl FnOnce() -> P) -> Arc<P> {
        let plan = match self
            .recent
            .iter()
            .position(|(cached_key, _)| *cached_key == key)
        {
            Some(index) => self.recent.remove(index).1,
            None => Arc::new(build()),
        };
        self.recent.truncate(Self::CAPACITY - 1);
        self.recent.insert(0, (key, plan.clone()));
        plan
    }
}

thread_local! {
    static PLANS: RefCell<PlanCache<Plan>> = const { RefCell::new(PlanCache::new()) };
}

impl Plan {
    /// `Plan::new`, reusing this thread's recent plans.
    pub(crate) fn cached(geometry: Geometry, kernel: Kernel, channels: usize) -> Arc<Plan> {
        let key = (geometry, kernel, channels);
        PLANS.with_borrow_mut(|plans| {
            plans.get_or_build(key, || Plan::over(geometry, kernel, channels))
        })
    }
}

/// Whether `width -> ow` is a large reduction, judged from the widest
/// possible window: at an integer factor 3 or 4, windows past 20 taps; at
/// factor 2 never; at other ratios, windows such that 10 consecutive outputs
/// read more than 64 inputs. Past it RGB takes the transposed path even where
/// the tier has per-row tap runs (`runs_transposed`).
fn is_large_reduction(width: usize, ow: usize, kernel: Kernel) -> bool {
    const SPAN_INPUTS: f64 = 64.0;
    const SPAN_OUTPUTS: f64 = 10.0;
    const MAX_TAPS_FACTOR_3_PLUS: usize = 20;
    let scale = width as f64 / ow as f64;
    let max_taps = 2 * (kernel.support() * scale.max(1.0)).ceil() as usize + 1;
    let factor = width / ow;
    if width == factor * ow && (2..=4).contains(&factor) {
        // max_taps bounds the window from above: one past the widest a
        // window at an integer factor gets.
        return factor >= 3 && max_taps > MAX_TAPS_FACTOR_3_PLUS + 1;
    }
    (SPAN_OUTPUTS - 1.0) * scale + max_taps as f64 > SPAN_INPUTS
}

/// Resize a u8 image of `plan.channels` samples per pixel by `plan`.
/// Every path computes the same fixed-point arithmetic (per-tap rounded
/// mulhrs, i16 sums, one final rounding), so the result is identical on
/// every CPU.
pub(crate) fn resize_u8(src: &[u8], (width, height): (usize, usize), plan: &Plan) -> Vec<u8> {
    let channels = plan.channels;
    debug_assert!(src.len() >= width * height * channels && plan.n_in == width);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if takes_transposed(plan, width) {
        return resize_transposed(src, width, channels, plan);
    }
    if channels > 1 {
        resize_pixel_rows(src, width, plan)
    } else {
        resize_rows(src, width, plan)
    }
}

/// Whether a resize takes the transposed path (else the per-row kernels;
/// either computes the same result). Its vertical pass takes rows of a
/// register or more. On aarch64 every per-row kernel is scalar.
///
/// Windows too long for the SIMD sums (`q15_sums_fit`) take the per-row
/// path, which computes them as the reference does: in the transposed path
/// their intermediate sums can leave the range its kernels assume, and a
/// clamped column sum near `i16::MAX` would wrap in the transposed narrowing.
#[cfg(target_arch = "x86_64")]
fn takes_transposed(plan: &Plan, width: usize) -> bool {
    crate::cpu::has_avx2()
        && width * plan.channels >= AVX2_I16_LANES
        && plan.rows.sums_fit()
        && plan.cols.sums_fit()
        && runs_transposed(plan)
}
#[cfg(target_arch = "aarch64")]
fn takes_transposed(plan: &Plan, width: usize) -> bool {
    crate::cpu::use_neon()
        && width * plan.channels >= TRANSPOSE_ROWS
        && plan.rows.sums_fit()
        && plan.cols.sums_fit()
}

/// On x86, whether the transposed path beats the per-row kernels the tier
/// has.
#[cfg(target_arch = "x86_64")]
fn runs_transposed(plan: &Plan) -> bool {
    // Large reductions (no block plan): transposed, which needs no widening.
    // RGB with wide windows: widening to RGBX costs the pixel kernels more
    // than the transposes.
    let unwidened = plan.channels == RGB && plan.widest_col >= RGB_TRANSPOSED_TAPS;
    // Transposed unless the tier has a per-row kernel: tap runs, for sparse
    // blocks only, where they pay for the widening and packing.
    let avx2_per_row = plan
        .tap_runs
        .as_ref()
        .is_some_and(|tap_runs| tap_runs.sparse_blocks);
    plan.large_reduction || unwidened || !avx2_per_row
}

/// The transposed path (AVX2 instructions, run on either tier). Without a
/// two-register 16-bit permute for the block gather, it batches
/// `TRANSPOSE_ROWS` output rows, transposes their intermediate rows, and runs
/// the horizontal pass as a vertical one: output column o of all the rows =
/// sum over its taps of (unit-stride load of the batch) x (broadcast weight),
/// for any ratio. Then transposes back. With `channels` samples per pixel,
/// each output sample's taps are `channels` samples apart.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn resize_transposed(src: &[u8], width: usize, channels: usize, plan: &Plan) -> Vec<u8> {
    #[cfg(target_arch = "aarch64")]
    use crate::convolve::neon::{
        horizontal_padded_neon as horizontal_padded,
        horizontal_transposed_neon as horizontal_transposed,
        transpose_rows_i16_neon as transpose_rows, untranspose_narrow_neon as untranspose_narrow,
    };
    #[cfg(target_arch = "x86_64")]
    use crate::convolve::x86::{
        horizontal_padded_avx2 as horizontal_padded,
        horizontal_transposed_avx2 as horizontal_transposed,
        transpose_rows_i16_avx2 as transpose_rows, untranspose_narrow_avx2 as untranspose_narrow,
    };
    // mid's row stride: whole cache lines plus half of one, so the batch's
    // rows don't share L1 sets (as a power-of-2 stride would).
    const LINE_I16: usize = 32;
    const ROWS: usize = TRANSPOSE_ROWS;
    let (ow, oh) = (plan.cols.len(), plan.rows.len());
    // Samples per row.
    let (ws, ows) = (width * channels, ow * channels);
    let wp = ws.next_multiple_of(2 * LINE_I16) + LINE_I16;
    let owp = ows.next_multiple_of(ROWS);
    let mut out = vec![0u8; ows * oh];
    // Scratch kept per thread between calls (see `TransposeScratch`). A call
    // writes everything it uses (the row padding passes through the
    // transposes unused).
    let TransposeScratch {
        mut mid,
        mut mid_t,
        mut out_t,
    } = SCRATCH.with_borrow_mut(std::mem::take);
    mid.resize(ROWS * wp, 0); // ROWS rows, stride wp
    // mid_t[x * ROWS + row] = mid[row][x], plus the samples the padded
    // windows' zero-weight taps read past the row.
    mid_t.resize((wp + PADDED_READS_PAST_ROW) * ROWS, 0);
    out_t.resize(owp * ROWS, 0); // Q6 sums, out_t[output * ROWS + row]
    let dp = out.as_mut_ptr();
    let batches = |lo: usize, hi: usize| {
        for batch in (lo..hi).step_by(ROWS) {
            let rows = ROWS.min(hi - batch);
            for row in 0..ROWS {
                // Short batch: repeat its last row (those outputs are dropped).
                let win = plan.rows.get((batch + row).min(hi - 1));
                vertical(
                    &src[win.start * ws..win.end() * ws],
                    ws,
                    win,
                    &mut mid[row * wp..row * wp + ws],
                );
            }
            // SAFETY: the tier was checked (takes_transposed); buffers sized
            // ROWS x wp / owp x ROWS.
            unsafe {
                transpose_rows(&mid, wp, &mut mid_t);
                match &plan.padded_cols {
                    Some(padded) => horizontal_padded(&mid_t, padded, channels, &mut out_t),
                    None => horizontal_transposed(&mid_t, &plan.cols, channels, &mut out_t),
                }
            }
            for &output in &plan.scalar_cols {
                let col = plan.cols.get(output);
                for (ch, row) in (0..channels).flat_map(|ch| (0..ROWS).map(move |row| (ch, row))) {
                    // Clamped, which narrows as the reference's i32 does.
                    let sum = col.dot_at(&mid_t, channels * ROWS, ch * ROWS + row);
                    out_t[(output * channels + ch) * ROWS + row] =
                        sum.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16;
                }
            }
            // SAFETY: rows [batch, batch + rows) of `out`, which only this
            // closure call writes (with_prefault's contract).
            let dst = unsafe { std::slice::from_raw_parts_mut(dp.add(batch * ows), rows * ows) };
            // SAFETY: the tier was checked; out_t holds owp x ROWS, dst rows x ows.
            unsafe { untranspose_narrow(&out_t, ows, dst, rows) };
        }
    };
    // SAFETY: `out` is this function's, and `batches(lo, hi)` writes only
    // rows [lo, hi) (whole rows, no slop).
    unsafe { crate::cpu::with_prefault(dp, ows * oh, ows, oh, batches) };
    TransposeScratch { mid, mid_t, out_t }.keep_in(&SCRATCH);
    out
}

/// The transposed paths' buffers (u8 in Q6 i16 here, float in
/// `convolve_float`), kept per thread between calls: fresh, a tile's call
/// would clear (or fault in) them every time. The intermediate rows, the
/// same transposed, and the transposed output sums.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[derive(Default)]
pub(crate) struct TransposeScratch<T> {
    pub(crate) mid: Vec<T>,
    pub(crate) mid_t: Vec<T>,
    pub(crate) out_t: Vec<T>,
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
impl<T: 'static> TransposeScratch<T> {
    /// Scratch kept per thread up to this size, so a very wide image doesn't
    /// pin its buffers to the thread.
    const KEPT_BYTES: usize = 8 << 20;

    pub(crate) const fn new() -> Self {
        Self {
            mid: Vec::new(),
            mid_t: Vec::new(),
            out_t: Vec::new(),
        }
    }

    /// Put the buffers back in the thread's `slot` for the next call, unless
    /// they are over `KEPT_BYTES`.
    pub(crate) fn keep_in(self, slot: &'static std::thread::LocalKey<RefCell<Self>>) {
        let capacity = self.mid.capacity() + self.mid_t.capacity() + self.out_t.capacity();
        if capacity * size_of::<T>() <= Self::KEPT_BYTES {
            slot.with_borrow_mut(|kept| *kept = self);
        }
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
thread_local! {
    static SCRATCH: RefCell<TransposeScratch<i16>> = const { RefCell::new(TransposeScratch::new()) };
}

/// The per-row planar path (the scalar reference).
fn resize_rows(src: &[u8], width: usize, plan: &Plan) -> Vec<u8> {
    let ow = plan.cols.len();
    let mut out = vec![0u8; ow * plan.rows.len()];
    let mut mid = vec![0i16; width];
    let dp = out.as_mut_ptr();
    let rows = |lo: usize, hi: usize| {
        for y in lo..hi {
            let win = plan.rows.get(y);
            vertical(
                &src[win.start * width..win.end() * width],
                width,
                win,
                &mut mid,
            );
            // SAFETY: row y of `out`, which only this closure call writes
            // (with_prefault's contract).
            let dst = unsafe { std::slice::from_raw_parts_mut(dp.add(y * ow), ow) };
            horizontal_scalar(plan, &mid, dst, 1);
        }
    };
    // SAFETY: `out` is this function's; `rows(lo, hi)` writes rows [lo, hi).
    unsafe { crate::cpu::with_prefault(dp, out.len(), ow, plan.rows.len(), rows) };
    out
}

/// The per-row RGB path (the AVX2 tap runs, or the scalar reference): rows
/// widened to the RGBX layout between the passes and the output rows packed
/// back.
fn resize_pixel_rows(src: &[u8], width: usize, plan: &Plan) -> Vec<u8> {
    // Padded: the widening reads whole groups of pixels, up to a register past.
    const WIDEN_GROUP_PIXELS: usize = 8;
    let channels = plan.channels;
    let ow = plan.cols.len();
    let ws = width * channels;
    let ows = ow * channels;
    let mut out = vec![0u8; ows * plan.rows.len()];
    // Padded so the SIMD kernels' windows may run past the row end.
    let row_pad = plan
        .tap_runs
        .as_ref()
        .map_or(0, |tap_runs| tap_runs.runs.overreach);
    let padded = (width + row_pad) * RGBX;
    let mut mids = [vec![0i16; padded], vec![0i16; padded]];
    // Rows widened to the RGBX layout between the passes (from
    // `narrow_mid`), output rows packed back (from `rgbx_rows`).
    let narrow_pad = channels * width.next_multiple_of(WIDEN_GROUP_PIXELS) + AVX2_I16_LANES;
    let mut narrow_mid = vec![0i16; narrow_pad];
    let mut rgbx_rows = [(); 2].map(|()| vec![0u8; ow * RGBX]);
    let dp = out.as_mut_ptr();
    let rows = |lo: usize, hi: usize| {
        for y in (lo..hi).step_by(2) {
            let pair_len = 2.min(hi - y);
            for (y, mid) in (y..y + pair_len).zip(mids.iter_mut()) {
                let win = plan.rows.get(y);
                vertical(
                    &src[win.start * ws..win.end() * ws],
                    ws,
                    win,
                    &mut narrow_mid[..ws],
                );
                widen_to_rgbx(&narrow_mid, width, mid);
            }
            let [rgbx0, rgbx1] = &mut rgbx_rows;
            let (rgbx_row0, rgbx_row1) = (rgbx0.as_mut_ptr(), rgbx1.as_mut_ptr());
            // SAFETY: the two RGBX row buffers, each written by one row at a
            // time.
            let dst = |y: usize| unsafe {
                let row = if y.is_multiple_of(2) {
                    rgbx_row0
                } else {
                    rgbx_row1
                };
                std::slice::from_raw_parts_mut(row, ow * RGBX)
            };
            #[cfg(target_arch = "x86_64")]
            let simd = horizontal_pixels_simd(plan, &mids, y, pair_len, dst);
            #[cfg(not(target_arch = "x86_64"))]
            let simd = false;
            for (y, mid) in (y..y + pair_len).zip(&mids) {
                let dst = dst(y);
                if simd {
                    narrow_scalar_cols(plan, mid, dst, RGBX);
                } else {
                    horizontal_scalar(plan, mid, dst, RGBX);
                }
                // SAFETY: row y of `out`, which only this closure call writes
                // (with_prefault's contract).
                pack_from_rgbx(dst, unsafe {
                    std::slice::from_raw_parts_mut(dp.add(y * ows), ows)
                });
            }
        }
    };
    // SAFETY: `out` is this function's; `rows(lo, hi)` writes rows [lo, hi).
    unsafe { crate::cpu::with_prefault(dp, out.len(), ows, plan.rows.len(), rows) };
    out
}

/// The x86 SIMD horizontal pass of output rows y .. y + `pair_len` (one or
/// two) of the per-row RGB path, from `mids`, into `dst(y)`: false where it
/// doesn't apply (the caller runs `horizontal_scalar`).
#[cfg(target_arch = "x86_64")]
fn horizontal_pixels_simd<'row>(
    plan: &Plan,
    mids: &[Vec<i16>; 2],
    y: usize,
    pair_len: usize,
    dst: impl Fn(usize) -> &'row mut [u8],
) -> bool {
    if let (Some(runs), true, 2) = (&plan.tap_runs, crate::cpu::has_avx2(), pair_len) {
        let [mid0, mid1] = mids;
        // SAFETY: avx2 detected; mids padded by the runs' overreach.
        unsafe {
            crate::convolve::x86::horizontal_taps_x2_avx2(
                [mid0, mid1],
                [dst(y), dst(y + 1)],
                &runs.runs,
            );
        };
        return true;
    }
    false
}

// ---- Pass kernels: dispatch + scalar reference -------------------------------

/// Intermediate RGB samples to RGBX-layout pixels (X 0) for the pixel
/// kernels: `rgbx[4p + c] = src[3p + c]`.
fn widen_to_rgbx(src: &[i16], pixels: usize, rgbx: &mut [i16]) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_avx2() {
        // SAFETY: avx2 detected; src padded past whole groups of 8 pixels.
        return unsafe { crate::convolve::x86::widen_to_rgbx_avx2(src, pixels, rgbx) };
    }
    let (rgb_pixels, _) = src[..RGB * pixels].as_chunks::<RGB>();
    let (rgbx_pixels, _) = rgbx.as_chunks_mut::<RGBX>();
    for (px, out) in rgb_pixels.iter().zip(rgbx_pixels) {
        *out = [px[0], px[1], px[2], 0];
    }
}

/// RGBX-layout output pixels to RGB: `out[3p + c] = rgbx[4p + c]`.
fn pack_from_rgbx(rgbx: &[u8], out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_avx2() {
        // SAFETY: avx2 detected; rgbx holds out.len() / 3 pixels.
        return unsafe { crate::convolve::x86::pack_from_rgbx_avx2(rgbx, out) };
    }
    let (rgbx_pixels, _) = rgbx.as_chunks::<RGBX>();
    let (rgb_pixels, _) = out.as_chunks_mut::<RGB>();
    for (px, out) in rgbx_pixels.iter().zip(rgb_pixels) {
        *out = [px[0], px[1], px[2]];
    }
}

/// Q6 sum back to u8: rounded, clamped (filters overshoot both ways).
#[inline]
fn narrow(acc: i32) -> u8 {
    ((acc + (1 << (MID_FRAC_BITS - 1))) >> MID_FRAC_BITS).clamp(0, 255) as u8
}

/// The scalar reference's outputs for the plan's scalar columns of one row
/// (`channels` samples per pixel), which the SIMD passes leave out.
fn narrow_scalar_cols(plan: &Plan, mid: &[i16], dst: &mut [u8], channels: usize) {
    for &output in &plan.scalar_cols {
        let col = plan.cols.get(output);
        for ch in 0..channels {
            dst[output * channels + ch] = narrow(col.dot_at(mid, channels, ch));
        }
    }
}

/// One row's horizontal pass by the scalar reference (`channels` samples per
/// pixel): the SIMD kernels' arithmetic where the weights fit Q15, exact
/// sums elsewhere.
fn horizontal_scalar(plan: &Plan, mid: &[i16], dst: &mut [u8], channels: usize) {
    for (output, col) in plan.cols.iter().enumerate() {
        for ch in 0..channels {
            let acc = if col.fits_q15() {
                col.mulhrs_dot_at(mid, channels, ch)
            } else {
                col.dot_at(mid, channels, ch)
            };
            dst[output * channels + ch] = narrow(acc);
        }
    }
}

/// `mid[x] = sum_t q15[t] * rows[t][x]` in Q6, `rows` = the window's source
/// rows.
fn vertical(rows: &[u8], width: usize, win: Window<'_>, mid: &mut [i16]) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_avx2() && win.fits_q15() && width >= AVX2_I16_LANES {
        // SAFETY: avx2 detected; rows holds q15.len() rows of width, mid width.
        return unsafe { crate::convolve::x86::vertical_q6_avx2(rows, width, win.q15, mid) };
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::use_neon() && win.fits_q15() && width >= TRANSPOSE_ROWS {
        // SAFETY: NEON is baseline; rows holds q15.len() rows of width, mid width.
        return unsafe { neon::vertical_q6_neon(rows, width, win.q15, mid) };
    }
    for (x, sum) in mid[..width].iter_mut().enumerate() {
        let taps = (0..win.q15.len()).map(|tap| i32::from(rows[tap * width + x]) << MID_FRAC_BITS);
        // SIMD-eligible windows: the Q15 weights and mulhrs, as the kernel
        // does (i16 wraparound == its per-term adds); others: exact i32.
        *sum = if win.fits_q15() {
            taps.zip(win.q15)
                .map(|(sample, &weight)| mulhrs(sample, weight))
                .sum::<i32>() as i16
        } else {
            win.dot(taps) as i16
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{CHANNELS, Rng, random_samples};

    const KERNELS: [Kernel; 4] = [
        Kernel::Box,
        Kernel::Bilinear,
        Kernel::CatmullRom,
        Kernel::Lanczos3,
    ];

    /// Column ratios (input width, output width) that reach every plan: the
    /// smallest images, ratios around the vector widths, integer factors 2 to
    /// 4, integer upscales x2 and x4, and large reductions.
    const COLUMN_CASES: [(usize, usize); 21] = [
        (1, 1),
        (1, 3),
        (3, 1),
        (2, 2),
        (7, 16),
        (16, 7),
        (15, 1),
        (17, 33),
        (31, 15),
        (33, 64),
        (64, 32),
        (130, 65),
        (201, 67),
        (268, 67),
        (130, 130),
        (64, 128),
        (33, 132),
        (130, 37),
        (130, 200),
        (100, 3),
        (640, 10),
    ];

    /// Row ratios: upscales (lanczos3's edge weights past Q15), odd output
    /// heights (a last row without a pair), a short last transposed batch,
    /// and windows long enough for the grouped vertical taps.
    const ROW_CASES: [(usize, usize); 7] = [
        (1, 1),
        (3, 7),
        (17, 9),
        (33, 16),
        (40, 5),
        (40, 37),
        (130, 2),
    ];

    /// Outputs a kernel must leave unwritten keep this value.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    const SENTINEL: u8 = 0xA5;
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    const SENTINEL_I16: i16 = i16::MIN;

    /// Rows per transposed batch.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    const ROWS: usize = TRANSPOSE_ROWS;

    /// The vertical pass and the transposed path's kernels of this
    /// architecture's first SIMD tier.
    #[cfg(target_arch = "aarch64")]
    use super::neon::{
        horizontal_padded_neon as horizontal_padded,
        horizontal_transposed_neon as horizontal_transposed,
        transpose_rows_i16_neon as transpose_rows, untranspose_narrow_neon as untranspose_narrow,
        vertical_q6_neon as vertical_q6,
    };
    #[cfg(target_arch = "x86_64")]
    use super::x86::{
        horizontal_padded_avx2 as horizontal_padded,
        horizontal_transposed_avx2 as horizontal_transposed,
        transpose_rows_i16_avx2 as transpose_rows, untranspose_narrow_avx2 as untranspose_narrow,
        vertical_q6_avx2 as vertical_q6,
    };

    /// Whether this host runs the kernels above.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    fn has_simd_rows() -> bool {
        #[cfg(target_arch = "x86_64")]
        return has_avx2();
        #[cfg(target_arch = "aarch64")]
        return crate::cpu::use_neon();
    }

    #[cfg(target_arch = "x86_64")]
    type PackKernel = unsafe fn(&[u8], &mut [u8]);

    fn kernel_name(kernel: Kernel) -> &'static str {
        match kernel {
            Kernel::Box => "box",
            Kernel::Bilinear => "bilinear",
            Kernel::CatmullRom => "catmullrom",
            Kernel::Lanczos3 => "lanczos3",
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn has_avx2() -> bool {
        std::is_x86_feature_detected!("avx2")
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    /// Intermediate samples as the vertical pass leaves them: Q6 u8 values,
    /// with some filter overshoot below 0.
    fn random_mid(rng: &mut Rng, len: usize) -> Vec<i16> {
        random_samples::<u8>(rng, len)
            .into_iter()
            .map(|sample| (i16::from(sample) - 16) << MID_FRAC_BITS)
            .collect()
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    /// Random Q15 weights. On aarch64 those of a window whose partial sums
    /// of Q6 samples stay within i16, as every SIMD window's do
    /// (`q15_sums_fit`): the RDM kernels saturate where the wrapping x86
    /// kernels and the reference agree for any weights. Absolute weights
    /// summing to at most `WEIGHT_TOTAL` keep the sums of mulhrs(16320, w)
    /// below 29 902.
    fn random_weights(rng: &mut Rng, taps: usize) -> Vec<i16> {
        const WEIGHT_TOTAL: i32 = 60_000;
        let weights: Vec<i32> = random_samples::<u16>(rng, taps)
            .into_iter()
            .map(|weight| i32::from(weight as i16))
            .collect();
        if cfg!(target_arch = "x86_64") {
            return weights.into_iter().map(|weight| weight as i16).collect();
        }
        let total = weights.iter().map(|weight| weight.abs()).sum::<i32>();
        let scale = |weight: i32| weight * WEIGHT_TOTAL / total.max(WEIGHT_TOTAL);
        weights
            .into_iter()
            .map(|weight| scale(weight) as i16)
            .collect()
    }

    fn assert_same<T: PartialEq + std::fmt::Debug>(actual: &[T], expected: &[T], context: &str) {
        assert_eq!(actual.len(), expected.len(), "{context}: length");
        if let Some(index) = (0..actual.len()).find(|&index| actual[index] != expected[index]) {
            panic!(
                "{context}: sample {index} is {:?}, expected {:?}",
                actual[index], expected[index]
            );
        }
    }

    /// The vertical pass by its definition: per tap a `mulhrs` of the Q6
    /// sample, i16 wraparound sums; `width` samples of rows `stride` apart.
    fn reference_vertical_q15(
        rows: &[u8],
        stride: usize,
        weights: &[i16],
        width: usize,
    ) -> Vec<i16> {
        (0..width)
            .map(|x| {
                let taps = weights.iter().enumerate();
                taps.map(|(tap, &weight)| {
                    mulhrs(i32::from(rows[tap * stride + x]) << MID_FRAC_BITS, weight)
                })
                .sum::<i32>() as i16
            })
            .collect()
    }

    /// One window's vertical pass: the Q15 arithmetic where its weights fit,
    /// else exact i32 sums.
    fn reference_vertical(rows: &[u8], width: usize, win: Window<'_>) -> Vec<i16> {
        if win.fits_q15() {
            return reference_vertical_q15(rows, width, win.q15, width);
        }
        (0..width)
            .map(|x| {
                let taps =
                    (0..win.q15.len()).map(|tap| i32::from(rows[tap * width + x]) << MID_FRAC_BITS);
                win.dot(taps) as i16
            })
            .collect()
    }

    /// Sample `channel` of output pixel `output` of a row of
    /// `channels`-sample pixels.
    fn reference_output(
        mid: &[i16],
        cols: &Windows,
        channels: usize,
        output: usize,
        channel: usize,
    ) -> u8 {
        let col = cols.get(output);
        narrow(if col.fits_q15() {
            col.mulhrs_dot_at(mid, channels, channel)
        } else {
            col.dot_at(mid, channels, channel)
        })
    }

    /// The whole resize by its definition, output row by output row.
    fn reference_resize(src: &[u8], width: usize, plan: &Plan) -> Vec<u8> {
        let channels = plan.channels;
        let ws = width * channels;
        let mut out = Vec::new();
        for win in plan.rows.iter() {
            let mid = reference_vertical(&src[win.start * ws..win.end() * ws], ws, win);
            for output in 0..plan.cols.len() {
                out.extend(
                    (0..channels).map(|channel| {
                        reference_output(&mid, &plan.cols, channels, output, channel)
                    }),
                );
            }
        }
        out
    }

    fn assert_resize_matches_reference(channels: usize) {
        let mut rng = Rng::new(channels as u64);
        for kernel in KERNELS {
            for (width, ow) in COLUMN_CASES {
                for (height, oh) in ROW_CASES {
                    let src: Vec<u8> = random_samples(&mut rng, width * height * channels);
                    let plan = Plan::new((width, height), (ow, oh), kernel, channels);
                    assert_same(
                        &resize_u8(&src, (width, height), &plan),
                        &reference_resize(&src, width, &plan),
                        &format!(
                            "{} {channels} channels {width} x {height} -> {ow} x {oh}",
                            kernel_name(kernel)
                        ),
                    );
                }
            }
        }
    }

    #[test]
    fn planar_resize_equals_reference() {
        assert_resize_matches_reference(1);
    }

    #[test]
    fn rgb_resize_equals_reference() {
        assert_resize_matches_reference(RGB);
    }

    /// The partial-sum check keeps every window of ordinary sizes on the SIMD
    /// path (only the weight range decides, as before it existed), and takes
    /// off it the windows of tens of thousands of taps whose rounded terms
    /// would add up past i16.
    #[test]
    fn q15_sum_check_takes_only_huge_windows_off_simd() {
        let sizes: Vec<usize> = (1..=40)
            .chain([
                63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 500, 512, 1000, 1024, 2048,
            ])
            .collect();
        for kernel in KERNELS {
            for &n_in in &sizes {
                for &n_out in &sizes {
                    let windows = Windows::new(n_in, n_out, kernel);
                    for (output, win) in windows.iter().enumerate() {
                        let in_range = win.q32.iter().all(|weight| {
                            (i32::from(i16::MIN)..=-i32::from(i16::MIN)).contains(weight)
                        });
                        assert_eq!(
                            win.fits_q15(),
                            in_range,
                            "{} {n_in} -> {n_out}, output {output}",
                            kernel_name(kernel)
                        );
                    }
                }
            }
        }
        for (kernel, n_in, n_out) in HUGE_REDUCTIONS {
            let windows = Windows::new(n_in, n_out, kernel);
            assert!(
                windows.iter().any(|win| !win.fits_q15()),
                "{}",
                kernel_name(kernel)
            );
        }
    }

    /// Reductions whose windows (tens of thousands of taps) would overflow
    /// the SIMD kernels' i16 sums.
    const HUGE_REDUCTIONS: [(Kernel, usize, usize); 3] = [
        (Kernel::Box, 40_000, 1),
        (Kernel::CatmullRom, 70_000, 2),
        (Kernel::Lanczos3, 50_000, 1),
    ];

    /// Those reductions resize as the reference does (their long windows on
    /// its exact path, not wrapped or saturated SIMD sums), including rows
    /// a lanczos3 upscale's overshoot lifts past the u8 range, whose rounded
    /// terms of weight 1 then each add 1.
    #[test]
    fn huge_reductions_equal_reference() {
        let mut rng = Rng::new(48);
        let step_rows = [0, 0, u8::MAX, u8::MAX];
        for (kernel, width, ow) in HUGE_REDUCTIONS {
            let plan = Plan::new((width, step_rows.len()), (ow, 7), kernel, 1);
            let steps: Vec<u8> = step_rows.iter().flat_map(|&row| vec![row; width]).collect();
            let random = random_samples(&mut rng, width * step_rows.len());
            for src in [steps, random] {
                assert_same(
                    &resize_u8(&src, (width, step_rows.len()), &plan),
                    &reference_resize(&src, width, &plan),
                    &format!("{} {width} -> {ow}", kernel_name(kernel)),
                );
            }
        }
    }

    /// The cases the resize tests run reach every plan, so those tests run
    /// every kernel this host's tier dispatches to.
    #[test]
    fn resize_cases_reach_every_plan() {
        let mut reached = std::collections::BTreeSet::new();
        for kernel in KERNELS {
            for channels in CHANNELS {
                for (width, ow) in COLUMN_CASES {
                    let plan = Plan::new((width, 33), (ow, 16), kernel, channels);
                    if plan.large_reduction {
                        reached.insert("large reduction");
                    }
                    if plan.tap_runs.is_some() {
                        reached.insert("tap runs");
                    }
                    if !plan.scalar_cols.is_empty() {
                        reached.insert("scalar columns");
                    }
                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    let transposed = takes_transposed(&plan, width);
                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                    let transposed = false;
                    reached.insert(match (transposed, channels) {
                        (true, 1) => "transposed planar",
                        (true, _) => "transposed RGB",
                        (false, 1) => "per-row planar",
                        (false, _) => "per-row RGB",
                    });
                }
            }
        }
        let mut expected = vec![
            "large reduction",
            "tap runs",
            "scalar columns",
            "per-row planar",
            "per-row RGB",
        ];
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        if has_simd_rows() {
            expected.extend(["transposed planar", "transposed RGB"]);
        }
        for plan in expected {
            assert!(reached.contains(plan), "no case reaches {plan}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn covered_by(ow: usize, outputs: impl IntoIterator<Item = usize>) -> Vec<bool> {
        let mut covered = vec![false; ow];
        for output in outputs {
            covered[output] = true;
        }
        covered
    }

    #[cfg(target_arch = "x86_64")]
    /// Every output of a row of `channels`-sample pixels: the reference where
    /// `covered`, else unwritten.
    fn assert_row(
        dst: &[u8],
        mid: &[i16],
        cols: &Windows,
        channels: usize,
        covered: &[bool],
        context: &str,
    ) {
        for (output, &is_covered) in covered.iter().enumerate() {
            for channel in 0..channels {
                let expected = if is_covered {
                    reference_output(mid, cols, channels, output, channel)
                } else {
                    SENTINEL
                };
                assert_eq!(
                    dst[output * channels + channel],
                    expected,
                    "{context}: output {output}, channel {channel}"
                );
            }
        }
    }

    /// The AVX2 tap runs on RGBX-layout rows padded as `resize_pixel_rows`
    /// pads them, for every case, whether or not the plan would choose them.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn tap_runs_equal_reference() {
        if !has_avx2() {
            return;
        }
        let mut rng = Rng::new(6);
        for kernel in KERNELS {
            for (width, ow) in COLUMN_CASES {
                let cols = Windows::new(width, ow, kernel);
                let context = format!("{} {width} -> {ow}", kernel_name(kernel));
                let runs = Runs::new(&cols, width);
                let mids = [(); 2].map(|()| random_mid(&mut rng, (width + runs.overreach) * RGBX));
                let mut dsts = [(); 2].map(|()| vec![SENTINEL; ow * RGBX]);
                // SAFETY: AVX2 detected; the mids padded by the runs' overreach.
                unsafe {
                    let [dst0, dst1] = &mut dsts;
                    x86::horizontal_taps_x2_avx2([&mids[0], &mids[1]], [dst0, dst1], &runs);
                }
                let covered = covered_by(ow, runs.outputs.iter().copied());
                for (mid, dst) in mids.iter().zip(&dsts) {
                    assert_row(dst, mid, &cols, RGBX, &covered, &context);
                }
            }
        }
    }

    /// The vertical kernels: widths around their registers and lines, rows
    /// starting off a 32-byte boundary, and the strides and window lengths where the
    /// taps run in groups (4 KB aliasing, 2 KB on AVX2, 40+ taps).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn vertical_kernels_equal_reference() {
        if !has_simd_rows() {
            return;
        }
        let mut rng = Rng::new(7);
        let widths = [16, 17, 31, 32, 33, 63, 64, 65, 100, 130, 257];
        let cases = widths
            .into_iter()
            .flat_map(|width| [1, 2, 3, 8, 9, 17, 41].map(|taps| (width, taps)))
            .chain([(2048, 16), (4096, 12), (4096, 16)]);
        for (width, taps) in cases {
            let weights = random_weights(&mut rng, taps);
            for offset in [0, 1, 31] {
                let buffer: Vec<u8> = random_samples(&mut rng, offset + taps * width);
                let rows = &buffer[offset..];
                let expected = reference_vertical_q15(rows, width, &weights, width);
                let context = format!("width {width}, {taps} taps, offset {offset}");
                let mut mid = vec![SENTINEL_I16; width];
                // SAFETY: the tier detected; rows holds taps rows of width >=
                // 16 samples, mid holds width.
                unsafe { vertical_q6(rows, width, &weights, &mut mid) };
                assert_same(&mid, &expected, &format!("{context}, first tier"));
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn widen_to_rgbx_kernels_equal_reference() {
        let avx2 = has_avx2();
        let mut rng = Rng::new(9);
        for pixels in 1..=70 {
            // The kernel's minimum padding: whole groups of 4 pixels, and half
            // a register's read past them.
            let mut check = |groups: usize,
                             overread: usize,
                             kernel: unsafe fn(&[i16], usize, &mut [i16]),
                             name: &str| {
                let src = random_mid(&mut rng, RGB * groups + overread);
                let mut rgbx = vec![SENTINEL_I16; RGBX * groups];
                // SAFETY: the caller detected the kernel's features; src and
                // rgbx hold the kernel's minimum lengths.
                unsafe { kernel(&src, pixels, &mut rgbx) };
                let expected: Vec<i16> = src[..RGB * pixels]
                    .as_chunks::<RGB>()
                    .0
                    .iter()
                    .flat_map(|px| [px[0], px[1], px[2], 0])
                    .collect();
                assert_same(
                    &rgbx[..RGBX * pixels],
                    &expected,
                    &format!("{pixels} pixels, {name}"),
                );
            };
            if avx2 {
                let (group, overread) = (AVX2_I16_LANES / RGBX, AVX2_I16_LANES / 2);
                check(
                    pixels.next_multiple_of(group),
                    overread,
                    x86::widen_to_rgbx_avx2,
                    "avx2",
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn pack_from_rgbx_kernels_equal_reference() {
        let avx2 = has_avx2();
        let mut rng = Rng::new(10);
        for pixels in 1..=70 {
            let rgbx: Vec<u8> = random_samples(&mut rng, RGBX * pixels);
            let expected: Vec<u8> = rgbx
                .as_chunks::<RGBX>()
                .0
                .iter()
                .flat_map(|px| [px[0], px[1], px[2]])
                .collect();
            let tiers: [(bool, PackKernel, &str); 1] = [(avx2, x86::pack_from_rgbx_avx2, "avx2")];
            for (available, kernel, name) in tiers {
                if available {
                    let mut out = vec![SENTINEL; RGB * pixels];
                    // SAFETY: the kernel's features detected; rgbx holds RGBX
                    // bytes per output pixel.
                    unsafe { kernel(&rgbx, &mut out) };
                    assert_same(&out, &expected, &format!("{pixels} pixels, {name}"));
                }
            }
        }
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn transpose_rows_equals_reference() {
        if !has_simd_rows() {
            return;
        }
        let mut rng = Rng::new(11);
        for stride in [16, 32, 48, 160] {
            let mid = random_mid(&mut rng, ROWS * stride);
            let mut mid_t = vec![SENTINEL_I16; stride * ROWS];
            // SAFETY: the tier detected; stride is a multiple of ROWS, both
            // buffers hold ROWS * stride samples.
            unsafe { transpose_rows(&mid, stride, &mut mid_t) };
            let expected: Vec<i16> = (0..stride)
                .flat_map(|x| (0..ROWS).map(move |row| (x, row)))
                .map(|(x, row)| mid[row * stride + x])
                .collect();
            assert_same(&mid_t, &expected, &format!("stride {stride}"));
        }
    }

    /// The transposed horizontal pass on buffers sized as
    /// `resize_transposed` sizes them.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn horizontal_transposed_equals_reference() {
        if !has_simd_rows() {
            return;
        }
        let mut rng = Rng::new(12);
        for kernel in KERNELS {
            for channels in CHANNELS {
                for (width, ow) in COLUMN_CASES {
                    let cols = Windows::new(width, ow, kernel);
                    let (ws, ows) = (width * channels, ow * channels);
                    // Padded as `resize_transposed` pads it.
                    let mid_stride = ws.next_multiple_of(64) + 32;
                    let mid_t = random_mid(&mut rng, mid_stride * ROWS);
                    let mut out_t = vec![SENTINEL_I16; ows.next_multiple_of(ROWS) * ROWS];
                    // SAFETY: the tier detected; every window ends by the
                    // row's width, within mid_t, and out_t holds every output.
                    unsafe { horizontal_transposed(&mid_t, &cols, channels, &mut out_t) };
                    let context = format!(
                        "{} {channels} channels {width} -> {ow}",
                        kernel_name(kernel)
                    );
                    for (output, col) in cols.iter().enumerate() {
                        for (channel, row) in (0..channels)
                            .flat_map(|channel| (0..ROWS).map(move |row| (channel, row)))
                        {
                            let expected = if col.fits_q15() {
                                col.mulhrs_dot_at(&mid_t, channels * ROWS, channel * ROWS + row)
                                    as i16
                            } else {
                                SENTINEL_I16
                            };
                            assert_eq!(
                                out_t[(output * channels + channel) * ROWS + row],
                                expected,
                                "{context}: output {output}, channel {channel}, row {row}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The padded windows' pass equals the plain pass on every output the
    /// SIMD path writes, on `mid_t` sized and padded as `resize_transposed`
    /// sizes it; and every case with windows narrow enough pads.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn padded_horizontal_equals_the_plain_pass() {
        if !has_simd_rows() {
            return;
        }
        let mut rng = Rng::new(14);
        let mut padded_cases = 0;
        for kernel in KERNELS {
            for channels in CHANNELS {
                for (width, ow) in COLUMN_CASES {
                    let cols = Windows::new(width, ow, kernel);
                    let Some(padded) = PaddedWindows::new(&cols) else {
                        let widest = cols
                            .iter()
                            .filter(|col| col.fits_q15())
                            .map(|col| col.q15.len());
                        assert!(widest.max().unwrap_or(0) > 20);
                        continue;
                    };
                    padded_cases += 1;
                    let (ws, ows) = (width * channels, ow * channels);
                    let mid_stride = ws.next_multiple_of(64) + 32;
                    let mid_t = random_mid(&mut rng, (mid_stride + PADDED_READS_PAST_ROW) * ROWS);
                    let run = |padded: Option<&PaddedWindows>| {
                        let mut out_t = vec![SENTINEL_I16; ows.next_multiple_of(ROWS) * ROWS];
                        // SAFETY: the tier detected (has_simd_rows); mid_t is
                        // padded as resize_transposed pads it, out_t holds
                        // every output.
                        unsafe {
                            match padded {
                                Some(padded) => {
                                    horizontal_padded(&mid_t, padded, channels, &mut out_t);
                                }
                                None => horizontal_transposed(&mid_t, &cols, channels, &mut out_t),
                            }
                        }
                        out_t
                    };
                    let (expected, actual) = (run(None), run(Some(&padded)));
                    let context = format!(
                        "{} {channels} channels {width} -> {ow}",
                        kernel_name(kernel)
                    );
                    for output in (0..ow).filter(|&output| cols.get(output).fits_q15()) {
                        let samples = (output * channels * ROWS)..((output + 1) * channels * ROWS);
                        assert_eq!(
                            actual[samples.clone()],
                            expected[samples],
                            "{context}: output {output}"
                        );
                    }
                }
            }
        }
        assert!(padded_cases > 0);
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn untranspose_narrow_equals_reference() {
        if !has_simd_rows() {
            return;
        }
        let mut rng = Rng::new(13);
        for ow in [1usize, 15, 16, 17, 63, 64, 65, 130] {
            for rows in [1, 7, ROWS] {
                let out_t = random_mid(&mut rng, ow.next_multiple_of(ROWS) * ROWS);
                let mut out = vec![SENTINEL; rows * ow];
                // SAFETY: the tier detected; rows <= ROWS, out_t holds whole
                // tiles of ow outputs, out holds rows * ow bytes.
                unsafe { untranspose_narrow(&out_t, ow, &mut out, rows) };
                let expected: Vec<u8> = (0..rows)
                    .flat_map(|row| (0..ow).map(move |output| (row, output)))
                    .map(|(row, output)| narrow(i32::from(out_t[output * ROWS + row])))
                    .collect();
                assert_same(&out, &expected, &format!("{ow} outputs, {rows} rows"));
            }
        }
    }
}

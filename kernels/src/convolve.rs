//! Native convolution resize of u8 images, planar or interleaved RGB or
//! 4-sample pixels, any ratio, with `fast_image_resize`'s filters and window
//! rules (clipped at the image edge, renormalized) — the same function, but
//! more exactly: the vertical pass keeps a Q6 i16 intermediate instead of
//! rounding and clamping to u8, which keeps every output within 1 LSB of the
//! exact convolution for reductions down to 1/128 per axis. Each tap's
//! product rounds to the Q6 grid, so the error grows with the tap count:
//! beyond 1/128 it could exceed 1 LSB, so larger reductions take the f32
//! engine (`crate::convolve_float`) instead.
//!
//! Vertical pass first, one output row at a time (contiguous, SIMD across the
//! row, no intermediate image), then the horizontal pass on that cached row,
//! by one of three plans:
//! - phased, for an integer factor k <= `MAX_PHASES`: the row is split into
//!   k phase arrays, after which every interior output has the same weights
//!   and each tap is a unit-stride load;
//! - blocks, for any ratio: runs of outputs whose taps fit a 64-sample window
//!   gather each tap with one two-register permute;
//! - sample runs, for windows too wide for blocks: each output's taps a
//!   register at a time.
//!
//! Where the per-row gathers don't apply (see `resize_u8`), the horizontal
//! pass runs on transposed 16-row batches instead (`resize_transposed`). The
//! scalar kernels mirror the SIMD arithmetic bit for bit, so results are
//! identical on every CPU.
//!
//! Interleaved images run the same vertical pass over cw samples and the
//! same weights per channel, so they equal resizing each channel as a planar
//! image. Their horizontal pass gathers whole pixels (see `PixelBlocks`; RGB
//! rows are widened to the RGBX layout for it and packed back), runs per
//! output over its taps (`TapRuns`), or runs transposed with a channel
//! stride.

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
pub(crate) mod x86;

use std::cell::RefCell;
use std::ops::Range;
use std::sync::Arc;

use crate::sample::{BICUBIC_U8_FRAC_BITS, mulhrs, q15};

/// Fraction bits of the intermediate row (u8 << 6 leaves i16 headroom for
/// the filters' overshoot).
const MID_FRAC_BITS: u32 = BICUBIC_U8_FRAC_BITS;
/// The weights' fixed point: Q15, as `mulhrs` multiplies by.
const Q15_BITS: u32 = 15;
const Q15_ONE: f64 = (1 << Q15_BITS) as f64;
/// Unnormalized weights this small are zeros that rounding left behind.
const ZERO_WEIGHT_EPS: f64 = 1e-9;
/// i16 lanes per register: the narrowest rows the vertical kernels take.
const AVX512_I16_LANES: usize = 32;
#[cfg(target_arch = "x86_64")]
const AVX2_I16_LANES: usize = 16;
/// Output rows batched per transposed horizontal pass (one AVX2 register of
/// i16 lanes, two NEON ones).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) const TRANSPOSE_ROWS: usize = 16;
/// Pixels per block of the RGB-to-RGBX vertical kernel.
#[cfg(target_arch = "x86_64")]
const FUSED_RGBX_BLOCK_PIXELS: usize = 32;
/// Integer factors split into phase arrays (others use blocks or sample runs).
const MAX_PHASES: usize = 4;
/// Samples one block's taps may span: the two registers of one permute.
pub(crate) const BLOCK_SPAN: usize = 64;
/// Outputs per block at most: one register of i16 lanes.
pub(crate) const BLOCK_OUTPUTS: usize = 32;
/// Narrower blocks leave too many lanes idle; below this the sample runs
/// are faster.
const MIN_BLOCK_OUTPUTS: usize = 10;
/// Channels of the interleaved layouts with a native path; RGB runs the
/// RGBA horizontal kernels on rows widened to RGBX.
pub(crate) const RGB: usize = 3;
pub(crate) const RGBA: usize = 4;
/// Output pixels per pixel block at most: one register of 8 x 4 i16 lanes.
pub(crate) const PIXEL_OUTPUTS: usize = 8;
/// Pixels one pixel-block tap may reach: the two registers of one permute.
pub(crate) const PIXEL_SPAN: usize = 16;
/// The AVX2 pixel blocks (`horizontal_pixels4_x2_avx2`): 4 outputs within an
/// 8-pixel window (two registers of 4 pixels).
pub(crate) const AVX2_PIXEL_OUTPUTS: usize = 4;
pub(crate) const AVX2_PIXEL_SPAN: usize = 8;
/// Longest window the phased plan takes for integer factors >= 3; longer
/// windows run as sample runs.
const MAX_TAPS_FACTOR_3_PLUS: usize = 20;
/// Pixel blocks this short (median outputs per block) leave tap runs ahead.
const SPARSE_BLOCK_OUTPUTS: usize = 5;
/// Longer pixel blocks still lose to tap runs on windows of
/// `TAP_RUN_WINDOW_TAPS`+ taps.
const SHORT_BLOCK_OUTPUTS: usize = 7;
const TAP_RUN_WINDOW_TAPS: usize = 12;
/// Mean outputs per AVX2 pixel block from which those blocks fill well
/// enough to beat the transposed path.
const AVX2_BLOCK_FILL: f64 = 3.5;
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
/// f32 for the uint16/float32 resize. Renormalizing a clipped edge window
/// can push a weight past 1.0 (1.13 for lanczos3 when upscaling), which i16
/// Q15 cannot hold.
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
/// starts at most at the row's last pixel, of at most 4 samples.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const PADDED_READS_PAST_ROW: usize = (MAX_PADDED_TAPS - 1) * 4;

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

/// Outputs of one block: `count` from `o0`, taps within `mid[base..base +
/// BLOCK_SPAN]`;
/// per tap, the lane index into that window and the weight, per output lane.
pub(crate) struct Block {
    pub(crate) o0: usize,
    pub(crate) count: usize,
    pub(crate) base: usize,
    pub(crate) idx: Vec<[i16; BLOCK_OUTPUTS]>,
    pub(crate) weights: Vec<[i16; BLOCK_OUTPUTS]>,
}

/// An interior tap of the phased plan: output `o` reads `phases[phase][o + rel]`.
#[derive(Clone, Copy)]
pub(crate) struct PhaseTap {
    pub(crate) phase: usize,
    pub(crate) rel: isize,
    pub(crate) weight: i16,
}

/// How the horizontal pass covers every output of a row.
enum Horizontal {
    Blocks(Vec<Block>),
    Phased {
        factor: usize,
        interior: Range<usize>,
        taps: Vec<PhaseTap>,
        edges: Vec<Block>,
    },
    /// Large reductions: each output's own taps, a register of samples
    /// (`BLOCK_OUTPUTS`) per group: the planar counterpart of `TapRuns`.
    SampleRuns(Runs),
}

/// Per-output runs over the taps (large reductions): per output, its
/// window's inputs a group at a time times its weights, the lanes then
/// summed to one. Planar rows group `BLOCK_OUTPUTS` samples (one per lane),
/// RGBA rows `PIXEL_OUTPUTS` pixels (4 lanes each).
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
    pub(crate) weights: Vec<[i16; BLOCK_OUTPUTS]>,
    /// Inputs (samples or pixels, as `width`) a group may read past the row end.
    overreach: usize,
}

impl Runs {
    /// Groups of `group_taps` taps, `BLOCK_OUTPUTS / group_taps` lanes per tap.
    fn new(cols: &Windows, width: usize, group_taps: usize) -> Runs {
        let lanes = BLOCK_OUTPUTS / group_taps;
        let outputs: Vec<usize> = (0..cols.len())
            .filter(|&output| cols.get(output).fits_q15())
            .collect();
        let taps = outputs
            .iter()
            .map(|&output| cols.get(output).q15.len())
            .max()
            .unwrap_or(0);
        let groups = taps.div_ceil(group_taps).max(1);
        let mut weights = vec![[0; BLOCK_OUTPUTS]; outputs.len() * groups];
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

/// The RGBA-layout horizontal pass (RGBA, and 2- and 3-channel rows widened
/// to RGBX), by runs of up to `PIXEL_OUTPUTS` output pixels whose first taps
/// lie within `PIXEL_SPAN` pixels of the run's first (the AVX2 kernel:
/// `AVX2_PIXEL_OUTPUTS` within `AVX2_PIXEL_SPAN`). A pixel of the
/// intermediate row is 4 x i16 = one qword; for tap t the window starting at
/// `base + t` is loaded and one qword permute (a fixed index per block)
/// gathers pixel `start_j + t` of every output j.
pub(crate) struct PixelBlocks {
    pub(crate) blocks: Vec<PixelBlock>,
    /// Per block tap, per lane: output j's weight in lanes 4j..4j + 4.
    pub(crate) weights: Vec<[i16; PIXEL_OUTPUTS * RGBA]>,
    /// Pixels a window may read past the row end.
    overreach: usize,
}

pub(crate) struct PixelBlock {
    pub(crate) o0: usize,
    pub(crate) count: usize,
    pub(crate) base: usize,
    /// Output j's first tap, relative to `base`.
    pub(crate) idx: [i64; PIXEL_OUTPUTS],
    /// This block's rows of `PixelBlocks::weights`, one per tap.
    pub(crate) taps: Range<usize>,
}

impl PixelBlocks {
    /// Outputs of the middle block (blocks are uniform away from the edges).
    fn median_outputs(&self) -> Option<usize> {
        self.blocks
            .get(self.blocks.len() / 2)
            .map(|block| block.count)
    }

    fn new(cols: &Windows) -> PixelBlocks {
        Self::with_size(cols, PIXEL_OUTPUTS, PIXEL_SPAN)
    }

    /// Blocks of up to `outputs` outputs whose first taps lie within `span`
    /// pixels (the weight rows keep their `PIXEL_OUTPUTS` lanes; the first
    /// `outputs` are used).
    fn with_size(cols: &Windows, outputs: usize, span: usize) -> PixelBlocks {
        let (mut blocks, mut weights, mut max_taps) = (Vec::new(), Vec::new(), 0);
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
                        .is_some_and(|offset| offset < span)
            };
            let count = (0..outputs).take_while(|&slot| fits(slot)).count();
            let taps = (output..output + count)
                .map(|x| cols.get(x).q15.len())
                .max()
                .unwrap_or(0);
            max_taps = max_taps.max(taps);
            let first = weights.len();
            weights.resize(first + taps, [0; PIXEL_OUTPUTS * RGBA]);
            let mut idx = [0; PIXEL_OUTPUTS];
            for (slot, first_tap) in idx.iter_mut().enumerate().take(count) {
                let win = cols.get(output + slot);
                *first_tap = (win.start - base) as i64;
                for (tap, &weight) in win.q15.iter().enumerate() {
                    weights[first + tap][RGBA * slot..RGBA * (slot + 1)].fill(weight);
                }
            }
            blocks.push(PixelBlock {
                o0: output,
                count,
                base,
                idx,
                taps: first..first + taps,
            });
            output += count;
        }
        PixelBlocks {
            blocks,
            weights,
            overreach: max_taps + span,
        }
    }
}

/// The RGB/RGBA horizontal pass for large reductions, vectorized over each
/// output's taps instead of across outputs: `Runs` of `PIXEL_OUTPUTS`
/// pixels per group. Pixel blocks fill their lanes only while several
/// outputs' taps fit `PIXEL_SPAN` pixels; this costs the same per tap at any
/// ratio. Used where blocks run short of outputs (`pay_off`); at milder
/// ratios the blocks are faster. Either kernel gives the same result.
#[cfg_attr(
    not(target_arch = "x86_64"),
    expect(
        dead_code,
        reason = "read by the x86 kernels; the scalar path has no use"
    )
)]
pub(crate) struct TapRuns {
    pub(crate) runs: Runs,
    /// Pixel blocks would hold fewer than `SPARSE_BLOCK_OUTPUTS` outputs.
    pub(crate) sparse_blocks: bool,
}

impl TapRuns {
    /// Whether tap runs beat `pixels` for these windows.
    fn pay_off(cols: &Windows, pixels: &PixelBlocks) -> bool {
        let Some(median) = pixels.median_outputs() else {
            return false;
        };
        let taps = cols.get(cols.len() / 2).q15.len();
        median < SPARSE_BLOCK_OUTPUTS
            || (median < SHORT_BLOCK_OUTPUTS && taps >= TAP_RUN_WINDOW_TAPS)
    }

    fn new(cols: &Windows, width: usize, pixels: &PixelBlocks) -> TapRuns {
        let sparse_blocks = pixels
            .median_outputs()
            .is_some_and(|median| median < SPARSE_BLOCK_OUTPUTS);
        TapRuns {
            runs: Runs::new(cols, width, PIXEL_OUTPUTS),
            sparse_blocks,
        }
    }
}

/// The RGBA horizontal pass of an integer-factor upscale (ow = P * width, P =
/// 2 or 4) on AVX-512. Output pixel P*m + j has its window at m + `rel_j` with
/// phase j's weights for every interior m, so each phase is a plain
/// convolution over the intermediate row — unit-stride loads and broadcast
/// weights, no gathers — and the P results interleave into consecutive
/// pixels. Outputs outside the interior take the scalar reference.
#[cfg_attr(
    not(target_arch = "x86_64"),
    expect(
        dead_code,
        reason = "read by the x86 kernels; the scalar path has no use"
    )
)]
pub(crate) struct IntegerUpscale {
    pub(crate) factor: usize,
    /// Per phase, per tap: (pixel offset from m, weight).
    pub(crate) taps: Vec<Vec<(isize, i16)>>,
    /// The m where every phase's window is its unclipped interior one.
    pub(crate) interior: Range<usize>,
}

impl IntegerUpscale {
    fn new(cols: &Windows, width: usize) -> Option<IntegerUpscale> {
        let factor = cols.len() / width;
        if !matches!(factor, 2 | 4) || cols.len() != factor * width {
            return None;
        }
        let m_ref = width / 2;
        let (mut taps, mut interior) = (Vec::new(), 0..width);
        for phase in 0..factor {
            let reference = cols.get(factor * m_ref + phase);
            let rel = reference.start as isize - m_ref as isize;
            let is_interior = |pixel: usize| {
                let window = cols.get(factor * pixel + phase);
                window.fits_q15()
                    && window.start as isize - pixel as isize == rel
                    && window.q15 == reference.q15
            };
            let run = interior_run(width, is_interior)?;
            interior = interior.start.max(run.start)..interior.end.min(run.end);
            taps.push(
                reference
                    .q15
                    .iter()
                    .enumerate()
                    .map(|(tap, &weight)| (rel + tap as isize, weight))
                    .collect(),
            );
        }
        (interior.len() >= PIXEL_OUTPUTS).then_some(IntegerUpscale {
            factor,
            taps,
            interior,
        })
    }

    /// Outputs the interior does not cover.
    #[cfg(target_arch = "x86_64")]
    fn edges(&self, ow: usize) -> impl Iterator<Item = usize> + use<> {
        (0..self.factor * self.interior.start).chain(self.factor * self.interior.end..ow)
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
    /// The planar horizontal pass. Also built for the interleaved layouts,
    /// where `Horizontal::SampleRuns` marks a large reduction for every layout.
    horizontal: Horizontal,
    /// Samples per pixel: 1, RGB or RGBA.
    channels: usize,
    /// Some for RGB and RGBA.
    pixels: Option<PixelBlocks>,
    /// RGBA on the AVX2 tier: the 4-output blocks of its pixel kernel.
    pixels_avx2: Option<PixelBlocks>,
    /// Some for an RGB/RGBA integer-factor upscale.
    upscale: Option<IntegerUpscale>,
    /// Some for RGB/RGBA reductions where it beats `pixels`.
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
    /// `channels`-sample image (pixel kernels for RGB and RGBA): block or
    /// phase plans where the windows are narrow enough to fill them,
    /// per-output tap runs past that (large reductions) — the same choice for
    /// every layout.
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
        debug_assert!([1, RGB, RGBA].contains(&channels));
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
        let simd_cols =
            |outputs: Range<usize>| outputs.filter(|output| cols.get(*output).fits_q15());
        let factor = width / ow;
        let planned = if !worth_planning(extent_width, ow, kernel) {
            None
        } else if width == factor * ow && (2..=MAX_PHASES).contains(&factor) {
            phased(&cols, factor, &scalar_cols)
        } else {
            blocks(&cols, simd_cols(0..ow).collect()).map(Horizontal::Blocks)
        };
        let horizontal = planned
            .unwrap_or_else(|| Horizontal::SampleRuns(Runs::new(&cols, width, BLOCK_OUTPUTS)));
        let interleaved = channels > 1;
        let pixels = interleaved.then(|| PixelBlocks::new(&cols));
        let upscale = if interleaved {
            IntegerUpscale::new(&cols, width)
        } else {
            None
        };
        let tap_runs = pixels
            .as_ref()
            .filter(|pixel_blocks| TapRuns::pay_off(&cols, pixel_blocks))
            .map(|pixel_blocks| TapRuns::new(&cols, width, pixel_blocks));
        let avx2_tier = crate::cpu::has_avx2() && !crate::cpu::has_avx512();
        // Only where the 4-output blocks fill; sparser blocks lose to the
        // transposed path.
        let pixels_avx2 = (channels == RGBA && avx2_tier && tap_runs.is_none())
            .then(|| PixelBlocks::with_size(&cols, AVX2_PIXEL_OUTPUTS, AVX2_PIXEL_SPAN))
            .filter(|pixel_blocks| pixel_blocks.blocks.len() as f64 * AVX2_BLOCK_FILL <= ow as f64);
        // AVX2 RGBA past that: tap runs, which beat the transposed path.
        let tap_runs =
            if tap_runs.is_none() && channels == RGBA && avx2_tier && pixels_avx2.is_none() {
                pixels
                    .as_ref()
                    .map(|pixel_blocks| TapRuns::new(&cols, width, pixel_blocks))
            } else {
                tap_runs
            };
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let padded_cols = PaddedWindows::new(&cols);
        Plan {
            rows,
            cols,
            horizontal,
            channels,
            pixels,
            pixels_avx2,
            upscale,
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

/// Whether the horizontal pass of `width -> ow` can get a block or phase
/// plan, judged from the widest possible window.
fn worth_planning(width: usize, ow: usize, kernel: Kernel) -> bool {
    let scale = width as f64 / ow as f64;
    let max_taps = 2 * (kernel.support() * scale.max(1.0)).ceil() as usize + 1;
    let factor = width / ow;
    if width == factor * ow && (2..=MAX_PHASES).contains(&factor) {
        // max_taps bounds the window from above: one past the widest a
        // window at an integer factor gets.
        return factor < 3 || max_taps <= MAX_TAPS_FACTOR_3_PLUS + 1;
    }
    // A block of MIN_BLOCK_OUTPUTS must fit its taps in one window.
    (MIN_BLOCK_OUTPUTS - 1) as f64 * scale + max_taps as f64 <= BLOCK_SPAN as f64
}

/// Group `outputs` (ascending) into blocks of up to `BLOCK_OUTPUTS`
/// consecutive outputs whose taps fit one `BLOCK_SPAN` window; None if that
/// leaves blocks narrower than `MIN_BLOCK_OUTPUTS` anywhere but at a run's
/// end.
fn blocks(cols: &Windows, outputs: Vec<usize>) -> Option<Vec<Block>> {
    // The samples the windows of `outputs` read: not simply from the first's
    // start to the last's end, since trimmed zero taps can start a window
    // after the next one's (or end it before the previous one's end).
    let span = |outputs: Range<usize>| {
        let start = outputs
            .clone()
            .map(|output| cols.get(output).start)
            .min()
            .unwrap_or(0);
        start
            ..outputs
                .map(|output| cols.get(output).end())
                .max()
                .unwrap_or(start)
    };
    let mut blocks = Vec::new();
    let mut next = 0;
    while next < outputs.len() {
        let o0 = outputs[next];
        // The consecutive run starting at o0, capped at one register.
        let run = outputs[next..]
            .iter()
            .enumerate()
            .take_while(|&(ref offset, &output)| output == o0 + offset)
            .take(BLOCK_OUTPUTS)
            .count();
        let (count, base) = (1..=run).rev().find_map(|count| {
            let span = span(o0..o0 + count);
            (span.len() <= BLOCK_SPAN).then_some((count, span.start))
        })?;
        if count < MIN_BLOCK_OUTPUTS && count < run {
            return None;
        }
        let taps = (o0..o0 + count)
            .map(|output| cols.get(output).q15.len())
            .max()
            .unwrap_or(0);
        let mut idx = vec![[0i16; BLOCK_OUTPUTS]; taps];
        let mut weights = vec![[0i16; BLOCK_OUTPUTS]; taps];
        for slot in 0..count {
            let win = cols.get(o0 + slot);
            for (tap, &weight) in win.q15.iter().enumerate() {
                idx[tap][slot] = (win.start - base + tap) as i16;
                weights[tap][slot] = weight;
            }
        }
        blocks.push(Block {
            o0,
            count,
            base,
            idx,
            weights,
        });
        next += count;
    }
    Some(blocks)
}

/// The first run of `0..count` where `is_interior` holds.
fn interior_run(count: usize, is_interior: impl Fn(usize) -> bool) -> Option<Range<usize>> {
    let first = (0..count).find(|&index| is_interior(index))?;
    Some(
        first
            ..(first..count)
                .find(|&index| !is_interior(index))
                .unwrap_or(count),
    )
}

/// The phased plan for an exact integer factor `factor`: interior outputs
/// (full window at the constant phase) read the phase arrays; the rest go to
/// blocks.
fn phased(cols: &Windows, factor: usize, scalar_cols: &[usize]) -> Option<Horizontal> {
    let n_out = cols.len();
    let mid = n_out / 2;
    // Interior windows start `lead` inputs before `factor * output`.
    let (len, lead) = (
        cols.get(mid).q15.len(),
        (factor * mid) as isize - cols.get(mid).start as isize,
    );
    if factor >= 3 && len > MAX_TAPS_FACTOR_3_PLUS {
        return None;
    }
    let is_interior = |output: usize| {
        let col = cols.get(output);
        col.q15.len() == len
            && (factor * output) as isize - col.start as isize == lead
            && col.q15 == cols.get(mid).q15
    };
    let interior = interior_run(n_out, is_interior)?;
    let taps = cols
        .get(mid)
        .q15
        .iter()
        .enumerate()
        .map(|(tap, &weight)| {
            let off = tap as isize - lead; // input factor*output + off
            PhaseTap {
                phase: off.rem_euclid(factor as isize) as usize,
                rel: off.div_euclid(factor as isize),
                weight,
            }
        })
        .collect();
    let edges = (0..interior.start)
        .chain(interior.end..n_out)
        .filter(|output| !scalar_cols.contains(output))
        .collect();
    Some(Horizontal::Phased {
        factor,
        interior,
        taps,
        edges: blocks(cols, edges)?,
    })
}

/// Resize a u8 image of `plan.channels` samples per pixel by `plan`; with
/// `rgbx_pad`, an RGB image to RGBX output, `rgbx_pad` the 4th sample.
/// Every path computes the same fixed-point arithmetic (per-tap rounded
/// mulhrs, i16 sums, one final rounding), so the result is identical on
/// every CPU.
pub(crate) fn resize_u8(
    src: &[u8],
    (width, height): (usize, usize),
    plan: &Plan,
    rgbx_pad: Option<u8>,
) -> Vec<u8> {
    let channels = plan.channels;
    debug_assert!(src.len() >= width * height * channels && plan.n_in == width);
    debug_assert!(rgbx_pad.is_none() || channels == RGB);
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if takes_transposed(plan, width, rgbx_pad) {
        let out = resize_transposed(src, width, channels, plan);
        let (ow, oh) = (plan.cols.len(), plan.rows.len());
        return match rgbx_pad {
            Some(pad) => crate::rgbx::expand_image(&out, oh, ow, pad),
            None => out,
        };
    }
    match &plan.pixels {
        Some(pixels) => resize_pixel_rows(src, width, plan, pixels, rgbx_pad),
        None => resize_rows(src, width, plan),
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
fn takes_transposed(plan: &Plan, width: usize, rgbx_pad: Option<u8>) -> bool {
    crate::cpu::has_avx2()
        && width * plan.channels >= AVX2_I16_LANES
        && plan.rows.sums_fit()
        && plan.cols.sums_fit()
        && runs_transposed(plan, rgbx_pad)
}
#[cfg(target_arch = "aarch64")]
fn takes_transposed(plan: &Plan, width: usize, _rgbx_pad: Option<u8>) -> bool {
    crate::cpu::use_neon()
        && width * plan.channels >= TRANSPOSE_ROWS
        && plan.rows.sums_fit()
        && plan.cols.sums_fit()
}

/// On x86, whether the transposed path beats the per-row kernels the tier
/// has.
#[cfg(target_arch = "x86_64")]
fn runs_transposed(plan: &Plan, rgbx_pad: Option<u8>) -> bool {
    let channels = plan.channels;
    // Large reductions (no block plan) of 1 and 3 channels: transposed on
    // either tier, which needs no widening; RGBA's tap runs stay faster.
    let large_reduction = matches!(plan.horizontal, Horizontal::SampleRuns(_)) && channels < RGBA;
    // RGB with wide windows unless written as RGBX: widening to RGBX costs
    // the pixel kernels more than the transposes.
    let unwidened = rgbx_pad.is_none() && channels == RGB && plan.widest_col >= RGB_TRANSPOSED_TAPS;
    // AVX2, which lacks the AVX-512 gathers, transposes unless it has a
    // per-row kernel: RGBA pixel blocks, or tap runs (RGB only for sparse
    // blocks, where they pay for the widening and packing).
    let avx2_per_row = plan.pixels_avx2.is_some()
        || plan
            .tap_runs
            .as_ref()
            .is_some_and(|tap_runs| channels == RGBA || (channels > 1 && tap_runs.sparse_blocks));
    let avx2_transposed = !crate::cpu::has_avx512() && !avx2_per_row;
    large_reduction || unwidened || avx2_transposed
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
    if (mid.capacity() + mid_t.capacity() + out_t.capacity()) * size_of::<i16>()
        <= SCRATCH_KEPT_BYTES
    {
        SCRATCH.with_borrow_mut(|scratch| *scratch = TransposeScratch { mid, mid_t, out_t });
    }
    out
}

/// Scratch kept per thread up to this size, so a very wide image doesn't pin
/// its buffers to the thread.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
const SCRATCH_KEPT_BYTES: usize = 8 << 20;

/// The transposed path's buffers, kept per thread between calls: fresh, a
/// tile's call would clear (or fault in) them every time. `TRANSPOSE_ROWS`
/// intermediate rows, the same transposed, and the transposed Q6 output sums.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[derive(Default)]
struct TransposeScratch {
    mid: Vec<i16>,
    mid_t: Vec<i16>,
    out_t: Vec<i16>,
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

/// The per-row path (AVX-512 kernels, or the scalar reference).
fn resize_rows(src: &[u8], width: usize, plan: &Plan) -> Vec<u8> {
    let ow = plan.cols.len();
    let mut out = vec![0u8; ow * plan.rows.len()];
    // Padded so a block's 64-sample window (or a run) may run past the row end.
    let row_pad = match &plan.horizontal {
        Horizontal::SampleRuns(runs) => runs.overreach.max(BLOCK_SPAN),
        _ => BLOCK_SPAN,
    };
    let mut mids = [vec![0i16; width + row_pad], vec![0i16; width + row_pad]];
    let factor = match plan.horizontal {
        Horizontal::Phased { factor, .. } => factor,
        _ => 1,
    };
    let phase_len = width.div_ceil(factor) + BLOCK_OUTPUTS;
    let mut phases = vec![vec![0i16; phase_len]; if factor > 1 { factor } else { 0 }];
    let dp = out.as_mut_ptr();
    let rows = |lo: usize, hi: usize| {
        for y in (lo..hi).step_by(2) {
            let pair_len = 2.min(hi - y);
            for (y, mid) in (y..y + pair_len).zip(mids.iter_mut()) {
                let win = plan.rows.get(y);
                vertical(
                    &src[win.start * width..win.end() * width],
                    width,
                    win,
                    &mut mid[..width],
                );
            }
            // SAFETY: rows y .. y + pair_len of `out`, which only this closure
            // call writes (with_prefault's contract).
            let dst = |y: usize| unsafe { std::slice::from_raw_parts_mut(dp.add(y * ow), ow) };
            match &plan.horizontal {
                #[cfg(target_arch = "x86_64")]
                Horizontal::Blocks(blocks) if pair_len == 2 && rows_in_pairs() => {
                    let [mid0, mid1] = &mids;
                    // SAFETY: avx512 detected; mids padded by BLOCK_SPAN.
                    unsafe {
                        crate::convolve::x86::horizontal_blocks_x2_avx512(
                            [mid0, mid1],
                            [dst(y), dst(y + 1)],
                            blocks,
                        );
                    };
                }
                Horizontal::Blocks(blocks) => {
                    for (y, mid) in (y..y + pair_len).zip(&mids) {
                        horizontal_blocks(mid, dst(y), blocks);
                    }
                }
                Horizontal::Phased {
                    factor,
                    interior,
                    taps,
                    edges,
                } => {
                    for (y, mid) in (y..y + pair_len).zip(&mids) {
                        split_phases(&mid[..width], *factor, &mut phases);
                        horizontal_phased(&phases, dst(y), interior.clone(), taps);
                        horizontal_blocks(mid, dst(y), edges);
                    }
                }
                #[cfg(target_arch = "x86_64")]
                Horizontal::SampleRuns(runs) if pair_len == 2 && rows_in_pairs() => {
                    let [mid0, mid1] = &mids;
                    // SAFETY: avx512 detected; mids padded by the runs' overreach.
                    unsafe {
                        crate::convolve::x86::horizontal_samples_x2_avx512(
                            [mid0, mid1],
                            [dst(y), dst(y + 1)],
                            runs,
                        );
                    };
                }
                Horizontal::SampleRuns(runs) => {
                    for (y, mid) in (y..y + pair_len).zip(&mids) {
                        let dst = dst(y);
                        for &output in &runs.outputs {
                            dst[output] = narrow(plan.cols.get(output).mulhrs_dot_at(mid, 1, 0));
                        }
                    }
                }
            }
            for (y, mid) in (y..y + pair_len).zip(&mids) {
                narrow_scalar_cols(plan, mid, dst(y), 1);
            }
        }
    };
    // SAFETY: `out` is this function's; `rows(lo, hi)` writes rows [lo, hi).
    unsafe { crate::cpu::with_prefault(dp, out.len(), ow, plan.rows.len(), rows) };
    out
}

/// The per-row interleaved path (the AVX-512 pixel gather, or the scalar
/// reference). 2- and 3-channel rows are widened to the RGBX layout between
/// the passes and the output rows packed back — or, with `rgbx_pad`, kept as
/// RGBX output with that 4th sample.
fn resize_pixel_rows(
    src: &[u8],
    width: usize,
    plan: &Plan,
    pixels: &PixelBlocks,
    rgbx_pad: Option<u8>,
) -> Vec<u8> {
    // Padded: the widening reads whole groups of pixels, up to a register past.
    const WIDEN_GROUP_PIXELS: usize = 8;
    let channels = plan.channels;
    let ow = plan.cols.len();
    let ws = width * channels;
    let ows = if rgbx_pad.is_some() {
        ow * RGBA
    } else {
        ow * channels
    };
    let mut out = vec![0u8; ows * plan.rows.len()];
    // Padded so a block's windows may run past the row end.
    let row_pad = plan
        .tap_runs
        .as_ref()
        .map_or(0, |tap_runs| tap_runs.runs.overreach)
        .max(pixels.overreach);
    // An odd last row through the AVX2 pixel kernel (which runs pairs) writes
    // its twin here.
    #[cfg(target_arch = "x86_64")]
    let mut spare_row = vec![
        0u8;
        if plan.pixels_avx2.is_some() {
            ow * RGBA
        } else {
            0
        }
    ];
    let padded = (width + row_pad) * RGBA;
    let mut mids = [vec![0i16; padded], vec![0i16; padded]];
    // Fewer than 4 channels: rows widened to the RGBX layout between the
    // passes (from `narrow_mid`), output rows packed back (from `rgbx_rows`)
    // unless the output is RGBX.
    let widened = channels < RGBA;
    let pack = widened && rgbx_pad.is_none();
    let narrow_pad = channels * width.next_multiple_of(WIDEN_GROUP_PIXELS) + AVX512_I16_LANES;
    let mut narrow_mid = vec![0i16; if widened { narrow_pad } else { 0 }];
    let mut rgbx_rows = [(); 2].map(|()| vec![0u8; if pack { ow * RGBA } else { 0 }]);
    let dp = out.as_mut_ptr();
    let rows = |lo: usize, hi: usize| {
        for y in (lo..hi).step_by(2) {
            let pair_len = 2.min(hi - y);
            for (y, mid) in (y..y + pair_len).zip(mids.iter_mut()) {
                let win = plan.rows.get(y);
                let rows = &src[win.start * ws..win.end() * ws];
                #[cfg(target_arch = "x86_64")]
                if channels == RGB && fused_rgbx(win, width) {
                    // SAFETY: avx512 detected (fused_rgbx); rows holds the window's
                    // rows of 3 * width samples, mid at least width pixels.
                    unsafe {
                        crate::convolve::x86::vertical_rgbx_avx512(rows, ws, win.q15, width, mid);
                    }
                    continue;
                }
                if widened {
                    vertical(rows, ws, win, &mut narrow_mid[..ws]);
                    widen_to_rgbx(&narrow_mid, width, mid);
                } else {
                    vertical(rows, ws, win, &mut mid[..ws]);
                }
            }
            let [rgbx0, rgbx1] = &mut rgbx_rows;
            let (rgbx_row0, rgbx_row1) = (rgbx0.as_mut_ptr(), rgbx1.as_mut_ptr());
            // SAFETY: rows y .. y + pair_len of `out`, which only this closure
            // call writes (with_prefault's contract); RGB: the two RGBX row
            // buffers.
            let dst = |y: usize| unsafe {
                let row = match (pack, y % 2) {
                    (false, _) => dp.add(y * ows),
                    (true, 0) => rgbx_row0,
                    (true, _) => rgbx_row1,
                };
                std::slice::from_raw_parts_mut(row, ow * RGBA)
            };
            #[cfg(target_arch = "x86_64")]
            let simd =
                horizontal_pixels_simd(plan, pixels, &mids, y, pair_len, dst, &mut spare_row);
            #[cfg(not(target_arch = "x86_64"))]
            let simd = false;
            if !simd {
                for (y, mid) in (y..y + pair_len).zip(&mids) {
                    horizontal_pixels(mid, dst(y), pixels);
                }
            }
            for (y, mid) in (y..y + pair_len).zip(&mids) {
                let dst = dst(y);
                narrow_scalar_cols(plan, mid, dst, RGBA);
                match rgbx_pad {
                    // SAFETY: row y of `out`, which only this closure call writes.
                    None if widened => pack_from_rgbx(dst, unsafe {
                        std::slice::from_raw_parts_mut(dp.add(y * ows), ows)
                    }),
                    // The 4th sample came out 0 (widened rows carry 0 there).
                    Some(pad) => dst
                        .as_chunks_mut::<RGBA>()
                        .0
                        .iter_mut()
                        .for_each(|px| px[3] = pad),
                    None => {}
                }
            }
        }
    };
    // SAFETY: `out` is this function's; `rows(lo, hi)` writes rows [lo, hi).
    unsafe { crate::cpu::with_prefault(dp, out.len(), ows, plan.rows.len(), rows) };
    out
}

/// The x86 SIMD horizontal passes of output rows y .. y + `pair_len` (one or
/// two) of the per-row interleaved path, from `mids`, into `dst(y)`: false
/// where none applies (the caller runs `horizontal_pixels`).
#[cfg(target_arch = "x86_64")]
fn horizontal_pixels_simd<'row>(
    plan: &Plan,
    pixels: &PixelBlocks,
    mids: &[Vec<i16>; 2],
    y: usize,
    pair_len: usize,
    dst: impl Fn(usize) -> &'row mut [u8],
    spare_row: &mut [u8],
) -> bool {
    let pairs = rows_in_pairs();
    if let (Some(up), true) = (&plan.upscale, pairs) {
        for (y, mid) in (y..y + pair_len).zip(mids) {
            let dst = dst(y);
            // SAFETY: avx512 detected; every interior window lies in the row.
            unsafe { crate::convolve::x86::horizontal_upscale_avx512(mid, dst, up) };
            for output in up
                .edges(plan.cols.len())
                .filter(|&output| plan.cols.get(output).fits_q15())
            {
                pixel_scalar(mid, dst, plan.cols.get(output), output);
            }
        }
        return true;
    }
    if let (Some(runs), true, 2) = (&plan.tap_runs, pairs, pair_len) {
        let [mid0, mid1] = mids;
        // SAFETY: avx512 detected; mids padded by the runs' overreach.
        unsafe {
            crate::convolve::x86::horizontal_taps_x2_avx512(
                [mid0, mid1],
                [dst(y), dst(y + 1)],
                &runs.runs,
            );
        };
        return true;
    }
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
    if pair_len == 2 && pairs {
        let [mid0, mid1] = mids;
        // SAFETY: avx512 detected; mids padded by `overreach` pixels.
        unsafe {
            crate::convolve::x86::horizontal_pixels_x2_avx512(
                [mid0, mid1],
                [dst(y), dst(y + 1)],
                pixels,
            );
        };
        return true;
    }
    if let (Some(pixels4), true) = (&plan.pixels_avx2, crate::cpu::has_avx2()) {
        let [mid0, mid1] = mids;
        let (mid1, dst1) = if pair_len == 2 {
            (mid1, dst(y + 1))
        } else {
            (mid0, &mut spare_row[..])
        };
        // SAFETY: avx2 detected; mids padded by `overreach` pixels.
        unsafe {
            crate::convolve::x86::horizontal_pixels4_x2_avx2([mid0, mid1], [dst(y), dst1], pixels4);
        };
        return true;
    }
    false
}

// ---- Pass kernels: dispatch + scalar reference -------------------------------

/// Whether an RGB row's vertical pass can write RGBX directly (AVX-512, the
/// SIMD weights, rows of at least one block).
#[cfg(target_arch = "x86_64")]
fn fused_rgbx(win: Window<'_>, width: usize) -> bool {
    crate::cpu::has_avx512() && win.fits_q15() && width >= FUSED_RGBX_BLOCK_PIXELS
}

/// Intermediate RGB samples to RGBX-layout pixels (X 0) for the pixel
/// kernels: `rgbx[4p + c] = src[3p + c]`.
fn widen_to_rgbx(src: &[i16], pixels: usize, rgbx: &mut [i16]) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_avx512() {
        // SAFETY: avx512bw detected; src padded past whole groups of 8 pixels.
        return unsafe { crate::convolve::x86::widen_to_rgbx_avx512(src, pixels, rgbx) };
    }
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_avx2() {
        // SAFETY: avx2 detected; src padded past whole groups of 8 pixels.
        return unsafe { crate::convolve::x86::widen_to_rgbx_avx2(src, pixels, rgbx) };
    }
    let (rgb_pixels, _) = src[..RGB * pixels].as_chunks::<RGB>();
    let (rgbx_pixels, _) = rgbx.as_chunks_mut::<RGBA>();
    for (px, out) in rgb_pixels.iter().zip(rgbx_pixels) {
        *out = [px[0], px[1], px[2], 0];
    }
}

/// RGBX-layout output pixels to RGB: `out[3p + c] = rgbx[4p + c]`.
fn pack_from_rgbx(rgbx: &[u8], out: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_avx512() {
        // SAFETY: avx512bw detected; rgbx holds out.len() / 3 pixels.
        return unsafe { crate::convolve::x86::pack_from_rgbx_avx512(rgbx, out) };
    }
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_avx2() {
        // SAFETY: avx2 detected; rgbx holds out.len() / 3 pixels.
        return unsafe { crate::convolve::x86::pack_from_rgbx_avx2(rgbx, out) };
    }
    let (rgbx_pixels, _) = rgbx.as_chunks::<RGBA>();
    let (rgb_pixels, _) = out.as_chunks_mut::<RGB>();
    for (px, out) in rgbx_pixels.iter().zip(rgb_pixels) {
        *out = [px[0], px[1], px[2]];
    }
}

/// Whether the per-row paths run the horizontal pass on two rows at once
/// (the AVX-512 kernels; the scalar reference goes row by row).
#[cfg(target_arch = "x86_64")]
fn rows_in_pairs() -> bool {
    crate::cpu::has_avx512()
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

/// `mid[x] = sum_t q15[t] * rows[t][x]` in Q6, `rows` = the window's source
/// rows.
fn vertical(rows: &[u8], width: usize, win: Window<'_>, mid: &mut [i16]) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_avx512() && win.fits_q15() && width >= AVX512_I16_LANES {
        // SAFETY: avx512f+bw detected; rows holds q15.len() rows of width, mid width.
        return unsafe { crate::convolve::x86::vertical_q6_avx512(rows, width, win.q15, mid) };
    }
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

/// `phases[p][m] = mid[factor*m + p]`.
fn split_phases(mid: &[i16], factor: usize, phases: &mut [Vec<i16>]) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_avx512() {
        // SAFETY: avx512f+bw detected; each phase holds len/factor + 32.
        return unsafe { crate::convolve::x86::split_phases_avx512(mid, factor, phases) };
    }
    for (index, group) in mid.chunks(factor).enumerate() {
        for (phase, &sample) in group.iter().enumerate() {
            phases[phase][index] = sample;
        }
    }
}

/// The phased plan's interior outputs from the phase arrays.
fn horizontal_phased(
    phases: &[Vec<i16>],
    dst: &mut [u8],
    interior: Range<usize>,
    taps: &[PhaseTap],
) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_avx512() && interior.len() >= BLOCK_OUTPUTS {
        // SAFETY: avx512f+bw detected; every tap index of `interior` lies in
        // its phase array (Plan: interior windows are full), dst covers it.
        return unsafe {
            crate::convolve::x86::horizontal_phased_avx512(phases, dst, interior, taps);
        };
    }
    for output in interior {
        let acc: i32 = taps
            .iter()
            .map(|tap| {
                mulhrs(
                    i32::from(phases[tap.phase][(output as isize + tap.rel) as usize]),
                    tap.weight,
                )
            })
            .sum();
        dst[output] = narrow(acc);
    }
}

/// One RGBA-layout row by pixel blocks.
fn horizontal_pixels(mid: &[i16], dst: &mut [u8], pixels: &PixelBlocks) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_avx512() {
        // SAFETY: avx512f+bw+vl detected; mid is padded by `overreach`
        // pixels, dst covers every block.
        return unsafe { crate::convolve::x86::horizontal_pixels_avx512(mid, dst, pixels) };
    }
    for block in &pixels.blocks {
        for (slot, ch) in (0..block.count).flat_map(|slot| (0..RGBA).map(move |ch| (slot, ch))) {
            let lane = slot * RGBA + ch;
            let taps = pixels.weights[block.taps.clone()].iter().enumerate();
            let acc: i32 = taps
                .map(|(tap, weights)| {
                    mulhrs(
                        i32::from(mid[(block.base + block.idx[slot] as usize + tap) * RGBA + ch]),
                        weights[lane],
                    )
                })
                .sum();
            dst[(block.o0 + slot) * RGBA + ch] = narrow(acc);
        }
    }
}

/// Output pixel `output` of an RGBA row by the scalar reference (Q15 weights).
#[cfg(target_arch = "x86_64")]
fn pixel_scalar(mid: &[i16], dst: &mut [u8], win: Window<'_>, output: usize) {
    for ch in 0..RGBA {
        dst[output * RGBA + ch] = narrow(win.mulhrs_dot_at(mid, RGBA, ch));
    }
}

/// One planar row's block outputs.
fn horizontal_blocks(mid: &[i16], dst: &mut [u8], blocks: &[Block]) {
    #[cfg(target_arch = "x86_64")]
    if crate::cpu::has_avx512() {
        // SAFETY: avx512f+bw+vl detected; mid is padded by BLOCK_SPAN past
        // every block base, dst covers every block.
        return unsafe { crate::convolve::x86::horizontal_blocks_avx512(mid, dst, blocks) };
    }
    for block in blocks {
        for slot in 0..block.count {
            let taps = block.idx.iter().zip(&block.weights);
            let acc: i32 = taps
                .map(|(idx, weights)| {
                    mulhrs(
                        i32::from(mid[block.base + idx[slot] as usize]),
                        weights[slot],
                    )
                })
                .sum();
            dst[block.o0 + slot] = narrow(acc);
        }
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
    /// smallest images, block ratios around the vector widths, the phased
    /// factors 2 to 4 (interiors past one register), integer upscales x2 and
    /// x4, and reductions past the block plans.
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

    /// The features of every AVX-512 kernel here.
    #[cfg(target_arch = "x86_64")]
    fn has_avx512() -> bool {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
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
    fn reference_resize(src: &[u8], width: usize, plan: &Plan, rgbx_pad: Option<u8>) -> Vec<u8> {
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
                out.extend(rgbx_pad);
            }
        }
        out
    }

    fn assert_resize_matches_reference(channels: usize, rgbx_pad: Option<u8>) {
        let mut rng = Rng::new(channels as u64);
        for kernel in KERNELS {
            for (width, ow) in COLUMN_CASES {
                for (height, oh) in ROW_CASES {
                    let src: Vec<u8> = random_samples(&mut rng, width * height * channels);
                    let plan = Plan::new((width, height), (ow, oh), kernel, channels);
                    assert_same(
                        &resize_u8(&src, (width, height), &plan, rgbx_pad),
                        &reference_resize(&src, width, &plan, rgbx_pad),
                        &format!(
                            "{} {channels} channels {width} x {height} -> {ow} x {oh}, pad \
                             {rgbx_pad:?}",
                            kernel_name(kernel)
                        ),
                    );
                }
            }
        }
    }

    #[test]
    fn planar_resize_equals_reference() {
        assert_resize_matches_reference(1, None);
    }

    #[test]
    fn rgb_resize_equals_reference() {
        assert_resize_matches_reference(RGB, None);
    }

    #[test]
    fn rgb_to_rgbx_resize_equals_reference() {
        assert_resize_matches_reference(RGB, Some(0x7F));
    }

    #[test]
    fn rgba_resize_equals_reference() {
        assert_resize_matches_reference(RGBA, None);
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
                    &resize_u8(&src, (width, step_rows.len()), &plan, None),
                    &reference_resize(&src, width, &plan, None),
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
                    reached.insert(match &plan.horizontal {
                        Horizontal::Phased { interior, .. } if interior.len() >= BLOCK_OUTPUTS => {
                            "phased, a register or more"
                        }
                        Horizontal::Phased { .. } => "phased",
                        Horizontal::Blocks(_) => "blocks",
                        Horizontal::SampleRuns(_) => "sample runs",
                    });
                    if plan.upscale.is_some() {
                        reached.insert("integer upscale");
                    }
                    if plan.tap_runs.is_some() {
                        reached.insert("tap runs");
                    }
                    if !plan.scalar_cols.is_empty() {
                        reached.insert("scalar columns");
                    }
                    for rgbx_pad in [None, Some(0)]
                        .into_iter()
                        .take(1 + usize::from(channels == RGB))
                    {
                        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                        let transposed = takes_transposed(&plan, width, rgbx_pad);
                        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                        let transposed = false;
                        reached.insert(match (transposed, channels, rgbx_pad) {
                            (true, 1, _) => "transposed planar",
                            (true, RGB, _) => "transposed RGB",
                            (true, _, _) => "transposed RGBA",
                            (false, 1, _) => "per-row planar",
                            (false, RGB, None) => "per-row RGB",
                            (false, RGB, Some(_)) => "per-row RGB to RGBX",
                            (false, _, _) => "per-row RGBA",
                        });
                    }
                }
            }
        }
        let mut expected = vec![
            "phased",
            "phased, a register or more",
            "blocks",
            "sample runs",
            "integer upscale",
            "tap runs",
            "scalar columns",
            "per-row planar",
            "per-row RGB",
            "per-row RGB to RGBX",
            "per-row RGBA",
        ];
        #[cfg(target_arch = "x86_64")]
        if has_simd_rows() {
            expected.extend(["transposed planar", "transposed RGB"]);
        }
        #[cfg(target_arch = "aarch64")]
        if has_simd_rows() {
            expected.extend(["transposed planar", "transposed RGB", "transposed RGBA"]);
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

    #[cfg(target_arch = "x86_64")]
    fn block_outputs(blocks: &[Block]) -> impl Iterator<Item = usize> + '_ {
        blocks
            .iter()
            .flat_map(|block| block.o0..block.o0 + block.count)
    }

    /// The planar horizontal kernels (blocks, phased with its phase split,
    /// sample runs) on rows padded as `resize_rows` pads them.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn planar_horizontal_avx512_equals_reference() {
        if !has_avx512() {
            return;
        }
        let mut rng = Rng::new(5);
        for kernel in KERNELS {
            for (width, ow) in COLUMN_CASES {
                let plan = Plan::new((width, 1), (ow, 1), kernel, 1);
                let context = format!("{} {width} -> {ow}", kernel_name(kernel));
                let row_pad = match &plan.horizontal {
                    Horizontal::SampleRuns(runs) => runs.overreach.max(BLOCK_SPAN),
                    _ => BLOCK_SPAN,
                };
                let mids = [
                    random_mid(&mut rng, width + row_pad),
                    random_mid(&mut rng, width + row_pad),
                ];
                let mut dsts = [vec![SENTINEL; ow], vec![SENTINEL; ow]];
                let [dst0, dst1] = &mut dsts;
                let covered = match &plan.horizontal {
                    Horizontal::Blocks(blocks) => {
                        let mut single = vec![SENTINEL; ow];
                        // SAFETY: AVX-512 F/BW/VL detected; the mids are padded
                        // by BLOCK_SPAN past every block base, the dsts hold
                        // every output.
                        unsafe {
                            x86::horizontal_blocks_avx512(&mids[0], &mut single, blocks);
                            x86::horizontal_blocks_x2_avx512(
                                [&mids[0], &mids[1]],
                                [dst0, dst1],
                                blocks,
                            );
                        }
                        let covered = covered_by(ow, block_outputs(blocks));
                        assert_row(&single, &mids[0], &plan.cols, 1, &covered, &context);
                        covered
                    }
                    Horizontal::Phased {
                        factor,
                        interior,
                        taps,
                        edges,
                    } => {
                        let phase_len = width.div_ceil(*factor) + BLOCK_OUTPUTS;
                        for (mid, dst) in mids.iter().zip([dst0, dst1]) {
                            let mut phases = vec![vec![0i16; phase_len]; *factor];
                            // SAFETY: AVX-512 F/BW/VL detected; each phase holds
                            // width / factor + BLOCK_OUTPUTS samples.
                            unsafe {
                                x86::split_phases_avx512(&mid[..width], *factor, &mut phases);
                            };
                            for (index, &sample) in mid[..width].iter().enumerate() {
                                assert_eq!(
                                    phases[index % factor][index / factor],
                                    sample,
                                    "{context}"
                                );
                            }
                            if interior.len() >= BLOCK_OUTPUTS {
                                // SAFETY: AVX-512 F/BW detected; interior windows
                                // are full (Plan) and the phases padded by a
                                // register, dst holds every output.
                                unsafe {
                                    x86::horizontal_phased_avx512(
                                        &phases,
                                        dst,
                                        interior.clone(),
                                        taps,
                                    );
                                }
                            } else {
                                horizontal_phased(&phases, dst, interior.clone(), taps);
                            }
                            // SAFETY: as for the blocks plan.
                            unsafe { x86::horizontal_blocks_avx512(mid, dst, edges) };
                        }
                        covered_by(ow, interior.clone().chain(block_outputs(edges)))
                    }
                    Horizontal::SampleRuns(runs) => {
                        // SAFETY: AVX-512 F/BW/VL detected; the mids are padded
                        // by the runs' overreach, the dsts hold every output.
                        unsafe {
                            x86::horizontal_samples_x2_avx512(
                                [&mids[0], &mids[1]],
                                [dst0, dst1],
                                runs,
                            );
                        }
                        covered_by(ow, runs.outputs.iter().copied())
                    }
                };
                for (mid, dst) in mids.iter().zip(&dsts) {
                    assert_row(dst, mid, &plan.cols, 1, &covered, &context);
                }
            }
        }
    }

    /// The RGBA-layout horizontal kernels of both tiers (pixel blocks, tap
    /// runs, integer upscale) on rows padded as `resize_pixel_rows` pads
    /// them, for every case, whichever the plan would choose.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn rgba_horizontal_kernels_equal_reference() {
        let (avx2, avx512) = (has_avx2(), has_avx512());
        let mut rng = Rng::new(6);
        for kernel in KERNELS {
            for (width, ow) in COLUMN_CASES {
                let cols = Windows::new(width, ow, kernel);
                let context = format!("{} {width} -> {ow}", kernel_name(kernel));
                let pixels = PixelBlocks::new(&cols);
                let pixels4 = PixelBlocks::with_size(&cols, AVX2_PIXEL_OUTPUTS, AVX2_PIXEL_SPAN);
                let runs = Runs::new(&cols, width, PIXEL_OUTPUTS);
                let mut padded_mids = |pad: usize| {
                    [
                        random_mid(&mut rng, (width + pad) * RGBA),
                        random_mid(&mut rng, (width + pad) * RGBA),
                    ]
                };
                let block_mids = padded_mids(pixels.overreach);
                let run_mids = padded_mids(runs.overreach.max(pixels.overreach));
                let fresh = || [vec![SENTINEL; ow * RGBA], vec![SENTINEL; ow * RGBA]];
                let assert_rows =
                    |mids: &[Vec<i16>; 2], dsts: &[Vec<u8>; 2], covered: &[bool], name: &str| {
                        for (mid, dst) in mids.iter().zip(dsts) {
                            assert_row(
                                dst,
                                mid,
                                &cols,
                                RGBA,
                                covered,
                                &format!("{context}, {name}"),
                            );
                        }
                    };
                let block_covered = covered_by(
                    ow,
                    pixels
                        .blocks
                        .iter()
                        .flat_map(|block| block.o0..block.o0 + block.count),
                );
                let run_covered = covered_by(ow, runs.outputs.iter().copied());
                if avx512 {
                    let mut single = fresh();
                    let mut pair = fresh();
                    let mut taps = fresh();
                    // SAFETY: AVX-512 F/BW/VL detected; the mids are padded by
                    // the blocks' and the runs' overreach, the dsts hold every
                    // output pixel.
                    unsafe {
                        x86::horizontal_pixels_avx512(&block_mids[0], &mut single[0], &pixels);
                        let [dst0, dst1] = &mut pair;
                        x86::horizontal_pixels_x2_avx512(
                            [&block_mids[0], &block_mids[1]],
                            [dst0, dst1],
                            &pixels,
                        );
                        let [dst0, dst1] = &mut taps;
                        x86::horizontal_taps_x2_avx512(
                            [&run_mids[0], &run_mids[1]],
                            [dst0, dst1],
                            &runs,
                        );
                    }
                    assert_row(
                        &single[0],
                        &block_mids[0],
                        &cols,
                        RGBA,
                        &block_covered,
                        &format!("{context}, pixels"),
                    );
                    assert_rows(&block_mids, &pair, &block_covered, "pixels x2");
                    assert_rows(&run_mids, &taps, &run_covered, "taps x2");
                    if let Some(up) = IntegerUpscale::new(&cols, width) {
                        let mut upscaled = fresh();
                        // SAFETY: AVX-512 F/BW/VL detected; every interior
                        // window lies in the row (IntegerUpscale), dst holds
                        // every output pixel.
                        unsafe {
                            x86::horizontal_upscale_avx512(&block_mids[0], &mut upscaled[0], &up);
                        };
                        let interior = up.factor * up.interior.start..up.factor * up.interior.end;
                        assert_row(
                            &upscaled[0],
                            &block_mids[0],
                            &cols,
                            RGBA,
                            &covered_by(ow, interior),
                            &format!("{context}, upscale"),
                        );
                    }
                }
                if avx2 {
                    let mut pair = fresh();
                    let mut taps = fresh();
                    // SAFETY: AVX2 detected; blocks of at most
                    // AVX2_PIXEL_OUTPUTS, the mids padded as above (the 8-pixel
                    // blocks' overreach covers the 4-pixel ones').
                    unsafe {
                        let [dst0, dst1] = &mut pair;
                        x86::horizontal_pixels4_x2_avx2(
                            [&block_mids[0], &block_mids[1]],
                            [dst0, dst1],
                            &pixels4,
                        );
                        let [dst0, dst1] = &mut taps;
                        x86::horizontal_taps_x2_avx2(
                            [&run_mids[0], &run_mids[1]],
                            [dst0, dst1],
                            &runs,
                        );
                    }
                    let covered4 = covered_by(
                        ow,
                        pixels4
                            .blocks
                            .iter()
                            .flat_map(|block| block.o0..block.o0 + block.count),
                    );
                    assert_rows(&block_mids, &pair, &covered4, "pixels4 x2 avx2");
                    assert_rows(&run_mids, &taps, &run_covered, "taps x2 avx2");
                }
            }
        }
    }

    /// The vertical kernels of every tier: widths around their registers and
    /// lines, rows starting off the 32-byte boundary the AVX-512 kernel
    /// aligns its lines to, and the strides and window lengths where the
    /// taps run in groups (4 KB aliasing, 2 KB on AVX2, 40+ taps).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn vertical_kernels_equal_reference() {
        if !has_simd_rows() {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        let avx512 = has_avx512();
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
                #[cfg(target_arch = "x86_64")]
                if avx512 && width >= BLOCK_OUTPUTS {
                    let mut mid = vec![SENTINEL_I16; width];
                    // SAFETY: AVX-512 F/BW detected; rows holds taps rows of
                    // width >= 32 samples, mid holds width.
                    unsafe { x86::vertical_q6_avx512(rows, width, &weights, &mut mid) };
                    assert_same(&mid, &expected, &format!("{context}, avx512"));
                }
            }
        }
    }

    /// The RGB vertical pass written as RGBX, on rows of exactly the length
    /// its contract requires.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vertical_rgbx_avx512_equals_reference() {
        if !has_avx512() {
            return;
        }
        let mut rng = Rng::new(8);
        for pixels in [32, 33, 47, 64, 67, 100, 130] {
            for taps in [1, 2, 5, 17] {
                for stride_pad in [0, 5] {
                    let stride = RGB * pixels + stride_pad;
                    let weights = random_weights(&mut rng, taps);
                    let rows: Vec<u8> =
                        random_samples(&mut rng, (taps - 1) * stride + RGB * pixels);
                    let expected: Vec<i16> =
                        reference_vertical_q15(&rows, stride, &weights, RGB * pixels)
                            .as_chunks::<RGB>()
                            .0
                            .iter()
                            .flat_map(|px| [px[0], px[1], px[2], 0])
                            .collect();
                    let mut rgbx = vec![SENTINEL_I16; pixels * RGBA];
                    // SAFETY: AVX-512 F/BW detected; taps >= 1, pixels >= 32,
                    // rows and rgbx sized as the contract requires.
                    unsafe {
                        x86::vertical_rgbx_avx512(&rows, stride, &weights, pixels, &mut rgbx);
                    };
                    assert_same(
                        &rgbx,
                        &expected,
                        &format!("{pixels} pixels, {taps} taps, stride {stride}"),
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn widen_to_rgbx_kernels_equal_reference() {
        let (avx2, avx512) = (has_avx2(), has_avx512());
        let mut rng = Rng::new(9);
        for pixels in 1..=70 {
            // Each tier's minimum padding: whole groups of 8 (AVX-512) or 4
            // pixels, and a register's read past them.
            let mut check = |groups: usize,
                             overread: usize,
                             kernel: unsafe fn(&[i16], usize, &mut [i16]),
                             name: &str| {
                let src = random_mid(&mut rng, RGB * groups + overread);
                let mut rgbx = vec![SENTINEL_I16; RGBA * groups];
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
                    &rgbx[..RGBA * pixels],
                    &expected,
                    &format!("{pixels} pixels, {name}"),
                );
            };
            if avx512 {
                let group = AVX512_I16_LANES / RGBA;
                check(
                    pixels.next_multiple_of(group),
                    AVX512_I16_LANES,
                    x86::widen_to_rgbx_avx512,
                    "avx512",
                );
            }
            if avx2 {
                let (group, overread) = (AVX2_I16_LANES / RGBA, AVX2_I16_LANES / 2);
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
        let (avx2, avx512) = (has_avx2(), has_avx512());
        let mut rng = Rng::new(10);
        for pixels in 1..=70 {
            let rgbx: Vec<u8> = random_samples(&mut rng, RGBA * pixels);
            let expected: Vec<u8> = rgbx
                .as_chunks::<RGBA>()
                .0
                .iter()
                .flat_map(|px| [px[0], px[1], px[2]])
                .collect();
            let tiers: [(bool, PackKernel, &str); 2] = [
                (avx512, x86::pack_from_rgbx_avx512, "avx512"),
                (avx2, x86::pack_from_rgbx_avx2, "avx2"),
            ];
            for (available, kernel, name) in tiers {
                if available {
                    let mut out = vec![SENTINEL; RGB * pixels];
                    // SAFETY: the kernel's features detected; rgbx holds RGBA
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

//! Exact x4 bilinear reduction of u8 images. At exactly a quarter of the
//! size the bilinear weights are 1, 3, 5, 7, 7, 5, 3, 1 over 32 per axis
//! (renormalized where the window is clipped at the image edge), so each
//! output is an integer sum over the samples, divided once.

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod x86;

/// The bilinear weights at x4 (x 32): output o reads inputs 4o - 2 ..= 4o + 5.
const TAPS: [u32; 8] = [1, 3, 5, 7, 7, 5, 3, 1];

/// The window of output `output` along an axis of `axis_len` inputs: its
/// first input and its weights, clipped to the axis.
fn window(output: usize, axis_len: usize) -> (usize, &'static [u32]) {
    let lo = 4 * output as isize - 2;
    let start = lo.max(0) as usize;
    let end = ((lo + 8) as usize).min(axis_len);
    (
        start,
        &TAPS[(start as isize - lo) as usize..(end as isize - lo) as usize],
    )
}

/// `numerator / denominator` rounded half up (both non-negative).
fn divide_rounded(numerator: u32, denominator: u32) -> u8 {
    ((2 * numerator + denominator) / (2 * denominator)) as u8
}

/// The scalar reference for any channel count.
pub(crate) fn bilinear_x4_scalar(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    let (ow, oh, ws) = (width / 4, height / 4, width * channels);
    let mut out = vec![0u8; ow * oh * channels];
    let mut sums = vec![0u32; ws];
    for y in 0..oh {
        let dst = &mut out[y * ow * channels..(y + 1) * ow * channels];
        row_scalar(src, (width, height, channels), y, &mut sums, dst);
    }
    out
}

/// Output row `y` by the scalar reference, `sums` holding a row's samples.
fn row_scalar(
    src: &[u8],
    (width, height, channels): (usize, usize, usize),
    y: usize,
    sums: &mut [u32],
    dst: &mut [u8],
) {
    let ws = width * channels;
    let (r0, wv) = window(y, height);
    // A row at a time, so the loop vectorizes.
    sums.fill(0);
    for (tap, &weight) in wv.iter().enumerate() {
        for (sum, &sample) in sums.iter_mut().zip(&src[(r0 + tap) * ws..][..ws]) {
            *sum += weight * u32::from(sample);
        }
    }
    let zv: u32 = wv.iter().sum();
    for output in 0..width / 4 {
        output_scalar(sums, (width, channels), zv, output, dst);
    }
}

/// Output pixel `output` of a row of vertical sums (`sums[x]` per sample,
/// weight total `zv`).
fn output_scalar<S: Copy + Into<u32>>(
    sums: &[S],
    (width, channels): (usize, usize),
    zv: u32,
    output: usize,
    dst: &mut [u8],
) {
    let (x0, wh) = window(output, width);
    let denominator = zv * wh.iter().sum::<u32>();
    for ch in 0..channels {
        let numerator: u32 = wh
            .iter()
            .enumerate()
            .map(|(tap, &weight)| weight * sums[(x0 + tap) * channels + ch].into())
            .sum();
        dst[output * channels + ch] = divide_rounded(numerator, denominator);
    }
}

/// The exact x4 bilinear reduction (each tier computes the same result).
pub(crate) fn bilinear_x4(src: &[u8], width: usize, height: usize, channels: usize) -> Vec<u8> {
    crate::sample::assert_image(src, width, height, channels);
    assert!(
        width.is_multiple_of(4) && height.is_multiple_of(4),
        "the exact x4 kernel takes sizes that are multiples of 4"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if crate::cpu::has_avx512() {
            // SAFETY: avx512f+bw+vl detected; src holds width * height *
            // channels samples, width and height multiples of 4 (the exact
            // kernel's condition), channels 1, 3 or 4 (the caller's shape
            // check).
            return unsafe {
                match channels {
                    1 => x86::bilinear_x4_avx512::<1, false>(src, width, height),
                    3 => x86::bilinear_x4_avx512::<4, true>(src, width, height),
                    _ => x86::bilinear_x4_avx512::<4, false>(src, width, height),
                }
            };
        }
        if crate::cpu::has_avx2() {
            // SAFETY: avx2 detected; src holds width * height * channels
            // samples, width and height multiples of 4 (the exact kernel's
            // condition), channels 1, 3 or 4 (the caller's shape check).
            return unsafe {
                match channels {
                    1 => x86::bilinear_x4_avx2::<1, false>(src, width, height),
                    3 => x86::bilinear_x4_avx2::<4, true>(src, width, height),
                    _ => x86::bilinear_x4_avx2::<4, false>(src, width, height),
                }
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::use_neon() {
        // SAFETY: NEON is baseline on aarch64; src holds width * height *
        // channels samples.
        return unsafe { neon::bilinear_x4_neon(src, width, height, channels) };
    }
    bilinear_x4_scalar(src, width, height, channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::tests::{assert_equals_reference, size_grid};
    use crate::testing::CHANNELS;

    /// Widths (multiples of 4) around the AVX2 and AVX-512 steps (16 and 32
    /// samples per horizontal step, so 64 and 128 gray input pixels) and
    /// their scalar and AVX2 fallbacks; heights from one output row (the
    /// window clipped at both edges) up.
    pub(super) const WIDTHS: [usize; 18] = [
        4, 8, 12, 16, 20, 28, 32, 36, 44, 60, 64, 68, 124, 128, 132, 136, 260, 516,
    ];
    pub(super) const HEIGHTS: [usize; 5] = [4, 8, 12, 16, 20];

    #[test]
    fn dispatch_equals_scalar() {
        assert_equals_reference(
            size_grid(&WIDTHS, &HEIGHTS),
            &CHANNELS,
            u8::MAX,
            &bilinear_x4,
            bilinear_x4_scalar,
        );
    }

    #[test]
    #[should_panic(expected = "channels")]
    fn dispatch_rejects_two_channels() {
        bilinear_x4(&[0; 8 * 8 * 2], 8, 8, 2);
    }

    #[test]
    #[should_panic(expected = "multiples of 4")]
    fn dispatch_rejects_width_not_multiple_of_four() {
        bilinear_x4(&[0; 6 * 8], 6, 8, 1);
    }

    #[test]
    #[should_panic(expected = "multiples of 4")]
    fn dispatch_rejects_height_not_multiple_of_four() {
        bilinear_x4(&[0; 8 * 10], 8, 10, 1);
    }

    #[test]
    #[should_panic(expected = "fewer than")]
    fn dispatch_rejects_short_source() {
        bilinear_x4(&[0; 8 * 8 - 1], 8, 8, 1);
    }
}

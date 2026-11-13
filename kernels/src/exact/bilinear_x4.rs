//! Exact x4 bilinear reduction. At exactly a quarter of the size the bilinear
//! weights are integers over 32 per axis (renormalized where the window is
//! clipped at the image edge), so each output is an integer sum divided once.

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod x86;

/// The bilinear weights at x4 (x 32): output o reads inputs 4o - 2 ..= 4o + 5.
const TAPS: [u32; 8] = [1, 3, 5, 7, 7, 5, 3, 1];

/// The first input and the weights of output `output`'s window, clipped to an
/// axis of `axis_len` inputs.
fn window(output: usize, axis_len: usize) -> (usize, &'static [u32]) {
    let unclipped_start = 4 * output as isize - 2;
    let start = unclipped_start.max(0) as usize;
    let end = ((unclipped_start + 8) as usize).min(axis_len);
    (
        start,
        &TAPS[(start as isize - unclipped_start) as usize
            ..(end as isize - unclipped_start) as usize],
    )
}

/// `numerator / denominator` rounded half up (both non-negative).
fn divide_rounded(numerator: u32, denominator: u32) -> u8 {
    ((2 * numerator + denominator) / (2 * denominator)) as u8
}

/// The scalar reference, for any channel count.
pub(crate) fn bilinear_x4_scalar(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    let (out_width, out_height, row_samples) = (width / 4, height / 4, width * channels);
    let mut out = vec![0u8; out_width * out_height * channels];
    let mut sums = vec![0u32; row_samples];
    for y in 0..out_height {
        let dst = &mut out[y * out_width * channels..(y + 1) * out_width * channels];
        row_scalar(src, (width, height, channels), y, &mut sums, dst);
    }
    out
}

/// Output row `y` by the scalar reference; `sums` holds a row's samples.
fn row_scalar(
    src: &[u8],
    (width, height, channels): (usize, usize, usize),
    y: usize,
    sums: &mut [u32],
    dst: &mut [u8],
) {
    let row_samples = width * channels;
    let (first_row, vertical_weights) = window(y, height);
    // A row at a time, so the loop vectorizes.
    sums.fill(0);
    for (tap, &weight) in vertical_weights.iter().enumerate() {
        for (sum, &sample) in sums
            .iter_mut()
            .zip(&src[(first_row + tap) * row_samples..][..row_samples])
        {
            *sum += weight * u32::from(sample);
        }
    }
    let vertical_total: u32 = vertical_weights.iter().sum();
    for output in 0..width / 4 {
        output_scalar(sums, (width, channels), vertical_total, output, dst);
    }
}

/// Output pixel `output` of a row of vertical sums.
fn output_scalar<S: Copy + Into<u32>>(
    sums: &[S],
    (width, channels): (usize, usize),
    vertical_total: u32,
    output: usize,
    dst: &mut [u8],
) {
    let (first_column, horizontal_weights) = window(output, width);
    let denominator = vertical_total * horizontal_weights.iter().sum::<u32>();
    for channel in 0..channels {
        let numerator: u32 = horizontal_weights
            .iter()
            .enumerate()
            .map(|(tap, &weight)| weight * sums[(first_column + tap) * channels + channel].into())
            .sum();
        dst[output * channels + channel] = divide_rounded(numerator, denominator);
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
        if crate::cpu::has_avx2() {
            // SAFETY: avx2 detected; src holds width * height * channels
            // samples, width and height multiples of 4, channels 1 or 3
            // (checked).
            return unsafe {
                match channels {
                    1 => x86::bilinear_x4_avx2::<false>(src, width, height),
                    _ => x86::bilinear_x4_avx2::<true>(src, width, height),
                }
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::use_neon() {
        // SAFETY: src holds width * height * channels samples.
        return unsafe { neon::bilinear_x4_neon(src, width, height, channels) };
    }
    bilinear_x4_scalar(src, width, height, channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::tests::{assert_equals_reference, size_grid};
    use crate::testing::CHANNELS;

    /// Widths (multiples of 4) around the SIMD steps (64 gray input pixels
    /// and twice that) and the scalar fallback; heights from one output row
    /// (both edges clipped) up.
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

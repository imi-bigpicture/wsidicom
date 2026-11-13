//! Exact x2 bilinear reduction. At exactly half size the bilinear weights are
//! 1/8 and 3/8 per axis (renormalized where the window is clipped at the image
//! edge), so each output is an integer sum divided once.

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
#[cfg(target_arch = "x86_64")]
mod x86;

/// The bilinear weights at x2 (x 8): output o reads inputs 2o - 1 ..= 2o + 2.
const TAPS: [u32; 4] = [1, 3, 3, 1];

/// The first input and the weights of output `output`'s window, clipped to an
/// axis of `axis_len` inputs.
pub(crate) fn window(output: usize, axis_len: usize) -> (usize, &'static [u32]) {
    let unclipped_start = 2 * output as isize - 1;
    let start = unclipped_start.max(0) as usize;
    let end = ((unclipped_start + 4) as usize).min(axis_len);
    (
        start,
        &TAPS[(start as isize - unclipped_start) as usize
            ..(end as isize - unclipped_start) as usize],
    )
}

/// `numerator / denominator` rounded half up (both non-negative). The
/// denominators are products of window totals 7 and 8; as constants the
/// divisions compile to multiplies.
fn divide_rounded(numerator: u32, denominator: u32) -> u32 {
    let rounded = |divisor: u32| (2 * numerator + divisor) / (2 * divisor);
    match denominator {
        64 => rounded(64),
        56 => rounded(56),
        49 => rounded(49),
        divisor => rounded(divisor),
    }
}

/// A sample type whose window sums (at most 64 x the largest sample) fit u32.
pub(crate) trait X2Sample: Copy + Default + Into<u32> {
    /// A rounded mean, which is within the type's range.
    fn narrow(mean: u32) -> Self;
}

impl X2Sample for u8 {
    fn narrow(mean: u32) -> u8 {
        mean as u8
    }
}

impl X2Sample for u16 {
    fn narrow(mean: u32) -> u16 {
        mean as u16
    }
}

/// The scalar reference, for any channel count.
pub(crate) fn bilinear_x2_scalar<T: X2Sample>(
    src: &[T],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<T> {
    let (out_width, out_height, row_samples) = (width / 2, height / 2, width * channels);
    let mut out = vec![T::default(); out_width * out_height * channels];
    let mut sums = vec![0u32; row_samples];
    for y in 0..out_height {
        let (first_row, vertical_weights) = window(y, height);
        vertical_sums_scalar(
            &src[first_row * row_samples..],
            row_samples,
            vertical_weights,
            &mut sums,
        );
        let vertical_total: u32 = vertical_weights.iter().sum();
        horizontal_scalar(
            &sums,
            channels,
            width,
            vertical_total,
            0..out_width,
            &mut out[y * out_width * channels..(y + 1) * out_width * channels],
        );
    }
    out
}

/// The weighted sums of the window's rows, a row at a time so the loop
/// vectorizes.
fn vertical_sums_scalar<T: X2Sample>(rows: &[T], stride: usize, weights: &[u32], sums: &mut [u32]) {
    sums.fill(0);
    for (tap, &weight) in weights.iter().enumerate() {
        let row = &rows[tap * stride..][..sums.len()];
        for (sum, &sample) in sums.iter_mut().zip(row) {
            *sum += weight * sample.into();
        }
    }
}

/// Outputs `outputs` of a row of vertical sums.
pub(crate) fn horizontal_scalar<S: Copy + Into<u32>, T: X2Sample>(
    sums: &[S],
    channels: usize,
    width: usize,
    vertical_total: u32,
    outputs: std::ops::Range<usize>,
    dst: &mut [T],
) {
    for output in outputs {
        let (first_column, horizontal_weights) = window(output, width);
        let denominator = vertical_total * horizontal_weights.iter().sum::<u32>();
        for channel in 0..channels {
            let numerator: u32 = horizontal_weights
                .iter()
                .enumerate()
                .map(|(tap, &weight)| {
                    weight * sums[(first_column + tap) * channels + channel].into()
                })
                .sum();
            dst[output * channels + channel] = T::narrow(divide_rounded(numerator, denominator));
        }
    }
}

/// The exact x2 bilinear reduction (each tier computes the same result).
pub(crate) fn bilinear_x2(src: &[u8], width: usize, height: usize, channels: usize) -> Vec<u8> {
    crate::sample::assert_image(src, width, height, channels);
    assert!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "the exact x2 kernels take even sizes"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if crate::cpu::has_avx2() {
            // SAFETY: avx2 detected; src holds width * height * channels
            // samples.
            return unsafe { x86::bilinear_x2_avx2(src, width, height, channels) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::use_neon() {
        // SAFETY: src holds width * height * channels samples.
        return unsafe { neon::bilinear_x2_neon(src, width, height, channels) };
    }
    bilinear_x2_scalar(src, width, height, channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::tests::{assert_equals_reference, size_grid};
    use crate::testing::CHANNELS;

    /// Even widths around the SIMD steps (16 samples and twice that) and the
    /// scalar fallbacks of narrow rows; even heights from one output row
    /// (both edges clipped) up.
    pub(super) const WIDTHS: [usize; 20] = [
        2, 4, 6, 8, 10, 12, 16, 22, 30, 32, 34, 36, 62, 64, 66, 70, 126, 130, 134, 262,
    ];
    pub(super) const HEIGHTS: [usize; 5] = [2, 4, 6, 8, 12];

    #[test]
    fn dispatch_equals_scalar() {
        assert_equals_reference(
            size_grid(&WIDTHS, &HEIGHTS),
            &CHANNELS,
            u8::MAX,
            &bilinear_x2,
            bilinear_x2_scalar,
        );
    }

    #[test]
    #[should_panic(expected = "channels")]
    fn dispatch_rejects_two_channels() {
        bilinear_x2(&[0; 8 * 8 * 2], 8, 8, 2);
    }

    #[test]
    #[should_panic(expected = "even sizes")]
    fn dispatch_rejects_odd_width() {
        bilinear_x2(&[0; 7 * 8], 7, 8, 1);
    }

    #[test]
    #[should_panic(expected = "even sizes")]
    fn dispatch_rejects_odd_height() {
        bilinear_x2(&[0; 8 * 7], 8, 7, 1);
    }

    #[test]
    #[should_panic(expected = "fewer than")]
    fn dispatch_rejects_short_source() {
        bilinear_x2(&[0; 8 * 8 * 3 - 1], 8, 8, 3);
    }
}

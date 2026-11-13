//! Exact x2 catmullrom reduction. At exactly half size the catmullrom weights
//! are integers over 256 per axis (renormalized where the window is clipped
//! at the image edge), so each output is an integer sum divided once and
//! clamped.

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod x86;

/// The catmullrom weights at x2 (x 256): output o reads inputs
/// 2o - 3 ..= 2o + 4.
const CATMULL_TAPS: [i32; 8] = [-3, -9, 29, 111, 111, 29, -9, -3];

/// The first input and the weights of output `output`'s window, clipped to an
/// axis of `axis_len` inputs.
fn catmull_window(output: usize, axis_len: usize) -> (usize, &'static [i32]) {
    let unclipped_start = 2 * output as isize - 3;
    let start = unclipped_start.max(0) as usize;
    let end = ((unclipped_start + 8) as usize).min(axis_len);
    (
        start,
        &CATMULL_TAPS[(start as isize - unclipped_start) as usize
            ..(end as isize - unclipped_start) as usize],
    )
}

/// `numerator / denominator` rounded half up, clamped to u8 (denominator > 0).
/// The denominators are products of window totals 239, 256 and 259; as
/// constants the divisions compile to multiplies.
fn divide_rounded_clamped(numerator: i64, denominator: i64) -> u8 {
    const TOTALS: [i64; 3] = [239, 256, 259];
    let rounded = |divisor: i64| (2 * numerator + divisor).div_euclid(2 * divisor);
    let quotient = match denominator {
        divisor if divisor == TOTALS[1] * TOTALS[1] => rounded(256 * 256),
        divisor if divisor == TOTALS[0] * TOTALS[1] => rounded(239 * 256),
        divisor if divisor == TOTALS[2] * TOTALS[1] => rounded(259 * 256),
        divisor if divisor == TOTALS[0] * TOTALS[0] => rounded(239 * 239),
        divisor if divisor == TOTALS[0] * TOTALS[2] => rounded(239 * 259),
        divisor if divisor == TOTALS[2] * TOTALS[2] => rounded(259 * 259),
        divisor => rounded(divisor),
    };
    quotient.clamp(0, 255) as u8
}

/// The scalar reference: integer sums, one rounded division per output.
pub(crate) fn catmull_x2_scalar(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    let (out_width, out_height, row_samples) = (width / 2, height / 2, width * channels);
    let mut out = vec![0u8; out_width * out_height * channels];
    let mut sums = vec![0i64; row_samples];
    for y in 0..out_height {
        let (first_row, vertical_weights) = catmull_window(y, height);
        for (x, sum) in sums.iter_mut().enumerate() {
            *sum = vertical_weights
                .iter()
                .enumerate()
                .map(|(tap, &weight)| {
                    i64::from(weight) * i64::from(src[(first_row + tap) * row_samples + x])
                })
                .sum();
        }
        let vertical_total: i64 = vertical_weights
            .iter()
            .map(|&weight| i64::from(weight))
            .sum();
        catmull_row_scalar(
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

/// Outputs `outputs` of a row of vertical sums.
fn catmull_row_scalar(
    sums: &[i64],
    channels: usize,
    width: usize,
    vertical_total: i64,
    outputs: std::ops::Range<usize>,
    dst: &mut [u8],
) {
    for output in outputs {
        let (first_column, horizontal_weights) = catmull_window(output, width);
        let denominator = vertical_total
            * horizontal_weights
                .iter()
                .map(|&weight| i64::from(weight))
                .sum::<i64>();
        for channel in 0..channels {
            let numerator: i64 = horizontal_weights
                .iter()
                .enumerate()
                .map(|(tap, &weight)| {
                    i64::from(weight) * sums[(first_column + tap) * channels + channel]
                })
                .sum();
            dst[output * channels + channel] = divide_rounded_clamped(numerator, denominator);
        }
    }
}

/// The exact x2 catmullrom reduction (each tier computes the same result).
pub(crate) fn catmull_x2(src: &[u8], width: usize, height: usize, channels: usize) -> Vec<u8> {
    crate::sample::assert_image(src, width, height, channels);
    assert!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "the exact x2 kernels take even sizes"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if crate::cpu::has_avx2_fma() {
            // SAFETY: avx2 and fma detected; src holds width * height *
            // channels samples, width and height even, channels 1 or 3
            // (checked).
            return unsafe { x86::catmull_x2_avx2(src, width, height, channels) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::use_neon() {
        // SAFETY: src holds width * height * channels samples.
        return unsafe { neon::catmull_x2_neon(src, width, height, channels) };
    }
    catmull_x2_scalar(src, width, height, channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::tests::{assert_equals_reference, assert_rejects};
    use crate::testing::{CHANNELS, size_grid};

    /// Even widths around the SIMD steps and the scalar fallback, the clipped
    /// edge windows (3 input pixels each side) and the vector tails; even
    /// heights from one output row (both edges clipped) up.
    pub(super) const WIDTHS: [usize; 22] = [
        2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 34, 36, 40, 62, 64, 66, 70, 128, 130, 134, 262,
    ];
    pub(super) const HEIGHTS: [usize; 6] = [2, 4, 6, 8, 10, 14];

    #[test]
    fn dispatch_equals_scalar() {
        assert_equals_reference(
            size_grid(&WIDTHS, &HEIGHTS),
            &CHANNELS,
            u8::MAX,
            &catmull_x2,
            catmull_x2_scalar,
        );
    }

    #[test]
    fn dispatch_rejects_bad_arguments() {
        assert_rejects(
            &catmull_x2,
            &[
                (8 * 8 * 2, 8, 8, 2, "channels"),
                (7 * 8, 7, 8, 1, "even sizes"),
                (8 * 7, 8, 7, 1, "even sizes"),
                (64 * 8 - 1, 64, 8, 1, "fewer than"),
            ],
        );
    }
}

//! Exact x2 catmullrom reduction of u8 images. At exactly half size the
//! catmullrom weights are integers over 256 per axis (renormalized where the
//! window is clipped at the image edge), so each output is an integer sum
//! over the samples, divided once and clamped.

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod x86;

/// The catmullrom weights at x2 (x 256): output o reads inputs 2o - 3 ..= 2o + 4.
const CATMULL_TAPS: [i32; 8] = [-3, -9, 29, 111, 111, 29, -9, -3];

/// The catmullrom window of output `output` along an axis of `axis_len`
/// inputs, clipped.
fn catmull_window(output: usize, axis_len: usize) -> (usize, &'static [i32]) {
    let lo = 2 * output as isize - 3;
    let start = lo.max(0) as usize;
    let end = ((lo + 8) as usize).min(axis_len);
    (
        start,
        &CATMULL_TAPS[(start as isize - lo) as usize..(end as isize - lo) as usize],
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
    let (ow, oh, ws) = (width / 2, height / 2, width * channels);
    let mut out = vec![0u8; ow * oh * channels];
    let mut sums = vec![0i64; ws];
    for y in 0..oh {
        let (r0, wv) = catmull_window(y, height);
        for (x, sum) in sums.iter_mut().enumerate() {
            *sum = wv
                .iter()
                .enumerate()
                .map(|(tap, &weight)| i64::from(weight) * i64::from(src[(r0 + tap) * ws + x]))
                .sum();
        }
        let zv: i64 = wv.iter().map(|&weight| i64::from(weight)).sum();
        catmull_row_scalar(
            &sums,
            channels,
            width,
            zv,
            0..ow,
            &mut out[y * ow * channels..(y + 1) * ow * channels],
        );
    }
    out
}

/// Outputs `outputs` of a row of vertical sums (weight total `zv`).
fn catmull_row_scalar(
    sums: &[i64],
    channels: usize,
    width: usize,
    zv: i64,
    outputs: std::ops::Range<usize>,
    dst: &mut [u8],
) {
    for output in outputs {
        let (x0, wh) = catmull_window(output, width);
        let denominator = zv * wh.iter().map(|&weight| i64::from(weight)).sum::<i64>();
        for ch in 0..channels {
            let numerator: i64 = wh
                .iter()
                .enumerate()
                .map(|(tap, &weight)| i64::from(weight) * sums[(x0 + tap) * channels + ch])
                .sum();
            dst[output * channels + ch] = divide_rounded_clamped(numerator, denominator);
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
        if crate::cpu::has_avx512() && crate::cpu::has_avx2_fma() && channels != 3 {
            // SAFETY: avx512f+bw+vl and fma detected; src holds width *
            // height * channels samples, width and height even (the exact
            // kernel's condition), channels 1 or 4 (the caller's shape
            // check, RGB excluded here).
            return unsafe { x86::catmull_x2_avx512_split(src, width, height, channels) };
        }
        if crate::cpu::has_avx512() {
            // SAFETY: avx512f+bw+vl detected; src holds width * height *
            // channels samples, width and height even (the exact kernel's
            // condition).
            return unsafe { x86::catmull_x2_avx512(src, width, height, channels) };
        }
        if crate::cpu::has_avx2_fma() {
            // SAFETY: avx2 and fma detected; src holds width * height *
            // channels samples, width and height even (the exact kernel's
            // condition), channels 1, 3 or 4 (the caller's shape check).
            return unsafe { x86::catmull_x2_avx2(src, width, height, channels) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::use_neon() {
        // SAFETY: NEON is baseline on aarch64; src holds width * height *
        // channels samples.
        return unsafe { neon::catmull_x2_neon(src, width, height, channels) };
    }
    catmull_x2_scalar(src, width, height, channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::tests::{assert_equals_reference, size_grid};
    use crate::testing::CHANNELS;

    /// Even widths around each kernel's step and its scalar fallback (the
    /// AVX-512 split kernel takes 64 gray pixels, the others fewer), the
    /// clipped edge windows (3 input pixels each side) and the vector tails;
    /// even heights from one output row (both edges clipped) up.
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
    #[should_panic(expected = "channels")]
    fn dispatch_rejects_two_channels() {
        catmull_x2(&[0; 8 * 8 * 2], 8, 8, 2);
    }

    #[test]
    #[should_panic(expected = "even sizes")]
    fn dispatch_rejects_odd_width() {
        catmull_x2(&[0; 7 * 8], 7, 8, 1);
    }

    #[test]
    #[should_panic(expected = "even sizes")]
    fn dispatch_rejects_odd_height() {
        catmull_x2(&[0; 8 * 7], 8, 7, 1);
    }

    #[test]
    #[should_panic(expected = "fewer than")]
    fn dispatch_rejects_short_source() {
        catmull_x2(&[0; 64 * 8 - 1], 64, 8, 1);
    }
}

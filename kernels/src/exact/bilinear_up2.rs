//! Exact x2 bilinear enlargement. At exactly twice the size the bilinear
//! weights are 1/4 and 3/4 per axis: output 2m reads inputs m - 1 and m,
//! output 2m + 1 inputs m and m + 1, at weights 1 and 3 (3 on m). At the
//! image edge the clipped window renormalizes to the edge sample alone, the
//! same sum with the edge sample repeated, so every output is `(sum + 8) >>
//! 4` of a sum of at most 16 * 255: u16 throughout.

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod x86;

/// The input an output reads at weight 1: before its near input `output / 2`
/// for even outputs, after it for odd ones, clamped to the axis.
fn far(output: usize, axis_len: usize) -> usize {
    let near = output / 2;
    if output.is_multiple_of(2) {
        near.saturating_sub(1)
    } else {
        (near + 1).min(axis_len - 1)
    }
}

/// The scalar reference, for any channel count.
pub(crate) fn bilinear_up2_scalar(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    let (out_width, out_height, row_samples) = (2 * width, 2 * height, width * channels);
    let mut out = vec![0u8; out_width * out_height * channels];
    let mut sums = vec![0u16; row_samples];
    for y in 0..out_height {
        let (near, far) = (
            &src[y / 2 * row_samples..][..row_samples],
            &src[far(y, height) * row_samples..][..row_samples],
        );
        for ((sum, &near_sample), &far_sample) in sums.iter_mut().zip(near).zip(far) {
            *sum = 3 * u16::from(near_sample) + u16::from(far_sample);
        }
        horizontal_scalar(
            &sums,
            channels,
            width,
            0..out_width,
            &mut out[y * out_width * channels..(y + 1) * out_width * channels],
        );
    }
    out
}

/// Outputs `outputs` of a row of vertical sums.
fn horizontal_scalar(
    sums: &[u16],
    channels: usize,
    width: usize,
    outputs: std::ops::Range<usize>,
    dst: &mut [u8],
) {
    for output in outputs {
        let (near, far) = (output / 2 * channels, far(output, width) * channels);
        for channel in 0..channels {
            dst[output * channels + channel] =
                ((3 * sums[near + channel] + sums[far + channel] + 8) >> 4) as u8;
        }
    }
}

/// The exact x2 bilinear enlargement (each tier computes the same result).
pub(crate) fn bilinear_up2(src: &[u8], width: usize, height: usize, channels: usize) -> Vec<u8> {
    crate::sample::assert_image(src, width, height, channels);
    #[cfg(target_arch = "x86_64")]
    {
        if crate::cpu::has_avx2() {
            // SAFETY: avx2 detected; src holds width * height * channels
            // samples.
            return unsafe { x86::bilinear_up2_avx2(src, width, height, channels) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::use_neon() {
        // SAFETY: src holds width * height * channels samples.
        return unsafe { neon::bilinear_up2_neon(src, width, height, channels) };
    }
    bilinear_up2_scalar(src, width, height, channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::tests::{assert_equals_reference, assert_rejects};
    use crate::testing::{CHANNELS, size_grid};

    /// Widths around 32 samples per row (the SIMD kernels' scalar fallback
    /// below it), the SIMD steps and a last step shifted back; heights from a
    /// single row (both neighbours the edge) up.
    pub(super) const WIDTHS: [usize; 20] = [
        1, 2, 3, 5, 7, 8, 10, 11, 15, 16, 17, 22, 31, 32, 33, 63, 64, 65, 129, 131,
    ];
    pub(super) const HEIGHTS: [usize; 4] = [1, 2, 3, 5];

    #[test]
    fn dispatch_equals_scalar() {
        assert_equals_reference(
            size_grid(&WIDTHS, &HEIGHTS),
            &CHANNELS,
            u8::MAX,
            &bilinear_up2,
            bilinear_up2_scalar,
        );
    }

    #[test]
    fn dispatch_rejects_bad_arguments() {
        assert_rejects(
            &bilinear_up2,
            &[
                (8 * 8 * 2, 8, 8, 2, "channels"),
                (40 * 3 * 3 - 1, 40, 3, 3, "fewer than"),
            ],
        );
    }
}

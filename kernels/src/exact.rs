//! Exact integer kernels: where a filter's weights at a ratio are integers
//! over a small total, each output is one integer sum divided once (rounded
//! half up), instead of the general kernel's per-tap rounding.

mod bilinear_up2;
mod bilinear_x2;
mod bilinear_x2_u16;
mod bilinear_x4;
mod catmull_x2;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod x86;

use crate::convolve::Kernel;

/// Takes the source (`width * height * channels` samples), `width`, `height`
/// and `channels`.
pub(crate) type ExactKernel<T> = fn(&[T], usize, usize, usize) -> Vec<T>;

/// Panics unless `src` holds a `width` x `height` image of 1 or 3 channels
/// whose sizes are multiples of `FACTOR`: the arguments the exact kernels of
/// every tier take (an enlargement's `FACTOR` is 1).
pub(crate) fn assert_exact_image<T, const FACTOR: usize>(
    src: &[T],
    width: usize,
    height: usize,
    channels: usize,
) {
    crate::sample::assert_image(src, width, height, channels);
    assert!(
        width.is_multiple_of(FACTOR) && height.is_multiple_of(FACTOR),
        "{}",
        match FACTOR {
            2 => "the exact x2 kernels take even sizes",
            _ => "the exact x4 kernel takes sizes that are multiples of 4",
        }
    );
}

/// Panics unless `channels` is 1 or 3, `full` holds `FACTOR * out_width`
/// pixels and `dst` `out_width`: the rows the kernels keep every `FACTOR`th
/// pixel of.
pub(crate) fn assert_kept_pixels<T, const FACTOR: usize>(
    full: &[T],
    channels: usize,
    out_width: usize,
    dst: &[T],
) {
    assert!(
        matches!(channels, 1 | crate::sample::RGB),
        "pixels have 1 or 3 channels, not {channels}"
    );
    assert!(
        full.len() >= FACTOR * out_width * channels && dst.len() >= out_width * channels,
        "the rows hold fewer pixels than the outputs need"
    );
}

/// The uint8 exact kernel of `kernel` resizing `width x height` to
/// `out_width x out_height`, if there is one.
pub(crate) fn exact_kernel(
    kernel: Kernel,
    (width, height): (usize, usize),
    (out_width, out_height): (usize, usize),
) -> Option<ExactKernel<u8>> {
    let half = width == 2 * out_width && height == 2 * out_height;
    let quarter = width == 4 * out_width && height == 4 * out_height;
    let double = out_width == 2 * width && out_height == 2 * height;
    match kernel {
        Kernel::Bilinear if half => Some(bilinear_x2::bilinear_x2),
        Kernel::CatmullRom if half => Some(catmull_x2::catmull_x2),
        Kernel::Bilinear if quarter => Some(bilinear_x4::bilinear_x4),
        Kernel::Bilinear if double => Some(bilinear_up2::bilinear_up2),
        _ => None,
    }
}

/// `exact_kernel` for uint16.
pub(crate) fn exact_kernel_u16(
    kernel: Kernel,
    (width, height): (usize, usize),
    (out_width, out_height): (usize, usize),
) -> Option<ExactKernel<u16>> {
    let half = width == 2 * out_width && height == 2 * out_height;
    match kernel {
        Kernel::Bilinear if half => Some(bilinear_x2_u16::bilinear_x2_u16),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::testing::{RandomSample, Rng, test_images};

    /// An `ExactKernel` of any sample type.
    pub(crate) type ChannelKernel<'kernel, T> =
        &'kernel dyn Fn(&[T], usize, usize, usize) -> Vec<T>;

    /// Bit for bit on every `test_images` image of each size and channel
    /// count, on sources of exactly `width * height * channels` samples.
    pub(crate) fn assert_equals_reference<T: RandomSample + Copy + Default>(
        sizes: impl Iterator<Item = (usize, usize)>,
        channel_counts: &[usize],
        max: T,
        kernel: ChannelKernel<'_, T>,
        reference: fn(&[T], usize, usize, usize) -> Vec<T>,
    ) {
        let mut rng = Rng::new(7);
        for (width, height) in sizes {
            for &channels in channel_counts {
                for (image, src) in test_images(&mut rng, width * height * channels, max) {
                    assert_eq!(
                        kernel(&src, width, height, channels),
                        reference(&src, width, height, channels),
                        "{image} {width} x {height} x {channels}"
                    );
                }
            }
        }
    }

    /// Bad arguments to a kernel: the source's sample count, `width`,
    /// `height`, `channels`, and a fragment of the panic message expected.
    pub(crate) type Rejection = (usize, usize, usize, usize, &'static str);

    /// Each rejection panics with its expected message.
    pub(crate) fn assert_rejects<T: Copy + Default>(
        kernel: ChannelKernel<'_, T>,
        rejections: &[Rejection],
    ) {
        for &(samples, width, height, channels, expected) in rejections {
            let src = vec![T::default(); samples];
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                drop(kernel(&src, width, height, channels));
            }))
            .expect_err(&format!(
                "{samples} samples {width} x {height} x {channels} did not panic"
            ));
            let message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or_default();
            assert!(
                message.contains(expected),
                "{width} x {height} x {channels}: {message:?} lacks {expected:?}"
            );
        }
    }
}

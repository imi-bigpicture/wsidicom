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
    use crate::testing::{RandomSample, Rng, random_samples};

    /// An `ExactKernel` of any sample type.
    pub(crate) type ChannelKernel<'kernel, T> =
        &'kernel dyn Fn(&[T], usize, usize, usize) -> Vec<T>;

    pub(crate) fn size_grid<'sizes>(
        widths: &'sizes [usize],
        heights: &'sizes [usize],
    ) -> impl Iterator<Item = (usize, usize)> + 'sizes {
        widths
            .iter()
            .flat_map(move |&width| heights.iter().map(move |&height| (width, height)))
    }

    /// Random samples, all at `max` (the largest sums), and random 0 or `max`
    /// (the largest overshoots of negative weights).
    pub(crate) fn test_images<T: RandomSample + Copy + Default>(
        rng: &mut Rng,
        len: usize,
        max: T,
    ) -> [(&'static str, Vec<T>); 3] {
        let extremes = (0..len)
            .map(|_| {
                if rng.next_u64() & 1 == 0 {
                    T::default()
                } else {
                    max
                }
            })
            .collect();
        [
            ("random", random_samples(rng, len)),
            ("saturated", vec![max; len]),
            ("extremes", extremes),
        ]
    }

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

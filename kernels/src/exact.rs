//! Resizes with an exact integer kernel. Where a filter's weights at a ratio
//! are integers over a small total, each output is one integer sum divided
//! once (rounded half up), instead of the general kernel's per-tap rounding.

mod bilinear_up2;
mod bilinear_x2;
pub(crate) mod bilinear_x2_u16;
mod bilinear_x4;
mod catmull_x2;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod x86;

use crate::convolve::Kernel;

pub(crate) type ExactKernel = fn(&[u8], usize, usize, usize) -> Vec<u8>;

/// The exact kernel for a resize of `(width, height)` to `(ow, oh)`, if one
/// applies: bilinear and catmullrom to exactly half the size, bilinear to
/// exactly a quarter and to exactly twice the size. It takes the source
/// (`width * height * channels` samples), `width`, `height` and `channels`.
pub(crate) fn exact_kernel(
    kernel: Kernel,
    (width, height): (usize, usize),
    (ow, oh): (usize, usize),
) -> Option<ExactKernel> {
    let half = width == 2 * ow && height == 2 * oh;
    let quarter = width == 4 * ow && height == 4 * oh;
    let double = ow == 2 * width && oh == 2 * height;
    match kernel {
        Kernel::Bilinear if half => Some(bilinear_x2::bilinear_x2),
        Kernel::CatmullRom if half => Some(catmull_x2::catmull_x2),
        Kernel::Bilinear if quarter => Some(bilinear_x4::bilinear_x4),
        Kernel::Bilinear if double => Some(bilinear_up2::bilinear_up2),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::testing::{RandomSample, Rng, random_samples};

    /// A kernel of the exact module's shape: source, width, height, channels.
    pub(crate) type ChannelKernel<'kernel, T> =
        &'kernel dyn Fn(&[T], usize, usize, usize) -> Vec<T>;

    /// Every (width, height) of `widths` x `heights`.
    pub(crate) fn size_grid<'sizes>(
        widths: &'sizes [usize],
        heights: &'sizes [usize],
    ) -> impl Iterator<Item = (usize, usize)> + 'sizes {
        widths
            .iter()
            .flat_map(move |&width| heights.iter().map(move |&height| (width, height)))
    }

    /// Images of `len` samples that between them exercise the arithmetic:
    /// uniform random samples, every sample at `max` (the largest sums), and
    /// samples at random extremes 0 or `max` (the largest overshoots of
    /// negative weights).
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

    /// Asserts that `kernel` equals `reference` bit for bit on every test
    /// image (see `test_images`) of each size and channel count. The sources
    /// hold exactly `width * height * channels` samples.
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
}

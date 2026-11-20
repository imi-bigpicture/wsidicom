//! Helpers shared by the unit tests: reproducible random images, the sizes
//! that exercise vector bodies, tails and image edges, CPU feature probes,
//! and a sample-by-sample comparison of two outputs.
//!
//! No test turns `set_simd` off, so the tests of a dispatcher run the SIMD
//! kernel this CPU has, against the scalar reference.

use crate::convolve::Kernel;
use crate::sample::Sample;

/// A xorshift64* generator: reproducible test data without a dependency.
pub(crate) struct Rng(u64);

/// 2^64 / the golden ratio: spreads small seeds over the state (which must
/// not be zero, hence the `| 1`).
const SEED_SPREAD: u64 = 0x9E37_79B9_7F4A_7C15;
const XORSHIFT64_STAR_MULTIPLIER: u64 = 0x2545_F491_4F6C_DD1D;

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(SEED_SPREAD) | 1)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(XORSHIFT64_STAR_MULTIPLIER)
    }
}

/// A sample type the tests can draw uniformly over its range.
pub(crate) trait RandomSample: Sample + PartialEq + std::fmt::Debug {
    fn random(rng: &mut Rng) -> Self;
}

impl RandomSample for u8 {
    fn random(rng: &mut Rng) -> Self {
        rng.next_u64().to_le_bytes()[0]
    }
}

impl RandomSample for u16 {
    fn random(rng: &mut Rng) -> Self {
        let [low, high, ..] = rng.next_u64().to_le_bytes();
        Self::from_le_bytes([low, high])
    }
}

/// `len` random samples.
pub(crate) fn random_samples<T: RandomSample>(rng: &mut Rng, len: usize) -> Vec<T> {
    (0..len).map(|_| T::random(rng)).collect()
}

/// Image widths and heights around the vector widths (16, 32 and 64 samples)
/// and the smallest sizes, so bodies, tails and edges all run.
pub(crate) const SIZES: [usize; 14] = [1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 33, 64, 67, 130];

/// The pixel layouts the kernels take: gray and RGB.
pub(crate) const CHANNELS: [usize; 2] = [1, 3];

/// Every (width, height) pair of `widths` and `heights`, width by width.
pub(crate) fn size_grid<'sizes>(
    widths: &'sizes [usize],
    heights: &'sizes [usize],
) -> impl Iterator<Item = (usize, usize)> + 'sizes {
    widths
        .iter()
        .flat_map(move |&width| heights.iter().map(move |&height| (width, height)))
}

/// Every (width, height) pair of `SIZES` with both at least `minimum`.
pub(crate) fn sizes_from(minimum: usize) -> Vec<(usize, usize)> {
    let sizes: Vec<usize> = SIZES.into_iter().filter(|&size| size >= minimum).collect();
    size_grid(&sizes, &sizes).collect()
}

/// Random samples, all at `max` (the largest sums), and random 0 or `max`
/// (the extremes, where integer sums and packs would overflow or saturate
/// and negative weights overshoot most).
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

/// Asserts `actual` equals `expected`, naming the first differing sample
/// (rather than printing whole outputs).
#[track_caller]
pub(crate) fn assert_same<T: PartialEq + std::fmt::Debug>(
    actual: &[T],
    expected: &[T],
    context: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{context}: length");
    if let Some(index) = (0..actual.len()).find(|&index| actual[index] != expected[index]) {
        panic!(
            "{context}: sample {index} is {:?}, expected {:?}",
            actual[index], expected[index]
        );
    }
}

/// The name `resize` takes for `kernel`.
pub(crate) fn kernel_name(kernel: Kernel) -> &'static str {
    match kernel {
        Kernel::Box => "box",
        Kernel::Bilinear => "bilinear",
        Kernel::CatmullRom => "catmullrom",
        Kernel::Lanczos3 => "lanczos3",
    }
}

/// Whether this CPU has AVX2, whatever `set_simd` says: the tests call the
/// AVX2 kernels directly.
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx2_detected() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// Whether this CPU has AVX2 and FMA, whatever `set_simd` says.
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx2_fma_detected() -> bool {
    avx2_detected() && std::is_x86_feature_detected!("fma")
}

/// A kernel of `(src, width, height, channels)`.
pub(crate) type ChannelKernel<'kernel, T> = &'kernel dyn Fn(&[T], usize, usize, usize) -> Vec<T>;

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

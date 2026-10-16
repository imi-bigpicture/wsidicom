//! Helpers shared by the unit tests: reproducible random images, the sizes
//! that exercise vector bodies, tails and image edges, and the comparison of
//! a kernel's output with its scalar reference.

use crate::sample::Sample;

/// A kernel from a `width` x `height` image to its output and output size.
pub(crate) type ImageKernel<T> = fn(&[T], usize, usize) -> (Vec<T>, usize, usize);
/// An [`ImageKernel`] that needs CPU features or a caller contract.
pub(crate) type UnsafeImageKernel<T> = unsafe fn(&[T], usize, usize) -> (Vec<T>, usize, usize);

/// A xorshift64* generator: reproducible test data without a dependency.
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

/// A sample type the tests can draw uniformly over its range (float32 over
/// [0, 1)).
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

impl RandomSample for f32 {
    fn random(rng: &mut Rng) -> Self {
        // The top 24 bits as a fraction: every value exactly representable.
        let fraction_bits = u32::try_from(rng.next_u64() >> 40).unwrap_or(0);
        fraction_bits as f32 / (1 << 24) as f32
    }
}

/// `len` random samples.
pub(crate) fn random_samples<T: RandomSample>(rng: &mut Rng, len: usize) -> Vec<T> {
    (0..len).map(|_| T::random(rng)).collect()
}

/// Image widths and heights around the vector widths (16, 32 and 64 samples)
/// and the smallest sizes, so bodies, tails and edges all run.
pub(crate) const SIZES: [usize; 14] = [1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 33, 64, 67, 130];

/// The pixel layouts the kernels take: gray, RGB and 4-sample pixels.
pub(crate) const CHANNELS: [usize; 3] = [1, 3, 4];

/// Every (width, height) pair of `SIZES` with both at least `minimum`.
pub(crate) fn sizes_from(minimum: usize) -> impl Iterator<Item = (usize, usize)> {
    SIZES
        .into_iter()
        .filter(move |&size| size >= minimum)
        .flat_map(move |width| {
            SIZES
                .into_iter()
                .filter(move |&size| size >= minimum)
                .map(move |height| (width, height))
        })
}

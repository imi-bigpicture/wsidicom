//! Helpers shared by the NEON exact kernels.
//!
//! # Safety
//! NEON is baseline on aarch64. Every `unsafe fn` here requires the bounds
//! its doc states.

use std::arch::aarch64::{
    uint8x16_t, uint8x16x3_t, uint8x16x4_t, uint16x8_t, uint16x8x3_t, uint16x8x4_t, vld2q_u8,
    vld2q_u16, vld3q_u8, vld3q_u16, vld4q_u8, vld4q_u16, vst1q_u8, vst1q_u16, vst3q_u8, vst3q_u16,
    vst4q_u8, vst4q_u16, vuzp1q_u8, vuzp1q_u16,
};

use crate::sample::{RGB, RGBX};

/// `dst[o]` = pixel `2 o` of `full`, for the `ow` output pixels: 16 per
/// step by `vld2`/`vld3`/`vld4` and `vuzp1` (the even lanes); scalar tail.
///
/// # Safety
/// `full` holds `2 * ow` pixels of `channels` (1, 3 or 4) samples and `dst`
/// `ow`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn keep_even_pixels(full: &[u8], channels: usize, ow: usize, dst: &mut [u8]) {
    const STEP_PX: usize = 16;
    debug_assert!(full.len() >= 2 * ow * channels && dst.len() >= ow * channels);
    let (full_ptr, dst_ptr) = (full.as_ptr(), dst.as_mut_ptr());
    let mut output = 0usize;
    while output + STEP_PX <= ow {
        let at = 2 * output * channels;
        // SAFETY: output + 16 <= ow: the loads read pixels 2 output ..
        // 2 output + 32 of full, the store output pixels output .. + 16.
        unsafe {
            let even = |plane: PlanePair| vuzp1q_u8(plane.0, plane.1);
            match channels {
                1 => vst1q_u8(dst_ptr.add(output), vld2q_u8(full_ptr.add(at)).0),
                RGB => {
                    let (left, right) =
                        (vld3q_u8(full_ptr.add(at)), vld3q_u8(full_ptr.add(at + 48)));
                    let planes = uint8x16x3_t(
                        even((left.0, right.0)),
                        even((left.1, right.1)),
                        even((left.2, right.2)),
                    );
                    vst3q_u8(dst_ptr.add(output * RGB), planes);
                }
                _ => {
                    let (left, right) =
                        (vld4q_u8(full_ptr.add(at)), vld4q_u8(full_ptr.add(at + 64)));
                    let planes = uint8x16x4_t(
                        even((left.0, right.0)),
                        even((left.1, right.1)),
                        even((left.2, right.2)),
                        even((left.3, right.3)),
                    );
                    vst4q_u8(dst_ptr.add(output * RGBX), planes);
                }
            }
        }
        output += STEP_PX;
    }
    for output in output..ow {
        let at = 2 * output * channels;
        dst[output * channels..(output + 1) * channels].copy_from_slice(&full[at..at + channels]);
    }
}

/// One channel plane of 32 pixels as two registers: the left and right 16.
type PlanePair = (uint8x16_t, uint8x16_t);

/// `keep_even_pixels` for uint16 samples: 8 output pixels per step.
///
/// # Safety
/// As `keep_even_pixels`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn keep_even_pixels_u16(
    full: &[u16],
    channels: usize,
    ow: usize,
    dst: &mut [u16],
) {
    const STEP_PX: usize = 8;
    debug_assert!(full.len() >= 2 * ow * channels && dst.len() >= ow * channels);
    let (full_ptr, dst_ptr) = (full.as_ptr(), dst.as_mut_ptr());
    let even = |left: uint16x8_t, right: uint16x8_t| vuzp1q_u16(left, right);
    let mut output = 0usize;
    while output + STEP_PX <= ow {
        let at = 2 * output * channels;
        // SAFETY: output + 8 <= ow: the loads read pixels 2 output ..
        // 2 output + 16 of full, the store output pixels output .. + 8.
        unsafe {
            match channels {
                1 => vst1q_u16(dst_ptr.add(output), vld2q_u16(full_ptr.add(at)).0),
                RGB => {
                    let (left, right) = (
                        vld3q_u16(full_ptr.add(at)),
                        vld3q_u16(full_ptr.add(at + 24)),
                    );
                    let planes = uint16x8x3_t(
                        even(left.0, right.0),
                        even(left.1, right.1),
                        even(left.2, right.2),
                    );
                    vst3q_u16(dst_ptr.add(output * RGB), planes);
                }
                _ => {
                    let (left, right) = (
                        vld4q_u16(full_ptr.add(at)),
                        vld4q_u16(full_ptr.add(at + 32)),
                    );
                    let planes = uint16x8x4_t(
                        even(left.0, right.0),
                        even(left.1, right.1),
                        even(left.2, right.2),
                        even(left.3, right.3),
                    );
                    vst4q_u16(dst_ptr.add(output * RGBX), planes);
                }
            }
        }
        output += STEP_PX;
    }
    for output in output..ow {
        let at = 2 * output * channels;
        dst[output * channels..(output + 1) * channels].copy_from_slice(&full[at..at + channels]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{CHANNELS, RandomSample, Rng, random_samples};

    /// Pixel `2 o` of `full` for every output `o`.
    fn even_pixels<T: Copy>(full: &[T], channels: usize) -> Vec<T> {
        full.chunks_exact(2 * channels)
            .flat_map(|pair| pair[..channels].to_vec())
            .collect()
    }

    /// Output widths around the steps of both kernels, on buffers of exactly
    /// the length their contract requires.
    fn assert_keeps_even_pixels<T: RandomSample + Default>(
        kernel: unsafe fn(&[T], usize, usize, &mut [T]),
    ) {
        let mut rng = Rng::new(3);
        for channels in CHANNELS {
            for ow in 0..80 {
                let full: Vec<T> = random_samples(&mut rng, 2 * ow * channels);
                let mut dst = vec![T::default(); ow * channels];
                // SAFETY: NEON is baseline; full holds 2 ow pixels, dst ow.
                unsafe { kernel(&full, channels, ow, &mut dst) };
                let expected = even_pixels(&full, channels);
                assert_eq!(dst, expected, "{channels} channels, {ow} pixels");
            }
        }
    }

    #[test]
    fn keep_even_pixels_equals_its_definition() {
        assert_keeps_even_pixels::<u8>(keep_even_pixels);
    }

    #[test]
    fn keep_even_pixels_u16_equals_its_definition() {
        assert_keeps_even_pixels::<u16>(keep_even_pixels_u16);
    }
}

//! Helpers shared by the NEON exact kernels.
//!
//! # Safety
//! NEON is baseline on aarch64; every `unsafe fn` here requires the bounds
//! its doc states.

use std::arch::aarch64::{
    uint8x16_t, uint8x16x3_t, uint16x8_t, uint16x8x3_t, vld2q_u8, vld2q_u16, vld3q_u8, vld3q_u16,
    vst1q_u8, vst1q_u16, vst3q_u8, vst3q_u16, vuzp1q_u8, vuzp1q_u16,
};

use crate::sample::RGB;

/// `dst` = the even pixels of `full`: `vld2`/`vld3` split the channel planes
/// and `vuzp1` keeps their even lanes.
///
/// # Safety
/// `full` holds `2 * out_width` pixels of `channels` (1 or 3) samples and
/// `dst` `out_width`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn keep_even_pixels(
    full: &[u8],
    channels: usize,
    out_width: usize,
    dst: &mut [u8],
) {
    const STEP_PIXELS: usize = 16;
    debug_assert!(full.len() >= 2 * out_width * channels && dst.len() >= out_width * channels);
    let (full_ptr, dst_ptr) = (full.as_ptr(), dst.as_mut_ptr());
    let mut output = 0usize;
    while output + STEP_PIXELS <= out_width {
        let at = 2 * output * channels;
        // SAFETY: output + 16 <= out_width: the loads read pixels 2 output ..
        // 2 output + 32 of full, the store output pixels output .. + 16.
        unsafe {
            let even = |plane: PlanePair| vuzp1q_u8(plane.0, plane.1);
            if channels == 1 {
                vst1q_u8(dst_ptr.add(output), vld2q_u8(full_ptr.add(at)).0);
            } else {
                let (left, right) = (vld3q_u8(full_ptr.add(at)), vld3q_u8(full_ptr.add(at + 48)));
                let planes = uint8x16x3_t(
                    even((left.0, right.0)),
                    even((left.1, right.1)),
                    even((left.2, right.2)),
                );
                vst3q_u8(dst_ptr.add(output * RGB), planes);
            }
        }
        output += STEP_PIXELS;
    }
    for output in output..out_width {
        let at = 2 * output * channels;
        dst[output * channels..(output + 1) * channels].copy_from_slice(&full[at..at + channels]);
    }
}

/// One channel plane of 32 pixels as two registers: the left and right 16.
type PlanePair = (uint8x16_t, uint8x16_t);

/// `keep_even_pixels` for uint16 samples.
///
/// # Safety
/// As `keep_even_pixels`.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn keep_even_pixels_u16(
    full: &[u16],
    channels: usize,
    out_width: usize,
    dst: &mut [u16],
) {
    const STEP_PIXELS: usize = 8;
    debug_assert!(full.len() >= 2 * out_width * channels && dst.len() >= out_width * channels);
    let (full_ptr, dst_ptr) = (full.as_ptr(), dst.as_mut_ptr());
    let even = |left: uint16x8_t, right: uint16x8_t| vuzp1q_u16(left, right);
    let mut output = 0usize;
    while output + STEP_PIXELS <= out_width {
        let at = 2 * output * channels;
        // SAFETY: output + 8 <= out_width: the loads read pixels 2 output ..
        // 2 output + 16 of full, the store output pixels output .. + 8.
        unsafe {
            if channels == 1 {
                vst1q_u16(dst_ptr.add(output), vld2q_u16(full_ptr.add(at)).0);
            } else {
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
        }
        output += STEP_PIXELS;
    }
    for output in output..out_width {
        let at = 2 * output * channels;
        dst[output * channels..(output + 1) * channels].copy_from_slice(&full[at..at + channels]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{CHANNELS, RandomSample, Rng, random_samples};

    fn even_pixels<T: Copy>(full: &[T], channels: usize) -> Vec<T> {
        full.chunks_exact(2 * channels)
            .flat_map(|pair| pair[..channels].to_vec())
            .collect()
    }

    /// Output widths around both kernels' steps, on buffers of exactly the
    /// length their contract requires.
    fn assert_keeps_even_pixels<T: RandomSample + Default>(
        kernel: unsafe fn(&[T], usize, usize, &mut [T]),
    ) {
        let mut rng = Rng::new(3);
        for channels in CHANNELS {
            for out_width in 0..80 {
                let full: Vec<T> = random_samples(&mut rng, 2 * out_width * channels);
                let mut dst = vec![T::default(); out_width * channels];
                // SAFETY: NEON is baseline; full holds 2 out_width pixels, dst
                // out_width.
                unsafe { kernel(&full, channels, out_width, &mut dst) };
                let expected = even_pixels(&full, channels);
                assert_eq!(dst, expected, "{channels} channels, {out_width} pixels");
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

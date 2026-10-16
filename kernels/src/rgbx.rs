//! RGB -> RGBX: the 4-byte layout of Pillow's "RGBX" images.

#[cfg(target_arch = "x86_64")]
use crate::cpu::has_avx2;
#[cfg(target_arch = "aarch64")]
use crate::cpu::use_neon;
use crate::sample::{RGB, RGBX};
#[cfg(target_arch = "x86_64")]
use crate::x86::expand_rgb_rgbx_avx2;

/// Dispatching RGB->RGBX expand (AVX2 pshufb, NEON vld3/vst4, else scalar).
#[inline]
pub(crate) fn expand_rgb_rgbx(src: &[u8], dst: &mut [u8], pixels: usize, pad: u8) {
    assert!(
        src.len() >= pixels * RGB && dst.len() >= pixels * RGBX,
        "src or dst holds fewer than `pixels` pixels"
    );
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        // SAFETY: avx2 detected at runtime; src holds `pixels` RGB pixels, dst as
        // many RGBX ones.
        return unsafe { expand_rgb_rgbx_avx2(src, dst, pixels, pad) };
    }
    #[cfg(target_arch = "aarch64")]
    if use_neon() {
        let (src, dst) = (&src[..pixels * RGB], &mut dst[..pixels * RGBX]);
        // SAFETY: NEON is baseline on aarch64.
        return unsafe { expand_rgb_rgbx_neon(src, dst, pad) };
    }
    expand_rgb_rgbx_scalar(&src[..pixels * RGB], &mut dst[..pixels * RGBX], pad);
}

/// RGB -> RGBX over whole slices (`dst` holds as many pixels as `src`).
fn expand_rgb_rgbx_scalar(src: &[u8], dst: &mut [u8], pad: u8) {
    let (rgb_pixels, _) = src.as_chunks::<RGB>();
    let (rgbx_pixels, _) = dst.as_chunks_mut::<RGBX>();
    for (rgbx, rgb) in rgbx_pixels.iter_mut().zip(rgb_pixels) {
        *rgbx = [rgb[0], rgb[1], rgb[2], pad];
    }
}

/// NEON RGB -> RGBX: `vld3` splits 16 pixels into channel planes, `vst4`
/// interleaves them with a pad plane; scalar tail.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
fn expand_rgb_rgbx_neon(src: &[u8], dst: &mut [u8], pad: u8) {
    use std::arch::aarch64::{uint8x16x4_t, vdupq_n_u8, vld3q_u8, vst4q_u8};
    const STEP_PX: usize = 16;
    let pad_plane = vdupq_n_u8(pad);
    let (rgb_blocks, rgb_tail) = src.as_chunks::<{ STEP_PX * RGB }>();
    let (rgbx_blocks, rgbx_tail) = dst.as_chunks_mut::<{ STEP_PX * RGBX }>();
    for (rgb, rgbx) in rgb_blocks.iter().zip(rgbx_blocks) {
        // SAFETY: the chunks hold exactly 16 RGB and 16 RGBX pixels.
        unsafe {
            let planes = vld3q_u8(rgb.as_ptr());
            vst4q_u8(
                rgbx.as_mut_ptr(),
                uint8x16x4_t(planes.0, planes.1, planes.2, pad_plane),
            );
        }
    }
    expand_rgb_rgbx_scalar(rgb_tail, rgbx_tail, pad);
}

/// RGB -> RGBX without a size change: one expand pass, with the page faults of
/// the new (larger) output overlapped with it.
pub(crate) fn expand_image(src: &[u8], height: usize, width: usize, pad: u8) -> Vec<u8> {
    let mut out = vec![0u8; height * width * RGBX];
    let dp = out.as_mut_ptr();
    // SAFETY: `out` is exclusively owned for the call and each `rows(y0, y1)`
    // writes only output rows [y0, y1) through its own sub-slice.
    unsafe {
        crate::cpu::with_prefault(dp, out.len(), width * RGBX, height, |y0, y1| {
            let rows = y1 - y0;
            let dst =
                std::slice::from_raw_parts_mut(dp.add(y0 * width * RGBX), rows * width * RGBX);
            expand_rgb_rgbx(
                &src[y0 * width * RGB..y1 * width * RGB],
                dst,
                rows * width,
                pad,
            );
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Rng, SIZES, random_samples};

    /// RGB pixels widened to RGBX with `pad` in X.
    fn rgb_to_rgbx(rgb: &[u8], pad: u8) -> Vec<u8> {
        rgb.as_chunks::<RGB>()
            .0
            .iter()
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], pad])
            .collect()
    }

    #[test]
    fn expand_rgb_rgbx_writes_only_its_pixels() {
        const UNTOUCHED: u8 = 0xCC;
        let mut rng = Rng::new(70);
        for pixels in 0..=100 {
            let src: Vec<u8> = random_samples(&mut rng, RGB * pixels);
            let mut dst = vec![UNTOUCHED; RGBX * pixels + 40];
            expand_rgb_rgbx(&src, &mut dst, pixels, 0x3E);
            assert_eq!(
                dst[..RGBX * pixels],
                rgb_to_rgbx(&src, 0x3E),
                "{pixels} pixels"
            );
            assert!(dst[RGBX * pixels..].iter().all(|&byte| byte == UNTOUCHED));
        }
    }

    #[test]
    #[should_panic(expected = "fewer than `pixels` pixels")]
    fn expand_rgb_rgbx_rejects_a_short_source() {
        expand_rgb_rgbx(&[0; 3 * 9 - 1], &mut [0; 4 * 9], 9, 0);
    }

    #[test]
    #[should_panic(expected = "fewer than `pixels` pixels")]
    fn expand_rgb_rgbx_rejects_a_short_destination() {
        expand_rgb_rgbx(&[0; 3 * 9], &mut [0; 4 * 9 - 1], 9, 0);
    }

    #[test]
    fn expand_image_equals_the_pixels_expanded() {
        let mut rng = Rng::new(71);
        for width in SIZES {
            for height in SIZES {
                let src: Vec<u8> = random_samples(&mut rng, width * height * RGB);
                assert_eq!(
                    expand_image(&src, height, width, 9),
                    rgb_to_rgbx(&src, 9),
                    "{width} x {height}"
                );
            }
        }
    }

    #[test]
    fn large_images_expand_in_prefaulted_bands() {
        let (width, height) = (1501, 1400);
        assert!(width * height * RGBX >= crate::cpu::PREFAULT_OUTPUT_BYTES);
        let src: Vec<u8> = random_samples(&mut Rng::new(72), width * height * RGB);
        assert!(expand_image(&src, height, width, u8::MAX) == rgb_to_rgbx(&src, u8::MAX));
    }
}

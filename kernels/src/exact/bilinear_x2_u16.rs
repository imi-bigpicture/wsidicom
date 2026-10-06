//! Exact x2 bilinear reduction of uint16 images: the uint8 kernel's integer
//! arithmetic (`bilinear_x2`) in 32-bit lanes. Vertical sums reach 8 x 65535
//! and the window's sum 64 x 65535, so every lane is i32.

#[cfg(target_arch = "x86_64")]
mod x86;

use super::bilinear_x2::bilinear_x2_scalar;

/// The exact x2 bilinear reduction of a uint16 image (each tier computes the
/// same result).
pub(crate) fn bilinear_x2_u16(
    src: &[u16],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u16> {
    crate::sample::assert_image(src, width, height, channels);
    assert!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "the exact x2 kernels take even sizes"
    );
    #[cfg(target_arch = "x86_64")]
    {
        if crate::cpu::has_avx2() {
            // SAFETY: avx2 detected; src holds width * height * channels
            // samples, width and height even, channels 1, 3 or 4 (the
            // caller's shape and exact-half checks).
            return unsafe {
                match channels {
                    1 => x86::bilinear_x2_u16_gray_avx2(src, width, height),
                    3 => x86::bilinear_x2_u16_hwc_avx2::<3>(src, width, height),
                    _ => x86::bilinear_x2_u16_hwc_avx2::<4>(src, width, height),
                }
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::use_neon() {
        // SAFETY: NEON is baseline on aarch64; src holds width * height *
        // channels samples.
        return unsafe {
            super::bilinear_x2::neon::bilinear_x2_u16_neon(src, width, height, channels)
        };
    }
    bilinear_x2_scalar(src, width, height, channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::tests::{assert_equals_reference, size_grid};
    use crate::testing::CHANNELS;

    /// Even widths around the one-channel kernel's 8-pixel step and the
    /// interleaved kernels' 4-pixel step and their scalar fallbacks.
    pub(super) const WIDTHS: [usize; 16] =
        [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 34, 36, 38, 66, 130];
    pub(super) const HEIGHTS: [usize; 5] = [2, 4, 6, 8, 12];

    #[test]
    fn dispatch_equals_scalar() {
        assert_equals_reference(
            size_grid(&WIDTHS, &HEIGHTS),
            &CHANNELS,
            u16::MAX,
            &bilinear_x2_u16,
            bilinear_x2_scalar,
        );
    }

    #[test]
    #[should_panic(expected = "channels")]
    fn dispatch_rejects_two_channels() {
        bilinear_x2_u16(&[0; 8 * 8 * 2], 8, 8, 2);
    }

    #[test]
    #[should_panic(expected = "even sizes")]
    fn dispatch_rejects_odd_width() {
        bilinear_x2_u16(&[0; 9 * 8], 9, 8, 1);
    }

    #[test]
    #[should_panic(expected = "even sizes")]
    fn dispatch_rejects_odd_height() {
        bilinear_x2_u16(&[0; 8 * 9], 8, 9, 1);
    }

    #[test]
    #[should_panic(expected = "fewer than")]
    fn dispatch_rejects_short_source() {
        bilinear_x2_u16(&[0; 8 * 8 * 4 - 1], 8, 8, 4);
    }
}

//! Exact x2 bilinear reduction of uint16 images: `bilinear_x2`'s arithmetic
//! in 32-bit lanes (the window's sum reaches 64 x 65535).

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "x86_64")]
mod x86;

use super::bilinear_x2::bilinear_x2_scalar;

/// Each tier computes the same result.
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
            // samples, width and height even, channels 1 or 3 (checked).
            return unsafe {
                match channels {
                    1 => x86::bilinear_x2_u16_gray_avx2(src, width, height),
                    _ => x86::bilinear_x2_u16_hwc_avx2(src, width, height),
                }
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if crate::cpu::use_neon() {
        // SAFETY: src holds width * height * channels samples.
        return unsafe { neon::bilinear_x2_u16_neon(src, width, height, channels) };
    }
    bilinear_x2_scalar(src, width, height, channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact::tests::{assert_equals_reference, assert_rejects};
    use crate::testing::{CHANNELS, size_grid};

    /// Even widths around the gray kernel's 8-pixel step, the RGB kernel's
    /// 4-pixel step and their scalar fallbacks.
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
    fn dispatch_rejects_bad_arguments() {
        assert_rejects(
            &bilinear_x2_u16,
            &[
                (8 * 8 * 2, 8, 8, 2, "channels"),
                (9 * 8, 9, 8, 1, "even sizes"),
                (8 * 9, 8, 9, 1, "even sizes"),
                (8 * 8 * 3 - 1, 8, 8, 3, "fewer than"),
            ],
        );
    }
}

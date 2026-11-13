//! The NEON kernel of the exact x2 bilinear enlargement.
//!
//! # Safety
//! NEON is baseline on aarch64, gated only by the `set_simd` kill switch;
//! every `unsafe fn` here requires the bounds its doc states.

use std::arch::aarch64::{
    uint8x16_t, uint8x16x3_t, uint16x8_t, vaddw_u8, vcombine_u8, vdup_n_u8, vget_high_u8,
    vget_low_u8, vld1q_u8, vld1q_u16, vld3q_u16, vmlaq_n_u16, vmull_u8, vrshrn_n_u16, vst1q_u8,
    vst1q_u16, vst3q_u8, vzip1_u8, vzip2_u8,
};

use super::{far, horizontal_scalar};

/// Input pixels per horizontal step: 16 output pixels.
const STEP_PIXELS: usize = 8;

/// Per output row the vertical sums `3 near + far`, with the row's first and
/// last pixel repeated once outside it (the scalar reference's clamped far
/// tap), so every input pixel gives its two outputs by `enlarged`. Pixels
/// past the last whole step by the scalar reference.
///
/// # Safety
/// `src` holds `width * height * channels` samples, `channels` 1 or 3.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bilinear_up2_neon(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    let (out_width, out_height, row_samples) = (2 * width, 2 * height, width * channels);
    debug_assert!(src.len() >= width * height * channels);
    let mut out_buf = crate::cpu::unwritten_output::<u8>(out_width * out_height * channels);
    // SAFETY: u8 has no invalid values; every sample is written below before
    // the vector is returned.
    let out = unsafe {
        std::slice::from_raw_parts_mut(
            out_buf.as_mut_ptr().cast::<u8>(),
            out_width * out_height * channels,
        )
    };
    // A pixel before and after the row.
    let mut sums = vec![0u16; row_samples + 2 * channels];
    let steps = width / STEP_PIXELS;
    for y in 0..out_height {
        let (near, far) = (
            &src[y / 2 * row_samples..][..row_samples],
            &src[far(y, height) * row_samples..][..row_samples],
        );
        let row_sums = &mut sums[channels..channels + row_samples];
        vertical_sums(near, far, row_sums);
        sums.copy_within(channels..2 * channels, 0);
        sums.copy_within(row_samples..row_samples + channels, row_samples + channels);
        let dst = &mut out[y * out_width * channels..(y + 1) * out_width * channels];
        let sums_ptr = sums.as_ptr();
        for step in 0..steps {
            let first = step * STEP_PIXELS;
            // SAFETY: first + 8 <= width: the loads read sums pixels first ..
            // first + 10 (the lead pixel at 0), the store output pixels
            // 2 first .. 2 first + 16.
            unsafe {
                let at = |pixel: usize| sums_ptr.add(pixel * channels);
                let dst_ptr = dst.as_mut_ptr().add(2 * first * channels);
                if channels == 1 {
                    let [before, center, after] =
                        [first, first + 1, first + 2].map(|pixel| vld1q_u16(at(pixel)));
                    vst1q_u8(dst_ptr, enlarged(before, center, after));
                } else {
                    let [before, center, after] =
                        [first, first + 1, first + 2].map(|pixel| vld3q_u16(at(pixel)));
                    let planes = uint8x16x3_t(
                        enlarged(before.0, center.0, after.0),
                        enlarged(before.1, center.1, after.1),
                        enlarged(before.2, center.2, after.2),
                    );
                    vst3q_u8(dst_ptr, planes);
                }
            }
        }
        horizontal_scalar(
            &sums[channels..channels + row_samples],
            channels,
            width,
            2 * steps * STEP_PIXELS..out_width,
            dst,
        );
    }
    // SAFETY: every row wrote all of its samples.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// `sums[x] = 3 near[x] + far[x]`.
#[target_feature(enable = "neon")]
fn vertical_sums(near: &[u8], far: &[u8], sums: &mut [u16]) {
    const BLOCK: usize = 16;
    let len = sums.len();
    let three = vdup_n_u8(3);
    let mut x = 0usize;
    while x + BLOCK <= len {
        // SAFETY: x + 16 <= len, the length of all three slices.
        unsafe {
            let (near_samples, far_samples) = (
                vld1q_u8(near.as_ptr().add(x)),
                vld1q_u8(far.as_ptr().add(x)),
            );
            let low = vaddw_u8(
                vmull_u8(vget_low_u8(near_samples), three),
                vget_low_u8(far_samples),
            );
            let high = vaddw_u8(
                vmull_u8(vget_high_u8(near_samples), three),
                vget_high_u8(far_samples),
            );
            vst1q_u16(sums.as_mut_ptr().add(x), low);
            vst1q_u16(sums.as_mut_ptr().add(x + 8), high);
        }
        x += BLOCK;
    }
    for x in x..len {
        sums[x] = 3 * u16::from(near[x]) + u16::from(far[x]);
    }
}

/// One channel plane's 16 outputs of 8 input pixels m (`center`, with the
/// planes of pixels m - 1 and m + 1): even `(3 s[m] + s[m - 1] + 8) >> 4`
/// and odd `(3 s[m] + s[m + 1] + 8) >> 4`, interleaved.
#[target_feature(enable = "neon")]
#[inline]
fn enlarged(before: uint16x8_t, center: uint16x8_t, after: uint16x8_t) -> uint8x16_t {
    let even = vrshrn_n_u16::<4>(vmlaq_n_u16(before, center, 3));
    let odd = vrshrn_n_u16::<4>(vmlaq_n_u16(after, center, 3));
    vcombine_u8(vzip1_u8(even, odd), vzip2_u8(even, odd))
}

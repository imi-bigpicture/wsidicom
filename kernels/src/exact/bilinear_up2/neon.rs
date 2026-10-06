//! The NEON kernel of the exact x2 bilinear enlargement.
//!
//! # Safety
//! NEON is baseline on aarch64, gated only by the `set_simd` kill switch.
//! Every `unsafe fn` here requires the bounds its doc states.

use std::arch::aarch64::{
    uint8x16_t, uint8x16x3_t, uint8x16x4_t, uint16x8_t, vaddw_u8, vcombine_u8, vdup_n_u8,
    vget_high_u8, vget_low_u8, vld1q_u8, vld1q_u16, vld3q_u16, vld4q_u16, vmlaq_n_u16, vmull_u8,
    vrshrn_n_u16, vst1q_u8, vst1q_u16, vst3q_u8, vst4q_u8, vzip1_u8, vzip2_u8,
};

use super::{far, horizontal_scalar};
use crate::sample::RGB;

/// Input pixels per horizontal step: 16 output pixels.
const STEP_PX: usize = 8;

/// The NEON kernel. Per output row: the vertical sums `3 near + far` (u16,
/// at most 4 x 255) with the row's first and last pixel repeated once
/// outside it — the scalar reference's clamped far tap, so every input pixel
/// m gives outputs 2m and 2m + 1 by the same formula, `(3 s[m] + s[m - 1] +
/// 8) >> 4` and `(3 s[m] + s[m + 1] + 8) >> 4` (`vrshrn` by 4), per channel
/// plane (`vld1`/`vld3`/`vld4`), interleaved by `vzip`. Pixels past the last
/// whole step by the scalar reference.
///
/// # Safety
/// `src` holds `width * height * channels` samples, `channels` 1, 3 or 4.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn bilinear_up2_neon(
    src: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Vec<u8> {
    let (ow, oh, ws) = (2 * width, 2 * height, width * channels);
    debug_assert!(src.len() >= width * height * channels);
    let mut out_buf = crate::cpu::unwritten_output::<u8>(ow * oh * channels);
    // SAFETY: u8 has no invalid values; every sample is written below before
    // the vector is returned.
    let out = unsafe {
        std::slice::from_raw_parts_mut(out_buf.as_mut_ptr().cast::<u8>(), ow * oh * channels)
    };
    // The vertical sums with one pixel before and after the row.
    let mut sums = vec![0u16; ws + 2 * channels];
    let steps = width / STEP_PX;
    for y in 0..oh {
        let (near, far) = (&src[y / 2 * ws..][..ws], &src[far(y, height) * ws..][..ws]);
        let row_sums = &mut sums[channels..channels + ws];
        vertical_sums(near, far, row_sums);
        // The edge pixels repeated before and after the row.
        sums.copy_within(channels..2 * channels, 0);
        sums.copy_within(ws..ws + channels, ws + channels);
        let dst = &mut out[y * ow * channels..(y + 1) * ow * channels];
        let sums_ptr = sums.as_ptr();
        for step in 0..steps {
            let first = step * STEP_PX;
            // SAFETY: first + 8 <= width: the loads read sums pixels first .. first + 10
            // (the lead pixel at 0), the store output pixels 2 first .. 2 first + 16.
            unsafe {
                let at = |pixel: usize| sums_ptr.add(pixel * channels);
                let dst_ptr = dst.as_mut_ptr().add(2 * first * channels);
                match channels {
                    1 => {
                        let [before, center, after] =
                            [first, first + 1, first + 2].map(|pixel| vld1q_u16(at(pixel)));
                        vst1q_u8(dst_ptr, enlarged(before, center, after));
                    }
                    RGB => {
                        let [before, center, after] =
                            [first, first + 1, first + 2].map(|pixel| vld3q_u16(at(pixel)));
                        let planes = uint8x16x3_t(
                            enlarged(before.0, center.0, after.0),
                            enlarged(before.1, center.1, after.1),
                            enlarged(before.2, center.2, after.2),
                        );
                        vst3q_u8(dst_ptr, planes);
                    }
                    _ => {
                        let [before, center, after] =
                            [first, first + 1, first + 2].map(|pixel| vld4q_u16(at(pixel)));
                        let planes = uint8x16x4_t(
                            enlarged(before.0, center.0, after.0),
                            enlarged(before.1, center.1, after.1),
                            enlarged(before.2, center.2, after.2),
                            enlarged(before.3, center.3, after.3),
                        );
                        vst4q_u8(dst_ptr, planes);
                    }
                }
            }
        }
        horizontal_scalar(
            &sums[channels..channels + ws],
            channels,
            width,
            2 * steps * STEP_PX..ow,
            dst,
        );
    }
    // SAFETY: every row wrote all of its samples.
    unsafe { crate::cpu::written_output(out_buf) }
}

/// `sums[x] = 3 near[x] + far[x]`: 16 samples per step; scalar tail.
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

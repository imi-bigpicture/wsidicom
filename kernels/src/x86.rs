//! Helpers shared by the `x86_64` kernels: tap grouping for the vertical
//! passes, streaming stores, and the RGB -> RGBX shuffles.

use std::ops::Range;

use crate::sample::CONVOLVE_U8_FRAC_BITS;
pub(crate) use crate::simd::cover_row;

/// `CONVOLVE_U8_FRAC_BITS` (Q6: the convolution's intermediate rows) as an
/// i32 shift immediate.
pub(crate) const CONVOLVE_FRAC_BITS: i32 = CONVOLVE_U8_FRAC_BITS as i32;

/// Whether a vertical pass over `taps` rows `row_bytes` apart goes through
/// `grouped_taps`. At row strides that are a multiple of 4 KB every window
/// row maps to the same L1 sets, so the rows' streams (and their prefetches)
/// evict each other; accumulating 8 rows at a time in the output row avoids
/// that. It pays there from 12 taps, and at half-aliased strides (a multiple
/// of 2 KB) from 16. At other strides it pays only for long windows.
#[inline]
pub(crate) fn groups_taps(row_bytes: usize, taps: usize) -> bool {
    const ALIASED_STRIDE: usize = 4096;
    const HALF_ALIASED_STRIDE: usize = 2048;
    const ALIASED_TAPS: usize = 12;
    const HALF_ALIASED_TAPS: usize = 16;
    const LONG_WINDOW_TAPS: usize = 40;
    let aliased = row_bytes.is_multiple_of(ALIASED_STRIDE) && taps >= ALIASED_TAPS;
    let half_aliased = row_bytes.is_multiple_of(HALF_ALIASED_STRIDE) && taps >= HALF_ALIASED_TAPS;
    taps >= LONG_WINDOW_TAPS || aliased || half_aliased
}

/// Prefetches the cache line at `x` of the two rows after a vertical window
/// of `taps` rows `stride` bytes apart from `rows`: the rows the next output
/// row's window adds on a 2x reduction (the window slides by the ratio), so
/// a source not in cache streams ahead of the pass. Past the image the
/// prefetch is a no-op (it never faults), so any `rows` is sound.
#[inline(always)]
pub(crate) fn prefetch_rows_after_window(rows: *const u8, taps: usize, stride: usize, x: usize) {
    use std::arch::x86_64::*;
    let next = rows.wrapping_add(taps * stride + x);
    // SAFETY: SSE is baseline on x86-64; a prefetch reads no memory
    // architecturally and cannot fault, whatever the address.
    unsafe {
        _mm_prefetch::<_MM_HINT_T0>(next as *const i8);
        _mm_prefetch::<_MM_HINT_T0>(next.wrapping_add(stride) as *const i8);
    }
}

/// Rows per group of `grouped_taps`.
const TAP_GROUP: usize = 8;
/// Output-row bytes per strip of `grouped_taps` (stays in L1 between groups).
const GROUP_STRIP_BYTES: usize = 4096;

/// The vertical pass over the blocks of `len` samples in `range` (a multiple
/// of `len` long) with the taps in groups: per strip of the output row, for
/// each group of `TAP_GROUP` taps, `block(x, taps)` over the strip's blocks,
/// which starts from zero for the first group and from the output row after
/// that. The sums are the same, in the same order.
#[inline(always)]
pub(crate) fn grouped_taps(
    range: Range<usize>,
    len: usize,
    sample_bytes: usize,
    taps: usize,
    mut block: impl FnMut(usize, Range<usize>),
) {
    let strip = (GROUP_STRIP_BYTES / sample_bytes).next_multiple_of(len);
    for lo in range.clone().step_by(strip) {
        let hi = (lo + strip).min(range.end);
        for t0 in (0..taps).step_by(TAP_GROUP) {
            let group = t0..(t0 + TAP_GROUP).min(taps);
            for x in (lo..hi).step_by(len) {
                block(x, group.clone());
            }
        }
    }
}

/// Rounding bias for a right shift by `frac_bits`: half of one output step.
pub(crate) const fn half_ulp(frac_bits: i32) -> i16 {
    1 << (frac_bits - 1)
}

/// 16-byte store that streams past the cache when `nt` is set and `dst` is
/// 16-aligned (the stream's requirement); otherwise a plain unaligned store.
///
/// # Safety
/// The 16 bytes at `dst` must be valid for writes.
#[inline(always)]
pub(crate) unsafe fn store_128i(
    dst: *mut std::arch::x86_64::__m128i,
    value: std::arch::x86_64::__m128i,
    nt: bool,
) {
    use std::arch::x86_64::*;
    if nt && (dst as usize).trailing_zeros() >= 4 {
        // SAFETY: `dst` is writable (the caller's contract) and 16-aligned
        // (checked); SSE2 is baseline on x86-64.
        unsafe { _mm_stream_si128(dst, value) };
    } else {
        // SAFETY: `dst` is writable (the caller's contract).
        unsafe { _mm_storeu_si128(dst, value) };
    }
}

/// `store_128i` for eight f32 (streams only when `dst` is 32-aligned).
///
/// # Safety
/// The CPU supports AVX, and the 8 floats at `dst` must be valid for writes.
#[target_feature(enable = "avx")]
#[inline]
pub(crate) unsafe fn store_256_ps(dst: *mut f32, value: std::arch::x86_64::__m256, nt: bool) {
    use std::arch::x86_64::*;
    if nt && (dst as usize).trailing_zeros() >= 5 {
        // SAFETY: `dst` is writable (the caller's contract) and 32-aligned
        // (checked).
        unsafe { _mm256_stream_ps(dst, value) };
    } else {
        // SAFETY: `dst` is writable (the caller's contract).
        unsafe { _mm256_storeu_ps(dst, value) };
    }
}

/// Orders a kernel's streaming stores before its buffer is handed out.
#[inline(always)]
pub(crate) fn sfence_if_streamed(nt: bool) {
    if nt {
        // SAFETY: SSE is baseline on x86-64; a store fence has no memory
        // preconditions.
        unsafe { std::arch::x86_64::_mm_sfence() };
    }
}

/// pshufb mask: per 128-bit lane, 4 RGB pixels (bytes 0..12) -> 4 RGBX slots
/// with the X byte zeroed (index high bit set), for `rgbx_pad_lanes` to fill.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) fn rgb_to_rgbx_shuffle() -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    _mm256_setr_epi8(
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1, //
        0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1,
    )
}

/// `pad` in every X byte (each 4th) and zero elsewhere, to OR into a shuffled
/// RGBX register whose X bytes are zero.
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) fn rgbx_pad_lanes(pad: u8) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let x_bytes = _mm256_setr_epi8(
        0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, //
        0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, -1,
    );
    _mm256_and_si256(_mm256_set1_epi8(pad as i8), x_bytes)
}

/// AVX2 uint8 RGB->RGBX expand: `pshufb` splices a `pad` byte after each pixel,
/// 8 px/iter — two 12-byte-offset 128-bit loads give each lane its 4 pixels,
/// one in-lane shuffle expands both. `src` is 3 * `pixels` RGB bytes, `dst`
/// 4 * `pixels` RGBX bytes.
///
/// # Safety
/// The CPU supports AVX2, `src.len() >= 3 * pixels` and
/// `dst.len() >= 4 * pixels` (neither is checked).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn expand_rgb_rgbx_avx2(src: &[u8], dst: &mut [u8], pixels: usize, pad: u8) {
    use std::arch::x86_64::*;
    let mask = rgb_to_rgbx_shuffle();
    let padv = rgbx_pad_lanes(pad);
    let (sp, dp) = (src.as_ptr(), dst.as_mut_ptr());
    let mut pixel = 0usize;
    // The lane-1 load reads src[3 * pixel + 12 .. 3 * pixel + 28]; the guard keeps
    // it in bounds.
    while pixel + 8 <= pixels && 3 * pixel + 28 <= src.len() {
        // SAFETY: src[3 * pixel .. 3 * pixel + 28] is in bounds (the guard).
        let rgb = unsafe {
            _mm256_inserti128_si256(
                _mm256_castsi128_si256(_mm_loadu_si128(sp.add(3 * pixel) as *const __m128i)),
                _mm_loadu_si128(sp.add(3 * pixel + 12) as *const __m128i),
                1,
            )
        };
        let rgbx = _mm256_or_si256(_mm256_shuffle_epi8(rgb, mask), padv);
        // SAFETY: pixel + 8 <= pixels, so dst[4 * pixel .. 4 * pixel + 32] is
        // inside the 4 * pixels bytes the caller guarantees.
        unsafe { _mm256_storeu_si256(dp.add(4 * pixel) as *mut __m256i, rgbx) };
        pixel += 8;
    }
    while pixel < pixels {
        // SAFETY: pixel < pixels: inside the 3 * pixels source and 4 * pixels
        // destination bytes the caller guarantees.
        unsafe {
            *dp.add(4 * pixel) = *sp.add(3 * pixel);
            *dp.add(4 * pixel + 1) = *sp.add(3 * pixel + 1);
            *dp.add(4 * pixel + 2) = *sp.add(3 * pixel + 2);
            *dp.add(4 * pixel + 3) = pad;
        }
        pixel += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Rng, random_samples};
    use std::arch::x86_64::*;

    /// RGB pixels widened to RGBX with `pad` in X.
    fn rgb_to_rgbx(rgb: &[u8], pad: u8) -> Vec<u8> {
        rgb.as_chunks::<3>()
            .0
            .iter()
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2], pad])
            .collect()
    }

    #[test]
    fn cover_row_covers_every_output_with_whole_blocks() {
        for width in [1, 4, 8, 16] {
            for len in width..=5 * width + 3 {
                let mut starts = Vec::new();
                cover_row(len, width, |start| starts.push(start));
                assert!(starts.iter().all(|&start| start + width <= len));
                assert!(starts.windows(2).all(|pair| pair[0] < pair[1]));
                assert_eq!(starts.len(), len.div_ceil(width), "{len} by {width}");
                let mut covered = vec![false; len];
                for start in starts {
                    covered[start..start + width].fill(true);
                }
                assert!(covered.iter().all(|&output| output), "{len} by {width}");
            }
        }
    }

    #[test]
    fn grouped_taps_visit_every_block_with_every_tap_group_in_order() {
        for (range, len, sample_bytes, taps) in [
            (0..48, 16, 1, 1),
            (0..48, 16, 1, 8),
            (32..(32 + 64 * 100), 64, 2, 21),
            (0..(32 * 300), 32, 4, 16),
            (16..(16 + 8 * 1000), 8, 1, 40),
        ] {
            let blocks = range.len() / len;
            let mut visits: Vec<Vec<Range<usize>>> = vec![Vec::new(); blocks];
            grouped_taps(range.clone(), len, sample_bytes, taps, |x, group| {
                assert!((x - range.start).is_multiple_of(len) && x < range.end);
                assert!(group.len() <= TAP_GROUP && group.end <= taps);
                visits[(x - range.start) / len].push(group);
            });
            for groups in visits {
                let taps_in_order: Vec<usize> = groups.into_iter().flatten().collect();
                assert_eq!(taps_in_order, (0..taps).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn taps_group_at_aliased_strides_and_long_windows() {
        assert!(groups_taps(3 * 4096, 12));
        assert!(!groups_taps(4096, 11));
        assert!(groups_taps(2048, 16));
        assert!(!groups_taps(2048, 15));
        assert!(groups_taps(1000, 40));
        assert!(!groups_taps(1000, 39));
    }

    #[test]
    fn half_ulp_is_half_of_one_output_step() {
        assert_eq!(half_ulp(1), 1);
        assert_eq!(half_ulp(CONVOLVE_FRAC_BITS), 32);
    }

    /// 64 bytes aligned to 32, so stores at every offset meet the streaming
    /// stores' alignment or miss it.
    #[repr(align(32))]
    struct AlignedBytes([u8; 64]);

    /// 24 floats aligned to 32, as `AlignedBytes`.
    #[repr(align(32))]
    struct AlignedFloats([f32; 24]);

    #[test]
    fn stores_write_their_values_streamed_or_not() {
        let value: [u8; 16] = std::array::from_fn(|index| index as u8 + 1);
        for nt in [false, true] {
            for offset in 0..=16 {
                let mut buffer = AlignedBytes([0; 64]);
                // SAFETY: offset + 16 <= 64: inside the buffer; `value` is 16
                // bytes.
                unsafe {
                    store_128i(
                        buffer.0.as_mut_ptr().add(offset) as *mut __m128i,
                        _mm_loadu_si128(value.as_ptr() as *const __m128i),
                        nt,
                    );
                }
                sfence_if_streamed(nt);
                assert_eq!(
                    buffer.0[offset..offset + 16],
                    value,
                    "offset {offset}, nt {nt}"
                );
                assert!(buffer.0[..offset].iter().all(|&byte| byte == 0));
                assert!(buffer.0[offset + 16..].iter().all(|&byte| byte == 0));
            }
        }
        if !std::is_x86_feature_detected!("avx") {
            return;
        }
        let floats: [f32; 8] = std::array::from_fn(|index| index as f32 + 0.5);
        for nt in [false, true] {
            for offset in 0..=8 {
                let mut buffer = AlignedFloats([0.0; 24]);
                // SAFETY: AVX is available; offset + 8 <= 24 floats: inside
                // the buffer; `floats` is 8 floats.
                unsafe {
                    store_256_ps(
                        buffer.0.as_mut_ptr().add(offset),
                        _mm256_loadu_ps(floats.as_ptr()),
                        nt,
                    );
                }
                sfence_if_streamed(nt);
                let written = &buffer.0;
                assert_eq!(
                    written[offset..offset + 8],
                    floats,
                    "offset {offset}, nt {nt}"
                );
                assert!(written[..offset].iter().all(|&sample| sample == 0.0));
                assert!(written[offset + 8..].iter().all(|&sample| sample == 0.0));
            }
        }
    }

    #[test]
    fn expand_rgb_rgbx_avx2_equals_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = Rng::new(60);
        for pixels in 0..=80 {
            // Exactly the pixels, then with source to spare (more vector
            // steps pass the source guard).
            for spare in [0, 32] {
                let src: Vec<u8> = random_samples(&mut rng, 3 * pixels + spare);
                for pad in [0, 0x6B, u8::MAX] {
                    let mut dst = vec![0x11; 4 * pixels];
                    // SAFETY: AVX2 is available; src holds `pixels` RGB pixels,
                    // dst as many RGBX ones.
                    unsafe { expand_rgb_rgbx_avx2(&src, &mut dst, pixels, pad) };
                    assert_eq!(dst, rgb_to_rgbx(&src[..3 * pixels], pad), "{pixels} pixels");
                }
            }
        }
    }

    #[test]
    fn prefetches_past_the_image_do_not_fault() {
        let rows = [0u8; 64];
        for (taps, stride, x) in [(0, 0, 0), (4, 4096, 63), (1000, 1 << 20, 1 << 12)] {
            prefetch_rows_after_window(rows.as_ptr(), taps, stride, x);
        }
    }
}

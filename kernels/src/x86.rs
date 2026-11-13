//! Helpers shared by the `x86_64` kernels: tap grouping for the vertical
//! passes and streaming stores.

use std::ops::Range;

use crate::sample::CONVOLVE_U8_FRAC_BITS;
pub(crate) use crate::simd::cover_row;

/// `CONVOLVE_U8_FRAC_BITS` as an i32 shift immediate.
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
    for strip_start in range.clone().step_by(strip) {
        let strip_end = (strip_start + strip).min(range.end);
        for group_start in (0..taps).step_by(TAP_GROUP) {
            let group = group_start..(group_start + TAP_GROUP).min(taps);
            for x in (strip_start..strip_end).step_by(len) {
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

/// Orders a kernel's streaming stores before its buffer is handed out.
#[inline(always)]
pub(crate) fn sfence_if_streamed(nt: bool) {
    if nt {
        // SAFETY: SSE is baseline on x86-64; a store fence has no memory
        // preconditions.
        unsafe { std::arch::x86_64::_mm_sfence() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::x86_64::*;

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

    /// 16-aligned, so offsets 0 and 16 meet the streaming store's alignment
    /// and the others miss it.
    #[repr(align(16))]
    struct AlignedBytes([u8; 64]);

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
    }
}

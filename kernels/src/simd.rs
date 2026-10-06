//! Helpers shared by the SIMD kernels of every instruction set.

/// Run `block(start)` for blocks of `width` outputs covering `0..len` (`len >= width`).
/// The last block is shifted back to end exactly at `len`, overlapping its
/// predecessor: it recomputes those outputs from the same inputs, so writing
/// them twice is harmless, and one full-width block replaces a narrow/scalar
/// tail, which on short rows costs as much as the vector body.
#[inline(always)]
pub(crate) fn cover_row(len: usize, width: usize, mut block: impl FnMut(usize)) {
    debug_assert!(len >= width);
    let mut start = 0;
    while start + width <= len {
        block(start);
        start += width;
    }
    if start < len {
        block(len - width);
    }
}

//! Helpers shared by the SIMD kernels of every instruction set.

/// Run `block(start)` for blocks of `width` outputs covering `0..len` (`len >= width`).
/// The last block is shifted back to end exactly at `len`, overlapping its
/// predecessor: it recomputes those outputs from the same inputs, so writing
/// them twice is harmless, and it replaces a scalar tail, which on short rows
/// costs as much as the vector body.
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

#[cfg(test)]
mod tests {
    use super::*;

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
}

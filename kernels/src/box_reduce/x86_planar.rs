//! Planar 2x2 and fused 4x4 box kernels (u8 / u16 / f32).
//!
//! # Safety
//! Every kernel requires the CPU features in its `#[target_feature]` (its
//! dispatch checks them at runtime) and `src.len() >= width * height`, which
//! it does not check. Loads are justified against that bound at their loop
//! guards; stores write the kernel's own output row.

use super::box2_mean;
use super::x86_output::BoxRows;
use crate::x86::*;

/// The frame every planar box kernel shares: `output`'s rows, row `oy`
/// written by `row(oy, out_row, nt)` (in a new image once its pages are
/// committed, see `BoxRows::write`), streamed stores fenced at the end.
/// `row` must write all of its row: the output is not cleared beforehand.
///
/// # Safety
/// `row(oy, out_row, nt)` must write each of the row's outputs at `out_row`
/// and nothing else.
#[inline(always)]
unsafe fn planar_box_to<T: Copy>(
    output: BoxRows<T>,
    mut row: impl FnMut(usize, *mut T, bool),
) -> Vec<T> {
    let (dp, layout, nt) = (output.dst(), output.layout(), output.streams());
    let rows = crate::cpu::row_bands(|oy_lo, oy_hi| {
        for oy in oy_lo..oy_hi {
            // SAFETY: oy is one of the output's rows, which start inside it.
            row(oy, unsafe { dp.add(layout.row_start(oy)) }, nt);
        }
    });
    // SAFETY: `row` writes every output of its row and nothing else (the
    // caller's contract; each kernel's body and tail cover the row).
    unsafe { output.write(nt, rows) }
}

/// `planar_box_to` into a new image of the `factor` reduction.
///
/// # Safety
/// As `planar_box_to`, for rows of `width / factor` outputs.
#[inline(always)]
unsafe fn planar_box<T: Copy>(
    width: usize,
    height: usize,
    factor: usize,
    row: impl FnMut(usize, *mut T, bool),
) -> (Vec<T>, usize, usize) {
    let (ow, oh) = (width / factor, height / factor);
    // SAFETY: the caller's contract.
    (
        unsafe { planar_box_to(BoxRows::new_image(oh, ow), row) },
        ow,
        oh,
    )
}

/// `row` typed as `planar_box`'s row closure (see `crate::cpu::row_bands`).
#[inline(always)]
fn planar_row<T, RowFn: FnMut(usize, *mut T, bool)>(row: RowFn) -> RowFn {
    row
}

/// Offsets into the source of the `ROWS` rows output row `oy` reduces.
fn source_rows<const ROWS: usize>(oy: usize, width: usize) -> [usize; ROWS] {
    std::array::from_fn(|row| (ROWS * oy + row) * width)
}

/// The exact sum of the 4x4 block at column `x` of `rows` (offsets into `sp`).
///
/// # Safety
/// The 4 samples from `row + x` of each of `rows` must be readable at `sp`.
#[inline(always)]
unsafe fn box4_sum<T: Copy + Into<u32>>(sp: *const T, rows: [usize; 4], x: usize) -> u32 {
    rows.iter()
        .map(|&row| {
            (0..4)
                // SAFETY: inside the block the caller guarantees readable.
                .map(|column| unsafe { *sp.add(row + x + column) }.into())
                .sum::<u32>()
        })
        .sum()
}

/// `box2x_u8_avx2_to` into a new image.
///
/// # Safety
/// As `box2x_u8_avx2_to`.
pub(crate) unsafe fn box2x_u8_avx2(
    src: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, usize, usize) {
    let (ow, oh) = (width / 2, height / 2);
    // SAFETY: the caller's contract.
    let image = unsafe { box2x_u8_avx2_to(src, width, height, BoxRows::new_image(oh, ow)) };
    (image, ow, oh)
}

/// AVX2 uint8 2x2 box average into `output`: maddubs sums horizontal pairs,
/// 16 out px/iter. Rounding (sum + 2) >> 2 is bit-identical to Pillow's
/// reduce(2).
///
/// # Safety
/// The CPU supports AVX2, `src.len() >= width * height`, and `output` holds
/// `height / 2` rows of `width / 2` samples.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box2x_u8_avx2_to(
    src: &[u8],
    width: usize,
    height: usize,
    output: BoxRows<u8>,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    let ow = width / 2;
    debug_assert_eq!(output.rows(), height / 2);
    let ones = _mm256_set1_epi8(1);
    let two = _mm256_set1_epi16(2);
    let sp = src.as_ptr();
    let row = planar_row::<u8, _>(|oy, out_row, nt| {
        let [r0, r1] = source_rows(oy, width);
        let mut ox = 0usize;
        while ox + 16 <= ow && 2 * ox + 32 <= width {
            let x = 2 * ox;
            // SAFETY: x + 32 <= width (the guard) in source rows 2oy, 2oy + 1
            // < height, inside the `width * height` samples of `src`.
            let (top, bottom) = unsafe {
                (
                    _mm256_loadu_si256(sp.add(r0 + x) as *const __m256i),
                    _mm256_loadu_si256(sp.add(r1 + x) as *const __m256i),
                )
            };
            // maddubs(x, 1) sums adjacent byte pairs -> 16 i16 horizontal sums.
            let sums = _mm256_add_epi16(
                _mm256_maddubs_epi16(top, ones),
                _mm256_maddubs_epi16(bottom, ones),
            );
            let rounded = _mm256_srli_epi16(_mm256_add_epi16(sums, two), 2);
            let packed = _mm_packus_epi16(
                _mm256_castsi256_si128(rounded),
                _mm256_extracti128_si256(rounded, 1),
            );
            // SAFETY: ox + 16 <= ow (the guard): inside the output row.
            unsafe { store_128i(out_row.add(ox) as *mut __m128i, packed, nt) };
            ox += 16;
        }
        while ox < ow {
            // SAFETY: ox < ow: the output is in the row, and its 2x2 block
            // (columns 2ox, 2ox + 1 < width of rows 2oy, 2oy + 1) in `src`.
            unsafe { *out_row.add(ox) = box2_mean(sp.add(r0 + 2 * ox), 1, width) };
            ox += 1;
        }
    });
    // SAFETY: `row` writes outputs 0..ow of its row (the vector body, then
    // the scalar tail up to ow) and nothing else.
    unsafe { planar_box_to::<u8>(output, row) }
}

/// AVX2 uint8 fused 4x4 box average (one pass of factor 4): maddubs sums byte
/// pairs per row (4 rows added: max 2040, fits i16), `madd_epi16` sums the i16
/// pairs to full 4x4 block sums in u32, then (sum + 8) >> 4. Single rounding —
/// closer to the true mean than two chained 2x passes. 16 output px/iter.
///
/// # Safety
/// The CPU supports AVX2, and `src.len() >= width * height`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box4x_u8_avx2(
    src: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, usize, usize) {
    use std::arch::x86_64::*;
    let ow = width / 4;
    let ones = _mm256_set1_epi8(1);
    let ones16 = _mm256_set1_epi16(1);
    let eight = _mm256_set1_epi32(8);
    let sp = src.as_ptr();
    let row = planar_row::<u8, _>(|oy, out_row, nt| {
        let rows = source_rows::<4>(oy, width);
        let mut ox = 0usize;
        // 8 rounded 4x4 quads (as u32) for the 32-column block starting at x.
        let quad8 = |x: usize| {
            let row_pair_sums = |row: usize| {
                // SAFETY: columns x..x + 32 <= width (the callers' guards) of
                // a source row < height, inside `src`.
                let pixels = unsafe { _mm256_loadu_si256(sp.add(row + x) as *const __m256i) };
                _mm256_maddubs_epi16(pixels, ones)
            };
            let row4_col2_sums = _mm256_add_epi16(
                _mm256_add_epi16(row_pair_sums(rows[0]), row_pair_sums(rows[1])),
                _mm256_add_epi16(row_pair_sums(rows[2]), row_pair_sums(rows[3])),
            );
            let quad_sums = _mm256_madd_epi16(row4_col2_sums, ones16);
            _mm256_srli_epi32(_mm256_add_epi32(quad_sums, eight), 4)
        };
        // Two 32-col blocks per iter fill one 16-byte (NT-streamable) store.
        while ox + 16 <= ow && 4 * ox + 64 <= width {
            let x = 4 * ox;
            let (ra, rb) = (quad8(x), quad8(x + 32));
            let p16a =
                _mm_packus_epi32(_mm256_castsi256_si128(ra), _mm256_extracti128_si256(ra, 1));
            let p16b =
                _mm_packus_epi32(_mm256_castsi256_si128(rb), _mm256_extracti128_si256(rb, 1));
            // SAFETY: ox + 16 <= ow (the guard): inside the output row.
            unsafe {
                store_128i(
                    out_row.add(ox) as *mut __m128i,
                    _mm_packus_epi16(p16a, p16b),
                    nt,
                );
            }
            ox += 16;
        }
        while ox + 8 <= ow && 4 * ox + 32 <= width {
            let means = quad8(4 * ox);
            let p16 = _mm_packus_epi32(
                _mm256_castsi256_si128(means),
                _mm256_extracti128_si256(means, 1),
            );
            let packed = _mm_packus_epi16(p16, p16);
            // SAFETY: ox + 8 <= ow (the guard): the 8 bytes are in the row.
            unsafe { _mm_storel_epi64(out_row.add(ox) as *mut __m128i, packed) };
            ox += 8;
        }
        while ox < ow {
            // SAFETY: ox < ow: the output is in the row, and its 4x4 block
            // (columns 4ox..4ox + 4 <= width of rows 4oy..4oy + 4 <= height)
            // in `src`.
            unsafe { *out_row.add(ox) = ((box4_sum(sp, rows, 4 * ox) + 8) >> 4) as u8 };
            ox += 1;
        }
    });
    // SAFETY: `row` writes outputs 0..ow of its row (the vector body, then
    // the scalar tail up to ow) and nothing else.
    unsafe { planar_box::<u8>(width, height, 4, row) }
}

/// AVX2 uint16 fused 4x4 box average: `(v & 0xFFFF) + (v >> 16)` sums adjacent
/// u16 pairs into ordered u32 (see `box2x_u16_avx2`), accumulated over the 4
/// rows; one hadd builds the 4-column quads (lane-scrambled to
/// [q0,q1,q4,q5|q2,q3,q6,q7] — one vpermd restores order), then single-rounded
/// (sum + 8) >> 4. 8 output px/iter.
///
/// # Safety
/// The CPU supports AVX2, and `src.len() >= width * height`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box4x_u16_avx2(
    src: &[u16],
    width: usize,
    height: usize,
) -> (Vec<u16>, usize, usize) {
    use std::arch::x86_64::*;
    let ow = width / 4;
    let eight = _mm256_set1_epi32(8);
    let lo16 = _mm256_set1_epi32(0xFFFF);
    let perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    let sp = src.as_ptr();
    let row = planar_row::<u16, _>(|oy, out_row, nt| {
        let rows = source_rows::<4>(oy, width);
        let mut ox = 0usize;
        while ox + 8 <= ow && 4 * ox + 32 <= width {
            let x = 4 * ox;
            let mut col_pairs_0_8 = _mm256_setzero_si256();
            let mut col_pairs_8_16 = _mm256_setzero_si256();
            for row in rows {
                // SAFETY: columns x..x + 32 <= width (the guard) of a source
                // row < height, inside `src`.
                let (cols_0_16, cols_16_32) = unsafe {
                    (
                        _mm256_loadu_si256(sp.add(row + x) as *const __m256i),
                        _mm256_loadu_si256(sp.add(row + x + 16) as *const __m256i),
                    )
                };
                col_pairs_0_8 = _mm256_add_epi32(
                    col_pairs_0_8,
                    _mm256_add_epi32(
                        _mm256_and_si256(cols_0_16, lo16),
                        _mm256_srli_epi32(cols_0_16, 16),
                    ),
                );
                col_pairs_8_16 = _mm256_add_epi32(
                    col_pairs_8_16,
                    _mm256_add_epi32(
                        _mm256_and_si256(cols_16_32, lo16),
                        _mm256_srli_epi32(cols_16_32, 16),
                    ),
                );
            }
            let quad_sums =
                _mm256_permutevar8x32_epi32(_mm256_hadd_epi32(col_pairs_0_8, col_pairs_8_16), perm);
            let means = _mm256_srli_epi32(_mm256_add_epi32(quad_sums, eight), 4);
            let p16 = _mm_packus_epi32(
                _mm256_castsi256_si128(means),
                _mm256_extracti128_si256(means, 1),
            );
            // SAFETY: ox + 8 <= ow (the guard): inside the output row.
            unsafe { store_128i(out_row.add(ox) as *mut __m128i, p16, nt) };
            ox += 8;
        }
        while ox < ow {
            // SAFETY: as box4x_u8_avx2's tail.
            unsafe { *out_row.add(ox) = ((box4_sum(sp, rows, 4 * ox) + 8) >> 4) as u16 };
            ox += 1;
        }
    });
    // SAFETY: `row` writes outputs 0..ow of its row (the vector body, then
    // the scalar tail up to ow) and nothing else.
    unsafe { planar_box::<u16>(width, height, 4, row) }
}

/// AVX2 float32 fused 4x4 box average: each row's 32 columns added down the
/// 4 rows as four registers c0..c3 (columns 8k..8k + 8, 4 per 128-bit lane);
/// two levels of `hadd_ps` reduce them to the 4-column quads, lane-scrambled to
/// [q0,q2,q4,q6|q1,q3,q5,q7] (lane L of ck holds quad 2k + L), one vpermps
/// restores order; * 1/16. 8 output px/iter.
///
/// # Safety
/// The CPU supports AVX2, and `src.len() >= width * height`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box4x_f32_avx2(
    src: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, usize, usize) {
    use std::arch::x86_64::*;
    let ow = width / 4;
    let sixteenth = _mm256_set1_ps(1.0 / 16.0);
    let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let sp = src.as_ptr();
    let row = planar_row::<f32, _>(|oy, out_row, nt| {
        let rows = source_rows::<4>(oy, width);
        let mut ox = 0usize;
        while ox + 8 <= ow && 4 * ox + 32 <= width {
            let x = 4 * ox;
            let mut col_sums = [_mm256_setzero_ps(); 4];
            for row in rows {
                for (block, block_sums) in col_sums.iter_mut().enumerate() {
                    // SAFETY: columns x..x + 32 <= width (the guard) of a
                    // source row < height, inside `src`.
                    let columns = unsafe { _mm256_loadu_ps(sp.add(row + x + 8 * block)) };
                    *block_sums = _mm256_add_ps(*block_sums, columns);
                }
            }
            let quads = _mm256_hadd_ps(
                _mm256_hadd_ps(col_sums[0], col_sums[1]),
                _mm256_hadd_ps(col_sums[2], col_sums[3]),
            );
            let quads = _mm256_permutevar8x32_ps(quads, perm);
            // SAFETY: ox + 8 <= ow (the guard): inside the output row.
            unsafe { store_256_ps(out_row.add(ox), _mm256_mul_ps(quads, sixteenth), nt) };
            ox += 8;
        }
        while ox < ow {
            // The body's order: columns down the rows, then the column pairs.
            let x = 4 * ox;
            // SAFETY: ox < ow: column x + column < 4ow <= width of rows
            // 4oy..4oy + 4 <= height, inside `src`.
            let col = |column: usize| unsafe {
                ((*sp.add(rows[0] + x + column) + *sp.add(rows[1] + x + column))
                    + *sp.add(rows[2] + x + column))
                    + *sp.add(rows[3] + x + column)
            };
            let mean = ((col(0) + col(1)) + (col(2) + col(3))) * (1.0 / 16.0);
            // SAFETY: ox < ow: inside the output row.
            unsafe { *out_row.add(ox) = mean };
            ox += 1;
        }
    });
    // SAFETY: `row` writes outputs 0..ow of its row (the vector body, then
    // the scalar tail up to ow) and nothing else.
    unsafe { planar_box::<f32>(width, height, 4, row) }
}

/// AVX2 uint16 2x2 box average. No maddubs for u16 (values >= 32768 would go
/// negative as i16), but `(v & 0xFFFF) + (v >> 16)` sums adjacent u16 pairs
/// into u32 — in order, no widen/extract/permute needed. Exact sums, so
/// rounding (sum + 2) >> 2 stays bit-identical to `box2x_scalar` and Pillow.
/// 8 output px/iter.
///
/// # Safety
/// The CPU supports AVX2, `src.len() >= width * height`, and `output` holds
/// `height / 2` rows of `width / 2` samples.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box2x_u16_avx2_to(
    src: &[u16],
    width: usize,
    height: usize,
    output: BoxRows<u16>,
) -> Vec<u16> {
    use std::arch::x86_64::*;
    let ow = width / 2;
    debug_assert_eq!(output.rows(), height / 2);
    let two = _mm256_set1_epi32(2);
    let lo16 = _mm256_set1_epi32(0xFFFF);
    let sp = src.as_ptr();
    let row = planar_row::<u16, _>(|oy, out_row, nt| {
        let [r0, r1] = source_rows(oy, width);
        let mut ox = 0usize;
        while ox + 8 <= ow && 2 * ox + 16 <= width {
            let x = 2 * ox;
            // SAFETY: x + 16 <= width (the guard) in source rows 2oy, 2oy + 1
            // < height, inside `src`.
            let (top, bottom) = unsafe {
                (
                    _mm256_loadu_si256(sp.add(r0 + x) as *const __m256i),
                    _mm256_loadu_si256(sp.add(r1 + x) as *const __m256i),
                )
            };
            let pair_sums0 =
                _mm256_add_epi32(_mm256_and_si256(top, lo16), _mm256_srli_epi32(top, 16));
            let pair_sums1 = _mm256_add_epi32(
                _mm256_and_si256(bottom, lo16),
                _mm256_srli_epi32(bottom, 16),
            );
            let block_sums = _mm256_add_epi32(pair_sums0, pair_sums1);
            let rounded = _mm256_srli_epi32(_mm256_add_epi32(block_sums, two), 2);
            let packed = _mm_packus_epi32(
                _mm256_castsi256_si128(rounded),
                _mm256_extracti128_si256(rounded, 1),
            );
            // SAFETY: ox + 8 <= ow (the guard): inside the output row.
            unsafe { store_128i(out_row.add(ox) as *mut __m128i, packed, nt) };
            ox += 8;
        }
        while ox < ow {
            // SAFETY: ox < ow: the output is in the row, and its 2x2 block
            // (columns 2ox, 2ox + 1 < width of rows 2oy, 2oy + 1) in `src`.
            unsafe { *out_row.add(ox) = box2_mean(sp.add(r0 + 2 * ox), 1, width) };
            ox += 1;
        }
    });
    // SAFETY: `row` writes outputs 0..ow of its row (the vector body, then
    // the scalar tail up to ow) and nothing else.
    unsafe { planar_box_to::<u16>(output, row) }
}

/// `box2x_u16_avx2_to` into a new image.
///
/// # Safety
/// As `box2x_u16_avx2_to`.
pub(crate) unsafe fn box2x_u16_avx2(
    src: &[u16],
    width: usize,
    height: usize,
) -> (Vec<u16>, usize, usize) {
    let (ow, oh) = (width / 2, height / 2);
    // SAFETY: the caller's contract.
    let image = unsafe { box2x_u16_avx2_to(src, width, height, BoxRows::new_image(oh, ow)) };
    (image, ow, oh)
}

/// `box2x_f32_avx2_to` into a new image.
///
/// # Safety
/// As `box2x_f32_avx2_to`.
pub(crate) unsafe fn box2x_f32_avx2(
    src: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, usize, usize) {
    let (ow, oh) = (width / 2, height / 2);
    // SAFETY: the caller's contract.
    let image = unsafe { box2x_f32_avx2_to(src, width, height, BoxRows::new_image(oh, ow)) };
    (image, ow, oh)
}

/// AVX2 float32 2x2 box average into `output`: sum the two rows, `hadd_ps`
/// the column pairs, fix lane order, * 0.25. 8 output px/iter. No
/// rounding/clamp needed.
///
/// # Safety
/// The CPU supports AVX2, `src.len() >= width * height`, and `output` holds
/// `height / 2` rows of `width / 2` samples.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box2x_f32_avx2_to(
    src: &[f32],
    width: usize,
    height: usize,
    output: BoxRows<f32>,
) -> Vec<f32> {
    use std::arch::x86_64::*;
    let ow = width / 2;
    debug_assert_eq!(output.rows(), height / 2);
    let quarter = _mm256_set1_ps(0.25);
    let perm = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    let sp = src.as_ptr();
    let row = planar_row::<f32, _>(|oy, out_row, nt| {
        let [r0, r1] = source_rows(oy, width);
        let mut ox = 0usize;
        while ox + 8 <= ow && 2 * ox + 16 <= width {
            let x = 2 * ox;
            // SAFETY: x + 16 <= width (the guard) in source rows 2oy, 2oy + 1
            // < height, inside `src`.
            let (row_lo, row_hi) = unsafe {
                (
                    _mm256_add_ps(
                        _mm256_loadu_ps(sp.add(r0 + x)),
                        _mm256_loadu_ps(sp.add(r1 + x)),
                    ),
                    _mm256_add_ps(
                        _mm256_loadu_ps(sp.add(r0 + x + 8)),
                        _mm256_loadu_ps(sp.add(r1 + x + 8)),
                    ),
                )
            };
            let hs = _mm256_permutevar8x32_ps(_mm256_hadd_ps(row_lo, row_hi), perm);
            // SAFETY: ox + 8 <= ow (the guard): inside the output row.
            unsafe { store_256_ps(out_row.add(ox), _mm256_mul_ps(hs, quarter), nt) };
            ox += 8;
        }
        while ox < ow {
            // SAFETY: ox < ow: the output is in the row, and its 2x2 block
            // (columns 2ox, 2ox + 1 < width of rows 2oy, 2oy + 1) in `src`.
            unsafe { *out_row.add(ox) = box2_mean(sp.add(r0 + 2 * ox), 1, width) };
            ox += 1;
        }
    });
    // SAFETY: `row` writes outputs 0..ow of its row (the vector body, then
    // the scalar tail up to ow) and nothing else.
    unsafe { planar_box_to::<f32>(output, row) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box_reduce::box2x_scalar;
    use crate::box_reduce::tests::{
        BoxTestSample, assert_same_image, box_sizes, box4x_chained_reference,
        box4x_fused_reference, test_images,
    };
    use crate::cpu::{LARGE_OUTPUT_BYTES, PREFAULT_OUTPUT_BYTES};
    use crate::testing::{ImageKernel, Rng, UnsafeImageKernel, random_samples, sizes_from};

    /// Runs `kernel` (needing AVX2) and `reference` on random and extreme
    /// images of every test size of at least `minimum`, and asserts the same
    /// output.
    fn assert_matches_reference<T: BoxTestSample>(
        minimum: usize,
        kernel: UnsafeImageKernel<T>,
        reference: ImageKernel<T>,
    ) {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = Rng::new(1);
        for (width, height) in box_sizes(minimum) {
            for src in test_images::<T>(&mut rng, width * height) {
                // SAFETY: AVX2 is available; src holds width * height samples.
                let simd = unsafe { kernel(&src, width, height) };
                assert_same_image(
                    &simd,
                    &reference(&src, width, height),
                    &format!("{width} x {height}"),
                );
            }
        }
    }

    #[test]
    fn box2x_avx2_equals_scalar() {
        assert_matches_reference::<u8>(1, box2x_u8_avx2, box2x_scalar);
        assert_matches_reference::<u16>(1, box2x_u16_avx2, box2x_scalar);
        assert_matches_reference::<f32>(1, box2x_f32_avx2, box2x_scalar);
    }

    #[test]
    fn box4x_avx2_equals_the_fused_reference() {
        assert_matches_reference::<u8>(1, box4x_u8_avx2, box4x_fused_reference);
        assert_matches_reference::<u16>(1, box4x_u16_avx2, box4x_fused_reference);
        assert_matches_reference::<f32>(1, box4x_f32_avx2, box4x_fused_reference);
    }

    #[test]
    fn box4x_avx2_integers_equal_two_scalar_halvings_within_one() {
        // The fused 4x kernel rounds once; two halvings round twice, so
        // integers may differ by 1.
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = Rng::new(2);
        for (width, height) in sizes_from(4) {
            let src: Vec<u8> = random_samples(&mut rng, width * height);
            // SAFETY: AVX2 is available; src holds width * height samples.
            let (fused, ..) = unsafe { box4x_u8_avx2(&src, width, height) };
            let (chained, ..) = box4x_chained_reference(&src, width, height);
            for (fused, chained) in fused.iter().zip(&chained) {
                assert!(fused.abs_diff(*chained) <= 1, "{width} x {height}");
            }
            let src: Vec<u16> = random_samples(&mut rng, width * height);
            // SAFETY: AVX2 is available; src holds width * height samples.
            let (fused, ..) = unsafe { box4x_u16_avx2(&src, width, height) };
            let (chained, ..) = box4x_chained_reference(&src, width, height);
            for (fused, chained) in fused.iter().zip(&chained) {
                assert!(fused.abs_diff(*chained) <= 1, "{width} x {height}");
            }
        }
    }

    /// `kernel` on an image whose halved output is `out_width` x
    /// `out_height`, against `reference`.
    fn assert_large_matches<T: BoxTestSample>(
        (out_width, out_height): (usize, usize),
        kernel: UnsafeImageKernel<T>,
        reference: ImageKernel<T>,
    ) {
        let (width, height) = (2 * out_width + 1, 2 * out_height + 1);
        let src: Vec<T> = random_samples(&mut Rng::new(3), width * height);
        // SAFETY: AVX2 is available (the caller checked); src holds width *
        // height samples.
        let simd = unsafe { kernel(&src, width, height) };
        assert_same_image(&simd, &reference(&src, width, height), "large");
    }

    #[test]
    fn large_outputs_stream_past_the_cache_and_prefault() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        // Outputs past both thresholds: streaming stores and the prefault
        // thread, with rows that are not a multiple of the vector width.
        let out_bytes = PREFAULT_OUTPUT_BYTES.max(LARGE_OUTPUT_BYTES);
        let out_width = 2052;
        let out_height = out_bytes.div_ceil(size_of::<u16>() * out_width);
        assert_large_matches::<u16>((out_width, out_height), box2x_u16_avx2, box2x_scalar);
        let out_height = out_bytes.div_ceil(size_of::<f32>() * out_width);
        assert_large_matches::<f32>((out_width, out_height), box2x_f32_avx2, box2x_scalar);
    }
}

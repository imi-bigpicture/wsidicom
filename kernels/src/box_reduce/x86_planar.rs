//! Planar 2x2 and fused 4x4 box kernels (u8 / u16).
//!
//! Every kernel checks `src.len() >= width * height`, so it requires only
//! the CPU features in its `#[target_feature]` (its dispatch checks them at
//! runtime). Loads are justified against that bound at their loop guards;
//! stores write the kernel's own output row.

use super::box2_mean;
use super::x86_output::BoxRows;
use crate::x86::*;

/// The frame every planar box kernel shares: `output`'s rows, row `out_y`
/// written by `row(out_y, out_row, stream)` (in a new image once its pages
/// are committed, see `BoxRows::write`), streamed stores fenced at the end.
///
/// # Safety
/// `row(out_y, out_row, stream)` must write each of the row's outputs at
/// `out_row` and nothing else: the output is not cleared beforehand.
#[inline(always)]
unsafe fn planar_box_into<T: Copy>(
    output: BoxRows<T>,
    mut row: impl FnMut(usize, *mut T, bool),
) -> Vec<T> {
    let (dst_ptr, layout, stream) = (output.dst(), output.layout(), output.streams());
    let rows = crate::cpu::row_bands(|first_out_y, end_out_y| {
        for out_y in first_out_y..end_out_y {
            // SAFETY: out_y is one of the output's rows, which start inside it.
            row(
                out_y,
                unsafe { dst_ptr.add(layout.row_start(out_y)) },
                stream,
            );
        }
    });
    // SAFETY: `row` writes every output of its row and nothing else (the
    // caller's contract; each kernel's body and tail cover the row).
    unsafe { output.write(stream, rows) }
}

/// `planar_box_into` into a new image of the `factor` reduction.
///
/// # Safety
/// As `planar_box_into`, for rows of `width / factor` outputs.
#[inline(always)]
unsafe fn planar_box<T: Copy>(
    width: usize,
    height: usize,
    factor: usize,
    row: impl FnMut(usize, *mut T, bool),
) -> (Vec<T>, usize, usize) {
    let (out_width, out_height) = (width / factor, height / factor);
    // SAFETY: the caller's contract.
    (
        unsafe { planar_box_into(BoxRows::new_image(out_height, out_width), row) },
        out_width,
        out_height,
    )
}

/// `row` typed as `planar_box`'s row closure (see `crate::cpu::row_bands`).
#[inline(always)]
fn planar_row<T, RowFn: FnMut(usize, *mut T, bool)>(row: RowFn) -> RowFn {
    row
}

/// Offsets into the source of the `ROWS` rows output row `out_y` reduces.
fn source_rows<const ROWS: usize>(out_y: usize, width: usize) -> [usize; ROWS] {
    std::array::from_fn(|row| (ROWS * out_y + row) * width)
}

/// The exact sum of the 4x4 block at column `x` of `rows` (offsets into
/// `src_ptr`).
///
/// # Safety
/// The 4 samples from `row + x` of each of `rows` must be readable at
/// `src_ptr`.
#[inline(always)]
unsafe fn box4_sum<T: Copy + Into<u32>>(src_ptr: *const T, rows: [usize; 4], x: usize) -> u32 {
    rows.iter()
        .map(|&row| {
            (0..4)
                // SAFETY: inside the block the caller guarantees readable.
                .map(|column| unsafe { *src_ptr.add(row + x + column) }.into())
                .sum::<u32>()
        })
        .sum()
}

/// `box2x_u8_into_avx2` into a new image.
#[target_feature(enable = "avx2")]
pub(crate) fn box2x_u8_avx2(src: &[u8], width: usize, height: usize) -> (Vec<u8>, usize, usize) {
    let (out_width, out_height) = (width / 2, height / 2);
    let image = box2x_u8_into_avx2(
        src,
        width,
        height,
        BoxRows::new_image(out_height, out_width),
    );
    (image, out_width, out_height)
}

/// AVX2 uint8 2x2 box into `output`, 16 outputs per step: maddubs sums
/// horizontal pairs, rounded `(sum + 2) >> 2` as Pillow's reduce(2).
///
/// Panics unless `src` holds a `width` x `height` image and `output` its
/// `height / 2` rows of `width / 2` samples.
#[target_feature(enable = "avx2")]
pub(crate) fn box2x_u8_into_avx2(
    src: &[u8],
    width: usize,
    height: usize,
    output: BoxRows<u8>,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    let out_width = width / 2;
    crate::sample::assert_image(src, width, height, 1);
    output.assert_rows(height / 2, width / 2);
    let ones = _mm256_set1_epi8(1);
    let two = _mm256_set1_epi16(2);
    let src_ptr = src.as_ptr();
    let row = planar_row::<u8, _>(|out_y, out_row, stream| {
        let [top_row, bottom_row] = source_rows(out_y, width);
        let mut out_x = 0usize;
        while out_x + 16 <= out_width && 2 * out_x + 32 <= width {
            let x = 2 * out_x;
            // SAFETY: x + 32 <= width (the guard) in source rows up to
            // 2 out_y + 1 < height, inside `src`.
            let (top, bottom) = unsafe {
                (
                    _mm256_loadu_si256(src_ptr.add(top_row + x) as *const __m256i),
                    _mm256_loadu_si256(src_ptr.add(bottom_row + x) as *const __m256i),
                )
            };
            let sums = _mm256_add_epi16(
                _mm256_maddubs_epi16(top, ones),
                _mm256_maddubs_epi16(bottom, ones),
            );
            let rounded = _mm256_srli_epi16(_mm256_add_epi16(sums, two), 2);
            let packed = _mm_packus_epi16(
                _mm256_castsi256_si128(rounded),
                _mm256_extracti128_si256(rounded, 1),
            );
            // SAFETY: out_x + 16 <= out_width (the guard): in the row.
            unsafe { store_128i(out_row.add(out_x) as *mut __m128i, packed, stream) };
            out_x += 16;
        }
        while out_x < out_width {
            // SAFETY: out_x < width / 2, out_y < height / 2: the output is in
            // the row, its 2x2 block in `src`.
            unsafe { *out_row.add(out_x) = box2_mean(src_ptr.add(top_row + 2 * out_x), 1, width) };
            out_x += 1;
        }
    });
    // SAFETY: `row` writes outputs 0..out_width of its row (the vector body,
    // then the scalar tail) and nothing else.
    unsafe { planar_box_into::<u8>(output, row) }
}

/// AVX2 uint8 fused 4x4 box, 16 outputs per step: maddubs sums byte pairs
/// per row (4 rows: at most 2040, fits i16), `madd_epi16` sums those pairs
/// to the blocks' sums in u32, then `(sum + 8) >> 4`.
///
/// Panics unless `src` holds a `width` x `height` image.
#[target_feature(enable = "avx2")]
pub(crate) fn box4x_u8_avx2(src: &[u8], width: usize, height: usize) -> (Vec<u8>, usize, usize) {
    use std::arch::x86_64::*;
    crate::sample::assert_image(src, width, height, 1);
    let out_width = width / 4;
    let ones = _mm256_set1_epi8(1);
    let ones_i16 = _mm256_set1_epi16(1);
    let eight = _mm256_set1_epi32(8);
    let src_ptr = src.as_ptr();
    let row = planar_row::<u8, _>(|out_y, out_row, stream| {
        let rows = source_rows::<4>(out_y, width);
        let mut out_x = 0usize;
        // The 8 rounded 4x4 means (as u32) of the 32 columns at x.
        let means_of_32_columns = |x: usize| {
            let row_pair_sums = |row: usize| {
                // SAFETY: columns x..x + 32 <= width (the callers' guards) of
                // a source row < height, inside `src`.
                let pixels = unsafe { _mm256_loadu_si256(src_ptr.add(row + x) as *const __m256i) };
                _mm256_maddubs_epi16(pixels, ones)
            };
            let row4_col2_sums = _mm256_add_epi16(
                _mm256_add_epi16(row_pair_sums(rows[0]), row_pair_sums(rows[1])),
                _mm256_add_epi16(row_pair_sums(rows[2]), row_pair_sums(rows[3])),
            );
            let quad_sums = _mm256_madd_epi16(row4_col2_sums, ones_i16);
            _mm256_srli_epi32(_mm256_add_epi32(quad_sums, eight), 4)
        };
        // Two 32-column blocks per step fill one 16-byte (streamable) store.
        while out_x + 16 <= out_width && 4 * out_x + 64 <= width {
            let x = 4 * out_x;
            let (first_means, second_means) = (means_of_32_columns(x), means_of_32_columns(x + 32));
            let first_u16 = _mm_packus_epi32(
                _mm256_castsi256_si128(first_means),
                _mm256_extracti128_si256(first_means, 1),
            );
            let second_u16 = _mm_packus_epi32(
                _mm256_castsi256_si128(second_means),
                _mm256_extracti128_si256(second_means, 1),
            );
            // SAFETY: out_x + 16 <= out_width (the guard): in the row.
            unsafe {
                store_128i(
                    out_row.add(out_x) as *mut __m128i,
                    _mm_packus_epi16(first_u16, second_u16),
                    stream,
                );
            }
            out_x += 16;
        }
        while out_x + 8 <= out_width && 4 * out_x + 32 <= width {
            let means = means_of_32_columns(4 * out_x);
            let means_u16 = _mm_packus_epi32(
                _mm256_castsi256_si128(means),
                _mm256_extracti128_si256(means, 1),
            );
            let packed = _mm_packus_epi16(means_u16, means_u16);
            // SAFETY: out_x + 8 <= out_width (the guard): in the row.
            unsafe { _mm_storel_epi64(out_row.add(out_x) as *mut __m128i, packed) };
            out_x += 8;
        }
        while out_x < out_width {
            // SAFETY: out_x < width / 4, out_y < height / 4: the output is in
            // the row, its 4x4 block in `src`.
            unsafe {
                *out_row.add(out_x) = ((box4_sum(src_ptr, rows, 4 * out_x) + 8) >> 4) as u8;
            };
            out_x += 1;
        }
    });
    // SAFETY: `row` writes outputs 0..out_width of its row (the vector body,
    // then the scalar tail) and nothing else.
    unsafe { planar_box::<u8>(width, height, 4, row) }
}

/// AVX2 uint16 fused 4x4 box, 8 outputs per step: adjacent u16 pairs summed
/// into u32 in order (see `box2x_u16_into_avx2`) down the 4 rows; one hadd
/// forms the blocks' sums, lane-scrambled to [q0,q1,q4,q5|q2,q3,q6,q7] (one
/// vpermd restores the order), then `(sum + 8) >> 4`.
///
/// Panics unless `src` holds a `width` x `height` image.
#[target_feature(enable = "avx2")]
pub(crate) fn box4x_u16_avx2(src: &[u16], width: usize, height: usize) -> (Vec<u16>, usize, usize) {
    use std::arch::x86_64::*;
    crate::sample::assert_image(src, width, height, 1);
    let out_width = width / 4;
    let eight = _mm256_set1_epi32(8);
    let low_u16_mask = _mm256_set1_epi32(0xFFFF);
    let quad_order = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    let src_ptr = src.as_ptr();
    let row = planar_row::<u16, _>(|out_y, out_row, stream| {
        let rows = source_rows::<4>(out_y, width);
        let mut out_x = 0usize;
        while out_x + 8 <= out_width && 4 * out_x + 32 <= width {
            let x = 4 * out_x;
            let mut col_pairs_0_8 = _mm256_setzero_si256();
            let mut col_pairs_8_16 = _mm256_setzero_si256();
            for row in rows {
                // SAFETY: columns x..x + 32 <= width (the guard) of a source
                // row < height, inside `src`.
                let (cols_0_16, cols_16_32) = unsafe {
                    (
                        _mm256_loadu_si256(src_ptr.add(row + x) as *const __m256i),
                        _mm256_loadu_si256(src_ptr.add(row + x + 16) as *const __m256i),
                    )
                };
                col_pairs_0_8 = _mm256_add_epi32(
                    col_pairs_0_8,
                    _mm256_add_epi32(
                        _mm256_and_si256(cols_0_16, low_u16_mask),
                        _mm256_srli_epi32(cols_0_16, 16),
                    ),
                );
                col_pairs_8_16 = _mm256_add_epi32(
                    col_pairs_8_16,
                    _mm256_add_epi32(
                        _mm256_and_si256(cols_16_32, low_u16_mask),
                        _mm256_srli_epi32(cols_16_32, 16),
                    ),
                );
            }
            let quad_sums = _mm256_permutevar8x32_epi32(
                _mm256_hadd_epi32(col_pairs_0_8, col_pairs_8_16),
                quad_order,
            );
            let means = _mm256_srli_epi32(_mm256_add_epi32(quad_sums, eight), 4);
            let means_u16 = _mm_packus_epi32(
                _mm256_castsi256_si128(means),
                _mm256_extracti128_si256(means, 1),
            );
            // SAFETY: out_x + 8 <= out_width (the guard): in the row.
            unsafe { store_128i(out_row.add(out_x) as *mut __m128i, means_u16, stream) };
            out_x += 8;
        }
        while out_x < out_width {
            // SAFETY: as box4x_u8_avx2's tail.
            unsafe {
                *out_row.add(out_x) = ((box4_sum(src_ptr, rows, 4 * out_x) + 8) >> 4) as u16;
            };
            out_x += 1;
        }
    });
    // SAFETY: `row` writes outputs 0..out_width of its row (the vector body,
    // then the scalar tail) and nothing else.
    unsafe { planar_box::<u16>(width, height, 4, row) }
}

/// AVX2 uint16 2x2 box into `output`, 8 outputs per step. maddubs would
/// treat u16 samples >= 32768 as negative, but `(v & 0xFFFF) + (v >> 16)`
/// sums adjacent u16 pairs into u32 in order; the sums are exact, so
/// `(sum + 2) >> 2` equals `box2x_scalar`.
///
/// Panics unless `src` holds a `width` x `height` image and `output` its
/// `height / 2` rows of `width / 2` samples.
#[target_feature(enable = "avx2")]
pub(crate) fn box2x_u16_into_avx2(
    src: &[u16],
    width: usize,
    height: usize,
    output: BoxRows<u16>,
) -> Vec<u16> {
    use std::arch::x86_64::*;
    let out_width = width / 2;
    crate::sample::assert_image(src, width, height, 1);
    output.assert_rows(height / 2, width / 2);
    let two = _mm256_set1_epi32(2);
    let low_u16_mask = _mm256_set1_epi32(0xFFFF);
    let src_ptr = src.as_ptr();
    let row = planar_row::<u16, _>(|out_y, out_row, stream| {
        let [top_row, bottom_row] = source_rows(out_y, width);
        let mut out_x = 0usize;
        while out_x + 8 <= out_width && 2 * out_x + 16 <= width {
            let x = 2 * out_x;
            // SAFETY: x + 16 <= width (the guard) in source rows up to
            // 2 out_y + 1 < height, inside `src`.
            let (top, bottom) = unsafe {
                (
                    _mm256_loadu_si256(src_ptr.add(top_row + x) as *const __m256i),
                    _mm256_loadu_si256(src_ptr.add(bottom_row + x) as *const __m256i),
                )
            };
            let top_pair_sums = _mm256_add_epi32(
                _mm256_and_si256(top, low_u16_mask),
                _mm256_srli_epi32(top, 16),
            );
            let bottom_pair_sums = _mm256_add_epi32(
                _mm256_and_si256(bottom, low_u16_mask),
                _mm256_srli_epi32(bottom, 16),
            );
            let block_sums = _mm256_add_epi32(top_pair_sums, bottom_pair_sums);
            let rounded = _mm256_srli_epi32(_mm256_add_epi32(block_sums, two), 2);
            let packed = _mm_packus_epi32(
                _mm256_castsi256_si128(rounded),
                _mm256_extracti128_si256(rounded, 1),
            );
            // SAFETY: out_x + 8 <= out_width (the guard): in the row.
            unsafe { store_128i(out_row.add(out_x) as *mut __m128i, packed, stream) };
            out_x += 8;
        }
        while out_x < out_width {
            // SAFETY: out_x < width / 2, out_y < height / 2: the output is in
            // the row, its 2x2 block in `src`.
            unsafe { *out_row.add(out_x) = box2_mean(src_ptr.add(top_row + 2 * out_x), 1, width) };
            out_x += 1;
        }
    });
    // SAFETY: `row` writes outputs 0..out_width of its row (the vector body,
    // then the scalar tail) and nothing else.
    unsafe { planar_box_into::<u16>(output, row) }
}

/// `box2x_u16_into_avx2` into a new image.
#[target_feature(enable = "avx2")]
pub(crate) fn box2x_u16_avx2(src: &[u16], width: usize, height: usize) -> (Vec<u16>, usize, usize) {
    let (out_width, out_height) = (width / 2, height / 2);
    let image = box2x_u16_into_avx2(
        src,
        width,
        height,
        BoxRows::new_image(out_height, out_width),
    );
    (image, out_width, out_height)
}

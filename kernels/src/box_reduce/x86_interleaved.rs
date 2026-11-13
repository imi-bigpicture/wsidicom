//! Interleaved (HWC) box kernels for RGB pixels.
//!
//! # Safety
//! Every kernel requires the CPU features in its `#[target_feature]` (its
//! dispatch checks them at runtime) and `src.len() >= width * height * 3`,
//! which it does not check. Loads are justified against that bound at their
//! loop guards; stores against the kernel's own output buffer.

use super::box2_mean;
use super::x86_output::BoxRows;
use crate::cpu::{unwritten_output, written_output};
use crate::sample::RGB;

type Box2xTo<T> = unsafe fn(&[T], usize, usize, BoxRows<T>) -> Vec<T>;

/// `kernel` into a new image.
///
/// # Safety
/// As `kernel`, with the CPU features it needs.
unsafe fn into_new_image<T>(
    kernel: Box2xTo<T>,
    src: &[T],
    width: usize,
    height: usize,
) -> (Vec<T>, usize, usize) {
    let (out_width, out_height) = (width / 2, height / 2);
    // SAFETY: the caller's contract; the image holds the output rows.
    let image = unsafe {
        kernel(
            src,
            width,
            height,
            BoxRows::new_image(out_height, out_width * RGB),
        )
    };
    (image, out_width, out_height)
}

/// The exact rounded 2x2 means `(sum + 2) >> 2` of u8 lanes: widened to u16
/// by unpacklo/hi (sums at most 1020), narrowed by packus. The unpack/packus
/// pair keeps the order within each 128-bit lane, so the result has the
/// loads' byte layout.
#[target_feature(enable = "avx2")]
#[inline]
fn rounded_mean4_u8(
    top: std::arch::x86_64::__m256i,
    bottom: std::arch::x86_64::__m256i,
    top_partner: std::arch::x86_64::__m256i,
    bottom_partner: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let two = _mm256_set1_epi16(2);
    let low_sums = _mm256_add_epi16(
        _mm256_add_epi16(
            _mm256_unpacklo_epi8(top, zero),
            _mm256_unpacklo_epi8(bottom, zero),
        ),
        _mm256_add_epi16(
            _mm256_unpacklo_epi8(top_partner, zero),
            _mm256_unpacklo_epi8(bottom_partner, zero),
        ),
    );
    let high_sums = _mm256_add_epi16(
        _mm256_add_epi16(
            _mm256_unpackhi_epi8(top, zero),
            _mm256_unpackhi_epi8(bottom, zero),
        ),
        _mm256_add_epi16(
            _mm256_unpackhi_epi8(top_partner, zero),
            _mm256_unpackhi_epi8(bottom_partner, zero),
        ),
    );
    _mm256_packus_epi16(
        _mm256_srli_epi16(_mm256_add_epi16(low_sums, two), 2),
        _mm256_srli_epi16(_mm256_add_epi16(high_sums, two), 2),
    )
}

/// The pshufb mask that compacts the kept (even) RGB pixels of a 32-byte
/// register of 2x2 means into the front of each 128-bit lane, up to the
/// first pixel that straddles the lanes -> (mask, output pixels per step,
/// bytes the low lane keeps).
fn rgb_compaction_mask() -> ([u8; 32], usize, usize) {
    let mut mask = [0x80u8; 32];
    let (mut low_lane_bytes, mut high_lane_bytes, mut pixels) = (0usize, 0usize, 0usize);
    loop {
        let summed_offset = 2 * pixels * RGB;
        if summed_offset + RGB > 32 {
            break;
        }
        if summed_offset + RGB <= 16 {
            for byte in 0..RGB {
                mask[low_lane_bytes + byte] = (summed_offset + byte) as u8;
            }
            low_lane_bytes += RGB;
        } else if summed_offset >= 16 {
            for byte in 0..RGB {
                mask[16 + high_lane_bytes + byte] = (summed_offset - 16 + byte) as u8;
            }
            high_lane_bytes += RGB;
        } else {
            break;
        }
        pixels += 1;
    }
    (mask, pixels, low_lane_bytes)
}

/// `box2x_u8_hwc_avx2_to` into a new image.
///
/// # Safety
/// As `box2x_u8_hwc_avx2_to`.
pub(crate) unsafe fn box2x_u8_hwc_avx2(
    src: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, usize, usize) {
    // SAFETY: the caller's contract.
    unsafe { into_new_image(box2x_u8_hwc_avx2_to, src, width, height) }
}

/// AVX2 uint8 RGB 2x2 box into `output`: `rounded_mean4_u8` of loads shifted
/// by a pixel for each sample's horizontal partner, then one pshufb compacts
/// the even pixels.
///
/// # Safety
/// The CPU supports AVX2, `src.len() >= width * height * 3`, and `output`
/// holds `height / 2` rows of `width / 2` pixels.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box2x_u8_hwc_avx2_to(
    src: &[u8],
    width: usize,
    height: usize,
    output: BoxRows<u8>,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    let (mask_bytes, step_pixels, low_lane_bytes) = rgb_compaction_mask();
    let out_width = width / 2;
    debug_assert_eq!(output.row_count(), height / 2);
    let stride = width * RGB;
    // SAFETY: `mask_bytes` is 32 bytes.
    let mask = unsafe { _mm256_loadu_si256(mask_bytes.as_ptr() as *const __m256i) };
    let (src_ptr, dst_ptr, layout) = (src.as_ptr(), output.dst(), output.layout());
    let rows = crate::cpu::row_bands(|first_out_y, end_out_y| {
        for out_y in first_out_y..end_out_y {
            let (top_row, bottom_row) = (2 * out_y * stride, (2 * out_y + 1) * stride);
            let out_row = layout.row_start(out_y);
            let mut out_x = 0usize;
            // Each step's two 16-byte stores write past its valid bytes; the
            // next step (or the scalar tail, or a new image's next row)
            // overwrites that slop, so only the row's store end bounds them.
            while out_x + step_pixels <= out_width
                && 2 * out_x * RGB + RGB + 32 <= stride
                && out_row + out_x * RGB + low_lane_bytes + 16 <= layout.store_end(out_y)
            {
                let src_offset = 2 * out_x * RGB;
                prefetch_next_row_pair(src_ptr, top_row + src_offset, stride);
                // SAFETY: shift <= RGB, so the 32 bytes end within the
                // guard's src_offset + RGB + 32 <= stride of row 2 out_y.
                let load_top = |shift: usize| unsafe {
                    _mm256_loadu_si256(src_ptr.add(top_row + src_offset + shift) as *const __m256i)
                };
                // SAFETY: as `load_top`, in row 2 out_y + 1 < height.
                let load_bottom = |shift: usize| unsafe {
                    _mm256_loadu_si256(
                        src_ptr.add(bottom_row + src_offset + shift) as *const __m256i
                    )
                };
                let means =
                    rounded_mean4_u8(load_top(0), load_bottom(0), load_top(RGB), load_bottom(RGB));
                let compacted = _mm256_shuffle_epi8(means, mask);
                let out_at = out_row + out_x * RGB;
                // SAFETY: out_at + low_lane_bytes + 16 <= the row's store end
                // (the guard), so both stores are inside the output.
                unsafe {
                    _mm_storeu_si128(
                        dst_ptr.add(out_at) as *mut __m128i,
                        _mm256_castsi256_si128(compacted),
                    );
                    _mm_storeu_si128(
                        dst_ptr.add(out_at + low_lane_bytes) as *mut __m128i,
                        _mm256_extracti128_si256(compacted, 1),
                    );
                }
                out_x += step_pixels;
            }
            while out_x < out_width {
                let top_left = top_row + 2 * out_x * RGB;
                for channel in 0..RGB {
                    // SAFETY: out_x < width / 2, out_y < height / 2: the sample
                    // is in the row, its 2x2 block in `src`.
                    unsafe {
                        *dst_ptr.add(out_row + out_x * RGB + channel) =
                            box2_mean(src_ptr.add(top_left + channel), RGB, stride);
                    }
                }
                out_x += 1;
            }
        }
    });
    // SAFETY: `rows(first_out_y, end_out_y)` writes all of those output rows
    // (each row's body and tail), its stores below each row's store end.
    unsafe { output.write(false, rows) }
}

/// Prefetches the lines that the loads at `at` will reach a row pair later
/// (rows 2 and 3 below; `at` and `stride` in samples), so a tile not in cache
/// streams four rows at once rather than two. A prefetch never faults, so any
/// `src_ptr` is sound.
#[inline(always)]
fn prefetch_next_row_pair<T>(src_ptr: *const T, at: usize, stride: usize) {
    use std::arch::x86_64::*;
    let next = src_ptr.wrapping_add(at + 2 * stride);
    // SAFETY: SSE is baseline on x86-64; a prefetch reads no memory
    // architecturally and cannot fault, whatever the address.
    unsafe {
        _mm_prefetch::<_MM_HINT_T0>(next as *const i8);
        _mm_prefetch::<_MM_HINT_T0>(next.wrapping_add(stride) as *const i8);
    }
}

/// `box2x_u16_hwc_avx2_to` into a new image.
///
/// # Safety
/// As `box2x_u16_hwc_avx2_to`.
pub(crate) unsafe fn box2x_u16_hwc_avx2(
    src: &[u16],
    width: usize,
    height: usize,
) -> (Vec<u16>, usize, usize) {
    // SAFETY: the caller's contract.
    unsafe { into_new_image(box2x_u16_hwc_avx2_to, src, width, height) }
}

/// AVX2 uint16 RGB 2x2 box into `output`, 3 output pixels per step: samples
/// widened in order to u32 (`cvtepu16_epi32`, no unpack lane scramble),
/// vertical sums plus the same shifted by a pixel, `(sum + 2) >> 2`, narrowed
/// by `packus_epi32` and one permute. The 16 sums hold the output pixels at
/// samples 0-2, 6-8 and 12-14; pixel 1 straddles the lanes, so a per-lane
/// pshufb and two overlapping 16-byte stores compact them, the next step
/// overwriting the slop as in the u8 kernel.
///
/// # Safety
/// The CPU supports AVX2, `src.len() >= width * height * 3`, and `output`
/// holds `height / 2` rows of `width / 2` pixels.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box2x_u16_hwc_avx2_to(
    src: &[u16],
    width: usize,
    height: usize,
    output: BoxRows<u16>,
) -> Vec<u16> {
    use std::arch::x86_64::*;
    const STEP_PIXELS: usize = 3;
    const LOAD_SAMPLES: usize = 8;
    /// Samples past a step's source offset its loads reach: the last starts
    /// a register and a pixel on.
    const READ_REACH: usize = LOAD_SAMPLES + RGB + LOAD_SAMPLES;
    const HIGH_STORE_BYTE: usize = 10;
    /// Output samples past a step's start its stores reach.
    const STORE_REACH: usize = (HIGH_STORE_BYTE + 16) / size_of::<u16>();
    let out_width = width / 2;
    debug_assert_eq!(output.row_count(), height / 2);
    let stride = width * RGB;
    let (src_ptr, dst_ptr, layout) = (src.as_ptr(), output.dst(), output.layout());
    let two = _mm256_set1_epi32(2);
    // Low lane: pixel 0 = bytes 0-5, pixel 1's head = bytes 12-15 (samples
    // 6, 7); high lane: pixel 1's tail = bytes 0-1 (sample 8), pixel 2 =
    // bytes 8-13.
    let compact = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, //
        0, 1, 8, 9, 10, 11, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let rows = crate::cpu::row_bands(|first_out_y, end_out_y| {
        for out_y in first_out_y..end_out_y {
            let (top_row, bottom_row) = (2 * out_y * stride, (2 * out_y + 1) * stride);
            let out_row = layout.row_start(out_y);
            let mut out_x = 0usize;
            while out_x + STEP_PIXELS <= out_width
                && 2 * out_x * RGB + READ_REACH <= stride
                && out_row + out_x * RGB + STORE_REACH <= layout.store_end(out_y)
            {
                let src_offset = 2 * out_x * RGB;
                prefetch_next_row_pair(src_ptr, top_row + src_offset, stride);
                // SAFETY: the loads end by src_offset + READ_REACH <= stride
                // (the guard) of source rows up to 2 out_y + 1 < height.
                let load_widened = |offset: usize| unsafe {
                    _mm256_cvtepu16_epi32(_mm_loadu_si128(src_ptr.add(offset) as *const __m128i))
                };
                let column_sums = |shift: usize| {
                    _mm256_add_epi32(
                        load_widened(top_row + src_offset + shift),
                        load_widened(bottom_row + src_offset + shift),
                    )
                };
                let block_sums_low = _mm256_add_epi32(column_sums(0), column_sums(RGB));
                let block_sums_high =
                    _mm256_add_epi32(column_sums(LOAD_SAMPLES), column_sums(LOAD_SAMPLES + RGB));
                let means_low = _mm256_srli_epi32(_mm256_add_epi32(block_sums_low, two), 2);
                let means_high = _mm256_srli_epi32(_mm256_add_epi32(block_sums_high, two), 2);
                // packus_epi32 interleaves lanes: [a.l0, b.l0 | a.l1, b.l1];
                // permute4x64(0xD8) restores sequential sample order.
                let ordered =
                    _mm256_permute4x64_epi64(_mm256_packus_epi32(means_low, means_high), 0xD8);
                let compacted = _mm256_shuffle_epi8(ordered, compact);
                // SAFETY: the stores end by out_row + 3 out_x + STORE_REACH <=
                // the row's store end (the guard).
                unsafe {
                    let out_byte_ptr = (dst_ptr.add(out_row + out_x * RGB)) as *mut u8;
                    _mm_storeu_si128(
                        out_byte_ptr as *mut __m128i,
                        _mm256_castsi256_si128(compacted),
                    );
                    _mm_storeu_si128(
                        out_byte_ptr.add(HIGH_STORE_BYTE) as *mut __m128i,
                        _mm256_extracti128_si256(compacted, 1),
                    );
                }
                out_x += STEP_PIXELS;
            }
            while out_x < out_width {
                let top_left = top_row + 2 * out_x * RGB;
                for channel in 0..RGB {
                    // SAFETY: out_x < width / 2, out_y < height / 2: the sample
                    // is in the row, its 2x2 block in `src`.
                    unsafe {
                        *dst_ptr.add(out_row + out_x * RGB + channel) =
                            box2_mean(src_ptr.add(top_left + channel), RGB, stride);
                    }
                }
                out_x += 1;
            }
        }
    });
    // SAFETY: `rows(first_out_y, end_out_y)` writes all of those output rows
    // (each row's body and tail), its stores below each row's store end.
    unsafe { output.write(false, rows) }
}

/// pshufb mask that groups, per 128-bit lane, each channel's 4 consecutive
/// columns of a 4-pixel RGB block, so `maddubs` then `madd` sum them: the
/// lane ends as one pixel of 4x4 sums and a zero.
fn quad_columns_mask() -> [u8; 32] {
    let mut mask = [0x80u8; 32];
    for lane in 0..2 {
        for channel in 0..RGB {
            for column in 0..4 {
                mask[16 * lane + 4 * channel + column] = (RGB * column + channel) as u8;
            }
        }
    }
    mask
}

/// AVX2 uint8 RGB box by 4 (`levels` 2) or 8 (`levels` 3) in one pass, with
/// the planar pyramid's rounding: exact 4x4 means `(sum + 8) >> 4`, and for 8x
/// the exact 2x2 mean `(sum + 2) >> 2` of those, so every channel equals that
/// channel reduced as a planar image.
///
/// Each 128-bit lane reads one 4-pixel block per row (12 of its 16 bytes),
/// groups each channel's columns with one pshufb and sums them with maddubs
/// (4 rows: at most 2040, fits i16) and madd. For 8x the two row groups'
/// rounded means add, and lane pairs add into one 8x pixel per lane. Either
/// way each lane ends as one output dword; four registers pack into 32 RGBX
/// bytes, shuffled to 24 RGB bytes.
///
/// # Safety
/// The CPU supports AVX2, `levels` is 2 or 3, and `src.len() >= width *
/// height * 3`. (The vector blocks check their reads against `src.len()`;
/// the scalar tail relies on the bound.)
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box_levels_u8_hwc_avx2(
    src: &[u8],
    (width, height): (usize, usize),
    levels: u32,
) -> (Vec<u8>, usize, usize) {
    use std::arch::x86_64::*;
    /// The bytes of a 4-pixel RGB block, of the 16 each lane loads.
    const LANE_BLOCK_BYTES: usize = 4 * RGB;
    const STEP_PIXELS: usize = 8;
    /// How far a step's two 16-byte stores (12 valid bytes each) reach.
    const STORE_REACH: usize = 12 + 16;
    debug_assert!((2..=3).contains(&levels));
    let factor = 1usize << levels;
    let (out_width, out_height) = (width / factor, height / factor);
    let (stride, out_stride) = (width * RGB, out_width * RGB);
    let mut out = unwritten_output::<u8>(out_height * out_stride);
    let out_len = out.len();
    // Source bytes per output register (8 lane dwords); a step reads 4
    // registers' worth from each row and overreads the last lane's 4 spare
    // bytes.
    let register_bytes = (factor / 2) * LANE_BLOCK_BYTES;
    let row_read_bytes = 4 * register_bytes + (16 - LANE_BLOCK_BYTES);
    // SAFETY: the mask is 32 bytes.
    let mask = unsafe { _mm256_loadu_si256(quad_columns_mask().as_ptr() as *const __m256i) };
    let (ones, ones_i16) = (_mm256_set1_epi8(1), _mm256_set1_epi16(1));
    let (two, eight) = (_mm256_set1_epi32(2), _mm256_set1_epi32(8));
    // packus leaves lane dwords as [r0.l0 r1.l0 r2.l0 r3.l0 | r0.l1 ...].
    let lane_order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let rgbx_to_rgb = _mm256_setr_epi8(
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1, //
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1,
    );
    let (src_ptr, dst_ptr) = (src.as_ptr(), out.as_mut_ptr().cast::<u8>());
    let rows = crate::cpu::row_bands(|first_out_y, end_out_y| {
        for out_y in first_out_y..end_out_y {
            // SAFETY: band_row < factor, so the row (factor * out_y +
            // band_row < height) starts inside `src`.
            let row =
                |band_row: usize| unsafe { src_ptr.add((factor * out_y + band_row) * stride) };
            // The rounded 4x4 means of the lanes' blocks at x and
            // x + LANE_BLOCK_BYTES in rows first_row..first_row + 4.
            let quads = |first_row: usize, x: usize| {
                let sums = |group_row: usize| {
                    // SAFETY: only reached from `block` at an out_x `fits`
                    // passed, which checked these reads against `src.len()`.
                    let pixels = unsafe {
                        let block_ptr = row(first_row + group_row).add(x);
                        _mm256_loadu2_m128i(
                            block_ptr.add(LANE_BLOCK_BYTES) as *const __m128i,
                            block_ptr as *const __m128i,
                        )
                    };
                    _mm256_maddubs_epi16(_mm256_shuffle_epi8(pixels, mask), ones)
                };
                let col_pairs = _mm256_add_epi16(
                    _mm256_add_epi16(sums(0), sums(1)),
                    _mm256_add_epi16(sums(2), sums(3)),
                );
                _mm256_srli_epi32(
                    _mm256_add_epi32(_mm256_madd_epi16(col_pairs, ones_i16), eight),
                    4,
                )
            };
            // One output dword per lane from the `register_bytes` source
            // bytes at x.
            let means = |x: usize| {
                if factor == 4 {
                    return quads(0, x);
                }
                // Per 4x block, the sum of its 2x2 group's quads (in the next
                // lane).
                let pairs = |x: usize| _mm256_add_epi32(quads(0, x), quads(4, x));
                let (first, second) = (pairs(x), pairs(x + register_bytes / 2));
                let sums = _mm256_add_epi32(
                    _mm256_permute2x128_si256(first, second, 0x20),
                    _mm256_permute2x128_si256(first, second, 0x31),
                );
                _mm256_srli_epi32(_mm256_add_epi32(sums, two), 2)
            };
            let out_row = out_y * out_stride;
            // Reads past the row may run into the next row; only the buffer
            // bounds them. The stores' slop is overwritten by the next block
            // or row.
            let fits = |out_x: usize| {
                (factor * out_y + factor - 1) * stride + factor * out_x * RGB + row_read_bytes
                    <= src.len()
                    && out_row + out_x * RGB + STORE_REACH <= out_len
            };
            // Only called at an out_x `fits` passed.
            let block = |out_x: usize| {
                let x = factor * out_x * RGB;
                let low = _mm256_packus_epi32(means(x), means(x + register_bytes));
                let high = _mm256_packus_epi32(
                    means(x + 2 * register_bytes),
                    means(x + 3 * register_bytes),
                );
                let rgbx = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(low, high), lane_order);
                // SAFETY: `fits(out_x)`: the stores end by out_row + out_x *
                // RGB + STORE_REACH <= out_len, inside `out`.
                unsafe {
                    let out_ptr = dst_ptr.add(out_row + out_x * RGB);
                    let rgb = _mm256_shuffle_epi8(rgbx, rgbx_to_rgb);
                    _mm_storeu_si128(out_ptr as *mut __m128i, _mm256_castsi256_si128(rgb));
                    _mm_storeu_si128(
                        out_ptr.add(12) as *mut __m128i,
                        _mm256_extracti128_si256(rgb, 1),
                    );
                }
            };
            let mut out_x = 0usize;
            while out_x + STEP_PIXELS <= out_width && fits(out_x) {
                block(out_x);
                out_x += STEP_PIXELS;
            }
            // A short remainder: one block shifted back to end at the row's
            // end (recomputing some outputs) instead of the scalar tail.
            if out_x < out_width && out_width >= STEP_PIXELS && fits(out_width - STEP_PIXELS) {
                block(out_width - STEP_PIXELS);
                out_x = out_width;
            }
            while out_x < out_width {
                for channel in 0..RGB {
                    let quad = |first_row: usize, first_column: usize| {
                        let sum: u32 = (first_row..first_row + 4)
                            .flat_map(|band_row| {
                                (first_column..first_column + 4).map(move |x| {
                                    // SAFETY: x < factor * out_width <= width
                                    // in a row of the band, inside `src`.
                                    u32::from(unsafe { *row(band_row).add(x * RGB + channel) })
                                })
                            })
                            .sum();
                        (sum + 8) >> 4
                    };
                    let x = factor * out_x;
                    let mean = if factor == 4 {
                        quad(0, x)
                    } else {
                        (quad(0, x) + quad(0, x + 4) + quad(4, x) + quad(4, x + 4) + 2) >> 2
                    };
                    // SAFETY: out_x < out_width: a sample of row out_y.
                    unsafe { *dst_ptr.add(out_row + out_x * RGB + channel) = mean as u8 };
                }
                out_x += 1;
            }
        }
    });
    // SAFETY: `out` is this function's; `rows(first_out_y, end_out_y)`
    // writes only those output rows, plus store slop inside the buffer.
    unsafe { crate::cpu::with_prefault(dst_ptr, out_len, out_stride, out_height, rows) };
    // SAFETY: every row's body and tail wrote all of it.
    (unsafe { written_output(out) }, out_width, out_height)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box_reduce::box2x_hwc_scalar;
    use crate::box_reduce::tests::{
        BoxTestSample, assert_same_image, box_sizes, interleaved_pyramid_reference, test_images,
    };
    use crate::cpu::{LARGE_OUTPUT_BYTES, PREFAULT_OUTPUT_BYTES};
    use crate::testing::{Rng, UnsafeImageKernel, random_samples};

    fn avx2_detected() -> bool {
        std::is_x86_feature_detected!("avx2")
    }

    /// Runs the 2x RGB `kernel` on random and extreme images of every box
    /// test size, against the scalar interleaved halving.
    ///
    /// # Safety
    /// The CPU supports `kernel`'s features.
    unsafe fn assert_halves_as_scalar<T: BoxTestSample>(kernel: UnsafeImageKernel<T>) {
        let mut rng = Rng::new(20);
        for (width, height) in box_sizes(1) {
            for src in test_images::<T>(&mut rng, width * height * RGB) {
                // SAFETY: the CPU features (the caller's contract); src holds
                // width * height pixels.
                let simd = unsafe { kernel(&src, width, height) };
                assert_same_image(
                    &simd,
                    &box2x_hwc_scalar(&src, width, height),
                    &format!("{width} x {height}"),
                );
            }
        }
    }

    #[test]
    fn box2x_avx2_equals_scalar() {
        if !avx2_detected() {
            return;
        }
        // SAFETY: AVX2 is available.
        unsafe {
            assert_halves_as_scalar::<u8>(box2x_u8_hwc_avx2);
            assert_halves_as_scalar::<u16>(box2x_u16_hwc_avx2);
        }
    }

    #[test]
    fn fused_u8_levels_avx2_equal_the_planar_pyramid() {
        if !avx2_detected() {
            return;
        }
        let mut rng = Rng::new(22);
        for levels in [2, 3] {
            for (width, height) in box_sizes(1) {
                for src in test_images::<u8>(&mut rng, width * height * RGB) {
                    let expected =
                        interleaved_pyramid_reference(&src, (width, height, RGB), levels);
                    // SAFETY: AVX2 is available; levels is 2 or 3; src holds
                    // the image.
                    let simd = unsafe { box_levels_u8_hwc_avx2(&src, (width, height), levels) };
                    let context = format!("{levels} levels, {width} x {height}");
                    assert_same_image(&simd, &expected, &context);
                }
            }
        }
    }

    #[test]
    fn fused_u8_levels_avx2_ignore_samples_past_the_image() {
        // The vector blocks bound their reads by the slice's length, so a
        // longer slice takes more of them; the output must not change.
        if !avx2_detected() {
            return;
        }
        let mut rng = Rng::new(23);
        for levels in [2, 3] {
            for (width, height) in box_sizes(8) {
                let len = width * height * RGB;
                let mut src: Vec<u8> = random_samples(&mut rng, len + 64 * RGB);
                // SAFETY: AVX2 is available; levels is 2 or 3; both slices
                // hold the image.
                let (exact, padded) = unsafe {
                    (
                        box_levels_u8_hwc_avx2(&src[..len], (width, height), levels),
                        box_levels_u8_hwc_avx2(&src, (width, height), levels),
                    )
                };
                src.truncate(len);
                let context = format!("{levels} levels, {width} x {height}");
                assert_same_image(&padded, &exact, &context);
                assert_same_image(
                    &exact,
                    &interleaved_pyramid_reference(&src, (width, height, RGB), levels),
                    &context,
                );
            }
        }
    }

    /// The 2x RGB `kernel` on an image whose output passes the
    /// streaming-store and prefault thresholds.
    ///
    /// # Safety
    /// The CPU supports `kernel`'s features.
    unsafe fn assert_large_halves_as_scalar<T: BoxTestSample>(kernel: UnsafeImageKernel<T>) {
        let out_bytes = PREFAULT_OUTPUT_BYTES.max(LARGE_OUTPUT_BYTES);
        let out_width = 1027;
        let out_height = out_bytes.div_ceil(out_width * RGB * size_of::<T>());
        let (width, height) = (2 * out_width + 1, 2 * out_height);
        let src: Vec<T> = random_samples(&mut Rng::new(24), width * height * RGB);
        // SAFETY: the CPU features (the caller's contract); src holds width *
        // height pixels.
        let simd = unsafe { kernel(&src, width, height) };
        assert_same_image(
            &simd,
            &box2x_hwc_scalar(&src, width, height),
            &format!("{width} x {height}"),
        );
    }

    #[test]
    fn large_outputs_stream_past_the_cache_and_prefault() {
        if !avx2_detected() {
            return;
        }
        // SAFETY: AVX2 is available.
        unsafe {
            assert_large_halves_as_scalar::<u8>(box2x_u8_hwc_avx2);
            assert_large_halves_as_scalar::<u16>(box2x_u16_hwc_avx2);
        }
    }
}

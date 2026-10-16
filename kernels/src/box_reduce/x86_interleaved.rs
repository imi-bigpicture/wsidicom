//! Interleaved (HWC) box kernels for RGB and 4-sample pixels, including
//! RGB -> RGBX.
//!
//! # Safety
//! Every kernel requires the CPU features in its `#[target_feature]` (its
//! dispatch checks them at runtime) and `src.len() >= width * height *
//! channels`, which it does not check. Loads are justified against that bound
//! at their loop guards; stores against the kernel's own output buffer.

use super::box2_mean;
use super::x86_output::BoxRows;
use crate::cpu::{unwritten_output, written_output};
use crate::sample::{RGB, RGBX};
use crate::x86::*;

/// An interleaved 2x box kernel writing into `BoxRows`.
type Box2xTo<T> = unsafe fn(&[T], usize, usize, BoxRows<T>) -> Vec<T>;

/// `kernel` (pixels of `channels` samples) into a new image.
///
/// # Safety
/// As `kernel`, with the CPU features it needs.
unsafe fn new_image<T>(
    kernel: Box2xTo<T>,
    src: &[T],
    (width, height, channels): (usize, usize, usize),
) -> (Vec<T>, usize, usize) {
    let (ow, oh) = (width / 2, height / 2);
    // SAFETY: the caller's contract; the image holds the output rows.
    let image = unsafe { kernel(src, width, height, BoxRows::new_image(oh, ow * channels)) };
    (image, ow, oh)
}

/// Exact per-element 2x2 box round for u8 lanes: (x + y + xc + yc + 2) >> 2,
/// bit-identical to the scalar path and Pillow reduce(2). Widen to u16 via
/// unpacklo/hi, sum (max 1020, fits), round, narrow via packus — the
/// unpack/packus pair is order-preserving per 128-bit lane, so the result has
/// the same byte layout as the loads.
#[target_feature(enable = "avx2")]
#[inline]
fn box_sum4_u8(
    x: std::arch::x86_64::__m256i,
    y: std::arch::x86_64::__m256i,
    xc: std::arch::x86_64::__m256i,
    yc: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let zero = _mm256_setzero_si256();
    let two = _mm256_set1_epi16(2);
    let lo = _mm256_add_epi16(
        _mm256_add_epi16(_mm256_unpacklo_epi8(x, zero), _mm256_unpacklo_epi8(y, zero)),
        _mm256_add_epi16(
            _mm256_unpacklo_epi8(xc, zero),
            _mm256_unpacklo_epi8(yc, zero),
        ),
    );
    let hi = _mm256_add_epi16(
        _mm256_add_epi16(_mm256_unpackhi_epi8(x, zero), _mm256_unpackhi_epi8(y, zero)),
        _mm256_add_epi16(
            _mm256_unpackhi_epi8(xc, zero),
            _mm256_unpackhi_epi8(yc, zero),
        ),
    );
    _mm256_packus_epi16(
        _mm256_srli_epi16(_mm256_add_epi16(lo, two), 2),
        _mm256_srli_epi16(_mm256_add_epi16(hi, two), 2),
    )
}

/// pshufb mask that compacts the kept (even) u8 output pixels (`channels` bytes each)
/// from a 32-byte horizontally-summed register into contiguous lo/hi 128-bit
/// lane groups. Returns (mask, output px/iter, lo bytes, hi bytes). A pixel
/// that would straddle the 128-bit lane boundary ends the run.
fn build_hwc_mask(channels: usize) -> ([u8; 32], usize, usize, usize) {
    let mut mask = [0x80u8; 32];
    let (mut lo, mut hi, mut npix, mut pixel) = (0usize, 0usize, 0usize, 0usize);
    loop {
        let off = 2 * pixel * channels; // byte offset of kept `pixel` in the summed register
        if off + channels > 32 {
            break;
        }
        if off + channels <= 16 {
            for byte in 0..channels {
                mask[lo + byte] = (off + byte) as u8;
            }
            lo += channels;
        } else if off >= 16 {
            for byte in 0..channels {
                mask[16 + hi + byte] = (off - 16 + byte) as u8;
            }
            hi += channels;
        } else {
            break; // straddles the lane boundary
        }
        npix += 1;
        pixel += 1;
    }
    (mask, npix, lo, hi)
}

/// `box2x_u8_hwc3_avx2_to` into a new image.
///
/// # Safety
/// As `box2x_u8_hwc3_avx2_to`.
pub(crate) unsafe fn box2x_u8_hwc3_avx2(
    src: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, usize, usize) {
    // SAFETY: the caller's contract.
    unsafe { new_image(box2x_u8_hwc3_avx2_to, src, (width, height, RGB)) }
}

/// AVX2 uint8 RGB 2x2 box into `output`: exact (sum+2)>>2 via
/// `box_sum4_u8` on 3-shifted loads + one 256-bit `pshufb` to compact the
/// even pixels. Bit-identical to the scalar path and Pillow reduce(2).
///
/// # Safety
/// The CPU supports AVX2, `src.len() >= width * height * 3`, and `output`
/// holds `height / 2` rows of `width / 2` pixels.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box2x_u8_hwc3_avx2_to(
    src: &[u8],
    width: usize,
    height: usize,
    output: BoxRows<u8>,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    let channels = RGB;
    let (mask_arr, npix, lo, _hi) = build_hwc_mask(channels);
    let ow = width / 2;
    debug_assert_eq!(output.rows(), height / 2);
    let stride = width * channels;
    // SAFETY: `mask_arr` is 32 bytes.
    let mask = unsafe { _mm256_loadu_si256(mask_arr.as_ptr() as *const __m256i) };
    let (sp, dp, layout) = (src.as_ptr(), output.dst(), output.layout());
    let rows = crate::cpu::row_bands(|oy_lo, oy_hi| {
        for oy in oy_lo..oy_hi {
            let (r0, r1) = (2 * oy * stride, (2 * oy + 1) * stride);
            let orow = layout.row_start(oy);
            let mut ox = 0usize;
            // Overlapping stores: each 16-byte store writes `lo`/`hi` valid bytes plus
            // slop, and the next iteration's store begins exactly at oo+lo+hi, so the
            // slop is always overwritten (by the next iteration, the scalar tail, or
            // a new image's next row). Only the end the row's stores may reach
            // needs a guard -- hence `+ lo + 16`.
            while ox + npix <= ow
                && 2 * ox * channels + channels + 32 <= stride
                && orow + ox * channels + lo + 16 <= layout.store_end(oy)
            {
                let src_offset = 2 * ox * channels;
                prefetch_next_row_pair(sp, r0 + src_offset, stride);
                // SAFETY: offset <= channels, so the 32 bytes stay within
                // the guard's src_offset + channels + 32 <= stride of source
                // rows 2oy, 2oy + 1 < height, inside `src`.
                let ld = |offset: usize| unsafe {
                    _mm256_loadu_si256(sp.add(r0 + src_offset + offset) as *const __m256i)
                };
                // SAFETY: as `ld`, in source row 2oy + 1.
                let ld1 = |offset: usize| unsafe {
                    _mm256_loadu_si256(sp.add(r1 + src_offset + offset) as *const __m256i)
                };
                // The channels-shifted loads supply each element's horizontal partner.
                let hh = box_sum4_u8(ld(0), ld1(0), ld(channels), ld1(channels));
                let compacted = _mm256_shuffle_epi8(hh, mask);
                let oo = orow + ox * channels;
                // SAFETY: oo + lo + 16 <= the row's store end (the guard), so
                // both stores are inside the output.
                unsafe {
                    _mm_storeu_si128(
                        dp.add(oo) as *mut __m128i,
                        _mm256_castsi256_si128(compacted),
                    );
                    _mm_storeu_si128(
                        dp.add(oo + lo) as *mut __m128i,
                        _mm256_extracti128_si256(compacted, 1),
                    );
                }
                ox += npix;
            }
            while ox < ow {
                let top_left = r0 + 2 * ox * channels;
                for ch in 0..channels {
                    // SAFETY: ox < ow: the output sample is in row oy, and its
                    // 2x2 block (pixels 2ox, 2ox + 1 < width of rows 2oy,
                    // 2oy + 1 < height) in `src`.
                    unsafe {
                        *dp.add(orow + ox * channels + ch) =
                            box2_mean(sp.add(top_left + ch), channels, stride);
                    }
                }
                ox += 1;
            }
        }
    });
    // SAFETY: `rows(oy_lo, oy_hi)` writes all of output rows [oy_lo, oy_hi)
    // (each row's body and tail), its stores below each row's store end.
    unsafe { output.write(false, rows) }
}

/// Prefetches the lines of the next row pair (rows 2 and 3 after `at`'s
/// row; `at` and `stride` in samples) that the loads at `at` will reach a row
/// pair later, so a tile not in cache streams four rows at once rather than
/// two. Past the image the prefetch is a no-op (it never faults), so any `sp`
/// is sound.
#[inline(always)]
fn prefetch_next_row_pair<T>(sp: *const T, at: usize, stride: usize) {
    use std::arch::x86_64::*;
    let next = sp.wrapping_add(at + 2 * stride);
    // SAFETY: SSE is baseline on x86-64; a prefetch reads no memory
    // architecturally and cannot fault, whatever the address.
    unsafe {
        _mm_prefetch::<_MM_HINT_T0>(next as *const i8);
        _mm_prefetch::<_MM_HINT_T0>(next.wrapping_add(stride) as *const i8);
    }
}

/// AVX2 uint8 HWC 2x2 box, RGB (3 samples) in -> RGBX (4 samples) out, fused.
///
/// Same exact `box_sum4_u8` as the 3-channel kernel; only the shuffle and store
/// differ. `hh[k]` holds the horizontally-averaged value of source element
/// `src_offset+k`, so output pixel `m` of this iteration lives at
/// `hh[6m .. 6m+3]`. Those land in lane 0 for m=0..2 and lane 1 for m=3..4, giving 5 output pixels per
/// iteration -- picked straight into 4-byte slots with the padding sample OR'd
/// in, so the expand costs nothing beyond the store it already had to do.
///
/// # Safety
/// The CPU supports AVX2, and `src.len() >= width * height * 3`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box2x_u8_hwc3to4_avx2(
    src: &[u8],
    width: usize,
    height: usize,
    pad: u8,
) -> (Vec<u8>, usize, usize) {
    use std::arch::x86_64::*;
    const IN_CHANNELS: usize = 3;
    const OUT_CHANNELS: usize = 4;
    let (ow, oh) = (width / 2, height / 2);
    let (stride, ostride) = (width * IN_CHANNELS, ow * OUT_CHANNELS);
    let mut out = unwritten_output::<u8>(oh * ostride);
    let olen = oh * ostride;
    let (sp, dp) = (src.as_ptr(), out.as_mut_ptr().cast::<u8>());
    // lane0: m=0,1,2 at hh 0..2, 6..8, 12..14; lane1: m=3,4 at local 2..4, 8..10
    let mask = _mm256_setr_epi8(
        0, 1, 2, -1, 6, 7, 8, -1, 12, 13, 14, -1, -1, -1, -1, -1, //
        2, 3, 4, -1, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let padv = rgbx_pad_lanes(pad);
    let rows = crate::cpu::row_bands(|oy_lo, oy_hi| {
        for oy in oy_lo..oy_hi {
            let (r0, r1) = (2 * oy * stride, (2 * oy + 1) * stride);
            let orow = oy * ostride;
            let mut ox = 0usize;
            // Lane 0 writes 12 valid bytes, lane 1 the next 8; the next iteration
            // starts exactly at +20, so the store slop is overwritten.
            while ox + 5 <= ow
                && 2 * ox * IN_CHANNELS + IN_CHANNELS + 32 <= stride
                && orow + ox * OUT_CHANNELS + 28 <= olen
            {
                let src_offset = 2 * ox * IN_CHANNELS;
                // SAFETY: offset <= IN_CHANNELS, so the 32 bytes stay
                // within the guard's src_offset + IN_CHANNELS + 32 <= stride of
                // source rows 2oy, 2oy + 1 < height, inside `src`.
                let ld = |offset: usize| unsafe {
                    _mm256_loadu_si256(sp.add(r0 + src_offset + offset) as *const __m256i)
                };
                // SAFETY: as `ld`, in source row 2oy + 1.
                let ld1 = |offset: usize| unsafe {
                    _mm256_loadu_si256(sp.add(r1 + src_offset + offset) as *const __m256i)
                };
                let hh = box_sum4_u8(ld(0), ld1(0), ld(IN_CHANNELS), ld1(IN_CHANNELS));
                let compacted = _mm256_or_si256(_mm256_shuffle_epi8(hh, mask), padv);
                let out_at = orow + ox * OUT_CHANNELS;
                // SAFETY: out_at + 28 <= olen (the guard), so both stores are
                // inside `out`.
                unsafe {
                    _mm_storeu_si128(
                        dp.add(out_at) as *mut __m128i,
                        _mm256_castsi256_si128(compacted),
                    );
                    _mm_storeu_si128(
                        dp.add(out_at + 12) as *mut __m128i,
                        _mm256_extracti128_si256(compacted, 1),
                    );
                }
                ox += 5;
            }
            while ox < ow {
                let top_left = r0 + 2 * ox * IN_CHANNELS;
                // SAFETY: ox < ow: the output pixel is in row oy, and its 2x2
                // block (pixels 2ox, 2ox + 1 < width of rows 2oy, 2oy + 1 <
                // height) in `src`.
                unsafe {
                    for ch in 0..IN_CHANNELS {
                        *dp.add(orow + ox * OUT_CHANNELS + ch) =
                            box2_mean(sp.add(top_left + ch), IN_CHANNELS, stride);
                    }
                    *dp.add(orow + ox * OUT_CHANNELS + 3) = pad;
                }
                ox += 1;
            }
        }
    });
    // SAFETY: `out` is this function's; `rows(oy_lo, oy_hi)` writes only
    // output rows [oy_lo, oy_hi), plus store slop inside the buffer.
    unsafe { crate::cpu::with_prefault(dp, olen, ostride, oh, rows) };
    // SAFETY: every row's body and tail wrote all of it.
    (unsafe { written_output(out) }, ow, oh)
}

/// AVX2 uint8 HWC 2x2 box, 4 channels (RGBX): one pixel = 4 bytes = one dword, so a
/// single `vpermd` compacts the four horizontally-averaged even pixels into the
/// low lane -> one exact 16-byte store, no overlap slop. 4 output px/iter.
/// Exact (sum+2)>>2, bit-identical to scalar and Pillow reduce(2).
///
/// # Safety
/// As `box2x_u8_hwc4_avx2_to`.
pub(crate) unsafe fn box2x_u8_hwc4_avx2(
    src: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, usize, usize) {
    // SAFETY: the caller's contract.
    unsafe { new_image(box2x_u8_hwc4_avx2_to, src, (width, height, RGBX)) }
}

/// `box2x_u8_hwc4_avx2` into `output`.
///
/// # Safety
/// The CPU supports AVX2, `src.len() >= width * height * 4`, and `output`
/// holds `height / 2` rows of `width / 2` pixels.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box2x_u8_hwc4_avx2_to(
    src: &[u8],
    width: usize,
    height: usize,
    output: BoxRows<u8>,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    const CHANNELS: usize = 4;
    let ow = width / 2;
    debug_assert_eq!(output.rows(), height / 2);
    let stride = width * CHANNELS;
    // Stores are exact (no overlap slop), so they can stream: an overlapping
    // stream would write each line twice.
    let nt = output.streams();
    let (sp, dp, layout) = (src.as_ptr(), output.dst(), output.layout());
    // dwords 0,2,4,6 are the even (horizontally-averaged) output pixels.
    let perm = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
    let rows = crate::cpu::row_bands(|oy_lo, oy_hi| {
        for oy in oy_lo..oy_hi {
            let (r0, r1) = (2 * oy * stride, (2 * oy + 1) * stride);
            let orow = layout.row_start(oy);
            let mut ox = 0usize;
            // ox+4<=ow bounds the store exactly: ox*CHANNELS+16 <= ow*CHANNELS = ostride.
            while ox + 4 <= ow && 2 * ox * CHANNELS + CHANNELS + 32 <= stride {
                let src_offset = 2 * ox * CHANNELS;
                prefetch_next_row_pair(sp, r0 + src_offset, stride);
                // SAFETY: offset <= CHANNELS, so the 32 bytes stay within
                // the guard's src_offset + CHANNELS + 32 <= stride of source
                // rows 2oy, 2oy + 1 < height, inside `src`.
                let ld = |offset: usize| unsafe {
                    _mm256_loadu_si256(sp.add(r0 + src_offset + offset) as *const __m256i)
                };
                // SAFETY: as `ld`, in source row 2oy + 1.
                let ld1 = |offset: usize| unsafe {
                    _mm256_loadu_si256(sp.add(r1 + src_offset + offset) as *const __m256i)
                };
                let hh = box_sum4_u8(ld(0), ld1(0), ld(CHANNELS), ld1(CHANNELS));
                let compacted = _mm256_permutevar8x32_epi32(hh, perm);
                // SAFETY: ox + 4 <= ow: the 16 bytes are inside output row oy.
                unsafe {
                    store_128i(
                        dp.add(orow + ox * CHANNELS) as *mut __m128i,
                        _mm256_castsi256_si128(compacted),
                        nt,
                    );
                }
                ox += 4;
            }
            while ox < ow {
                let top_left = r0 + 2 * ox * CHANNELS;
                for ch in 0..CHANNELS {
                    // SAFETY: ox < ow: the output sample is in row oy, and its
                    // 2x2 block (pixels 2ox, 2ox + 1 < width of rows 2oy,
                    // 2oy + 1 < height) in `src`.
                    unsafe {
                        *dp.add(orow + ox * CHANNELS + ch) =
                            box2_mean(sp.add(top_left + ch), CHANNELS, stride);
                    }
                }
                ox += 1;
            }
        }
    });
    // SAFETY: `rows(oy_lo, oy_hi)` writes all of output rows [oy_lo, oy_hi)
    // (each row's body and tail) and nothing else.
    unsafe { output.write(nt, rows) }
}

/// AVX2 uint16 HWC 2x2 box, 4 channels: exact (sum+2)>>2 (bit-identical to
/// scalar/NEON/cv2-AREA), 2 output px/iter from a single load per row:
/// vertical sums widen to u32, the horizontal partner (+CHANNELS elements) comes from
/// a cross-lane 128-bit shift, and `packus_epi32`'s lane interleave emits
/// [px0|px1] directly in the low 128.
///
/// # Safety
/// As `box2x_u16_hwc4_avx2_to`.
pub(crate) unsafe fn box2x_u16_hwc4_avx2(
    src: &[u16],
    width: usize,
    height: usize,
) -> (Vec<u16>, usize, usize) {
    // SAFETY: the caller's contract.
    unsafe { new_image(box2x_u16_hwc4_avx2_to, src, (width, height, RGBX)) }
}

/// `box2x_u16_hwc4_avx2` into `output`.
///
/// # Safety
/// The CPU supports AVX2, `src.len() >= width * height * 4`, and `output`
/// holds `height / 2` rows of `width / 2` pixels.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box2x_u16_hwc4_avx2_to(
    src: &[u16],
    width: usize,
    height: usize,
    output: BoxRows<u16>,
) -> Vec<u16> {
    use std::arch::x86_64::*;
    const CHANNELS: usize = 4;
    let ow = width / 2;
    debug_assert_eq!(output.rows(), height / 2);
    let stride = width * CHANNELS;
    let (sp, dp, layout) = (src.as_ptr(), output.dst(), output.layout());
    let two = _mm256_set1_epi32(2);
    let nt = output.streams();
    let rows = crate::cpu::row_bands(|oy_lo, oy_hi| {
        for oy in oy_lo..oy_hi {
            let (r0, r1) = (2 * oy * stride, (2 * oy + 1) * stride);
            let orow = layout.row_start(oy);
            let mut ox = 0usize;
            // Reads are exactly [src_offset, src_offset+16) per row: no shifted load to over-read.
            while ox + 2 <= ow && 2 * ox + 4 <= width {
                let src_offset = 2 * ox * CHANNELS;
                prefetch_next_row_pair(sp, r0 + src_offset, stride);
                // SAFETY: src_offset + 16 <= stride (2ox + 4 <= width) in source
                // rows 2oy, 2oy + 1 < height, inside `src`.
                let (top, bottom) = unsafe {
                    (
                        _mm256_loadu_si256(sp.add(r0 + src_offset) as *const __m256i),
                        _mm256_loadu_si256(sp.add(r1 + src_offset) as *const __m256i),
                    )
                };
                // vertical sums as u32: v_lo = elements 0..8, v_hi = 8..16
                let v_lo = _mm256_add_epi32(
                    _mm256_cvtepu16_epi32(_mm256_castsi256_si128(top)),
                    _mm256_cvtepu16_epi32(_mm256_castsi256_si128(bottom)),
                );
                let v_hi = _mm256_add_epi32(
                    _mm256_cvtepu16_epi32(_mm256_extracti128_si256(top, 1)),
                    _mm256_cvtepu16_epi32(_mm256_extracti128_si256(bottom, 1)),
                );
                // shift by 4 u32 (= CHANNELS elements): partners of v_lo are elements 4..12
                let s_lo = _mm256_permute2x128_si256(v_lo, v_hi, 0x21);
                let s_hi = _mm256_permute2x128_si256(v_hi, v_hi, 0x81); // upper half, then zero
                let r_lo =
                    _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(v_lo, s_lo), two), 2);
                let r_hi =
                    _mm256_srli_epi32(_mm256_add_epi32(_mm256_add_epi32(v_hi, s_hi), two), 2);
                // packus lane interleave: low 128 = [r_lo.l0 (px0), r_hi.l0 (px1)]
                let packed = _mm256_packus_epi32(r_lo, r_hi);
                let pixels = _mm256_castsi256_si128(packed);
                // SAFETY: ox + 2 <= ow: the 8 samples are inside output row oy.
                unsafe { store_128i(dp.add(orow + ox * CHANNELS) as *mut __m128i, pixels, nt) };
                ox += 2;
            }
            while ox < ow {
                let top_left = r0 + 2 * ox * CHANNELS;
                for ch in 0..CHANNELS {
                    // SAFETY: ox < ow: the output sample is in row oy, and its
                    // 2x2 block (pixels 2ox, 2ox + 1 < width of rows 2oy,
                    // 2oy + 1 < height) in `src`.
                    unsafe {
                        *dp.add(orow + ox * CHANNELS + ch) =
                            box2_mean(sp.add(top_left + ch), CHANNELS, stride);
                    }
                }
                ox += 1;
            }
        }
    });
    // SAFETY: `rows(oy_lo, oy_hi)` writes all of output rows [oy_lo, oy_hi)
    // (each row's body and tail) and nothing else.
    unsafe { output.write(nt, rows) }
}

/// AVX2 uint16 HWC 2x2 box, 3 channels: exact (sum+2)>>2, 3 output px per iteration.
/// Sequential widening via `cvtepu16_epi32` (no unpack lane-scramble), vertical
/// and CHANNELS-shifted horizontal sums in u32, narrow via `packus_epi32` and one
/// permute4x64 to restore element order. The 16 h-values cover output pixels
/// at element offsets 0-2 / 6-8 / 12-14; pixel 1 straddles the lane boundary,
/// so compaction is a per-lane pshufb + two overlapping 16-byte stores (10 +
/// 8 valid bytes; the next write starts 18 bytes on and overwrites the slop —
/// same contract as the u8 generic kernel, guarded at the end a row's
/// stores may reach).
///
/// # Safety
/// As `box2x_u16_hwc3_avx2_to`.
pub(crate) unsafe fn box2x_u16_hwc3_avx2(
    src: &[u16],
    width: usize,
    height: usize,
) -> (Vec<u16>, usize, usize) {
    // SAFETY: the caller's contract.
    unsafe { new_image(box2x_u16_hwc3_avx2_to, src, (width, height, RGB)) }
}

/// `box2x_u16_hwc3_avx2` into `output`.
///
/// # Safety
/// The CPU supports AVX2, `src.len() >= width * height * 3`, and `output`
/// holds `height / 2` rows of `width / 2` pixels.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box2x_u16_hwc3_avx2_to(
    src: &[u16],
    width: usize,
    height: usize,
    output: BoxRows<u16>,
) -> Vec<u16> {
    use std::arch::x86_64::*;
    const CHANNELS: usize = 3;
    let ow = width / 2;
    debug_assert_eq!(output.rows(), height / 2);
    let stride = width * CHANNELS;
    let (sp, dp, layout) = (src.as_ptr(), output.dst(), output.layout());
    let two = _mm256_set1_epi32(2);
    // lane0: px0 = bytes 0-5, px1 head = bytes 12-15 (elements 6,7);
    // lane1: px1 tail = local bytes 0-1 (element 8), px2 = local bytes 8-13.
    let compact = _mm256_setr_epi8(
        0, 1, 2, 3, 4, 5, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, //
        0, 1, 8, 9, 10, 11, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let rows = crate::cpu::row_bands(|oy_lo, oy_hi| {
        for oy in oy_lo..oy_hi {
            let (r0, r1) = (2 * oy * stride, (2 * oy + 1) * stride);
            let orow = layout.row_start(oy);
            let mut ox = 0usize;
            // Reads reach element src_offset+19; the second store ends at output element
            // orow + ox*CHANNELS + 13.
            while ox + 3 <= ow
                && 6 * ox + 19 <= stride
                && orow + ox * CHANNELS + 13 <= layout.store_end(oy)
            {
                let src_offset = 2 * ox * CHANNELS;
                prefetch_next_row_pair(sp, r0 + src_offset, stride);
                // SAFETY: every call reads 8 samples from offset (row r0 or r1)
                // + src_offset + at most 8 + CHANNELS, ending by src_offset + 19
                // <= stride (the guard) in source rows 2oy, 2oy + 1 < height,
                // inside `src`.
                let cv = |offset: usize| unsafe {
                    _mm256_cvtepu16_epi32(_mm_loadu_si128(sp.add(offset) as *const __m128i))
                };
                // vertical sums, elements src_offset..src_offset+8 and src_offset+8..src_offset+16
                let v0 = _mm256_add_epi32(cv(r0 + src_offset), cv(r1 + src_offset));
                let v1 = _mm256_add_epi32(cv(r0 + src_offset + 8), cv(r1 + src_offset + 8));
                // CHANNELS-shifted vertical sums -> horizontal pair completion
                let h0 = _mm256_add_epi32(
                    v0,
                    _mm256_add_epi32(
                        cv(r0 + src_offset + CHANNELS),
                        cv(r1 + src_offset + CHANNELS),
                    ),
                );
                let h1 = _mm256_add_epi32(
                    v1,
                    _mm256_add_epi32(
                        cv(r0 + src_offset + 8 + CHANNELS),
                        cv(r1 + src_offset + 8 + CHANNELS),
                    ),
                );
                let rounded0 = _mm256_srli_epi32(_mm256_add_epi32(h0, two), 2);
                let rounded1 = _mm256_srli_epi32(_mm256_add_epi32(h1, two), 2);
                // packus_epi32 interleaves lanes: [a.l0, b.l0 | a.l1, b.l1] ->
                // permute4x64(0xD8) restores sequential element order.
                let ordered =
                    _mm256_permute4x64_epi64(_mm256_packus_epi32(rounded0, rounded1), 0xD8);
                let compacted = _mm256_shuffle_epi8(ordered, compact);
                // SAFETY: the stores cover output samples orow + 3ox ..
                // orow + 3ox + 13 <= the row's store end (the guard).
                unsafe {
                    let ob = (dp.add(orow + ox * CHANNELS)) as *mut u8;
                    _mm_storeu_si128(ob as *mut __m128i, _mm256_castsi256_si128(compacted));
                    _mm_storeu_si128(
                        ob.add(10) as *mut __m128i,
                        _mm256_extracti128_si256(compacted, 1),
                    );
                }
                ox += 3;
            }
            while ox < ow {
                let top_left = r0 + 2 * ox * CHANNELS;
                for ch in 0..CHANNELS {
                    // SAFETY: ox < ow: the output sample is in row oy, and its
                    // 2x2 block (pixels 2ox, 2ox + 1 < width of rows 2oy,
                    // 2oy + 1 < height) in `src`.
                    unsafe {
                        *dp.add(orow + ox * CHANNELS + ch) =
                            box2_mean(sp.add(top_left + ch), CHANNELS, stride);
                    }
                }
                ox += 1;
            }
        }
    });
    // SAFETY: `rows(oy_lo, oy_hi)` writes all of output rows [oy_lo, oy_hi)
    // (each row's body and tail), its stores below each row's store end.
    unsafe { output.write(false, rows) }
}

/// AVX2 float32 HWC 2x2 box, 4 channels: one pixel = one 128-bit lane, so no gather
/// and no mask games — vertical add, then vperm2f128 pairs the lanes of two
/// 256-bit registers into (px0|px2) and (px1|px3); one add + mul emits two
/// complete output pixels per 256-bit store.
///
/// # Safety
/// As `box2x_f32_hwc4_avx2_to`.
pub(crate) unsafe fn box2x_f32_hwc4_avx2(
    src: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, usize, usize) {
    // SAFETY: the caller's contract.
    unsafe { new_image(box2x_f32_hwc4_avx2_to, src, (width, height, RGBX)) }
}

/// `box2x_f32_hwc4_avx2` into `output`.
///
/// # Safety
/// The CPU supports AVX2, `src.len() >= width * height * 4`, and `output`
/// holds `height / 2` rows of `width / 2` pixels.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box2x_f32_hwc4_avx2_to(
    src: &[f32],
    width: usize,
    height: usize,
    output: BoxRows<f32>,
) -> Vec<f32> {
    use std::arch::x86_64::*;
    const CHANNELS: usize = 4;
    let ow = width / 2;
    debug_assert_eq!(output.rows(), height / 2);
    let stride = width * CHANNELS;
    let quarter = _mm256_set1_ps(0.25);
    let (sp, dp, layout) = (src.as_ptr(), output.dst(), output.layout());
    let nt = output.streams();
    let rows = crate::cpu::row_bands(|oy_lo, oy_hi| {
        for oy in oy_lo..oy_hi {
            let (r0, r1) = (2 * oy * stride, (2 * oy + 1) * stride);
            let orow = layout.row_start(oy);
            let mut ox = 0usize;
            while ox + 2 <= ow && 2 * ox + 4 <= width {
                let src_offset = 2 * ox * CHANNELS;
                prefetch_next_row_pair(sp, r0 + src_offset, stride);
                // SAFETY: src_offset + 16 <= stride (2ox + 4 <= width) in source
                // rows 2oy, 2oy + 1 < height, inside `src`.
                let (pixels_0_1, pixels_2_3) = unsafe {
                    (
                        _mm256_add_ps(
                            _mm256_loadu_ps(sp.add(r0 + src_offset)),
                            _mm256_loadu_ps(sp.add(r1 + src_offset)),
                        ),
                        _mm256_add_ps(
                            _mm256_loadu_ps(sp.add(r0 + src_offset + 8)),
                            _mm256_loadu_ps(sp.add(r1 + src_offset + 8)),
                        ),
                    )
                };
                // lanes: pixels_0_1 = [px0|px1], pixels_2_3 = [px2|px3] -> lo = [px0|px2], hi = [px1|px3]
                let lo = _mm256_permute2f128_ps(pixels_0_1, pixels_2_3, 0x20);
                let hi = _mm256_permute2f128_ps(pixels_0_1, pixels_2_3, 0x31);
                let means = _mm256_mul_ps(_mm256_add_ps(lo, hi), quarter);
                // SAFETY: ox + 2 <= ow: the 8 floats are inside output row oy.
                unsafe { store_256_ps(dp.add(orow + ox * CHANNELS), means, nt) };
                ox += 2;
            }
            while ox < ow {
                let top_left = r0 + 2 * ox * CHANNELS;
                for ch in 0..CHANNELS {
                    // SAFETY: ox < ow: the output sample is in row oy, and its
                    // 2x2 block (pixels 2ox, 2ox + 1 < width of rows 2oy,
                    // 2oy + 1 < height) in `src`.
                    unsafe {
                        *dp.add(orow + ox * CHANNELS + ch) =
                            box2_mean(sp.add(top_left + ch), CHANNELS, stride);
                    }
                }
                ox += 1;
            }
        }
    });
    // SAFETY: `rows(oy_lo, oy_hi)` writes all of output rows [oy_lo, oy_hi)
    // (each row's body and tail) and nothing else.
    unsafe { output.write(nt, rows) }
}

/// AVX2 float32 HWC 2x2 box, 3 channels: the horizontal pair-add is two loads offset
/// by 3 elements (no gather), and the 4-float store overlaps into the next
/// pixel's slot — which the next iteration overwrites. The vector loop stops
/// one pixel early per row so the overlap never leaves the row; the scalar
/// tail finishes the row.
///
/// # Safety
/// As `box2x_f32_hwc3_avx2_to`.
pub(crate) unsafe fn box2x_f32_hwc3_avx2(
    src: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, usize, usize) {
    // SAFETY: the caller's contract.
    unsafe { new_image(box2x_f32_hwc3_avx2_to, src, (width, height, RGB)) }
}

/// `box2x_f32_hwc3_avx2` into `output`.
///
/// # Safety
/// The CPU supports AVX2, `src.len() >= width * height * 3`, and `output`
/// holds `height / 2` rows of `width / 2` pixels.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box2x_f32_hwc3_avx2_to(
    src: &[f32],
    width: usize,
    height: usize,
    output: BoxRows<f32>,
) -> Vec<f32> {
    use std::arch::x86_64::*;
    const CHANNELS: usize = 3;
    let ow = width / 2;
    debug_assert_eq!(output.rows(), height / 2);
    let stride = width * CHANNELS;
    let quarter = _mm_set1_ps(0.25);
    let (sp, dp, layout) = (src.as_ptr(), output.dst(), output.layout());
    let rows = crate::cpu::row_bands(|oy_lo, oy_hi| {
        for oy in oy_lo..oy_hi {
            let (r0, r1) = (2 * oy * stride, (2 * oy + 1) * stride);
            let orow = layout.row_start(oy);
            let mut ox = 0usize;
            // 1 output px per iter; ox+2<=ow keeps the read (src_offset+3+4 floats) and the
            // overlapping 4-float store inside the row+1 window on every row.
            while ox + 2 <= ow {
                let src_offset = 2 * ox * CHANNELS;
                prefetch_next_row_pair(sp, r0 + src_offset, stride);
                // SAFETY: the reads end by src_offset + 7 < 6(ox + 2) <= 6ow <=
                // stride in source rows 2oy, 2oy + 1 < height, inside `src`.
                let (s0, s1) = unsafe {
                    (
                        _mm_add_ps(
                            _mm_loadu_ps(sp.add(r0 + src_offset)),
                            _mm_loadu_ps(sp.add(r1 + src_offset)),
                        ),
                        _mm_add_ps(
                            _mm_loadu_ps(sp.add(r0 + src_offset + CHANNELS)),
                            _mm_loadu_ps(sp.add(r1 + src_offset + CHANNELS)),
                        ),
                    )
                };
                let means = _mm_mul_ps(_mm_add_ps(s0, s1), quarter);
                // SAFETY: ox + 2 <= ow: the 4 floats end inside output row oy.
                unsafe { _mm_storeu_ps(dp.add(orow + ox * CHANNELS), means) }; // 4th float = next px, overwritten
                ox += 1;
            }
            while ox < ow {
                let top_left = r0 + 2 * ox * CHANNELS;
                for ch in 0..CHANNELS {
                    // SAFETY: ox < ow: the output sample is in row oy, and its
                    // 2x2 block (pixels 2ox, 2ox + 1 < width of rows 2oy,
                    // 2oy + 1 < height) in `src`.
                    unsafe {
                        *dp.add(orow + ox * CHANNELS + ch) =
                            box2_mean(sp.add(top_left + ch), CHANNELS, stride);
                    }
                }
                ox += 1;
            }
        }
    });
    // SAFETY: `rows(oy_lo, oy_hi)` writes all of output rows [oy_lo, oy_hi)
    // (each row's body and tail) and nothing else.
    unsafe { output.write(false, rows) }
}

/// pshufb mask that groups, per 128-bit lane, each channel's 4 consecutive
/// columns of a 4-pixel block, so `maddubs` then `madd` sum them: the lane
/// ends as one pixel of 4x4 sums (RGB: and a zero).
fn quad_columns_mask(channels: usize) -> [u8; 32] {
    let mut mask = [0x80u8; 32];
    for lane in 0..2 {
        for ch in 0..channels {
            for col in 0..4 {
                mask[16 * lane + 4 * ch + col] = (channels * col + ch) as u8;
            }
        }
    }
    mask
}

/// AVX2 uint8 interleaved (HWC, RGB or 4 samples) box reduction by 4 (`levels` 2)
/// or 8 (`levels` 3) in one pass, with the planar pyramid's rounding: exact
/// 4x4 means ((sum + 8) >> 4, as `box4x_u8_avx2`), and for 8x the exact 2x2
/// mean of those ((sum + 2) >> 2, as `box2x`) — so every channel equals that
/// channel reduced as a planar image. With `pad`, RGB in -> RGBX out (`pad`
/// in X).
///
/// Each 128-bit lane reads one 4-pixel block per row (RGB: 12 of its 16
/// bytes), groups each channel's columns with one pshufb and sums them with
/// maddubs (4 rows: max 2040, fits i16) and madd. For 8x the two row groups'
/// rounded means add, and lane pairs add into one 8x pixel per lane. Either
/// way each lane ends as one output dword, and four registers pack into 32
/// output bytes.
///
/// # Safety
/// The CPU supports AVX2, `channels` is 3 or 4 (3 with `pad`), `levels` 2 or
/// 3, and `src.len() >= width * height * channels`. (The vector blocks check
/// their reads against `src.len()`; the scalar tail relies on the bound.)
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn box_reduce_u8_hwc_avx2(
    src: &[u8],
    (width, height): (usize, usize),
    channels: usize,
    levels: u32,
    pad: Option<u8>,
) -> (Vec<u8>, usize, usize) {
    use std::arch::x86_64::*;
    debug_assert!(
        (channels == RGB || channels == RGBX)
            && (2..=3).contains(&levels)
            && (pad.is_none() || channels == RGB)
    );
    let factor = 1usize << levels;
    let out_channels = if pad.is_some() { RGBX } else { channels };
    let (ow, oh) = (width / factor, height / factor);
    let (stride, ostride) = (width * channels, ow * out_channels);
    let mut out = unwritten_output::<u8>(oh * ostride);
    let olen = out.len();
    let lane_bytes = if channels == RGB { 12 } else { 16 };
    // Source bytes per output register (8 lane dwords) and per iteration,
    // output pixels per iteration, the bytes each source row reads (RGB
    // overreads 4) and how far the stores reach (RGB: two 16-byte stores of
    // 12 valid bytes).
    let reg = (factor / 2) * lane_bytes;
    let step_px = 8;
    let reads = 4 * reg + (16 - lane_bytes);
    let store_reach = if channels == RGB && pad.is_none() {
        28
    } else {
        32
    };
    // SAFETY: the mask is 32 bytes.
    let mask =
        unsafe { _mm256_loadu_si256(quad_columns_mask(channels).as_ptr() as *const __m256i) };
    let (ones, ones16) = (_mm256_set1_epi8(1), _mm256_set1_epi16(1));
    let (two, eight) = (_mm256_set1_epi32(2), _mm256_set1_epi32(8));
    // packus leaves lane dwords as [r0.l0 r1.l0 r2.l0 r3.l0 | r0.l1 ...].
    let lane_order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let rgbx_to_rgb = _mm256_setr_epi8(
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1, //
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1,
    );
    let pad_lanes = rgbx_pad_lanes(pad.unwrap_or(0));
    let (sp, dp) = (src.as_ptr(), out.as_mut_ptr().cast::<u8>());
    let rows = crate::cpu::row_bands(|oy_lo, oy_hi| {
        for oy in oy_lo..oy_hi {
            // SAFETY: band_row < factor, so the row (factor * oy + band_row <
            // height) starts inside `src`.
            let row = |band_row: usize| unsafe { sp.add((factor * oy + band_row) * stride) };
            // The rounded 4x4 means of the lanes' blocks at x and x + lane_bytes
            // in rows r0..r0+4.
            let quads = |r0: usize, x: usize| {
                let sums = |group_row: usize| {
                    // SAFETY: called only from `block` at an ox `fits` passed,
                    // whose reads (x + reads bytes into the band's last row at
                    // most) it checked against `src.len()`.
                    let pixels = unsafe {
                        let block_ptr = row(r0 + group_row).add(x);
                        if channels == RGB {
                            _mm256_loadu2_m128i(
                                block_ptr.add(12) as *const __m128i,
                                block_ptr as *const __m128i,
                            )
                        } else {
                            _mm256_loadu_si256(block_ptr as *const __m256i)
                        }
                    };
                    _mm256_maddubs_epi16(_mm256_shuffle_epi8(pixels, mask), ones)
                };
                let col_pairs = _mm256_add_epi16(
                    _mm256_add_epi16(sums(0), sums(1)),
                    _mm256_add_epi16(sums(2), sums(3)),
                );
                _mm256_srli_epi32(
                    _mm256_add_epi32(_mm256_madd_epi16(col_pairs, ones16), eight),
                    4,
                )
            };
            // One output dword per lane from the `reg` source bytes at x.
            let means = |x: usize| {
                if factor == 4 {
                    return quads(0, x);
                }
                // Per 4x block, the sum of its 2x2 group's quads (in the next
                // lane).
                let pairs = |x: usize| _mm256_add_epi32(quads(0, x), quads(4, x));
                let (first, second) = (pairs(x), pairs(x + reg / 2));
                let sums = _mm256_add_epi32(
                    _mm256_permute2x128_si256(first, second, 0x20),
                    _mm256_permute2x128_si256(first, second, 0x31),
                );
                _mm256_srli_epi32(_mm256_add_epi32(sums, two), 2)
            };
            let orow = oy * ostride;
            // Reads past the row may run into the next row; only the buffer
            // bounds them. The RGB stores' slop is overwritten by the next block
            // or row.
            let fits = |ox: usize| {
                (factor * oy + factor - 1) * stride + factor * ox * channels + reads <= src.len()
                    && orow + ox * out_channels + store_reach <= olen
            };
            // Only called at an ox `fits` passed.
            let block = |ox: usize| {
                let x = factor * ox * channels;
                let low = _mm256_packus_epi32(means(x), means(x + reg));
                let high = _mm256_packus_epi32(means(x + 2 * reg), means(x + 3 * reg));
                let px = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(low, high), lane_order);
                // SAFETY: `fits(ox)`: the stores end by orow + ox * out_channels
                // + store_reach <= olen, inside `out`.
                unsafe {
                    let out_ptr = dp.add(orow + ox * out_channels);
                    if pad.is_some() {
                        _mm256_storeu_si256(
                            out_ptr as *mut __m256i,
                            _mm256_or_si256(px, pad_lanes),
                        );
                    } else if channels == RGB {
                        let rgb = _mm256_shuffle_epi8(px, rgbx_to_rgb);
                        _mm_storeu_si128(out_ptr as *mut __m128i, _mm256_castsi256_si128(rgb));
                        _mm_storeu_si128(
                            out_ptr.add(12) as *mut __m128i,
                            _mm256_extracti128_si256(rgb, 1),
                        );
                    } else {
                        _mm256_storeu_si256(out_ptr as *mut __m256i, px);
                    }
                }
            };
            let mut ox = 0usize;
            while ox + step_px <= ow && fits(ox) {
                block(ox);
                ox += step_px;
            }
            // A short remainder: one block shifted back to end at the row's end
            // (recomputing some outputs) instead of the scalar tail.
            if ox < ow && ow >= step_px && fits(ow - step_px) {
                block(ow - step_px);
                ox = ow;
            }
            while ox < ow {
                for ch in 0..channels {
                    let quad = |r0: usize, x0: usize| {
                        let sum: u32 = (r0..r0 + 4)
                            .flat_map(|band_row| {
                                (x0..x0 + 4).map(move |x| {
                                    // SAFETY: x < factor * ow <= width in a row
                                    // of the band, inside `src`.
                                    u32::from(unsafe { *row(band_row).add(x * channels + ch) })
                                })
                            })
                            .sum();
                        (sum + 8) >> 4
                    };
                    let x0 = factor * ox;
                    let mean = if factor == 4 {
                        quad(0, x0)
                    } else {
                        (quad(0, x0) + quad(0, x0 + 4) + quad(4, x0) + quad(4, x0 + 4) + 2) >> 2
                    };
                    // SAFETY: ox < ow: the sample is in row oy of `out`.
                    unsafe { *dp.add(orow + ox * out_channels + ch) = mean as u8 };
                }
                if let Some(pad) = pad {
                    // SAFETY: as above (out_channels is 4 with `pad`).
                    unsafe { *dp.add(orow + ox * out_channels + 3) = pad };
                }
                ox += 1;
            }
        }
    });
    // SAFETY: `out` is this function's; `rows(oy_lo, oy_hi)` writes only
    // output rows [oy_lo, oy_hi), plus store slop inside the buffer.
    unsafe { crate::cpu::with_prefault(dp, olen, ostride, oh, rows) };
    // SAFETY: every row's body and tail wrote all of it.
    (unsafe { written_output(out) }, ow, oh)
}

// ---- AVX-512: RGB and RGBA ---------------------------------------------------

/// Exact sums of horizontally adjacent pixel pairs, per channel, as u16:
/// `pair_bytes` (an in-lane byte shuffle) puts each channel of a pair side
/// by side, and maddubs by 1 adds them.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn pair_sums_u8_avx512(
    pixels: std::arch::x86_64::__m512i,
    pair_bytes: std::arch::x86_64::__m512i,
) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::*;
    _mm512_maddubs_epi16(_mm512_shuffle_epi8(pixels, pair_bytes), _mm512_set1_epi8(1))
}

/// `(sum + 2) >> 2` of 2x2 sums: the scalar path's (and Pillow's) rounding.
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
fn round_quarter_u16_avx512(sums: std::arch::x86_64::__m512i) -> std::arch::x86_64::__m512i {
    use std::arch::x86_64::*;
    _mm512_srli_epi16::<2>(_mm512_add_epi16(sums, _mm512_set1_epi16(2)))
}

/// AVX-512 uint8 RGBA 2x2 box: per 128-bit lane of 4 pixels, the two
/// horizontal pairs' sums per channel by one shuffle and maddubs
/// (`pair_sums_u8_avx512`), the rows' sums added and rounded exactly as the
/// scalar path. 16 output pixels per iteration.
///
/// # Safety
/// As `box2x_u8_hwc4_avx512_to`.
pub(crate) unsafe fn box2x_u8_hwc4_avx512(
    src: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, usize, usize) {
    // SAFETY: the caller's contract.
    unsafe { new_image(box2x_u8_hwc4_avx512_to, src, (width, height, RGBX)) }
}

/// `box2x_u8_hwc4_avx512` into `output`.
///
/// # Safety
/// The CPU supports AVX-512 F and BW, `src.len() >= width * height * 4`, and
/// `output` holds `height / 2` rows of `width / 2` pixels.
#[target_feature(enable = "avx512f,avx512bw")]
pub(crate) unsafe fn box2x_u8_hwc4_avx512_to(
    src: &[u8],
    width: usize,
    height: usize,
    output: BoxRows<u8>,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    const CHANNELS: usize = 4;
    // From 32 input pixels of each row: two registers of 16.
    const OUT_PIXELS: usize = 16;
    let ow = width / 2;
    debug_assert_eq!(output.rows(), height / 2);
    let stride = width * CHANNELS;
    let (sp, dp, layout) = (src.as_ptr(), output.dst(), output.layout());
    // Per lane: each channel of pixels 0 and 1, then of pixels 2 and 3.
    let pair_bytes = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15,
    ));
    // packus leaves lane L = [first register's output pixels 2L, 2L + 1 |
    // second's 8 + 2L, 9 + 2L]; the qwords back in pixel order.
    let order = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    let rows = crate::cpu::row_bands(|oy_lo, oy_hi| {
        for oy in oy_lo..oy_hi {
            let (r0, r1) = (2 * oy * stride, (2 * oy + 1) * stride);
            let orow = layout.row_start(oy);
            let mut ox = 0usize;
            // ox + 16 <= ow keeps both the 128 input bytes and the 64-byte store
            // inside their rows. No software prefetch: with two line-aligned
            // loads per row the hardware prefetchers keep up (prefetching the
            // next row pair, as the RGB kernels do, didn't help here).
            while ox + OUT_PIXELS <= ow {
                let src_offset = 2 * ox * CHANNELS;
                let quarter = |offset: usize| {
                    let row = |row_start: usize| {
                        // SAFETY: offset <= 64, so the 64 bytes end by
                        // src_offset + 128 <= stride (ox + 16 <= ow) in source
                        // rows 2oy, 2oy + 1 < height, inside `src`.
                        let pixels = unsafe {
                            _mm512_loadu_si512(
                                sp.add(row_start + src_offset + offset) as *const __m512i
                            )
                        };
                        pair_sums_u8_avx512(pixels, pair_bytes)
                    };
                    round_quarter_u16_avx512(_mm512_add_epi16(row(r0), row(r1)))
                };
                let packed = _mm512_packus_epi16(quarter(0), quarter(64));
                // SAFETY: ox + 16 <= ow: the 64 bytes are inside output row oy.
                unsafe {
                    _mm512_storeu_si512(
                        dp.add(orow + ox * CHANNELS) as *mut __m512i,
                        _mm512_permutexvar_epi64(order, packed),
                    );
                }
                ox += OUT_PIXELS;
            }
            while ox < ow {
                let top_left = r0 + 2 * ox * CHANNELS;
                for ch in 0..CHANNELS {
                    // SAFETY: ox < ow: the output sample is in row oy, and its
                    // 2x2 block (pixels 2ox, 2ox + 1 < width of rows 2oy,
                    // 2oy + 1 < height) in `src`.
                    unsafe {
                        *dp.add(orow + ox * CHANNELS + ch) =
                            box2_mean(sp.add(top_left + ch), CHANNELS, stride);
                    }
                }
                ox += 1;
            }
        }
    });
    // SAFETY: `rows(oy_lo, oy_hi)` writes all of output rows [oy_lo, oy_hi)
    // (each row's body and tail) and nothing else.
    unsafe { output.write(false, rows) }
}

/// AVX-512 uint8 RGB 2x2 box, 32 output pixels per iteration: per row
/// three line-aligned loads of 64 pixels, cut by two-register dword permutes
/// into four groups of 16 with 4 whole pixels (12 bytes) per 128-bit lane;
/// then as the RGBA kernel, and each group's 6 sums per lane gathered by a
/// word permute and narrowed to 24 bytes. (A 64-byte load per 16 pixels
/// would split cache lines three times in four, which costs when the rows
/// come from memory.)
///
/// # Safety
/// As `box2x_u8_hwc3_avx512_to`.
pub(crate) unsafe fn box2x_u8_hwc3_avx512(
    src: &[u8],
    width: usize,
    height: usize,
) -> (Vec<u8>, usize, usize) {
    // SAFETY: the caller's contract.
    unsafe { new_image(box2x_u8_hwc3_avx512_to, src, (width, height, RGB)) }
}

/// `box2x_u8_hwc3_avx512` into `output`.
///
/// # Safety
/// The CPU supports AVX-512 F, BW and VL, `src.len() >= width * height * 3`,
/// and `output` holds `height / 2` rows of `width / 2` pixels.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(crate) unsafe fn box2x_u8_hwc3_avx512_to(
    src: &[u8],
    width: usize,
    height: usize,
    output: BoxRows<u8>,
) -> Vec<u8> {
    use std::arch::x86_64::*;
    const CHANNELS: usize = 3;
    // Per iteration and row: 64 pixels = 48 dwords = 3 registers, cut into
    // 4 groups of 16 pixels (12 dwords).
    const LOADS: usize = 3;
    const GROUPS: usize = 4;
    const GROUP_DWORDS: usize = 12;
    const GROUP_OUT_BYTES: usize = 8 * CHANNELS;
    const OUT_PIXELS: usize = 32;
    let ow = width / 2;
    debug_assert_eq!(output.rows(), height / 2);
    let stride = width * CHANNELS;
    let (sp, dp, layout) = (src.as_ptr(), output.dst(), output.layout());
    // Group k: dwords 12k.. of the loads, from load `12k / 16` and the next;
    // lane L gets pixels 4L..4L + 4 (dwords 3L..3L + 3, its 4th unused).
    let groups: [(usize, __m512i); GROUPS] = std::array::from_fn(|group| {
        let first = GROUP_DWORDS * group / 16;
        let idx: [i32; 16] = std::array::from_fn(|dword| {
            (GROUP_DWORDS * group + 3 * (dword / 4) + (dword % 4).min(2) - 16 * first) as i32
        });
        // SAFETY: `idx` is 64 bytes.
        (first, unsafe {
            _mm512_loadu_si512(idx.as_ptr() as *const __m512i)
        })
    });
    // Per lane: each channel of pixels 0 and 1, then of pixels 2 and 3; the
    // last 4 bytes zero.
    let pair_bytes = _mm512_broadcast_i32x4(_mm_setr_epi8(
        0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11, -1, -1, -1, -1,
    ));
    // The 6 sums of each lane's 8 words, in pixel order.
    let mut compact = [0i16; 32];
    for (word_index, word) in compact.iter_mut().enumerate().take(GROUP_OUT_BYTES) {
        *word = (8 * (word_index / 6) + word_index % 6) as i16;
    }
    // SAFETY: `compact` is 64 bytes.
    let compact = unsafe { _mm512_loadu_si512(compact.as_ptr() as *const __m512i) };
    let store_mask = u32::MAX >> (32 - GROUP_OUT_BYTES);
    let rows = crate::cpu::row_bands(|oy_lo, oy_hi| {
        for oy in oy_lo..oy_hi {
            let (r0, r1) = (2 * oy * stride, (2 * oy + 1) * stride);
            let (orow, store_end) = (layout.row_start(oy), layout.store_end(oy));
            let mut ox = 0usize;
            // ox + 32 <= ow keeps the 192 input bytes inside the row.
            while ox + OUT_PIXELS <= ow {
                let src_offset = 2 * ox * CHANNELS;
                for line in 0..LOADS {
                    prefetch_next_row_pair(sp, r0 + src_offset + 64 * line, stride);
                }
                let load_row = |row_start: usize| -> [__m512i; LOADS] {
                    std::array::from_fn(|load| {
                        // SAFETY: load < 3, so the 64 bytes end by src_offset
                        // + 192 <= stride (ox + 32 <= ow) in source rows 2oy,
                        // 2oy + 1 < height, inside `src`.
                        unsafe {
                            _mm512_loadu_si512(
                                sp.add(row_start + src_offset + 64 * load) as *const __m512i
                            )
                        }
                    })
                };
                let (top, bottom) = (load_row(r0), load_row(r1));
                for (group, &(first, idx)) in groups.iter().enumerate() {
                    let second = (first + 1).min(LOADS - 1);
                    let sums = |row_loads: &[__m512i; LOADS]| {
                        pair_sums_u8_avx512(
                            _mm512_permutex2var_epi32(row_loads[first], idx, row_loads[second]),
                            pair_bytes,
                        )
                    };
                    let rounded =
                        round_quarter_u16_avx512(_mm512_add_epi16(sums(&top), sums(&bottom)));
                    let bytes = _mm512_cvtepi16_epi8(_mm512_permutexvar_epi16(compact, rounded));
                    let at = orow + ox * CHANNELS + GROUP_OUT_BYTES * group;
                    if at + 32 <= store_end {
                        // 8 bytes of slop, which the next store (or the tail, or
                        // a new image's next row) overwrites.
                        // SAFETY: at + 32 <= the row's store end (checked).
                        unsafe { _mm256_storeu_si256(dp.add(at) as *mut __m256i, bytes) };
                    } else {
                        // SAFETY: the mask writes bytes at..at + 24 <= orow +
                        // 3(ox + 32) <= orow + 3ow (ox + 32 <= ow), inside output
                        // row oy; masked-off bytes are not accessed.
                        unsafe {
                            _mm256_mask_storeu_epi8(dp.add(at) as *mut i8, store_mask, bytes);
                        }
                    }
                }
                ox += OUT_PIXELS;
            }
            while ox < ow {
                let top_left = r0 + 2 * ox * CHANNELS;
                for ch in 0..CHANNELS {
                    // SAFETY: ox < ow: the output sample is in row oy, and its
                    // 2x2 block (pixels 2ox, 2ox + 1 < width of rows 2oy,
                    // 2oy + 1 < height) in `src`.
                    unsafe {
                        *dp.add(orow + ox * CHANNELS + ch) =
                            box2_mean(sp.add(top_left + ch), CHANNELS, stride);
                    }
                }
                ox += 1;
            }
        }
    });
    // SAFETY: `rows(oy_lo, oy_hi)` writes all of output rows [oy_lo, oy_hi)
    // (each row's body and tail), its stores below each row's store end.
    unsafe { output.write(false, rows) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box_reduce::box2x_interleaved;
    use crate::box_reduce::tests::{
        BoxTestSample, assert_same_image, box_sizes, interleaved_pyramid_reference, rgb_to_rgbx,
        test_images,
    };
    use crate::cpu::{LARGE_OUTPUT_BYTES, PREFAULT_OUTPUT_BYTES};
    use crate::testing::{Rng, UnsafeImageKernel, random_samples};

    fn avx2_detected() -> bool {
        std::is_x86_feature_detected!("avx2")
    }

    fn avx512_detected() -> bool {
        std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512vl")
    }

    /// Runs the 2x `kernel` of `channels`-sample pixels on random and extreme
    /// images of every box test size, against the scalar interleaved halving.
    ///
    /// # Safety
    /// The CPU supports `kernel`'s features.
    unsafe fn assert_halves_as_scalar<T: BoxTestSample>(
        channels: usize,
        kernel: UnsafeImageKernel<T>,
    ) {
        let mut rng = Rng::new(20);
        for (width, height) in box_sizes(1) {
            for src in test_images::<T>(&mut rng, width * height * channels) {
                // SAFETY: the CPU features (the caller's contract); src holds
                // width * height pixels.
                let simd = unsafe { kernel(&src, width, height) };
                assert_same_image(
                    &simd,
                    &box2x_interleaved(&src, width, height, channels),
                    &format!("{width} x {height} x {channels}"),
                );
            }
        }
    }

    #[test]
    fn box2x_u8_avx2_equals_scalar() {
        if !avx2_detected() {
            return;
        }
        // SAFETY: AVX2 is available.
        unsafe {
            assert_halves_as_scalar::<u8>(RGB, box2x_u8_hwc3_avx2);
            assert_halves_as_scalar::<u8>(RGBX, box2x_u8_hwc4_avx2);
        }
    }

    #[test]
    fn box2x_u8_avx512_equals_scalar() {
        if !avx512_detected() {
            return;
        }
        // SAFETY: AVX-512 F, BW and VL are available.
        unsafe {
            assert_halves_as_scalar::<u8>(RGB, box2x_u8_hwc3_avx512);
            assert_halves_as_scalar::<u8>(RGBX, box2x_u8_hwc4_avx512);
        }
    }

    #[test]
    fn box2x_u16_avx2_equals_scalar() {
        if !avx2_detected() {
            return;
        }
        // SAFETY: AVX2 is available.
        unsafe {
            assert_halves_as_scalar::<u16>(RGB, box2x_u16_hwc3_avx2);
            assert_halves_as_scalar::<u16>(RGBX, box2x_u16_hwc4_avx2);
        }
    }

    #[test]
    fn box2x_f32_avx2_equals_scalar() {
        if !avx2_detected() {
            return;
        }
        // SAFETY: AVX2 is available.
        unsafe {
            assert_halves_as_scalar::<f32>(RGB, box2x_f32_hwc3_avx2);
            assert_halves_as_scalar::<f32>(RGBX, box2x_f32_hwc4_avx2);
        }
    }

    #[test]
    fn box2x_rgb_to_rgbx_avx2_equals_the_scalar_halving_padded() {
        if !avx2_detected() {
            return;
        }
        let mut rng = Rng::new(21);
        for pad in [0, 0x5A, u8::MAX] {
            for (width, height) in box_sizes(1) {
                for src in test_images::<u8>(&mut rng, width * height * RGB) {
                    let (rgb, ow, oh) = box2x_interleaved(&src, width, height, RGB);
                    // SAFETY: AVX2 is available; src holds width * height RGB
                    // pixels.
                    let simd = unsafe { box2x_u8_hwc3to4_avx2(&src, width, height, pad) };
                    assert_same_image(
                        &simd,
                        &(rgb_to_rgbx(&rgb, pad), ow, oh),
                        &format!("pad {pad}, {width} x {height}"),
                    );
                }
            }
        }
    }

    /// The fused u8 reduction (`levels`, `channels`, `pad`) against each
    /// channel reduced by the planar pyramid reference, padded.
    fn assert_fused_u8_equals_planar_pyramid(channels: usize, levels: u32, pad: Option<u8>) {
        let mut rng = Rng::new(22);
        for (width, height) in box_sizes(1) {
            for src in test_images::<u8>(&mut rng, width * height * channels) {
                let reference =
                    interleaved_pyramid_reference(&src, (width, height, channels), levels);
                let expected = match pad {
                    Some(pad) => (rgb_to_rgbx(&reference.0, pad), reference.1, reference.2),
                    None => reference,
                };
                // SAFETY: AVX2 is available (the callers check); channels is 3
                // or 4 (3 with a pad), levels 2 or 3; src holds the image.
                let simd =
                    unsafe { box_reduce_u8_hwc_avx2(&src, (width, height), channels, levels, pad) };
                let context = format!("{levels} levels, {width} x {height} x {channels}, {pad:?}");
                assert_same_image(&simd, &expected, &context);
            }
        }
    }

    #[test]
    fn fused_u8_levels_avx2_equal_the_planar_pyramid() {
        if !avx2_detected() {
            return;
        }
        for levels in [2, 3] {
            assert_fused_u8_equals_planar_pyramid(RGB, levels, None);
            assert_fused_u8_equals_planar_pyramid(RGBX, levels, None);
            assert_fused_u8_equals_planar_pyramid(RGB, levels, Some(0x81));
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
        for channels in [RGB, RGBX] {
            for levels in [2, 3] {
                for (width, height) in box_sizes(8) {
                    let len = width * height * channels;
                    let mut src: Vec<u8> = random_samples(&mut rng, len + 64 * channels);
                    // SAFETY: AVX2 is available; channels is 3 or 4, levels 2
                    // or 3; both slices hold the image.
                    let (exact, padded) = unsafe {
                        (
                            box_reduce_u8_hwc_avx2(
                                &src[..len],
                                (width, height),
                                channels,
                                levels,
                                None,
                            ),
                            box_reduce_u8_hwc_avx2(&src, (width, height), channels, levels, None),
                        )
                    };
                    src.truncate(len);
                    let context = format!("{levels} levels, {width} x {height} x {channels}");
                    assert_same_image(&padded, &exact, &context);
                    assert_same_image(
                        &exact,
                        &interleaved_pyramid_reference(&src, (width, height, channels), levels),
                        &context,
                    );
                }
            }
        }
    }

    /// The 2x `kernel` on an image of `channels`-sample pixels whose output
    /// passes the streaming-store and prefault thresholds.
    ///
    /// # Safety
    /// The CPU supports `kernel`'s features.
    unsafe fn assert_large_halves_as_scalar<T: BoxTestSample>(
        channels: usize,
        kernel: UnsafeImageKernel<T>,
    ) {
        let out_bytes = PREFAULT_OUTPUT_BYTES.max(LARGE_OUTPUT_BYTES);
        let out_width = 1027;
        let out_height = out_bytes.div_ceil(out_width * channels * size_of::<T>());
        let (width, height) = (2 * out_width + 1, 2 * out_height);
        let src: Vec<T> = random_samples(&mut Rng::new(24), width * height * channels);
        // SAFETY: the CPU features (the caller's contract); src holds width *
        // height pixels.
        let simd = unsafe { kernel(&src, width, height) };
        assert_same_image(
            &simd,
            &box2x_interleaved(&src, width, height, channels),
            &format!("{width} x {height} x {channels}"),
        );
    }

    #[test]
    fn large_outputs_stream_past_the_cache_and_prefault() {
        if !avx2_detected() {
            return;
        }
        // SAFETY: AVX2 is available.
        unsafe {
            assert_large_halves_as_scalar::<u8>(RGBX, box2x_u8_hwc4_avx2);
            assert_large_halves_as_scalar::<u16>(RGBX, box2x_u16_hwc4_avx2);
            assert_large_halves_as_scalar::<u16>(RGB, box2x_u16_hwc3_avx2);
        }
        if avx512_detected() {
            // SAFETY: AVX-512 F, BW and VL are available.
            unsafe { assert_large_halves_as_scalar::<u8>(RGB, box2x_u8_hwc3_avx512) };
        }
    }
}

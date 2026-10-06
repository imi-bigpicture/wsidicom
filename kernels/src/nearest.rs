//! Nearest-neighbour resize, any dtype and channel count, at `fast_image_resize`'s
//! sample positions. Every output row samples the same source columns, so a
//! call plans one row's gather once, as byte shuffles: each 16 output bytes
//! from one or two 16-byte windows of the source row (x86 `pshufb`), or from
//! up to four consecutive ones (NEON `tbl`). An output row that samples the
//! same source row as the one above is a copy of it.

use crate::convolve::Geometry;

/// The output rows past the last sampled source row stay zero, as
/// `fast_image_resize` leaves them. Columns are `floor((x + 0.5) * sx)`; rows
/// accumulate `sy` from `sy / 2`, as it steps them.
#[cfg(test)]
pub(crate) fn resize_nearest<T: bytemuck::Pod>(
    src: &[T],
    src_size: (usize, usize),
    dst: (usize, usize),
    channels: usize,
) -> Vec<T> {
    resize_nearest_over(src, Geometry::resize(src_size, dst), channels)
}

/// `resize_nearest` for any `geometry`: the extent sets the steps, and a
/// position past the image takes its last row or column.
pub(crate) fn resize_nearest_over<T: bytemuck::Pod>(
    src: &[T],
    geometry: Geometry,
    channels: usize,
) -> Vec<T> {
    let Geometry {
        src: (width, height),
        extent: (extent_width, extent_height),
        dst: (ow, oh),
    } = geometry;
    let (sx, sy) = (
        extent_width as f64 / ow as f64,
        extent_height as f64 / oh as f64,
    );
    let cols: Vec<u32> = (0..ow)
        .map(|x| ((sx * 0.5 + sx * x as f64) as usize).min(width - 1) as u32)
        .collect();
    let rows = ((((extent_height as f64 - sy * 0.5) / sy).max(0.0)).ceil() as usize).min(oh);
    let src_rows: Vec<usize> = std::iter::successors(Some(sy * 0.5), |y| Some(y + sy))
        .take(rows)
        .map(|y| (y as usize).min(height - 1))
        .collect();
    let pixel_bytes = channels * size_of::<T>();
    let (in_row, out_row) = (width * pixel_bytes, ow * pixel_bytes);
    // Rows that gather (the others copy the row before).
    let gathered_rows = 1 + src_rows
        .windows(2)
        .filter(|pair| pair[0] != pair[1])
        .count();
    let gather = RowGather::new(&cols, pixel_bytes, in_row, gathered_rows);
    let src_bytes: &[u8] = bytemuck::cast_slice(src);
    let mut out = crate::cpu::unwritten_output::<T>(ow * oh * channels);
    let dp = out.as_mut_ptr().cast::<u8>();
    let fill = |lo: usize, hi: usize| {
        for y in lo..hi {
            // SAFETY: row y of `out`, which only this call writes (with_prefault);
            // row y - 1 (same band) is written.
            unsafe {
                let dst = dp.add(y * out_row);
                if y > lo && src_rows[y] == src_rows[y - 1] {
                    std::ptr::copy_nonoverlapping(dst.sub(out_row), dst, out_row);
                } else {
                    gather.row(&src_bytes[src_rows[y] * in_row..][..in_row], dst);
                }
            }
        }
    };
    // SAFETY: `out` is this function's; `fill(lo, hi)` writes rows [lo, hi).
    unsafe { crate::cpu::with_prefault(dp, out_row * oh, out_row, rows, fill) };
    // SAFETY: rows `rows..oh` are inside `out` and nothing wrote them.
    unsafe { std::ptr::write_bytes(dp.add(rows * out_row), 0, (oh - rows) * out_row) };
    // SAFETY: every row is gathered, copied or zeroed.
    unsafe { crate::cpu::written_output(out) }
}

/// One output row's gather from a source row, planned once per call.
struct RowGather {
    /// The source pixel of every output pixel.
    cols: Vec<u32>,
    /// 16-byte chunks covering the output row (the last shifted back to end
    /// at the row's end); empty where the SIMD path does not apply, or where
    /// some 16 output bytes span more than the gather's windows (reductions
    /// beyond about 2x on x86, 4x on NEON), which then go a pixel at a time.
    chunks: Vec<Chunk>,
    pixel_bytes: usize,
    /// An exact 2x reduction of 1- or 2-byte pixels: every output pixel is
    /// the odd one of its source pair, gathered by a shift and a pack.
    odd_pixels: bool,
}

/// 16 output bytes at `out_at`: from the 16-byte source windows at `windows`
/// through byte shuffles `masks` (0x80 selects nothing; with two windows the
/// two results OR together).
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Chunk {
    out_at: u32,
    windows: [u32; 2],
    masks: [[u8; 16]; 2],
    two_windows: bool,
}

/// 16 output bytes at `out_at`: `index[k]` is the source byte `start +
/// index[k]` of a span of `tables` consecutive 16-byte windows (one `tbl`
/// over 1 to 4 registers).
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct Chunk {
    out_at: u32,
    start: u32,
    tables: u8,
    index: [u8; 16],
}

/// The placeholder chunk where no SIMD gather exists.
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[derive(Clone, Copy)]
struct Chunk;

const CHUNK: usize = 16;
/// Windows a NEON `tbl` gathers from.
#[cfg(target_arch = "aarch64")]
const MAX_TABLES: usize = 4;
/// The rows that repay planning a chunk whose table loads replace twice as
/// many pixel moves (fewer rows where they replace more, more where fewer).
#[cfg(target_arch = "aarch64")]
const PLAN_ROWS: f64 = 32.0;

impl RowGather {
    /// The gather of `cols` for `rows` rows of `in_row` bytes.
    fn new(cols: &[u32], pixel_bytes: usize, in_row: usize, rows: usize) -> RowGather {
        let mut chunks = Vec::new();
        let out_row = cols.len() * pixel_bytes;
        let odd_pixels = simd_gather()
            && matches!(pixel_bytes, 1 | 2)
            && out_row >= 32
            && cols
                .iter()
                .enumerate()
                .all(|(x, &col)| col as usize == 2 * x + 1);
        if simd_gather()
            && !odd_pixels
            && out_row >= CHUNK
            && in_row >= CHUNK
            && Self::gather_may_pay(in_row, out_row, pixel_bytes, rows)
        {
            chunks = Self::plan(cols, pixel_bytes, in_row)
                .filter(|chunks| Self::gather_pays(chunks, pixel_bytes))
                .unwrap_or_default();
        }
        RowGather {
            cols: cols.to_vec(),
            chunks,
            pixel_bytes,
            odd_pixels,
        }
    }

    /// The chunks covering a row of `cols` (the last one overlapping its
    /// predecessor), or None where one spans more than the gather's windows.
    fn plan(cols: &[u32], pixel_bytes: usize, in_row: usize) -> Option<Vec<Chunk>> {
        let out_row = cols.len() * pixel_bytes;
        // Byte `position` of a run starting at a pixel is byte `position %
        // pixel_bytes` of its pixel `position / pixel_bytes` (positions below
        // 32: a chunk starts within a pixel).
        let pixel_of: [usize; 2 * CHUNK] = std::array::from_fn(|position| position / pixel_bytes);
        let byte_of: [u32; 2 * CHUNK] =
            std::array::from_fn(|position| (position % pixel_bytes) as u32);
        let mut chunks = Vec::with_capacity(out_row.div_ceil(CHUNK));
        for start in (0..out_row).step_by(CHUNK) {
            let at = start.min(out_row - CHUNK);
            let (first, offset) = (at / pixel_bytes, at % pixel_bytes);
            let pixels = &cols[first..];
            // The source byte of each of the 16 output bytes.
            let bytes: [u32; CHUNK] = std::array::from_fn(|slot| {
                pixels[pixel_of[offset + slot]] * pixel_bytes as u32 + byte_of[offset + slot]
            });
            chunks.push(Self::chunk(&bytes, at, in_row)?);
        }
        Some(chunks)
    }

    /// Whether the planned gather beats moving pixels one at a time: a chunk
    /// costs a load per table, a pixel a fixed-size move per power-of-2 part
    /// (`copy_pixels`), so on average a chunk must load fewer tables than its
    /// pixels take moves (wide pixels, and large reductions of 4-byte ones,
    /// go a pixel at a time).
    #[cfg(target_arch = "aarch64")]
    fn gather_pays(chunks: &[Chunk], pixel_bytes: usize) -> bool {
        let tables: usize = chunks.iter().map(|chunk| usize::from(chunk.tables)).sum();
        tables * pixel_bytes < chunks.len() * CHUNK * pixel_bytes.count_ones() as usize
    }

    /// x86 plans at most two windows per chunk, which always pays.
    #[cfg(not(target_arch = "aarch64"))]
    fn gather_pays(_chunks: &[Chunk], _pixel_bytes: usize) -> bool {
        true
    }

    /// `gather_pays` before planning, also counting the plan's cost against
    /// the `rows` it serves, so rows that can't repay it skip it: a chunk's 16
    /// output bytes span about `(16 - pixel_bytes) * ratio + pixel_bytes`
    /// source bytes, so it loads at least that over 16 tables; per row it
    /// saves the pixel moves its table loads replace.
    #[cfg(target_arch = "aarch64")]
    fn gather_may_pay(in_row: usize, out_row: usize, pixel_bytes: usize, rows: usize) -> bool {
        let ratio = in_row as f64 / out_row as f64;
        let span = (CHUNK - pixel_bytes.min(CHUNK)) as f64 * ratio + pixel_bytes as f64;
        let tables = (span / CHUNK as f64).ceil();
        let moves = (CHUNK / pixel_bytes) as f64 * f64::from(pixel_bytes.count_ones());
        rows as f64 * (moves - tables) > PLAN_ROWS * tables
    }

    /// x86 plans every row.
    #[cfg(not(target_arch = "aarch64"))]
    fn gather_may_pay(_in_row: usize, _out_row: usize, _pixel_bytes: usize, _rows: usize) -> bool {
        true
    }

    /// The span and indices for the 16 output bytes taking source `bytes`, or
    /// None where they span more than `MAX_TABLES` windows.
    #[cfg(target_arch = "aarch64")]
    fn chunk(bytes: &[u32; CHUNK], at: usize, in_row: usize) -> Option<Chunk> {
        let lo = bytes.iter().fold(u32::MAX, |lo, &byte| lo.min(byte)) as usize;
        let hi = bytes.iter().fold(0, |hi, &byte| hi.max(byte)) as usize;
        let tables = (hi - lo) / CHUNK + 1;
        if tables > MAX_TABLES || in_row < tables * CHUNK {
            return None;
        }
        let start = lo.min(in_row - tables * CHUNK);
        let index = bytes.map(|src_byte| (src_byte as usize - start) as u8);
        Some(Chunk {
            out_at: at as u32,
            start: start as u32,
            tables: tables as u8,
            index,
        })
    }

    /// No SIMD gather here.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fn chunk(_bytes: &[u32; CHUNK], _at: usize, _in_row: usize) -> Option<Chunk> {
        None
    }

    /// The windows and masks for the 16 output bytes taking source `bytes`,
    /// or None where they span more than two windows.
    #[cfg(target_arch = "x86_64")]
    fn chunk(bytes: &[u32; CHUNK], at: usize, in_row: usize) -> Option<Chunk> {
        let lo = bytes.iter().fold(u32::MAX, |lo, &byte| lo.min(byte)) as usize;
        let hi = bytes.iter().fold(0, |hi, &byte| hi.max(byte)) as usize;
        let masks_for = |windows: [usize; 2]| -> [[u8; 16]; 2] {
            let mut masks = [[0x80u8; 16]; 2];
            for (out_byte, &src_byte) in bytes.iter().enumerate() {
                let src_byte = src_byte as usize;
                let window_index =
                    usize::from(!(src_byte >= windows[0] && src_byte < windows[0] + CHUNK));
                masks[window_index][out_byte] = (src_byte - windows[window_index]) as u8;
            }
            masks
        };
        let one = lo.min(in_row - CHUNK);
        if hi < one + CHUNK {
            return Some(Chunk {
                out_at: at as u32,
                windows: [one as u32; 2],
                masks: masks_for([one, one]),
                two_windows: false,
            });
        }
        if in_row < 2 * CHUNK {
            return None;
        }
        let first = lo.min(in_row - 2 * CHUNK);
        let windows = [first, first + CHUNK];
        (hi < first + 2 * CHUNK).then(|| Chunk {
            out_at: at as u32,
            windows: windows.map(|window| window as u32),
            masks: masks_for(windows),
            two_windows: true,
        })
    }

    /// Gathers one output row from source row `row` into `dst`.
    ///
    /// # Safety
    /// `row` is a source row (the `in_row` bytes the plan was made for);
    /// `dst` is valid for writes of an output row (`cols.len() * pixel_bytes`
    /// bytes).
    unsafe fn row(&self, row: &[u8], dst: *mut u8) {
        if self.odd_pixels {
            // SAFETY: planned only where simd_gather (AVX2).
            #[cfg(target_arch = "x86_64")]
            return unsafe { self.row_odd_pixels_avx2(row, dst) };
            // SAFETY: planned only where simd_gather (NEON).
            #[cfg(target_arch = "aarch64")]
            return unsafe { self.row_odd_pixels_neon(row, dst) };
        }
        if !self.chunks.is_empty() {
            // SAFETY: chunks are planned only where simd_gather (AVX2).
            #[cfg(target_arch = "x86_64")]
            return unsafe { self.row_ssse3(row, dst) };
            // SAFETY: chunks are planned only where simd_gather (NEON).
            #[cfg(target_arch = "aarch64")]
            return unsafe { self.row_tbl(row, dst) };
        }
        // SAFETY: the caller's contract.
        unsafe { self.row_by_pixel(row, dst) }
    }

    /// The row a pixel at a time: one fixed-size move per pixel (1, 3 or 4
    /// samples of 1, 2 or 4 bytes).
    ///
    /// # Safety
    /// As `row`.
    unsafe fn row_by_pixel(&self, row: &[u8], dst: *mut u8) {
        // SAFETY: (every arm) the columns are inside the row; `dst` holds the
        // output row.
        unsafe {
            match self.pixel_bytes {
                1 => self.copy_pixels::<1>(row, dst),
                2 => self.copy_pixels::<2>(row, dst),
                3 => self.copy_pixels::<3>(row, dst),
                4 => self.copy_pixels::<4>(row, dst),
                6 => self.copy_pixels::<6>(row, dst),
                8 => self.copy_pixels::<8>(row, dst),
                12 => self.copy_pixels::<12>(row, dst),
                16 => self.copy_pixels::<16>(row, dst),
                pixel_bytes => unreachable!("{pixel_bytes}-byte pixels"),
            }
        }
    }

    /// `row_by_pixel` for `PIXEL_BYTES`-byte pixels.
    ///
    /// # Safety
    /// As `row`.
    unsafe fn copy_pixels<const PIXEL_BYTES: usize>(&self, row: &[u8], dst: *mut u8) {
        let (sp, dp) = (
            row.as_ptr() as *const [u8; PIXEL_BYTES],
            dst as *mut [u8; PIXEL_BYTES],
        );
        for (out_pixel, &x) in self.cols.iter().enumerate() {
            // SAFETY: as row_by_pixel.
            unsafe {
                dp.add(out_pixel)
                    .write_unaligned(sp.add(x as usize).read_unaligned());
            };
        }
    }

    /// The odd pixel of every source pair: the high byte (word) of each word
    /// (dword) shifted down and packed, the pack's lane order fixed by one
    /// permute; 32 output bytes from 64 source bytes, the last step shifted
    /// back to end at the row's end.
    ///
    /// # Safety
    /// The CPU supports AVX2; otherwise as `row`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn row_odd_pixels_avx2(&self, row: &[u8], dst: *mut u8) {
        use std::arch::x86_64::*;
        let out_row = self.cols.len() * self.pixel_bytes;
        debug_assert!(out_row >= 32 && row.len() >= 2 * out_row);
        let sp = row.as_ptr();
        // Called with at + 32 <= out_row.
        let step = |at: usize| {
            // SAFETY: 2 * at + 64 <= 2 * out_row <= the source row (the plan's
            // columns 2x + 1 reach its end).
            let (first, second) = unsafe {
                (
                    _mm256_loadu_si256(sp.add(2 * at) as *const __m256i),
                    _mm256_loadu_si256(sp.add(2 * at + 32) as *const __m256i),
                )
            };
            // Per 128-bit lane: first's outputs, then second's.
            let packed = if self.pixel_bytes == 1 {
                _mm256_packus_epi16(
                    _mm256_srli_epi16::<8>(first),
                    _mm256_srli_epi16::<8>(second),
                )
            } else {
                _mm256_packus_epi32(
                    _mm256_srli_epi32::<16>(first),
                    _mm256_srli_epi32::<16>(second),
                )
            };
            let ordered = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);
            // SAFETY: at + 32 <= out_row, inside the output row at `dst`
            // (the caller's contract).
            unsafe { _mm256_storeu_si256(dst.add(at) as *mut __m256i, ordered) };
        };
        let mut at = 0;
        while at + 32 <= out_row {
            step(at);
            at += 32;
        }
        if at < out_row {
            step(out_row - 32);
        }
    }

    /// The row by the planned chunks' byte shuffles.
    ///
    /// # Safety
    /// The CPU supports AVX2; otherwise as `row`.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn row_ssse3(&self, row: &[u8], dst: *mut u8) {
        use std::arch::x86_64::*;
        let sp = row.as_ptr();
        for chunk in &self.chunks {
            // SAFETY: the plan puts every chunk at most out_row - 16 into the
            // output row at `dst` (the caller's contract).
            let out = unsafe { dst.add(chunk.out_at as usize) };
            let shuffled = |window_index: usize| {
                // SAFETY: the plan puts every window at most in_row - 16 into
                // the source row; the mask is 16 bytes.
                unsafe {
                    let window = _mm_loadu_si128(
                        sp.add(chunk.windows[window_index] as usize) as *const __m128i
                    );
                    _mm_shuffle_epi8(
                        window,
                        _mm_loadu_si128(chunk.masks[window_index].as_ptr() as *const __m128i),
                    )
                }
            };
            let bytes = if chunk.two_windows {
                _mm_or_si128(shuffled(0), shuffled(1))
            } else {
                shuffled(0)
            };
            // SAFETY: `out` has 16 bytes of the output row (see above).
            unsafe { _mm_storeu_si128(out as *mut __m128i, bytes) };
        }
    }
}

/// Whether rows gather by byte shuffles here.
fn simd_gather() -> bool {
    #[cfg(target_arch = "x86_64")]
    return crate::cpu::has_avx2();
    #[cfg(target_arch = "aarch64")]
    return crate::cpu::use_neon();
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    false
}

#[cfg(target_arch = "aarch64")]
impl RowGather {
    /// The planned chunks: per chunk, `tables` consecutive 16-byte windows
    /// loaded and one `tbl` over them.
    ///
    /// # Safety
    /// `row` is the source row the chunks were planned for; `dst` holds an
    /// output row.
    #[target_feature(enable = "neon")]
    unsafe fn row_tbl(&self, row: &[u8], dst: *mut u8) {
        use std::arch::aarch64::{
            uint8x16x2_t, uint8x16x3_t, uint8x16x4_t, vld1q_u8, vqtbl1q_u8, vqtbl2q_u8, vqtbl3q_u8,
            vqtbl4q_u8, vst1q_u8,
        };
        let sp = row.as_ptr();
        for chunk in &self.chunks {
            // SAFETY: the plan keeps start + 16 tables inside the row and
            // out_at + 16 inside the output row.
            unsafe {
                let window = |table: usize| vld1q_u8(sp.add(chunk.start as usize + CHUNK * table));
                let index = vld1q_u8(chunk.index.as_ptr());
                let bytes = match chunk.tables {
                    1 => vqtbl1q_u8(window(0), index),
                    2 => vqtbl2q_u8(uint8x16x2_t(window(0), window(1)), index),
                    3 => vqtbl3q_u8(uint8x16x3_t(window(0), window(1), window(2)), index),
                    _ => vqtbl4q_u8(
                        uint8x16x4_t(window(0), window(1), window(2), window(3)),
                        index,
                    ),
                };
                vst1q_u8(dst.add(chunk.out_at as usize), bytes);
            }
        }
    }

    /// The odd pixel of every source pair (1- or 2-byte pixels): `vld2`'s
    /// second plane, 16 output bytes from 32 source bytes, the last step
    /// shifted back to end at the row's end.
    ///
    /// # Safety
    /// `row` holds at least twice the output row's bytes; `dst` holds an
    /// output row of at least 32 bytes.
    #[target_feature(enable = "neon")]
    unsafe fn row_odd_pixels_neon(&self, row: &[u8], dst: *mut u8) {
        use std::arch::aarch64::{vld2q_u8, vld2q_u16, vreinterpretq_u8_u16, vst1q_u8};
        let out_row = self.cols.len() * self.pixel_bytes;
        debug_assert!(out_row >= 32 && row.len() >= 2 * out_row);
        let sp = row.as_ptr();
        let step = |at: usize| {
            // SAFETY: at + 16 <= out_row: source bytes 2 at .. 2 at + 32 and
            // output bytes at .. at + 16.
            unsafe {
                let odd = if self.pixel_bytes == 1 {
                    vld2q_u8(sp.add(2 * at)).1
                } else {
                    vreinterpretq_u8_u16(vld2q_u16(sp.add(2 * at).cast::<u16>()).1)
                };
                vst1q_u8(dst.add(at), odd);
            }
        };
        let mut at = 0;
        while at + CHUNK <= out_row {
            step(at);
            at += CHUNK;
        }
        if at < out_row {
            step(out_row - CHUNK);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::RGBX;
    use crate::testing::{CHANNELS, RandomSample, Rng, random_samples};

    /// Every pixel size the kernels take: 1, 3 or 4 samples of 1, 2 or 4
    /// bytes.
    const PIXEL_BYTES: [usize; 8] = [1, 2, 3, 4, 6, 8, 12, 16];

    /// The source column of every output pixel, as `resize_nearest` places
    /// them.
    fn source_columns(width: usize, ow: usize) -> Vec<u32> {
        let sx = width as f64 / ow as f64;
        (0..ow)
            .map(|x| {
                let column = ((sx * 0.5 + sx * x as f64) as usize).min(width - 1);
                u32::try_from(column).expect("test widths fit u32")
            })
            .collect()
    }

    /// The output row gathered a pixel at a time from a byte slice.
    fn gather_reference(row: &[u8], cols: &[u32], pixel_bytes: usize) -> Vec<u8> {
        cols.iter()
            .flat_map(|&col| &row[col as usize * pixel_bytes..][..pixel_bytes])
            .copied()
            .collect()
    }

    /// The paths a planned row gather takes (`two_windows`: two or more).
    #[derive(Default)]
    struct PathsTaken {
        odd_pixels: bool,
        one_window: bool,
        two_windows: bool,
        by_pixel: bool,
    }

    /// Plans the gather of `ow` pixels from a `width`-pixel row, checks the
    /// plan's windows and stores stay inside the rows, and compares its
    /// output (and the per-pixel path's) with the reference.
    fn assert_row_gather(
        rng: &mut Rng,
        (width, ow): (usize, usize),
        pixel_bytes: usize,
        paths: &mut PathsTaken,
    ) {
        let cols = source_columns(width, ow);
        let (in_row, out_row) = (width * pixel_bytes, ow * pixel_bytes);
        // As many rows as repay any plan that pays.
        let gather = RowGather::new(&cols, pixel_bytes, in_row, usize::MAX);
        let context = format!("{width} -> {ow} pixels of {pixel_bytes} bytes");
        for chunk in &gather.chunks {
            assert!(chunk.out_at as usize + CHUNK <= out_row, "{context}");
            #[cfg(target_arch = "x86_64")]
            let windows = {
                let windows = if chunk.two_windows { 2 } else { 1 };
                for &window in &chunk.windows[..windows] {
                    assert!(window as usize + CHUNK <= in_row, "{context}");
                }
                windows
            };
            #[cfg(target_arch = "aarch64")]
            let windows = {
                let tables = usize::from(chunk.tables);
                assert!(chunk.start as usize + tables * CHUNK <= in_row, "{context}");
                assert!(
                    chunk
                        .index
                        .iter()
                        .all(|&index| usize::from(index) < tables * CHUNK),
                    "{context}"
                );
                tables
            };
            paths.two_windows |= windows > 1;
            paths.one_window |= windows == 1;
        }
        paths.odd_pixels |= gather.odd_pixels;
        paths.by_pixel |= !gather.odd_pixels && gather.chunks.is_empty();
        if gather.odd_pixels {
            assert!(2 * out_row <= in_row, "{context}");
        }
        let row: Vec<u8> = random_samples(rng, in_row);
        let expected = gather_reference(&row, &cols, pixel_bytes);
        let mut planned = vec![0u8; out_row];
        let mut by_pixel = vec![0u8; out_row];
        // SAFETY: `row` is the source row the plan was made for; each output
        // holds an output row.
        unsafe {
            gather.row(&row, planned.as_mut_ptr());
            gather.row_by_pixel(&row, by_pixel.as_mut_ptr());
        }
        assert_eq!(planned, expected, "{context}");
        assert_eq!(by_pixel, expected, "{context}");
    }

    #[test]
    fn row_gathers_equal_a_pixel_by_pixel_copy() {
        const WIDTHS: [usize; 15] = [1, 2, 3, 5, 8, 15, 16, 17, 31, 32, 33, 64, 100, 257, 1000];
        const OUT_WIDTHS: [usize; 20] = [
            1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 50, 64, 100, 128, 129, 300, 500, 514,
        ];
        let mut rng = Rng::new(50);
        let mut paths = PathsTaken::default();
        for pixel_bytes in PIXEL_BYTES {
            for width in WIDTHS {
                for ow in OUT_WIDTHS {
                    assert_row_gather(&mut rng, (width, ow), pixel_bytes, &mut paths);
                }
            }
        }
        assert!(paths.by_pixel);
        if simd_gather() {
            assert!(paths.odd_pixels && paths.one_window && paths.two_windows);
        }
    }

    #[test]
    fn exact_halvings_of_small_pixels_take_the_odd_pixels() {
        if !simd_gather() {
            return;
        }
        let mut rng = Rng::new(51);
        for pixel_bytes in [1, 2] {
            // From the smallest planned output row (32 bytes) through
            // several 32-byte steps and a shifted last one.
            for ow in (32 / pixel_bytes)..(200 / pixel_bytes) {
                let mut paths = PathsTaken::default();
                assert_row_gather(&mut rng, (2 * ow, ow), pixel_bytes, &mut paths);
                assert!(paths.odd_pixels, "{ow} pixels of {pixel_bytes} bytes");
            }
        }
    }

    /// The nearest resize straight from its definition: output pixel (x, y)
    /// is source pixel (`source_columns`, the y-th accumulated row), and
    /// rows past the image's last row are zero.
    fn resize_reference<T: Copy + Default>(
        src: &[T],
        (width, height): (usize, usize),
        (ow, oh): (usize, usize),
        channels: usize,
    ) -> Vec<T> {
        let cols = source_columns(width, ow);
        let sy = height as f64 / oh as f64;
        let mut out = vec![T::default(); ow * oh * channels];
        let mut source_y = sy * 0.5;
        for out_row in out.chunks_exact_mut(ow * channels) {
            if source_y >= height as f64 {
                break;
            }
            let row = &src[source_y as usize * width * channels..][..width * channels];
            for (pixel, &col) in out_row.chunks_exact_mut(channels).zip(&cols) {
                pixel.copy_from_slice(&row[col as usize * channels..][..channels]);
            }
            source_y += sy;
        }
        out
    }

    const RESIZE_WIDTHS: [usize; 6] = [1, 3, 16, 17, 33, 100];
    const RESIZE_OUT_WIDTHS: [usize; 8] = [1, 2, 7, 16, 33, 50, 129, 300];
    const RESIZE_HEIGHTS: [(usize, usize); 5] = [(1, 1), (3, 8), (8, 3), (17, 5), (4, 9)];

    fn assert_resizes_match<T: RandomSample + bytemuck::Pod + Default>() {
        let mut rng = Rng::new(52);
        for channels in CHANNELS {
            for width in RESIZE_WIDTHS {
                for ow in RESIZE_OUT_WIDTHS {
                    for (height, oh) in RESIZE_HEIGHTS {
                        let src: Vec<T> = random_samples(&mut rng, width * height * channels);
                        let context = format!("{width} x {height} -> {ow} x {oh} x {channels}");
                        let out = resize_nearest(&src, (width, height), (ow, oh), channels);
                        let expected = resize_reference(&src, (width, height), (ow, oh), channels);
                        assert_eq!(out, expected, "{context}");
                    }
                }
            }
        }
    }

    #[test]
    fn resize_nearest_equals_its_definition() {
        assert_resizes_match::<u8>();
        assert_resizes_match::<u16>();
        assert_resizes_match::<f32>();
    }

    #[test]
    fn large_resizes_prefault_and_copy_repeated_rows() {
        // An output past the prefault threshold, upscaled vertically so rows
        // repeat (copied from the row above) and within bands.
        let (width, height) = (1000, 300);
        let (ow, oh) = (1500, 1500);
        assert!(ow * oh * RGBX >= crate::cpu::PREFAULT_OUTPUT_BYTES);
        let src: Vec<u8> = random_samples(&mut Rng::new(53), width * height * RGBX);
        assert_eq!(
            resize_nearest(&src, (width, height), (ow, oh), RGBX),
            resize_reference(&src, (width, height), (ow, oh), RGBX)
        );
    }
}

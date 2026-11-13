//! AVX2 kernels of the native u8 convolution resize, planar and
//! interleaved (see `crate::convolve` for the passes, plans and the scalar
//! reference).
//!
//! # Safety
//! Every kernel requires the CPU features in its `#[target_feature]` (its
//! dispatch checks them at runtime). The register-only helpers are safe
//! `fn`s; the kernels that load and store are `unsafe fn`s whose `# Safety`
//! section gives the bounds the caller upholds, and each load and store is
//! justified against those bounds at its loop guard.

use std::arch::x86_64::__m256i;
use std::ops::Range;

use crate::convolve::{
    AVX2_PIXEL_OUTPUTS, AVX2_PIXEL_SPAN, MAX_PADDED_TAPS, PIXEL_OUTPUTS, PaddedWindows,
    PixelBlocks, RGB, RGBA, Runs, TRANSPOSE_ROWS, Windows,
};
use crate::x86::*;

/// Bytes of a 256-bit register.
const YMM_BYTES: usize = 32;
/// `permute2x128` immediates: the two sources' low 128-bit lanes, and their
/// high ones.
const LOW_LANES: i32 = 0x20;
const HIGH_LANES: i32 = 0x31;
/// `permute4x64` immediate gathering qwords 0 and 2 (each 128-bit lane's
/// low qword) into the low 128 bits.
const LOW_QWORD_PER_LANE: i32 = 0b10_00;

/// Q6 sums rounded to integers, `(acc + 32) >> 6`, unclamped (a `packus`
/// clamps them to u8).
#[target_feature(enable = "avx2")]
#[inline]
fn round_q6_avx2(acc: __m256i) -> __m256i {
    use std::arch::x86_64::*;
    _mm256_srai_epi16::<{ CONVOLVE_FRAC_BITS }>(_mm256_add_epi16(
        acc,
        _mm256_set1_epi16(half_ulp(CONVOLVE_FRAC_BITS)),
    ))
}

/// The `STEP` outputs (indices into `outputs`, which is not empty) a run
/// kernel computes from `i0`, and whether they are `i0..i0 + STEP`
/// themselves. A run that is short (the row's end) or broken (a scalar
/// column) repeats `i0`, and the kernel then stores only `i0` and moves on
/// by one.
fn run_step<const STEP: usize>(outputs: &[usize], i0: usize) -> ([usize; STEP], bool) {
    let len = outputs.len();
    let idx = std::array::from_fn(|offset| {
        let index = (i0 + offset).min(len - 1);
        if outputs[index] - outputs[i0] == offset {
            index
        } else {
            i0
        }
    });
    let consecutive = (1..STEP).all(|offset| idx[offset] == i0 + offset);
    (idx, consecutive)
}

/// Stores a run step's four RGBA pixels `px` (a dword each, see `run_step`):
/// at once where the outputs are consecutive, else one by one.
///
/// # Safety
/// `outputs[index]` is a pixel of `dst` (`(outputs[index] + 1) * RGBA <=
/// dst.len()`) for every index in `step_outputs`, and with `consecutive` those
/// four are consecutive pixels (as `run_step` reports).
unsafe fn store_step_pixels(
    dst: &mut [u8],
    outputs: &[usize],
    step_outputs: [usize; 4],
    consecutive: bool,
    px: std::arch::x86_64::__m128i,
) {
    use std::arch::x86_64::*;
    if consecutive {
        // SAFETY: the four consecutive pixels from `outputs[step_outputs[0]]`
        // lie in `dst` (contract).
        unsafe {
            _mm_storeu_si128(
                dst.as_mut_ptr().add(outputs[step_outputs[0]] * RGBA) as *mut __m128i,
                px,
            );
        }
    } else {
        let mut four = [0u32; 4];
        // SAFETY: `four` is the 16 bytes stored.
        unsafe { _mm_storeu_si128(four.as_mut_ptr() as *mut __m128i, px) };
        for (slot, &index) in step_outputs.iter().enumerate() {
            // SAFETY: `outputs[index]` is a pixel of `dst` (contract).
            unsafe {
                (dst.as_mut_ptr().add(outputs[index] * RGBA) as *mut u32)
                    .write_unaligned(four[slot]);
            }
        }
    }
}

/// The RGBX-to-RGB pack's tables: per 128-bit lane, the byte shuffle moving
/// its 4 pixels' RGB bytes to the front, and the dword indices joining the
/// two lanes' packed bytes.
const fn pack_tables() -> ([i8; YMM_BYTES], [i32; 8]) {
    let channels = RGB;
    let mut lane_bytes = [-1i8; YMM_BYTES];
    let mut dword_idx = [0i32; 8];
    let mut lane = 0;
    while lane < 2 {
        let mut byte = 0;
        while byte < RGBA * channels {
            lane_bytes[16 * lane + byte] = (RGBA * (byte / channels) + byte % channels) as i8;
            byte += 1;
        }
        let mut dword = 0;
        while dword < channels {
            dword_idx[channels * lane + dword] = (4 * lane + dword) as i32;
            dword += 1;
        }
        lane += 1;
    }
    (lane_bytes, dword_idx)
}

/// `pack_tables`, built at compile time: the pack runs once per output row.
const PACK_TABLES: ([i8; YMM_BYTES], [i32; 8]) = pack_tables();

/// The AVX2 widening's in-lane byte shuffle (2 pixels per 128-bit lane).
const fn widen_shuffle_ymm() -> [i8; 32] {
    let channels = RGB;
    let mut shuffle = [-1i8; 32];
    let mut lane = 0;
    while lane < 2 {
        let mut word = 0;
        while word < 8 {
            let (pixel, channel) = (word / RGBA, word % RGBA);
            if channel < channels {
                let from = 2 * (channels * pixel + channel);
                shuffle[16 * lane + 2 * word] = from as i8;
                shuffle[16 * lane + 2 * word + 1] = (from + 1) as i8;
            }
            word += 1;
        }
        lane += 1;
    }
    shuffle
}
const WIDEN_SHUFFLE_YMM: [i8; 32] = widen_shuffle_ymm();

// ---- AVX2 tier ----------------------------------------------------------------

/// The RGBA horizontal pass on AVX2, two rows at a time, by blocks of up to
/// 4 outputs whose first taps lie within 8 pixels (`AVX2_PIXEL_OUTPUTS`):
/// per tap the window of 8 pixels from `base + t` is two registers of 4
/// pixels (a pixel = 4 x i16 = 2 dwords), and two dword permutes and a
/// blend (fixed per block) gather pixel `start_j + t` of every output j —
/// no transposes, which are a large share of the transposed path's work.
///
/// # Safety
/// Every block holds at most `AVX2_PIXEL_OUTPUTS` outputs, and for every
/// block and both rows `(block.base + block.taps.len() + AVX2_PIXEL_SPAN) *
/// RGBA <= mid.len()` and `(block.o0 + block.count) * RGBA <= dst.len()`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn horizontal_pixels4_x2_avx2(
    mids: [&[i16]; 2],
    dsts: [&mut [u8]; 2],
    pixels: &PixelBlocks,
) {
    use std::arch::x86_64::*;
    let [dst0, dst1] = dsts;
    for block in &pixels.blocks {
        debug_assert!(
            mids.iter()
                .all(|mid| (block.base + block.taps.len() + AVX2_PIXEL_SPAN) * RGBA <= mid.len())
        );
        debug_assert!((block.o0 + block.count) * RGBA <= dst0.len().min(dst1.len()));
        // Output slot's pixel: dwords 2 pixel, 2 pixel + 1 of the low
        // (pixel < 4) or high register.
        let mut dwords = [0i32; 8];
        let mut high = [0i32; 8];
        for slot in 0..AVX2_PIXEL_OUTPUTS {
            let pixel = block.idx[slot] as i32;
            dwords[2 * slot] = 2 * (pixel % 4);
            dwords[2 * slot + 1] = 2 * (pixel % 4) + 1;
            high[2 * slot] = if pixel >= 4 { -1 } else { 0 };
            high[2 * slot + 1] = high[2 * slot];
        }
        // SAFETY: `dwords` and `high` are the 8 lanes loaded each.
        let (dwords, high) = unsafe {
            (
                _mm256_loadu_si256(dwords.as_ptr() as *const __m256i),
                _mm256_loadu_si256(high.as_ptr() as *const __m256i),
            )
        };
        let gather = |window: *const i16| {
            // SAFETY: every caller passes the window of a tap t <
            // block.taps.len(), pixels base + t .. base + t + AVX2_PIXEL_SPAN
            // of its row, within the row (contract).
            let (lo, hi) = unsafe {
                (
                    _mm256_loadu_si256(window as *const __m256i),
                    _mm256_loadu_si256(window.add(16) as *const __m256i),
                )
            };
            let lo = _mm256_permutevar8x32_epi32(lo, dwords);
            let hi = _mm256_permutevar8x32_epi32(hi, dwords);
            _mm256_blendv_epi8(lo, hi, high)
        };
        // SAFETY: the block's windows start in both mids (contract).
        let (mut win0, mut win1) = unsafe {
            (
                mids[0].as_ptr().add(block.base * RGBA),
                mids[1].as_ptr().add(block.base * RGBA),
            )
        };
        let (mut acc0, mut acc1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
        for weights in &pixels.weights[block.taps.clone()] {
            // SAFETY: a weight row holds the 16 lanes loaded.
            let wv = unsafe { _mm256_loadu_si256(weights.as_ptr() as *const __m256i) };
            acc0 = _mm256_add_epi16(acc0, _mm256_mulhrs_epi16(gather(win0), wv));
            acc1 = _mm256_add_epi16(acc1, _mm256_mulhrs_epi16(gather(win1), wv));
            // SAFETY: at most one pixel past the last tap's window start,
            // still within the rows (contract).
            unsafe {
                win0 = win0.add(RGBA);
                win1 = win1.add(RGBA);
            }
        }
        for (acc, dst) in [(acc0, &mut *dst0), (acc1, &mut *dst1)] {
            // packus clamps to 0..255; the lanes' low halves hold the 4 pixels.
            let rounded = round_q6_avx2(acc);
            let bytes = _mm256_castsi256_si128(_mm256_permute4x64_epi64::<LOW_QWORD_PER_LANE>(
                _mm256_packus_epi16(rounded, rounded),
            ));
            // SAFETY: the block's block.count <= AVX2_PIXEL_OUTPUTS output
            // pixels from block.o0 lie in `dst` (contract).
            unsafe {
                let at = dst.as_mut_ptr().add(block.o0 * RGBA);
                if block.count == AVX2_PIXEL_OUTPUTS {
                    _mm_storeu_si128(at as *mut __m128i, bytes);
                } else {
                    let mut px = [0u8; 16];
                    _mm_storeu_si128(px.as_mut_ptr() as *mut __m128i, bytes);
                    std::ptr::copy_nonoverlapping(px.as_ptr(), at, block.count * RGBA);
                }
            }
        }
    }
}

/// The horizontal pass by tap runs (`TapRuns`), two rows and four outputs
/// per step: each 8-pixel weight group as two registers of 4 pixels, one
/// unit-stride load per row times the output's weights (loaded once for both
/// rows); per output a register of 4 pixel lanes, summed to one pixel for
/// four outputs at once (i16 wraparound sums: the same result as the per-tap
/// order).
///
/// # Safety
/// Both mids hold `runs.starts[i] + runs.groups * PIXEL_OUTPUTS` pixels for
/// every i (the runs' overreach past the row), and every output in
/// `runs.outputs` is a pixel of both dsts.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn horizontal_taps_x2_avx2(mids: [&[i16]; 2], dsts: [&mut [u8]; 2], runs: &Runs) {
    use std::arch::x86_64::*;
    // A group's pixels as two registers of 4.
    const HALF_GROUP_PIXELS: usize = PIXEL_OUTPUTS / 2;
    let output_count = runs.outputs.len();
    if output_count == 0 {
        return;
    }
    let [dst0, dst1] = dsts;
    // Registers of 4 pixels a, b, c, d -> one register of their 4 sums.
    let reduce = |acc: [__m256i; 4]| {
        let pair = |x: __m256i, y: __m256i| {
            _mm256_add_epi16(_mm256_unpacklo_epi64(x, y), _mm256_unpackhi_epi64(x, y))
        };
        let (ab, cd) = (pair(acc[0], acc[1]), pair(acc[2], acc[3]));
        _mm256_add_epi16(
            _mm256_permute2x128_si256::<LOW_LANES>(ab, cd),
            _mm256_permute2x128_si256::<HIGH_LANES>(ab, cd),
        )
    };
    let mut i0 = 0;
    while i0 < output_count {
        let (step_outputs, consecutive) = run_step::<4>(&runs.outputs, i0);
        let mut acc = [[_mm256_setzero_si256(); 4]; 2];
        for group in 0..runs.groups {
            for (slot, &index) in step_outputs.iter().enumerate() {
                let weights_ptr = runs.weights[index * runs.groups + group].as_ptr();
                for half in 0..2 {
                    // SAFETY: half < 2: the half's 16 lanes lie in the 32-lane
                    // weight row.
                    let wv = unsafe {
                        _mm256_loadu_si256(
                            weights_ptr.add(half * HALF_GROUP_PIXELS * RGBA) as *const __m256i
                        )
                    };
                    let at =
                        (runs.starts[index] + group * PIXEL_OUTPUTS + half * HALF_GROUP_PIXELS)
                            * RGBA;
                    for (row, mid) in mids.iter().enumerate() {
                        // SAFETY: group < runs.groups: the group's
                        // PIXEL_OUTPUTS pixels lie in the mid (contract).
                        let samples =
                            unsafe { _mm256_loadu_si256(mid.as_ptr().add(at) as *const __m256i) };
                        acc[row][slot] =
                            _mm256_add_epi16(acc[row][slot], _mm256_mulhrs_epi16(samples, wv));
                    }
                }
            }
        }
        for (acc, dst) in acc.iter().zip([&mut *dst0, &mut *dst1]) {
            let sums = round_q6_avx2(reduce(*acc));
            // [a b | c d] as bytes per lane (packus clamping) -> a b c d in the
            // low 16 bytes.
            let packed =
                _mm256_permute4x64_epi64::<LOW_QWORD_PER_LANE>(_mm256_packus_epi16(sums, sums));
            // SAFETY: the step's outputs are pixels of `dst` (contract), and
            // `run_step` reports whether they are consecutive.
            unsafe {
                store_step_pixels(
                    dst,
                    &runs.outputs,
                    step_outputs,
                    consecutive,
                    _mm256_castsi256_si128(packed),
                );
            }
        }
        i0 += if consecutive { 4 } else { 1 };
    }
}

/// `rgbx[4p + c] = src[3p + c]` for c < 3, X 0: 4 pixels per register, 2
/// per 128-bit lane by an in-lane byte shuffle. `src` is read up to 8 samples past.
///
/// # Safety
/// With `groups = pixels.next_multiple_of(4)`: `src` holds `RGB * groups + 8`
/// samples and `rgbx` holds `RGBA * groups`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn widen_to_rgbx_avx2(src: &[i16], pixels: usize, rgbx: &mut [i16]) {
    use std::arch::x86_64::*;
    let channels = RGB;
    debug_assert!(
        src.len() >= channels * pixels.next_multiple_of(4) + 8
            && rgbx.len() >= RGBA * pixels.next_multiple_of(4)
    );
    // SAFETY: the table is the 32 bytes loaded.
    let shuffle = unsafe { _mm256_loadu_si256(WIDEN_SHUFFLE_YMM.as_ptr() as *const __m256i) };
    for pixel in (0..pixels).step_by(4) {
        // SAFETY: pixel + 4 <= groups: the two 8-sample halves read from
        // `src` and the register written to `rgbx` lie in them (contract).
        unsafe {
            let at = src.as_ptr().add(channels * pixel);
            let samples =
                _mm256_loadu2_m128i(at.add(2 * channels) as *const __m128i, at as *const __m128i);
            _mm256_storeu_si256(
                rgbx.as_mut_ptr().add(RGBA * pixel) as *mut __m256i,
                _mm256_shuffle_epi8(samples, shuffle),
            );
        }
    }
}

/// `out[3p + c] = rgbx[4p + c]` for c < 3: 8 pixels per in-lane byte
/// shuffle and dword permute, scalar for the tail.
///
/// # Safety
/// `rgbx` holds `RGBA * (out.len() / RGB)` bytes (the stores to `out` are
/// bounded here).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn pack_from_rgbx_avx2(rgbx: &[u8], out: &mut [u8]) {
    use std::arch::x86_64::*;
    let channels = RGB;
    let pixels = out.len() / channels;
    debug_assert!(rgbx.len() >= RGBA * pixels);
    let (lane_bytes, dword_idx) = &PACK_TABLES;
    // SAFETY: the tables hold the 32 bytes loaded each.
    let (shuffle, dwords) = unsafe {
        (
            _mm256_loadu_si256(lane_bytes.as_ptr() as *const __m256i),
            _mm256_loadu_si256(dword_idx.as_ptr() as *const __m256i),
        )
    };
    let whole = pixels / 8 * 8;
    for pixel in (0..whole).step_by(8) {
        // SAFETY: pixel + 8 <= pixels: the 8 pixels lie in `rgbx`
        // (contract).
        let bytes =
            unsafe { _mm256_loadu_si256(rgbx.as_ptr().add(RGBA * pixel) as *const __m256i) };
        let packed = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(bytes, shuffle), dwords);
        // SAFETY: pixel + 8 <= out.len() / RGB: the 16 + 8 bytes stored lie
        // in `out`.
        unsafe {
            let dst = out.as_mut_ptr().add(channels * pixel);
            _mm_storeu_si128(dst as *mut __m128i, _mm256_castsi256_si128(packed));
            _mm_storel_epi64(
                dst.add(16) as *mut __m128i,
                _mm256_extracti128_si256::<1>(packed),
            );
        }
    }
    for pixel in whole..pixels {
        out[channels * pixel..channels * (pixel + 1)]
            .copy_from_slice(&rgbx[RGBA * pixel..RGBA * pixel + channels]);
    }
}

/// Vertical pass: `mid[x] = sum_t weights[t] * rows[t][x]`, Q6 (u8 << 6) x
/// Q15 via mulhrs, i16 sums, 16 samples per register, a whole cache line per tap where the row allows, the taps in
/// groups where the stride aliases (`groups_taps`).
///
/// # Safety
/// `rows` holds `weights.len()` rows of `width` samples, `width >= 16`, and
/// `mid` holds at least `width` samples.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vertical_q6_avx2(rows: &[u8], width: usize, weights: &[i16], mid: &mut [i16]) {
    use std::arch::x86_64::*;
    const BLOCK: usize = 16;
    // A line: a whole cache line of samples, four registers.
    const LINE: usize = 4 * BLOCK;
    debug_assert!(rows.len() >= weights.len() * width && mid.len() >= width && width >= BLOCK);
    let vq: Vec<__m256i> = weights
        .iter()
        .map(|&weight| _mm256_set1_epi16(weight))
        .collect();
    let (src_ptr, mid_ptr) = (rows.as_ptr(), mid.as_mut_ptr());
    let widen = |sample_ptr: *const u8| {
        // SAFETY: every caller passes the start of `BLOCK` samples of one row
        // in `rows` (bounded at its guard below).
        let bytes = unsafe { _mm_loadu_si128(sample_ptr as *const __m128i) };
        _mm256_slli_epi16::<{ CONVOLVE_FRAC_BITS }>(_mm256_cvtepu8_epi16(bytes))
    };
    let block = |x: usize| {
        let mut acc = _mm256_setzero_si256();
        for (tap, vq) in vq.iter().enumerate() {
            // SAFETY: tap < weights.len() and `cover_row` keeps x + BLOCK <=
            // width: the block lies in the tap's row of `rows` (contract).
            let tap_ptr = unsafe { src_ptr.add(tap * width + x) };
            acc = _mm256_add_epi16(acc, _mm256_mulhrs_epi16(widen(tap_ptr), *vq));
        }
        // SAFETY: x + BLOCK <= width <= mid.len() (contract).
        unsafe { _mm256_storeu_si256(mid_ptr.add(x) as *mut __m256i, acc) };
    };
    // Taps `taps` of the line at x, added to the line's sums so far.
    let line = |x: usize, taps: Range<usize>| {
        if taps.start == 0 {
            prefetch_rows_after_window(src_ptr, weights.len(), width, x);
        }
        let mut acc = [_mm256_setzero_si256(); LINE / BLOCK];
        if taps.start > 0 {
            for (register, acc) in acc.iter_mut().enumerate() {
                // SAFETY: every line ends by `width` (x + LINE <= width <=
                // mid.len(), contract).
                *acc = unsafe {
                    _mm256_loadu_si256(mid_ptr.add(x + register * BLOCK) as *const __m256i)
                };
            }
        }
        for tap in taps {
            for (register, acc) in acc.iter_mut().enumerate() {
                // SAFETY: tap < weights.len() and x + LINE <= width: the
                // register's samples lie in the tap's row of `rows`
                // (contract).
                let register_ptr = unsafe { src_ptr.add(tap * width + x + register * BLOCK) };
                *acc = _mm256_add_epi16(*acc, _mm256_mulhrs_epi16(widen(register_ptr), vq[tap]));
            }
        }
        for (register, acc) in acc.iter().enumerate() {
            // SAFETY: x + LINE <= width <= mid.len() (contract).
            unsafe { _mm256_storeu_si256(mid_ptr.add(x + register * BLOCK) as *mut __m256i, *acc) };
        }
    };
    if width < LINE {
        return cover_row(width, BLOCK, block);
    }
    if groups_taps(width, weights.len()) {
        let lines = 0..width / LINE * LINE;
        grouped_taps(lines.clone(), LINE, 2, weights.len(), &line);
        if lines.end < width {
            line(width - LINE, 0..weights.len());
        }
    } else {
        cover_row(width, LINE, |x| line(x, 0..weights.len()));
    }
}

/// Transpose a 16x16 tile of i16: `rows[i]` = row i on entry, column i on
/// exit. An 8x8 transpose per 128-bit lane (unpack 16/32/64), then lane
/// pairing.
#[target_feature(enable = "avx2")]
#[inline]
fn transpose_16x16_i16(rows: &mut [__m256i; 16]) {
    use std::arch::x86_64::*;
    let lo = transpose_8x8_i16_lanes(&rows[..8]);
    let hi = transpose_8x8_i16_lanes(&rows[8..]);
    for column in 0..8 {
        rows[column] = _mm256_permute2x128_si256::<LOW_LANES>(lo[column], hi[column]);
        rows[column + 8] = _mm256_permute2x128_si256::<HIGH_LANES>(lo[column], hi[column]);
    }
}

/// Transposes 8 rows per 128-bit lane: `columns[j]` = [column j | column
/// j + 8].
#[target_feature(enable = "avx2")]
#[inline]
fn transpose_8x8_i16_lanes(rows: &[__m256i]) -> [__m256i; 8] {
    use std::arch::x86_64::*;
    let mut pairs = [_mm256_setzero_si256(); 8];
    for pair in 0..4 {
        pairs[2 * pair] = _mm256_unpacklo_epi16(rows[2 * pair], rows[2 * pair + 1]);
        pairs[2 * pair + 1] = _mm256_unpackhi_epi16(rows[2 * pair], rows[2 * pair + 1]);
    }
    // quads[0..4]: cols (0,1) (2,3) (4,5) (6,7) of rows 0-3; quads[4..8]: rows 4-7.
    let mut quads = [_mm256_setzero_si256(); 8];
    for group in 0..2 {
        for half in 0..2 {
            let (upper, lower) = (pairs[4 * group + half], pairs[4 * group + 2 + half]);
            quads[4 * group + 2 * half] = _mm256_unpacklo_epi32(upper, lower);
            quads[4 * group + 2 * half + 1] = _mm256_unpackhi_epi32(upper, lower);
        }
    }
    let mut columns = [_mm256_setzero_si256(); 8];
    for quad in 0..4 {
        columns[2 * quad] = _mm256_unpacklo_epi64(quads[quad], quads[4 + quad]);
        columns[2 * quad + 1] = _mm256_unpackhi_epi64(quads[quad], quads[4 + quad]);
    }
    columns
}

/// `TRANSPOSE_ROWS` (= 16) rows of `mid` (stride `wp`, a multiple of 16)
/// into column-major `mid_t[x * 16 + r]`.
///
/// # Safety
/// `wp` is a multiple of `TRANSPOSE_ROWS`, and `mid` and `mid_t` hold
/// `TRANSPOSE_ROWS * wp` samples each.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn transpose_rows_i16_avx2(mid: &[i16], wp: usize, mid_t: &mut [i16]) {
    use std::arch::x86_64::*;
    const ROWS: usize = TRANSPOSE_ROWS;
    debug_assert!(wp.is_multiple_of(ROWS) && mid.len() >= ROWS * wp && mid_t.len() >= ROWS * wp);
    for x0 in (0..wp).step_by(ROWS) {
        let mut tile = [_mm256_setzero_si256(); ROWS];
        for (row, register) in tile.iter_mut().enumerate() {
            // SAFETY: row < ROWS and x0 + ROWS <= wp (wp a multiple of ROWS):
            // within the ROWS * wp samples of `mid` (contract).
            *register =
                unsafe { _mm256_loadu_si256(mid.as_ptr().add(row * wp + x0) as *const __m256i) };
        }
        transpose_16x16_i16(&mut tile);
        for (column, register) in tile.iter().enumerate() {
            // SAFETY: x0 + column < wp: within the wp * ROWS samples of
            // `mid_t` (contract).
            unsafe {
                _mm256_storeu_si256(
                    mid_t.as_mut_ptr().add((x0 + column) * ROWS) as *mut __m256i,
                    *register,
                );
            }
        }
    }
}

/// The horizontal pass on transposed rows: for every output whose weights
/// fit Q15 and each of its `channels` samples c, with `s(x, c) = x *
/// channels + c`: `out_t[s(o, c) * 16 + r] = sum_t mulhrs(mid_t[s(start + t,
/// c) * 16 + r], q15[t])` for all 16 (`TRANSPOSE_ROWS`) rows at once — a
/// unit-stride load per tap, any ratio.
///
/// # Safety
/// For every output o whose window fits Q15, `col.end() * channels *
/// TRANSPOSE_ROWS <= mid_t.len()` and `(o + 1) * channels * TRANSPOSE_ROWS
/// <= out_t.len()`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn horizontal_transposed_avx2(
    mid_t: &[i16],
    cols: &Windows,
    channels: usize,
    out_t: &mut [i16],
) {
    use std::arch::x86_64::*;
    const ROWS: usize = TRANSPOSE_ROWS;
    for (output, col) in cols.iter().enumerate() {
        if !col.fits_q15() {
            continue;
        }
        debug_assert!(
            col.end() * channels * ROWS <= mid_t.len()
                && (output + 1) * channels * ROWS <= out_t.len()
        );
        for channel in 0..channels {
            let mut acc = _mm256_setzero_si256();
            for (tap, &weight) in col.q15.iter().enumerate() {
                let x = (col.start + tap) * channels + channel;
                // SAFETY: start + tap < col.end() and channel < channels: the
                // ROWS samples lie in `mid_t` (contract).
                let samples =
                    unsafe { _mm256_loadu_si256(mid_t.as_ptr().add(x * ROWS) as *const __m256i) };
                acc =
                    _mm256_add_epi16(acc, _mm256_mulhrs_epi16(samples, _mm256_set1_epi16(weight)));
            }
            // SAFETY: channel < channels: the ROWS sums lie in `out_t`
            // (contract).
            unsafe {
                _mm256_storeu_si256(
                    out_t.as_mut_ptr().add((output * channels + channel) * ROWS) as *mut __m256i,
                    acc,
                );
            }
        }
    }
}

/// `horizontal_transposed_avx2` over padded windows: one loop of the
/// windows' common tap count, fixed per kernel.
///
/// # Safety
/// The CPU supports AVX2, `mid_t` holds every padded window's samples,
/// `(start + taps) * channels * 16`, and `out_t` every output's, `outputs *
/// channels * 16`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn horizontal_padded_avx2(
    mid_t: &[i16],
    padded: &PaddedWindows,
    channels: usize,
    out_t: &mut [i16],
) {
    let (starts, weights) = (&padded.starts[..], &padded.weights[..]);
    macro_rules! by_taps {
        ($($taps:literal)*) => {
            match padded.taps {
                $($taps => horizontal_padded_taps_avx2::<$taps>(mid_t, starts, weights, channels, out_t),)*
                _ => horizontal_padded_taps_avx2::<{ MAX_PADDED_TAPS }>(
                    mid_t, starts, weights, channels, out_t,
                ),
            }
        };
    }
    // SAFETY: the contract, passed on.
    unsafe { by_taps!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19) }
}

/// `horizontal_padded_avx2` for `TAPS` taps per window.
///
/// # Safety
/// As `horizontal_padded_avx2`, `weights` holding `TAPS` per start.
#[target_feature(enable = "avx2")]
unsafe fn horizontal_padded_taps_avx2<const TAPS: usize>(
    mid_t: &[i16],
    starts: &[usize],
    weights: &[i16],
    channels: usize,
    out_t: &mut [i16],
) {
    use std::arch::x86_64::*;
    const ROWS: usize = TRANSPOSE_ROWS;
    let (mid_t_ptr, out_t_ptr) = (mid_t.as_ptr(), out_t.as_mut_ptr());
    let windows = starts.iter().zip(weights.as_chunks::<TAPS>().0);
    for (output, (&start, weights)) in windows.enumerate() {
        debug_assert!(
            (start + TAPS) * channels * ROWS <= mid_t.len()
                && (output + 1) * channels * ROWS <= out_t.len()
        );
        let weights: [__m256i; TAPS] = std::array::from_fn(|tap| _mm256_set1_epi16(weights[tap]));
        for channel in 0..channels {
            let mut acc = _mm256_setzero_si256();
            for (tap, &weight) in weights.iter().enumerate() {
                let at = ((start + tap) * channels + channel) * ROWS;
                // SAFETY: inside the padded window's samples (the caller's
                // contract).
                let samples = unsafe { _mm256_loadu_si256(mid_t_ptr.add(at) as *const __m256i) };
                acc = _mm256_add_epi16(acc, _mm256_mulhrs_epi16(samples, weight));
            }
            // SAFETY: inside the outputs' sums (the caller's contract).
            unsafe {
                _mm256_storeu_si256(
                    out_t_ptr.add((output * channels + channel) * ROWS) as *mut __m256i,
                    acc,
                );
            }
        }
    }
}

/// Q6 sums `out_t[o * 16 + r]` back to row-major u8 `out[r * ow + o]` for the
/// first `rows` rows: 16x16 transposes, then rounded, clamped narrow.
///
/// # Safety
/// `rows <= TRANSPOSE_ROWS`, `out_t` holds `ow.next_multiple_of(16) * 16`
/// samples, and `out` holds `rows * ow` bytes.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn untranspose_narrow_avx2(
    out_t: &[i16],
    ow: usize,
    out: &mut [u8],
    rows: usize,
) {
    use std::arch::x86_64::*;
    // Whole cache lines per output row: 16-byte pieces of 16 rows at once
    // would, at power-of-2 row strides (which share L1 sets), evict each
    // line before it is complete.
    const LINE: usize = 64;
    const ROWS: usize = TRANSPOSE_ROWS;
    let mut staged = [[0u8; LINE]; ROWS];
    for l0 in (0..ow).step_by(LINE) {
        for o0 in (l0..(l0 + LINE).min(ow)).step_by(ROWS) {
            debug_assert!((o0 + ROWS) * ROWS <= out_t.len());
            let mut tile = [_mm256_setzero_si256(); ROWS];
            for (output, register) in tile.iter_mut().enumerate() {
                // SAFETY: o0 < ow, a multiple of ROWS: outputs o0..o0 + ROWS
                // lie in `out_t` (contract).
                *register = unsafe {
                    _mm256_loadu_si256(out_t.as_ptr().add((o0 + output) * ROWS) as *const __m256i)
                };
            }
            transpose_16x16_i16(&mut tile);
            for (row, register) in tile.iter().enumerate().take(rows) {
                let rounded = round_q6_avx2(*register);
                let bytes = _mm_packus_epi16(
                    _mm256_castsi256_si128(rounded),
                    _mm256_extracti128_si256::<1>(rounded),
                );
                // SAFETY: o0 - l0 <= LINE - ROWS: the 16 bytes lie in the
                // staged line.
                unsafe {
                    _mm_storeu_si128(staged[row].as_mut_ptr().add(o0 - l0) as *mut __m128i, bytes);
                }
            }
        }
        let len = LINE.min(ow - l0);
        for (row, line) in staged.iter().enumerate().take(rows) {
            // SAFETY: row < rows and l0 + len <= ow: bytes l0..l0 + len of
            // the row lie in `out` (contract); the staged line holds LINE.
            unsafe {
                let dst = out.as_mut_ptr().add(row * ow + l0);
                if len == LINE {
                    let (lo, hi) = line.split_at(LINE / 2);
                    _mm256_storeu_si256(
                        dst as *mut __m256i,
                        _mm256_loadu_si256(lo.as_ptr() as *const __m256i),
                    );
                    _mm256_storeu_si256(
                        dst.add(LINE / 2) as *mut __m256i,
                        _mm256_loadu_si256(hi.as_ptr() as *const __m256i),
                    );
                } else {
                    std::ptr::copy_nonoverlapping(line.as_ptr(), dst, len);
                }
            }
        }
    }
}

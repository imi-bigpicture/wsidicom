//! x86 helpers shared by the exact kernels: covering a row in fixed steps,
//! prefetching the next rows, keeping the even pixels of a row.
//!
//! # Safety
//! `keep_even_pixels` requires AVX2 (its dispatch checks it at runtime) and
//! the row lengths its caller documents; its loads and stores are justified
//! against those lengths at their loop guards.

/// Runs `step_at(o0)` for steps of `step` outputs covering `0..len`
/// (`len >= step`), the last one shifted back to end at `len`. One call
/// site, so a large `step_at` stays inlined.
#[inline(always)]
pub(crate) fn cover_steps(len: usize, step: usize, step_at: &impl Fn(usize)) {
    let mut o0 = 0;
    loop {
        let at = o0.min(len - step);
        step_at(at);
        if at + step >= len {
            break;
        }
        o0 += step;
    }
}

/// Prefetches the line at sample `x` of rows `next` and `next + 1` (`ws`
/// samples apart), the two rows the next output row's window adds, so a
/// tile not in cache streams ahead of the vertical pass. Past the image the
/// prefetch is a no-op (it never faults).
#[inline(always)]
pub(crate) fn prefetch_next_rows(src: &[u8], next: usize, ws: usize, x: usize) {
    use std::arch::x86_64::*;
    let next = src.as_ptr().wrapping_add(next * ws + x);
    // SAFETY: a prefetch reads nothing architecturally and never faults.
    unsafe {
        _mm_prefetch::<_MM_HINT_T0>(next as *const i8);
        _mm_prefetch::<_MM_HINT_T0>(next.wrapping_add(ws) as *const i8);
    }
}

/// `dst` = the even pixels (`channels` samples each) of `full`: byte
/// shuffles, a copy per pixel for the tail.
///
/// # Safety
/// AVX2 must be available, and `full` holds at least `2 * dst.len()` samples
/// (the shuffles read it unchecked; `dst` is bounded by its own length).
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn keep_even_pixels(full: &[u8], channels: usize, dst: &mut [u8]) {
    use std::arch::x86_64::*;
    let ow = dst.len() / channels;
    let (fp, dp) = (full.as_ptr(), dst.as_mut_ptr());
    // Output pixels done by the shuffles.
    let mut output = 0;
    match channels {
        1 => {
            // Per 32 input bytes, 16 output bytes: the even samples.
            let lane: [i8; 16] = [0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1];
            // SAFETY: both halves read the 16 bytes of `lane`.
            let shuffle = unsafe {
                _mm256_loadu2_m128i(
                    lane.as_ptr() as *const __m128i,
                    lane.as_ptr() as *const __m128i,
                )
            };
            while output + 16 <= ow {
                // SAFETY: output + 16 <= ow (the loop guard): the load reads
                // full[2 output..2 output + 32], within 2 ow samples (the
                // caller's contract).
                let input = unsafe { _mm256_loadu_si256(fp.add(2 * output) as *const __m256i) };
                let kept = _mm256_shuffle_epi8(input, shuffle);
                let packed = _mm256_permute4x64_epi64::<0b10_00>(kept);
                // SAFETY: the store writes dst[output..output + 16], within ow
                // by the loop guard.
                unsafe {
                    _mm_storeu_si128(
                        dp.add(output) as *mut __m128i,
                        _mm256_castsi256_si128(packed),
                    );
                }
                output += 16;
            }
        }
        3 => {
            // Per 12 input pixels, 6 output pixels: each 16-byte half from a
            // pixel pair (bytes 0 and 18), its pixels 0, 2 and 4 kept.
            let lane: [i8; 16] = [0, 1, 2, 6, 7, 8, 12, 13, 14, -1, -1, -1, -1, -1, -1, -1];
            // SAFETY: both halves read the 16 bytes of `lane`.
            let shuffle = unsafe {
                _mm256_loadu2_m128i(
                    lane.as_ptr() as *const __m128i,
                    lane.as_ptr() as *const __m128i,
                )
            };
            // 16-byte stores at 9-byte steps: the last must stay in the row.
            while output + 6 <= ow && output * 3 + 9 + 16 <= dst.len() {
                // SAFETY: output + 6 <= ow (the loop guard): the loads read
                // full[6 output..6 output + 34], within 6 ow = 2 dst.len()
                // samples (the caller's contract).
                let input = unsafe {
                    let at = fp.add(2 * output * 3);
                    _mm256_loadu2_m128i(at.add(18) as *const __m128i, at as *const __m128i)
                };
                let kept = _mm256_shuffle_epi8(input, shuffle);
                // SAFETY: the stores write dst[3 output..3 output + 25], within
                // dst.len() by the loop guard.
                unsafe {
                    _mm_storeu_si128(
                        dp.add(output * 3) as *mut __m128i,
                        _mm256_castsi256_si128(kept),
                    );
                    _mm_storeu_si128(
                        dp.add(output * 3 + 9) as *mut __m128i,
                        _mm256_extracti128_si256::<1>(kept),
                    );
                }
                output += 6;
            }
        }
        4 => {
            // Per 8 input pixels, 4 output pixels: the even dwords.
            let even = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
            while output + 4 <= ow {
                // SAFETY: output + 4 <= ow (the loop guard): the load reads
                // full[8 output..8 output + 32], within 8 ow = 2 dst.len()
                // samples (the caller's contract).
                let input = unsafe { _mm256_loadu_si256(fp.add(2 * output * 4) as *const __m256i) };
                let kept = _mm256_permutevar8x32_epi32(input, even);
                // SAFETY: the store writes dst[4 output..4 output + 16], within
                // 4 ow by the loop guard.
                unsafe {
                    _mm_storeu_si128(
                        dp.add(output * 4) as *mut __m128i,
                        _mm256_castsi256_si128(kept),
                    );
                }
                output += 4;
            }
        }
        _ => unreachable!("{channels} channels"),
    }
    for output in output..ow {
        dst[output * channels..(output + 1) * channels]
            .copy_from_slice(&full[2 * output * channels..2 * output * channels + channels]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Rng, random_samples};

    #[test]
    fn cover_steps_covers_every_output_within_bounds() {
        for step in [1, 4, 16] {
            for len in step..step * 5 {
                let starts = std::cell::RefCell::new(Vec::new());
                cover_steps(len, step, &|start| starts.borrow_mut().push(start));
                let mut covered = vec![false; len];
                for start in starts.into_inner() {
                    assert!(start + step <= len, "step {step}, len {len}");
                    covered[start..start + step].fill(true);
                }
                assert!(covered.iter().all(|&done| done), "step {step}, len {len}");
            }
        }
    }

    #[test]
    fn keep_even_pixels_equals_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let mut rng = Rng::new(3);
        for channels in [1, 3, 4] {
            for kept_pixels in 0..80 {
                let full: Vec<u8> = random_samples(&mut rng, 2 * kept_pixels * channels);
                let mut dst = vec![0; kept_pixels * channels];
                // SAFETY: AVX2 is available; full holds 2 dst.len() samples.
                unsafe { keep_even_pixels(&full, channels, &mut dst) };
                let expected: Vec<u8> = full
                    .chunks_exact(2 * channels)
                    .flat_map(|pair| pair[..channels].to_vec())
                    .collect();
                assert_eq!(dst, expected, "{channels} channels, {kept_pixels} pixels");
            }
        }
    }

    #[test]
    fn prefetch_past_the_image_does_not_fault() {
        let src = vec![0u8; 64];
        prefetch_next_rows(&src, 1000, 64, 0);
    }
}

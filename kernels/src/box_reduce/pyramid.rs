//! The power-of-2 box pyramid: chained 2x and fused 4x/8x box passes, band
//! by band for interleaved images, optionally widening RGB to RGBX in the
//! last pass.

use std::borrow::Cow;
use std::mem::MaybeUninit;

use super::{BoxSample, box_levels_rgb_to_rgbx, box2x_rgb_to_rgbx};
use crate::cpu::{unwritten_output, written_output};
use crate::rgbx::expand_image;
use crate::sample::{RGB, RGBX};

/// A sample type's mean of a block from the block's exact sum: integers
/// rounded half up (as the box kernels round), float32 the quotient.
pub(crate) trait BlockMean: Sized {
    fn block_mean(sum: f64, count: usize) -> Self;
}

/// `sum / count` rounded half up, for integer sums (exact in f64).
fn rounded_mean(sum: f64, count: usize) -> u64 {
    let (sum, count) = (sum as u64, count as u64);
    (sum + count / 2) / count
}

impl BlockMean for u8 {
    fn block_mean(sum: f64, count: usize) -> u8 {
        rounded_mean(sum, count) as u8
    }
}

impl BlockMean for u16 {
    fn block_mean(sum: f64, count: usize) -> u16 {
        rounded_mean(sum, count) as u16
    }
}

impl BlockMean for f32 {
    fn block_mean(sum: f64, count: usize) -> f32 {
        (sum / count as f64) as f32
    }
}

/// The `2^times` box reduction at `ceil` size: the pyramid of the whole
/// blocks, and where the factor doesn't divide the image, a last column and
/// row of partial blocks, each the mean of the pixels it covers.
pub(crate) fn box_downscale<T: BoxSample>(
    src: &[T],
    dims @ (height, width, channels): (usize, usize, usize),
    times: u32,
) -> (Vec<T>, usize, usize) {
    let block = 1usize << times;
    if height.is_multiple_of(block) && width.is_multiple_of(block) {
        return box_pyramid(src, dims, times);
    }
    let (ow, oh) = (width.div_ceil(block), height.div_ceil(block));
    let mut out = unwritten_output::<T>(ow * oh * channels);
    if has_whole_blocks(dims, times) {
        // SAFETY: `out` holds the whole blocks' `height >> times` rows of
        // `width >> times` pixels, `ow * channels` samples apart.
        let in_place = unsafe {
            box_pyramid_into(
                src,
                dims,
                times,
                out.as_mut_ptr().cast::<T>(),
                ow * channels,
            )
        };
        if !in_place {
            let (whole, whole_width, _) = box_pyramid(src, dims, times);
            place_rows(&whole, whole_width * channels, &mut out, ow * channels);
        }
    }
    write_partial_blocks(src, dims, times, &mut out, channels, |means, pixel| {
        for (sample, &mean) in pixel.iter_mut().zip(means) {
            sample.write(mean);
        }
    });
    // SAFETY: the whole blocks and the partial ones cover every pixel.
    (unsafe { written_output(out) }, ow, oh)
}

/// `box_downscale` of an RGB image with RGBX output.
pub(crate) fn box_downscale_rgbx(
    src: &[u8],
    (height, width): (usize, usize),
    times: u32,
    pad: u8,
) -> (Vec<u8>, usize, usize) {
    let block = 1usize << times;
    if height.is_multiple_of(block) && width.is_multiple_of(block) {
        return box_pyramid_rgbx(src, (height, width), times, pad);
    }
    let dims = (height, width, RGB);
    let (ow, oh) = (width.div_ceil(block), height.div_ceil(block));
    let mut out = unwritten_output::<u8>(ow * oh * RGBX);
    if has_whole_blocks(dims, times) {
        let (whole, whole_width, _) = box_pyramid_rgbx(src, (height, width), times, pad);
        place_rows(&whole, whole_width * RGBX, &mut out, ow * RGBX);
    }
    write_partial_blocks(src, dims, times, &mut out, RGBX, |means, pixel| {
        for (sample, &mean) in pixel.iter_mut().zip(means) {
            sample.write(mean);
        }
        pixel[RGB].write(pad);
    });
    // SAFETY: the whole blocks and the partial ones cover every pixel.
    (unsafe { written_output(out) }, ow, oh)
}

/// Whether the image has a whole `2^times` block (the kernels reduce those
/// and skip the rest).
fn has_whole_blocks((height, width, _): (usize, usize, usize), times: u32) -> bool {
    (height >> times) > 0 && (width >> times) > 0
}

/// Copies the rows of `rows` (each `row_len` samples) into `out`, `stride`
/// samples apart from its start.
fn place_rows<T: Copy>(rows: &[T], row_len: usize, out: &mut [MaybeUninit<T>], stride: usize) {
    for (y, row) in rows.chunks_exact(row_len).enumerate() {
        for (sample, &value) in out[y * stride..][..row_len].iter_mut().zip(row) {
            sample.write(value);
        }
    }
}

/// Writes the partial blocks of a `2^times` reduction at `ceil` size into
/// `out` (`out_channels` samples per pixel): the last column's and row's
/// blocks the image ends inside. `pixel` writes each from its channels' means.
fn write_partial_blocks<T: BoxSample, O>(
    src: &[T],
    dims @ (height, width, channels): (usize, usize, usize),
    times: u32,
    out: &mut [MaybeUninit<O>],
    out_channels: usize,
    pixel: impl Fn(&[T], &mut [MaybeUninit<O>]),
) {
    let block = 1usize << times;
    let (ow, oh) = (width.div_ceil(block), height.div_ceil(block));
    let (whole_width, whole_height) = if has_whole_blocks(dims, times) {
        (width >> times, height >> times)
    } else {
        (0, 0)
    };
    let mut sums = vec![0.0; channels];
    let mut means = vec![src[0]; channels];
    for y in 0..oh {
        let first_partial = if y < whole_height { whole_width } else { 0 };
        let rows = y * block..((y + 1) * block).min(height);
        for x in first_partial..ow {
            let cols = x * block..((x + 1) * block).min(width);
            sums.fill(0.0);
            for row in rows.clone() {
                let samples = &src
                    [(row * width + cols.start) * channels..(row * width + cols.end) * channels];
                for (index, &sample) in samples.iter().enumerate() {
                    sums[index % channels] += f64::from(sample.to_f32());
                }
            }
            let count = rows.len() * cols.len();
            for (mean, &sum) in means.iter_mut().zip(&sums) {
                *mean = T::block_mean(sum, count);
            }
            pixel(
                &means,
                &mut out[(y * ow + x) * out_channels..][..out_channels],
            );
        }
    }
}

/// Halvings that make `factor` a power-of-2 box reduction, if it is one.
pub(crate) fn box_halvings(filter: &str, factor: usize) -> Option<u32> {
    (filter == "box" && factor.is_power_of_two()).then(|| factor.trailing_zeros())
}

/// `times` successive 2x box halvings of a (height, width, channels) image ->
/// (pixels, width, height). Two levels per pass through the fused 4x passes
/// while at least two remain (an interleaved image's last pass folds in a
/// remaining 2x), so an interleaved image equals its channels reduced as
/// planar ones. Interleaved images with more than one pass go band by band
/// (see `box_pyramid_banded`).
pub(crate) fn box_pyramid<T: BoxSample>(
    src: &[T],
    (height, width, channels): (usize, usize, usize),
    times: u32,
) -> (Vec<T>, usize, usize) {
    if channels > 1 && box_passes(times) >= 2 {
        return box_pyramid_banded(
            src,
            (height, width, channels),
            times,
            |band, width, height| box_levels(band, (height, width, channels), times),
        );
    }
    box_levels(src, (height, width, channels), times)
}

/// `box_pyramid` written to `dst_ptr`, the output rows `dst_stride` samples
/// apart, where the CPU has a kernel that writes them in place (a single 2x
/// level): true if it did, false (nothing written) if not.
///
/// # Safety
/// `dst_ptr` holds the output rows of `box_pyramid`'s result, `dst_stride`
/// samples apart.
pub(crate) unsafe fn box_pyramid_into<T: BoxSample>(
    src: &[T],
    dims: (usize, usize, usize),
    times: u32,
    dst_ptr: *mut T,
    dst_stride: usize,
) -> bool {
    // SAFETY: the caller's contract.
    times == 1 && unsafe { T::box2x_into(src, dims, dst_ptr, dst_stride) }
}

/// The passes of a `times`-level interleaved pyramid.
fn box_passes(times: u32) -> u32 {
    if times >= 2 { times / 2 } else { times }
}

/// The levels of the next fused pass with `left` levels to go: 4x passes,
/// the last one 8x when `left` is odd (the planar pyramid's 4x then 2x).
fn fused_levels(left: u32) -> u32 {
    if left == 3 { 3 } else { 2 }
}

/// Levels the next interleaved pass halves, of `left` to go: a fused pass
/// while at least two remain, else one 2x pass.
fn next_pass_levels(left: u32) -> u32 {
    if left >= 2 { fused_levels(left) } else { 1 }
}

/// The pyramid's levels over a whole image (or band).
fn box_levels<T: BoxSample>(
    src: &[T],
    (height, width, channels): (usize, usize, usize),
    times: u32,
) -> (Vec<T>, usize, usize) {
    let mut level = (Cow::Borrowed(src), width, height);
    let mut left = times;
    while left > 0 {
        let (px, width, height) = (&level.0, level.1, level.2);
        let (next, nw, nh) = if channels > 1 {
            let levels = next_pass_levels(left);
            left -= levels;
            if levels > 1 {
                T::box_levels_hwc(px, width, height, channels, levels)
            } else {
                T::box2x_hwc(px, width, height, channels)
            }
        } else if left >= 2 {
            left -= 2;
            T::box4x(px, width, height)
        } else {
            left -= 1;
            T::box2x(px, width, height)
        };
        level = (Cow::Owned(next), nw, nh);
    }
    (level.0.into_owned(), level.1, level.2)
}

/// Input bytes per band of `box_pyramid_banded`, so its levels stay in L2.
const BAND_BYTES: usize = 256 << 10;

/// A `times`-level pyramid of an interleaved image one band of input rows at
/// a time: each output row depends only on its own 2^times input rows, so
/// running the chained halvings (`levels`) per band gives the same pixels,
/// while the intermediate levels stay small and cache-resident instead of
/// full-size images written and read back.
fn box_pyramid_banded<T: Copy, LevelsFn>(
    src: &[T],
    (height, width, channels): (usize, usize, usize),
    times: u32,
    levels: LevelsFn,
) -> (Vec<T>, usize, usize)
where
    LevelsFn: Fn(&[T], usize, usize) -> (Vec<T>, usize, usize),
{
    let block = 1usize << times;
    let (ow, oh) = (width >> times, height >> times);
    let row_bytes = width * channels * std::mem::size_of::<T>();
    let band_out = (BAND_BYTES / (row_bytes * block).max(1)).max(1);
    let mut out: Vec<T> = Vec::new();
    for oy in (0..oh).step_by(band_out) {
        let rows = band_out.min(oh - oy);
        let band = &src[oy * block * width * channels..(oy + rows) * block * width * channels];
        let (px, bw, bh) = levels(band, width, rows * block);
        debug_assert!(bw == ow && bh == rows);
        if out.is_empty() {
            out.reserve_exact(oh * (px.len() / rows));
        }
        out.extend_from_slice(&px);
    }
    (out, ow, oh)
}

/// `box_pyramid` of an RGB image with RGBX output. Intermediate levels stay
/// RGB; only the last halving widens, inside the pass that writes every output
/// pixel anyway.
pub(crate) fn box_pyramid_rgbx(
    src: &[u8],
    (height, width): (usize, usize),
    times: u32,
    pad: u8,
) -> (Vec<u8>, usize, usize) {
    if times == 0 {
        return (expand_image(src, height, width, pad), width, height);
    }
    if box_passes(times) >= 2 {
        return box_pyramid_banded(src, (height, width, RGB), times, |band, width, height| {
            box_levels_rgbx(band, (height, width), times, pad)
        });
    }
    box_levels_rgbx(src, (height, width), times, pad)
}

/// `box_pyramid_rgbx`'s levels over a whole image (or band): `box_levels`'
/// passes, the last one writing RGBX.
fn box_levels_rgbx(
    src: &[u8],
    (height, width): (usize, usize),
    times: u32,
    pad: u8,
) -> (Vec<u8>, usize, usize) {
    let mut level = (Cow::Borrowed(src), width, height);
    let mut left = times;
    loop {
        let (px, width, height) = (&level.0, level.1, level.2);
        let levels = next_pass_levels(left);
        if left == levels {
            return if levels > 1 {
                box_levels_rgb_to_rgbx(px, width, height, levels, pad)
            } else {
                box2x_rgb_to_rgbx(px, width, height, pad)
            };
        }
        let (next, nw, nh) = if levels > 1 {
            u8::box_levels_hwc(px, width, height, RGB, levels)
        } else {
            u8::box2x_hwc(px, width, height, RGB)
        };
        left -= levels;
        level = (Cow::Owned(next), nw, nh);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::box_reduce::box2x_scalar;
    use crate::box_reduce::tests::{
        BoxTestSample, assert_same_image, integer_differences, interleaved_pyramid_reference,
        per_channel, planar_pyramid_reference, rgb_to_rgbx, test_images,
    };
    use crate::testing::{Rng, random_samples, sizes_from};

    /// What `box_pyramid` must return: the planar pyramid (fused 4x passes)
    /// of every channel.
    fn pyramid_reference<T: BoxTestSample>(
        src: &[T],
        (height, width, channels): (usize, usize, usize),
        times: u32,
    ) -> (Vec<T>, usize, usize) {
        if channels == 1 {
            planar_pyramid_reference(src, width, height, times)
        } else {
            interleaved_pyramid_reference(src, (width, height, channels), times)
        }
    }

    /// `times` scalar 2x halvings of every channel, each rounded.
    fn chained_halvings<T: BoxTestSample>(
        src: &[T],
        (height, width, channels): (usize, usize, usize),
        times: u32,
    ) -> (Vec<T>, usize, usize) {
        per_channel(src, (width, height, channels), |plane, width, height| {
            let mut level = (plane.to_vec(), width, height);
            for _ in 0..times {
                level = box2x_scalar(&level.0, level.1, level.2);
            }
            level
        })
    }

    fn assert_small_pyramids_match<T: BoxTestSample>() {
        let mut rng = Rng::new(30);
        for channels in [1, RGB, RGBX] {
            for times in 0..=4 {
                for (width, height) in sizes_from(1) {
                    let dims = (height, width, channels);
                    for src in test_images::<T>(&mut rng, width * height * channels) {
                        assert_same_image(
                            &box_pyramid(&src, dims, times),
                            &pyramid_reference(&src, dims, times),
                            &format!("{times} times, {width} x {height} x {channels}"),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn box_pyramid_equals_the_planar_pyramid_of_every_channel() {
        assert_small_pyramids_match::<u8>();
        assert_small_pyramids_match::<u16>();
        assert_small_pyramids_match::<f32>();
    }

    /// An image of `channels`-sample pixels whose `times`-level pyramid
    /// runs in two full bands and a partial one, with rows left over below
    /// the last block: the banded path against the whole-image levels and
    /// the reference.
    fn assert_banded_pyramid_matches<T: BoxTestSample>(width: usize, channels: usize, times: u32) {
        let block = 1usize << times;
        let band_rows = BAND_BYTES / (width * channels * size_of::<T>() * block);
        assert!(
            band_rows >= 1,
            "the image is too wide for bands of whole output rows"
        );
        let height = (2 * band_rows + band_rows / 2 + 1) * block + block - 1;
        let dims = (height, width, channels);
        assert!(box_passes(times) >= 2);
        let mut rng = Rng::new(31);
        let src: Vec<T> = random_samples(&mut rng, width * height * channels);
        let context = format!("{times} times, {width} x {height} x {channels}");
        let banded = box_pyramid(&src, dims, times);
        assert_same_image(&banded, &box_levels(&src, dims, times), &context);
        assert_same_image(&banded, &pyramid_reference(&src, dims, times), &context);
    }

    #[test]
    fn banded_pyramids_equal_the_whole_image_levels() {
        assert_banded_pyramid_matches::<u8>(100, RGB, 4);
        assert_banded_pyramid_matches::<u8>(67, RGBX, 5);
        assert_banded_pyramid_matches::<u16>(70, RGB, 4);
        assert_banded_pyramid_matches::<u16>(130, RGBX, 6);
        assert_banded_pyramid_matches::<f32>(40, RGBX, 5);
        assert_banded_pyramid_matches::<f32>(31, RGB, 4);
    }

    #[test]
    fn one_halving_equals_the_scalar_halving() {
        let mut rng = Rng::new(32);
        for channels in [1, RGB, RGBX] {
            for (width, height) in sizes_from(2) {
                let dims = (height, width, channels);
                let src: Vec<u8> = random_samples(&mut rng, width * height * channels);
                assert_same_image(
                    &box_pyramid(&src, dims, 1),
                    &chained_halvings(&src, dims, 1),
                    &format!("{width} x {height} x {channels}"),
                );
            }
        }
    }

    #[test]
    fn fused_pyramids_differ_from_chained_halvings_by_at_most_one() {
        // The documented rounding difference: a fused 4x rounds once where
        // two chained halvings round twice.
        let mut rng = Rng::new(33);
        for channels in [1, RGB, RGBX] {
            let (width, height) = (64, 64);
            let dims = (height, width, channels);
            let src: Vec<u8> = random_samples(&mut rng, width * height * channels);
            let (largest, count) = integer_differences(
                &box_pyramid(&src, dims, 2).0,
                &chained_halvings(&src, dims, 2).0,
            );
            assert!(largest <= 1, "{channels} channels");
            // Every tier fuses, so the difference is real on ordinary data.
            assert!(count > 0, "{channels} channels");
        }
    }

    #[test]
    fn rgbx_pyramids_equal_the_rgb_pyramid_padded() {
        let mut rng = Rng::new(34);
        for times in 0..=4 {
            for (width, height) in sizes_from(1) {
                let src: Vec<u8> = random_samples(&mut rng, width * height * RGB);
                let (rgb, ow, oh) = box_pyramid(&src, (height, width, RGB), times);
                assert_same_image(
                    &box_pyramid_rgbx(&src, (height, width), times, 0xEE),
                    &(rgb_to_rgbx(&rgb, 0xEE), ow, oh),
                    &format!("{times} times, {width} x {height}"),
                );
            }
        }
        let (width, times) = (90, 4);
        let band_rows = BAND_BYTES / (width * RGB * (1 << times));
        let height = (2 * band_rows + 3) * (1 << times);
        let src: Vec<u8> = random_samples(&mut rng, width * height * RGB);
        let (rgb, ow, oh) = interleaved_pyramid_reference(&src, (width, height, RGB), times);
        assert_same_image(
            &box_pyramid_rgbx(&src, (height, width), times, 1),
            &(rgb_to_rgbx(&rgb, 1), ow, oh),
            "banded",
        );
    }

    #[test]
    fn box_halvings_are_the_power_of_two_box_factors() {
        assert_eq!(box_halvings("box", 1), Some(0));
        assert_eq!(box_halvings("box", 2), Some(1));
        assert_eq!(box_halvings("box", 64), Some(6));
        assert_eq!(box_halvings("box", 3), None);
        assert_eq!(box_halvings("box", 12), None);
        assert_eq!(box_halvings("bilinear", 2), None);
    }

    #[test]
    fn fused_passes_end_with_an_8x_pass_for_odd_levels() {
        let pass_levels = |times: u32| {
            let mut passes = Vec::new();
            let mut left = times;
            while left > 0 {
                let levels = next_pass_levels(left);
                passes.push(levels);
                left -= levels;
            }
            passes
        };
        assert_eq!(pass_levels(1), [1]);
        assert_eq!(pass_levels(3), [3]);
        assert_eq!(pass_levels(5), [2, 3]);
        assert_eq!(pass_levels(6), [2, 2, 2]);
        for times in 0..8 {
            assert_eq!(pass_levels(times).len() as u32, box_passes(times));
        }
    }
}

//! Grids of tiles as one image: stitched, or each tile box-reduced and
//! written in place, optionally inside a filled frame.

use numpy::{Element, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::box_reduce::{self, BoxSample};
use crate::sample::Sample;
use crate::{c_samples, image_dims, output_samples, read_only, to_numpy};

/// A row-major grid of tiles: `grid_width` per grid row, a grid row sharing
/// its height and a grid column its width.
struct TileGrid {
    dims: Vec<(usize, usize, usize)>,
    heights: Vec<usize>,
    widths: Vec<usize>,
    channels: usize,
}

impl TileGrid {
    fn new(tiles: &[Bound<'_, PyAny>], grid_width: usize) -> PyResult<TileGrid> {
        if grid_width == 0 || tiles.is_empty() || !tiles.len().is_multiple_of(grid_width) {
            return Err(PyValueError::new_err(
                "tiles must fill whole grid rows of grid_width tiles",
            ));
        }
        let dims = tiles.iter().map(image_dims).collect::<PyResult<Vec<_>>>()?;
        let channels = dims[0].2;
        let heights: Vec<usize> = (0..tiles.len() / grid_width)
            .map(|grid_row| dims[grid_row * grid_width].0)
            .collect();
        let widths: Vec<usize> = (0..grid_width).map(|col| dims[col].1).collect();
        for (tile_index, &(tile_height, tile_width, tile_channels)) in dims.iter().enumerate() {
            if tile_height != heights[tile_index / grid_width]
                || tile_width != widths[tile_index % grid_width]
                || tile_channels != channels
            {
                return Err(PyValueError::new_err(
                    "every tile must share its grid row's height, its grid column's width \
                     and the channel count",
                ));
            }
        }
        Ok(TileGrid {
            dims,
            heights,
            widths,
            channels,
        })
    }

    fn grid_width(&self) -> usize {
        self.widths.len()
    }
}

/// The tiles' samples, borrowed read-only as `T`.
fn tile_samples<'py, T: Element>(
    tiles: &[Bound<'py, PyAny>],
) -> PyResult<Vec<PyReadonlyArrayDyn<'py, T>>> {
    tiles
        .iter()
        .map(|tile| {
            let array = tile.cast::<PyArrayDyn<T>>().map_err(|_| {
                PyValueError::new_err("every tile must have the first tile's dtype")
            })?;
            read_only(array)
        })
        .collect()
}

/// Copies `rows` rows of `row_len` samples from `src` (contiguous) into `out`
/// from sample `at`, `stride` samples apart.
///
/// # Safety
/// The destination rows are inside `out`: for each of the
/// `src.len() / row_len` rows, `at + y * stride + row_len <= out.len()`.
unsafe fn place_rows<T: Copy>(
    src: &[T],
    row_len: usize,
    out: &mut [std::mem::MaybeUninit<T>],
    at: usize,
    stride: usize,
) {
    for (y, row) in src.chunks_exact(row_len).enumerate() {
        // SAFETY: row y's destination `at + y * stride .. + row_len` is inside
        // `out` (the caller's contract); `src` and `out` are distinct borrows.
        unsafe {
            std::ptr::copy_nonoverlapping(
                row.as_ptr(),
                out.as_mut_ptr().add(at + y * stride).cast::<T>(),
                row_len,
            );
        };
    }
}

/// The output a grid reduction is placed in: `(height, width)`, at least the
/// reduced grid's, the reduced grid at its top left and `fill` (one pixel's
/// samples) around it.
struct Frame<T> {
    height: usize,
    width: usize,
    fill: Vec<T>,
}

impl<T: Element + Copy> Frame<T> {
    /// The frame `size` asks for, or none; `fill` must be a (`channels`,)
    /// array, given exactly when `size` is.
    fn new(
        size: Option<(usize, usize)>,
        fill: Option<&Bound<'_, PyAny>>,
        channels: usize,
    ) -> PyResult<Option<Frame<T>>> {
        let Some((height, width)) = size else {
            return match fill {
                Some(_) => Err(PyValueError::new_err("fill needs size")),
                None => Ok(None),
            };
        };
        let fill = fill.ok_or_else(|| PyValueError::new_err("size needs fill"))?;
        let samples = fill
            .cast::<PyArrayDyn<T>>()
            .map_err(|_| PyValueError::new_err("fill must have the tiles' dtype"))?;
        if samples.shape() != [channels] {
            return Err(PyValueError::new_err(
                "fill must be a (C,) array: one value per channel",
            ));
        }
        let fill = c_samples(&read_only(samples)?, "fill")?.to_vec();
        output_samples((height, width, channels), size_of::<T>())?;
        Ok(Some(Frame {
            height,
            width,
            fill,
        }))
    }

    /// The frame's (height, width) for a reduced image of
    /// `(reduced_height, reduced_width)`, checked.
    fn size_around(
        &self,
        (reduced_height, reduced_width): (usize, usize),
    ) -> PyResult<(usize, usize)> {
        if self.height < reduced_height || self.width < reduced_width {
            return Err(PyValueError::new_err(
                "size must be at least the reduced grid's",
            ));
        }
        Ok((self.height, self.width))
    }

    /// Fills `out` (the frame's rows, `channels` samples per pixel) outside
    /// its top-left `(block_height, block_width)` pixels: whole fill rows copied from one built
    /// row.
    ///
    /// # Safety
    /// `out` holds the frame (`height * width * channels` samples) and the
    /// block lies inside it (`block_height <= height`, `block_width <= width`).
    unsafe fn fill_margins(
        &self,
        out: &mut [std::mem::MaybeUninit<T>],
        (block_height, block_width): (usize, usize),
    ) {
        let channels = self.fill.len();
        let row_stride = self.width * channels;
        let fill_row: Vec<T> = self.fill.iter().copied().cycle().take(row_stride).collect();
        let right = (self.width - block_width) * channels;
        if right > 0 {
            for y in 0..block_height {
                // SAFETY: row y's samples past the block are inside `out`.
                unsafe {
                    place_rows(
                        &fill_row[..right],
                        right,
                        out,
                        y * row_stride + block_width * channels,
                        row_stride,
                    );
                };
            }
        }
        for y in block_height..self.height {
            // SAFETY: row y is inside `out`, below the block.
            unsafe { place_rows(&fill_row, row_stride, out, y * row_stride, row_stride) };
        }
    }
}

/// `downscale_grid`'s box path: the tiles of the grid, each reduced by
/// `2^times` on its own and its rows written straight into place in the one
/// output (in `frame`, if given). Equal to reducing the stitched image when
/// every tile but the last row's and column's divides by the factor, which
/// the caller checks: no box window then crosses a seam, and only the last
/// row's and column's tiles end in partial blocks.
pub(crate) fn box_grid<T: BoxSample>(
    py: Python<'_>,
    tiles: &[Bound<'_, PyAny>],
    grid_width: usize,
    times: u32,
    (size, fill): (Option<(usize, usize)>, Option<&Bound<'_, PyAny>>),
) -> PyResult<Py<PyAny>> {
    let factor = 1usize << times;
    let grid = TileGrid::new(tiles, grid_width)?;
    let reads = tile_samples::<T>(tiles)?;
    let srcs = reads
        .iter()
        .map(|read| c_samples(read, "tile"))
        .collect::<PyResult<Vec<_>>>()?;
    let (dims, heights, widths, channels) =
        (&grid.dims, &grid.heights, &grid.widths, grid.channels);
    let grid_rows = heights.len();
    for (tile_index, &(tile_height, tile_width, _)) in dims.iter().enumerate() {
        let (grid_row, col) = (tile_index / grid_width, tile_index % grid_width);
        if (grid_row + 1 < grid_rows && tile_height % factor != 0)
            || (col + 1 < grid_width && tile_width % factor != 0)
        {
            return Err(PyValueError::new_err(
                "inner tiles must divide by the factor",
            ));
        }
    }
    let out_rows: Vec<usize> = heights
        .iter()
        .map(|height| height.div_ceil(factor))
        .collect();
    let out_cols: Vec<usize> = widths.iter().map(|width| width.div_ceil(factor)).collect();
    let (oh, ow): (usize, usize) = (out_rows.iter().sum(), out_cols.iter().sum());
    let frame = Frame::<T>::new(size, fill, channels)?;
    let (fh, fw) = match &frame {
        Some(frame) => frame.size_around((oh, ow))?,
        None => (oh, ow),
    };
    let row_stride = fw * channels;
    let out = py.detach(|| {
        let mut out = crate::cpu::unwritten_output::<T>(fh * row_stride);
        let mut y0 = 0;
        for (grid_row, &tile_rows) in out_rows.iter().enumerate() {
            let mut x0 = 0;
            for (col, &tile_cols) in out_cols.iter().enumerate() {
                if tile_rows == 0 || tile_cols == 0 {
                    // Nothing to write (an empty tile): no pointer to
                    // its place, which may lie past the output.
                    x0 += tile_cols;
                    continue;
                }
                let tile_index = grid_row * grid_width + col;
                let tile_at = y0 * row_stride + x0 * channels;
                let (tile_height, tile_width, _) = dims[tile_index];
                let whole_blocks = tile_height % factor == 0 && tile_width % factor == 0;
                // SAFETY: rows y0..y0 + tile_rows, samples x0 * channels.. of
                // `out` are this tile's alone and inside it.
                let in_place = whole_blocks
                    && unsafe {
                        box_reduce::pyramid::box_pyramid_into(
                            srcs[tile_index],
                            dims[tile_index],
                            times,
                            out.as_mut_ptr().add(tile_at).cast::<T>(),
                            row_stride,
                        )
                    };
                if !in_place {
                    let (tile, tw, th) = box_reduce::pyramid::box_downscale(
                        srcs[tile_index],
                        dims[tile_index],
                        times,
                    );
                    debug_assert!((tw, th) == (tile_cols, tile_rows));
                    // SAFETY: as above.
                    unsafe { place_rows(&tile, tw * channels, &mut out, tile_at, row_stride) };
                }
                x0 += tile_cols;
            }
            y0 += tile_rows;
        }
        if let Some(frame) = &frame {
            // SAFETY: `out` holds the (fh, fw) frame, and `size_around` checked
            // that the (oh, ow) block fits in it.
            unsafe { frame.fill_margins(&mut out, (oh, ow)) };
        }
        // SAFETY: the tiles' outputs tile the top-left block and the frame's
        // fill the rest, each sample written once.
        unsafe { crate::cpu::written_output(out) }
    });
    to_numpy(py, out, fh, fw, channels)
}

/// `image` placed at the top left of `frame`, the rest filled.
pub(crate) fn framed<T: Sample>(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    (size, fill): (Option<(usize, usize)>, Option<&Bound<'_, PyAny>>),
) -> PyResult<Py<PyAny>> {
    let (height, width, channels) = image_dims(image)?;
    let frame = Frame::<T>::new(size, fill, channels)?
        .ok_or_else(|| PyValueError::new_err("size is required"))?;
    let (fh, fw) = frame.size_around((height, width))?;
    let array = image
        .cast::<PyArrayDyn<T>>()
        .map_err(|_| PyValueError::new_err("image must have fill's dtype"))?;
    let read = read_only(array)?;
    let src = c_samples(&read, "image")?;
    let out = py.detach(|| {
        let mut out = crate::cpu::unwritten_output::<T>(fh * fw * channels);
        // SAFETY: `out` holds the (fh, fw) frame and `size_around` checked that
        // the (height, width) image fits in it: the image's rows land inside
        // `out` and fill its top-left block, the frame's fill the rest, so
        // every sample is written.
        unsafe {
            place_rows(src, width * channels, &mut out, 0, fw * channels);
            frame.fill_margins(&mut out, (height, width));
            crate::cpu::written_output(out)
        }
    });
    to_numpy(py, out, fh, fw, channels)
}

/// The grid's tiles stitched into one image (recycled memory, no clearing),
/// handed to numpy without a copy.
pub(crate) fn stitch_grid<T: Sample>(
    py: Python<'_>,
    tiles: &[Bound<'_, PyAny>],
    grid_width: usize,
) -> PyResult<Py<PyAny>> {
    let grid = TileGrid::new(tiles, grid_width)?;
    let reads = tile_samples::<T>(tiles)?;
    let srcs = reads
        .iter()
        .map(|read| c_samples(read, "tile"))
        .collect::<PyResult<Vec<_>>>()?;
    let (height, width, channels) = (
        grid.heights.iter().sum::<usize>(),
        grid.widths.iter().sum::<usize>(),
        grid.channels,
    );
    let stitched = py.detach(|| {
        let mut out = crate::cpu::unwritten_output::<T>(height * width * channels);
        let mut y0 = 0;
        for (grid_row, &th) in grid.heights.iter().enumerate() {
            let mut x0 = 0;
            for (col, &tw) in grid.widths.iter().enumerate() {
                // SAFETY: rows y0..y0 + th, samples x0 * channels.. of `out` are this
                // tile's alone and inside it.
                unsafe {
                    place_rows(
                        srcs[grid_row * grid.grid_width() + col],
                        tw * channels,
                        &mut out,
                        (y0 * width + x0) * channels,
                        width * channels,
                    );
                };
                x0 += tw;
            }
            y0 += th;
        }
        // SAFETY: the tiles cover the stitched image, each sample written once.
        unsafe { crate::cpu::written_output(out) }
    });
    to_numpy(py, stitched, height, width, channels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::CHANNELS;
    use std::mem::MaybeUninit;

    const SENTINEL: u16 = 0xDEAD;

    fn sentinels(len: usize) -> Vec<MaybeUninit<u16>> {
        vec![MaybeUninit::new(SENTINEL); len]
    }

    fn read(out: &[MaybeUninit<u16>]) -> Vec<u16> {
        // SAFETY: every element was initialized by `sentinels` (and maybe
        // overwritten since).
        out.iter()
            .map(|sample| unsafe { sample.assume_init() })
            .collect()
    }

    #[test]
    fn place_rows_writes_each_row_a_stride_apart() {
        let src: Vec<u16> = (0..12).collect();
        let mut out = sentinels(40);
        // SAFETY: the last row lands at 3 + 2 * 10 + 4 = 27 <= 40.
        unsafe { place_rows(&src, 4, &mut out, 3, 10) };
        let out = read(&out);
        for (index, &sample) in out.iter().enumerate() {
            let (row, column) = (index.saturating_sub(3) / 10, index.saturating_sub(3) % 10);
            let expected = if index >= 3 && row < 3 && column < 4 {
                u16::try_from(row * 4 + column).unwrap()
            } else {
                SENTINEL
            };
            assert_eq!(sample, expected, "sample {index}");
        }
    }

    #[test]
    fn place_rows_ending_at_the_output_end() {
        let src: Vec<u16> = (0..6).collect();
        let mut out = sentinels(8);
        // SAFETY: the second row ends at 2 + 3 + 3 = 8, the output's end.
        unsafe { place_rows(&src, 3, &mut out, 2, 3) };
        assert_eq!(read(&out), [SENTINEL, SENTINEL, 0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn fill_margins_fills_exactly_outside_the_block() {
        for channels in CHANNELS {
            let fill: Vec<u16> = (100..100 + channels as u16).collect();
            for (height, width) in [(1, 1), (4, 5), (3, 3), (6, 2)] {
                let frame = Frame {
                    height,
                    width,
                    fill: fill.clone(),
                };
                for block_height in 0..=height {
                    for block_width in 0..=width {
                        let mut out = sentinels(height * width * channels);
                        // SAFETY: `out` holds the frame and the block fits in it.
                        unsafe { frame.fill_margins(&mut out, (block_height, block_width)) };
                        let out = read(&out);
                        for (index, pixel) in out.chunks_exact(channels).enumerate() {
                            let (y, x) = (index / width, index % width);
                            let in_block = y < block_height && x < block_width;
                            let expected: &[u16] = if in_block {
                                &[SENTINEL; 4][..channels]
                            } else {
                                &fill
                            };
                            assert_eq!(
                                pixel, expected,
                                "{height} x {width} x {channels}, block \
                                 {block_height} x {block_width}, pixel ({x}, {y})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn size_around_accepts_frames_at_least_the_image() {
        let frame = Frame::<u8> {
            height: 4,
            width: 6,
            fill: vec![0],
        };
        assert_eq!(frame.size_around((4, 6)).ok(), Some((4, 6)));
        assert_eq!(frame.size_around((1, 2)).ok(), Some((4, 6)));
        assert!(frame.size_around((5, 6)).is_err());
        assert!(frame.size_around((4, 7)).is_err());
    }
}

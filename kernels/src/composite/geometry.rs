//! Where a tile lands on the canvas: pixel-grid views, the fractional
//! placement of a translated tile, and the separable bicubic row cache.

use std::ops::Range;

/// Pixels of edge replication a bicubic source row needs on each side: taps
/// reach one pixel left of x0 and two right of it.
pub(crate) const EDGE_PAD_LEFT: usize = 1;
pub(crate) const EDGE_PAD_RIGHT: usize = 2;

/// A read-only `width x height` pixel grid, `channels` interleaved samples per
/// pixel (channels = 1: planar). Each row is contiguous, and row `y` starts
/// at sample `y * stride`: a crop of a larger grid shares its rows.
pub(crate) struct Tile<'a, T> {
    pub(crate) px: &'a [T],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) channels: usize,
    pub(crate) stride: usize,
}

impl<'a, T: Copy> Tile<'a, T> {
    /// The grid in `px`, rows back to back.
    pub(crate) fn new(px: &'a [T], width: usize, height: usize, channels: usize) -> Self {
        Tile {
            px,
            width,
            height,
            channels,
            stride: width * channels,
        }
    }

    /// The `width x height` pixels from (`x`, `y`), which must lie inside.
    pub(crate) fn crop(&self, (x, y): (usize, usize), (width, height): (usize, usize)) -> Self {
        let start = y * self.stride + x * self.channels;
        let len = if height == 0 {
            0
        } else {
            (height - 1) * self.stride + width * self.channels
        };
        Tile {
            px: &self.px[start..start + len],
            width,
            height,
            ..*self
        }
    }

    pub(crate) fn row(&self, y: usize) -> &'a [T] {
        let start = y * self.stride;
        &self.px[start..start + self.width * self.channels]
    }

    /// Row `y`, with rows above/below the tile replicating its edge rows.
    pub(crate) fn row_clamped(&self, y: isize) -> &'a [T] {
        self.row(y.clamp(0, self.height as isize - 1) as usize)
    }

    pub(crate) fn rows4_clamped(&self, top: isize) -> [&'a [T]; 4] {
        std::array::from_fn(|row_offset| self.row_clamped(top + row_offset as isize))
    }

    /// Row `y` (clamped, see `row_clamped`) into `buf`, widened by
    /// `EDGE_PAD_LEFT` / `EDGE_PAD_RIGHT` copies of its first / last pixel:
    /// every bicubic tap (x0 - 1 ..= x0 + 2) of every column then lies inside
    /// it, at padded index x + `EDGE_PAD_LEFT`.
    pub(crate) fn edge_padded_row(&self, y: isize, buf: &mut Vec<T>) {
        let row = self.row_clamped(y);
        let (first, last) = (&row[..self.channels], &row[row.len() - self.channels..]);
        buf.clear();
        for _ in 0..EDGE_PAD_LEFT {
            buf.extend_from_slice(first);
        }
        buf.extend_from_slice(row);
        for _ in 0..EDGE_PAD_RIGHT {
            buf.extend_from_slice(last);
        }
    }

    /// Copy the source pixels that output pixels with left taps
    /// `x0 .. x0 + n_px` read (`x0 - 1 ..= x0 + n_px + 1`) from `row` into `buf`,
    /// replicating the row's edge pixels past its ends. In `buf`, the first
    /// output pixel's x0 tap sits at element `channels`.
    pub(crate) fn edge_replicated(&self, row: &[T], x0: usize, n_px: usize, buf: &mut Vec<T>) {
        let last = self.width as isize - 1;
        buf.clear();
        for x in (x0 as isize - 1)..=(x0 + n_px + 1) as isize {
            let x = x.clamp(0, last) as usize * self.channels;
            buf.extend_from_slice(&row[x..x + self.channels]);
        }
    }
}

/// The mutable destination grid (same layout rules as `Tile`).
pub(crate) struct Canvas<'a, T> {
    pub(crate) px: &'a mut [T],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) channels: usize,
}

impl<T> Canvas<'_, T> {
    /// The `pixel_count` pixels starting at (x, y), all samples.
    pub(crate) fn run(&mut self, x: isize, y: isize, pixel_count: usize) -> &mut [T] {
        let start = (y as usize * self.width + x as usize) * self.channels;
        &mut self.px[start..start + pixel_count * self.channels]
    }

    /// While resampling into row `y` of a run of rows ending at `last_y`,
    /// prefetch the same run on the next row. Rows of a tile land a canvas
    /// stride apart, a jump the hardware prefetcher doesn't follow, so without
    /// this each row starts on cache misses. (Plain row copies finish too fast
    /// for the hint to pay; the resample kernels leave it time to land.)
    pub(crate) fn prefetch_next_row(&self, x: isize, y: isize, last_y: isize, pixel_count: usize) {
        if y >= last_y {
            return;
        }
        let start = ((y + 1) as usize * self.width + x as usize) * self.channels;
        let next_row_ptr = self.px.as_ptr().wrapping_add(start) as *const u8;
        crate::cpu::prefetch_for_write(
            next_row_ptr,
            pixel_count * self.channels * std::mem::size_of::<T>(),
        );
    }
}

/// Inclusive rectangle of canvas pixels.
#[derive(Clone, Copy)]
pub(crate) struct Rect {
    pub(crate) x0: isize,
    pub(crate) x1: isize,
    pub(crate) y0: isize,
    pub(crate) y1: isize,
}

impl Rect {
    pub(crate) fn nonempty(self) -> Option<Rect> {
        (self.x0 <= self.x1 && self.y0 <= self.y1).then_some(self)
    }

    pub(crate) fn width(&self) -> usize {
        (self.x1 - self.x0 + 1) as usize
    }
}

/// The covered columns of a bicubic output row, split by where their
/// horizontal taps (x0 - 1 ..= x0 + 2) fall: output pixels in `inside` read
/// the tile row directly; the ones before and after it (at most 1 and 2) have
/// taps past the tile's left/right edge.
pub(crate) struct Columns {
    /// Tile column of the first covered pixel's x0 tap.
    pub(crate) first_x0: usize,
    /// Covered pixels in the row.
    pub(crate) count: usize,
    pub(crate) inside: Range<usize>,
}

impl Columns {
    /// Output pixel ranges whose taps cross the tile's left or right edge.
    pub(crate) fn edges(&self) -> [Range<usize>; 2] {
        [0..self.inside.start, self.inside.end..self.count]
    }
}

/// A tile translated to a fractional canvas position (px, py). Canvas pixel
/// (cx, cy) samples the tile at (`src_x(cx)` + fx, `src_y(cy)` + fy): one sub-pixel
/// phase for the whole tile, so resampling needs constant weights only and
/// whole rows can go through the SIMD row kernels. Only canvas pixels fully
/// backed by tile samples are written (`covered`).
pub(crate) struct Placement {
    x_origin: isize,
    y_origin: isize,
    pub(crate) fx: f32,
    pub(crate) fy: f32,
    pub(crate) covered: Rect,
}

impl Placement {
    #[cfg(test)]
    pub(crate) fn new<T, U>(
        px: f64,
        py: f64,
        tile: &Tile<T>,
        canvas: &Canvas<U>,
    ) -> Option<Placement> {
        Self::inset(px, py, tile, canvas, 0)
    }

    /// As `new`, with `covered` shrunk by `inset` pixels at both ends of each
    /// fractional axis: by 1, it keeps bicubic's outer taps (-1, +2) inside
    /// the tile too.
    pub(crate) fn inset<T, U>(
        px: f64,
        py: f64,
        tile: &Tile<T>,
        canvas: &Canvas<U>,
        inset: isize,
    ) -> Option<Placement> {
        let (x_origin, y_origin) = (px.ceil() as isize, py.ceil() as isize);
        // The last canvas pixel fully backed by the tile along an axis: the
        // tile's last one, or its second last at a fractional position (the
        // last one's right/bottom tap would lie past the tile). In integers:
        // `floor(p + len - 1)` in f64 rounds up when p's fraction is within
        // half an ulp of 1, which would cover a pixel whose tap reads past
        // the tile.
        let last_backed = |origin: isize, len: usize, position: f64| {
            origin + len as isize - 1 - isize::from(position.fract() != 0.0)
        };
        let inset_x = if px.fract() == 0.0 { 0 } else { inset };
        let inset_y = if py.fract() == 0.0 { 0 } else { inset };
        let covered = Rect {
            x0: (x_origin + inset_x).max(0),
            x1: (last_backed(x_origin, tile.width, px) - inset_x).min(canvas.width as isize - 1),
            y0: (y_origin + inset_y).max(0),
            y1: (last_backed(y_origin, tile.height, py) - inset_y).min(canvas.height as isize - 1),
        }
        .nonempty()?;
        Some(Placement {
            x_origin,
            y_origin,
            fx: (px.ceil() - px) as f32,
            fy: (py.ceil() - py) as f32,
            covered,
        })
    }

    pub(crate) fn src_x(&self, cx: isize) -> usize {
        (cx - self.x_origin) as usize
    }

    pub(crate) fn src_y(&self, cy: isize) -> usize {
        (cy - self.y_origin) as usize
    }

    /// [w00, w10, w01, w11]: weights of the (left, top), (right, top),
    /// (left, bottom), (right, bottom) taps.
    pub(crate) fn bilinear_weights(&self) -> [f32; 4] {
        let (fx, fy) = (self.fx, self.fy);
        [
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        ]
    }

    /// Element offset from the left bilinear tap to the right one. When x is
    /// integer-aligned (fx == 0) `covered` reaches the tile's last column, and
    /// the right tap — weight 0 there — would read past the row; 0 re-reads
    /// the left tap instead.
    pub(crate) fn right_tap_step(&self, channels: usize) -> usize {
        if self.fx == 0.0 { 0 } else { channels }
    }

    /// Tile rows of the top and bottom bilinear taps for canvas row `cy`; the
    /// bottom one clamped for the same reason as `right_tap_step` (fy == 0).
    pub(crate) fn bilinear_rows(&self, cy: isize, tile_h: usize) -> (usize, usize) {
        let top = self.src_y(cy);
        (top, (top + 1).min(tile_h - 1))
    }

    /// Tile row of the top bicubic tap for canvas row `cy`; -1 on the tile's
    /// first row (see `Tile::row_clamped`).
    pub(crate) fn bicubic_top(&self, cy: isize) -> isize {
        cy - self.y_origin - 1
    }

    pub(crate) fn bicubic_columns(&self, tile_w: usize) -> Columns {
        let count = self.covered.width();
        let first_x0 = self.src_x(self.covered.x0);
        // Pixel i is inside when 1 <= first_x0 + i <= tile_w - 3.
        let start = (1 - first_x0 as isize).clamp(0, count as isize);
        let end = (tile_w as isize - 2 - first_x0 as isize).clamp(start, count as isize);
        Columns {
            first_x0,
            count,
            inside: start as usize..end as usize,
        }
    }
}

/// Catmull-Rom cubic kernel (Keys parameter a = -0.5) at signed tap distance
/// `offset`.
#[inline]
fn cubic_weight(offset: f32) -> f32 {
    let keys_a = -0.5f32;
    let distance = offset.abs();
    if distance <= 1.0 {
        ((keys_a + 2.0) * distance - (keys_a + 3.0)) * distance * distance + 1.0
    } else if distance < 2.0 {
        (((distance - 5.0) * distance + 8.0) * distance - 4.0) * keys_a
    } else {
        0.0
    }
}

/// Weights of the taps at -1, 0, +1, +2 around a sample at phase `phase`.
pub(crate) fn cubic_weights(phase: f32) -> [f32; 4] {
    [
        cubic_weight(phase + 1.0),
        cubic_weight(phase),
        cubic_weight(phase - 1.0),
        cubic_weight(phase - 2.0),
    ]
}

/// The separable bicubic's horizontal pass, cached for the 4 tile rows under
/// the current output row, so sliding down one row convolves one new row.
/// Rows are logical (they may lie past the tile; the fill clamps them).
pub(crate) struct HconvRing<A> {
    buf: Vec<A>,
    row_len: usize,
    slots: [usize; 4],
    top: Option<isize>,
}

impl<A: Copy + Default> HconvRing<A> {
    pub(crate) fn new(row_len: usize) -> Self {
        HconvRing {
            buf: vec![A::default(); 4 * row_len],
            row_len,
            slots: [0, 1, 2, 3],
            top: None,
        }
    }

    /// Hold rows `top..top + 4`, calling `hconv(row, out)` for each row not
    /// already cached.
    pub(crate) fn load(&mut self, top: isize, mut hconv: impl FnMut(isize, &mut [A])) {
        let row_len = self.row_len;
        if self.top == Some(top - 1) {
            self.slots.rotate_left(1);
            let slot = self.slots[3];
            hconv(top + 3, &mut self.buf[slot * row_len..(slot + 1) * row_len]);
        } else if self.top != Some(top) {
            for (row_offset, &slot) in self.slots.iter().enumerate() {
                hconv(
                    top + row_offset as isize,
                    &mut self.buf[slot * row_len..(slot + 1) * row_len],
                );
            }
        }
        self.top = Some(top);
    }

    pub(crate) fn rows(&self) -> [&[A]; 4] {
        self.slots
            .map(|slot| &self.buf[slot * self.row_len..(slot + 1) * self.row_len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fractional and integer positions from well off the start of a
    /// 10-pixel canvas axis to well past its end, quarter-pixel apart.
    fn positions() -> impl Iterator<Item = f64> {
        (-60..=60).map(|quarter| f64::from(quarter) * 0.25)
    }

    fn tile_of(samples: &[u8], width: usize, height: usize, channels: usize) -> Tile<'_, u8> {
        Tile::new(
            &samples[..width * height * channels],
            width,
            height,
            channels,
        )
    }

    #[test]
    fn covered_pixels_have_every_bilinear_tap_inside_the_tile() {
        let samples = vec![0u8; 12 * 12];
        let mut canvas_samples = vec![0u8; 10 * 10];
        let canvas = Canvas {
            px: &mut canvas_samples,
            width: 10,
            height: 10,
            channels: 1,
        };
        for width in 1..=12 {
            let tile = tile_of(&samples, width, 1, 1);
            for px in positions() {
                let Some(placement) = Placement::new(px, 0.0, &tile, &canvas) else {
                    continue;
                };
                let covered = placement.covered;
                assert!(covered.x0 >= 0 && covered.x1 < 10, "width {width} at {px}");
                let right_tap = |cx: isize| placement.src_x(cx) + placement.right_tap_step(1);
                assert!(right_tap(covered.x1) < width, "width {width} at {px}");
                assert_eq!(
                    placement.src_x(covered.x0) as f64 + f64::from(placement.fx),
                    covered.x0 as f64 - px,
                    "width {width} at {px}"
                );
            }
        }
    }

    #[test]
    fn bilinear_rows_stay_inside_the_tile() {
        let samples = vec![0u8; 12];
        let mut canvas_samples = vec![0u8; 10 * 10];
        let canvas = Canvas {
            px: &mut canvas_samples,
            width: 10,
            height: 10,
            channels: 1,
        };
        for height in 1..=12 {
            let tile = tile_of(&samples, 1, height, 1);
            for py in positions() {
                let Some(placement) = Placement::new(0.0, py, &tile, &canvas) else {
                    continue;
                };
                for cy in placement.covered.y0..=placement.covered.y1 {
                    let (top, bottom) = placement.bilinear_rows(cy, height);
                    assert!(top < height && bottom < height, "height {height} at {py}");
                    let expected_bottom = if placement.fy == 0.0 { bottom } else { top + 1 };
                    assert_eq!(bottom, expected_bottom, "height {height} at {py}");
                }
            }
        }
    }

    #[test]
    fn bicubic_columns_split_where_taps_leave_the_tile() {
        let samples = vec![0u8; 12];
        let mut canvas_samples = vec![0u8; 10];
        let canvas = Canvas {
            px: &mut canvas_samples,
            width: 10,
            height: 1,
            channels: 1,
        };
        for width in 1..=12 {
            let tile = tile_of(&samples, width, 1, 1);
            for px in positions() {
                let Some(placement) = Placement::new(px, 0.0, &tile, &canvas) else {
                    continue;
                };
                let cols = placement.bicubic_columns(width);
                let context = format!("width {width} at {px}");
                assert_eq!(cols.count, placement.covered.width(), "{context}");
                assert!(cols.first_x0 + cols.count <= width, "{context}");
                for pixel in 0..cols.count {
                    let x0 = cols.first_x0 + pixel;
                    let taps_inside = x0 >= 1 && x0 + 2 < width;
                    assert_eq!(
                        cols.inside.contains(&pixel),
                        taps_inside,
                        "{context}: {pixel}"
                    );
                }
                let [left, right] = cols.edges();
                assert!(left.len() <= 1 && right.len() <= 2, "{context}");
                assert_eq!(
                    left.len() + cols.inside.len() + right.len(),
                    cols.count,
                    "{context}"
                );
            }
        }
    }

    #[test]
    fn edge_padded_row_replicates_the_edge_pixels() {
        let samples: Vec<u8> = (0..60).collect();
        for (width, height, channels) in [(1, 3, 1), (5, 4, 3), (4, 3, 4)] {
            let tile = tile_of(&samples, width, height, channels);
            let mut padded = Vec::new();
            for y in -2..height as isize + 2 {
                tile.edge_padded_row(y, &mut padded);
                let row = tile.row(y.clamp(0, height as isize - 1) as usize);
                let expected: Vec<u8> = (-(EDGE_PAD_LEFT as isize)
                    ..(width + EDGE_PAD_RIGHT) as isize)
                    .flat_map(|x| {
                        let x = x.clamp(0, width as isize - 1) as usize;
                        row[x * channels..(x + 1) * channels].to_vec()
                    })
                    .collect();
                assert_eq!(padded, expected, "{width} x {height} x {channels}, row {y}");
            }
        }
    }

    #[test]
    fn edge_replicated_copies_the_clamped_taps() {
        let samples: Vec<u8> = (0..60).collect();
        for (width, channels) in [(1, 1), (2, 3), (5, 4), (7, 1)] {
            let tile = tile_of(&samples, width, 1, channels);
            let row = tile.row(0);
            let mut buf = Vec::new();
            for x0 in 0..width {
                for pixel_count in 1..=width - x0 {
                    tile.edge_replicated(row, x0, pixel_count, &mut buf);
                    let expected: Vec<u8> = (x0 as isize - 1..=(x0 + pixel_count + 1) as isize)
                        .flat_map(|x| {
                            let x = x.clamp(0, width as isize - 1) as usize;
                            row[x * channels..(x + 1) * channels].to_vec()
                        })
                        .collect();
                    assert_eq!(buf, expected, "width {width} x0 {x0} count {pixel_count}");
                }
            }
        }
    }

    #[test]
    fn canvas_run_is_the_pixels_from_x_y() {
        let mut samples: Vec<u16> = (0..5 * 4 * 3).collect();
        let mut canvas = Canvas {
            px: &mut samples,
            width: 5,
            height: 4,
            channels: 3,
        };
        assert_eq!(canvas.run(1, 2, 3), &[33, 34, 35, 36, 37, 38, 39, 40, 41]);
        assert_eq!(canvas.run(4, 3, 1), &[57, 58, 59]);
        // The last row has no next row to prefetch; the call must not index.
        canvas.prefetch_next_row(0, 3, 3, 5);
        canvas.prefetch_next_row(0, 2, 3, 5);
    }

    #[test]
    fn cubic_weights_interpolate_and_sum_to_one() {
        assert_eq!(cubic_weights(0.0), [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(cubic_weights(0.5), [-0.0625, 0.5625, 0.5625, -0.0625]);
        for quarter in 0..4 {
            let weights = cubic_weights(f32::from(quarter as u8) * 0.25 + 0.1);
            let sum: f32 = weights.iter().sum();
            // Each weight is a cubic in f32: a few roundings apiece.
            assert!((sum - 1.0).abs() <= 8.0 * f32::EPSILON, "{weights:?}");
        }
    }

    #[test]
    fn hconv_ring_holds_the_four_rows_from_top() {
        // Each cached row holds its own row index; `load` counts the rows it
        // convolves.
        let mut ring = HconvRing::<isize>::new(3);
        for (top, new_rows) in [(-1, 4), (0, 1), (1, 1), (1, 0), (5, 4), (6, 1), (3, 4)] {
            let mut convolved = 0;
            ring.load(top, |row, out| {
                out.fill(row);
                convolved += 1;
            });
            assert_eq!(convolved, new_rows, "top {top}");
            let held = ring.rows().map(|row| {
                assert!(row.iter().all(|&sample| sample == row[0]), "top {top}");
                row[0]
            });
            assert_eq!(held, [top, top + 1, top + 2, top + 3]);
        }
    }
}

//! Tile edges blended by coverage. A tile at a fractional position only partly
//! covers the canvas pixels along its edges: some of their taps lie outside
//! the tile. Each such pixel gets every tile's partial sum (the taps inside
//! it) and the canvas keeps the weight no tile covers, so where two tiles
//! butt, the pixel is the resampling of both together, as if one image.
//!
//! A pixel between tiles along both axes (a seam, not the mosaic's outer
//! edge) takes no canvas: what weight the tiles miss there is a gap between
//! them narrower than the taps, and the tiles fill it (`resolve_pixel`).
//!
//! The interior pass (`compose_tile_interior`) runs after this one and writes
//! only pixels with every tap inside a tile, last write winning, so a pixel
//! one tile covers fully is that tile's, whatever the edges of others. This
//! pass leaves out the pixels the interiors write (`covered`): tiles that
//! overlap by more than their taps cost no more than skipping the edges.

use std::ops::Range;

use super::geometry::{Canvas, Rect, Tile, cubic_weights};
use super::{Interpolation, is_copy};
use crate::sample::Sample;

/// The taps of a tile along one axis: canvas pixel `c` reads tile samples
/// `c - origin + first_offset ..`, one per weight.
struct Axis {
    origin: isize,
    len: isize,
    first_offset: isize,
    weights: [f32; 4],
    tap_count: usize,
}

impl Axis {
    /// The taps the interior kernels use (`Placement`): bilinear at 0 and +1,
    /// bicubic at -1 ..= +2, around `ceil(position)`, with the same f32
    /// weights. An integer position is a copy along the axis: one tap.
    fn new(position: f64, len: usize, filter: Interpolation) -> Axis {
        let phase = (position.ceil() - position) as f32;
        let (first_offset, weights, tap_count) = if position.fract() == 0.0 {
            (0, [1.0, 0.0, 0.0, 0.0], 1)
        } else if filter == Interpolation::Bilinear {
            (0, [1.0 - phase, phase, 0.0, 0.0], 2)
        } else {
            (-1, cubic_weights(phase), 4)
        };
        Axis {
            origin: position.ceil() as isize,
            len: len as isize,
            first_offset,
            weights,
            tap_count,
        }
    }

    fn last_offset(&self) -> isize {
        self.first_offset + self.tap_count as isize - 1
    }

    /// Canvas pixels with at least one tap inside the tile, clipped to
    /// `0..canvas_len`.
    fn footprint(&self, canvas_len: usize) -> Range<isize> {
        let start = (self.origin - self.last_offset()).max(0);
        let end = (self.origin + self.len - self.first_offset).min(canvas_len as isize);
        start..end.max(start)
    }

    /// Canvas pixels with every tap inside the tile, clipped to `within`.
    fn interior(&self, within: &Range<isize>) -> Range<isize> {
        let start = (self.origin - self.first_offset).clamp(within.start, within.end);
        let end = (self.origin + self.len - self.last_offset()).clamp(start, within.end);
        start..end
    }

    /// Tile index of canvas pixel `pixel`'s first tap.
    fn first_tap(&self, pixel: isize) -> isize {
        pixel - self.origin + self.first_offset
    }

    /// The taps of canvas pixel `pixel` that lie inside the tile, as (tile
    /// index, weight), how many, and their total weight.
    fn taps_inside(&self, pixel: isize) -> ([(usize, f32); 4], usize, f32) {
        let mut taps = [(0, 0.0); 4];
        let (mut count, mut total) = (0, 0.0);
        let first = self.first_tap(pixel);
        for (tap, &weight) in self.weights[..self.tap_count].iter().enumerate() {
            let index = first + tap as isize;
            if (0..self.len).contains(&index) {
                taps[count] = (index as usize, weight);
                count += 1;
                total += weight;
            }
        }
        (taps, count, total)
    }
}

/// The taps of one canvas pixel along an axis that lie inside the tile.
#[derive(Clone, Copy)]
struct Taps {
    taps: [(usize, f32); 4],
    count: usize,
    weight: f32,
}

impl Taps {
    fn inside(axis: &Axis, pixel: isize) -> Taps {
        let (taps, count, weight) = axis.taps_inside(pixel);
        Taps {
            taps,
            count,
            weight,
        }
    }

    fn as_slice(&self) -> &[(usize, f32)] {
        &self.taps[..self.count]
    }
}

/// Rows ahead the column pass prefetches the canvas: down a column the
/// pixels lie a canvas row apart, further than the hardware prefetcher
/// follows.
const PREFETCH_ROWS: usize = 8;

/// Zero rows around a band column's x sums (`add_band_column`): as many as
/// y taps can fall outside the tile, three for bicubic.
const COLUMN_PAD: usize = 3;

/// From which sides tiles reach a pixel, a bit per axis and side: the pixel
/// inside a tile's interior along the axis, or before it (the tile lies
/// after the pixel), or after it.
mod sides {
    pub(super) const X_INSIDE: u8 = 1;
    pub(super) const X_TILE_AFTER: u8 = 2;
    pub(super) const X_TILE_BEFORE: u8 = 4;
    pub(super) const Y_INSIDE: u8 = X_INSIDE << 3;
    pub(super) const Y_TILE_AFTER: u8 = X_TILE_AFTER << 3;
    pub(super) const Y_TILE_BEFORE: u8 = X_TILE_BEFORE << 3;

    /// Whether tiles cover the pixel along both axes: inside one, or reaching
    /// it from both sides.
    pub(super) fn between_tiles(sides: u8) -> bool {
        let covered = |axis: u8| {
            axis & X_INSIDE != 0
                || axis & (X_TILE_AFTER | X_TILE_BEFORE) == X_TILE_AFTER | X_TILE_BEFORE
        };
        covered(sides & 7) && covered(sides >> 3)
    }

    /// The side bit along y of canvas row `cy` for a tile with interior rows
    /// `interior`.
    pub(super) fn y_side(cy: isize, interior: &std::ops::Range<isize>) -> u8 {
        if interior.contains(&cy) {
            Y_INSIDE
        } else if cy < interior.start {
            Y_TILE_AFTER
        } else {
            Y_TILE_BEFORE
        }
    }
}

/// Least summed weight with which tiles fill a gap between them: below it
/// (bicubic's negative lobes reaching across a gap of pixels) the canvas
/// shows through.
const GAP_FILL_MIN_WEIGHT: f32 = 0.25;

/// The rectangles the interior pass writes, filed under every square canvas
/// cell they reach, so a line finds the ones crossing it without reading
/// them all (thousands of tiny tiles make that quadratic). A cell is about
/// a rectangle across, so each rectangle is in a few cells and each cell
/// holds a few rectangles.
struct Covered<'r> {
    rects: &'r [Rect],
    cell: isize,
    columns: usize,
    /// Where each cell's rectangles start in `indices`, and where the last
    /// ends.
    starts: Vec<u32>,
    /// The rectangles of each cell in turn, as indices into `rects`.
    indices: Vec<u32>,
}

impl<'r> Covered<'r> {
    fn new(rects: &'r [Rect], width: usize, height: usize) -> Self {
        let extent: isize = rects
            .iter()
            .map(|rect| rect.x1 - rect.x0 + rect.y1 - rect.y0 + 2)
            .sum();
        let cell = (extent / (2 * rects.len().max(1) as isize)).clamp(4, 1024);
        let columns = (width as isize + cell - 1) / cell;
        let rows = (height as isize + cell - 1) / cell;
        let cells_of = |rect: Rect| {
            (rect.y0 / cell..=rect.y1 / cell).flat_map(move |row| {
                (rect.x0 / cell..=rect.x1 / cell)
                    .map(move |column| (row * columns + column) as usize)
            })
        };
        let mut starts = vec![0u32; (columns * rows) as usize + 1];
        for rect in rects {
            for at in cells_of(*rect) {
                starts[at + 1] += 1;
            }
        }
        for at in 1..starts.len() {
            starts[at] += starts[at - 1];
        }
        let mut filled = starts.clone();
        let mut indices = vec![0u32; starts[starts.len() - 1] as usize];
        for (index, rect) in rects.iter().enumerate() {
            for at in cells_of(*rect) {
                indices[filled[at] as usize] = index as u32;
                filled[at] += 1;
            }
        }
        Covered {
            rects,
            cell,
            columns: columns as usize,
            starts,
            indices,
        }
    }

    /// The rectangles that meet the canvas pixels `columns` x `rows` (both
    /// nonempty), into `near`. A rectangle in several of their cells comes
    /// once per cell.
    fn find(&self, columns: &Range<isize>, rows: &Range<isize>, near: &mut Vec<Rect>) {
        near.clear();
        for row in rows.start / self.cell..=(rows.end - 1) / self.cell {
            for column in columns.start / self.cell..=(columns.end - 1) / self.cell {
                let at = row as usize * self.columns + column as usize;
                let indices = &self.indices[self.starts[at] as usize..self.starts[at + 1] as usize];
                near.extend(
                    indices
                        .iter()
                        .map(|&index| self.rects[index as usize])
                        .filter(|rect| {
                            rect.x0 < columns.end
                                && columns.start <= rect.x1
                                && rect.y0 < rows.end
                                && rows.start <= rect.y1
                        }),
                );
            }
        }
    }
}

/// A column past a tile's interior, with its x taps, its side and the rows
/// of it no interior writes over (`Bands::runs`).
struct Band {
    tile: usize,
    column: isize,
    taps: Taps,
    side: u8,
    rows: Range<usize>,
}

/// A row past a tile's interior, with the interior columns of it no
/// interior writes over (`Bands::runs`).
struct BandRow {
    cy: isize,
    tile: usize,
    columns: Range<usize>,
}

/// The band columns and rows of every placed tile, their runs of pixels in
/// one list: thousands of tiny tiles allocate nothing apiece.
#[derive(Default)]
struct Bands {
    columns: Vec<Band>,
    rows: Vec<BandRow>,
    runs: Vec<Range<isize>>,
    /// The covered rectangles that meet the tile being placed (`Covered::find`).
    near: Vec<Rect>,
    /// Scratch for `add_uncovered`.
    holes: Vec<Range<isize>>,
}

impl Bands {
    /// Add the parts of `span` that no rectangle of `near` holds to `runs`:
    /// along x on row `at` (`along_x`), or along y on column `at`. Their
    /// indices in `runs`.
    fn add_uncovered(&mut self, span: Range<isize>, at: isize, along_x: bool) -> Range<usize> {
        let first = self.runs.len();
        if span.is_empty() {
            return first..first;
        }
        self.holes.clear();
        self.holes.extend(self.near.iter().filter_map(|rect| {
            let (across, along) = if along_x {
                ((rect.y0, rect.y1), (rect.x0, rect.x1))
            } else {
                ((rect.x0, rect.x1), (rect.y0, rect.y1))
            };
            (across.0 <= at && at <= across.1).then_some(along.0..along.1 + 1)
        }));
        self.holes.sort_unstable_by_key(|hole| hole.start);
        let mut from = span.start;
        for hole in &self.holes {
            if hole.start >= span.end {
                break;
            }
            if hole.start > from {
                self.runs.push(from..hole.start);
            }
            from = from.max(hole.end);
            if from >= span.end {
                break;
            }
        }
        if from < span.end {
            self.runs.push(from..span.end);
        }
        first..self.runs.len()
    }
}

/// A fractionally placed tile, by the canvas pixels it touches that no
/// interior writes over (`covered`); its band columns and rows are in
/// `Bands`.
struct Placed<'t, 'a, T> {
    tile: &'t Tile<'a, T>,
    y: Axis,
    interior_rows: Range<isize>,
    interior_columns: Range<isize>,
    first_interior_tap: usize,
    x_weights: [f32; 4],
    x_tap_count: usize,
}

impl<'t, 'a, T: Sample> Placed<'t, 'a, T> {
    /// Place `tile`, the `index`th placed, adding its band columns and rows
    /// to `bands`; none if it has neither.
    fn new(
        (tile, index): (&'t Tile<'a, T>, usize),
        (px, py): (f64, f64),
        filter: Interpolation,
        canvas: &Canvas<T>,
        (covered, bands): (&Covered, &mut Bands),
    ) -> Option<Self> {
        let (x, y) = (
            Axis::new(px, tile.width, filter),
            Axis::new(py, tile.height, filter),
        );
        let (columns, rows) = (x.footprint(canvas.width), y.footprint(canvas.height));
        if columns.is_empty() || rows.is_empty() {
            return None;
        }
        let interior_columns = x.interior(&columns);
        let interior_rows = y.interior(&rows);
        let before = (columns.start..interior_columns.start).map(|cx| (cx, sides::X_TILE_AFTER));
        let after = (interior_columns.end..columns.end).map(|cx| (cx, sides::X_TILE_BEFORE));
        let counts = (bands.columns.len(), bands.rows.len());
        covered.find(&columns, &rows, &mut bands.near);
        for (cx, side) in before.chain(after) {
            let band_rows = bands.add_uncovered(rows.clone(), cx, false);
            if !band_rows.is_empty() {
                bands.columns.push(Band {
                    tile: index,
                    column: cx,
                    taps: Taps::inside(&x, cx),
                    side,
                    rows: band_rows,
                });
            }
        }
        for cy in (rows.start..interior_rows.start).chain(interior_rows.end..rows.end) {
            let columns = bands.add_uncovered(interior_columns.clone(), cy, true);
            if !columns.is_empty() {
                bands.rows.push(BandRow {
                    cy,
                    tile: index,
                    columns,
                });
            }
        }
        if counts == (bands.columns.len(), bands.rows.len()) {
            return None;
        }
        Some(Placed {
            tile,
            interior_rows,
            first_interior_tap: x.first_tap(interior_columns.start).max(0) as usize,
            interior_columns,
            x_weights: x.weights,
            x_tap_count: x.tap_count,
            y,
        })
    }

    /// Add the run `columns` of the interior columns of band row `cy`: every
    /// x tap inside, so the run is the y taps of the tile rows (into
    /// `row.vertical`), then the x taps along it, sample by sample.
    fn add_band_row<const C: usize>(&self, cy: isize, columns: Range<isize>, row: &mut RowSums<C>) {
        let y_taps = Taps::inside(&self.y, cy);
        let len = columns.len() * C;
        let x_weights = &self.x_weights[..self.x_tap_count];
        let first_tap =
            self.first_interior_tap + (columns.start - self.interior_columns.start) as usize;
        let read = first_tap * C..(first_tap + columns.len() + x_weights.len() - 1) * C;
        let vertical = &mut row.vertical;
        vertical.clear();
        vertical.resize(read.len(), 0.0);
        for &(tile_y, weight_y) in y_taps.as_slice() {
            let samples = &self.tile.row(tile_y)[read.clone()];
            for (value, &sample) in vertical.iter_mut().zip(samples) {
                *value += weight_y * sample.to_f32();
            }
        }
        let sums = &mut row.sums.as_flattened_mut()[columns.start as usize * C..][..len];
        for (tap, &weight_x) in x_weights.iter().enumerate() {
            for (sum, &value) in sums.iter_mut().zip(&vertical[tap * C..][..len]) {
                *sum += weight_x * value;
            }
        }
        let weight = y_taps.weight * x_weights.iter().sum::<f32>();
        let pixels = columns.start as usize..columns.end as usize;
        for total in &mut row.weights[pixels.clone()] {
            *total += weight;
        }
        let side = sides::X_INSIDE | sides::y_side(cy, &self.interior_rows);
        for pixel_sides in &mut row.sides[pixels.clone()] {
            *pixel_sides |= side;
        }
        row.touched.push(pixels);
    }

    /// Add `band`'s rows (`rows`, its runs) to its strip: first the x taps of
    /// the tile rows they read into `column` (between `COLUMN_PAD` zero rows,
    /// which stand for the y taps outside the tile), then the y taps down it.
    fn add_band_column<const C: usize>(
        &self,
        (band, rows): (&Band, &[Range<isize>]),
        strip: &mut Strip<C>,
        column: &mut Vec<[f32; C]>,
    ) {
        column.clear();
        column.resize(self.tile.height + 2 * COLUMN_PAD, [0.0; C]);
        let (Some(first), Some(last)) = (rows.first(), rows.last()) else {
            return;
        };
        let tile_rows = self.y.first_tap(first.start).max(0) as usize
            ..((self.y.first_tap(last.end - 1) + self.y.tap_count as isize) as usize)
                .min(self.tile.height);
        let x_taps = band.taps.as_slice();
        for tile_y in tile_rows {
            let tile_row = self.tile.row(tile_y);
            let value = &mut column[COLUMN_PAD + tile_y];
            for &(tile_x, weight_x) in x_taps {
                let samples = &tile_row[tile_x * C..][..C];
                for (value, &sample) in value.iter_mut().zip(samples) {
                    *value += weight_x * sample.to_f32();
                }
            }
        }
        let column_weight: f32 = x_taps.iter().map(|tap| tap.1).sum();
        match self.y.tap_count {
            1 => self.add_down_column::<C, 1>(column, (band.side, rows), column_weight, strip),
            2 => self.add_down_column::<C, 2>(column, (band.side, rows), column_weight, strip),
            _ => self.add_down_column::<C, 4>(column, (band.side, rows), column_weight, strip),
        }
    }

    /// The y taps (`TAPS` of them) down a column from `add_band_column`.
    fn add_down_column<const C: usize, const TAPS: usize>(
        &self,
        column: &[[f32; C]],
        (side, rows): (u8, &[Range<isize>]),
        column_weight: f32,
        strip: &mut Strip<C>,
    ) {
        let weights: [f32; TAPS] = std::array::from_fn(|tap| self.y.weights[tap]);
        let rows_weight = column_weight * weights.iter().sum::<f32>();
        for cy in rows.iter().flat_map(Clone::clone) {
            let first = (self.y.first_tap(cy) + COLUMN_PAD as isize) as usize;
            let taps: &[[f32; C]; TAPS] = column[first..][..TAPS]
                .try_into()
                .unwrap_or_else(|_| unreachable!());
            let mut values = [0.0f32; C];
            for (tap, &weight) in taps.iter().zip(&weights) {
                for (value, &sample) in values.iter_mut().zip(tap) {
                    *value += weight * sample;
                }
            }
            let weight = if self.interior_rows.contains(&cy) {
                rows_weight
            } else {
                column_weight * Taps::inside(&self.y, cy).weight
            };
            let at = cy as usize - strip.rows.start;
            strip.weights[at] += weight;
            strip.sides[at] |= side | sides::y_side(cy, &self.interior_rows);
            for (sum, value) in strip.sums[at].iter_mut().zip(values) {
                *sum += value;
            }
        }
    }
}

/// One canvas row of partial sums, `C` per pixel, and the runs of pixels
/// that have any. Runs may overlap: resolving a pixel clears its sums, and
/// a pixel with none resolves to itself.
struct RowSums<const C: usize> {
    sums: Vec<[f32; C]>,
    weights: Vec<f32>,
    touched: Vec<Range<usize>>,
    sides: Vec<u8>,
    /// A tile row's y taps, before its x taps (`add_band_row`).
    vertical: Vec<f32>,
}

/// The partial sums down one canvas column that is a band column of some
/// tile, over the rows of the tiles it is a band column of.
struct Strip<const C: usize> {
    column: usize,
    rows: Range<usize>,
    sums: Vec<[f32; C]>,
    weights: Vec<f32>,
    sides: Vec<u8>,
}

/// The strips, and for each canvas column the index of its strip, if any.
struct Strips<const C: usize> {
    strips: Vec<Strip<C>>,
    of_column: Vec<Option<usize>>,
}

impl<const C: usize> Strips<C> {
    fn new(width: usize, bands: &Bands) -> Self {
        let mut rows: Vec<Range<usize>> = vec![0..0; width];
        for band in &bands.columns {
            let runs = &bands.runs[band.rows.clone()];
            let (Some(first), Some(last)) = (runs.first(), runs.last()) else {
                continue;
            };
            let band_rows = first.start as usize..last.end as usize;
            let column = &mut rows[band.column as usize];
            *column = if Range::is_empty(column) {
                band_rows
            } else {
                column.start.min(band_rows.start)..column.end.max(band_rows.end)
            };
        }
        let mut strips = Vec::new();
        let mut of_column = vec![None; width];
        for (column, rows) in rows.into_iter().enumerate() {
            if !rows.is_empty() {
                of_column[column] = Some(strips.len());
                strips.push(Strip {
                    column,
                    sums: vec![[0.0; C]; rows.len()],
                    weights: vec![0.0; rows.len()],
                    sides: vec![0; rows.len()],
                    rows,
                });
            }
        }
        Strips { strips, of_column }
    }

    /// Move the sums of row `cy`'s pixels in a strip to the strip.
    fn take_row(&mut self, cy: usize, row: &mut RowSums<C>) {
        for strip in &mut self.strips {
            let Some(at) = cy
                .checked_sub(strip.rows.start)
                .filter(|&at| at < strip.rows.len())
            else {
                continue;
            };
            let x = strip.column;
            for (strip_sum, sum) in strip.sums[at].iter_mut().zip(&mut row.sums[x]) {
                *strip_sum += *sum;
                *sum = 0.0;
            }
            strip.weights[at] += row.weights[x];
            row.weights[x] = 0.0;
            strip.sides[at] |= row.sides[x];
            row.sides[x] = 0;
        }
    }
}

/// Blend the edges of every fractionally placed tile into `canvas` (see the
/// module docs), except the pixels the interior pass writes (`covered`, a
/// rectangle a tile). Where the tiles' weights at a pixel sum past 1 (tiles
/// overlapping by less than a pixel), the pixel is their weighted mean. On
/// an RGBX canvas (`pad` set) the tiles are RGB, and the fourth sample is
/// `pad`.
pub(crate) fn blend_edges<T: Sample>(
    canvas: &mut Canvas<T>,
    (tiles, positions): (&[Tile<T>], &[(f64, f64)]),
    filter: Interpolation,
    covered: &[Rect],
    pad: Option<T>,
) {
    let covered = Covered::new(covered, canvas.width, canvas.height);
    let mut bands = Bands::default();
    let mut placed: Vec<Placed<T>> = Vec::new();
    for (tile, &position) in tiles.iter().zip(positions) {
        if is_copy(position.0, position.1, filter) {
            continue;
        }
        let index = placed.len();
        let context = (&covered, &mut bands);
        placed.extend(Placed::new(
            (tile, index),
            position,
            filter,
            canvas,
            context,
        ));
    }
    let Some(tile_channels) = placed.first().map(|placed| placed.tile.channels) else {
        return;
    };
    match tile_channels {
        1 => blend::<T, 1>(canvas, &placed, &mut bands, pad),
        3 => blend::<T, 3>(canvas, &placed, &mut bands, pad),
        _ => blend::<T, 4>(canvas, &placed, &mut bands, pad),
    }
}

/// The edges in two passes: the tiles' band rows, row by row, then their
/// band columns, column by column, each pixel summed in one place (a band
/// row's pixels in a band column move to its strip).
fn blend<T: Sample, const C: usize>(
    canvas: &mut Canvas<T>,
    placed: &[Placed<T>],
    bands: &mut Bands,
    pad: Option<T>,
) {
    let mut strips = Strips::<C>::new(canvas.width, bands);
    // Band rows by row, then tile index, so each pixel sums its tiles in order.
    bands
        .rows
        .sort_unstable_by_key(|band_row| (band_row.cy, band_row.tile));
    let mut row = RowSums::<C> {
        sums: vec![[0.0; C]; canvas.width],
        weights: vec![0.0; canvas.width],
        touched: Vec::new(),
        sides: vec![0; canvas.width],
        vertical: Vec::new(),
    };
    for on_row in bands.rows.chunk_by(|first, second| first.cy == second.cy) {
        let cy = on_row[0].cy;
        for band_row in on_row {
            for run in &bands.runs[band_row.columns.clone()] {
                placed[band_row.tile].add_band_row(cy, run.clone(), &mut row);
            }
        }
        resolve_row(canvas, cy, &mut row, &mut strips, pad);
    }
    let mut column = Vec::new();
    for band in &bands.columns {
        let strip = &mut strips.strips
            [strips.of_column[band.column as usize].unwrap_or_else(|| unreachable!())];
        let rows = &bands.runs[band.rows.clone()];
        placed[band.tile].add_band_column((band, rows), strip, &mut column);
    }
    for strip in &mut strips.strips {
        resolve_strip(canvas, strip, pad);
    }
}

/// Move the sums of canvas row `cy`'s pixels in strips to the strips, write
/// its other touched pixels from theirs, and clear them.
fn resolve_row<T: Sample, const C: usize>(
    canvas: &mut Canvas<T>,
    cy: isize,
    row: &mut RowSums<C>,
    strips: &mut Strips<C>,
    pad: Option<T>,
) {
    strips.take_row(cy as usize, row);
    let canvas_channels = canvas.channels;
    let canvas_row = canvas.run(0, cy, canvas.width);
    for run in row.touched.drain(..) {
        let pixels = canvas_row[run.start * canvas_channels..run.end * canvas_channels]
            .chunks_exact_mut(canvas_channels);
        let sums = &mut row.sums[run.clone()];
        let weights = &mut row.weights[run.clone()];
        let sides = &mut row.sides[run];
        for (((pixel, sums), weight), sides) in pixels.zip(sums).zip(weights).zip(sides) {
            resolve_pixel(pixel, (sums, weight, sides), pad);
        }
    }
}

/// Write the pixels of a strip from their sums.
fn resolve_strip<T: Sample, const C: usize>(
    canvas: &mut Canvas<T>,
    strip: &mut Strip<C>,
    pad: Option<T>,
) {
    let channels = canvas.channels;
    let stride = canvas.width * channels;
    let first = strip.rows.start * stride + strip.column * channels;
    let samples = &mut canvas.px[first..];
    let rows = strip
        .sums
        .iter_mut()
        .zip(&mut strip.weights)
        .zip(&mut strip.sides);
    for (row, ((sums, weight), sides)) in rows.enumerate() {
        let ahead = (row + PREFETCH_ROWS) * stride;
        crate::cpu::prefetch_for_read(samples.as_ptr().wrapping_add(ahead) as *const u8);
        let pixel = &mut samples[row * stride..][..channels];
        resolve_pixel(pixel, (sums, weight, sides), pad);
    }
}

/// Write a pixel from its sums, weight and sides, and clear them.
#[inline(always)]
fn resolve_pixel<T: Sample, const C: usize>(
    pixel: &mut [T],
    (sums, weight, sides): (&mut [f32; C], &mut f32, &mut u8),
    pad: Option<T>,
) {
    // Covered past 1 (sub-pixel overlaps), or between tiles (a gap they
    // fill): the tiles' weighted mean; otherwise the canvas keeps the weight
    // no tile covers.
    let fills_gap = sides::between_tiles(*sides) && *weight >= GAP_FILL_MIN_WEIGHT;
    *sides = 0;
    let (scale, keep) = if *weight >= 1.0 || fills_gap {
        (1.0 / *weight, 0.0)
    } else {
        (1.0, 1.0 - *weight)
    };
    for (sample, sum) in pixel.iter_mut().zip(sums.iter_mut()) {
        // Not `keep * sample` when nothing is kept: a float canvas may
        // hold inf or NaN.
        let kept = if keep == 0.0 {
            0.0
        } else {
            keep * sample.to_f32()
        };
        *sample = T::from_f32(*sum * scale + kept);
        *sum = 0.0;
    }
    if let Some(pad) = pad {
        pixel[C..].fill(pad);
    }
    *weight = 0.0;
}

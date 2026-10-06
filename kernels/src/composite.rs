//! Tile compositing: paste (integer positions) or resample (fractional) each
//! tile onto the canvas, planar/HWC, plus the RGB -> RGBX variant.

mod edges;
pub(crate) mod geometry;
#[cfg(target_arch = "aarch64")]
mod neon;
pub(crate) mod rows;
#[cfg(target_arch = "x86_64")]
mod x86_rows;
#[cfg(target_arch = "x86_64")]
mod x86_separable;

use std::ops::Range;

#[cfg(target_arch = "aarch64")]
use crate::cpu::use_neon;
#[cfg(target_arch = "x86_64")]
use crate::cpu::{has_avx2_fma, has_avx512};
use crate::rgbx::expand_rgb_rgbx;
use crate::sample::RGB;
use geometry::{Canvas, Columns, EDGE_PAD_LEFT, HconvRing, Placement, Rect, Tile, cubic_weights};
use rows::{ComposeSample, separable_bicubic, vconv4_rgbx};

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Interpolation {
    Nearest,
    Bilinear,
    Bicubic,
}

/// Compose `tiles` onto `canvas` at `positions`: the pixels fractionally
/// placed tiles cover only partly blended by coverage (`edges`), then the
/// interiors in order, last write winning.
pub(crate) fn compose_tiles<T: ComposeSample>(
    canvas: &mut Canvas<T>,
    tiles: &[Tile<T>],
    positions: &[(f64, f64)],
    filter: Interpolation,
) {
    let inset = interior_inset(filter);
    let covered = interiors(canvas, tiles, positions, filter, inset);
    edges::blend_edges(canvas, (tiles, positions), filter, &covered, None);
    for (tile, &(px, py)) in tiles.iter().zip(positions) {
        compose_tile_interior(canvas, tile, px, py, filter, inset);
    }
}

/// `compose_tiles` of RGB tiles onto an RGBX canvas (see `compose_tile_rgbx`).
pub(crate) fn compose_tiles_rgbx(
    canvas: &mut Canvas<u8>,
    tiles: &[Tile<u8>],
    positions: &[(f64, f64)],
    filter: Interpolation,
    pad: u8,
) {
    let inset = interior_inset(filter);
    let covered = interiors(canvas, tiles, positions, filter, inset);
    edges::blend_edges(canvas, (tiles, positions), filter, &covered, Some(pad));
    let mut scratch = Vec::new();
    for (tile, &(px, py)) in tiles.iter().zip(positions) {
        compose_tile_rgbx_interior(canvas, tile, (px, py), filter, pad, inset, &mut scratch);
    }
}

/// Bicubic's interior keeps all four taps inside the tile (the edge pass
/// has the pixels whose outer taps fall outside).
fn interior_inset(filter: Interpolation) -> isize {
    isize::from(filter == Interpolation::Bicubic)
}

/// The canvas pixels the interior pass writes, a rectangle a tile (none for
/// a tile off the canvas): the pasted tile for a copy, else `covered` of
/// `Placement::inset` by `inset`, as `compose_tile_interior` takes them.
fn interiors<T, U>(
    canvas: &Canvas<U>,
    tiles: &[Tile<T>],
    positions: &[(f64, f64)],
    filter: Interpolation,
    inset: isize,
) -> Vec<Rect> {
    tiles
        .iter()
        .zip(positions)
        .filter_map(|(tile, &(px, py))| {
            if is_copy(px, py, filter) {
                let (ox, oy) = (px.round() as isize, py.round() as isize);
                Rect {
                    x0: ox.max(0),
                    x1: (ox + tile.width as isize).min(canvas.width as isize) - 1,
                    y0: oy.max(0),
                    y1: (oy + tile.height as isize).min(canvas.height as isize) - 1,
                }
                .nonempty()
            } else {
                Placement::inset(px, py, tile, canvas, inset).map(|placement| placement.covered)
            }
        })
        .collect()
}

/// Paste `tile` with its top-left at integer canvas position (ox, oy), handing
/// `put` each visible tile-row segment with the canvas segment it lands on.
fn paste_aligned<T, U: Copy>(
    canvas: &mut Canvas<T>,
    tile: &Tile<U>,
    ox: isize,
    oy: isize,
    mut put: impl FnMut(&mut [T], &[U]),
) {
    let x0 = ox.max(0);
    let x1 = (ox + tile.width as isize).min(canvas.width as isize);
    if x1 <= x0 {
        return;
    }
    let (skip, count) = ((x0 - ox) as usize, (x1 - x0) as usize);
    for cy in oy.max(0)..(oy + tile.height as isize).min(canvas.height as isize) {
        let row = tile.row((cy - oy) as usize);
        put(
            canvas.run(x0, cy, count),
            &row[skip * tile.channels..(skip + count) * tile.channels],
        );
    }
}

/// Integer positions copy; so does `Nearest` anywhere, since nearest-neighbour
/// resampling of a pure translation is a copy from the rounded position.
fn is_copy(px: f64, py: f64, filter: Interpolation) -> bool {
    filter == Interpolation::Nearest || (px.fract() == 0.0 && py.fract() == 0.0)
}

/// Paste one tile onto a same-layout canvas at (px, py). Last write wins.
#[cfg(test)]
pub(crate) fn compose_tile<T: ComposeSample>(
    canvas: &mut Canvas<T>,
    tile: &Tile<T>,
    px: f64,
    py: f64,
    filter: Interpolation,
) {
    compose_tile_interior(canvas, tile, px, py, filter, 0);
}

/// Paste one tile onto a same-layout canvas at (px, py), writing the pixels
/// fully backed by the tile (`Placement::inset` by `inset`). Last write wins.
fn compose_tile_interior<T: ComposeSample>(
    canvas: &mut Canvas<T>,
    tile: &Tile<T>,
    px: f64,
    py: f64,
    filter: Interpolation,
    inset: isize,
) {
    if is_copy(px, py, filter) {
        paste_aligned(
            canvas,
            tile,
            px.round() as isize,
            py.round() as isize,
            |dst, src| dst.copy_from_slice(src),
        );
        return;
    }
    let Some(placement) = Placement::inset(px, py, tile, canvas, inset) else {
        return;
    };
    if filter == Interpolation::Bilinear {
        if !T::compose_bilinear_tile(canvas, tile, &placement) {
            compose_bilinear(canvas, tile, &placement);
        }
    } else {
        compose_bicubic(canvas, tile, &placement);
    }
}

/// Paste one RGB tile onto an RGBX canvas at (px, py), expanding as it writes
/// (`pad` fills the fourth sample). Pillow stores RGB as 4-byte RGBX, so such
/// a canvas maps into a Pillow image with no copy, and fusing the expand into
/// the compose write replaces a separate full-canvas expand pass.
#[cfg(test)]
pub(crate) fn compose_tile_rgbx(
    canvas: &mut Canvas<u8>,
    tile: &Tile<u8>,
    px: f64,
    py: f64,
    filter: Interpolation,
    pad: u8,
    scratch: &mut Vec<u8>,
) {
    compose_tile_rgbx_interior(canvas, tile, (px, py), filter, pad, 0, scratch);
}

/// `compose_tile_rgbx`, `covered` inset as in `compose_tile_interior`.
fn compose_tile_rgbx_interior(
    canvas: &mut Canvas<u8>,
    tile: &Tile<u8>,
    (px, py): (f64, f64),
    filter: Interpolation,
    pad: u8,
    inset: isize,
    scratch: &mut Vec<u8>,
) {
    if is_copy(px, py, filter) {
        paste_aligned(
            canvas,
            tile,
            px.round() as isize,
            py.round() as isize,
            |dst, src| expand_rgb_rgbx(src, dst, src.len() / RGB, pad),
        );
        return;
    }
    let Some(placement) = Placement::inset(px, py, tile, canvas, inset) else {
        return;
    };
    if filter == Interpolation::Bilinear {
        compose_bilinear_rgbx(canvas, tile, &placement, pad, scratch);
    } else {
        compose_bicubic_rgbx(canvas, tile, &placement, pad, scratch);
    }
}

/// Planar (channels = 1) and interleaved rows alike: an interleaved row is a
/// planar row whose same-channel neighbour is `channels` elements away, and every sample of
/// a pixel shares that pixel's weights.
///
/// The taps stay inside the tile rows (the row kernels' contract): `covered`
/// ends at canvas column `floor(px) + tile.width - 1` and
/// `src_x(cx) = cx - ceil(px)`, so the last output's left tap is at most
/// column `tile.width - 2` when `fx != 0` (the right tap, `right_tap_step` =
/// `channels` later, is then column `tile.width - 1`) and at most
/// `tile.width - 1` when `fx == 0` (step 0). Each row is
/// `tile.width * channels` elements long.
fn compose_bilinear<T: ComposeSample>(
    canvas: &mut Canvas<T>,
    tile: &Tile<T>,
    placement: &Placement,
) {
    let channels = tile.channels;
    let span = placement.covered;
    let (src_x0, step, weights) = (
        placement.src_x(span.x0) * channels,
        placement.right_tap_step(channels),
        placement.bilinear_weights(),
    );
    for cy in span.y0..=span.y1 {
        canvas.prefetch_next_row(span.x0, cy, span.y1, span.width());
        let (top, bottom) = placement.bilinear_rows(cy, tile.height);
        let dst = canvas.run(span.x0, cy, span.width());
        // SAFETY: taps inside the rows (see above); `dst` holds the run's
        // `span.width() * channels` elements.
        unsafe {
            T::bilinear_row(
                tile.row(top),
                tile.row(bottom),
                dst,
                src_x0,
                span.width() * channels,
                step,
                weights,
            );
        }
    }
}

/// `compose_bilinear` for uint16, each tile row the runs read widened to
/// f32 once into a two-row ring (a canvas row's taps are rows `top` and
/// `top + 1`, or one row twice: row parity picks the slot), so a sample is
/// widened once rather than once per tap.
///
/// # Safety
/// The CPU has the tier's kernels: NEON, or AVX2 and FMA.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) unsafe fn compose_bilinear_widened(
    canvas: &mut Canvas<u16>,
    tile: &Tile<u16>,
    placement: &Placement,
) {
    #[cfg(target_arch = "aarch64")]
    use neon::{
        bilinear_row_widened_u16_neon as bilinear_row_widened, widen_u16_row_neon as widen_row,
    };
    #[cfg(target_arch = "x86_64")]
    use x86_rows::{
        bilinear_row_widened_u16_avx2 as bilinear_row_widened, widen_u16_row_avx2 as widen_row,
    };
    let channels = tile.channels;
    let span = placement.covered;
    let (src_x0, step, weights) = (
        placement.src_x(span.x0) * channels,
        placement.right_tap_step(channels),
        placement.bilinear_weights(),
    );
    let len = span.width() * channels;
    // The samples a run reads from a tile row: its taps, `step` past the run.
    let read = src_x0..src_x0 + len + step;
    let mut ring = [vec![0f32; read.len()], vec![0f32; read.len()]];
    let mut held = [usize::MAX; 2];
    for cy in span.y0..=span.y1 {
        canvas.prefetch_next_row(span.x0, cy, span.y1, span.width());
        let (top, bottom) = placement.bilinear_rows(cy, tile.height);
        for row in [top, bottom] {
            if held[row % 2] != row {
                let (samples, widened) = (&tile.row(row)[read.clone()], &mut ring[row % 2]);
                // SAFETY: the CPU has the tier's features (the caller's
                // contract).
                unsafe { widen_row(samples, widened) };
                held[row % 2] = row;
            }
        }
        let dst = canvas.run(span.x0, cy, span.width());
        // SAFETY: the CPU has the tier's features (the caller's contract);
        // the taps stay inside the widened `read` samples (see
        // `compose_bilinear`), and `dst` holds the run's `len` elements.
        unsafe {
            bilinear_row_widened(
                &ring[top % 2],
                &ring[bottom % 2],
                dst,
                0,
                len,
                step,
                weights,
            );
        }
    }
}

fn compose_bilinear_rgbx(
    canvas: &mut Canvas<u8>,
    tile: &Tile<u8>,
    placement: &Placement,
    pad: u8,
    scratch: &mut Vec<u8>,
) {
    let span = placement.covered;
    let width = span.width();
    let (src_x0, step, weights) = (
        placement.src_x(span.x0) * RGB,
        placement.right_tap_step(RGB),
        placement.bilinear_weights(),
    );
    scratch.clear();
    scratch.resize(width * RGB, 0);
    for cy in span.y0..=span.y1 {
        canvas.prefetch_next_row(span.x0, cy, span.y1, width);
        let (row0, row1) = {
            let (top, bottom) = placement.bilinear_rows(cy, tile.height);
            (tile.row(top), tile.row(bottom))
        };
        let dst = canvas.run(span.x0, cy, width);
        #[cfg(target_arch = "x86_64")]
        if has_avx512() {
            // SAFETY: avx512f+bw detected; taps inside the rows (see
            // `compose_bilinear`), `dst` holds `width` RGBX pixels.
            unsafe {
                x86_rows::bilinear_row_rgb_to_rgbx_fx_avx512(
                    row0, row1, dst, src_x0, width, step, pad, weights,
                );
            };
            continue;
        }
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            // SAFETY: avx2+fma detected; taps inside the rows (see
            // `compose_bilinear`), `dst` holds `width` RGBX pixels.
            unsafe {
                x86_rows::bilinear_row_rgb_to_rgbx_fx_avx2(
                    row0, row1, dst, src_x0, width, step, pad, weights,
                );
            };
            continue;
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon() {
            // SAFETY: NEON is baseline on aarch64; taps inside the rows (see
            // `compose_bilinear`), `dst` holds `width` RGBX pixels.
            unsafe {
                neon::bilinear_row_rgb_to_rgbx_fx_neon(
                    row0, row1, dst, src_x0, width, step, pad, weights,
                );
            };
            continue;
        }
        // SAFETY: taps inside the rows (see `compose_bilinear`);
        // `scratch` holds `width * RGB` elements.
        unsafe {
            <u8 as ComposeSample>::bilinear_row(
                row0,
                row1,
                scratch,
                src_x0,
                width * RGB,
                step,
                weights,
            );
        }
        expand_rgb_rgbx(scratch, dst, width, pad);
    }
}

/// Bicubic over every covered pixel, through the row kernels. Taps past the
/// tile's edge replicate the edge: rows by clamping the row index, columns by
/// `over_columns` — so the edge band takes the same (SIMD) path as the rest.
fn compose_bicubic<T: ComposeSample>(
    canvas: &mut Canvas<T>,
    tile: &Tile<T>,
    placement: &Placement,
) {
    let channels = tile.channels;
    let (wx, wy) = (cubic_weights(placement.fx), cubic_weights(placement.fy));
    let cols = placement.bicubic_columns(tile.width);
    let mut edge_bufs = EdgeBufs::default();
    let mut ring = (T::SEP_BICUBIC && separable_bicubic())
        .then(|| HconvRing::<T::HAcc>::new(cols.count * channels));
    for cy in placement.covered.y0..=placement.covered.y1 {
        canvas.prefetch_next_row(placement.covered.x0, cy, placement.covered.y1, cols.count);
        let top = placement.bicubic_top(cy);
        let dst = canvas.run(placement.covered.x0, cy, cols.count);
        match &mut ring {
            Some(ring) => {
                ring.load(top, |y, out| {
                    hconv_covered(tile, y, &cols, wx, &mut edge_bufs, out);
                });
                // SAFETY: the ring's rows and `dst` hold `cols.count * channels`
                // elements each.
                unsafe { T::vconv4_row(ring.rows(), dst, cols.count * channels, wy) };
            }
            None => bicubic_covered(tile, top, &cols, (wx, wy), &mut edge_bufs, dst),
        }
    }
}

fn compose_bicubic_rgbx(
    canvas: &mut Canvas<u8>,
    tile: &Tile<u8>,
    placement: &Placement,
    pad: u8,
    scratch: &mut Vec<u8>,
) {
    let (wx, wy) = (cubic_weights(placement.fx), cubic_weights(placement.fy));
    let cols = placement.bicubic_columns(tile.width);
    let mut edge_bufs = EdgeBufs::default();
    let mut ring = (<u8 as ComposeSample>::SEP_BICUBIC && separable_bicubic())
        .then(|| HconvRing::<i16>::new(cols.count * RGB));
    scratch.clear();
    scratch.resize(cols.count * RGB, 0);
    for cy in placement.covered.y0..=placement.covered.y1 {
        canvas.prefetch_next_row(placement.covered.x0, cy, placement.covered.y1, cols.count);
        let top = placement.bicubic_top(cy);
        let dst = canvas.run(placement.covered.x0, cy, cols.count);
        if let Some(ring) = &mut ring {
            ring.load(top, |y, out| {
                hconv_covered(tile, y, &cols, wx, &mut edge_bufs, out);
            });
            // SAFETY: the ring's rows hold `cols.count * RGB` elements, `dst`
            // `cols.count` RGBX pixels.
            unsafe { vconv4_rgbx(ring.rows(), dst, cols.count, pad, wy) };
        } else {
            bicubic_covered(tile, top, &cols, (wx, wy), &mut edge_bufs, scratch);
            expand_rgb_rgbx(scratch, dst, cols.count, pad);
        }
    }
}

/// Reusable buffers for the edge-replicated source rows of `over_columns`.
type EdgeBufs<T> = [Vec<T>; 4];

/// Longest row (in elements) for which `hconv_covered` copies the whole row
/// edge-padded and makes one kernel call; longer rows are convolved in place
/// with separate calls for the edge columns. Measured crossover: padding wins
/// up to ~1k elements (512 px planar -18%, 256 px RGB -12%) and loses on long
/// rows (2048 px RGB +19%), where the copy outweighs the two small calls.
const PAD_WHOLE_ROW_MAX_ELEMENTS: usize = 1024;

/// The separable horizontal pass of (logical) tile row `y` over every covered
/// column, into `out`.
fn hconv_covered<T: ComposeSample>(
    tile: &Tile<T>,
    y: isize,
    cols: &Columns,
    wx: [f32; 4],
    edge_bufs: &mut EdgeBufs<T>,
    out: &mut [T::HAcc],
) {
    let channels = tile.channels;
    if tile.width * channels <= PAD_WHOLE_ROW_MAX_ELEMENTS {
        let padded = &mut edge_bufs[0];
        tile.edge_padded_row(y, padded);
        // SAFETY: the padded row holds tile.width + 3 pixels, column x at
        // x + EDGE_PAD_LEFT; the covered columns' x0 taps are first_x0 ..
        // first_x0 + count <= tile.width (see `Placement::bicubic_columns`), so
        // the taps x0 - 1 ..= x0 + 2 lie in padded pixels 0 ..= tile.width + 2.
        // `out` is a ring row of `cols.count * channels` elements.
        unsafe {
            T::hconv4_row(
                padded,
                out,
                (cols.first_x0 + EDGE_PAD_LEFT) * channels,
                cols.count * channels,
                channels,
                wx,
            );
        }
        return;
    }
    let [edge_buf, ..] = edge_bufs;
    over_columns(
        tile,
        [tile.row_clamped(y)],
        cols,
        std::array::from_mut(edge_buf),
        |[src], src_x0, px| {
            // SAFETY: `over_columns` keeps every tap inside `src`; the output
            // slice holds `px.len() * channels` elements.
            unsafe {
                T::hconv4_row(
                    src,
                    &mut out[px.start * channels..px.end * channels],
                    src_x0,
                    px.len() * channels,
                    channels,
                    wx,
                );
            }
        },
    );
}

/// The direct 4x4 pass of the output row whose top tap is (logical) tile row
/// `top`, over every covered column, into `dst`.
fn bicubic_covered<T: ComposeSample>(
    tile: &Tile<T>,
    top: isize,
    cols: &Columns,
    (wx, wy): ([f32; 4], [f32; 4]),
    edge_bufs: &mut EdgeBufs<T>,
    dst: &mut [T],
) {
    let channels = tile.channels;
    over_columns(
        tile,
        tile.rows4_clamped(top),
        cols,
        edge_bufs,
        |rows, src_x0, px| {
            // SAFETY: `over_columns` keeps every tap inside `rows`; the output
            // slice holds `px.len() * channels` elements.
            unsafe {
                T::bicubic_row(
                    rows,
                    &mut dst[px.start * channels..px.end * channels],
                    src_x0,
                    px.len() * channels,
                    channels,
                    wx,
                    wy,
                );
            }
        },
    );
}

/// Run a 4-tap horizontal row kernel over every covered column of an output
/// row: `kernel(rows, src_x0, px)` fills output pixels `px`, reading `rows`
/// with the first pixel's x0 tap at element `src_x0`. Columns whose taps are
/// inside the tile read `rows` directly; the few at the edges read
/// edge-replicated copies. Either way every 4-tap window of `px` (elements
/// `src_x0 - channels ..= src_x0 + (px.len() + 2) * channels - 1`)
/// lies inside every row handed to `kernel`, which bicubic row kernels rely on:
/// `cols.inside` holds exactly the pixels with taps `0 ..= tile.width - 1`,
/// and an edge copy holds `px.len() + 3` pixels with the first x0 tap at
/// pixel 1.
fn over_columns<T: ComposeSample, const R: usize>(
    tile: &Tile<T>,
    rows: [&[T]; R],
    cols: &Columns,
    edge_bufs: &mut [Vec<T>; R],
    mut kernel: impl FnMut([&[T]; R], usize, Range<usize>),
) {
    if !cols.inside.is_empty() {
        kernel(
            rows,
            (cols.first_x0 + cols.inside.start) * tile.channels,
            cols.inside.clone(),
        );
    }
    for px in cols.edges() {
        if px.is_empty() {
            continue;
        }
        for (buf, row) in edge_bufs.iter_mut().zip(rows) {
            tile.edge_replicated(row, cols.first_x0 + px.start, px.len(), buf);
        }
        kernel(edge_bufs.each_ref().map(Vec::as_slice), tile.channels, px);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composite::rows::tests::{
        BICUBIC_ROUNDING_STEPS, BILINEAR_ROUNDING_STEPS, TestSample, bilinear_weights, l1, test_row,
    };
    use crate::sample::RGBX;
    use crate::testing::{CHANNELS, Rng, random_samples};

    const FILTERS: [Interpolation; 3] = [
        Interpolation::Nearest,
        Interpolation::Bilinear,
        Interpolation::Bicubic,
    ];

    /// Tile (width, height)s: single pixels, rows shorter than the bicubic's
    /// four taps, widths around the vector blocks, and rows past
    /// `PAD_WHOLE_ROW_MAX_ELEMENTS` (300 px at 4 channels, 1030 planar), whose
    /// horizontal pass runs in place with separate edge calls.
    const TILE_SIZES: [(usize, usize); 9] = [
        (1, 1),
        (2, 2),
        (3, 5),
        (5, 3),
        (17, 4),
        (40, 3),
        (70, 2),
        (300, 2),
        (1030, 1),
    ];

    /// Canvas margin around a tile: room for interior positions while tiles
    /// can still hang off every edge.
    const MARGIN: usize = 3;

    /// Positions along one axis of a tile `tile_len` long on a canvas
    /// `canvas_len` long: integer and fractional, inside, flush with either
    /// end, hanging off either end by part of the tile, and fully off.
    fn axis_positions(tile_len: usize, canvas_len: usize) -> [f64; 10] {
        let (tile, canvas) = (tile_len as f64, canvas_len as f64);
        [
            1.0 - tile,
            -0.5 * tile - 0.3,
            0.0,
            2.0,
            1.7,
            canvas - tile,
            canvas - tile + 0.5,
            canvas - (0.5 * tile).floor(),
            canvas - 1.35,
            -tile - 0.25,
        ]
    }

    fn is_integer(position: f64) -> bool {
        position.fract() == 0.0
    }

    /// A row-major image: `channels` samples per pixel.
    struct Image<T> {
        samples: Vec<T>,
        width: usize,
        height: usize,
        channels: usize,
    }

    impl<T: TestSample> Image<T> {
        /// Test samples, then samples no correct compose reads past the end.
        fn random(rng: &mut Rng, (width, height): (usize, usize), channels: usize) -> Self {
            let extremes = rng.next_u64().is_multiple_of(3);
            Image {
                samples: test_row(rng, width * height * channels, extremes),
                width,
                height,
                channels,
            }
        }

        fn len(&self) -> usize {
            self.width * self.height * self.channels
        }

        fn tile(&self) -> Tile<'_, T> {
            Tile::new(
                &self.samples[..self.len()],
                self.width,
                self.height,
                self.channels,
            )
        }

        /// The sample at (x, y) with coordinates clamped into the image.
        fn clamped(&self, x: isize, y: isize, channel: usize) -> f32 {
            let x = x.clamp(0, self.width as isize - 1) as usize;
            let y = y.clamp(0, self.height as isize - 1) as usize;
            self.samples[(y * self.width + x) * self.channels + channel].to_f32()
        }
    }

    /// What the scalar reference rows compute for canvas pixel (cx, cy),
    /// sample by sample, or `None` where the compose leaves the canvas: the
    /// definition of compose written out per pixel, with the scalar kernels'
    /// association order and edge replication by clamping.
    fn reference_pixel<T: TestSample>(
        tile: &Image<T>,
        (px, py): (f64, f64),
        filter: Interpolation,
        (cx, cy): (isize, isize),
    ) -> Option<Vec<T>> {
        let (tile_width, tile_height) = (tile.width as isize, tile.height as isize);
        if filter == Interpolation::Nearest || (is_integer(px) && is_integer(py)) {
            let (x, y) = (cx - px.round() as isize, cy - py.round() as isize);
            let inside = (0..tile_width).contains(&x) && (0..tile_height).contains(&y);
            return inside.then(|| {
                let start = (y as usize * tile.width + x as usize) * tile.channels;
                tile.samples[start..start + tile.channels].to_vec()
            });
        }
        let (x_origin, y_origin) = (px.ceil() as isize, py.ceil() as isize);
        let last_x = (px + tile.width as f64 - 1.0).floor() as isize;
        let last_y = (py + tile.height as f64 - 1.0).floor() as isize;
        if !(x_origin..=last_x).contains(&cx) || !(y_origin..=last_y).contains(&cy) {
            return None;
        }
        let (x, y) = (cx - x_origin, cy - y_origin);
        let (fx, fy) = ((px.ceil() - px) as f32, (py.ceil() - py) as f32);
        let blend = |channel: usize| {
            if filter == Interpolation::Bilinear {
                let weights = bilinear_weights(fx, fy);
                return weights[0] * tile.clamped(x, y, channel)
                    + weights[1] * tile.clamped(x + 1, y, channel)
                    + weights[2] * tile.clamped(x, y + 1, channel)
                    + weights[3] * tile.clamped(x + 1, y + 1, channel);
            }
            let (wx, wy) = (cubic_weights(fx), cubic_weights(fy));
            let mut acc = 0.0f32;
            for (row_offset, row_weight) in (-1..=2).zip(wy) {
                let row = y + row_offset;
                let sum = wx[0] * tile.clamped(x - 1, row, channel)
                    + wx[1] * tile.clamped(x, row, channel)
                    + wx[2] * tile.clamped(x + 1, row, channel)
                    + wx[3] * tile.clamped(x + 2, row, channel);
                acc += row_weight * sum;
            }
            acc
        };
        Some(
            (0..tile.channels)
                .map(|channel| T::from_f32(blend(channel)))
                .collect(),
        )
    }

    /// How far a composed sample may be from the reference's.
    fn tolerance<T: TestSample>((px, py): (f64, f64), filter: Interpolation) -> f32 {
        let (fx, fy) = ((px.ceil() - px) as f32, (py.ceil() - py) as f32);
        match filter {
            _ if filter == Interpolation::Nearest || (is_integer(px) && is_integer(py)) => 0.0,
            Interpolation::Bilinear => T::tolerance(
                BILINEAR_ROUNDING_STEPS,
                T::MAX.to_f32() * l1(bilinear_weights(fx, fy)),
            ),
            _ => T::tolerance(
                BICUBIC_ROUNDING_STEPS,
                T::MAX.to_f32() * l1(cubic_weights(fx)) * l1(cubic_weights(fy)),
            ),
        }
    }

    /// Asserts `composed` (canvas `original` after composing `tile` at
    /// `position`) holds the reference pixels where the compose writes, within
    /// `tolerance`, and `original` everywhere else.
    fn assert_composed<T: TestSample>(
        composed: &Image<T>,
        original: &Image<T>,
        tile: &Image<T>,
        position: (f64, f64),
        filter: Interpolation,
        context: &str,
    ) {
        let tolerance = tolerance::<T>(position, filter);
        let channels = composed.channels;
        for (index, (pixel, original_pixel)) in composed.samples[..composed.len()]
            .chunks_exact(channels)
            .zip(original.samples.chunks_exact(channels))
            .enumerate()
        {
            let (cx, cy) = (
                (index % composed.width) as isize,
                (index / composed.width) as isize,
            );
            let (expected, tolerance) = match reference_pixel(tile, position, filter, (cx, cy)) {
                Some(expected) => (expected, tolerance),
                None => (original_pixel.to_vec(), 0.0),
            };
            for (&sample, &expected) in pixel.iter().zip(&expected) {
                assert!(
                    sample.distance(expected) <= tolerance,
                    "{context}: pixel ({cx}, {cy}): {pixel:?} vs {expected:?}"
                );
            }
        }
    }

    fn assert_compose_tile_matches_reference<T: TestSample>(seed: u64) {
        let mut rng = Rng::new(seed);
        for (tile_width, tile_height) in TILE_SIZES {
            let canvas_size = (tile_width + MARGIN, tile_height + MARGIN);
            for channels in CHANNELS {
                let tile = Image::<T>::random(&mut rng, (tile_width, tile_height), channels);
                let original = Image::<T>::random(&mut rng, canvas_size, channels);
                for py in axis_positions(tile_height, canvas_size.1) {
                    for px in axis_positions(tile_width, canvas_size.0) {
                        for filter in FILTERS {
                            let mut composed = Image {
                                samples: original.samples.clone(),
                                ..original
                            };
                            let mut canvas = Canvas {
                                px: &mut composed.samples[..original.len()],
                                width: canvas_size.0,
                                height: canvas_size.1,
                                channels,
                            };
                            compose_tile(&mut canvas, &tile.tile(), px, py, filter);
                            let context = format!(
                                "{tile_width} x {tile_height} x {channels} at ({px}, {py}), {:?}",
                                filter as u8
                            );
                            assert_composed(
                                &composed,
                                &original,
                                &tile,
                                (px, py),
                                filter,
                                &context,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn compose_tile_u8_matches_reference() {
        assert_compose_tile_matches_reference::<u8>(41);
    }

    #[test]
    fn compose_tile_u16_matches_reference() {
        assert_compose_tile_matches_reference::<u16>(42);
    }

    #[test]
    fn compose_tile_f32_matches_reference() {
        assert_compose_tile_matches_reference::<f32>(43);
    }

    const PAD: u8 = 0x77;

    /// The RGBX canvas `original` after composing an RGB tile as the
    /// reference does: reference RGB samples and `PAD` where the compose
    /// writes, the original pixel elsewhere.
    fn rgbx_reference(
        original: &Image<u8>,
        tile: &Image<u8>,
        position: (f64, f64),
        filter: Interpolation,
    ) -> Vec<u8> {
        let mut expected = original.samples[..original.len()].to_vec();
        for (index, pixel) in expected.as_chunks_mut::<RGBX>().0.iter_mut().enumerate() {
            let (cx, cy) = (
                (index % original.width) as isize,
                (index / original.width) as isize,
            );
            if let Some(rgb) = reference_pixel(tile, position, filter, (cx, cy)) {
                pixel[..RGB].copy_from_slice(&rgb);
                pixel[RGB] = PAD;
            }
        }
        expected
    }

    /// Composes `tile` at `position` onto a copy of the RGBX canvas
    /// `original` with `compose_tile_rgbx`.
    fn compose_rgbx(
        original: &Image<u8>,
        tile: &Image<u8>,
        (px, py): (f64, f64),
        filter: Interpolation,
        scratch: &mut Vec<u8>,
    ) -> Vec<u8> {
        let mut composed = original.samples.clone();
        let mut canvas = Canvas {
            px: &mut composed[..original.len()],
            width: original.width,
            height: original.height,
            channels: RGBX,
        };
        compose_tile_rgbx(&mut canvas, &tile.tile(), px, py, filter, PAD, scratch);
        assert_eq!(
            composed[original.len()..],
            original.samples[original.len()..]
        );
        composed.truncate(original.len());
        composed
    }

    /// Composes `tile` at `position` onto an RGB canvas holding the RGB
    /// samples of the RGBX canvas `original`, with `compose_tile`, and
    /// expands the result with the X samples of `original`.
    fn compose_rgb_expanded(
        original: &Image<u8>,
        tile: &Image<u8>,
        (px, py): (f64, f64),
        filter: Interpolation,
    ) -> Vec<u8> {
        let mut rgb: Vec<u8> = original.samples[..original.len()]
            .as_chunks::<RGBX>()
            .0
            .iter()
            .flat_map(|pixel| pixel[..RGB].to_vec())
            .collect();
        let mut canvas = Canvas {
            px: &mut rgb,
            width: original.width,
            height: original.height,
            channels: RGB,
        };
        compose_tile(&mut canvas, &tile.tile(), px, py, filter);
        let (rgb_pixels, _) = rgb.as_chunks::<RGB>();
        let (rgbx_pixels, _) = original.samples.as_chunks::<RGBX>();
        rgb_pixels
            .iter()
            .zip(rgbx_pixels)
            .flat_map(|(rgb, rgbx)| [rgb[0], rgb[1], rgb[2], rgbx[3]])
            .collect()
    }

    #[test]
    fn compose_tile_rgbx_matches_reference() {
        let mut rng = Rng::new(44);
        let mut scratch = Vec::new();
        for (tile_width, tile_height) in TILE_SIZES {
            let canvas_size = (tile_width + MARGIN, tile_height + MARGIN);
            let tile = Image::<u8>::random(&mut rng, (tile_width, tile_height), RGB);
            let original = Image::<u8>::random(&mut rng, canvas_size, RGBX);
            for py in axis_positions(tile_height, canvas_size.1) {
                for px in axis_positions(tile_width, canvas_size.0) {
                    for filter in FILTERS {
                        let context = format!(
                            "{tile_width} x {tile_height} at ({px}, {py}), {:?}",
                            filter as u8
                        );
                        let composed =
                            compose_rgbx(&original, &tile, (px, py), filter, &mut scratch);
                        let expected = rgbx_reference(&original, &tile, (px, py), filter);
                        let tolerance = tolerance::<u8>((px, py), filter);
                        for (index, (&sample, &expected)) in
                            composed.iter().zip(&expected).enumerate()
                        {
                            let is_pad = index % RGBX == RGB;
                            assert!(
                                sample.distance(expected) <= if is_pad { 0.0 } else { tolerance },
                                "{context}: sample {index}: {sample} vs {expected}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn compose_tile_rgbx_equals_rgb_compose_expanded() {
        // The fused RGBX kernels share the RGB kernels' arithmetic, except
        // that the bilinear one blends rows under one 16-pixel block in f32,
        // where the RGB row (48 elements and more go fixed point) may round
        // differently.
        let mut rng = Rng::new(45);
        let mut scratch = Vec::new();
        for (tile_width, tile_height) in TILE_SIZES {
            let canvas_size = (tile_width + MARGIN, tile_height + MARGIN);
            let tile = Image::<u8>::random(&mut rng, (tile_width, tile_height), RGB);
            let mut original = Image::<u8>::random(&mut rng, canvas_size, RGBX);
            for pixel in original.samples.as_chunks_mut::<RGBX>().0 {
                pixel[RGB] = PAD;
            }
            for py in axis_positions(tile_height, canvas_size.1) {
                for px in axis_positions(tile_width, canvas_size.0) {
                    for filter in FILTERS {
                        let composed =
                            compose_rgbx(&original, &tile, (px, py), filter, &mut scratch);
                        let expected = compose_rgb_expanded(&original, &tile, (px, py), filter);
                        // Placement reads only the canvas size.
                        let canvas_size_only = Canvas {
                            px: &mut [0u8; 0][..],
                            width: canvas_size.0,
                            height: canvas_size.1,
                            channels: RGBX,
                        };
                        let short_bilinear = filter == Interpolation::Bilinear
                            && !is_copy(px, py, filter)
                            && Placement::new(px, py, &tile.tile(), &canvas_size_only)
                                .is_some_and(|placement| placement.covered.width() < 16);
                        let tolerance = if short_bilinear { 1.0 } else { 0.0 };
                        for (index, (&sample, &expected)) in
                            composed.iter().zip(&expected).enumerate()
                        {
                            assert!(
                                sample.distance(expected) <= tolerance,
                                "{tile_width} x {tile_height} at ({px}, {py}), {:?}: sample {index}: {sample} vs {expected}",
                                filter as u8
                            );
                        }
                    }
                }
            }
        }
    }

    /// The uint16 tile path (rows widened once) writes exactly what the
    /// row-by-row path writes, for tiles hanging off every edge and positions
    /// with and without a fraction in each axis.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    #[test]
    fn widened_uint16_bilinear_equals_the_row_path() {
        #[cfg(target_arch = "x86_64")]
        if !has_avx2_fma() {
            return;
        }
        let mut rng = Rng::new(47);
        for channels in CHANNELS {
            for (width, height) in TILE_SIZES {
                let tile_samples: Vec<u16> = random_samples(&mut rng, width * height * channels);
                let tile = Tile::new(&tile_samples, width, height, channels);
                let (canvas_width, canvas_height) = (width + 2 * MARGIN, height + 2 * MARGIN);
                for (px, py) in [
                    (0.5, 0.25),
                    (-1.75, 2.0),
                    (2.0, -0.5),
                    (1.3, 1.6),
                    (-0.4, -2.9),
                ] {
                    let compose = |widened: bool| {
                        let mut samples = vec![0u16; canvas_width * canvas_height * channels];
                        let mut canvas = Canvas {
                            px: &mut samples,
                            width: canvas_width,
                            height: canvas_height,
                            channels,
                        };
                        if let Some(placement) = Placement::new(px, py, &tile, &canvas) {
                            if widened {
                                // SAFETY: the tier's features are present
                                // (NEON is baseline; AVX2 and FMA checked).
                                unsafe {
                                    compose_bilinear_widened(&mut canvas, &tile, &placement);
                                }
                            } else {
                                compose_bilinear(&mut canvas, &tile, &placement);
                            }
                        }
                        samples
                    };
                    assert_eq!(
                        compose(true),
                        compose(false),
                        "{width} x {height} x {channels} at ({px}, {py})"
                    );
                }
            }
        }
    }

    #[test]
    fn compose_tile_onto_an_empty_canvas_writes_nothing() {
        let mut rng = Rng::new(46);
        let tile_samples: Vec<f32> = random_samples(&mut rng, 4 * 4);
        let tile = Tile::new(&tile_samples, 4, 4, 1);
        for (width, height) in [(0, 0), (0, 3), (3, 0)] {
            let mut canvas = Canvas {
                px: &mut [][..],
                width,
                height,
                channels: 1,
            };
            for filter in FILTERS {
                compose_tile(&mut canvas, &tile, 0.5, 0.25, filter);
                compose_tile(&mut canvas, &tile, 0.0, 0.0, filter);
            }
        }
    }

    /// `tiles` composed onto a copy of `original` with blended edges.
    fn compose_blended<T: TestSample>(
        original: &Image<T>,
        tiles: &[Image<T>],
        positions: &[(f64, f64)],
        filter: Interpolation,
    ) -> Image<T> {
        let mut composed = Image {
            samples: original.samples.clone(),
            ..*original
        };
        let mut canvas = Canvas {
            px: &mut composed.samples[..original.len()],
            width: original.width,
            height: original.height,
            channels: original.channels,
        };
        let tiles: Vec<Tile<T>> = tiles.iter().map(Image::tile).collect();
        compose_tiles(&mut canvas, &tiles, positions, filter);
        composed
    }

    /// The `width x height` part of `image` at (x, y).
    fn crop<T: TestSample>(
        image: &Image<T>,
        (x, y): (usize, usize),
        (width, height): (usize, usize),
    ) -> Image<T> {
        let samples = (y..y + height)
            .flat_map(|row| {
                let start = (row * image.width + x) * image.channels;
                image.samples[start..start + width * image.channels]
                    .iter()
                    .copied()
            })
            .collect();
        Image {
            samples,
            width,
            height,
            channels: image.channels,
        }
    }

    /// An image cut into a grid of tiles and composed at a fractional offset
    /// equals the image composed whole there: the pixels along the cuts,
    /// which no tile covers fully, are both tiles' resampling together.
    fn assert_split_equals_whole<T: TestSample>(seed: u64) {
        const WIDTH: usize = 23;
        const HEIGHT: usize = 17;
        const CUTS_X: [usize; 4] = [0, 7, 15, WIDTH];
        const CUTS_Y: [usize; 4] = [0, 1, 6, HEIGHT];
        let mut rng = Rng::new(seed);
        for channels in CHANNELS {
            let image = Image::<T>::random(&mut rng, (WIDTH, HEIGHT), channels);
            let original = Image::<T>::random(&mut rng, (WIDTH + 4, HEIGHT + 4), channels);
            for offset in [
                (0.5, 0.25),
                (1.3, 2.75),
                (2.0, 0.6),
                (-1.5, 1.0),
                (3.9, -0.1),
            ] {
                let mut tiles = Vec::new();
                let mut positions = Vec::new();
                for rows in CUTS_Y.windows(2) {
                    for columns in CUTS_X.windows(2) {
                        let size = (columns[1] - columns[0], rows[1] - rows[0]);
                        tiles.push(crop(&image, (columns[0], rows[0]), size));
                        positions.push((offset.0 + columns[0] as f64, offset.1 + rows[0] as f64));
                    }
                }
                for filter in [Interpolation::Bilinear, Interpolation::Bicubic] {
                    let whole = compose_blended(
                        &original,
                        &[crop(&image, (0, 0), (WIDTH, HEIGHT))],
                        &[offset],
                        filter,
                    );
                    let split = compose_blended(&original, &tiles, &positions, filter);
                    let tolerance = 2.0 * tolerance::<T>(offset, filter);
                    for (index, (&split, &whole)) in split.samples[..original.len()]
                        .iter()
                        .zip(&whole.samples)
                        .enumerate()
                    {
                        assert!(
                            split.distance(whole) <= tolerance,
                            "{channels} channels at {offset:?}, {:?}: sample {index}: {split:?} vs {whole:?}",
                            filter as u8
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn split_u8_equals_whole() {
        assert_split_equals_whole::<u8>(47);
    }

    #[test]
    fn split_u16_equals_whole() {
        assert_split_equals_whole::<u16>(48);
    }

    #[test]
    fn split_f32_equals_whole() {
        assert_split_equals_whole::<f32>(49);
    }

    /// `compose_blended` with the edge pass over every partly covered pixel,
    /// the ones the interiors then write included.
    fn compose_unpruned<T: TestSample>(
        original: &Image<T>,
        tiles: &[Image<T>],
        positions: &[(f64, f64)],
        filter: Interpolation,
    ) -> Image<T> {
        let mut composed = Image {
            samples: original.samples.clone(),
            ..*original
        };
        let mut canvas = Canvas {
            px: &mut composed.samples[..original.len()],
            width: original.width,
            height: original.height,
            channels: original.channels,
        };
        let tiles: Vec<Tile<T>> = tiles.iter().map(Image::tile).collect();
        edges::blend_edges(&mut canvas, (&tiles, positions), filter, &[], None);
        let inset = interior_inset(filter);
        for (tile, &(px, py)) in tiles.iter().zip(positions) {
            compose_tile_interior(&mut canvas, tile, px, py, filter, inset);
        }
        composed
    }

    /// Leaving the pixels the interiors write out of the edge pass changes no
    /// pixel: random grids of `rows` x `columns` tiles of random sizes,
    /// neighbours apart by whole or sub-pixel gaps and overlaps or butting,
    /// some at integer positions, on a canvas of `size`.
    fn assert_pruning_changes_nothing<T: TestSample>(
        seed: u64,
        (rows, columns): (usize, usize),
        size: (usize, usize),
    ) {
        const DELTAS: [f64; 8] = [-4.0, -2.0, -1.0, -0.6, -0.25, 0.0, 0.3, 1.5];
        const PHASES: [f64; 5] = [0.0, 0.25, 0.5, 0.7, 0.0];
        let mut rng = Rng::new(seed);
        let mut pick = |len: usize| (rng.next_u64() % len as u64) as usize;
        for channels in CHANNELS {
            for _ in 0..12 {
                let original = Image::<T>::random(&mut Rng::new(seed + 1), size, channels);
                let widths: Vec<usize> = (0..columns).map(|_| 4 + pick(9)).collect();
                let heights: Vec<usize> = (0..rows).map(|_| 4 + pick(9)).collect();
                let origin = (PHASES[pick(5)] + 1.0, PHASES[pick(5)] - 1.0);
                let (mut tiles, mut positions) = (Vec::new(), Vec::new());
                let mut y = origin.1;
                for &height in &heights {
                    let mut x = origin.0;
                    for &width in &widths {
                        let tile = Image::<T>::random(
                            &mut Rng::new(rng_seed(&mut pick)),
                            (width, height),
                            channels,
                        );
                        tiles.push(tile);
                        positions.push((x, y));
                        x += width as f64 + DELTAS[pick(DELTAS.len())];
                    }
                    y += height as f64 + DELTAS[pick(DELTAS.len())];
                }
                for filter in [Interpolation::Bilinear, Interpolation::Bicubic] {
                    let pruned = compose_blended(&original, &tiles, &positions, filter);
                    let unpruned = compose_unpruned(&original, &tiles, &positions, filter);
                    for (index, (&pruned, &unpruned)) in
                        pruned.samples.iter().zip(&unpruned.samples).enumerate()
                    {
                        assert!(
                            pruned.distance(unpruned) == 0.0
                                || (pruned.to_f32().is_nan() && unpruned.to_f32().is_nan()),
                            "{channels} channels, {positions:?}, {:?}: sample {index}: {pruned:?} vs {unpruned:?}",
                            filter as u8
                        );
                    }
                }
            }
        }
    }

    fn rng_seed(pick: &mut impl FnMut(usize) -> usize) -> u64 {
        pick(1 << 30) as u64
    }

    #[test]
    fn pruning_u8_changes_nothing() {
        assert_pruning_changes_nothing::<u8>(61, (2, 3), (40, 34));
    }

    #[test]
    fn pruning_f32_changes_nothing() {
        assert_pruning_changes_nothing::<f32>(62, (2, 3), (40, 34));
    }

    /// Many tiles, each in a few cells of the index of interiors.
    #[test]
    fn pruning_many_tiles_changes_nothing() {
        assert_pruning_changes_nothing::<u8>(63, (9, 11), (110, 96));
    }

    /// Flat values make the blend easy to state: a tile's edge pixels take
    /// the canvas for the part the tile leaves uncovered, and where two tiles
    /// overlap by less than a pixel, their weighted mean.
    #[test]
    fn edges_blend_by_coverage() {
        let flat = |value: f32, (width, height): (usize, usize)| Image {
            samples: vec![value; width * height],
            width,
            height,
            channels: 1,
        };
        let original = flat(0.0, (10, 3));
        let tile = flat(1.0, (4, 3));

        let composed = compose_blended(&original, &[tile], &[(0.5, 0.0)], Interpolation::Bilinear);
        assert_eq!(
            &composed.samples[..10],
            &[0.5, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        );

        let tiles = [flat(1.0, (4, 3)), flat(3.0, (4, 3))];
        let composed = compose_blended(
            &original,
            &tiles,
            &[(0.5, 0.0), (4.0, 0.0)],
            Interpolation::Bilinear,
        );
        // Column 4: half of the first tile's last pixel and the second
        // tile's first pixel, fully covered; the second tile's interior
        // wins over the first tile's edge.
        assert_eq!(
            &composed.samples[..10],
            &[0.5, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 3.0, 0.0, 0.0]
        );

        let composed = compose_blended(
            &original,
            &tiles,
            &[(0.5, 0.0), (4.25, 0.0)],
            Interpolation::Bilinear,
        );
        // Column 4: 0.5 of the first tile and 0.75 of the second.
        assert_eq!(composed.samples[4], (0.5 * 1.0 + 0.75 * 3.0) / 1.25);
    }

    /// A gap between tiles narrower than the taps is filled by them, their
    /// weighted mean; on the mosaic's outer edge, and across a gap wider than
    /// the taps, the canvas shows through.
    #[test]
    fn gaps_between_tiles_fill() {
        let flat = |value: f32, (width, height): (usize, usize)| Image {
            samples: vec![value; width * height],
            width,
            height,
            channels: 1,
        };
        let original = flat(0.0, (12, 7));
        let tiles = [flat(1.0, (4, 4)), flat(3.0, (4, 4))];
        let pixel = |image: &Image<f32>, (x, y): (usize, usize)| image.samples[y * image.width + x];
        let near = |actual: f32, expected: f32| {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        };

        // The first tile ends at 4.5 (its last sample at 3.5), the second
        // starts at 4.75: column 4 has 0.5 of the first and 0.25 of the
        // second.
        let composed = compose_blended(
            &original,
            &tiles,
            &[(0.5, 1.5), (4.75, 1.5)],
            Interpolation::Bilinear,
        );
        let (first, second) = (0.5, 0.25);
        // Rows 2 to 4: inside both tiles along y, between them along x.
        near(
            pixel(&composed, (4, 3)),
            (first * 1.0 + second * 3.0) / (first + second),
        );
        // Row 1, the mosaic's top edge: half covered along y, the canvas
        // keeps the rest.
        near(pixel(&composed, (4, 1)), 0.5 * (first * 1.0 + second * 3.0));
        near(pixel(&composed, (2, 1)), 0.5);

        // The second tile at 6.0: a gap of 1.5 pixels, column 5 untouched,
        // column 4 the first tile's edge over the canvas.
        let composed = compose_blended(
            &original,
            &tiles,
            &[(0.5, 1.5), (6.0, 1.5)],
            Interpolation::Bilinear,
        );
        near(pixel(&composed, (4, 3)), 0.5);
        near(pixel(&composed, (5, 3)), 0.0);

        // Four flat tiles 0.3 apart along both axes, bicubic: inside the
        // mosaic, gaps and their crossing included, every pixel is the
        // tiles' value; its outer edge is not.
        let tiles = [
            flat(2.0, (6, 6)),
            flat(2.0, (6, 6)),
            flat(2.0, (6, 6)),
            flat(2.0, (6, 6)),
        ];
        let original = flat(0.0, (16, 16));
        let composed = compose_blended(
            &original,
            &tiles,
            &[(1.6, 1.6), (7.9, 1.6), (1.6, 7.9), (7.9, 7.9)],
            Interpolation::Bicubic,
        );
        for y in 3..12 {
            for x in 3..12 {
                near(pixel(&composed, (x, y)), 2.0);
            }
        }
        assert!(pixel(&composed, (1, 5)) < 2.0);
    }
}

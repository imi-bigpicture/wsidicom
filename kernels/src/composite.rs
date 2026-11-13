//! Tile compositing: paste each tile onto the canvas at an integer position,
//! later tiles overwriting earlier ones.

/// A read-only `width x height` pixel grid of `channels` interleaved samples
/// per pixel. Row `y` starts at sample `y * stride`, so a crop of a larger
/// grid shares its rows.
pub(crate) struct Tile<'a, T> {
    pub(crate) samples: &'a [T],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) channels: usize,
    pub(crate) stride: usize,
}

impl<'a, T: Copy> Tile<'a, T> {
    /// The grid in `samples`, rows back to back.
    pub(crate) fn new(samples: &'a [T], width: usize, height: usize, channels: usize) -> Self {
        Tile {
            samples,
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
            samples: &self.samples[start..start + len],
            width,
            height,
            ..*self
        }
    }

    fn row(&self, y: usize) -> &'a [T] {
        let start = y * self.stride;
        &self.samples[start..start + self.width * self.channels]
    }
}

/// The mutable destination grid, rows back to back.
pub(crate) struct Canvas<'a, T> {
    pub(crate) samples: &'a mut [T],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) channels: usize,
}

impl<T> Canvas<'_, T> {
    /// The samples of the `pixel_count` pixels from (x, y).
    fn run(&mut self, x: isize, y: isize, pixel_count: usize) -> &mut [T] {
        let start = (y as usize * self.width + x as usize) * self.channels;
        &mut self.samples[start..start + pixel_count * self.channels]
    }
}

/// Paste `tiles` onto `canvas`, each with its top-left at its position in
/// `positions`, in order: the last write wins.
pub(crate) fn compose_tiles<T: Copy>(
    canvas: &mut Canvas<T>,
    tiles: &[Tile<T>],
    positions: &[(isize, isize)],
) {
    for (tile, &position) in tiles.iter().zip(positions) {
        paste(canvas, tile, position);
    }
}

/// Paste the part of `tile` on the canvas, its top-left at canvas position
/// (left, top).
fn paste<T: Copy>(canvas: &mut Canvas<T>, tile: &Tile<T>, (left, top): (isize, isize)) {
    let visible_left = left.max(0);
    let visible_right = left
        .saturating_add_unsigned(tile.width)
        .min(canvas.width as isize);
    if visible_right <= visible_left {
        return;
    }
    let hidden_left = (visible_left - left) as usize;
    let visible_width = (visible_right - visible_left) as usize;
    let visible_bottom = top
        .saturating_add_unsigned(tile.height)
        .min(canvas.height as isize);
    let channels = tile.channels;
    for canvas_y in top.max(0)..visible_bottom {
        let row = tile.row((canvas_y - top) as usize);
        canvas
            .run(visible_left, canvas_y, visible_width)
            .copy_from_slice(
                &row[hidden_left * channels..(hidden_left + visible_width) * channels],
            );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{CHANNELS, Rng};

    /// Each canvas sample as a reference paste leaves it: the last tile
    /// covering the pixel, else the canvas's own value.
    fn reference(
        canvas: &[u16],
        (width, height, channels): (usize, usize, usize),
        tiles: &[Tile<u16>],
        positions: &[(isize, isize)],
    ) -> Vec<u16> {
        let mut out = canvas.to_vec();
        for (tile, &(left, top)) in tiles.iter().zip(positions) {
            for tile_y in 0..tile.height {
                for tile_x in 0..tile.width {
                    let (x, y) = (left + tile_x as isize, top + tile_y as isize);
                    if x < 0 || y < 0 || x >= width as isize || y >= height as isize {
                        continue;
                    }
                    let at = (y as usize * width + x as usize) * channels;
                    let from = tile_y * tile.stride + tile_x * channels;
                    out[at..at + channels].copy_from_slice(&tile.samples[from..from + channels]);
                }
            }
        }
        out
    }

    #[test]
    fn pastes_match_the_reference() {
        let mut rng = Rng::new(7);
        for &channels in &CHANNELS {
            let (width, height) = (13, 9);
            let sizes = [(1, 1), (4, 3), (13, 9), (20, 2), (3, 15)];
            let images: Vec<Vec<u16>> = sizes
                .iter()
                .map(|&(tile_width, tile_height)| {
                    (0..tile_width * tile_height * channels)
                        .map(|_| rng.next_u64() as u16)
                        .collect()
                })
                .collect();
            let tiles: Vec<Tile<u16>> = images
                .iter()
                .zip(sizes)
                .map(|(samples, (tile_width, tile_height))| {
                    Tile::new(samples, tile_width, tile_height, channels)
                })
                .collect();
            for positions in [
                [(0, 0), (2, 3), (0, 0), (-5, 4), (11, -10)],
                [(12, 8), (-1, -1), (-13, 0), (9, 7), (0, 8)],
                [(13, 0), (0, 9), (-4, -3), (-20, 1), (5, -15)],
            ] {
                let start: Vec<u16> = (0..width * height * channels)
                    .map(|sample| sample as u16)
                    .collect();
                let expected = reference(&start, (width, height, channels), &tiles, &positions);
                let mut samples = start.clone();
                let mut canvas = Canvas {
                    samples: &mut samples,
                    width,
                    height,
                    channels,
                };
                compose_tiles(&mut canvas, &tiles, &positions);
                assert_eq!(
                    samples, expected,
                    "channels {channels}, positions {positions:?}"
                );
            }
        }
    }

    #[test]
    fn crops_paste_their_pixels() {
        let samples: Vec<u16> = (0..6 * 5).collect();
        let tile = Tile::new(&samples, 6, 5, 1);
        let piece = tile.crop((2, 1), (3, 2));
        let mut out = vec![0u16; 4 * 3];
        let mut canvas = Canvas {
            samples: &mut out,
            width: 4,
            height: 3,
            channels: 1,
        };
        compose_tiles(&mut canvas, &[piece], &[(1, 1)]);
        assert_eq!(out, [0, 0, 0, 0, 0, 8, 9, 10, 0, 14, 15, 16]);
    }
}

//! Python surface of `wsidicom_kernels` (`wsidicom_kernels._kernels`): argument validation,
//! array borrows, GIL release and numpy conversion around the pure-Rust
//! kernel modules.

use numpy::ndarray::{ArrayD, ArrayView2, Dimension, IxDyn};
use numpy::{
    Element, IntoPyArray, PyArray, PyArray2, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyReadonlyArray, PyReadonlyArray2, PyReadonlyArrayDyn, PyReadwriteArray, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

mod box_reduce;
mod composite;
mod convolve;
mod convolve_float;
mod cpu;
mod exact;
mod grid;
mod nearest;
mod rgbx;
mod sample;
mod simd;
#[cfg(test)]
mod testing;
#[cfg(target_arch = "x86_64")]
mod x86;

use box_reduce::BoxSample;
use composite::geometry::{Canvas, Tile};
use composite::rows::ComposeSample;
use composite::{Interpolation, compose_tiles, compose_tiles_rgbx};
use convolve::Geometry;
use sample::{RGB, RGBX};

/// Freed outputs go back to mimalloc's caches, so the next call's output
/// (a tile stream's every call) reuses committed pages instead of faulting
/// in fresh ones.
#[cfg(target_arch = "x86_64")]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// How long freed memory stays committed for reuse before mimalloc returns
/// it to the OS: long enough that a stream of slow tiles (a lanczos3 2048
/// tile takes milliseconds, with several outputs queued) still finds its
/// pages committed (mimalloc's 10 ms default re-faulted them), short enough
/// that a one-off large output doesn't stay resident.
#[cfg(target_arch = "x86_64")]
const PURGE_DELAY_MS: std::ffi::c_long = 1000;

/// `mi_option_purge_delay` of mimalloc's `mi_option_t` (v2 and v3).
#[cfg(target_arch = "x86_64")]
const MI_OPTION_PURGE_DELAY: std::ffi::c_int = 15;

#[cfg(target_arch = "x86_64")]
unsafe extern "C" {
    fn mi_option_set(option: std::ffi::c_int, value: std::ffi::c_long);
}

// ---- Errors and conversions ----------------------------------------------------

fn value_error(msg: String) -> PyErr {
    PyValueError::new_err(msg)
}

/// Error for an input array that a concurrent compose holds mutably borrowed
/// (numpy-crate borrow tracking is process-wide).
fn in_use() -> PyErr {
    PyValueError::new_err("array is being written by a concurrent compose — pass a copy")
}

fn not_c_contiguous(what: &str) -> PyErr {
    PyValueError::new_err(format!("{what} must be C-contiguous"))
}

/// Error for an array whose data does not start on a multiple of its sample
/// size (a view at an odd byte offset): it can't be read as a slice.
fn misaligned(what: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{what} must be aligned to the sample size — pass a copy"
    ))
}

/// Why an array can't be read as a C-ordered slice, if it can't.
fn check_c_slice<T: Element, D: Dimension>(
    array: &Bound<'_, PyArray<T, D>>,
    what: &str,
) -> PyResult<()> {
    if !array.is_c_contiguous() {
        return Err(not_c_contiguous(what));
    }
    if !array.is_aligned() {
        return Err(misaligned(what));
    }
    Ok(())
}

/// The samples of `array` in (H, W, C) order, so it must be C-contiguous:
/// `as_slice` alone also takes a Fortran-ordered array.
fn c_samples<'a, T: Element, D: Dimension>(
    array: &'a PyReadonlyArray<'_, T, D>,
    what: &str,
) -> PyResult<&'a [T]> {
    check_c_slice(array, what)?;
    array.as_slice().map_err(|_| not_c_contiguous(what))
}

/// Outputs from this many bytes are allocated only after the allocator
/// confirms it can provide them: a failed allocation aborts the process.
const CHECKED_ALLOCATION_BYTES: usize = 1 << 30;

/// The samples of an output of `height` x `width` pixels of `channels`
/// samples of `sample_bytes` each: a size whose samples or bytes overflow is a `ValueError`,
/// one the allocator can't provide a `MemoryError`, rather than a wrapped
/// allocation or an aborted process.
pub(crate) fn output_samples(
    (height, width, channels): (usize, usize, usize),
    sample_bytes: usize,
) -> PyResult<usize> {
    let too_large = || {
        PyValueError::new_err(format!(
            "an output of {height} x {width} x {channels} samples is too large"
        ))
    };
    let samples = height
        .checked_mul(width)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(too_large)?;
    let bytes = samples
        .checked_mul(sample_bytes)
        .filter(|&bytes| isize::try_from(bytes).is_ok())
        .ok_or_else(too_large)?;
    if bytes >= CHECKED_ALLOCATION_BYTES {
        Vec::<u8>::new().try_reserve_exact(bytes).map_err(|_| {
            PyMemoryError::new_err(format!("cannot allocate an output of {bytes} bytes"))
        })?;
    }
    Ok(samples)
}

/// Positions' largest magnitude: f64 holds every integer up to it, and
/// pixel arithmetic on it cannot overflow.
const MAX_POSITION: f64 = 4_503_599_627_370_496.0; // 2^52

/// `c_samples` of an array borrowed for writing.
fn c_samples_mut<'a, T: Element, D: Dimension>(
    array: &'a mut PyReadwriteArray<'_, T, D>,
    what: &str,
) -> PyResult<&'a mut [T]> {
    check_c_slice(array, what)?;
    array.as_slice_mut().map_err(|_| not_c_contiguous(what))
}

/// `value` as a size or factor, a `ValueError` with `message` below 1.
fn at_least_one(value: i64, message: &str) -> PyResult<usize> {
    usize::try_from(value)
        .ok()
        .filter(|&value| value >= 1)
        .ok_or_else(|| PyValueError::new_err(message.to_owned()))
}

fn parse_interpolation(name: &str) -> PyResult<Interpolation> {
    match name {
        "nearest" => Ok(Interpolation::Nearest),
        "bilinear" => Ok(Interpolation::Bilinear),
        "bicubic" => Ok(Interpolation::Bicubic),
        other => Err(PyValueError::new_err(format!("unknown filter: {other}"))),
    }
}

/// Channel counts an interleaved (H, W, C) array may have: gray, RGB, and
/// 4-sample pixels (RGBA, RGBX).
const CHANNELS: [usize; 3] = [1, RGB, RGBX];

/// Extract (height, width, channels) from a 2D (channels=1) or 3D (HWC) shape.
fn hwc_dims(shape: &[usize]) -> PyResult<(usize, usize, usize)> {
    match *shape {
        [height, width] => Ok((height, width, 1)),
        [height, width, channels] if CHANNELS.contains(&channels) => Ok((height, width, channels)),
        [_, _, _] => Err(PyValueError::new_err(
            "3D arrays must have 1, 3 or 4 channels",
        )),
        _ => Err(PyValueError::new_err(
            "arrays must be 2D (planar) or 3D (H, W, C)",
        )),
    }
}

/// (H, W, C) of a Python array-like to resample, C = 1 for 2D; at least one
/// pixel and one channel.
fn image_dims(image: &Bound<'_, PyAny>) -> PyResult<(usize, usize, usize)> {
    let shape: Vec<usize> = image
        .getattr("shape")?
        .extract()
        .map_err(|_| PyValueError::new_err("image must be a 2D or 3D array"))?;
    let dims @ (height, width, channels) = hwc_dims(&shape)?;
    if height == 0 || width == 0 || channels == 0 {
        return Err(PyValueError::new_err(
            "image must have at least one pixel and one channel",
        ));
    }
    Ok(dims)
}

/// Borrow an input image read-only; fallible, as a concurrent compose may hold it.
fn read_only<'py, T: Element>(
    array: &Bound<'py, PyArrayDyn<T>>,
) -> PyResult<PyReadonlyArrayDyn<'py, T>> {
    array.try_readonly().map_err(|_| in_use())
}

/// Wrap `data` as a numpy array: (height, width) when channels == 1, else
/// (height, width, channels).
fn to_numpy<T: Element>(
    py: Python<'_>,
    data: Vec<T>,
    height: usize,
    width: usize,
    channels: usize,
) -> PyResult<Py<PyAny>> {
    let shape = if channels == 1 {
        vec![height, width]
    } else {
        vec![height, width, channels]
    };
    let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(array.into_pyarray(py).into_any().unbind())
}

/// `out` with `like`'s channel axis: the kernels return one channel as
/// (H, W), and an (H, W, 1) input gives an (H, W, 1) output (a view).
fn shaped_like(py: Python<'_>, like: &Bound<'_, PyAny>, out: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let ndim = |array: &Bound<'_, PyAny>| {
        array
            .cast::<PyUntypedArray>()
            .map_or(0, PyUntypedArrayMethods::ndim)
    };
    let image = out.bind(py);
    if ndim(like) != 3 || ndim(image) != 2 {
        return Ok(out);
    }
    let (height, width): (usize, usize) = image.getattr("shape")?.extract()?;
    Ok(image
        .call_method1("reshape", ((height, width, 1),))?
        .unbind())
}

/// Evaluate `$body` with `$binding` bound to `$array` downcast to `PyArrayDyn<$t>`,
/// `$t` being whichever supported dtype it has.
macro_rules! with_dtype {
    ($array:expr_2021, $what:literal, |$binding:ident: $t:ident| $body:expr_2021) => {
        if let Ok($binding) = $array.cast::<PyArrayDyn<u8>>() {
            type $t = u8;
            $body
        } else if let Ok($binding) = $array.cast::<PyArrayDyn<u16>>() {
            type $t = u16;
            $body
        } else if let Ok($binding) = $array.cast::<PyArrayDyn<f32>>() {
            type $t = f32;
            $body
        } else {
            Err(PyValueError::new_err(concat!(
                $what,
                " must be a 2D/3D uint8, uint16, or float32 array in native byte order"
            )))
        }
    };
}

// ---- compose -------------------------------------------------------------------

/// A borrowed tile's samples, from its first to its last, and how many
/// samples apart its rows start. Each row must be contiguous: a C-contiguous
/// array, or rows of one cut out by slicing (`tile[y0:y1, x0:x1]`).
fn tile_samples<'a, T: Element>(
    array: &'a PyReadonlyArrayDyn<'_, T>,
    (height, width, channels): (usize, usize, usize),
) -> PyResult<(&'a [T], usize)> {
    let row_len = width * channels;
    if array.is_c_contiguous() {
        return Ok((c_samples(array, "tiles")?, row_len));
    }
    if !array.is_aligned() {
        return Err(misaligned("tiles"));
    }
    let sample = std::mem::size_of::<T>() as isize;
    let strides = array.strides();
    let pixels_contiguous = channels == 1 || strides.get(2).copied() == Some(sample);
    let row_contiguous = width <= 1 || strides[1] == channels as isize * sample;
    let row_stride = if height <= 1 {
        Some(row_len)
    } else {
        (strides[0] % sample == 0)
            .then(|| usize::try_from(strides[0] / sample).ok())
            .flatten()
            .filter(|&stride| stride >= row_len)
    };
    let (true, true, Some(row_stride)) = (pixels_contiguous, row_contiguous, row_stride) else {
        return Err(PyValueError::new_err(
            "each tile row must be contiguous: a C-contiguous array, or a slice of rows \
             and columns of one",
        ));
    };
    let len = if height == 0 || width == 0 {
        0
    } else {
        (height - 1) * row_stride + row_len
    };
    // SAFETY: sample (y, x, c) of the array lies `y * row_stride + x *
    // channels + c` samples past its data pointer, and its buffer holds
    // every sample from its first to its last; the read-only borrow keeps
    // them alive and unwritten by other borrowers for 'a.
    let samples = unsafe { std::slice::from_raw_parts(array.data().cast_const(), len) };
    Ok((samples, row_stride))
}

/// The pieces `crops` cuts from `tiles`, one per row: (tile index, x, y,
/// width, height), each nonempty and inside its tile.
fn crop_tiles<'a, T: Copy>(
    tiles: &[Tile<'a, T>],
    crops: &ArrayView2<i64>,
) -> PyResult<Vec<Tile<'a, T>>> {
    crops
        .rows()
        .into_iter()
        .map(|crop| {
            let field = |at: usize| usize::try_from(crop[at]).ok();
            let piece = (field(0), field(1), field(2), field(3), field(4));
            let (Some(index), Some(x), Some(y), Some(width), Some(height)) = piece else {
                return Err(PyValueError::new_err("crops must not be negative"));
            };
            let tile = tiles.get(index).ok_or_else(|| {
                PyValueError::new_err("crops must name a tile by its index in tiles")
            })?;
            let inside = width > 0
                && height > 0
                && x.checked_add(width).is_some_and(|end| end <= tile.width)
                && y.checked_add(height).is_some_and(|end| end <= tile.height);
            if !inside {
                return Err(PyValueError::new_err(
                    "crops must be nonempty and inside their tile",
                ));
            }
            Ok(tile.crop((x, y), (width, height)))
        })
        .collect()
}

/// Validate and borrow all tiles, then compose them with the GIL released (the
/// kernels are pure Rust on raw slices). Borrows are fallible so aliasing — a
/// tile sharing the canvas's buffer, or another thread composing onto it — is
/// a `ValueError` rather than a panic.
fn compose_all<T: ComposeSample>(
    canvas_array: &Bound<'_, PyArrayDyn<T>>,
    tile_arrays: &[Bound<'_, PyAny>],
    (pos, crops): (&ArrayView2<f64>, Option<&ArrayView2<i64>>),
    tile_c: Option<usize>,
    compose: impl FnOnce(&mut Canvas<T>, &[Tile<T>], &[(f64, f64)]) + Send,
) -> PyResult<()> {
    let (height, width, channels) = hwc_dims(canvas_array.shape())?;
    let tile_c = tile_c.unwrap_or(channels);
    let Ok(mut rw) = canvas_array.try_readwrite() else {
        let writable = canvas_array
            .getattr("flags")?
            .getattr("writeable")?
            .extract::<bool>()?;
        return Err(PyValueError::new_err(if writable {
            "canvas is already borrowed (in use by another compose?)"
        } else {
            "canvas must be writable"
        }));
    };
    let mut canvas = Canvas {
        px: c_samples_mut(&mut rw, "canvas")?,
        width,
        height,
        channels,
    };

    let mut guards: Vec<(PyReadonlyArrayDyn<T>, (usize, usize, usize))> =
        Vec::with_capacity(tile_arrays.len());
    for tile_object in tile_arrays {
        let tile_array = tile_object
            .cast::<PyArrayDyn<T>>()
            .map_err(|_| PyValueError::new_err("every tile must match the canvas dtype"))?;
        let (th, tw, tc) = hwc_dims(tile_array.shape())?;
        if tc != tile_c {
            return Err(PyValueError::new_err(if tile_c == channels {
                "every tile must have the same channels as the canvas"
            } else {
                "an RGBX (H, W, 4) canvas takes RGB (H, W, 3) tiles; \
                 for a 4-sample tile, use a 4-sample canvas"
            }));
        }
        let ro = tile_array.try_readonly().map_err(|_| {
            PyValueError::new_err("a tile shares memory with the canvas — pass a copy")
        })?;
        guards.push((ro, (th, tw, tc)));
    }
    let mut tiles = guards
        .iter()
        .map(|(guard, dims @ (tile_height, tile_width, _))| {
            let (px, stride) = tile_samples(guard, *dims)?;
            Ok(Tile {
                stride,
                ..Tile::new(px, *tile_width, *tile_height, tile_c)
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    if let Some(crops) = crops {
        tiles = crop_tiles(&tiles, crops)?;
    }

    let positions: Vec<(f64, f64)> = pos.rows().into_iter().map(|row| (row[0], row[1])).collect();
    canvas_array
        .py()
        .detach(|| compose(&mut canvas, &tiles, &positions));
    Ok(())
}

/// compose(canvas, tiles, positions, filter="bilinear", pad=255, crops=None)
///
/// Paste `tiles` onto `canvas` (mutated in place) at `positions` (N x 2 float64,
/// each (x, y) = (column, row)); with `crops` (N x 5 int64, each (tile index,
/// x, y, width, height)), paste those pieces of `tiles` instead. Overlaps: last wins; the pixels tiles at
/// fractional positions cover only partly are blended by coverage (see
/// `composite::edges`). Canvas and tiles are all
/// either planar 2D (H, W) or interleaved HWC (H, W, C) with matching channels,
/// except that a uint8 RGBX (H, W, 4) canvas takes RGB (H, W, 3) tiles, `pad`
/// filling the fourth sample (see `compose_tile_rgbx`).
#[pyfunction]
#[pyo3(signature = (canvas, tiles, positions, filter="bilinear", pad=255, crops=None))]
fn compose(
    canvas: &Bound<'_, PyAny>,
    tiles: Vec<Bound<'_, PyAny>>,
    positions: PyReadonlyArray2<f64>,
    filter: &str,
    pad: u8,
    crops: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let filter = parse_interpolation(filter)?;
    if !positions.is_c_contiguous() {
        return Err(PyValueError::new_err("positions must be C-contiguous"));
    }
    let pos = positions.as_array();
    if pos.shape()[1] != 2 {
        return Err(PyValueError::new_err("positions must have shape (N, 2)"));
    }
    let crops = crops
        .map(|crops| {
            crops
                .cast::<PyArray2<i64>>()
                .ok()
                .filter(|crops| crops.shape()[1] == 5)
                .ok_or_else(|| PyValueError::new_err("crops must be an (N, 5) int64 array"))?
                .try_readonly()
                .map_err(|_| in_use())
        })
        .transpose()?;
    let crops = crops.as_ref().map(PyReadonlyArray2::as_array);
    let (pieces, what) = match &crops {
        Some(crops) => (crops.shape()[0], "crops"),
        None => (tiles.len(), "tiles"),
    };
    if pos.shape()[0] != pieces {
        return Err(value_error(format!(
            "number of positions must equal number of {what}"
        )));
    }
    if !pos
        .iter()
        .all(|position| position.is_finite() && position.abs() <= MAX_POSITION)
    {
        return Err(PyValueError::new_err(
            "positions must be finite and at most 2^52 in magnitude",
        ));
    }

    if let Ok(rgbx) = canvas.cast::<PyArrayDyn<u8>>() {
        let first_tile_channels = tiles
            .first()
            .and_then(|tile| tile.cast::<PyArrayDyn<u8>>().ok())
            .and_then(|tile| hwc_dims(tile.shape()).ok())
            .map(|(_, _, channels)| channels);
        if hwc_dims(rgbx.shape())?.2 == RGBX && first_tile_channels == Some(RGB) {
            let placements = (&pos, crops.as_ref());
            return compose_all(
                rgbx,
                &tiles,
                placements,
                Some(RGB),
                |canvas, tiles, positions| {
                    compose_tiles_rgbx(canvas, tiles, positions, filter, pad);
                },
            );
        }
    }
    with_dtype!(canvas, "canvas", |array: T| compose_all::<T>(
        array,
        &tiles,
        (&pos, crops.as_ref()),
        None,
        |canvas, tiles, positions| compose_tiles(canvas, tiles, positions, filter)
    ))
}

// ---- downscale / resize / rgb_to_rgbx ------------------------------------------

fn box_reduce<T: BoxSample>(
    py: Python<'_>,
    array: &Bound<'_, PyArrayDyn<T>>,
    dims @ (_, _, channels): (usize, usize, usize),
    times: u32,
) -> PyResult<Py<PyAny>> {
    let ro = read_only(array)?;
    let src = c_samples(&ro, "image")?;
    let (out, width, height) = py.detach(|| box_reduce::pyramid::box_downscale(src, dims, times));
    to_numpy(py, out, height, width, channels)
}

/// _`stitch_grid(tiles`, `grid_width`)
///
/// The grid's tiles as one image; see `wsidicom_kernels.downscale_grid`.
#[pyfunction]
fn stitch_grid(
    py: Python<'_>,
    tiles: Vec<Bound<'_, PyAny>>,
    grid_width: usize,
) -> PyResult<Py<PyAny>> {
    let first = tiles
        .first()
        .ok_or_else(|| PyValueError::new_err("Cannot stitch an empty tile grid"))?;
    let out = with_dtype!(first, "tile", |array: T| {
        let _ = array;
        grid::stitch_grid::<T>(py, &tiles, grid_width)
    })?;
    shaped_like(py, first, out)
}

/// _`downscale_grid_box(tiles`, `grid_width`, factor, size=None, fill=None)
///
/// The power-of-2 box reduction of a grid of tiles as one image, in a
/// `size` frame with `fill` around it if given; see `wsidicom_kernels.downscale_grid`,
/// which checks when this path is exact.
#[pyfunction]
#[pyo3(signature = (tiles, grid_width, factor, size=None, fill=None))]
fn downscale_grid_box(
    py: Python<'_>,
    tiles: Vec<Bound<'_, PyAny>>,
    grid_width: usize,
    factor: usize,
    size: Option<(usize, usize)>,
    fill: Option<Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let times = box_reduce::pyramid::box_halvings("box", factor)
        .ok_or_else(|| PyValueError::new_err("factor must be a power of 2"))?;
    let first = tiles
        .first()
        .ok_or_else(|| PyValueError::new_err("Cannot downscale an empty tile grid"))?;
    let out = with_dtype!(first, "tile", |array: T| {
        let _ = array;
        grid::box_grid::<T>(py, &tiles, grid_width, times, (size, fill.as_ref()))
    })?;
    shaped_like(py, first, out)
}

/// framed(image, size, fill)
///
/// `image` at the top left of a `size` frame, `fill` around it; see
/// `wsidicom_kernels.downscale_grid`.
#[pyfunction]
fn framed(
    py: Python<'_>,
    image: Bound<'_, PyAny>,
    size: (usize, usize),
    fill: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let out = with_dtype!(image, "image", |array: T| {
        let _ = array;
        grid::framed::<T>(py, &image, (Some(size), Some(&fill)))
    })?;
    shaped_like(py, &image, out)
}

/// Resize: nearest-neighbour (`nearest`) or by convolution (`convolve` for
/// uint8, `convolve_float` for uint16 and float32).
fn resize_image(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    geometry: Geometry,
    channels: usize,
    filter: &str,
) -> PyResult<Py<PyAny>> {
    if filter == "nearest" {
        return with_dtype!(image, "image", |array: T| resize_nearest_array::<T>(
            py, array, geometry, channels
        ));
    }
    if convolve::Kernel::parse(filter).is_none() {
        return Err(unknown_filter(filter));
    }
    if let Some(result) = resize_native_u8(py, image, geometry, channels, filter, None) {
        return result;
    }
    if let Some(result) = resize_native_float(py, image, geometry, channels, filter) {
        return result;
    }
    Err(PyValueError::new_err(
        "image must be a 2D/3D uint8, uint16, or float32 array in native byte order",
    ))
}

fn unknown_filter(filter: &str) -> PyErr {
    value_error(format!("unknown filter: {filter}"))
}

/// The native uint16/float32 convolution resize, or None
/// where it doesn't apply.
fn resize_native_float(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    geometry: Geometry,
    channels: usize,
    filter: &str,
) -> Option<PyResult<Py<PyAny>>> {
    fn run<T: convolve_float::FloatResample + Element>(
        py: Python<'_>,
        array: &Bound<'_, PyArrayDyn<T>>,
        geometry: Geometry,
        channels: usize,
        kernel: convolve::Kernel,
    ) -> PyResult<Py<PyAny>> {
        let ro = read_only(array)?;
        let src = c_samples(&ro, "image")?;
        let out = py.detach(|| {
            let plan = convolve_float::FloatPlan::cached(geometry, kernel, channels);
            convolve_float::resize_float(src, geometry.src, &plan)
        });
        let (ow, oh) = geometry.dst;
        to_numpy(py, out, oh, ow, channels)
    }
    let kernel = convolve::Kernel::parse(filter)?;
    if let Ok(array) = image.cast::<PyArrayDyn<f32>>() {
        return Some(run(py, array, geometry, channels, kernel));
    }
    if let Ok(array) = image.cast::<PyArrayDyn<u16>>() {
        let ((width, height), (ow, oh)) = (geometry.src, geometry.dst);
        if kernel == convolve::Kernel::Bilinear && width == 2 * ow && height == 2 * oh {
            return Some(exact_half_u16(py, array, (height, width, channels)));
        }
        return Some(run(py, array, geometry, channels, kernel));
    }
    None
}

/// The exact x2 bilinear reduction of a uint16 image.
fn exact_half_u16(
    py: Python<'_>,
    array: &Bound<'_, PyArrayDyn<u16>>,
    (height, width, channels): (usize, usize, usize),
) -> PyResult<Py<PyAny>> {
    let ro = read_only(array)?;
    let src = c_samples(&ro, "image")?;
    let out = py.detach(|| exact::bilinear_x2_u16::bilinear_x2_u16(src, width, height, channels));
    to_numpy(py, out, height / 2, width / 2, channels)
}

/// The native nearest-neighbour resize, any dtype and channel count.
fn resize_nearest_array<T: Element + bytemuck::Pod>(
    py: Python<'_>,
    array: &Bound<'_, PyArrayDyn<T>>,
    geometry: Geometry,
    channels: usize,
) -> PyResult<Py<PyAny>> {
    let ro = read_only(array)?;
    let src = c_samples(&ro, "image")?;
    let out = py.detach(|| nearest::resize_nearest_over(src, geometry, channels));
    let (ow, oh) = geometry.dst;
    to_numpy(py, out, oh, ow, channels)
}

/// The native u8 resize (RGB to RGBX with `rgbx_pad`), or
/// None for other dtypes and filters.
fn resize_native_u8(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    geometry: Geometry,
    channels: usize,
    filter: &str,
    rgbx_pad: Option<u8>,
) -> Option<PyResult<Py<PyAny>>> {
    let kernel = convolve::Kernel::parse(filter)?;
    let array = image.cast::<PyArrayDyn<u8>>().ok()?;
    if geometry.reduces_past(FIXED_POINT_MAX_REDUCTION) {
        return Some(resize_u8_float(
            py, array, geometry, channels, kernel, rgbx_pad,
        ));
    }
    let ((width, height), (ow, oh)) = (geometry.src, geometry.dst);
    if let Some(exact_kernel) = geometry
        .fits_image()
        .then(|| exact::exact_kernel(kernel, (width, height), (ow, oh)))
        .flatten()
    {
        let exact = || {
            let ro = read_only(array)?;
            let src = c_samples(&ro, "image")?;
            let out = py.detach(|| {
                let resized = exact_kernel(src, width, height, channels);
                match rgbx_pad {
                    Some(pad) => rgbx::expand_image(&resized, oh, ow, pad),
                    None => resized,
                }
            });
            to_numpy(
                py,
                out,
                oh,
                ow,
                if rgbx_pad.is_some() { RGBX } else { channels },
            )
        };
        return Some(exact());
    }
    let plan = py.detach(|| convolve::Plan::cached(geometry, kernel, channels));
    let resized = || {
        let ro = read_only(array)?;
        let src = c_samples(&ro, "image")?;
        let out = py.detach(|| convolve::resize_u8(src, (width, height), &plan, rgbx_pad));
        to_numpy(
            py,
            out,
            oh,
            ow,
            if rgbx_pad.is_some() { RGBX } else { channels },
        )
    };
    Some(resized())
}

/// The reduction (per axis) past which the uint8 fixed-point convolution
/// loses accuracy: each of a window's taps is rounded, and over thousands of
/// taps the roundings add up. Past it uint8 takes the float32 engine.
const FIXED_POINT_MAX_REDUCTION: usize = 128;

/// A uint8 resize by the float32 engine (RGB to RGBX with `rgbx_pad`).
fn resize_u8_float(
    py: Python<'_>,
    array: &Bound<'_, PyArrayDyn<u8>>,
    geometry: Geometry,
    channels: usize,
    kernel: convolve::Kernel,
    rgbx_pad: Option<u8>,
) -> PyResult<Py<PyAny>> {
    let ro = read_only(array)?;
    let src = c_samples(&ro, "image")?;
    let (ow, oh) = geometry.dst;
    let out = py.detach(|| {
        let plan = convolve_float::FloatPlan::cached(geometry, kernel, channels);
        let resized = convolve_float::resize_float(src, geometry.src, &plan);
        match rgbx_pad {
            Some(pad) => rgbx::expand_image(&resized, oh, ow, pad),
            None => resized,
        }
    });
    to_numpy(
        py,
        out,
        oh,
        ow,
        if rgbx_pad.is_some() { RGBX } else { channels },
    )
}

/// Downscale/resize an RGB image to RGBX: the fused box pyramid for a
/// power-of-2 box reduction (`box_times` halvings), else the convolution
/// resize writing RGBX, else (nearest) the nearest resize expanded.
fn resize_to_rgbx(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    geometry: Geometry,
    channels: usize,
    filter: &str,
    box_times: Option<u32>,
    pad: u8,
) -> PyResult<Py<PyAny>> {
    enum Method {
        BoxPyramid(u32),
        Nearest,
    }
    if channels != RGB {
        return Err(PyValueError::new_err(
            "pad (RGBX output) needs a 3-sample (H, W, 3) uint8 image",
        ));
    }
    let array = image
        .cast::<PyArrayDyn<u8>>()
        .map_err(|_| PyValueError::new_err("pad (RGBX output) needs a uint8 image"))?;
    if box_times.is_none()
        && let Some(result) = resize_native_u8(py, image, geometry, channels, filter, Some(pad))
    {
        return result;
    }
    let method = match box_times {
        Some(times) => Method::BoxPyramid(times),
        None if filter == "nearest" => Method::Nearest,
        None => return Err(unknown_filter(filter)),
    };
    let ro = read_only(array)?;
    let src = c_samples(&ro, "image")?;
    let (width, height) = geometry.src;
    let (out, ow, oh) = py.detach(|| match method {
        Method::BoxPyramid(times) => {
            box_reduce::pyramid::box_downscale_rgbx(src, (height, width), times, pad)
        }
        Method::Nearest => {
            let rgb = nearest::resize_nearest_over(src, geometry, RGB);
            let (ow, oh) = geometry.dst;
            (rgbx::expand_image(&rgb, oh, ow, pad), ow, oh)
        }
    });
    to_numpy(py, out, oh, ow, RGBX)
}

/// downscale(image, factor, filter="box", pad=None)
///
/// Reduce `image` by integer `factor` per spatial axis. Accepts a planar 2D
/// (H, W) array or an interleaved HWC 3D (H, W, C) array; call planar per
/// component for subsampled chroma. Filters: "box" (power-of-2 uses the native
/// SIMD box kernels) / "bilinear" / "catmullrom" / "lanczos3" (native
/// convolution) / "nearest".
#[pyfunction]
#[pyo3(signature = (image, factor, filter="box", pad=None))]
fn downscale(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    factor: i64,
    filter: &str,
    pad: Option<u8>,
) -> PyResult<Py<PyAny>> {
    let factor = at_least_one(factor, "factor must be >= 1")?;
    let out = downscale_image(py, image, factor, filter, pad)?;
    shaped_like(py, image, out)
}

fn downscale_image(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    factor: usize,
    filter: &str,
    pad: Option<u8>,
) -> PyResult<Py<PyAny>> {
    let dims @ (height, width, channels) = image_dims(image)?;
    let geometry = Geometry::downscale((width, height), factor);
    let box_times = box_reduce::pyramid::box_halvings(filter, factor);
    if let Some(pad) = pad {
        return resize_to_rgbx(py, image, geometry, channels, filter, box_times, pad);
    }
    if let Some(times) = box_times {
        return with_dtype!(image, "image", |array: T| box_reduce::<T>(
            py, array, dims, times
        ));
    }
    resize_image(py, image, geometry, channels, filter)
}

/// resize(image, `out_h`, `out_w`, filter="box", pad=None)
///
/// Resample `image` to an exact `(out_h, out_w)` — any ratio, including
/// fractional (e.g. a requested resolution) and upscaling. Accepts a planar
/// 2D (H, W) array or an interleaved HWC 3D (H, W, C) array. Filters: "box" /
/// "bilinear" / "catmullrom" / "lanczos3"; the kernel is scaled to the reduction
/// ratio, so every filter here antialiases at any ratio. An exact power-of-2
/// "box" reduction takes the same native kernel as `downscale`.
#[pyfunction]
#[pyo3(signature = (image, out_h, out_w, filter="box", pad=None))]
fn resize(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    out_h: i64,
    out_w: i64,
    filter: &str,
    pad: Option<u8>,
) -> PyResult<Py<PyAny>> {
    let message = "out_h and out_w must be >= 1";
    let size = (at_least_one(out_h, message)?, at_least_one(out_w, message)?);
    let out = resize_to(py, image, size, filter, pad)?;
    shaped_like(py, image, out)
}

fn resize_to(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    (out_h, out_w): (usize, usize),
    filter: &str,
    pad: Option<u8>,
) -> PyResult<Py<PyAny>> {
    let dims @ (height, width, channels) = image_dims(image)?;
    let out_channels = if pad.is_some() { RGBX } else { channels };
    output_samples(
        (out_h, out_w, out_channels),
        image.cast::<PyUntypedArray>()?.dtype().itemsize(),
    )?;
    let geometry = Geometry::resize((width, height), (out_w, out_h));
    let exact_factor =
        (width % out_w == 0 && height % out_h == 0 && width / out_w == height / out_h)
            .then_some(width / out_w);
    let box_times =
        exact_factor.and_then(|factor| box_reduce::pyramid::box_halvings(filter, factor));
    if let Some(pad) = pad {
        return resize_to_rgbx(py, image, geometry, channels, filter, box_times, pad);
    }
    if let Some(times) = box_times {
        return with_dtype!(image, "image", |array: T| box_reduce::<T>(
            py, array, dims, times
        ));
    }
    resize_image(py, image, geometry, channels, filter)
}

/// `rgb_to_rgbx(rgb`, pad=0)
///
/// Expand a contiguous (H, W, 3) uint8 RGB array to a new (H, W, 4) RGBX array,
/// with the 4th byte set to `pad`. Pairs with PIL's zero-copy handoff to skip
/// the RGB->RGBX repack:
///   Image.frombuffer("RGB", (w, h), `rgb_to_rgbx(a)`, "raw", "RGBX", 0, 1)
#[pyfunction]
#[pyo3(signature = (rgb, pad=0))]
fn rgb_to_rgbx(py: Python<'_>, rgb: &Bound<'_, PyAny>, pad: u8) -> PyResult<Py<PyAny>> {
    let array = rgb
        .cast::<PyArrayDyn<u8>>()
        .map_err(|_| PyValueError::new_err("rgb must be a uint8 array"))?;
    let [height, width, RGB] = *array.shape() else {
        return Err(PyValueError::new_err("rgb must have shape (H, W, 3)"));
    };
    let ro = read_only(array)?;
    let src = c_samples(&ro, "rgb")?;
    let out = py.detach(|| rgbx::expand_image(src, height, width, pad));
    to_numpy(py, out, height, width, RGBX)
}

/// `set_simd(enabled)`
///
/// Test/benchmark hook: force the scalar kernels (False) or restore SIMD
/// dispatch (True, the default). Lets parity tests and interleaved A/B
/// benchmarks compare both paths within one process.
#[pyfunction]
fn set_simd(enabled: bool) {
    crate::cpu::set_simd(enabled);
}

#[pymodule]
fn _kernels(module: &Bound<'_, PyModule>) -> PyResult<()> {
    // SAFETY: setting a mimalloc option is thread-safe and takes effect for
    // later frees.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        mi_option_set(MI_OPTION_PURGE_DELAY, PURGE_DELAY_MS);
    };
    module.add_function(wrap_pyfunction!(compose, module)?)?;
    module.add_function(wrap_pyfunction!(downscale, module)?)?;
    module.add_function(wrap_pyfunction!(resize, module)?)?;
    module.add_function(wrap_pyfunction!(downscale_grid_box, module)?)?;
    module.add_function(wrap_pyfunction!(stitch_grid, module)?)?;
    module.add_function(wrap_pyfunction!(framed, module)?)?;
    module.add_function(wrap_pyfunction!(rgb_to_rgbx, module)?)?;
    module.add_function(wrap_pyfunction!(set_simd, module)?)?;
    Ok(())
}

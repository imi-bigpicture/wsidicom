//! Python surface of `wsidicom_kernels` (`wsidicom_kernels._kernels`): argument validation,
//! array borrows, GIL release and numpy conversion around the pure-Rust
//! kernel modules.

use numpy::ndarray::{ArrayD, Dimension, IxDyn};
use numpy::{
    Element, IntoPyArray, PyArray, PyArrayDescrMethods, PyArrayDyn, PyArrayMethods,
    PyReadonlyArray, PyReadonlyArrayDyn, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

mod box_reduce;
mod convolve;
mod convolve_float;
mod cpu;
mod exact;
mod grid;
mod nearest;
mod sample;
mod simd;
#[cfg(test)]
mod testing;
#[cfg(target_arch = "x86_64")]
mod x86;

use box_reduce::BoxSample;
use convolve::Geometry;
use sample::RGB;

/// Freed outputs go back to mimalloc's caches, so the next call's output
/// reuses committed pages instead of faulting in fresh ones.
#[cfg(target_arch = "x86_64")]
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// How long freed memory stays committed before mimalloc returns it to the
/// OS: long enough for a stream of slow tiles to reuse its pages (the 10 ms
/// default is not), short enough that a one-off large output doesn't stay
/// resident.
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

/// numpy's borrow tracking is process-wide: another Rust extension may hold
/// the array for writing.
fn borrowed_for_writing() -> PyErr {
    PyValueError::new_err("Array is borrowed for writing elsewhere — pass a copy.")
}

fn not_c_contiguous(what: &str) -> PyErr {
    PyValueError::new_err(format!("{what} must be C-contiguous."))
}

/// A view at an odd byte offset can't be read as a slice of samples.
fn misaligned(what: &str) -> PyErr {
    PyValueError::new_err(format!(
        "{what} must be aligned to the sample size — pass a copy."
    ))
}

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

/// The samples of `array` in (H, W, C) order. Checks C order because
/// `as_slice` alone also takes a Fortran-ordered array.
fn c_samples<'a, T: Element, D: Dimension>(
    array: &'a PyReadonlyArray<'_, T, D>,
    what: &str,
) -> PyResult<&'a [T]> {
    check_c_slice(array, what)?;
    array.as_slice().map_err(|_| not_c_contiguous(what))
}

/// Outputs from this many bytes are reserved before the kernel allocates
/// them, since a failed allocation aborts the process.
const CHECKED_ALLOCATION_BYTES: usize = 1 << 30;

/// The sample count of a (height, width, channels) output of `sample_bytes`
/// samples: a `ValueError` if the size overflows, a `MemoryError` if the
/// allocator can't provide it, instead of a wrapped size or an abort.
pub(crate) fn output_samples(
    (height, width, channels): (usize, usize, usize),
    sample_bytes: usize,
) -> PyResult<usize> {
    let too_large = || {
        PyValueError::new_err(format!(
            "An output of {height} x {width} x {channels} samples is too large."
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
            PyMemoryError::new_err(format!("Cannot allocate an output of {bytes} bytes."))
        })?;
    }
    Ok(samples)
}

/// `value` as a size or factor; a `ValueError` with `message` below 1.
fn at_least_one(value: i64, message: &str) -> PyResult<usize> {
    usize::try_from(value)
        .ok()
        .filter(|&value| value >= 1)
        .ok_or_else(|| PyValueError::new_err(message.to_owned()))
}

const GRAY_OR_RGB: [usize; 2] = [1, RGB];

/// (height, width, channels) of a 2D (one channel) or 3D (H, W, C) shape.
fn hwc_dims(shape: &[usize]) -> PyResult<(usize, usize, usize)> {
    match *shape {
        [height, width] => Ok((height, width, 1)),
        [height, width, channels] if GRAY_OR_RGB.contains(&channels) => {
            Ok((height, width, channels))
        }
        [_, _, _] => Err(PyValueError::new_err(
            "3D arrays must have 1 or 3 channels.",
        )),
        _ => Err(PyValueError::new_err(
            "Arrays must be 2D (planar) or 3D (H, W, C).",
        )),
    }
}

/// `hwc_dims` of an array-like to resample, which must hold a pixel.
fn image_dims(image: &Bound<'_, PyAny>) -> PyResult<(usize, usize, usize)> {
    let shape: Vec<usize> = image
        .getattr("shape")?
        .extract()
        .map_err(|_| PyValueError::new_err("Image must be a 2D or 3D array."))?;
    let dims @ (height, width, channels) = hwc_dims(&shape)?;
    if height == 0 || width == 0 || channels == 0 {
        return Err(PyValueError::new_err(
            "Image must have at least one pixel and one channel.",
        ));
    }
    Ok(dims)
}

fn read_only<'py, T: Element>(
    array: &Bound<'py, PyArrayDyn<T>>,
) -> PyResult<PyReadonlyArrayDyn<'py, T>> {
    array.try_readonly().map_err(|_| borrowed_for_writing())
}

/// `data` as a numpy array, (height, width) for one channel.
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

/// `$body` with `$binding` bound to `$array` cast to `PyArrayDyn<$t>`, `$t`
/// being whichever supported dtype it has.
macro_rules! with_dtype {
    ($array:expr_2021, $what:literal, |$binding:ident: $t:ident| $body:expr_2021) => {
        if let Ok($binding) = $array.cast::<PyArrayDyn<u8>>() {
            type $t = u8;
            $body
        } else if let Ok($binding) = $array.cast::<PyArrayDyn<u16>>() {
            type $t = u16;
            $body
        } else {
            Err(PyValueError::new_err(concat!(
                $what,
                " must be a 2D/3D uint8 or uint16 array in native byte order."
            )))
        }
    };
}

// ---- downscale / resize --------------------------------------------------------

fn box_downscale_array<T: BoxSample>(
    py: Python<'_>,
    array: &Bound<'_, PyArrayDyn<T>>,
    dims @ (_, _, channels): (usize, usize, usize),
    halvings: u32,
) -> PyResult<Py<PyAny>> {
    let borrow = read_only(array)?;
    let src = c_samples(&borrow, "Image")?;
    let (out, width, height) =
        py.detach(|| box_reduce::pyramid::box_downscale(src, dims, halvings));
    to_numpy(py, out, height, width, channels)
}

/// The row-major grid of `tiles`, `grid_width` per row, as one image; see
/// `wsidicom_kernels.downscale_grid`.
#[pyfunction]
fn stitch_grid(
    py: Python<'_>,
    tiles: Vec<Bound<'_, PyAny>>,
    grid_width: usize,
) -> PyResult<Py<PyAny>> {
    let first = tiles
        .first()
        .ok_or_else(|| PyValueError::new_err("Cannot stitch an empty tile grid."))?;
    let out = with_dtype!(first, "Tile", |_tile: T| grid::stitch_grid::<T>(
        py, &tiles, grid_width
    ))?;
    shaped_like(py, first, out)
}

/// The box reduction by a power-of-2 `factor` of the row-major grid of
/// `tiles`, `grid_width` per row, as one image; with `size` ((height, width))
/// and `fill` (one sample per channel), at the top left of a frame of that
/// size filled with `fill`. See `wsidicom_kernels.downscale_grid`, which
/// checks when this path is exact.
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
    let halvings = box_reduce::pyramid::box_halvings("box", factor)
        .ok_or_else(|| PyValueError::new_err("Factor must be a power of 2."))?;
    let first = tiles
        .first()
        .ok_or_else(|| PyValueError::new_err("Cannot downscale an empty tile grid."))?;
    let out = with_dtype!(first, "Tile", |_tile: T| grid::box_grid::<T>(
        py,
        &tiles,
        grid_width,
        halvings,
        (size, fill.as_ref())
    ))?;
    shaped_like(py, first, out)
}

/// `image` at the top left of a frame of `size` ((height, width)) filled
/// with `fill` (one sample per channel); see `wsidicom_kernels.downscale_grid`.
#[pyfunction]
fn framed(
    py: Python<'_>,
    image: Bound<'_, PyAny>,
    size: (usize, usize),
    fill: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let out = with_dtype!(image, "Image", |_image: T| grid::framed::<T>(
        py,
        &image,
        (Some(size), Some(&fill))
    ))?;
    shaped_like(py, &image, out)
}

/// Resize: nearest-neighbour (`nearest`) or by convolution (`convolve` for
/// uint8, `convolve_float` for uint16).
fn resize_image(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    geometry: Geometry,
    channels: usize,
    resample: &str,
) -> PyResult<Py<PyAny>> {
    if resample == "nearest" {
        return with_dtype!(image, "Image", |array: T| resize_nearest_array::<T>(
            py, array, geometry, channels
        ));
    }
    if convolve::Kernel::parse(resample).is_none() {
        return Err(unknown_filter(resample));
    }
    if let Some(result) = resize_native_u8(py, image, geometry, channels, resample) {
        return result;
    }
    if let Some(result) = resize_native_u16(py, image, geometry, channels, resample) {
        return result;
    }
    Err(PyValueError::new_err(
        "Image must be a 2D/3D uint8 or uint16 array in native byte order.",
    ))
}

fn unknown_filter(resample: &str) -> PyErr {
    PyValueError::new_err(format!("Unknown filter: {resample}."))
}

/// The native uint16 convolution resize, or None for other dtypes and
/// filters.
fn resize_native_u16(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    geometry: Geometry,
    channels: usize,
    resample: &str,
) -> Option<PyResult<Py<PyAny>>> {
    let kernel = convolve::Kernel::parse(resample)?;
    let array = image.cast::<PyArrayDyn<u16>>().ok()?;
    if let Some(exact_kernel) = geometry
        .fits_image()
        .then(|| exact::exact_kernel_u16(kernel, geometry.src, geometry.dst))
        .flatten()
    {
        return Some(exact_resize(py, array, exact_kernel, geometry, channels));
    }
    Some(resize_float_engine(py, array, geometry, channels, kernel))
}

/// A resize by an exact kernel, which `geometry` fits.
fn exact_resize<T: Element>(
    py: Python<'_>,
    array: &Bound<'_, PyArrayDyn<T>>,
    exact_kernel: exact::ExactKernel<T>,
    geometry: Geometry,
    channels: usize,
) -> PyResult<Py<PyAny>> {
    let ((width, height), (out_width, out_height)) = (geometry.src, geometry.dst);
    let borrow = read_only(array)?;
    let src = c_samples(&borrow, "Image")?;
    let out = py.detach(|| exact_kernel(src, width, height, channels));
    to_numpy(py, out, out_height, out_width, channels)
}

fn resize_nearest_array<T: Element + bytemuck::Pod>(
    py: Python<'_>,
    array: &Bound<'_, PyArrayDyn<T>>,
    geometry: Geometry,
    channels: usize,
) -> PyResult<Py<PyAny>> {
    let borrow = read_only(array)?;
    let src = c_samples(&borrow, "Image")?;
    let out = py.detach(|| nearest::resize_nearest_over(src, geometry, channels));
    let (out_width, out_height) = geometry.dst;
    to_numpy(py, out, out_height, out_width, channels)
}

/// The native uint8 resize, or None for other dtypes and filters.
fn resize_native_u8(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    geometry: Geometry,
    channels: usize,
    resample: &str,
) -> Option<PyResult<Py<PyAny>>> {
    let kernel = convolve::Kernel::parse(resample)?;
    let array = image.cast::<PyArrayDyn<u8>>().ok()?;
    if geometry.reduces_past(FIXED_POINT_MAX_REDUCTION) {
        return Some(resize_float_engine(py, array, geometry, channels, kernel));
    }
    if let Some(exact_kernel) = geometry
        .fits_image()
        .then(|| exact::exact_kernel(kernel, geometry.src, geometry.dst))
        .flatten()
    {
        return Some(exact_resize(py, array, exact_kernel, geometry, channels));
    }
    let ((width, height), (out_width, out_height)) = (geometry.src, geometry.dst);
    let plan = py.detach(|| convolve::Plan::cached(geometry, kernel, channels));
    let resized = || {
        let borrow = read_only(array)?;
        let src = c_samples(&borrow, "Image")?;
        let out = py.detach(|| convolve::resize_u8(src, (width, height), &plan));
        to_numpy(py, out, out_height, out_width, channels)
    };
    Some(resized())
}

/// The reduction (per axis) past which the uint8 fixed-point convolution
/// loses accuracy: each tap is rounded, and over thousands of taps the
/// roundings add up.
const FIXED_POINT_MAX_REDUCTION: usize = 128;

/// A resize by the float32 convolution: all of uint16's, and uint8's past
/// `FIXED_POINT_MAX_REDUCTION`.
fn resize_float_engine<T: convolve_float::FloatResample + Element>(
    py: Python<'_>,
    array: &Bound<'_, PyArrayDyn<T>>,
    geometry: Geometry,
    channels: usize,
    kernel: convolve::Kernel,
) -> PyResult<Py<PyAny>> {
    let borrow = read_only(array)?;
    let src = c_samples(&borrow, "Image")?;
    let (out_width, out_height) = geometry.dst;
    let out = py.detach(|| {
        let plan = convolve_float::FloatPlan::cached(geometry, kernel, channels);
        convolve_float::resize_float(src, geometry.src, &plan)
    });
    to_numpy(py, out, out_height, out_width, channels)
}

/// Reduce `image` by integer `factor` per spatial axis. `image` is a planar
/// (H, W) or interleaved (H, W, C) array; call planar per component for
/// subsampled chroma. `resample`: "box" (native SIMD box kernels at a
/// power-of-2 factor), "bilinear", "catmullrom", "lanczos3" or "nearest".
#[pyfunction]
#[pyo3(signature = (image, factor, resample))]
fn downscale(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    factor: i64,
    resample: &str,
) -> PyResult<Py<PyAny>> {
    let factor = at_least_one(factor, "Factor must be >= 1.")?;
    let out = downscale_image(py, image, factor, resample)?;
    shaped_like(py, image, out)
}

fn downscale_image(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    factor: usize,
    resample: &str,
) -> PyResult<Py<PyAny>> {
    let (height, width, channels) = image_dims(image)?;
    let geometry = Geometry::downscale((width, height), factor);
    if let Some(halvings) = box_reduce::pyramid::box_halvings(resample, factor) {
        return with_dtype!(image, "Image", |array: T| box_downscale_array::<T>(
            py,
            array,
            (width, height, channels),
            halvings
        ));
    }
    resize_image(py, image, geometry, channels, resample)
}

/// Resample `image` to exactly `height` x `width` pixels, at any ratio,
/// including fractional reductions and enlargements. `image` is a planar
/// (H, W) or interleaved (H, W, C) array. `resample`: "box", "bilinear",
/// "catmullrom", "lanczos3" or "nearest"; the kernel scales with the
/// reduction, so every filter but "nearest" antialiases. A power-of-2 "box"
/// reduction takes the same native kernels as `downscale`.
#[pyfunction]
#[pyo3(signature = (image, height, width, resample))]
fn resize(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    height: i64,
    width: i64,
    resample: &str,
) -> PyResult<Py<PyAny>> {
    let message = "Height and width must be >= 1.";
    let size = (
        at_least_one(height, message)?,
        at_least_one(width, message)?,
    );
    let out = resize_to(py, image, size, resample)?;
    shaped_like(py, image, out)
}

fn resize_to(
    py: Python<'_>,
    image: &Bound<'_, PyAny>,
    (out_height, out_width): (usize, usize),
    resample: &str,
) -> PyResult<Py<PyAny>> {
    let (height, width, channels) = image_dims(image)?;
    output_samples(
        (out_height, out_width, channels),
        image.cast::<PyUntypedArray>()?.dtype().itemsize(),
    )?;
    let geometry = Geometry::resize((width, height), (out_width, out_height));
    let same_integer_factor = (width % out_width == 0
        && height % out_height == 0
        && width / out_width == height / out_height)
        .then_some(width / out_width);
    if let Some(halvings) =
        same_integer_factor.and_then(|factor| box_reduce::pyramid::box_halvings(resample, factor))
    {
        return with_dtype!(image, "Image", |array: T| box_downscale_array::<T>(
            py,
            array,
            (width, height, channels),
            halvings
        ));
    }
    resize_image(py, image, geometry, channels, resample)
}

/// Test and benchmark hook: `False` forces the scalar kernels, `True` (the
/// default) restores SIMD dispatch, so both paths compare in one process.
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
    module.add_function(wrap_pyfunction!(downscale, module)?)?;
    module.add_function(wrap_pyfunction!(resize, module)?)?;
    module.add_function(wrap_pyfunction!(downscale_grid_box, module)?)?;
    module.add_function(wrap_pyfunction!(stitch_grid, module)?)?;
    module.add_function(wrap_pyfunction!(framed, module)?)?;
    module.add_function(wrap_pyfunction!(set_simd, module)?)?;
    Ok(())
}

//! Where the x86 2x box kernels write: a new image, or rows of the caller's
//! (a tile's place in a grid).

use std::mem::MaybeUninit;

use crate::cpu::{nt_stores_for, unwritten_output, with_prefault, written_output};
use crate::x86::sfence_if_streamed;

/// The output rows of a 2x box kernel: `height` rows of `row_len` samples,
/// row `out_y` at `dst + out_y * stride`.
pub(super) struct BoxRows<T> {
    /// The new image the rows are, or none when they are the caller's.
    image: Option<Vec<MaybeUninit<T>>>,
    dst: *mut T,
    layout: RowLayout,
}

#[derive(Clone, Copy)]
pub(super) struct RowLayout {
    pub(super) stride: usize,
    row_len: usize,
    height: usize,
    /// Whether a row's stores may run past its end into the rows after it,
    /// which overwrite them: in a new image yes, up to its end; in the
    /// caller's rows no, the samples past a row are someone else's.
    spills: bool,
}

impl RowLayout {
    pub(super) fn row_start(self, out_y: usize) -> usize {
        out_y * self.stride
    }

    /// The offset no store of row `out_y` may reach (exclusive).
    pub(super) fn store_end(self, out_y: usize) -> usize {
        if self.spills {
            self.height * self.stride
        } else {
            out_y * self.stride + self.row_len
        }
    }
}

impl<T> BoxRows<T> {
    pub(super) fn new_image(height: usize, row_len: usize) -> BoxRows<T> {
        let mut image = unwritten_output::<T>(height * row_len);
        let dst = image.as_mut_ptr().cast::<T>();
        BoxRows {
            image: Some(image),
            dst,
            layout: RowLayout {
                stride: row_len,
                row_len,
                height,
                spills: true,
            },
        }
    }

    /// The caller's `height` rows of `row_len` samples at `dst`, `stride`
    /// samples apart.
    ///
    /// # Safety
    /// `dst` holds those rows, and nothing else writes them until the kernel
    /// is done.
    pub(super) unsafe fn in_place(
        dst: *mut T,
        stride: usize,
        height: usize,
        row_len: usize,
    ) -> BoxRows<T> {
        BoxRows {
            image: None,
            dst,
            layout: RowLayout {
                stride,
                row_len,
                height,
                spills: false,
            },
        }
    }

    pub(super) fn dst(&self) -> *mut T {
        self.dst
    }

    pub(super) fn layout(&self) -> RowLayout {
        self.layout
    }

    pub(super) fn row_count(&self) -> usize {
        self.layout.height
    }

    /// Whether to stream the stores past the cache: a large new image (see
    /// `nt_stores_for`); never the caller's rows, which it reads next.
    pub(super) fn streams(&self) -> bool {
        self.image.is_some()
            && nt_stores_for(self.layout.height * self.layout.row_len * size_of::<T>())
    }

    /// Runs `rows(first_row, end_row)` over every row (a new image's behind
    /// the prefault helper, see `with_prefault`), fences streamed stores, and
    /// returns the new image, or an empty one for the caller's rows.
    ///
    /// # Safety
    /// `rows(first_row, end_row)` writes every sample of those rows, and its
    /// stores for row `out_y` stay below `layout().store_end(out_y)`.
    pub(super) unsafe fn write(self, streamed: bool, mut rows: impl FnMut(usize, usize)) -> Vec<T> {
        let RowLayout { stride, height, .. } = self.layout;
        let Some(image) = self.image else {
            rows(0, height);
            sfence_if_streamed(streamed);
            return Vec::new();
        };
        // SAFETY: `image` is this function's, its rows `stride` samples long;
        // a band's stores stay in its rows plus the spill into the next ones
        // (the caller's contract).
        unsafe {
            with_prefault(
                self.dst.cast::<u8>(),
                height * stride * size_of::<T>(),
                stride * size_of::<T>(),
                height,
                rows,
            );
        };
        sfence_if_streamed(streamed);
        // SAFETY: every row was written in full (the caller's contract).
        unsafe { written_output(image) }
    }
}

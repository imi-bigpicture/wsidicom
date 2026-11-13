//! Runtime SIMD detection (with the scalar switch), environment switches,
//! output buffers allocated without clearing, and the page prefault of large
//! outputs.

/// Off forces the scalar kernels, so tests and benchmarks can compare them
/// with SIMD in one process.
static SIMD_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[inline]
pub(crate) fn simd_enabled() -> bool {
    SIMD_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn set_simd(enabled: bool) {
    SIMD_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

// The crate builds for baseline x86-64, so every call into an AVX2 kernel is
// gated on these (std caches the detection).
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn has_avx2() -> bool {
    simd_enabled() && std::is_x86_feature_detected!("avx2")
}
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn has_avx2_fma() -> bool {
    has_avx2() && std::is_x86_feature_detected!("fma")
}

/// Outputs at least this big are not re-read from cache before they leave,
/// so non-temporal stores pay.
#[cfg(target_arch = "x86_64")]
pub(crate) const LARGE_OUTPUT_BYTES: usize = 4 << 20;

/// Outputs from which `with_prefault` touches pages in a helper thread.
/// Smaller outputs (tiles) mostly reuse pages the allocator recycled, which
/// have no faults to hide.
pub(crate) const PREFAULT_OUTPUT_BYTES: usize = 8 << 20;

/// On unless set to "0"; `default` when unset.
#[cfg(target_arch = "x86_64")]
fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name).map_or(default, |value| value != "0")
}

/// Bytes an overlapping store may write past the last row of a
/// `with_prefault` band.
const MAX_STORE_SLOP: usize = 64;

/// Run a row-range kernel body over `out_height` output rows. For a large
/// output, a scoped helper thread touches its pages in order while `rows`
/// follows behind, called on bands of rows only once their pages are
/// committed: the page faults (serialized within a process) then overlap the
/// compute instead of adding to it. Splitting the output between kernel and
/// helper instead would serialize their faults. Not compute parallelism — the
/// caller still owns threading.
///
/// # Safety
/// `dst_ptr..dst_ptr+total_bytes` must be a live, exclusively-owned output
/// buffer and `rows(start, end)` must only write rows in `[start, end)` (row =
/// `row_bytes`), plus at most `MAX_STORE_SLOP` bytes into row `end`.
pub(crate) unsafe fn with_prefault(
    dst_ptr: *mut u8,
    total_bytes: usize,
    row_bytes: usize,
    out_height: usize,
    mut rows: impl FnMut(usize, usize),
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    const PAGE_BYTES: usize = 4096;
    const PROGRESS_STEP_BYTES: usize = 16 * PAGE_BYTES;
    const BAND_TARGET_BYTES: usize = 64 << 10;
    /// Bands are whole batches of the transposed kernels' rows.
    const BATCH_ROWS: usize = 16;
    const SPINS_BEFORE_YIELD: u32 = 256;
    // x86-64 only: on aarch64 (macOS) large outputs come back as recycled
    // pages, so the helper only costs.
    if !(cfg!(target_arch = "x86_64") && total_bytes >= PREFAULT_OUTPUT_BYTES && out_height >= 2) {
        return rows(0, out_height);
    }
    // The helper has touched every page below this offset and writes nothing
    // past it that `rows` has written.
    let committed_bytes = AtomicUsize::new(0);
    let (dst_addr, committed_bytes) = (dst_ptr as usize, &committed_bytes);
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let dst = dst_addr as *mut u8;
            let mut offset = 0usize;
            while offset < total_bytes {
                // Volatile so the page-committing write isn't elided.
                // SAFETY: dst+offset is inside the caller's buffer, at or past
                // `committed_bytes`, which `rows` has not been let past.
                unsafe { std::ptr::write_volatile(dst.add(offset), 0) };
                offset += PAGE_BYTES;
                if offset.is_multiple_of(PROGRESS_STEP_BYTES) {
                    committed_bytes.store(offset, Ordering::Release);
                }
            }
            committed_bytes.store(usize::MAX, Ordering::Release);
        });
        let band_rows = (BAND_TARGET_BYTES / row_bytes.max(1))
            .next_multiple_of(BATCH_ROWS)
            .max(BATCH_ROWS);
        for band_start in (0..out_height).step_by(band_rows) {
            let band_end = (band_start + band_rows).min(out_height);
            let band_end_bytes = (band_end * row_bytes + MAX_STORE_SLOP).min(total_bytes);
            let mut spins = 0u32;
            while committed_bytes.load(Ordering::Acquire) < band_end_bytes {
                // Yield after a brief spin: under a caller's full thread pool
                // the helper needs the core.
                if spins < SPINS_BEFORE_YIELD {
                    spins += 1;
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
            rows(band_start, band_end);
        }
    });
}

/// `rows` typed as `with_prefault`'s row-band closure, so a closure bound
/// before an `unsafe { with_prefault(..) }` call (keeping its body out of the
/// unsafe block) compiles as an inline argument would.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn row_bands<RowsFn: FnMut(usize, usize)>(rows: RowsFn) -> RowsFn {
    rows
}

/// NEON is baseline on aarch64: only the SIMD switch applies.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn use_neon() -> bool {
    simd_enabled()
}

/// The ARMv8.1 rounding multiply-accumulate (`vqrdmlah`).
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn has_rdm() -> bool {
    static DETECTED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::arch::is_aarch64_feature_detected!("rdm"));
    *DETECTED
}

/// Whether a box kernel streams an `out_bytes` output past the cache
/// (`WSIDICOM_KERNELS_NT=0` disables); pays under multi-worker bandwidth
/// contention.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn nt_stores_for(out_bytes: usize) -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| env_flag("WSIDICOM_KERNELS_NT", true));
    *ENABLED && out_bytes >= LARGE_OUTPUT_BYTES
}

/// An output the kernel writes completely before anything reads it,
/// allocated without clearing.
pub(crate) fn unwritten_output<T>(len: usize) -> Vec<std::mem::MaybeUninit<T>> {
    let mut out = Vec::with_capacity(len);
    // SAFETY: MaybeUninit elements need no initialization.
    unsafe { out.set_len(len) };
    out
}

/// `unwritten_output`'s vector once the kernel has written it.
///
/// # Safety
/// Every element of `out` has been written.
pub(crate) unsafe fn written_output<T>(out: Vec<std::mem::MaybeUninit<T>>) -> Vec<T> {
    let mut out = std::mem::ManuallyDrop::new(out);
    // SAFETY: MaybeUninit<T> has T's layout, and every element is initialized
    // (the caller's contract); the allocation moves over unchanged.
    unsafe { Vec::from_raw_parts(out.as_mut_ptr().cast::<T>(), out.len(), out.capacity()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn written_outputs_keep_their_samples() {
        let mut out = unwritten_output::<u16>(1000);
        assert_eq!(out.len(), 1000);
        for (index, sample) in out.iter_mut().enumerate() {
            sample.write(index as u16 * 3);
        }
        // SAFETY: every element is written.
        let out = unsafe { written_output(out) };
        assert!(
            out.iter()
                .enumerate()
                .all(|(index, &sample)| sample == index as u16 * 3)
        );
        assert!(unwritten_output::<u8>(0).is_empty());
    }

    /// Runs `with_prefault` over an `out_height`-row output, each band
    /// filling its rows with their row numbers; checks the bands cover the
    /// rows once and in order, and returns them.
    fn assert_bands_cover_the_rows(row_bytes: usize, out_height: usize) -> Vec<(usize, usize)> {
        let mut out = vec![0xFFu8; row_bytes * out_height];
        let dst_ptr = out.as_mut_ptr();
        let mut bands = Vec::new();
        // SAFETY: `out` is this function's; each band writes only its rows.
        unsafe {
            with_prefault(
                dst_ptr,
                row_bytes * out_height,
                row_bytes,
                out_height,
                |start, end| {
                    bands.push((start, end));
                    for row in start..end {
                        std::ptr::write_bytes(dst_ptr.add(row * row_bytes), row as u8, row_bytes);
                    }
                },
            );
        }
        assert_eq!(bands.first().map(|band| band.0), Some(0));
        assert_eq!(bands.last().map(|band| band.1), Some(out_height));
        assert!(bands.windows(2).all(|pair| pair[0].1 == pair[1].0));
        assert!(bands.iter().all(|&(start, end)| start < end));
        for (row, samples) in out.chunks_exact(row_bytes).enumerate() {
            assert!(
                samples.iter().all(|&sample| sample == row as u8),
                "row {row}"
            );
        }
        bands
    }

    #[test]
    fn small_outputs_run_as_one_band() {
        assert_eq!(assert_bands_cover_the_rows(1000, 50), [(0, 50)]);
    }

    #[test]
    fn large_outputs_run_in_bands_behind_the_prefault() {
        let row_bytes = 4096 + 12;
        let out_height = PREFAULT_OUTPUT_BYTES.div_ceil(row_bytes) + 7;
        let bands = assert_bands_cover_the_rows(row_bytes, out_height);
        if cfg!(target_arch = "x86_64") {
            assert!(bands.len() > 1);
        }
        // A single row, however large, is one band.
        assert_eq!(
            assert_bands_cover_the_rows(PREFAULT_OUTPUT_BYTES, 1),
            [(0, 1)]
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn only_large_outputs_stream() {
        assert!(!nt_stores_for(0));
        assert!(!nt_stores_for(LARGE_OUTPUT_BYTES - 1));
        assert_eq!(
            nt_stores_for(LARGE_OUTPUT_BYTES),
            env_flag("WSIDICOM_KERNELS_NT", true)
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn unset_environment_flags_take_their_default() {
        let name = "WSIDICOM_KERNELS_TEST_FLAG_NEVER_SET";
        assert!(env_flag(name, true));
        assert!(!env_flag(name, false));
    }
}

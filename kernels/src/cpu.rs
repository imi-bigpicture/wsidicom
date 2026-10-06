//! What the CPU offers and how outputs are written: runtime SIMD detection
//! (with the scalar kill switch and the AVX-512 opt-out), the environment
//! switches, output buffers allocated without clearing, the page-prefault
//! helper for large outputs, and the write prefetch.

// Global SIMD kill switch (default on). `set_simd(False)` from Python forces
// the scalar paths so tests and benchmarks can compare SIMD and scalar
// in-process.
static SIMD_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
#[inline]
pub(crate) fn simd_enabled() -> bool {
    SIMD_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

// Runtime CPU feature detection (cached by std after first call). The crate is
// built at baseline x86-64, so these gate every call into an AVX2 kernel; when
// they return false (non-AVX2 x86) the scalar path runs instead.
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
// AVX-512 tier: F+BW+VL is the working set for the 512-bit integer
// kernels. WSIDICOM_KERNELS_AVX512=0 drops to the AVX2 tier (per-tier testing; an
// escape hatch for hosts where 512-bit execution downclocks).
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn has_avx512() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| env_flag("WSIDICOM_KERNELS_AVX512", true));
    *ON && simd_enabled()
        && std::is_x86_feature_detected!("avx512f")
        && std::is_x86_feature_detected!("avx512bw")
        && std::is_x86_feature_detected!("avx512vl")
        && std::is_x86_feature_detected!("fma")
}

/// No x86 tier off x86-64: tier-dependent planning then picks the scalar
/// (or NEON) paths.
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(crate) fn has_avx2() -> bool {
    false
}
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(crate) fn has_avx512() -> bool {
    false
}

/// Outputs at least this big are not re-read from cache before they leave,
/// so non-temporal stores pay.
#[cfg(target_arch = "x86_64")]
pub(crate) const LARGE_OUTPUT_BYTES: usize = 4 << 20;

/// Outputs from which `with_prefault` touches pages in a helper thread.
/// Smaller outputs (tiles) mostly reuse pages the allocator recycled from
/// earlier outputs, which have no faults to hide; larger ones are mostly
/// fresh memory.
pub(crate) const PREFAULT_OUTPUT_BYTES: usize = 8 << 20;

/// An environment switch: on unless set to "0" (unset: `default`).
fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name).map_or(default, |value| value != "0")
}

/// `WSIDICOM_KERNELS_PREFAULT=0` disables `with_prefault`'s helper thread.
/// x86-64 only: on aarch64 macOS the helper was measured without gain (large
/// outputs come back from recycled pages) and slowed the fastest kernels.
#[inline]
fn prefault_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| env_flag("WSIDICOM_KERNELS_PREFAULT", true));
    cfg!(target_arch = "x86_64") && *ON
}

/// Run a row-range kernel body over `oh` output rows. For a large output, a
/// scoped helper thread touches its pages in order while `rows` follows
/// behind, called on bands of rows only once their pages are committed: the
/// kernel never takes a page fault itself, so the faults (serialized within
/// a process) overlap the compute instead of adding to it — splitting the
/// output instead, the kernel faulting one half while the helper faults the
/// other, would serialize the two. A transient page-toucher, not compute
/// parallelism — the caller still owns threading.
///
/// # Safety
/// `dp..dp+total_bytes` must be a live, exclusively-owned output buffer and
/// `rows(lo, hi)` must only write rows in `[lo, hi)` (row = `row_bytes`), plus
/// at most a few bytes of overlapping-store slop into row `hi`.
pub(crate) unsafe fn with_prefault(
    dp: *mut u8,
    total_bytes: usize,
    row_bytes: usize,
    oh: usize,
    mut rows: impl FnMut(usize, usize),
) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    const PAGE: usize = 4096;
    // The helper publishes its progress every this many bytes.
    const COMMIT_STEP: usize = 16 * PAGE;
    // The kernel runs in bands of about this many bytes, in whole batches of
    // `BATCH_ROWS` (the transposed kernels' batch).
    const BAND_TARGET_BYTES: usize = 64 << 10;
    const BATCH_ROWS: usize = 16;
    // Bytes an overlapping store may write past a band's last row.
    const MAX_STORE_SLOP: usize = 64;
    // Polls of `committed` before the waiting kernel yields its core.
    const SPINS_BEFORE_YIELD: u32 = 256;
    if !(prefault_enabled() && total_bytes >= PREFAULT_OUTPUT_BYTES && oh >= 2) {
        return rows(0, oh);
    }
    // Bytes from dp below which the helper has touched every page (and past
    // which it writes nothing the kernel has written).
    let committed = AtomicUsize::new(0);
    let (addr, committed) = (dp as usize, &committed);
    std::thread::scope(|scope| {
        scope.spawn(move || {
            let base = addr as *mut u8;
            let mut offset = 0usize;
            while offset < total_bytes {
                // Volatile so the page-committing write isn't elided.
                // SAFETY: base+offset is inside the caller's buffer, at or past
                // `committed`, which `rows` has not been let past.
                unsafe { std::ptr::write_volatile(base.add(offset), 0) };
                offset += PAGE;
                if offset.is_multiple_of(COMMIT_STEP) {
                    committed.store(offset, Ordering::Release);
                }
            }
            committed.store(usize::MAX, Ordering::Release);
        });
        // Each band runs once its rows and their store slop are committed.
        let band = (BAND_TARGET_BYTES / row_bytes.max(1))
            .next_multiple_of(BATCH_ROWS)
            .max(BATCH_ROWS);
        for lo in (0..oh).step_by(band) {
            let hi = (lo + band).min(oh);
            let need = (hi * row_bytes + MAX_STORE_SLOP).min(total_bytes);
            let mut spins = 0u32;
            while committed.load(Ordering::Acquire) < need {
                // Briefly spin, then yield: under a caller's full thread pool
                // the helper needs the core.
                if spins < SPINS_BEFORE_YIELD {
                    spins += 1;
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
            rows(lo, hi);
        }
    });
}

/// `rows` typed as `with_prefault`'s row-band closure. A closure bound
/// before an `unsafe { with_prefault(..) }` call (keeping its body out of the
/// unsafe block) takes its signature from this bound, as an inline argument
/// does, and so compiles to the same code.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) fn row_bands<RowsFn: FnMut(usize, usize)>(rows: RowsFn) -> RowsFn {
    rows
}

// NEON is baseline on aarch64 — no CPU detection, only the kill switch.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn use_neon() -> bool {
    simd_enabled()
}

/// The ARMv8.1 rounding multiply-accumulate (`vqrdmlah`), on Apple silicon
/// and most cores since; detected once.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn has_rdm() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::arch::is_aarch64_feature_detected!("rdm"));
    *ON
}

/// Whether a box kernel should stream an `out_bytes` output past the cache
/// (`WSIDICOM_KERNELS_NT=0` disables). Pays under multi-worker bandwidth contention.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn nt_stores_for(out_bytes: usize) -> bool {
    static NT: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| env_flag("WSIDICOM_KERNELS_NT", true));
    *NT && out_bytes >= LARGE_OUTPUT_BYTES
}

/// An output a kernel writes completely before anything reads it, allocated
/// without clearing: a recycled allocation would be cleared for nothing.
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

/// The most bytes of a destination run worth prefetching: past the first few
/// cache lines the hardware prefetcher has picked up the row's stream.
#[cfg(target_arch = "x86_64")]
const PREFETCH_MAX_BYTES: usize = 512;

/// Hint that the `bytes` at `dst` are about to be written, so their cache lines
/// are fetched (for ownership, where the CPU supports it) ahead of the stores.
/// Prefetches never fault, so `dst` need not be dereferenceable.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn prefetch_for_write(dst: *const u8, bytes: usize) {
    let lines = (0..bytes.min(PREFETCH_MAX_BYTES))
        .step_by(64)
        .map(|off| dst.wrapping_add(off));
    if has_prefetchw() {
        for line in lines {
            // SAFETY: the CPU implements PREFETCHW (checked); a prefetch
            // reads no memory architecturally and cannot fault.
            unsafe {
                std::arch::asm!(
                    "prefetchw byte ptr [{0}]",
                    in(reg) line,
                    options(nostack, preserves_flags, readonly)
                );
            };
        }
    } else {
        for line in lines {
            // SAFETY: SSE is baseline on x86-64; a prefetch cannot fault.
            unsafe {
                std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(
                    line as *const i8,
                );
            };
        }
    }
}

/// No-op off x86-64. On aarch64 an `prfm pstl1keep` twin was measured on
/// Apple silicon: no gain for integer canvases, a small one only for large
/// float32 tiles.
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(crate) fn prefetch_for_write(_dst: *const u8, _bytes: usize) {}

/// Hint that the cache line at `src` is about to be read. Prefetches never
/// fault, so `src` need not be dereferenceable.
#[cfg(target_arch = "x86_64")]
#[inline]
pub(crate) fn prefetch_for_read(src: *const u8) {
    // SAFETY: SSE is baseline on x86-64; a prefetch cannot fault.
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(src as *const i8);
    }
}

/// No-op off x86-64.
#[cfg(not(target_arch = "x86_64"))]
#[inline]
pub(crate) fn prefetch_for_read(_src: *const u8) {}

/// PREFETCHW support (CPUID `0x8000_0001` ECX bit 8). `is_x86_feature_detected`
/// can't name it on stable Rust.
#[cfg(target_arch = "x86_64")]
fn has_prefetchw() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        use std::arch::x86_64::__cpuid;
        // Leaf 0x8000_0001 is only read after 0x8000_0000 reports it exists.
        __cpuid(0x8000_0000).eax >= 0x8000_0001 && __cpuid(0x8000_0001).ecx & (1 << 8) != 0
    });
    *ON
}

pub(crate) fn set_simd(enabled: bool) {
    SIMD_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
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
        assert!(unwritten_output::<f32>(0).is_empty());
    }

    /// Runs `with_prefault` over an `oh`-row output of `row_bytes` rows,
    /// each band filling its rows with their row numbers, and checks the
    /// bands cover the rows in order, once, and the output is theirs.
    fn assert_bands_cover_the_rows(row_bytes: usize, oh: usize) -> Vec<(usize, usize)> {
        let mut out = vec![0xFFu8; row_bytes * oh];
        let dp = out.as_mut_ptr();
        let mut bands = Vec::new();
        // SAFETY: `out` is this function's; each band writes only its rows.
        unsafe {
            with_prefault(dp, row_bytes * oh, row_bytes, oh, |lo, hi| {
                bands.push((lo, hi));
                for row in lo..hi {
                    std::ptr::write_bytes(dp.add(row * row_bytes), row as u8, row_bytes);
                }
            });
        }
        assert_eq!(bands.first().map(|band| band.0), Some(0));
        assert_eq!(bands.last().map(|band| band.1), Some(oh));
        assert!(bands.windows(2).all(|pair| pair[0].1 == pair[1].0));
        assert!(bands.iter().all(|&(lo, hi)| lo < hi));
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
        let oh = PREFAULT_OUTPUT_BYTES.div_ceil(row_bytes) + 7;
        let bands = assert_bands_cover_the_rows(row_bytes, oh);
        if prefault_enabled() {
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
    fn simd_tiers_follow_the_cpu() {
        // The tests never turn the SIMD switch off.
        assert!(simd_enabled());
        assert_eq!(has_avx2(), std::is_x86_feature_detected!("avx2"));
        assert!(!has_avx2_fma() || has_avx2());
        assert!(!has_avx512() || has_avx2_fma());
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

    #[test]
    fn unset_environment_flags_take_their_default() {
        let name = "WSIDICOM_KERNELS_TEST_FLAG_NEVER_SET";
        assert!(env_flag(name, true));
        assert!(!env_flag(name, false));
    }

    #[test]
    fn write_prefetches_do_not_fault() {
        let out = [0u8; 100];
        prefetch_for_write(out.as_ptr(), 0);
        prefetch_for_write(out.as_ptr(), 100);
        // Past the buffer, and nowhere at all: prefetches never fault.
        prefetch_for_write(out.as_ptr(), 1 << 20);
        prefetch_for_write(std::ptr::dangling(), 4096);
    }
}

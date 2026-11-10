//! Test-build-only heap observation. Never linked into release binaries.
//! Counts requested Rust heap bytes on the current thread inside `capture`,
//! excluding stack and allocator overhead.
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
};

#[derive(Clone, Copy, Default, Debug)]
pub(super) struct Heap {
    pub allocations: usize,
    pub live: usize,
    pub peak: usize,
}

#[derive(Debug)]
pub(super) struct Snapshot {
    pub heap: Heap,
}

thread_local! {
    static HEAP: Cell<Option<Heap>> = const { Cell::new(None) };
}

struct ObservedAllocator;
#[global_allocator]
static ALLOCATOR: ObservedAllocator = ObservedAllocator;

fn record(allocated: usize, freed: usize) {
    // Thread-local storage may already be destroyed during thread exit.
    let _ = HEAP.try_with(|cell| {
        if let Some(mut heap) = cell.get() {
            if allocated != 0 {
                heap.allocations += 1;
            }
            // A moving realloc can require both old and new storage.
            heap.peak = heap.peak.max(heap.live + allocated);
            heap.live = (heap.live + allocated).saturating_sub(freed);
            cell.set(Some(heap));
        }
    });
}

// SAFETY: forwards every pointer/layout contract unchanged to System. Counter
// updates allocate no memory, never unwind, and tolerate destroyed thread TLS.
unsafe impl GlobalAlloc for ObservedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            record(layout.size(), 0);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc_zeroed(layout);
        if !ptr.is_null() {
            record(layout.size(), 0);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        record(0, layout.size());
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = System.realloc(ptr, layout, size);
        if result.is_null() {
            return result;
        }
        if result == ptr {
            record(
                size.saturating_sub(layout.size()),
                layout.size().saturating_sub(size),
            );
        } else {
            record(size, layout.size());
        }
        result
    }
}

struct Reset;
impl Drop for Reset {
    fn drop(&mut self) {
        HEAP.with(|heap| heap.set(None));
    }
}

/// Runs `f` and reports the heap it used on this thread.
pub(super) fn capture<T>(f: impl FnOnce() -> T) -> (T, Snapshot) {
    HEAP.with(|heap| {
        assert!(heap.get().is_none(), "nested diagnostic scope");
        heap.set(Some(Heap::default()));
    });
    let reset = Reset;
    let result = f();
    let heap = HEAP.with(|heap| heap.get().unwrap());
    drop(reset);
    (result, Snapshot { heap })
}

#[test]
fn observations_are_scoped_and_track_allocation_lifetimes() {
    let ((), s) = capture(|| {
        let a = super::memory::try_buffer(8192).unwrap();
        let b = super::memory::try_copy(&a).unwrap();
        std::hint::black_box(&b);
    });
    assert!(s.heap.allocations >= 2 && s.heap.peak >= 16384);
    assert_eq!(s.heap.live, 0);
    let ((), empty) = capture(|| {});
    assert_eq!(empty.heap.allocations, 0);
}

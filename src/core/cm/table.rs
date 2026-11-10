//! Original aligned nibble-context cache with deterministic priority replacement.
//! Provenance: cache-line buckets and nibble bit-history trees described in
//! Mahoney, Data Compression Explained and the public-domain ZPAQ specification.
//! Layout, four-way lookup and allocation are original Rust.

use super::state;
use crate::core::types::BatoError;
use std::{
    alloc::{alloc_zeroed, dealloc, Layout},
    ptr::NonNull,
};

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Slot {
    check: u8,
    histories: [u8; 15],
}
#[repr(C, align(64))]
struct Bucket {
    slots: [Slot; 4],
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Handle {
    bucket: usize,
    slot: usize,
    check: u8,
}
pub(super) struct Table {
    buckets: NonNull<Bucket>,
    count: usize,
    layout: Layout,
}
/// Buckets reserved per modeled byte before the geometry's cap: each byte
/// touches two nibble slots in every context, and fewer buckets made small
/// inputs evict useful statistics (measured: 8x fewer cost 0.5-0.7%).
const BUCKETS_PER_BYTE: usize = 4;
fn bucket_count(length: usize, cap: u8) -> usize {
    length
        .saturating_mul(BUCKETS_PER_BYTE)
        .max(4)
        .next_power_of_two()
        .min(1usize << cap.min(24))
}
pub(super) fn memory_required(length: usize, cap: u8) -> usize {
    bucket_count(length, cap) * std::mem::size_of::<Bucket>()
}
impl Table {
    pub(super) fn new(length: usize, cap: u8) -> Result<Self, BatoError> {
        let count = bucket_count(length, cap);
        let layout = Layout::array::<Bucket>(count)
            .map_err(|_| BatoError::AllocationFailed { bytes: usize::MAX })?;
        // SAFETY: layout has nonzero size, a valid power-of-two alignment and
        // checked array size. Every Bucket/Slot consists solely of byte arrays,
        // so all-zero bytes are valid. Drop releases this exact allocation.
        let pointer = unsafe { alloc_zeroed(layout) }.cast::<Bucket>();
        let buckets = NonNull::new(pointer).ok_or(BatoError::AllocationFailed {
            bytes: layout.size(),
        })?;
        Ok(Self {
            buckets,
            count,
            layout,
        })
    }
    #[inline]
    fn bucket(&self, index: usize) -> &Bucket {
        // SAFETY: callers derive index by masking with count-1 or from a handle
        // created by this table. The allocation contains count live Buckets.
        debug_assert!(index < self.count);
        unsafe { &*self.buckets.as_ptr().add(index) }
    }
    #[inline]
    fn bucket_mut(&mut self, index: usize) -> &mut Bucket {
        debug_assert!(index < self.count);
        // SAFETY: same bounds as bucket(); exclusive self prevents aliasing.
        unsafe { &mut *self.buckets.as_ptr().add(index) }
    }
    /// Hints the cache line for `key`; a no-op where no prefetch is available.
    #[inline]
    // Without x86-64 prefetch the key is legitimately unused.
    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
    pub(super) fn prefetch(&self, key: u64) {
        // SAFETY: the masked index names an allocated cache line; prefetch
        // neither reads values into the model nor writes memory.
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(
                self.buckets
                    .as_ptr()
                    .add(key as usize & (self.count - 1))
                    .cast(),
                std::arch::x86_64::_MM_HINT_T0,
            );
        }
    }
    #[inline]
    pub(super) fn find(&mut self, key: u64) -> Handle {
        let index = key as usize & (self.count - 1);
        let check = (key >> 32) as u8;
        let bucket = self.bucket_mut(index);
        let slot = if let Some(slot) = bucket.slots.iter().position(|s| s.check == check) {
            slot
        } else {
            let mut victim = 0;
            for i in 1..4 {
                if bucket.slots[i].histories[0] < bucket.slots[victim].histories[0] {
                    victim = i;
                }
            }
            bucket.slots[victim] = Slot {
                check,
                histories: [0; 15],
            };
            victim
        };
        Handle {
            bucket: index,
            slot,
            check,
        }
    }
    #[inline]
    pub(super) fn history(&self, handle: Handle, node: usize) -> u8 {
        let slot = &self.bucket(handle.bucket).slots[handle.slot];
        if slot.check == handle.check {
            slot.histories[node]
        } else {
            0
        }
    }
    #[inline]
    pub(super) fn observe(&mut self, handle: Handle, node: usize, bit: u8) {
        let slot = &mut self.bucket_mut(handle.bucket).slots[handle.slot];
        if slot.check == handle.check {
            slot.histories[node] = state::next(slot.histories[node], bit);
        }
    }
}
impl Drop for Table {
    fn drop(&mut self) {
        // SAFETY: pointer/layout came from one successful alloc_zeroed call;
        // Table owns that allocation and this is its only release.
        unsafe {
            dealloc(self.buckets.as_ptr().cast(), self.layout);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn alignment_size_and_actual_heap_equal_the_model_reservation() {
        assert_eq!(std::mem::size_of::<Slot>(), 16);
        assert_eq!(std::mem::size_of::<Bucket>(), 64);
        for length in [1, 17, 1024, 65536] {
            for cap in [0, 4, 12, 20] {
                let (table, observed) =
                    crate::core::diagnostics::capture(|| Table::new(length, cap).unwrap());
                assert_eq!(table.buckets.as_ptr() as usize & 63, 0);
                assert_eq!(table.layout.size(), memory_required(length, cap));
                assert_eq!(observed.heap.peak, memory_required(length, cap));
            }
        }
    }
    #[test]
    fn replacement_prefers_low_priority_and_stale_handles_cannot_update() {
        let mut table = Table::new(1, 0).unwrap();
        let mut handles = Vec::new();
        for check in 1..=4u64 {
            let h = table.find(check << 32);
            handles.push(h);
            for _ in 0..check {
                table.observe(h, 0, 0);
            }
        }
        let old = handles[0];
        let new = table.find(5u64 << 32);
        assert_eq!(new.slot, old.slot);
        assert_eq!(table.history(new, 0), 0);
        table.observe(old, 0, 1);
        assert_eq!(table.history(new, 0), 0);
        for h in handles.into_iter().skip(1) {
            assert_ne!(table.history(h, 0), 0);
        }
    }
}

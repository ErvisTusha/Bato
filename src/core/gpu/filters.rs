//! GPU filter limits, operation direction, and the stable archive-facing API.
//! Device discovery lives in `device.rs`; scheduling lives in `scheduler.rs`.
//! Backend-specific kernels remain in `cuda/filters.rs` and `wgpu_filters.rs`.

pub(crate) use super::{
    device::FilterPool,
    scheduler::WorkerFilters,
};

/// Bound a single transfer and its temporary device buffers.
pub(super) const MAX_BYTES: usize = 64 * 1024 * 1024;
/// Small inputs rarely repay device startup and host transfers.
pub(super) const AUTO_MIN_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy)]
pub(crate) enum Direction {
    Forward,
    Inverse,
}

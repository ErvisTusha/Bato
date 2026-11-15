//! Common reversible byte-filter operation description.
//! Every backend accepts the same transform IDs, lengths and direction.
//! Archive coding does not depend on the chosen device.

/// Bound a transfer and its temporary device buffers.
pub(in crate::core::gpu) const MAX_BYTES: usize = 64 * 1024 * 1024;
/// Small batches rarely repay GPU startup and host transfers.
pub(in crate::core::gpu) const AUTO_MIN_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy)]
pub(crate) enum Direction {
    Forward,
    Inverse,
}

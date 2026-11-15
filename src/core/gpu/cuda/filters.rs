//! CUDA implementations of the format's delta and byte-plane filters.
//! Provenance: original implementation of the byte operations in cm/filters.rs,
//! using parallel prefix sums for delta inversion.
use super::{Argument, Cuda, GpuError, Memory};
use crate::core::{
    memory::try_filled,
    types::{BatoError, Compute, ExecutionStats, TransformId},
};
use std::sync::{Mutex, TryLockError};

/// Bytes processed by a workgroup of the inverse delta kernel.
const DELTA_TILE_BYTES: usize = 4096;
/// Cap the device working set at two 64 MiB buffers plus tile carries.
pub(crate) const MAX_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy)]
pub(crate) enum Direction {
    Forward,
    Inverse,
}

/// Dispatches the selected filter and downloads the exact output bytes.
pub(crate) fn apply(
    cuda: &Cuda,
    input: &[u8],
    output: &mut [u8],
    (transform, direction): (TransformId, Direction),
) -> Result<u64, GpuError> {
    let length = input.len();
    if length == 0 || length > MAX_BYTES || output.len() != length {
        return Err(GpuError::Shape(format!("byte filter of length {length}")));
    }
    let kind = match (transform, direction) {
        (TransformId::Delta1, Direction::Inverse) => return inverse_delta(cuda, input, output, 1),
        (TransformId::Delta2, Direction::Inverse) => return inverse_delta(cuda, input, output, 2),
        (TransformId::Delta4, Direction::Inverse) => return inverse_delta(cuda, input, output, 4),
        (TransformId::Shuffle2, Direction::Inverse) => 7,
        (TransformId::Shuffle4, Direction::Inverse) => 8,
        (
            TransformId::Delta1
            | TransformId::Delta2
            | TransformId::Delta4
            | TransformId::Shuffle2
            | TransformId::Shuffle4,
            Direction::Forward,
        ) => transform as u32,
        _ => {
            return Err(GpuError::Shape(format!(
                "unsupported byte filter {transform:?}"
            )))
        }
    };
    let source = Memory::uploaded(cuda, input)?;
    let result = Memory::allocate::<u8>(cuda, length)?;
    cuda.launch(
        cuda.function(c"filter_bytes")?,
        length,
        &[
            source.argument(),
            result.argument(),
            Argument::Word(length as u32),
            Argument::Word(kind),
        ],
    )?;
    result.read(output)?;
    Ok(1)
}

fn inverse_delta(
    cuda: &Cuda,
    input: &[u8],
    output: &mut [u8],
    distance: u32,
) -> Result<u64, GpuError> {
    let length = input.len();
    let count = length.div_ceil(DELTA_TILE_BYTES);
    let source = Memory::uploaded(cuda, input)?;
    let result = Memory::allocate::<u8>(cuda, length)?;
    let tiles = Memory::allocate::<u32>(cuda, count)?;
    let size = Argument::Word(length as u32);
    let distance = Argument::Word(distance);
    cuda.launch(
        cuda.function(c"filter_delta_local")?,
        count * super::BLOCK_THREADS,
        &[
            source.argument(),
            result.argument(),
            tiles.argument(),
            size,
            distance,
        ],
    )?;
    cuda.launch(
        cuda.function(c"filter_delta_carries")?,
        1,
        &[tiles.argument(), Argument::Word(count as u32)],
    )?;
    cuda.launch(
        cuda.function(c"filter_delta_finish")?,
        length,
        &[result.argument(), tiles.argument(), size, distance],
    )?;
    result.read(output)?;
    Ok(3)
}

/// CUDA driver startup and host transfers need a complete large batch to amortize.
const AUTO_MIN_BYTES: usize = 8 * 1024 * 1024;

/// Worker-local CUDA session. The pool shares a single bounded transfer slot
/// so other workers can continue on the CPU while one uses the device.
pub(crate) struct WorkerFilters<'a> {
    compute: Compute,
    slot: &'a Mutex<bool>,
    stats: &'a ExecutionStats,
    cuda: Option<Cuda>,
}

impl<'a> WorkerFilters<'a> {
    pub(crate) fn new(compute: Compute, slot: &'a Mutex<bool>, stats: &'a ExecutionStats) -> Self {
        Self {
            compute,
            slot,
            stats,
            cuda: None,
        }
    }

    /// Attempts a CUDA byte filter. Returning no output leaves the input
    /// unchanged so the caller can run the CPU reference.
    pub(crate) fn run(
        &mut self,
        input: &[u8],
        transform: TransformId,
        direction: Direction,
    ) -> Result<Option<Vec<u8>>, BatoError> {
        let min_bytes = match self.compute {
            Compute::Cpu => return Ok(None),
            Compute::Auto => AUTO_MIN_BYTES,
            Compute::Cuda => 1,
        };
        if !(min_bytes..=MAX_BYTES).contains(&input.len()) {
            return Ok(None);
        }
        let mut unavailable = match self.compute {
            Compute::Auto => match self.slot.try_lock() {
                Err(TryLockError::WouldBlock) => return Ok(None),
                result => result.map_err(|_| BatoError::Cuda("transfer slot poisoned".into()))?,
            },
            _ => self
                .slot
                .lock()
                .map_err(|_| BatoError::Cuda("transfer slot poisoned".into()))?,
        };
        if *unavailable {
            return Ok(None);
        }
        let mut output = match try_filled(input.len(), 0u8) {
            Ok(output) => output,
            Err(error) if self.compute == Compute::Auto => {
                self.stats
                    .update(|stats| stats.cuda_fallback = error.to_string());
                return Ok(None);
            }
            Err(error) => return Err(error),
        };
        let result = self.apply(input, &mut output, (transform, direction));
        match result {
            Ok(dispatches) => {
                self.stats.update(|stats| {
                    stats.cuda_filter_dispatches += dispatches;
                    stats.cuda_filter_bytes += input.len() as u64;
                });
                Ok(Some(output))
            }
            Err(error) if self.compute == Compute::Auto => {
                *unavailable = true;
                self.stats
                    .update(|stats| stats.cuda_fallback = error.to_string());
                Ok(None)
            }
            Err(error) => Err(BatoError::Cuda(error.to_string())),
        }
    }

    fn apply(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        operation: (TransformId, Direction),
    ) -> Result<u64, GpuError> {
        if self.cuda.is_none() {
            let cuda = Cuda::open_for_filters()?;
            self.stats
                .update(|stats| stats.cuda_device = cuda.description());
            self.cuda = Some(cuda);
        }
        let cuda = self
            .cuda
            .as_ref()
            .ok_or_else(|| GpuError::Execution("CUDA session unavailable".into()))?;
        apply(cuda, input, output, operation)
    }
}

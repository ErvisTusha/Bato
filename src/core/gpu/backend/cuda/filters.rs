//! CUDA implementations of the format's delta and byte-plane filters.
//! Provenance: original implementation of the byte operations in cm/filters.rs,
//! using parallel prefix sums for delta inversion.
use super::{Argument, Cuda, Memory};
use super::super::super::{filters::{Direction, MAX_BYTES}, GpuError};
use crate::core::types::TransformId;

/// Bytes processed by a workgroup of the inverse delta kernel.
const DELTA_TILE_BYTES: usize = 4096;
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

//! Bato: a pure-Rust context-mixing compressor with bounded parallel execution.
pub mod core;
pub use crate::core::archive::{
    create_solid_archive, create_solid_archive_excluding, extract_solid_archive,
    extract_solid_archive_with_execution, extract_solid_archive_with_options, ArchiveFileEntry,
};
pub use crate::core::frame::{
    compress_stream, decompress_stream, decompress_stream_with_limits,
    decompress_stream_with_options, read_block_header, read_geometry, read_stream_header,
    read_stream_layout, DecompressReader, StreamLayout,
};
pub use crate::core::types::{
    BatoError, BlockHeader, BlockType, CompressionOptions, DecodeLimits, DecompressionOptions,
    ExecutionOptions, ExecutionStats, ExecutionStatsSnapshot, Geometry, MemoryBudget, Overwrite,
    StreamHeader, StreamKind, TransformId, DEFAULT_CHUNK_SIZE, DEFAULT_MEMORY_BUDGET,
    DEFAULT_SEGMENT_SIZE, MAX_CHUNK_SIZE, VERSION_MAJOR, VERSION_MINOR,
};
pub use crate::core::TimingReport;
use std::io::Cursor;
pub fn compress(data: &[u8]) -> Result<Vec<u8>, BatoError> {
    compress_with_options(data, &CompressionOptions::default())
}
pub fn compress_with_options(
    data: &[u8],
    options: &CompressionOptions,
) -> Result<Vec<u8>, BatoError> {
    let host = core::memory::HostBudget::new(options.memory_budget, &options.execution.stats)?;
    let mut out = core::memory::ManagedVec::new(&host)?;
    core::frame::compress_with_budget(&mut Cursor::new(data), &mut out, options, host)?;
    Ok(out.into_vec())
}
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, BatoError> {
    decompress_with_options(data, &DecompressionOptions::default())
}
pub fn decompress_with_options(
    data: &[u8],
    options: &DecompressionOptions,
) -> Result<Vec<u8>, BatoError> {
    let host =
        core::memory::HostBudget::new(options.limits.max_working_memory, &options.execution.stats)?;
    let mut out = core::memory::ManagedVec::new(&host)?;
    core::frame::decompress_with_budget(&mut Cursor::new(data), &mut out, options, host)?;
    Ok(out.into_vec())
}
pub fn decompress_with_limit(data: &[u8], max_output: usize) -> Result<Vec<u8>, BatoError> {
    decompress_with_limits(
        data,
        &DecodeLimits {
            max_output_bytes: max_output as u64,
            ..Default::default()
        },
    )
}
pub fn decompress_with_limits(data: &[u8], limits: &DecodeLimits) -> Result<Vec<u8>, BatoError> {
    decompress_with_options(
        data,
        &DecompressionOptions {
            limits: limits.clone(),
            ..Default::default()
        },
    )
}

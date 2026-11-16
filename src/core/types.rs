//! Bato format constants and resource limits.
use super::cm::profile::{Timing, TimingReport, TimingSink};
pub use super::memory::MemoryBudget;
use std::{fmt, io};
pub const MAGIC: [u8; 4] = *b"BATO";
/// Format version. Any change to the container or the model that alters
/// encoded bytes increments it; readers accept exactly this version.
pub const VERSION_MAJOR: u8 = 1;
pub const VERSION_MINOR: u8 = 22;
/// Largest chunk the format allows (chunk log2 <= 20).
pub const MAX_CHUNK_SIZE: usize = 1024 * 1024;
pub const DEFAULT_SEGMENT_SIZE: usize = 64 * 1024 * 1024;
pub const DEFAULT_CHUNK_SIZE: usize = 256 * 1024;
/// Smallest automatic segment, in bytes. Below two of these an input stays one
/// stream. Small segments cost size: on 508 KB of text, 2 streams cost +4.1%
/// and 4 cost +8.0%; the owner accepts that so all cores are used.
pub const MIN_AUTO_SEGMENT: usize = 64 * 1024;
pub const DEFAULT_TABLE_LOG2: u8 = 22;
/// Largest number of worker threads a caller may request.
pub const MAX_THREADS: usize = 256;
/// A fixed reference allowance for reproducible callers and benchmarks.
/// Compression and decompression options default to `MemoryBudget::Auto`.
pub const DEFAULT_MEMORY_BUDGET: usize = 256 * 1024 * 1024;
/// The only header flag: the stream holds a directory archive catalogue.
pub const FLAG_DIRECTORY_ARCHIVE: u8 = 1;
#[derive(Debug)]
pub enum BatoError {
    InvalidMagic([u8; 4]),
    UnsupportedVersion { major: u8, minor: u8 },
    TruncatedHeader,
    TruncatedPayload { expected: usize, available: usize },
    ChecksumMismatch { expected: u32, computed: u32 },
    StreamChecksumMismatch { expected: u32, computed: u32 },
    DecompressionLimitExceeded { size: usize, limit: usize },
    MemoryLimitExceeded { required: usize, limit: usize },
    AllocationFailed { bytes: usize },
    InvalidOptions(String),
    Cuda(String),
    Wgpu(String),
    Native(String),
    CorruptStream(String),
    UnknownBlockType(u8),
    UnknownTransformId(u8),
    Io(io::Error),
}
impl fmt::Display for BatoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic(m) => {
                write!(f, "Not a Bato stream (magic {m:02X?})")
            }
            Self::UnsupportedVersion { major, minor } => {
                write!(f, "Unsupported format version: {major}.{minor}")
            }
            Self::TruncatedHeader => write!(f, "Truncated header encountered"),
            Self::TruncatedPayload {
                expected,
                available,
            } => write!(
                f,
                "Truncated payload: expected {expected} bytes, got {available}"
            ),
            Self::ChecksumMismatch { expected, computed } => write!(
                f,
                "Block CRC32C mismatch: expected {expected:08X}, computed {computed:08X}"
            ),
            Self::StreamChecksumMismatch { expected, computed } => write!(
                f,
                "Stream CRC32C mismatch: expected {expected:08X}, computed {computed:08X}"
            ),
            Self::DecompressionLimitExceeded { size, limit } => write!(
                f,
                "Decompression size limit exceeded: {size} bytes > {limit} bytes"
            ),
            Self::MemoryLimitExceeded { required, limit } => write!(
                f,
                "Working memory limit exceeded: {required} bytes > {limit} bytes"
            ),
            Self::AllocationFailed { bytes } => write!(f, "Unable to allocate {bytes} bytes"),
            Self::InvalidOptions(s) => write!(f, "Invalid options: {s}"),
            Self::Cuda(reason) => write!(f, "CUDA byte filters: {reason}"),
            Self::Wgpu(reason) => write!(f, "wgpu byte filters: {reason}"),
            Self::Native(reason) => write!(f, "native GPU byte filters: {reason}"),
            Self::CorruptStream(s) => write!(f, "Corrupted stream: {s}"),
            Self::UnknownBlockType(v) => write!(f, "Unknown block type: {v}"),
            Self::UnknownTransformId(v) => write!(f, "Unknown transform ID: {v}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}
impl std::error::Error for BatoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}
impl From<io::Error> for BatoError {
    fn from(e: io::Error) -> Self {
        if e.get_ref().is_some_and(|inner| inner.is::<BatoError>()) {
            // Managed writers and decompression readers preserve typed
            // admission/format failures through std::io traits. This only
            // recovers our own error after the `is` check above; it does not
            // dispatch on arbitrary types.
            return *e.into_inner().unwrap().downcast::<BatoError>().unwrap();
        }
        Self::Io(e)
    }
}
#[derive(Debug, Clone, Default)]
pub struct ExecutionStatsSnapshot {
    pub cpu_workers: usize,
    pub cpu_blocks: u64,
    /// Successful CUDA kernel launches that produced archive filter output.
    pub cuda_filter_dispatches: u64,
    /// Input bytes processed by CUDA, counted once per filter.
    pub cuda_filter_bytes: u64,
    pub cuda_device: String,
    pub cuda_fallback: String,
    pub wgpu_filter_dispatches: u64,
    pub wgpu_filter_bytes: u64,
    /// Successfully dispatched native HIP, Level Zero or Metal filters.
    pub native_filter_dispatches: u64,
    pub native_filter_bytes: u64,
    pub native_device: String,
    /// Devices that successfully completed at least one archive filter.
    pub gpu_devices: Vec<String>,
    pub gpu_fallback: String,
    /// Filter operations completed on any GPU backend (one per offloaded filter).
    pub gpu_filter_calls: u64,
    /// Bytes copied host to device: one upload of the filter input per call.
    pub gpu_host_to_device_bytes: u64,
    /// Bytes copied device to host: one download of the same length per call.
    pub gpu_device_to_host_bytes: u64,
    /// Wall nanoseconds inside successful GPU filter calls: upload, kernels and
    /// download together. Backends do not report the split.
    pub gpu_call_nanos: u64,
    pub estimated_host_memory: usize,
    /// Startup allowance, shared by the entire operation.
    pub memory_budget: usize,
    /// Peak simultaneous conservative reservations, not process RSS.
    pub managed_host_peak: usize,
}
impl ExecutionStatsSnapshot {
    /// Records one successful GPU filter call of `bytes` in and out.
    pub(crate) fn record_gpu_call(&mut self, bytes: usize, elapsed: std::time::Duration) {
        self.gpu_filter_calls += 1;
        self.gpu_host_to_device_bytes += bytes as u64;
        self.gpu_device_to_host_bytes += bytes as u64;
        self.gpu_call_nanos = self
            .gpu_call_nanos
            .saturating_add(u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX));
    }
}
#[derive(Debug, Default)]
pub struct ExecutionStats {
    data: std::sync::Mutex<ExecutionStatsSnapshot>,
    timing: TimingSink,
}
impl ExecutionStats {
    /// Statistics that also time every step of the operation; see
    /// `timing_report`. Timing never changes the coded bytes.
    pub fn with_timing() -> Self {
        let stats = Self::default();
        stats.timing.enable();
        stats
    }
    /// Step times and model activity, if these statistics were created with
    /// `with_timing`.
    pub fn timing_report(&self) -> Option<TimingReport> {
        self.timing.report()
    }
    pub(crate) fn timing(&self) -> Timing<'_> {
        self.timing.handle()
    }
    pub fn snapshot(&self) -> ExecutionStatsSnapshot {
        self.data.lock().unwrap().clone()
    }
    pub(crate) fn update(&self, f: impl FnOnce(&mut ExecutionStatsSnapshot)) {
        f(&mut self.data.lock().unwrap())
    }
}
/// Where eligible byte filters run. Every setting produces identical archives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compute {
    /// Try every available compiled GPU backend for large batches, with CPU fallback.
    #[default]
    Auto,
    /// Run all filters on the CPU.
    Cpu,
    /// Require CUDA for supported batches. A device failure is returned.
    Cuda,
    /// Require a supported hardware adapter through wgpu for eligible filters.
    Wgpu,
    /// Require a native AMD HIP plugin (compiled with ROCm/HIP SDK).
    Hip,
    /// Require Intel oneAPI's native Level Zero SYCL backend plugin.
    LevelZero,
    /// Require a direct Apple Metal compute plugin.
    Metal,
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionOptions {
    /// Worker threads; 0 uses all available CPUs.
    pub threads: usize,
    pub compute: Compute,
    pub stats: std::sync::Arc<ExecutionStats>,
}
impl ExecutionOptions {
    /// Requested worker threads, with 0 meaning every available CPU.
    pub(crate) fn worker_threads(&self) -> usize {
        if self.threads == 0 {
            std::thread::available_parallelism()
                .map_or(1, usize::from)
                .min(MAX_THREADS)
        } else {
            self.threads
        }
    }
    pub fn validate(&self) -> Result<(), BatoError> {
        if self.threads > MAX_THREADS {
            return Err(BatoError::InvalidOptions(format!(
                "threads must be at most {MAX_THREADS}"
            )));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BlockType {
    /// Chunk stored uncompressed (coding did not make it smaller).
    Raw = 0,
    /// Chunk coded by the model.
    Coded = 1,
    /// Independent uncompressed segment, with no predictor state.
    Stored = 2,
    /// Independent literal/copy command stream.
    Lz = 3,
    /// Independent canonical static Huffman byte stream.
    Huffman = 4,
    /// Static Huffman table over an independent literal/copy command stream.
    LzHuffman = 5,
    /// Independent repetition of one byte.
    Repeat = 6,
    /// Independent contextual digit-run lifting with static entropy tables.
    Numeric = 7,
    EndOfStream = 255,
}
impl BlockType {
    pub(crate) fn independent(self) -> bool {
        matches!(
            self,
            Self::Stored
                | Self::Lz
                | Self::Huffman
                | Self::LzHuffman
                | Self::Repeat
                | Self::Numeric
        )
    }
    pub(crate) fn implicit_payload(self) -> bool {
        matches!(self, Self::Raw | Self::Stored | Self::Repeat)
    }
    pub fn from_u8(v: u8) -> Result<Self, BatoError> {
        match v {
            0 => Ok(Self::Raw),
            1 => Ok(Self::Coded),
            2 => Ok(Self::Stored),
            3 => Ok(Self::Lz),
            4 => Ok(Self::Huffman),
            5 => Ok(Self::LzHuffman),
            6 => Ok(Self::Repeat),
            7 => Ok(Self::Numeric),
            255 => Ok(Self::EndOfStream),
            _ => Err(BatoError::UnknownBlockType(v)),
        }
    }
}
/// Whether a CM segment's model first read the built-in primer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Primer {
    Read,
    Skipped,
}

/// Largest transformed length of a deflate-transformed segment, in segment
/// sizes. Inflated streams are larger than the compressed bytes they replace,
/// so a full segment needs room to grow; 4 covers the 4:1 streams of text,
/// code and executables that dominate real archives.
pub const DEFLATE_EXPANSION: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TransformId {
    None = 0,
    /// Segment whose x86 call/jump operands were made absolute before coding.
    X86Calls = 1,
    Delta1 = 2,
    Delta2 = 3,
    Delta4 = 4,
    Shuffle2 = 5,
    Shuffle4 = 6,
    /// Deflate streams inside the segment replaced by their inflated bytes
    /// and recipes; the chunks hold the transformed bytes.
    Deflate = 7,
    /// As `X86Calls`, with RIP-relative displacements made absolute too.
    X86Relative = 8,
    /// As `Deflate`, with the `X86Calls` rewrite applied to the transformed
    /// bytes.
    DeflateX86Calls = 9,
    /// As `Deflate`, with the `X86Relative` rewrite applied to the
    /// transformed bytes.
    DeflateX86Relative = 10,
    /// As `X86Calls`, reaching targets anywhere in the segment.
    X86CallsSegment = 11,
    /// As `X86Relative`, reaching targets anywhere in the segment.
    X86RelativeSegment = 12,
    /// As `DeflateX86Calls`, reaching targets anywhere in the container.
    DeflateX86CallsSegment = 13,
    /// As `DeflateX86Relative`, reaching targets anywhere in the container.
    DeflateX86RelativeSegment = 14,
}
impl TransformId {
    pub fn from_u8(v: u8) -> Result<Self, BatoError> {
        match v {
            0 => Ok(Self::None),
            1 => Ok(Self::X86Calls),
            2 => Ok(Self::Delta1),
            3 => Ok(Self::Delta2),
            4 => Ok(Self::Delta4),
            5 => Ok(Self::Shuffle2),
            6 => Ok(Self::Shuffle4),
            7 => Ok(Self::Deflate),
            8 => Ok(Self::X86Relative),
            9 => Ok(Self::DeflateX86Calls),
            10 => Ok(Self::DeflateX86Relative),
            11 => Ok(Self::X86CallsSegment),
            12 => Ok(Self::X86RelativeSegment),
            13 => Ok(Self::DeflateX86CallsSegment),
            14 => Ok(Self::DeflateX86RelativeSegment),
            _ => Err(BatoError::UnknownTransformId(v)),
        }
    }
    /// Whether the chunks hold a deflate container, whose length differs
    /// from the segment's.
    pub fn is_deflate_container(self) -> bool {
        matches!(
            self,
            Self::Deflate
                | Self::DeflateX86Calls
                | Self::DeflateX86Relative
                | Self::DeflateX86CallsSegment
                | Self::DeflateX86RelativeSegment
        )
    }
}
/// What a stream's decoded bytes are, recorded in header flag bit 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamKind {
    /// The bytes of one file or input stream.
    Bytes,
    /// A serialized directory tree (see `docs/format.md`).
    DirectoryArchive,
}
impl StreamKind {
    pub(crate) fn flags(self) -> u8 {
        match self {
            Self::Bytes => 0,
            Self::DirectoryArchive => FLAG_DIRECTORY_ARCHIVE,
        }
    }
    pub(crate) fn from_flags(flags: u8) -> Result<Self, BatoError> {
        match flags {
            0 => Ok(Self::Bytes),
            FLAG_DIRECTORY_ARCHIVE => Ok(Self::DirectoryArchive),
            _ => Err(BatoError::CorruptStream("unsupported stream flags".into())),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamHeader {
    pub version_major: u8,
    pub version_minor: u8,
    pub kind: StreamKind,
}
/// Whether publishing output may replace an existing regular file.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Overwrite {
    /// Fail when the destination exists.
    #[default]
    Refuse,
    /// Replace an existing regular file; other entry types are still refused.
    Replace,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockHeader {
    pub block_type: BlockType,
    pub transform_id: TransformId,
    pub primer: Primer,
    pub uncompressed_size: u32,
    pub compressed_size: u32,
}
/// Segment geometry: the only encoder settings recorded in the stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub segment_size: usize,
    pub chunk_size: usize,
    pub table_log2: u8,
}
impl Default for Geometry {
    fn default() -> Self {
        Self {
            segment_size: DEFAULT_SEGMENT_SIZE,
            chunk_size: DEFAULT_CHUNK_SIZE,
            table_log2: DEFAULT_TABLE_LOG2,
        }
    }
}
impl Geometry {
    /// The geometry to code an input of `length` bytes with, spread over
    /// `workers` threads. A segment size left at `DEFAULT_SEGMENT_SIZE` is
    /// chosen from the length: inputs below two minimum segments stay one
    /// stream; anything larger is cut into about one independent stream per
    /// worker (a power-of-two segment between `MIN_AUTO_SEGMENT` and the
    /// default), so that every worker codes and decodes at once. The archive
    /// therefore depends on the thread count used to make it. Measured on a
    /// 7.5 MB executable (1 stream = 1,691,899 bytes): 2 streams cost +2.0%
    /// and decode 1.8x faster, 4 cost +3.6% and 3.0x, 8 cost +6.2% and 3.3x.
    /// Other segment sizes are kept.
    pub fn for_input(self, length: u64, workers: usize) -> Geometry {
        if self.segment_size != DEFAULT_SEGMENT_SIZE || length < 2 * MIN_AUTO_SEGMENT as u64 {
            return self;
        }
        let share = length.div_ceil(workers.max(1) as u64);
        let segment = usize::try_from(share.next_power_of_two()).unwrap_or(DEFAULT_SEGMENT_SIZE);
        let segment_size = segment.clamp(MIN_AUTO_SEGMENT, DEFAULT_SEGMENT_SIZE);
        Geometry {
            segment_size,
            // A chunk never exceeds its segment.
            chunk_size: self.chunk_size.min(segment_size),
            ..self
        }
    }
    pub fn validate(&self) -> Result<(), BatoError> {
        self.descriptor().map(|_| ())
    }
    pub(crate) fn descriptor(self) -> Result<super::cm::descriptor::Descriptor, BatoError> {
        if !self.segment_size.is_power_of_two() || !self.chunk_size.is_power_of_two() {
            return Err(BatoError::InvalidOptions(
                "segment and chunk sizes must be powers of two".into(),
            ));
        }
        let descriptor = super::cm::descriptor::Descriptor {
            segment_log2: self.segment_size.trailing_zeros() as u8,
            chunk_log2: self.chunk_size.trailing_zeros() as u8,
            table_log2: self.table_log2,
        };
        descriptor.validate()?;
        Ok(descriptor)
    }
    pub(crate) fn from_descriptor(descriptor: super::cm::descriptor::Descriptor) -> Self {
        Self {
            segment_size: descriptor.segment_bytes(),
            chunk_size: descriptor.chunk_bytes(),
            table_log2: descriptor.table_log2,
        }
    }
}
#[derive(Debug, Clone, Default)]
pub struct CompressionOptions {
    pub geometry: Geometry,
    /// Length of the input in bytes when the caller knows it. It lets the
    /// encoder pick the segment size (see `Geometry::for_input`); without it
    /// the geometry is used as given.
    pub input_length: Option<u64>,
    pub memory_budget: MemoryBudget,
    pub execution: ExecutionOptions,
}
impl CompressionOptions {
    /// These options with the geometry the known input length calls for.
    pub(crate) fn resolved(&self) -> CompressionOptions {
        let workers = self.execution.worker_threads();
        CompressionOptions {
            geometry: self.input_length.map_or(self.geometry, |length| {
                self.geometry.for_input(length, workers)
            }),
            ..self.clone()
        }
    }
    pub fn validate(&self) -> Result<(), BatoError> {
        self.execution.validate()?;
        self.geometry.validate()?;
        if self.memory_budget == MemoryBudget::Bytes(0) {
            return Err(BatoError::InvalidOptions(
                "memory budget must be positive".into(),
            ));
        }
        Ok(())
    }
}
/// Most entries a directory archive may hold, when created or extracted.
pub const MAX_ARCHIVE_ENTRIES: usize = 100_000;
/// Longest archive path, in UTF-8 bytes.
pub const MAX_PATH_BYTES: usize = 4096;
#[derive(Debug, Clone)]
pub struct DecodeLimits {
    /// Maximum restored bytes. Defaults to the full format length range.
    pub max_output_bytes: u64,
    pub max_working_memory: MemoryBudget,
    pub max_archive_entries: usize,
    pub max_path_bytes: usize,
    pub max_metadata_bytes: usize,
}
impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_output_bytes: u64::MAX,
            max_working_memory: MemoryBudget::Auto,
            max_archive_entries: MAX_ARCHIVE_ENTRIES,
            max_path_bytes: MAX_PATH_BYTES,
            max_metadata_bytes: 32 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DecompressionOptions {
    pub limits: DecodeLimits,
    pub execution: ExecutionOptions,
}

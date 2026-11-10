//! Strict versioned Bato framing with ordered, bounded parallel execution.
use super::cm::{
    descriptor::Descriptor,
    profile::Part,
    segment::{Chunk, Coding, Direction},
    varint,
};
use super::memory::{try_buffer, try_capacity, HostBudget, MemoryLease};
use super::{
    checksum::crc32c_combine,
    codec::{self, Output, Pool, Work},
    types::*,
};
use std::collections::VecDeque;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::sync::Arc;
fn exact<R: Read>(r: &mut R, b: &mut [u8]) -> Result<(), BatoError> {
    r.read_exact(b).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            BatoError::TruncatedHeader
        } else {
            e.into()
        }
    })
}
/// Stream header: magic (4), version (2), flags (1), segment geometry (3).
pub(super) const HEADER_BYTES: u64 = 7 + Descriptor::BYTES as u64;
/// CRC32C follows each chunk or independent segment.
const BLOCK_CRC_BYTES: u64 = 4;
/// The final CRC follows a canonical base-128 total length.
const FOOTER_CRC_BYTES: usize = 4;

pub(crate) fn block_cost(kind: BlockType, length: usize, payload: usize) -> usize {
    if kind == BlockType::EndOfStream {
        return 1;
    }
    1 + varint::len(length as u64)
        + if kind.implicit_payload() {
            0
        } else {
            varint::len(payload as u64)
        }
        + payload
        + BLOCK_CRC_BYTES as usize
}
fn header_len(h: &BlockHeader) -> usize {
    block_cost(
        h.block_type,
        h.uncompressed_size as usize,
        h.compressed_size as usize,
    )
    .saturating_sub(
        h.compressed_size as usize
            + if h.block_type == BlockType::EndOfStream {
                0
            } else {
                BLOCK_CRC_BYTES as usize
            },
    )
}

pub fn write_stream_header<W: Write>(w: &mut W, o: &CompressionOptions) -> Result<(), BatoError> {
    write_header(w, o, StreamKind::Bytes)
}
fn write_header<W: Write>(
    w: &mut W,
    o: &CompressionOptions,
    kind: StreamKind,
) -> Result<(), BatoError> {
    o.validate()?;
    w.write_all(&MAGIC)?;
    w.write_all(&[VERSION_MAJOR, VERSION_MINOR, kind.flags()])?;
    w.write_all(&o.geometry.descriptor()?.encode()?)?;
    Ok(())
}
/// Reads the 7-byte stream header (magic, version, flags). The 3-byte
/// segment geometry follows; read it with `read_geometry`.
pub fn read_stream_header<R: Read>(r: &mut R) -> Result<StreamHeader, BatoError> {
    let mut b = [0; 7];
    exact(r, &mut b)?;
    let magic = b[..4].try_into().unwrap();
    if magic != MAGIC {
        return Err(BatoError::InvalidMagic(magic));
    }
    if b[4] != VERSION_MAJOR || b[5] != VERSION_MINOR {
        return Err(BatoError::UnsupportedVersion {
            major: b[4],
            minor: b[5],
        });
    }
    Ok(StreamHeader {
        version_major: b[4],
        version_minor: b[5],
        kind: StreamKind::from_flags(b[6])?,
    })
}
/// Read the segment geometry immediately following the stream header.
pub fn read_geometry<R: Read>(r: &mut R) -> Result<Geometry, BatoError> {
    let mut bytes = [0; Descriptor::BYTES];
    exact(r, &mut bytes)?;
    Ok(Geometry::from_descriptor(Descriptor::decode(bytes)?))
}
fn write_block<W: Write>(
    w: &mut W,
    h: &BlockHeader,
    p: &[u8],
    crc: Option<u32>,
) -> Result<(), BatoError> {
    if h.block_type == BlockType::EndOfStream {
        w.write_all(&[255])?;
        return Ok(());
    }
    let skipped = u8::from(h.primer == Primer::Skipped) << 6;
    let transform = h.transform_id as u8;
    w.write_all(&[(h.block_type as u8) | (transform & 7) << 3 | skipped | (transform & 8) << 4])?;
    varint::write(w, u64::from(h.uncompressed_size))?;
    if !h.block_type.implicit_payload() {
        varint::write(w, u64::from(h.compressed_size))?;
    }
    w.write_all(p)?;
    if let Some(c) = crc {
        w.write_all(&c.to_le_bytes())?;
    }
    Ok(())
}
pub fn read_block_header<R: Read>(r: &mut R) -> Result<BlockHeader, BatoError> {
    let mut byte = [0];
    exact(r, &mut byte)?;
    if byte[0] == 255 {
        return Ok(BlockHeader {
            block_type: BlockType::EndOfStream,
            transform_id: TransformId::None,
            primer: Primer::Read,
            uncompressed_size: 0,
            compressed_size: 0,
        });
    }
    let block_type = BlockType::from_u8(byte[0] & 7)?;
    let transform_id = TransformId::from_u8((byte[0] >> 3) & 7 | (byte[0] >> 4) & 8)?;
    let primer = if byte[0] & 0x40 != 0 {
        Primer::Skipped
    } else {
        Primer::Read
    };
    if primer == Primer::Skipped && block_type.independent() {
        return Err(BatoError::CorruptStream(
            "independent block declares a primer choice".into(),
        ));
    }
    let n = varint::read(r)?;
    let c = if block_type == BlockType::Repeat {
        1
    } else if block_type.implicit_payload() {
        n
    } else {
        varint::read(r)?
    };
    let limit = if block_type.independent() {
        1u64 << 30
    } else {
        MAX_CHUNK_SIZE as u64
    };
    if n == 0
        || n > limit
        || c > n
        || (!block_type.implicit_payload() && c >= n)
        || (block_type == BlockType::Repeat && n < 2)
    {
        return Err(BatoError::CorruptStream("invalid block sizes".into()));
    }
    Ok(BlockHeader {
        block_type,
        transform_id,
        primer,
        uncompressed_size: n as u32,
        compressed_size: c as u32,
    })
}
/// A stream's framing, read without decoding or verifying any payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamLayout {
    pub header: StreamHeader,
    pub geometry: Geometry,
    /// Every block header in stream order, ending with the end-of-stream block.
    pub blocks: Vec<BlockHeader>,
}

/// Reads the header, geometry and block headers of a stream, skipping block
/// payloads. Checksums and coded data are not checked; decode for that.
pub fn read_stream_layout<R: Read>(r: &mut R) -> Result<StreamLayout, BatoError> {
    let header = read_stream_header(r)?;
    let geometry = read_geometry(r)?;
    let mut blocks = Vec::new();
    loop {
        let block = read_block_header(r)?;
        if block.block_type == BlockType::EndOfStream {
            blocks.push(block);
            return Ok(StreamLayout {
                header,
                geometry,
                blocks,
            });
        }
        let skip = u64::from(block.compressed_size) + BLOCK_CRC_BYTES;
        let available = io::copy(&mut r.by_ref().take(skip), &mut io::sink())?;
        if available != skip {
            return Err(BatoError::TruncatedPayload {
                expected: skip as usize,
                available: available as usize,
            });
        }
        blocks.push(block);
    }
}

/// Workers the stream would admit, checked before any input is read.
pub fn admitted_workers(o: &CompressionOptions) -> Result<usize, BatoError> {
    o.validate()?;
    Ok(codec::admit(
        (o.geometry.descriptor()?, Direction::Encode),
        o.memory_budget.resolve()?,
        &o.execution,
    )?
    .workers)
}
/// Largest single read request when filling a segment.
const READ_SIZE: usize = 64 * 1024;
fn read_segment<R: Read>(r: &mut R, descriptor: Descriptor) -> Result<Vec<u8>, BatoError> {
    let mut data = Vec::new();
    let mut at = 0;
    while at < descriptor.segment_bytes() {
        if at == data.len() {
            let next = data
                .len()
                .saturating_mul(2)
                .max(READ_SIZE)
                .min(descriptor.segment_bytes());
            data.try_reserve_exact(next - data.len())
                .map_err(|_| BatoError::AllocationFailed { bytes: next })?;
            data.resize(next, 0);
        }
        let end = data.len().min(at + READ_SIZE);
        match r.read(&mut data[at..end]) {
            Ok(0) => break,
            Ok(read) => at += read,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.into()),
        }
    }
    data.truncate(at);
    Ok(data)
}

pub fn compress_stream<R: Read, W: Write>(
    r: &mut R,
    w: &mut W,
    o: &CompressionOptions,
) -> Result<(usize, usize), BatoError> {
    let host = HostBudget::new(o.memory_budget, &o.execution.stats)?;
    compress_kind(r, w, o, Destination::new(StreamKind::Bytes, host))
}
pub(crate) fn compress_with_budget<R: Read, W: Write>(
    r: &mut R,
    w: &mut W,
    o: &CompressionOptions,
    host: Arc<HostBudget>,
) -> Result<(usize, usize), BatoError> {
    compress_kind(r, w, o, Destination::new(StreamKind::Bytes, host))
}
pub(super) fn compress_archive_stream<R: Read, W: Write>(
    r: &mut R,
    w: &mut W,
    o: &CompressionOptions,
    host: Arc<HostBudget>,
) -> Result<(usize, usize), BatoError> {
    compress_kind(
        r,
        w,
        o,
        Destination::new(StreamKind::DirectoryArchive, host),
    )
}
/// State of the last complete segment in the output stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegmentCheckpoint {
    pub decoded: u64,
    pub encoded: u64,
    pub crc32c: u32,
}

pub fn compress_checkpointed<R: Read + Seek, W: Write>(
    input: &mut R,
    output: &mut W,
    options: &CompressionOptions,
    kind: StreamKind,
    progress: EncodeProgress<'_, W>,
) -> Result<(usize, usize), BatoError> {
    if let Some(state) = progress.0 {
        input.seek(SeekFrom::Start(state.decoded))?;
    }
    let host = HostBudget::new(options.memory_budget, &options.execution.stats)?;
    encode_segments(
        input,
        output,
        options,
        Destination::new(kind, host),
        progress,
    )
}

/// What an encoded stream holds and the host budget its workers draw from.
struct Destination {
    kind: StreamKind,
    host: Arc<HostBudget>,
}
impl Destination {
    fn new(kind: StreamKind, host: Arc<HostBudget>) -> Self {
        Self { kind, host }
    }
}
fn compress_kind<R: Read, W: Write>(
    r: &mut R,
    w: &mut W,
    o: &CompressionOptions,
    destination: Destination,
) -> Result<(usize, usize), BatoError> {
    encode_segments(r, w, o, destination, (None, &mut |_, _| Ok(())))
}

type EncodeProgress<'a, W> = (
    Option<SegmentCheckpoint>,
    &'a mut dyn FnMut(&mut W, SegmentCheckpoint) -> Result<(), BatoError>,
);

fn encode_segments<R: Read, W: Write>(
    r: &mut R,
    w: &mut W,
    o: &CompressionOptions,
    destination: Destination,
    (resume, checkpoint): EncodeProgress<'_, W>,
) -> Result<(usize, usize), BatoError> {
    o.validate()?;
    let descriptor = o.geometry.descriptor()?;
    let mut pool = Pool::new(
        (descriptor, Direction::Encode),
        destination.host,
        &o.execution,
    )?;
    let (mut total, mut compressed, mut crc) = if let Some(resume) = resume {
        (resume.decoded, resume.encoded, resume.crc32c)
    } else {
        write_header(w, o, destination.kind)?;
        o.execution
            .stats
            .timing()
            .part(Part::StreamHeader, HEADER_BYTES as usize);
        (0, HEADER_BYTES, 0)
    };
    let mut ended = false;
    loop {
        while !ended && pool.outstanding() < pool.capacity() {
            let lease = pool.prepare_job()?;
            let raw = read_segment(r, descriptor)?;
            if raw.is_empty() {
                ended = true;
                break;
            }
            let trials = pool.trials();
            pool.submit(
                Work::Encode {
                    raw,
                    descriptor,
                    trials,
                },
                lease,
            )?;
        }
        if pool.outstanding() == 0 {
            break;
        }
        let result = pool.receive()?;
        let Output::Encoded(chunks, original) = result.output else {
            return Err(BatoError::CorruptStream(
                "unexpected worker output during compression".into(),
            ));
        };
        total = total
            .checked_add(original.length)
            .ok_or_else(|| BatoError::CorruptStream("size overflow".into()))?;
        crc = crc32c_combine(crc, original.crc, original.length);
        for result in chunks {
            let cost = block_cost(
                result.header.block_type,
                result.header.uncompressed_size as usize,
                result.bytes.len(),
            );
            compressed += cost as u64;
            let timing = o.execution.stats.timing();
            let payload = if result.header.block_type == BlockType::Coded {
                Part::Coded
            } else {
                Part::Stored
            };
            timing.part(payload, result.bytes.len());
            timing.part(Part::BlockChecksums, BLOCK_CRC_BYTES as usize);
            timing.part(
                Part::BlockHeaders,
                cost - result.bytes.len() - BLOCK_CRC_BYTES as usize,
            );
            write_block(w, &result.header, &result.bytes, Some(result.crc))?;
        }
        checkpoint(
            w,
            SegmentCheckpoint {
                decoded: total,
                encoded: compressed,
                crc32c: crc,
            },
        )?;
    }
    write_block(
        w,
        &BlockHeader {
            block_type: BlockType::EndOfStream,
            transform_id: TransformId::None,
            primer: Primer::Read,
            uncompressed_size: 0,
            compressed_size: 0,
        },
        &[],
        None,
    )?;
    varint::write(w, total)?;
    w.write_all(&crc.to_le_bytes())?;
    w.flush()?;
    compressed += (1 + varint::len(total) + FOOTER_CRC_BYTES) as u64;
    o.execution
        .stats
        .timing()
        .part(Part::Footer, 1 + varint::len(total) + FOOTER_CRC_BYTES);
    Ok((
        usize::try_from(total)
            .map_err(|_| BatoError::CorruptStream("output counter overflow".into()))?,
        usize::try_from(compressed)
            .map_err(|_| BatoError::CorruptStream("input counter overflow".into()))?,
    ))
}
/// A segment is verified before its bytes are exposed. The reader stays on its
/// calling thread; worker jobs own their payloads and never borrow the reader.
pub struct DecompressReader<R: Read> {
    reader: R,
    kind: StreamKind,
    descriptor: Descriptor,
    short_final_chunk: bool,
    limits: DecodeLimits,
    pool: Pool,
    block: Vec<u8>,
    block_memory: Option<MemoryLease>,
    pos: usize,
    total: u64,
    scheduled: u64,
    compressed: u64,
    crc: u32,
    footer: Option<(u64, u32)>,
    segment_ends: Option<VecDeque<u64>>,
    verified_end: u64,
    done: bool,
    failed: bool,
}
impl<R: Read> DecompressReader<R> {
    pub fn new(reader: R, limits: &DecodeLimits) -> Result<Self, BatoError> {
        Self::new_with_options(
            reader,
            &DecompressionOptions {
                limits: limits.clone(),
                ..Default::default()
            },
        )
    }
    pub fn new_with_options(reader: R, o: &DecompressionOptions) -> Result<Self, BatoError> {
        let host = HostBudget::new(o.limits.max_working_memory, &o.execution.stats)?;
        Self::with_budget(reader, o, host)
    }
    pub(crate) fn with_budget(
        mut reader: R,
        o: &DecompressionOptions,
        host: Arc<HostBudget>,
    ) -> Result<Self, BatoError> {
        let header = read_stream_header(&mut reader)?;
        let descriptor = read_geometry(&mut reader)?.descriptor()?;
        let pool = Pool::new((descriptor, Direction::Decode), host, &o.execution)?;
        Ok(Self {
            reader,
            kind: header.kind,
            descriptor,
            short_final_chunk: false,
            limits: o.limits.clone(),
            pool,
            block: Vec::new(),
            block_memory: None,
            pos: 0,
            total: 0,
            scheduled: 0,
            compressed: HEADER_BYTES,
            crc: 0,
            footer: None,
            segment_ends: None,
            verified_end: HEADER_BYTES,
            done: false,
            failed: false,
        })
    }
    pub fn stats(&self) -> (u64, u64) {
        (self.total, self.compressed)
    }
    pub fn kind(&self) -> StreamKind {
        self.kind
    }
    fn read_footer(&mut self) -> Result<(), BatoError> {
        let total = varint::read(&mut self.reader)?;
        let mut f = [0; FOOTER_CRC_BYTES];
        exact(&mut self.reader, &mut f)?;
        self.compressed += (varint::len(total) + FOOTER_CRC_BYTES) as u64;
        self.footer = Some((total, u32::from_le_bytes(f)));
        let mut extra = [0];
        loop {
            match self.reader.read(&mut extra) {
                Ok(0) => return Ok(()),
                Ok(_) => {
                    return Err(BatoError::CorruptStream(
                        "trailing bytes after footer".into(),
                    ))
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
    }
    fn schedule(&mut self, length: u32) -> Result<(), BatoError> {
        let next = self
            .scheduled
            .checked_add(u64::from(length))
            .ok_or_else(|| BatoError::CorruptStream("output size overflow".into()))?;
        if next > self.limits.max_output_bytes {
            return Err(BatoError::DecompressionLimitExceeded {
                size: usize::try_from(next).unwrap_or(usize::MAX),
                limit: usize::try_from(self.limits.max_output_bytes).unwrap_or(usize::MAX),
            });
        }
        self.scheduled = next;
        Ok(())
    }
    fn prefetch(&mut self) -> Result<(), BatoError> {
        let descriptor = self.descriptor;
        while self.footer.is_none() && self.pool.outstanding() < self.pool.capacity() {
            let lease = self.pool.prepare_job()?;
            let mut chunks = Vec::new();
            let mut length = 0usize;
            let mut coding = None;
            let mut limit = descriptor.segment_bytes();
            while length < limit {
                let header = read_block_header(&mut self.reader)?;
                self.compressed += header_len(&header) as u64;
                if header.block_type == BlockType::EndOfStream {
                    self.read_footer()?;
                    break;
                }
                let n = header.uncompressed_size as usize;
                let deflate = header.transform_id.is_deflate_container();
                if deflate && length == 0 {
                    limit = descriptor.segment_bytes() * DEFLATE_EXPANSION;
                }
                if self.short_final_chunk
                    || (header.block_type.independent() && length != 0)
                    || (!header.block_type.independent() && n > descriptor.chunk_bytes())
                    || n > limit - length
                {
                    return Err(BatoError::CorruptStream(
                        "invalid chunk type, size, or segment boundary".into(),
                    ));
                }
                let declared = Coding {
                    transform: header.transform_id,
                    primer: header.primer,
                };
                if *coding.get_or_insert(declared) != declared {
                    return Err(BatoError::CorruptStream(
                        "chunks of one segment declare different transforms or primer use".into(),
                    ));
                }
                self.schedule(header.uncompressed_size)?;
                // Validate the size and output limit before allocating payloads
                // or metadata; model/output allocation happens in the worker.
                if chunks.capacity() == 0 {
                    chunks = try_capacity(descriptor.segment_bytes() / descriptor.chunk_bytes())?;
                }
                let mut payload = try_buffer(header.compressed_size as usize)?;
                self.reader.read_exact(&mut payload)?;
                let mut crc = [0; BLOCK_CRC_BYTES as usize];
                exact(&mut self.reader, &mut crc)?;
                self.compressed += payload.len() as u64 + BLOCK_CRC_BYTES;
                chunks.push(Chunk {
                    kind: header.block_type,
                    length: header.uncompressed_size,
                    payload,
                    crc: u32::from_le_bytes(crc),
                });
                length += n;
                let short = if header.block_type.independent() {
                    n != descriptor.segment_bytes()
                } else {
                    n != descriptor.chunk_bytes()
                };
                // A short chunk ends a deflate-transformed segment; any other
                // short chunk can only end the stream.
                self.short_final_chunk = short && !deflate;
                if header.block_type.independent() || (deflate && short) {
                    break;
                }
            }
            if !chunks.is_empty() {
                if let Some(ends) = &mut self.segment_ends {
                    let end = self.footer.map_or(self.compressed, |(total, _)| {
                        self.compressed - (1 + varint::len(total) + FOOTER_CRC_BYTES) as u64
                    });
                    ends.push_back(end);
                }
                self.pool.submit(
                    Work::Decode {
                        chunks,
                        descriptor,
                        coding: coding.unwrap_or(Coding {
                            transform: TransformId::None,
                            primer: Primer::Read,
                        }),
                    },
                    lease,
                )?;
            }
        }
        Ok(())
    }
    fn next(&mut self) -> Result<(), BatoError> {
        self.block = Vec::new();
        self.block_memory = None;
        self.pos = 0;
        self.prefetch()?;
        if self.pool.outstanding() > 0 {
            let result = self.pool.receive()?;
            let Output::Decoded { bytes, crc } = result.output else {
                return Err(BatoError::CorruptStream(
                    "unexpected worker output during decompression".into(),
                ));
            };
            let length = bytes.len() as u64;
            self.total += length;
            if let Some(ends) = &mut self.segment_ends {
                self.verified_end = ends
                    .pop_front()
                    .ok_or_else(|| BatoError::CorruptStream("missing segment boundary".into()))?;
            }
            self.crc = crc32c_combine(self.crc, crc, length);
            self.block = bytes;
            self.block_memory = result.memory;
            return Ok(());
        }
        let (total, crc) = self
            .footer
            .ok_or_else(|| BatoError::CorruptStream("missing footer".into()))?;
        if total != self.total {
            return Err(BatoError::CorruptStream("footer size mismatch".into()));
        }
        if crc != self.crc {
            return Err(BatoError::StreamChecksumMismatch {
                expected: crc,
                computed: self.crc,
            });
        }
        self.done = true;
        Ok(())
    }
    pub fn finish(&mut self) -> Result<(), BatoError> {
        if self.failed {
            return Err(BatoError::CorruptStream("reader previously failed".into()));
        }
        if self.pos != self.block.len() {
            return Err(BatoError::CorruptStream(
                "unconsumed decompressed bytes".into(),
            ));
        }
        if !self.done {
            if let Err(e) = self.next() {
                self.failed = true;
                return Err(e);
            }
            if !self.done {
                return Err(BatoError::CorruptStream(
                    "unconsumed decompressed segments".into(),
                ));
            }
        }
        Ok(())
    }
}
pub fn decompress_checkpointed<R: Read + Seek, W: Write>(
    input: &mut R,
    output: &mut W,
    options: &DecompressionOptions,
    (resume, checkpoint): EncodeProgress<'_, W>,
) -> Result<StreamKind, BatoError> {
    let mut decoder = DecompressReader::new_with_options(input, options)?;
    decoder.segment_ends = Some(VecDeque::new());
    if let Some(resume) = resume {
        if resume.encoded < HEADER_BYTES || resume.decoded > options.limits.max_output_bytes {
            return Err(BatoError::CorruptStream(
                "invalid segment checkpoint".into(),
            ));
        }
        decoder.reader.seek(SeekFrom::Start(resume.encoded))?;
        decoder.total = resume.decoded;
        decoder.scheduled = resume.decoded;
        decoder.compressed = resume.encoded;
        decoder.verified_end = resume.encoded;
        decoder.crc = resume.crc32c;
        decoder.short_final_chunk = resume.decoded % decoder.descriptor.segment_bytes() as u64 != 0;
    }
    loop {
        decoder.next()?;
        if decoder.done {
            return Ok(decoder.kind);
        }
        output.write_all(&decoder.block)?;
        checkpoint(
            output,
            SegmentCheckpoint {
                decoded: decoder.total,
                encoded: decoder.verified_end,
                crc32c: decoder.crc,
            },
        )?;
    }
}

impl<R: Read> Read for DecompressReader<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        if self.failed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "reader previously failed",
            ));
        }
        if self.pos == self.block.len() {
            if self.done {
                return Ok(0);
            }
            if let Err(e) = self.next() {
                self.failed = true;
                return Err(io::Error::new(io::ErrorKind::InvalidData, e));
            }
            if self.done {
                return Ok(0);
            }
        }
        let n = out.len().min(self.block.len() - self.pos);
        out[..n].copy_from_slice(&self.block[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }
}
pub(super) fn read_error(e: io::Error) -> BatoError {
    let kind = e.kind();
    if let Some(inner) = e.into_inner() {
        match inner.downcast::<BatoError>() {
            Ok(z) => *z,
            Err(e) => BatoError::Io(io::Error::new(kind, e)),
        }
    } else {
        BatoError::Io(io::Error::from(kind))
    }
}
pub fn decompress_stream_with_options<R: Read, W: Write>(
    r: &mut R,
    w: &mut W,
    o: &DecompressionOptions,
) -> Result<(usize, usize), BatoError> {
    let mut z = DecompressReader::new_with_options(r, o)?;
    drain_reader(&mut z, w)
}
pub(crate) fn decompress_with_budget<R: Read, W: Write>(
    r: &mut R,
    w: &mut W,
    o: &DecompressionOptions,
    host: Arc<HostBudget>,
) -> Result<(usize, usize), BatoError> {
    let mut z = DecompressReader::with_budget(r, o, host)?;
    drain_reader(&mut z, w)
}
fn drain_reader<R: Read, W: Write>(
    z: &mut DecompressReader<R>,
    w: &mut W,
) -> Result<(usize, usize), BatoError> {
    let mut b = [0; READ_SIZE];
    loop {
        let n = z.read(&mut b).map_err(read_error)?;
        if n == 0 {
            break;
        }
        w.write_all(&b[..n])?
    }
    z.finish()?;
    w.flush()?;
    let (a, b) = z.stats();
    Ok((
        usize::try_from(a)
            .map_err(|_| BatoError::CorruptStream("output counter overflow".into()))?,
        usize::try_from(b)
            .map_err(|_| BatoError::CorruptStream("input counter overflow".into()))?,
    ))
}
pub fn decompress_stream_with_limits<R: Read, W: Write>(
    r: &mut R,
    w: &mut W,
    l: &DecodeLimits,
) -> Result<(usize, usize), BatoError> {
    decompress_stream_with_options(
        r,
        w,
        &DecompressionOptions {
            limits: l.clone(),
            ..Default::default()
        },
    )
}
pub fn decompress_stream<R: Read, W: Write>(
    r: &mut R,
    w: &mut W,
    max_output: usize,
) -> Result<(usize, usize), BatoError> {
    decompress_stream_with_limits(
        r,
        w,
        &DecodeLimits {
            max_output_bytes: max_output as u64,
            ..Default::default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primer_choice_bit_is_read_for_cm_chunks_and_rejected_elsewhere() {
        // Coded CM chunk, no transform: n = 10, c = 4.
        let header = read_block_header(&mut &[0x41, 10, 4][..]).unwrap();
        assert_eq!(header.block_type, BlockType::Coded);
        assert_eq!(header.primer, Primer::Skipped);
        let header = read_block_header(&mut &[0x01, 10, 4][..]).unwrap();
        assert_eq!(header.primer, Primer::Read);
        // An independent segment declaring a primer choice is corrupt.
        assert!(read_block_header(&mut &[0x42, 10][..]).is_err());
        // The top bit is the fourth transform bit; id 15 is undefined.
        let header = read_block_header(&mut &[0x81, 10, 4][..]).unwrap();
        assert_eq!(header.transform_id, TransformId::X86Relative);
        assert!(read_block_header(&mut &[0xb9, 10, 4][..]).is_err());
    }

    #[test]
    fn fragmented_segment_reads_reuse_space_and_grow_with_the_data() {
        struct Fragments<'a> {
            bytes: &'a [u8],
            calls: usize,
            max_request: usize,
        }
        impl Read for Fragments<'_> {
            fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
                self.calls += 1;
                self.max_request = self.max_request.max(output.len());
                if self.calls.is_multiple_of(7) {
                    return Err(io::ErrorKind::Interrupted.into());
                }
                let size = output.len().min(1).min(self.bytes.len());
                output[..size].copy_from_slice(&self.bytes[..size]);
                self.bytes = &self.bytes[size..];
                Ok(size)
            }
        }
        let raw: Vec<_> = (0..20001).map(|i| (i % 251) as u8).collect();
        let descriptor = Descriptor {
            segment_log2: 20,
            chunk_log2: 14,
            table_log2: 20,
        };
        let mut reader = Fragments {
            bytes: &raw,
            calls: 0,
            max_request: 0,
        };
        let (decoded, observed) =
            super::super::diagnostics::capture(|| read_segment(&mut reader, descriptor));
        assert_eq!(decoded.unwrap(), raw);
        assert!(reader.bytes.is_empty());
        assert!(reader.max_request <= READ_SIZE);
        assert!(
            observed.heap.allocations <= 8,
            "allocation count must not scale with short-read count: {:?}",
            observed.heap
        );
        assert!(
            observed.heap.peak <= 2 * READ_SIZE,
            "small input must not preallocate its declared 1 MiB segment"
        );
    }
}

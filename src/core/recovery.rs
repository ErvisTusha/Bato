//! Offline salvage index for plain Bato streams. Every recovered extent is
//! checked against the index prepared from the intact, verified archive.
use super::{
    archive::{restore_catalogue, validate_path},
    checksum::Crc32c,
    cm::varint,
    frame::{
        decompress_checkpointed, decompress_stream_with_options, read_stream_header, HEADER_BYTES,
    },
    safe_fs::{absolute_path, open_regular_file, AtomicOutput},
    types::{BatoError, DecompressionOptions, Overwrite, StreamKind},
};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Cursor, Read, Seek, SeekFrom, Write},
    path::Path,
};

const BOUND_MAGIC: &[u8; 4] = b"BAR2";
const MAGIC: &[u8; 4] = b"BAR1";
pub(crate) const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;

fn corrupt(message: &str) -> BatoError {
    BatoError::CorruptStream(message.into())
}

fn fixed<const N: usize>(input: &mut impl Read) -> io::Result<[u8; N]> {
    let mut bytes = [0; N];
    input.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn number(input: &mut impl Read) -> io::Result<u64> {
    Ok(u64::from_le_bytes(fixed(input)?))
}

fn copy_exact(input: &mut impl Read, output: &mut impl Write, length: u64) -> io::Result<()> {
    if io::copy(&mut input.take(length), output)? != length {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "truncated recovery extent",
        ));
    }
    Ok(())
}

#[derive(Default)]
struct Checks {
    crc: Crc32c,
    sha: Sha256,
}

impl Write for Checks {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.crc.update(data);
        self.sha.update(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct Record {
    name: String,
    offset: u64,
    length: u64,
    crc: u32,
    directory: bool,
}

impl Record {
    fn header(&self, output: &mut impl Write) -> io::Result<()> {
        output.write_all(&[if self.directory { 1 } else { 2 }])?;
        output.write_all(&(self.name.len() as u16).to_le_bytes())?;
        output.write_all(self.name.as_bytes())
    }

    fn index(&self, output: &mut impl Write) -> io::Result<()> {
        self.header(output)?;
        if !self.directory {
            output.write_all(&self.offset.to_le_bytes())?;
            output.write_all(&self.length.to_le_bytes())?;
            output.write_all(&self.crc.to_le_bytes())?;
        }
        Ok(())
    }

    fn recover(&self, raw: &mut File, output: &mut impl Write) -> io::Result<()> {
        self.header(output)?;
        if !self.directory {
            output.write_all(&self.length.to_le_bytes())?;
            raw.seek(SeekFrom::Start(self.offset))?;
            copy_exact(raw, output, self.length)?;
            output.write_all(&self.crc.to_le_bytes())?;
        }
        Ok(())
    }
}

fn read_name(input: &mut impl Read, limit: usize) -> Result<String, BatoError> {
    let length = u16::from_le_bytes(fixed(input)?) as usize;
    if length > limit {
        return Err(corrupt("recovery path exceeds the configured limit"));
    }
    let mut name = vec![0; length];
    input.read_exact(&mut name)?;
    let name = String::from_utf8(name).map_err(|_| corrupt("invalid recovery path"))?;
    validate_path(&name)?;
    Ok(name)
}

fn index_catalogue(
    raw: &mut File,
    index: &mut File,
    options: &DecompressionOptions,
    verify: bool,
) -> Result<(), BatoError> {
    // Untrusted input must exercise full catalogue validation. The freshly
    // created indexed archive has already validated the source and encoded the
    // catalogue, so materializing its entire tree to disk again is redundant.
    raw.rewind()?;
    if verify {
        let stage = tempfile::tempdir()?;
        restore_catalogue(raw, &stage.path().join("tree"), Overwrite::Refuse, options)?;
        raw.rewind()?;
    }
    let mut originals = Vec::new();
    let mut input = BufReader::new(raw);
    loop {
        let [kind] = fixed(&mut input)?;
        if kind == 0 {
            return Ok(());
        }
        let name = read_name(&mut input, options.limits.max_path_bytes)?;
        let mut record = Record {
            name,
            offset: 0,
            length: 0,
            crc: 0,
            directory: kind == 1,
        };
        if kind == 1 {
            record.index(index)?;
            continue;
        }
        record.length = number(&mut input)?;
        if kind == 2 {
            record.offset = input.stream_position()?;
            let end = record
                .offset
                .checked_add(record.length)
                .ok_or_else(|| corrupt("catalogue offset overflow"))?;
            input.seek(SeekFrom::Start(end))?;
        } else {
            let id = u32::from_le_bytes(fixed(&mut input)?) as usize;
            record.offset = *originals
                .get(id)
                .ok_or_else(|| corrupt("invalid duplicate index"))?;
        }
        originals.push(record.offset);
        record.crc = u32::from_le_bytes(fixed(&mut input)?);
        record.index(index)?;
        if index.stream_position()? > MAX_INDEX_BYTES {
            return Err(corrupt("recovery index exceeds its size limit"));
        }
    }
}

pub fn create_index(
    source: &Path,
    index_path: &Path,
    overwrite: Overwrite,
    options: &DecompressionOptions,
) -> Result<(), BatoError> {
    let mut source_file = open_regular_file(source)?;
    let staged = AtomicOutput::new(index_path, overwrite, Some(source))?;
    let mut index = tempfile::tempfile()?;
    write_index(&mut source_file, &mut index, options)?;
    let mut output = staged.writer()?;
    io::copy(&mut index, &mut output)?;
    output.sync_all()?;
    drop(output);
    staged.publish()?;
    Ok(())
}

pub(crate) fn write_index(
    source_file: &mut File,
    index: &mut File,
    options: &DecompressionOptions,
) -> Result<(), BatoError> {
    write_index_inner(source_file, index, options, false)
}

/// Index a stream just encoded into a private temporary file by Bato itself.
/// Validation of untrusted files remains mandatory in `write_index`.
pub(crate) fn write_index_generated(
    source_file: &mut File,
    index: &mut File,
    options: &DecompressionOptions,
) -> Result<(), BatoError> {
    write_index_inner(source_file, index, options, true)
}

fn write_index_inner(
    source_file: &mut File,
    index: &mut File,
    options: &DecompressionOptions,
    generated: bool,
) -> Result<(), BatoError> {
    let source_hash = super::jobs::fingerprint(source_file)?;
    let header = fixed::<{ HEADER_BYTES as usize }>(source_file)?;
    let kind = read_stream_header(&mut header.as_slice())?.kind;
    source_file.rewind()?;
    index.write_all(BOUND_MAGIC)?;
    index.write_all(&source_hash)?;
    index.write_all(&header)?;
    let mut raw = tempfile::tempfile()?;
    let mut previous = (0, HEADER_BYTES);
    let mut limits = options.clone();
    if kind == StreamKind::DirectoryArchive {
        limits.limits.max_output_bytes = limits
            .limits
            .max_output_bytes
            .saturating_add(limits.limits.max_metadata_bytes as u64)
            .saturating_add(5);
    }
    decompress_checkpointed(
        &mut BufReader::new(&mut *source_file),
        &mut raw,
        &limits,
        (None, &mut |raw, state| {
            let length = state.decoded - previous.0;
            raw.seek(SeekFrom::Start(previous.0))?;
            let mut checks = Checks::default();
            copy_exact(raw, &mut checks, length)?;
            index.write_all(&length.to_le_bytes())?;
            index.write_all(&(state.encoded - previous.1).to_le_bytes())?;
            index.write_all(&checks.crc.finalize().to_le_bytes())?;
            index.write_all(&checks.sha.finalize())?;
            previous = (state.decoded, state.encoded);
            if index.stream_position()? > MAX_INDEX_BYTES {
                return Err(corrupt("recovery index exceeds its size limit"));
            }
            Ok(())
        }),
    )?;
    if !generated && super::jobs::fingerprint(source_file)? != source_hash {
        return Err(corrupt("archive changed while building its index"));
    }
    index.write_all(&0u64.to_le_bytes())?;
    if kind == StreamKind::DirectoryArchive {
        index_catalogue(&mut raw, index, options, !generated)?;
    }
    index.write_all(&[0])?;
    if index.stream_position()? > MAX_INDEX_BYTES - 32 {
        return Err(corrupt("recovery index exceeds its size limit"));
    }
    let digest = super::jobs::fingerprint(index)?;
    index.seek(SeekFrom::End(0))?;
    index.write_all(&digest)?;
    index.rewind()?;
    Ok(())
}

struct Extent {
    length: u64,
    offset: u64,
    encoded: u64,
    crc: u32,
    sha: [u8; 32],
}

fn decode_extent(
    source: &mut (impl Read + Seek),
    header: &[u8],
    segment: &Extent,
    options: &DecompressionOptions,
) -> Result<Option<File>, BatoError> {
    let source_length = source.seek(SeekFrom::End(0))?;
    if segment
        .offset
        .checked_add(segment.encoded)
        .is_none_or(|end| end > source_length)
    {
        return Ok(None);
    }
    source.seek(SeekFrom::Start(segment.offset))?;
    let mut footer = vec![255];
    varint::write(&mut footer, segment.length)?;
    footer.extend_from_slice(&segment.crc.to_le_bytes());
    let mut input = header
        .chain(BufReader::new(source.take(segment.encoded)))
        .chain(Cursor::new(footer));
    let mut raw = tempfile::tempfile()?;
    let mut limits = options.clone();
    limits.limits.max_output_bytes = segment.length;
    match decompress_stream_with_options(&mut input, &mut raw, &limits) {
        Ok(_) => {}
        Err(
            error @ (BatoError::MemoryLimitExceeded { .. } | BatoError::AllocationFailed { .. }),
        ) => return Err(error),
        Err(BatoError::Io(error))
            if !matches!(
                error.kind(),
                io::ErrorKind::UnexpectedEof | io::ErrorKind::InvalidData
            ) =>
        {
            return Err(error.into())
        }
        Err(_) => return Ok(None),
    }
    if super::jobs::fingerprint(&mut raw)? != segment.sha {
        return Ok(None);
    }
    Ok(Some(raw))
}

#[derive(Default)]
pub struct Recovery {
    pub recovered: usize,
    pub missing: usize,
    pub damaged_segments: usize,
}

fn checked_index(path: &Path) -> Result<io::Take<BufReader<File>>, BatoError> {
    checked_index_file(open_regular_file(path)?)
}

fn checked_index_file(file: File) -> Result<io::Take<BufReader<File>>, BatoError> {
    let mut index = BufReader::new(file);
    index.rewind()?;
    let size = index.get_ref().metadata()?.len();
    if !(55..=MAX_INDEX_BYTES).contains(&size) {
        return Err(corrupt("invalid recovery index size"));
    }
    let length = size - 32;
    let mut digest = Sha256::new();
    copy_exact(&mut index, &mut digest, length)?;
    if digest.finalize()[..] != fixed::<32>(&mut index)? {
        return Err(corrupt("recovery index SHA-256 mismatch"));
    }
    index.rewind()?;
    Ok(index.take(length))
}

pub(crate) fn index_binding(file: File) -> Result<[u8; 32], BatoError> {
    let mut index = checked_index_file(file)?;
    if &fixed::<4>(&mut index)? != BOUND_MAGIC {
        return Err(corrupt("embedded extraction requires a BAR2 index"));
    }
    Ok(fixed(&mut index)?)
}

pub fn salvage(
    archive: &Path,
    index_path: &Path,
    destination: &Path,
    options: &DecompressionOptions,
) -> Result<Recovery, BatoError> {
    let destination = absolute_path(destination)?;
    if absolute_path(archive)?.starts_with(&destination)
        || absolute_path(index_path)?.starts_with(&destination)
    {
        return Err(corrupt("recovery destination overlaps an input"));
    }
    if destination.try_exists()? {
        return Err(corrupt("recovery requires a new output directory"));
    }
    let mut index = checked_index(index_path)?;
    match &fixed::<4>(&mut index)? {
        BOUND_MAGIC => {
            fixed::<32>(&mut index)?;
        }
        MAGIC => {}
        _ => return Err(corrupt("invalid recovery index header")),
    }
    let header = fixed::<{ HEADER_BYTES as usize }>(&mut index)?;
    let kind = read_stream_header(&mut header.as_slice())?.kind;
    let geometry = super::frame::read_geometry(&mut &header[7..])?;
    let mut source = open_regular_file(archive)?;
    let mut raw = tempfile::tempfile()?;
    let mut catalogue = BufWriter::new(tempfile::tempfile()?);
    let mut verified = Vec::<(u64, u64)>::new();
    let mut report = Recovery::default();
    let mut position = (0u64, HEADER_BYTES);
    let mut count = 0u64;
    loop {
        let length = number(&mut index)?;
        if length == 0 {
            break;
        }
        let encoded = number(&mut index)?;
        let extent = Extent {
            length,
            offset: position.1,
            encoded,
            crc: u32::from_le_bytes(fixed(&mut index)?),
            sha: fixed(&mut index)?,
        };
        if length > geometry.segment_size as u64
            || encoded == 0
            || !position.0.is_multiple_of(geometry.segment_size as u64)
        {
            return Err(corrupt("invalid recovery segment size"));
        }
        let end = position
            .0
            .checked_add(length)
            .ok_or_else(|| corrupt("recovery output offset overflow"))?;
        let logical_limit = if kind == StreamKind::DirectoryArchive {
            options
                .limits
                .max_output_bytes
                .saturating_add(options.limits.max_metadata_bytes as u64)
                .saturating_add(5)
        } else {
            options.limits.max_output_bytes
        };
        if end > logical_limit {
            return Err(corrupt("recovery input exceeds the output limit"));
        }
        if let Some(mut recovered) = decode_extent(&mut source, &header, &extent, options)? {
            raw.seek(SeekFrom::Start(position.0))?;
            copy_exact(&mut recovered, &mut raw, length)?;
            match verified.last_mut() {
                Some(last) if last.1 == position.0 => last.1 = end,
                _ => verified.push((position.0, end)),
            }
            if kind == StreamKind::Bytes {
                Record {
                    name: format!("segment-{count:08}-offset-{}", position.0),
                    offset: position.0,
                    length,
                    crc: extent.crc,
                    directory: false,
                }
                .recover(&mut raw, &mut catalogue)?;
                report.recovered += 1;
            }
        } else {
            report.damaged_segments += 1;
            if kind == StreamKind::Bytes {
                report.missing += 1;
            }
        }
        count += 1;
        position = (
            end,
            position
                .1
                .checked_add(encoded)
                .ok_or_else(|| corrupt("recovery archive offset overflow"))?,
        );
    }
    if kind == StreamKind::DirectoryArchive {
        recover_catalogue(
            &mut index,
            &mut raw,
            &mut catalogue,
            (&verified, options, &mut report),
        )?;
    } else if fixed::<1>(&mut index)? != [0] {
        return Err(corrupt("unexpected file recovery index record"));
    }
    if index.limit() != 0 {
        return Err(corrupt("trailing recovery index records"));
    }
    catalogue.write_all(&[0])?;
    catalogue.flush()?;
    let mut catalogue = catalogue.into_inner().map_err(|e| e.into_error())?;
    catalogue.rewind()?;
    restore_catalogue(
        &mut BufReader::new(catalogue),
        &destination,
        Overwrite::Refuse,
        options,
    )?;
    Ok(report)
}

fn recover_catalogue(
    index: &mut impl Read,
    raw: &mut File,
    output: &mut impl Write,
    (verified, options, report): (&[(u64, u64)], &DecompressionOptions, &mut Recovery),
) -> Result<(), BatoError> {
    let mut total = 0u64;
    let mut entries = 0usize;
    loop {
        let [kind] = fixed(index)?;
        if kind == 0 {
            return Ok(());
        }
        if kind > 2 || entries >= options.limits.max_archive_entries {
            return Err(corrupt("invalid or excessive recovery catalogue entries"));
        }
        entries += 1;
        let name = read_name(index, options.limits.max_path_bytes)?;
        let mut record = Record {
            name,
            offset: 0,
            length: 0,
            crc: 0,
            directory: kind == 1,
        };
        if record.directory {
            record.recover(raw, output)?;
            continue;
        }
        record.offset = number(index)?;
        record.length = number(index)?;
        record.crc = u32::from_le_bytes(fixed(index)?);
        let end = record
            .offset
            .checked_add(record.length)
            .ok_or_else(|| corrupt("recovery range overflow"))?;
        let candidate = verified.partition_point(|(_, end)| *end <= record.offset);
        if record.length != 0
            && !verified
                .get(candidate)
                .is_some_and(|&(start, stop)| start <= record.offset && stop >= end)
        {
            eprintln!("Unrecoverable member: {:?}", record.name);
            report.missing += 1;
            continue;
        }
        total = total
            .checked_add(record.length)
            .ok_or_else(|| corrupt("recovery size overflow"))?;
        if total > options.limits.max_output_bytes {
            return Err(corrupt("recovered files exceed the output limit"));
        }
        record.recover(raw, output)?;
        report.recovered += 1;
    }
}

/// Restore selected paths by seeking only to their indexed compressed segments.
pub fn extract_indexed(
    archive: &Path,
    index_path: &Path,
    destination: &Path,
    selection: (&[std::path::PathBuf], Overwrite, &DecompressionOptions),
) -> Result<usize, BatoError> {
    if selection.0.is_empty() {
        return Err(corrupt("indexed extraction requires selected members"));
    }
    let destination = absolute_path(destination)?;
    if absolute_path(archive)?.starts_with(&destination)
        || absolute_path(index_path)?.starts_with(&destination)
    {
        return Err(corrupt("indexed extraction destination overlaps an input"));
    }
    extract_indexed_stream(
        &mut open_regular_file(archive)?,
        (open_regular_file(index_path)?, IndexTrust::CheckStream),
        &destination,
        selection,
    )
}

pub(crate) enum IndexTrust {
    CheckStream,
    Authenticated,
}

pub(crate) fn extract_indexed_stream(
    source: &mut (impl Read + Seek),
    (index, trust): (File, IndexTrust),
    destination: &Path,
    (paths, overwrite, options): (&[std::path::PathBuf], Overwrite, &DecompressionOptions),
) -> Result<usize, BatoError> {
    let mut index = checked_index_file(index)?;
    if &fixed::<4>(&mut index)? != BOUND_MAGIC {
        return Err(corrupt("indexed extraction requires a BAR2 index"));
    }
    let expected = fixed::<32>(&mut index)?;
    source.rewind()?;
    if matches!(trust, IndexTrust::CheckStream) && super::jobs::fingerprint(source)? != expected {
        return Err(corrupt("archive does not match its extraction index"));
    }
    let header = fixed::<{ HEADER_BYTES as usize }>(&mut index)?;
    if read_stream_header(&mut header.as_slice())?.kind != StreamKind::DirectoryArchive
        || fixed::<{ HEADER_BYTES as usize }>(source)? != header
    {
        return Err(corrupt(
            "indexed extraction requires a matching directory archive",
        ));
    }
    let geometry = super::frame::read_geometry(&mut &header[7..])?;
    let mut extents = Vec::new();
    let (mut raw_pos, mut archive_pos) = (0u64, HEADER_BYTES);
    let archive_len = source.seek(SeekFrom::End(0))?;
    loop {
        let length = number(&mut index)?;
        if length == 0 {
            break;
        }
        let encoded = number(&mut index)?;
        if length > geometry.segment_size as u64
            || encoded == 0
            || !raw_pos.is_multiple_of(geometry.segment_size as u64)
        {
            return Err(corrupt("invalid indexed segment"));
        }
        let end = raw_pos
            .checked_add(length)
            .ok_or_else(|| corrupt("indexed length overflow"))?;
        let next = archive_pos
            .checked_add(encoded)
            .filter(|&end| end < archive_len)
            .ok_or_else(|| corrupt("indexed segment outside archive"))?;
        extents.push((
            raw_pos,
            end,
            Extent {
                length,
                offset: archive_pos,
                encoded,
                crc: u32::from_le_bytes(fixed(&mut index)?),
                sha: fixed(&mut index)?,
            },
        ));
        raw_pos = end;
        archive_pos = next;
    }
    let mut selected = Vec::new();
    let mut found = vec![false; paths.len()];
    let mut needed = std::collections::BTreeSet::new();
    let mut entries = 0;
    loop {
        let [kind] = fixed(&mut index)?;
        if kind == 0 {
            break;
        }
        entries += 1;
        if kind > 2 || entries > options.limits.max_archive_entries {
            return Err(corrupt("invalid indexed catalogue"));
        }
        let name = read_name(&mut index, options.limits.max_path_bytes)?;
        let (offset, length, crc) = if kind == 1 {
            (0, 0, 0)
        } else {
            (
                number(&mut index)?,
                number(&mut index)?,
                u32::from_le_bytes(fixed(&mut index)?),
            )
        };
        let member = Path::new(&name);
        for (path, found) in paths.iter().zip(&mut found) {
            *found |= member == path;
        }
        if !paths.iter().any(|path| member.starts_with(path)) {
            continue;
        }
        let end = offset
            .checked_add(length)
            .filter(|&end| end <= raw_pos)
            .ok_or_else(|| corrupt("indexed member outside catalogue"))?;
        if length != 0 {
            let start = extents.partition_point(|(_, end, _)| *end <= offset);
            let stop = extents.partition_point(|(start, _, _)| *start < end);
            needed.extend(start..stop);
        }
        selected.push(Record {
            name,
            offset,
            length,
            crc,
            directory: kind == 1,
        });
    }
    if index.limit() != 0 || found.iter().any(|&found| !found) {
        return Err(corrupt(
            "selection missing from archive or trailing index records",
        ));
    }
    let mut raw = tempfile::tempfile()?;
    let decoded_count = needed.len();
    for id in needed {
        let (offset, _, extent) = &extents[id];
        let mut data = decode_extent(source, &header, extent, options)?
            .ok_or_else(|| corrupt("selected segment failed integrity verification"))?;
        raw.seek(SeekFrom::Start(*offset))?;
        copy_exact(&mut data, &mut raw, extent.length)?;
    }
    let mut catalogue = BufWriter::new(tempfile::tempfile()?);
    let mut total = 0u64;
    for record in selected {
        total = total
            .checked_add(record.length)
            .filter(|&n| n <= options.limits.max_output_bytes)
            .ok_or_else(|| corrupt("selected members exceed the output limit"))?;
        record.recover(&mut raw, &mut catalogue)?;
    }
    catalogue.write_all(&[0])?;
    let mut catalogue = catalogue.into_inner().map_err(|error| error.into_error())?;
    catalogue.rewind()?;
    restore_catalogue(
        &mut BufReader::new(catalogue),
        destination,
        overwrite,
        options,
    )?;
    Ok(decoded_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{CompressionOptions, ExecutionOptions, Geometry};

    #[test]
    fn generated_index_matches_verified_index() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("source");
        std::fs::create_dir(&source).unwrap();
        std::fs::write(source.join("a"), b"the same contents".repeat(300)).unwrap();
        std::fs::write(source.join("duplicate"), b"the same contents".repeat(300)).unwrap();
        std::fs::write(source.join("other"), b"another entry").unwrap();

        let options = CompressionOptions {
            geometry: Geometry {
                segment_size: 1 << 16,
                chunk_size: 1 << 14,
                table_log2: 16,
            },
            execution: ExecutionOptions {
                threads: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut encoded = tempfile::tempfile().unwrap();
        crate::core::archive::create_solid_archive(&source, &mut encoded, &options).unwrap();

        let decode = DecompressionOptions::default();
        let mut generated = tempfile::tempfile().unwrap();
        let mut verified = tempfile::tempfile().unwrap();
        write_index_generated(&mut encoded, &mut generated, &decode).unwrap();
        write_index(&mut encoded, &mut verified, &decode).unwrap();
        let mut first = Vec::new();
        let mut second = Vec::new();
        generated.rewind().unwrap();
        verified.rewind().unwrap();
        generated.read_to_end(&mut first).unwrap();
        verified.read_to_end(&mut second).unwrap();
        assert_eq!(first, second);
    }
}

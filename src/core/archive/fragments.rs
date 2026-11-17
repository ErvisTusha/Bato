//! Bounded cross-file copies in the native directory catalogue.
//! Provenance: the LZ77 distance/length principle and Bato's cross_fragments experiment.
use super::{catalogue_exact, open_source_file, ArchiveInput, FileDigest, HostBudget};
use crate::core::{
    checksum::crc32c,
    cm::{segment::Direction, varint},
    codec,
    memory::{try_buffer, try_capacity, MemoryLease},
    types::{BatoError, CompressionOptions},
};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, io::{self, Read}};

pub(super) const LIMIT: usize = 8 * 1024 * 1024;
pub(super) const START: u8 = 5;
pub(super) const MEMBER: u8 = 4;
pub(super) const VERSION: u8 = 25;
const MAX_MEMBERS: usize = 4096;
const MIN_COPY: usize = 192;
const MAX_RUNS: usize = 1 << 14;

type Spot = (usize, usize);
type Command = [u32; 4];

fn hash(data: &[u8]) -> u64 {
    let value = u64::from_le_bytes(data.try_into().unwrap());
    let mixed = (value ^ (value >> 33)).wrapping_mul(0xff51_afd7_ed55_8ccd);
    mixed ^ (mixed >> 33)
}

fn anchor(data: &[u8], offset: usize) -> Option<u64> {
    let word = data.get(offset..offset + 8)?;
    let key = hash(word);
    (key & 31 == 0).then_some(key)
}

fn matched(a: &[u8], b: &[u8]) -> usize {
    let limit = a.len().min(b.len());
    let mut length = 0;
    while length + 8 <= limit {
        let x = u64::from_le_bytes(a[length..length + 8].try_into().unwrap());
        let y = u64::from_le_bytes(b[length..length + 8].try_into().unwrap());
        let diff = x ^ y;
        if diff != 0 {
            return length + (diff.trailing_zeros() / 8) as usize;
        }
        length += 8;
    }
    length
        + a[length..limit]
            .iter()
            .zip(&b[length..limit])
            .take_while(|(x, y)| x == y)
            .count()
}

fn factor(
    raw: &[u8],
    previous: &[Vec<u8>],
    index: &HashMap<u64, [Option<Spot>; 2]>,
) -> (Vec<u8>, Vec<Command>) {
    let (mut literals, mut commands) = (Vec::new(), Vec::new());
    let (mut at, mut copied) = (0, 0);
    while at + MIN_COPY <= raw.len() && commands.len() < MAX_RUNS {
        let mut best = None;
        let mut length = MIN_COPY - 1;
        if let Some(spots) = anchor(raw, at).and_then(|key| index.get(&key)) {
            for &(file, offset) in spots.iter().flatten() {
                let n = matched(&previous[file][offset..], &raw[at..]);
                if n > length {
                    length = n;
                    best = Some((file, offset));
                }
            }
        }
        if let Some((file, mut offset)) = best {
            let mut start = at;
            while start > copied && offset > 0 && raw[start - 1] == previous[file][offset - 1] {
                start -= 1;
                offset -= 1;
            }
            let literal_len = start - copied;
            literals.extend_from_slice(&raw[copied..start]);
            copied = at + length;
            commands.push([
                literal_len as u32,
                file as u32,
                offset as u32,
                (copied - start) as u32,
            ]);
            at = copied;
        } else {
            at += 1;
        }
    }
    literals.extend_from_slice(&raw[copied..]);
    (literals, commands)
}


fn invalid() -> BatoError {
    BatoError::CorruptStream("invalid archive fragment recipe".into())
}

pub(super) fn reserve(
    input: &ArchiveInput<'_>,
    options: &CompressionOptions,
    host: &std::sync::Arc<HostBudget>,
) -> Result<Option<MemoryLease>, BatoError> {
    if !(2..=MAX_MEMBERS).contains(&input.files.len()) {
        return Ok(None);
    }
    let Some(bytes) = input.expected_files.iter().try_fold(0u64, |sum, file| {
        sum.checked_add(file.0).filter(|&size| size <= LIMIT as u64)
    }) else {
        return Ok(None);
    };
    let descriptor = options.resolved().geometry.descriptor()?;
    let current = codec::admit((descriptor, Direction::Encode), host.available(), &options.execution)?;
    let estimate = 8 * bytes as usize + 1024 * 1024;
    // Factoring is optional when the original worker pool needs this memory.
    if host.available().saturating_sub(estimate) < current.host_memory {
        return Ok(None);
    }
    Ok(Some(host.reserve(estimate)?))
}

fn load(path: &std::path::Path, expected: FileDigest) -> Result<Vec<u8>, BatoError> {
    let size = expected.0 as usize;
    let mut file = open_source_file(path)?;
    let mut data = try_buffer(size)?;
    file.read_exact(&mut data)?;
    let digest: [u8; 32] = Sha256::digest(&data).into();
    if file.read(&mut [0])? != 0 || digest != expected.1 || crc32c(&data) != expected.2 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "file changed while archiving").into());
    }
    Ok(data)
}

pub(super) fn prepare(
    input: &ArchiveInput<'_>,
    length: usize,
) -> Result<Option<Vec<u8>>, BatoError> {
    let mut original = try_capacity(input.files.len())?;
    for ((path, _), &expected) in input.files.iter().zip(input.expected_files) {
        if let Some(root) = input.root {
            root.verify()?;
        }
        original.push(load(path, expected)?);
    }
    let mut index = HashMap::<u64, [Option<Spot>; 2]>::new();
    let mut output = try_capacity(length + 1)?;
    output.push(START);
    output.extend_from_slice(&input.stage);
    let mut found = false;
    for (id, raw) in original.iter().enumerate() {
        let duplicate = input.duplicates[id];
        let (literals, commands) = if duplicate.is_some() {
            (Vec::new(), Vec::new())
        } else {
            factor(raw, &original[..id], &index)
        };
        let recipe = !commands.is_empty()
            && literals.len() + 16 * commands.len() + 24 < raw.len() * 3 / 4;
        let name = input.files[id].1.as_bytes();
        output.push(if duplicate.is_some() { 3 } else if recipe { MEMBER } else { 2 });
        output.extend_from_slice(&(name.len() as u16).to_le_bytes());
        output.extend_from_slice(name);
        output.extend_from_slice(&input.expected_files[id].0.to_le_bytes());
        if let Some(reference) = duplicate {
            output.extend_from_slice(&reference.to_le_bytes());
        } else if recipe {
            found = true;
            varint::write(&mut output, commands.len() as u64)?;
            let mut at = 0;
            for command in commands {
                for value in command {
                    varint::write(&mut output, u64::from(value))?;
                }
                let end = at + command[0] as usize;
                output.extend_from_slice(&literals[at..end]);
                at = end;
            }
            output.extend_from_slice(&literals[at..]);
        } else {
            output.extend_from_slice(raw);
        }
        output.extend_from_slice(&input.expected_files[id].2.to_le_bytes());
        if duplicate.is_none() {
            for offset in 0..raw.len().saturating_sub(MIN_COPY - 1) {
                let Some(key) = anchor(raw, offset) else { continue };
                let spot = Some((id, offset));
                if let Some(spots) = index.get_mut(&key) {
                    *spots = [spot, spots[0]];
                } else if index.len() < LIMIT / 16 {
                    index.insert(key, [spot, None]);
                }
            }
        }
    }
    output.push(0);
    Ok((found && output.len() < length).then_some(output))
}

/// Expands an earlier-file recipe into its exact declared length. The reader
/// supplied by the catalogue validates both the earlier ID and its range.
pub(super) fn read_member(
    source: &mut impl Read,
    size: u64,
    mut read_copy: impl FnMut(usize, usize, &mut [u8]) -> Result<(), BatoError>,
) -> Result<Vec<u8>, BatoError> {
    let size = usize::try_from(size).map_err(|_| invalid())?;
    if size > LIMIT {
        return Err(invalid());
    }
    let count = read_number(source)?;
    if !(1..=MAX_RUNS).contains(&count) || count > size / MIN_COPY {
        return Err(invalid());
    }
    let mut output = try_buffer(size)?;
    let mut at = 0;
    for _ in 0..count {
        let literal = read_number(source)?;
        let file = read_number(source)?;
        let offset = read_number(source)?;
        let length = read_number(source)?;
        let split = at.checked_add(literal).ok_or_else(invalid)?;
        let end = split.checked_add(length).ok_or_else(invalid)?;
        if end > size || length < MIN_COPY {
            return Err(invalid());
        }
        catalogue_exact(source, &mut output[at..split], BatoError::TruncatedHeader)?;
        read_copy(file, offset, &mut output[split..end])?;
        at = end;
    }
    catalogue_exact(source, &mut output[at..], BatoError::TruncatedHeader)?;
    Ok(output)
}

fn read_number(source: &mut impl Read) -> Result<usize, BatoError> {
    usize::try_from(varint::read(source)?).map_err(|_| invalid())
}

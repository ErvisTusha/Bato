//! Verified catalogue inspection without materializing members.
use super::{
    catalogue_exact, validate_path, BatoError, DecompressReader, DecompressionOptions, HostBudget,
    Read, Sha256, StreamKind, COPY_BUFFER,
};
use crate::core::checksum::Crc32c;
use sha2::Digest;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Entry {
    pub size: u64,
    pub sha256: [u8; 32],
}

pub type Inventory = BTreeMap<String, Option<Entry>>;

pub(super) fn entry_cost(path_len: usize) -> usize {
    256 + 4 * path_len
}

fn invalid(text: &str) -> BatoError {
    BatoError::CorruptStream(text.into())
}

fn read<const N: usize>(source: &mut impl Read) -> Result<[u8; N], BatoError> {
    let mut bytes = [0; N];
    catalogue_exact(source, &mut bytes, BatoError::TruncatedHeader)?;
    Ok(bytes)
}

pub(super) fn inspect(
    source: &mut impl Read,
    options: &DecompressionOptions,
) -> Result<Inventory, BatoError> {
    let limits = &options.limits;
    let host = HostBudget::new(limits.max_working_memory, &options.execution.stats)?;
    let metadata_limit = limits.max_metadata_bytes.min(host.limit() / 4);
    let _metadata = host.reserve(metadata_limit.saturating_add(COPY_BUFFER))?;
    let mut stream = options.clone();
    stream.limits.max_output_bytes = limits
        .max_output_bytes
        .saturating_add(metadata_limit as u64)
        .saturating_add(5);
    let mut source = DecompressReader::with_budget(source, &stream, host)?;
    if source.kind() != StreamKind::DirectoryArchive {
        return Err(invalid("inspection requires a directory archive"));
    }
    let mut entries = Inventory::new();
    let mut files = Vec::<(Entry, u32)>::new();
    let mut buffer = vec![0; COPY_BUFFER];
    let mut total = 0u64;
    let mut metadata = 0usize;
    loop {
        let [kind] = read(&mut source)?;
        if kind == 0 {
            break;
        }
        let len = u16::from_le_bytes(read(&mut source)?) as usize;
        metadata = metadata.saturating_add(entry_cost(len));
        if len > limits.max_path_bytes
            || entries.len() >= limits.max_archive_entries
            || metadata > metadata_limit
        {
            return Err(invalid("archive catalogue exceeds inspection limits"));
        }
        let mut name = vec![0; len];
        catalogue_exact(&mut source, &mut name, BatoError::TruncatedHeader)?;
        let name = String::from_utf8(name).map_err(|_| invalid("archive path is not UTF-8"))?;
        validate_path(&name)?;
        let entry = if kind == 1 {
            None
        } else {
            let size = u64::from_le_bytes(read(&mut source)?);
            total = total
                .checked_add(size)
                .filter(|&n| n <= limits.max_output_bytes)
                .ok_or_else(|| invalid("archive files exceed inspection output limit"))?;
            let (entry, crc) = match kind {
                2 => {
                    let mut crc = Crc32c::new();
                    let mut sha = Sha256::new();
                    let mut remaining = size;
                    while remaining != 0 {
                        let n = remaining.min(buffer.len() as u64) as usize;
                        catalogue_exact(&mut source, &mut buffer[..n], BatoError::TruncatedHeader)?;
                        crc.update(&buffer[..n]);
                        sha.update(&buffer[..n]);
                        remaining -= n as u64;
                    }
                    (
                        Entry {
                            size,
                            sha256: sha.finalize().into(),
                        },
                        crc.finalize(),
                    )
                }
                3 => {
                    let index = u32::from_le_bytes(read(&mut source)?) as usize;
                    let (entry, crc) = files
                        .get(index)
                        .ok_or_else(|| invalid("invalid duplicate reference"))?;
                    if entry.size != size {
                        return Err(invalid("duplicate size mismatch"));
                    }
                    (entry.clone(), *crc)
                }
                _ => return Err(invalid("unknown archive entry type")),
            };
            let expected = u32::from_le_bytes(read(&mut source)?);
            if expected != crc {
                return Err(BatoError::ChecksumMismatch {
                    expected,
                    computed: crc,
                });
            }
            files.push((entry.clone(), crc));
            Some(entry)
        };
        if entries.insert(name, entry).is_some() {
            return Err(invalid("duplicate archive path"));
        }
    }
    if source.read(&mut [0])? != 0 {
        return Err(invalid("trailing archive catalogue data"));
    }
    source.finish()?;
    validate_tree(&entries)?;
    Ok(entries)
}

pub(super) fn validate_tree(entries: &Inventory) -> Result<(), BatoError> {
    for name in entries.keys() {
        for (position, _) in name.match_indices('/') {
            if matches!(entries.get(&name[..position]), Some(Some(_))) {
                return Err(invalid("archive file is also a parent directory"));
            }
        }
    }
    Ok(())
}

pub fn inspect_directory(
    root: &std::path::Path,
    options: &DecompressionOptions,
) -> Result<Inventory, BatoError> {
    use super::{Directory, IfMissing, Kind, Path};
    fn visit(
        dir: &Directory,
        prefix: &str,
        result: &mut Inventory,
        (limits, metadata): (&super::DecodeLimits, &mut usize),
    ) -> Result<(), BatoError> {
        for name in dir.names()? {
            let name = name?;
            let name_utf8 = name
                .to_str()
                .ok_or_else(|| invalid("directory path is not UTF-8"))?;
            let path = if prefix.is_empty() {
                name_utf8.to_owned()
            } else {
                format!("{prefix}/{name_utf8}")
            };
            validate_path(&path)?;
            *metadata = metadata.saturating_add(entry_cost(path.len()));
            if result.len() >= limits.max_archive_entries
                || path.len() > limits.max_path_bytes
                || *metadata > limits.max_metadata_bytes
            {
                return Err(invalid("directory exceeds inspection limits"));
            }
            match dir.kind(&name)? {
                Kind::Directory => {
                    result.insert(path.clone(), None);
                    visit(
                        &dir.child(&name, IfMissing::Fail)?,
                        &path,
                        result,
                        (limits, metadata),
                    )?;
                }
                Kind::Regular => {
                    let mut file = dir.open_file(Path::new(&name))?;
                    let mut sha = Sha256::new();
                    let size = std::io::copy(&mut file, &mut sha)?;
                    result.insert(
                        path,
                        Some(Entry {
                            size,
                            sha256: sha.finalize().into(),
                        }),
                    );
                }
                Kind::Special => {
                    return Err(invalid("directory contains a symlink or special file"))
                }
            }
        }
        Ok(())
    }
    let host = HostBudget::new(options.limits.max_working_memory, &options.execution.stats)?;
    let mut limits = options.limits.clone();
    limits.max_metadata_bytes = limits.max_metadata_bytes.min(host.limit() / 4);
    let _metadata = host.reserve(limits.max_metadata_bytes)?;
    let mut result = Inventory::new();
    visit(
        &Directory::open(root, IfMissing::Fail)?,
        "",
        &mut result,
        (&limits, &mut 0),
    )?;
    Ok(result)
}

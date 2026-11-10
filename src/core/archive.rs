//! Streaming Bato directory archives with safe staged extraction.

use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
#[cfg(test)]
use std::fs;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use super::checksum::Crc32c;
use super::frame::{compress_archive_stream, read_error, DecompressReader};
use super::memory::{try_buffer, try_capacity, try_filled, HostBudget};
use super::safe_fs::{
    absolute_path, open_regular_file, Directory, IfMissing, Kind, StagingDirectory,
};
pub mod inventory;
pub mod managed;
mod selection;
pub use selection::MemberSelection;
use selection::{member, member_kind, PublishScope};

use super::types::{
    BatoError, CompressionOptions, DecodeLimits, DecompressionOptions, Overwrite, StreamKind,
    MAX_ARCHIVE_ENTRIES, MAX_PATH_BYTES,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveFileEntry {
    pub rel_path: String,
    pub file_size: u64,
    pub file_crc: u32,
}

fn err(e: io::Error) -> BatoError {
    BatoError::Io(e)
}
fn catalogue_exact<R: Read>(
    reader: &mut R,
    bytes: &mut [u8],
    eof: BatoError,
) -> Result<(), BatoError> {
    reader.read_exact(bytes).map_err(|error| {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            eof
        } else {
            read_error(error)
        }
    })
}
fn path_name(root: &Path, p: &Path) -> Result<String, BatoError> {
    let rel = p
        .strip_prefix(root)
        .map_err(|_| BatoError::CorruptStream("path outside root".into()))?;
    let mut out = String::new();
    for c in rel.components() {
        let Component::Normal(s) = c else {
            return Err(BatoError::CorruptStream("unsafe archive path".into()));
        };
        let s = s
            .to_str()
            .ok_or_else(|| BatoError::CorruptStream("non-UTF8 path".into()))?;
        if s.contains('\\') || s == "." || s == ".." {
            return Err(BatoError::CorruptStream("unsafe archive path".into()));
        }
        if !out.is_empty() {
            out.push('/');
        }
        out.push_str(s);
    }
    if out.is_empty() || out.len() > u16::MAX as usize {
        return Err(BatoError::CorruptStream("invalid archive path".into()));
    }
    validate_path(&out)?;
    Ok(out)
}
// Charge conservatively for path buffers, vector growth, hash buckets, saved
// fingerprints and the streamed directory prefix. Check before retaining each
// entry; never collect an unbounded read_dir result.
const INVENTORY_METADATA: usize = 32 * 1024 * 1024;
/// Stack buffer for streaming file contents, in bytes.
const COPY_BUFFER: usize = 128 * 1024;
/// Fixed metadata charged per inventory entry, in bytes.
const ENTRY_CHARGE: usize = 512;
/// Charge per byte of each entry's source path and archive name.
const NAME_CHARGE: usize = 4;

/// Directory and file entries found under an archive root, with the metadata
/// charged so far against `INVENTORY_METADATA`.
struct Inventory<'a> {
    root: &'a Path,
    excluded: &'a [PathBuf],
    dirs: Vec<(PathBuf, String)>,
    files: Vec<(PathBuf, String)>,
    charged: usize,
}
impl Inventory<'_> {
    /// Adds every entry below `cur`, depth first.
    fn walk(&mut self, cur: &Path) -> Result<(), BatoError> {
        let dir = Directory::open(cur, IfMissing::Fail).map_err(err)?;
        for entry in dir.names().map_err(err)? {
            let name = entry.map_err(err)?;
            let path = cur.join(&name);
            if self
                .excluded
                .iter()
                .any(|x| path == *x || path.starts_with(x))
            {
                continue;
            }
            let archived = path_name(self.root, &path)?;
            let kind = dir.kind(&name).map_err(err)?;
            if kind == Kind::Special {
                return Err(BatoError::CorruptStream(format!(
                    "unsupported file or symlink: {archived}"
                )));
            }
            self.admit(&path, &archived)?;
            if kind == Kind::Directory {
                super::memory::try_reserve(&mut self.dirs, 1)?;
                self.dirs.push((path.clone(), archived));
                self.walk(&path)?;
            } else {
                super::memory::try_reserve(&mut self.files, 1)?;
                self.files.push((path, archived));
            }
        }
        Ok(())
    }
    /// Checks the path, entry and metadata limits before retaining an entry.
    fn admit(&mut self, path: &Path, archived: &str) -> Result<(), BatoError> {
        if archived.len() > MAX_PATH_BYTES {
            return Err(BatoError::CorruptStream("path exceeds limit".into()));
        }
        if self.dirs.len().saturating_add(self.files.len()) >= MAX_ARCHIVE_ENTRIES {
            return Err(BatoError::CorruptStream(
                "archive entry limit exceeded".into(),
            ));
        }
        let names = path.as_os_str().len().saturating_add(archived.len());
        let charge = ENTRY_CHARGE.saturating_add(NAME_CHARGE.saturating_mul(names));
        self.charged = self.charged.saturating_add(charge);
        if self.charged > INVENTORY_METADATA {
            return Err(BatoError::MemoryLimitExceeded {
                required: self.charged,
                limit: INVENTORY_METADATA,
            });
        }
        Ok(())
    }
}

fn files_equal(a: &Path, b: &Path) -> io::Result<bool> {
    let mut fa = open_source_file(a)?;
    let mut fb = open_source_file(b)?;
    if fa.metadata()?.len() != fb.metadata()?.len() {
        return Ok(false);
    }
    let mut aa = [0u8; COPY_BUFFER];
    let mut bb = [0u8; COPY_BUFFER];
    loop {
        let na = fa.read(&mut aa)?;
        let nb = fb.read(&mut bb)?;
        if na != nb {
            return Ok(false);
        }
        if na == 0 {
            return Ok(true);
        }
        if aa[..na] != bb[..nb] {
            return Ok(false);
        }
    }
}
fn fingerprint_file(p: &Path) -> io::Result<(u64, [u8; 32], u32)> {
    let mut f = open_source_file(p)?;
    let size = f.metadata()?.len();
    let mut total = 0u64;
    let mut h = Sha256::new();
    let mut crc = Crc32c::new();
    let mut b = [0u8; COPY_BUFFER];
    loop {
        let n = f.read(&mut b)?;
        if n == 0 {
            break;
        }
        total = total
            .checked_add(n as u64)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "file too large"))?;
        h.update(&b[..n]);
        crc.update(&b[..n]);
    }
    if total != size || f.metadata()?.len() != size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file changed while archiving",
        ));
    }
    Ok((size, h.finalize().into(), crc.finalize()))
}
fn open_source_file(path: &Path) -> io::Result<File> {
    open_regular_file(path)
}

struct ArchiveInput<'a> {
    root: Option<&'a Directory>,
    files: &'a [(PathBuf, String)],
    stage: Vec<u8>,
    pos: usize,
    file: Option<File>,
    index: usize,
    remaining: u64,
    crc: Crc32c,
    sha: Sha256,
    crc_pending: bool,
    crc_bytes: [u8; 4],
    ended: bool,
    duplicates: &'a [Option<u32>],
    expected_files: &'a [(u64, [u8; 32], u32)],
    started: bool,
}
impl<'a> ArchiveInput<'a> {
    fn start_file(&mut self) -> io::Result<()> {
        if let Some(root) = self.root {
            root.verify()?;
        }
        let (p, n) = &self.files[self.index];
        self.stage.clear();
        if let Some(id) = self.duplicates[self.index] {
            self.stage.extend_from_slice(&[3]);
            self.stage
                .extend_from_slice(&(n.len() as u16).to_le_bytes());
            self.stage.extend_from_slice(n.as_bytes());
            let fingerprint = fingerprint_file(p)?;
            if fingerprint != self.expected_files[self.index] {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "file changed while archiving",
                ));
            }
            self.stage.extend_from_slice(&fingerprint.0.to_le_bytes());
            self.stage.extend_from_slice(&id.to_le_bytes());
            self.stage.extend_from_slice(&fingerprint.2.to_le_bytes());
            self.pos = 0;
            self.file = None;
            self.crc_pending = true;
            return Ok(());
        }
        self.stage.extend_from_slice(&[2]);
        self.stage
            .extend_from_slice(&(n.len() as u16).to_le_bytes());
        self.stage.extend_from_slice(n.as_bytes());
        let f = open_source_file(p)?;
        let m = f.metadata()?;
        if m.len() != self.expected_files[self.index].0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file changed while archiving",
            ));
        }
        self.stage.extend_from_slice(&m.len().to_le_bytes());
        self.pos = 0;
        self.remaining = m.len();
        self.crc = Crc32c::new();
        self.sha = Sha256::new();
        self.crc_pending = false;
        self.file = Some(f);
        Ok(())
    }
}
impl<'a> Read for ArchiveInput<'a> {
    fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
        if b.is_empty() {
            return Ok(0);
        }
        if self.pos < self.stage.len() {
            let n = (self.stage.len() - self.pos).min(b.len());
            b[..n].copy_from_slice(&self.stage[self.pos..self.pos + n]);
            self.pos += n;
            return Ok(n);
        }
        if self.index < self.files.len() {
            if !self.started {
                self.start_file()?;
                self.started = true;
                return self.read(b);
            }
            if self.file.is_none() {
                self.started = false;
                self.index += 1;
                self.stage.clear();
                self.pos = 0;
                self.crc_pending = false;
                return self.read(b);
            }
            let take = self.remaining.min(b.len() as u64) as usize;
            let n = if take == 0 {
                0
            } else {
                self.file.as_mut().unwrap().read(&mut b[..take])?
            };
            if n > 0 {
                self.remaining = self.remaining.saturating_sub(n as u64);
                self.crc.update(&b[..n]);
                self.sha.update(&b[..n]);
                return Ok(n);
            }
            if self.remaining != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "file changed while archiving",
                ));
            }
            if !self.crc_pending {
                let f = self.file.as_mut().unwrap();
                let mut extra = [0u8; 1];
                if f.read(&mut extra)? != 0
                    || f.metadata()?.len() != self.expected_files[self.index].0
                    || self.sha.clone().finalize().as_slice() != self.expected_files[self.index].1
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "file changed while archiving",
                    ));
                }
                self.crc_bytes = self.crc.finalize().to_le_bytes();
                self.pos = self.stage.len();
                self.stage.extend_from_slice(&self.crc_bytes);
                self.crc_pending = true;
                return self.read(b);
            }
            self.file = None;
            self.index += 1;
            self.started = false;
            return self.read(b);
        }
        if !self.ended {
            self.ended = true;
            b[0] = 0;
            return Ok(1);
        }
        Ok(0)
    }
}

pub fn create_solid_archive<P: AsRef<Path>, W: Write>(
    dir: P,
    w: &mut W,
    o: &CompressionOptions,
) -> Result<usize, BatoError> {
    create_solid_archive_excluding(dir, w, o, &[])
}
pub fn create_solid_archive_excluding<P: AsRef<Path>, W: Write>(
    dir: P,
    w: &mut W,
    o: &CompressionOptions,
    excluded: &[PathBuf],
) -> Result<usize, BatoError> {
    let host = HostBudget::new(o.memory_budget, &o.execution.stats)?;
    let _metadata = host.reserve(INVENTORY_METADATA + 512 * 1024)?;
    super::codec::admit(
        (
            o.geometry.descriptor()?,
            super::cm::segment::Direction::Encode,
        ),
        host.available(),
        &o.execution,
    )?;
    with_catalogue(dir.as_ref(), excluded, |input| {
        compress_archive_stream(input, w, o, host).map(|(_, n)| n)
    })
}

/// Serializes the same catalogue used by solid archives to a seekable staging file.
pub fn stage_catalogue(
    source: &Path,
    output: &mut impl Write,
    options: &CompressionOptions,
) -> Result<(), BatoError> {
    let host = HostBudget::new(options.memory_budget, &options.execution.stats)?;
    let _metadata = host.reserve(INVENTORY_METADATA + 512 * 1024)?;
    with_catalogue(source, &[], |input| Ok(io::copy(input, output)?))?;
    Ok(())
}

fn with_catalogue<T>(
    dir: &Path,
    excluded: &[PathBuf],
    consume: impl FnOnce(&mut ArchiveInput<'_>) -> Result<T, BatoError>,
) -> Result<T, BatoError> {
    // Reject links before lexical normalization, then use consistent absolute
    // paths for inventory and exclusions (including a new in-tree output).
    let root_guard = Directory::open(dir, IfMissing::Fail).map_err(err)?;
    let root_meta = root_guard.file().metadata().map_err(err)?;
    let root_path = absolute_path(dir).map_err(err)?;
    let root = root_path.as_path();
    let excluded = excluded
        .iter()
        .map(|p| absolute_path(p))
        .collect::<io::Result<Vec<_>>>()
        .map_err(err)?;
    root_guard.verify().map_err(err)?;
    if root_meta.file_type().is_symlink() || !root_meta.is_dir() {
        return Err(BatoError::CorruptStream(
            "archive root is not a directory".into(),
        ));
    }
    let mut inventory = Inventory {
        root,
        excluded: &excluded,
        dirs: Vec::new(),
        files: Vec::new(),
        charged: 0,
    };
    inventory.walk(root)?;
    let Inventory {
        dirs: mut ds,
        files: mut fs_,
        ..
    } = inventory;
    ds.sort_by(|a, b| a.1.cmp(&b.1));
    fs_.sort_by(|a, b| a.1.cmp(&b.1));
    let mut duplicates = try_filled(fs_.len(), None)?;
    let mut expected_files = try_capacity(fs_.len())?;
    let mut seen: HashMap<(u64, [u8; 32]), u32> = HashMap::new();
    for (i, (path, _)) in fs_.iter().enumerate() {
        root_guard.verify().map_err(err)?;
        let fingerprint = fingerprint_file(path).map_err(err)?;
        expected_files.push(fingerprint);
        let key = (fingerprint.0, fingerprint.1);
        if let Some(&id) = seen.get(&key) {
            if files_equal(path, &fs_[id as usize].0).map_err(err)? {
                duplicates[i] = Some(id);
                continue;
            }
        }
        seen.try_reserve(1)
            .map_err(|_| BatoError::AllocationFailed {
                bytes: INVENTORY_METADATA,
            })?;
        seen.insert(key, i as u32);
    }
    let mut prefix = Vec::new();
    for (_, p) in &ds {
        super::memory::try_reserve(&mut prefix, 3 + p.len())?;
        prefix.extend_from_slice(&[1]);
        prefix.extend_from_slice(&(p.len() as u16).to_le_bytes());
        prefix.extend_from_slice(p.as_bytes());
    }
    let mut input = ArchiveInput {
        root: Some(&root_guard),
        files: &fs_,
        stage: prefix,
        pos: 0,
        file: None,
        index: 0,
        remaining: 0,
        crc: Crc32c::new(),
        sha: Sha256::new(),
        crc_pending: false,
        crc_bytes: [0; 4],
        ended: false,
        duplicates: &duplicates,
        expected_files: &expected_files,
        started: false,
    };
    let result = consume(&mut input)?;
    root_guard.verify().map_err(err)?;
    Ok(result)
}

pub fn extract_solid_archive<R: Read, P: AsRef<Path>>(
    r: &mut R,
    d: P,
    max: usize,
) -> Result<usize, BatoError> {
    let limits = DecodeLimits {
        max_output_bytes: max as u64,
        ..Default::default()
    };
    extract_solid_archive_with_options(r, d, &limits, Overwrite::Refuse)
}
pub fn extract_solid_archive_with_options<R: Read, P: AsRef<Path>>(
    r: &mut R,
    d: P,
    l: &DecodeLimits,
    overwrite: Overwrite,
) -> Result<usize, BatoError> {
    let options = DecompressionOptions {
        limits: l.clone(),
        ..Default::default()
    };
    extract_solid_archive_with_execution(r, d, overwrite, &options)
}

pub fn extract_solid_archive_with_execution<R: Read, P: AsRef<Path>>(
    r: &mut R,
    d: P,
    overwrite: Overwrite,
    options: &DecompressionOptions,
) -> Result<usize, BatoError> {
    Ok(extract_details(
        r,
        d.as_ref(),
        overwrite,
        (options, &MemberSelection::default()),
    )?
    .entries)
}

/// Verifies a solid directory archive and publishes only the selected members.
pub fn extract_solid_archive_members<R: Read, P: AsRef<Path>>(
    r: &mut R,
    d: P,
    overwrite: Overwrite,
    settings: (&DecompressionOptions, &MemberSelection),
) -> Result<(), BatoError> {
    extract_details(r, d.as_ref(), overwrite, settings)?;
    Ok(())
}

pub(crate) struct Extracted {
    entries: usize,
    bytes: u64,
    metadata: usize,
}

struct CatalogueTarget {
    path: PathBuf,
    existing: Option<Directory>,
    stage: StagingDirectory,
}

impl CatalogueTarget {
    fn open(path: &Path) -> Result<Self, BatoError> {
        let path = absolute_path(path)?;
        let existing = open_destination(&path)?;
        let stage = stage_near(&path)?;
        Ok(Self {
            path,
            existing,
            stage,
        })
    }

    fn publish(&self, overwrite: Overwrite, members: &MemberSelection) -> Result<(), BatoError> {
        publish_tree(
            self.stage.directory(),
            &self.path,
            self.existing.as_ref(),
            (overwrite, members),
        )
    }
}

fn extract_details<R: Read>(
    r: &mut R,
    destination: &Path,
    overwrite: Overwrite,
    (options, members): (&DecompressionOptions, &MemberSelection),
) -> Result<Extracted, BatoError> {
    let l = &options.limits;
    let host = HostBudget::new(l.max_working_memory, &options.execution.stats)?;
    let metadata_limit = l.max_metadata_bytes.min(host.limit() / 4);
    let reserve = metadata_limit.saturating_add(512 * 1024);
    let _metadata = host.reserve(reserve)?;
    let mut stream_limits = l.clone();
    // Materialized file bytes have their own exact limit. The outer record
    // stream also carries bounded path/record metadata, including empty dirs.
    stream_limits.max_output_bytes = l
        .max_output_bytes
        .saturating_add(metadata_limit as u64)
        .saturating_add(5);
    let target = CatalogueTarget::open(destination)?;
    let mut z = DecompressReader::with_budget(
        r,
        &DecompressionOptions {
            limits: stream_limits,
            execution: options.execution.clone(),
        },
        host,
    )?;
    if z.kind() != StreamKind::DirectoryArchive {
        return Err(BatoError::CorruptStream(
            "byte stream is not a directory archive".into(),
        ));
    }
    let mut catalogue_limits = l.clone();
    catalogue_limits.max_metadata_bytes = metadata_limit;
    let extracted = materialize_catalogue(&mut z, target.stage.directory(), &catalogue_limits)?;
    z.finish()?;
    target.publish(overwrite, members)?;
    Ok(extracted)
}

/// Materializes a catalogue whose containing stream has already been verified.
pub fn restore_catalogue(
    catalogue: &mut impl Read,
    destination: &Path,
    overwrite: Overwrite,
    options: &DecompressionOptions,
) -> Result<(), BatoError> {
    restore_catalogue_members(
        catalogue,
        destination,
        overwrite,
        (options, &MemberSelection::default()),
    )
}

/// Publishes selected members from an already verified directory stream.
pub fn restore_catalogue_members(
    catalogue: &mut impl Read,
    destination: &Path,
    overwrite: Overwrite,
    settings: (&DecompressionOptions, &MemberSelection),
) -> Result<(), BatoError> {
    restore_catalogue_details(catalogue, destination, overwrite, settings)?;
    Ok(())
}

pub(crate) fn restore_catalogue_details(
    catalogue: &mut impl Read,
    destination: &Path,
    overwrite: Overwrite,
    (options, members): (&DecompressionOptions, &MemberSelection),
) -> Result<Extracted, BatoError> {
    let host = HostBudget::new(options.limits.max_working_memory, &options.execution.stats)?;
    let mut limits = options.limits.clone();
    limits.max_metadata_bytes = limits.max_metadata_bytes.min(host.limit() / 4);
    let _metadata = host.reserve(limits.max_metadata_bytes.saturating_add(512 * 1024))?;
    let target = CatalogueTarget::open(destination)?;
    let extracted = materialize_catalogue(catalogue, target.stage.directory(), &limits)?;
    target.publish(overwrite, members)?;
    Ok(extracted)
}

fn materialize_catalogue(
    z: &mut impl Read,
    stage: &Directory,
    l: &DecodeLimits,
) -> Result<Extracted, BatoError> {
    let metadata_limit = l.max_metadata_bytes;
    let mut file_paths = Vec::<PathBuf>::new();
    let mut count = 0usize;
    let mut total_output = 0u64;
    let mut metadata_bytes = 0usize;
    let mut seen_paths = HashSet::new();
    loop {
        let mut k = [0];
        catalogue_exact(z, &mut k, BatoError::TruncatedHeader)?;
        if k[0] == 0 {
            break;
        }
        let mut x = [0; 2];
        catalogue_exact(z, &mut x, BatoError::TruncatedHeader)?;
        let n = u16::from_le_bytes(x) as usize;
        if n > l.max_path_bytes {
            return Err(BatoError::DecompressionLimitExceeded {
                size: n,
                limit: l.max_path_bytes,
            });
        }
        if count >= l.max_archive_entries {
            return Err(BatoError::DecompressionLimitExceeded {
                size: count.saturating_add(1),
                limit: l.max_archive_entries,
            });
        }
        // Includes HashSet buckets, file-id paths, vector growth and temporary
        // staging paths, rather than only the serialized UTF-8 bytes.
        metadata_bytes = metadata_bytes.saturating_add(256usize.saturating_add(
            4usize.saturating_mul(n.saturating_add(stage.path().as_os_str().len())),
        ));
        if metadata_bytes > metadata_limit {
            return Err(BatoError::MemoryLimitExceeded {
                required: metadata_bytes,
                limit: metadata_limit,
            });
        }
        let mut pb = try_buffer(n)?;
        catalogue_exact(z, &mut pb, BatoError::TruncatedHeader)?;
        let p = String::from_utf8(pb)
            .map_err(|_| BatoError::CorruptStream("path is not UTF-8".into()))?;
        validate_path(&p)?;
        seen_paths
            .try_reserve(1)
            .map_err(|_| BatoError::AllocationFailed {
                bytes: metadata_limit,
            })?;
        if !seen_paths.insert(p.clone()) {
            return Err(BatoError::CorruptStream("duplicate archive path".into()));
        }
        let out = PathBuf::from(&p);
        if k[0] == 1 {
            stage.descend(&out, IfMissing::Create).map_err(err)?;
            count += 1;
            continue;
        }
        let mut sz = [0; 8];
        catalogue_exact(z, &mut sz, BatoError::TruncatedHeader)?;
        let size = u64::from_le_bytes(sz);
        total_output =
            total_output
                .checked_add(size)
                .ok_or(BatoError::DecompressionLimitExceeded {
                    size: usize::MAX,
                    limit: l.max_output_bytes as usize,
                })?;
        if size > l.max_output_bytes || total_output > l.max_output_bytes {
            return Err(BatoError::DecompressionLimitExceeded {
                size: total_output as usize,
                limit: l.max_output_bytes as usize,
            });
        }
        if k[0] == 2 {
            let mut f = stage.create_file(&out).map_err(err)?;
            let mut left = size;
            let mut crc = Crc32c::new();
            let mut b = [0; COPY_BUFFER];
            while left > 0 {
                let q = left.min(b.len() as u64) as usize;
                catalogue_exact(
                    z,
                    &mut b[..q],
                    BatoError::TruncatedPayload {
                        expected: q,
                        available: 0,
                    },
                )?;
                f.write_all(&b[..q]).map_err(err)?;
                crc.update(&b[..q]);
                left -= q as u64;
            }
            let mut crc_bytes = [0; 4];
            catalogue_exact(z, &mut crc_bytes, BatoError::TruncatedHeader)?;
            let computed = crc.finalize();
            let expected = u32::from_le_bytes(crc_bytes);
            if computed != expected {
                return Err(BatoError::ChecksumMismatch { expected, computed });
            }
            super::memory::try_reserve(&mut file_paths, 1)?;
            file_paths.push(out);
        } else if k[0] == 3 {
            let mut id = [0; 4];
            catalogue_exact(z, &mut id, BatoError::TruncatedHeader)?;
            let src = file_paths
                .get(u32::from_le_bytes(id) as usize)
                .ok_or_else(|| BatoError::CorruptStream("invalid duplicate id".into()))?;
            let mut source = stage.open_file(src).map_err(err)?;
            let sm = source.metadata().map_err(err)?;
            if sm.len() != size {
                return Err(BatoError::CorruptStream("duplicate size mismatch".into()));
            }
            let mut expected_bytes = [0; 4];
            catalogue_exact(z, &mut expected_bytes, BatoError::TruncatedHeader)?;
            let expected = u32::from_le_bytes(expected_bytes);
            let mut target = stage.create_file(&out).map_err(err)?;
            let mut left = size;
            let mut crc = Crc32c::new();
            let mut buffer = [0; COPY_BUFFER];
            while left != 0 {
                let n = left.min(buffer.len() as u64) as usize;
                source.read_exact(&mut buffer[..n]).map_err(err)?;
                target.write_all(&buffer[..n]).map_err(err)?;
                crc.update(&buffer[..n]);
                left -= n as u64;
            }
            let mut extra = [0];
            if source.read(&mut extra).map_err(err)? != 0 {
                return Err(BatoError::CorruptStream("duplicate size changed".into()));
            }
            let computed = crc.finalize();
            if expected != computed {
                return Err(BatoError::ChecksumMismatch { expected, computed });
            }
            super::memory::try_reserve(&mut file_paths, 1)?;
            file_paths.push(out);
        } else {
            return Err(BatoError::CorruptStream("unknown archive record".into()));
        }
        count += 1;
    }
    let mut trailing = [0u8; 1];
    if z.read(&mut trailing).map_err(read_error)? != 0 {
        return Err(BatoError::CorruptStream("trailing archive bytes".into()));
    }
    Ok(Extracted {
        entries: count,
        bytes: total_output,
        metadata: metadata_bytes,
    })
}

fn stage_near(destination: &Path) -> Result<StagingDirectory, BatoError> {
    let mut parent = destination.parent().unwrap_or(destination);
    while !parent.exists() {
        parent = parent
            .parent()
            .ok_or_else(|| BatoError::InvalidOptions("missing destination ancestor".into()))?;
    }
    Ok(StagingDirectory::new(Directory::open(
        parent,
        IfMissing::Fail,
    )?)?)
}

fn open_destination(destination: &Path) -> Result<Option<Directory>, BatoError> {
    match Directory::open(destination, IfMissing::Fail) {
        Ok(dir) => Ok(Some(dir)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn publish_tree(
    stage: &Directory,
    destination: &Path,
    destination_guard: Option<&Directory>,
    (overwrite, members): (Overwrite, &MemberSelection),
) -> Result<(), BatoError> {
    let scope = PublishScope::new(stage, members)?;
    preflight_collisions(stage, destination_guard, overwrite, &scope)?;
    let target = Directory::open(destination, IfMissing::Create)?;
    publish_dir(stage, &target, overwrite, &scope)
}

/// All checksums and catalogue validation finish before publication. Directory
/// descriptors/locked ancestor handles keep every later operation anchored.
fn preflight_collisions(
    src: &Directory,
    dst: Option<&Directory>,
    overwrite: Overwrite,
    scope: &PublishScope<'_>,
) -> Result<(), BatoError> {
    src.verify().map_err(err)?;
    if let Some(dst) = dst {
        dst.verify().map_err(err)?;
    }
    for name in src.names().map_err(err)? {
        let name = name.map_err(err)?;
        if !scope.includes(src, &name)? {
            continue;
        }
        match src.kind(&name).map_err(err)? {
            Kind::Directory => {
                let source = src.child(&name, IfMissing::Fail).map_err(err)?;
                let target = if let Some(dst) = dst {
                    match dst.kind(&name) {
                        Ok(Kind::Directory) => {
                            Some(dst.child(&name, IfMissing::Fail).map_err(err)?)
                        }
                        Ok(_) => {
                            return Err(BatoError::CorruptStream(
                                "destination directory collision".into(),
                            ))
                        }
                        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
                        Err(e) => return Err(err(e)),
                    }
                } else {
                    None
                };
                preflight_collisions(&source, target.as_ref(), overwrite, scope)?;
            }
            Kind::Regular => {
                if let Some(dst) = dst {
                    dst.file_target(&name, overwrite).map_err(err)?;
                }
            }
            Kind::Special => return Err(BatoError::CorruptStream("staged special file".into())),
        }
    }
    Ok(())
}
fn publish_dir(
    src: &Directory,
    dst: &Directory,
    overwrite: Overwrite,
    scope: &PublishScope<'_>,
) -> Result<(), BatoError> {
    for name in src.names().map_err(err)? {
        let name = name.map_err(err)?;
        if !scope.includes(src, &name)? {
            continue;
        }
        match src.kind(&name).map_err(err)? {
            Kind::Directory => publish_dir(
                &src.child(&name, IfMissing::Fail).map_err(err)?,
                &dst.child(&name, IfMissing::Create).map_err(err)?,
                overwrite,
                scope,
            )?,
            Kind::Regular => dst
                .publish_file(src, &name, &name, overwrite)
                .map_err(err)?,
            Kind::Special => return Err(BatoError::CorruptStream("staged special file".into())),
        }
    }
    Ok(())
}

pub(crate) fn validate_path(p: &str) -> Result<(), BatoError> {
    if p.split('/').count() > 128 {
        return Err(BatoError::CorruptStream(
            "archive path nesting exceeds 128 components".into(),
        ));
    }
    if p.is_empty()
        || p.starts_with('/')
        || p.contains('\\')
        || p.contains('\0')
        || p.contains(':')
        || p.split('/').any(|x| x.is_empty() || x == "." || x == "..")
    {
        return Err(BatoError::CorruptStream("unsafe archive path".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn append_after_declared_bytes_is_detected() {
        let t = tempfile::tempdir().unwrap();
        let path = t.path().join("file");
        fs::write(&path, b"original").unwrap();
        let files = vec![(path.clone(), "file".into())];
        let fingerprints = vec![fingerprint_file(&path).unwrap()];
        let mut input = ArchiveInput {
            root: None,
            files: &files,
            stage: Vec::new(),
            pos: 0,
            file: None,
            index: 0,
            remaining: 0,
            crc: Crc32c::new(),
            sha: Sha256::new(),
            crc_pending: false,
            crc_bytes: [0; 4],
            ended: false,
            duplicates: &[None],
            expected_files: &fingerprints,
            started: false,
        };
        let mut b = [0u8; 1024];
        assert!(input.read(&mut b).unwrap() > 0); // file record header
        assert_eq!(input.read(&mut b).unwrap(), 8);
        fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(b"append")
            .unwrap();
        assert!(input.read(&mut b).is_err());
    }
}

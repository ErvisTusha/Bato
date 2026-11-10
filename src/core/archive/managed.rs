//! Editable archive envelopes. Bato streams are compressed before encryption.
//! Encryption provenance: age v1, https://c2sp.org/age.
use super::{
    absolute_path, create_solid_archive, extract_details, member, member_kind, open_destination,
    open_regular_file, publish_tree, stage_near, BatoError, CompressionOptions,
    DecompressionOptions, Directory, HostBudget, IfMissing, Inventory, Kind, MemberSelection,
    Overwrite, INVENTORY_METADATA, MAX_PATH_BYTES,
};
use crate::core::jobs::{Job, Section};
use crate::core::safe_fs::AtomicOutput;
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    collections::BTreeSet,
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Seek, Write},
    path::{Path, PathBuf},
};
use tempfile::{tempdir, tempfile};

mod indexed;
mod inspection;
pub use inspection::{diff, health, inspect, Change};
pub(crate) mod protection;
mod volumes;
use crate::core::recovery::IndexTrust;
use protection::{decrypt, encrypt, password, Password};
use volumes::Volumes;

const WHOLE: [u8; 4] = *b"BAE1";
const SELECTIVE: [u8; 4] = *b"BAP1";
const MAX_SCOPE_BYTES: usize = 1024 * 1024;
const MAX_SCOPES: usize = 4096;

#[derive(Default)]
pub struct ArchiveOptions {
    pub password_file: Option<PathBuf>,
    pub protect: Vec<PathBuf>,
    pub split_size: Option<u64>,
    pub public_only: bool,
    pub parity: bool,
    pub parity_volumes: u8,
    pub indexed: bool,
}

impl std::fmt::Debug for ArchiveOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("ArchiveOptions");
        debug
            .field("password_file", &self.password_file)
            .field("protect", &self.protect)
            .field("split_size", &self.split_size)
            .field("public_only", &self.public_only)
            .field("parity", &self.parity);
        // Existing checkpoints bind to this representation.
        if self.parity_volumes != 0 {
            debug.field("parity_volumes", &self.parity_volumes);
        }
        if self.indexed {
            debug.field("indexed", &self.indexed);
        }
        debug.finish()
    }
}

pub enum Edit<'a> {
    Add(&'a [PathBuf]),
    Remove(&'a [PathBuf]),
}

enum Protection {
    Plain,
    Whole,
    Selected(BTreeSet<PathBuf>),
}

fn invalid(message: impl Into<String>) -> BatoError {
    BatoError::InvalidOptions(message.into())
}

fn scopes(paths: &[PathBuf]) -> Result<BTreeSet<PathBuf>, BatoError> {
    let paths = paths
        .iter()
        .map(|p| member(p).map(PathBuf::from))
        .collect::<Result<BTreeSet<_>, _>>()?;
    if paths.len() > MAX_SCOPES
        || paths.iter().map(|p| p.as_os_str().len()).sum::<usize>() > MAX_SCOPE_BYTES
    {
        return Err(invalid("too many protected paths"));
    }
    Ok(paths)
}

fn covered(path: &Path, roots: &BTreeSet<PathBuf>) -> bool {
    path.ancestors().any(|ancestor| roots.contains(ancestor))
}

fn read_number<const N: usize>(input: &mut impl Read) -> io::Result<[u8; N]> {
    let mut bytes = [0; N];
    input.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn require_eof(input: &mut impl Read) -> Result<(), BatoError> {
    if input.read(&mut [0])? != 0 {
        return Err(BatoError::CorruptStream("trailing archive data".into()));
    }
    Ok(())
}

fn write_scopes(writer: &mut impl Write, roots: &BTreeSet<PathBuf>) -> Result<(), BatoError> {
    writer.write_all(&(roots.len() as u16).to_le_bytes())?;
    for root in roots {
        let name = member(root)?;
        writer.write_all(&(name.len() as u16).to_le_bytes())?;
        writer.write_all(name.as_bytes())?;
    }
    Ok(())
}

fn read_scopes(reader: &mut impl Read) -> Result<BTreeSet<PathBuf>, BatoError> {
    let count = u16::from_le_bytes(read_number(reader)?) as usize;
    if count > MAX_SCOPES {
        return Err(invalid("invalid protected path count"));
    }
    let mut bytes = 0;
    let mut roots = BTreeSet::new();
    for _ in 0..count {
        let length = u16::from_le_bytes(read_number(reader)?) as usize;
        bytes += length;
        if length > MAX_PATH_BYTES || bytes > MAX_SCOPE_BYTES {
            return Err(invalid("protected path list exceeds limit"));
        }
        let mut name = vec![0; length];
        reader.read_exact(&mut name)?;
        let path =
            PathBuf::from(String::from_utf8(name).map_err(|_| invalid("invalid protected path"))?);
        member(&path)?;
        if !roots.insert(path) {
            return Err(invalid("duplicate protected path"));
        }
    }
    Ok(roots)
}

fn inventory<'a>(root: &'a Path) -> Result<Inventory<'a>, BatoError> {
    let mut result = Inventory {
        root,
        excluded: &[],
        dirs: Vec::new(),
        files: Vec::new(),
        charged: 0,
    };
    result.walk(root)?;
    Ok(result)
}

fn check_password_file(file: &File, password: Option<&Password>) -> Result<(), BatoError> {
    if let Some(password) = password {
        if same_file::Handle::from_file(file.try_clone()?)? == password.file {
            return Err(invalid(
                "the password file must be outside the archived tree",
            ));
        }
    }
    Ok(())
}

fn validate_scopes(root: &Path, roots: &BTreeSet<PathBuf>) -> Result<(), BatoError> {
    let directory = Directory::open(root, IfMissing::Fail)?;
    for path in roots {
        member_kind(&directory, path)
            .map_err(|e| invalid(format!("protected path {}: {e}", path.display())))?;
    }
    Ok(())
}

enum ScopeSection {
    Public,
    Private,
}

fn check_section(
    directory: &Directory,
    relative: &Path,
    roots: &BTreeSet<PathBuf>,
    section: &ScopeSection,
) -> Result<(), BatoError> {
    for name in directory.names()? {
        let name = name?;
        let path = relative.join(&name);
        let kind = directory.kind(&name)?;
        let private = covered(&path, roots);
        let valid = match section {
            ScopeSection::Public => !private,
            ScopeSection::Private => {
                private
                    || (kind == Kind::Directory
                        && roots
                            .range(path.clone()..)
                            .next()
                            .is_some_and(|scope| scope.starts_with(&path)))
            }
        };
        if !valid {
            return Err(BatoError::CorruptStream(
                "member does not match protection scopes".into(),
            ));
        }
        if kind == Kind::Directory {
            check_section(
                &directory.child(&name, IfMissing::Fail)?,
                &path,
                roots,
                section,
            )?;
        }
    }
    Ok(())
}

fn partition(
    entries: Inventory<'_>,
    stage: &Path,
    roots: &BTreeSet<PathBuf>,
    pass: &Password,
) -> Result<(PathBuf, PathBuf), BatoError> {
    let public = stage.join("public");
    let private = stage.join("private");
    fs::create_dir(&public)?;
    fs::create_dir(&private)?;
    let target = |path: &Path| {
        if covered(path, roots) {
            &private
        } else {
            &public
        }
    };
    for (_, name) in entries.dirs {
        let path = Path::new(&name);
        fs::create_dir_all(target(path).join(path))?;
    }
    for (src, name) in entries.files {
        let mut input = open_regular_file(&src)?;
        check_password_file(&input, Some(pass))?;
        let path = Path::new(&name);
        let dst = target(path).join(path);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        io::copy(&mut input, &mut File::create_new(dst)?)?;
    }
    Ok((public, private))
}

fn encode_section(
    root: &Path,
    output: &mut impl Write,
    section: Section,
    settings: (&CompressionOptions, Option<&Job>),
) -> Result<(), BatoError> {
    let (options, job) = settings;
    if let Some(job) = job {
        return job.encode_tree(section, root, output, options);
    }
    create_solid_archive(root, output, options)?;
    Ok(())
}

fn write_envelope(
    root: &Path,
    output: &mut impl Write,
    protection: (&Protection, Option<&Password>),
    settings: (&CompressionOptions, Option<&Job>),
) -> Result<(), BatoError> {
    let (options, _) = settings;
    let (mode, pass) = protection;
    if matches!(mode, Protection::Plain) {
        encode_section(root, output, Section::Plain, settings)?;
        return Ok(());
    }
    let pass = pass.ok_or_else(|| invalid("this archive requires --password-file"))?;
    let budget = HostBudget::new(options.memory_budget, &options.execution.stats)?;
    let metadata = budget.reserve(INVENTORY_METADATA)?;
    let entries = inventory(root)?;
    match mode {
        Protection::Whole => {
            for (path, _) in &entries.files {
                check_password_file(&open_regular_file(path)?, Some(pass))?;
            }
            drop(entries);
            drop(metadata);
            output.write_all(&WHOLE)?;
            let mut encrypted = encrypt(output, pass, options)?;
            encode_section(root, &mut encrypted, Section::Private, settings)?;
            encrypted.finish()?;
        }
        Protection::Selected(roots) => {
            validate_scopes(root, roots)?;
            let stage = tempdir()?;
            let (public, private) = partition(entries, stage.path(), roots, pass)?;
            drop(metadata);
            let mut public_stream = tempfile()?;
            encode_section(&public, &mut public_stream, Section::Public, settings)?;
            let len = public_stream.stream_position()?;
            public_stream.rewind()?;
            output.write_all(&SELECTIVE)?;
            output.write_all(&len.to_le_bytes())?;
            let mut digest = Sha256::new();
            io::copy(
                &mut public_stream,
                &mut HashWriter {
                    output,
                    digest: &mut digest,
                },
            )?;
            let mut encrypted = encrypt(output, pass, options)?;
            encrypted.write_all(&digest.finalize())?;
            write_scopes(&mut encrypted, roots)?;
            encode_section(&private, &mut encrypted, Section::Private, settings)?;
            encrypted.finish()?;
        }
        Protection::Plain => unreachable!(),
    }
    Ok(())
}

struct HashWriter<'a, W> {
    output: &'a mut W,
    digest: &'a mut Sha256,
}

impl<W: Write> Write for HashWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let written = self.output.write(bytes)?;
        self.digest.update(&bytes[..written]);
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.output.flush()
    }
}

struct HashReader<'a, R> {
    input: R,
    digest: &'a mut Sha256,
}

impl<R: Read> Read for HashReader<'_, R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let n = self.input.read(bytes)?;
        self.digest.update(&bytes[..n]);
        Ok(n)
    }
}

fn decode_section(
    input: &mut impl Read,
    destination: &Path,
    (section, overwrite, members): (Section, Overwrite, &MemberSelection),
    (options, job): (&DecompressionOptions, Option<&Job>),
) -> Result<super::Extracted, BatoError> {
    match job {
        Some(job) => job.extract_tree(section, input, destination, (overwrite, options, members)),
        None => extract_details(input, destination, overwrite, (options, members)),
    }
}

fn extract_envelope(
    input: &mut (impl Read + Seek),
    destination: &Path,
    (overwrite, members): (Overwrite, &MemberSelection),
    access: (Option<&Password>, bool),
    (options, job): (&DecompressionOptions, Option<&Job>),
) -> Result<Protection, BatoError> {
    let (pass, public_only) = access;
    let magic = read_number(input)?;
    if let Some(trust) = index_trust(magic) {
        if public_only {
            return Err(invalid("indexed archives have no separate public section"));
        }
        let mode = match trust {
            IndexTrust::Authenticated => Protection::Whole,
            IndexTrust::CheckStream => Protection::Plain,
        };
        indexed::read(
            input,
            indexed::Operation::Extract {
                destination,
                overwrite,
                members,
                job,
            },
            (pass, trust),
            options,
        )?;
        return Ok(mode);
    }
    if !matches!(magic, WHOLE | SELECTIVE) {
        if public_only {
            return Err(invalid(
                "--public-only requires a selectively protected archive",
            ));
        }
        input.rewind()?;
        decode_section(
            input,
            destination,
            (Section::Plain, overwrite, members),
            (options, job),
        )?;
        return Ok(Protection::Plain);
    }
    if magic == WHOLE && public_only {
        return Err(invalid("this archive has no public section"));
    }
    if !public_only && pass.is_none() {
        return Err(invalid("this archive requires --password-file"));
    }
    let destination = absolute_path(destination)?;
    let destination_guard = open_destination(&destination)?;
    let stage = stage_near(&destination)?;
    let combined = stage.directory().path().join("contents");
    let all_members = &MemberSelection::default();
    let mut remaining = options.clone();
    let mut public_digest = Sha256::new();
    if magic == SELECTIVE {
        let length = u64::from_le_bytes(read_number(input)?);
        let mut bounded = input.take(length);
        let public = decode_section(
            &mut HashReader {
                input: &mut bounded,
                digest: &mut public_digest,
            },
            &combined,
            (Section::Public, Overwrite::Refuse, all_members),
            (options, job),
        )?;
        if bounded.limit() != 0 {
            return Err(invalid("invalid public stream length"));
        }
        remaining.limits.max_output_bytes -= public.bytes;
        remaining.limits.max_archive_entries -= public.entries;
        remaining.limits.max_metadata_bytes = remaining
            .limits
            .max_metadata_bytes
            .saturating_sub(public.metadata);
    }
    let protection = if public_only {
        Protection::Plain
    } else {
        let pass = pass.ok_or_else(|| invalid("this archive requires --password-file"))?;
        let header_limit = Cell::new(None);
        let mut plaintext = decrypt(&mut *input, &header_limit, pass, &remaining)?;
        if magic == WHOLE {
            decode_section(
                &mut plaintext,
                &combined,
                (Section::Private, Overwrite::Refuse, all_members),
                (&remaining, job),
            )?;
            require_eof(&mut plaintext)?;
            drop(plaintext);
            require_eof(input)?;
            Protection::Whole
        } else {
            let expected = read_number::<32>(&mut plaintext)?;
            if expected[..] != public_digest.finalize()[..] {
                return Err(BatoError::CorruptStream(
                    "public archive SHA-256 mismatch".into(),
                ));
            }
            let roots = read_scopes(&mut plaintext)?;
            // Keep the sections separate until authentication and scope
            // validation finish, so an edit cannot reclassify private data.
            let private = stage.directory().path().join("protected");
            decode_section(
                &mut plaintext,
                &private,
                (Section::Private, Overwrite::Refuse, all_members),
                (&remaining, job),
            )?;
            require_eof(&mut plaintext)?;
            drop(plaintext);
            require_eof(input)?;
            validate_scopes(&private, &roots)?;
            let public_tree = Directory::open(&combined, IfMissing::Fail)?;
            let private_tree = Directory::open(&private, IfMissing::Fail)?;
            check_section(&public_tree, Path::new(""), &roots, &ScopeSection::Public)?;
            check_section(&private_tree, Path::new(""), &roots, &ScopeSection::Private)?;
            publish_tree(
                &private_tree,
                &combined,
                Some(&public_tree),
                (Overwrite::Refuse, all_members),
            )?;
            Protection::Selected(roots)
        }
    };
    let combined = Directory::open(&combined, IfMissing::Fail)?;
    publish_tree(
        &combined,
        &destination,
        destination_guard.as_ref(),
        (overwrite, members),
    )?;
    Ok(protection)
}

struct ArchiveFile {
    path: PathBuf,
    index: PathBuf,
    file: File,
    split: Option<Volumes>,
}

impl ArchiveFile {
    fn open(path: &Path, operation: Access) -> Result<Self, BatoError> {
        let path = absolute_path(path)?;
        let is_split = fs::symlink_metadata(&path)?.is_dir();
        let (index, file) = if is_split {
            volumes::open_index(&path)?
        } else {
            (path.clone(), open_regular_file(&path)?)
        };
        match operation {
            Access::Edit => file.lock()?,
            Access::Read => file.lock_shared()?,
        }
        ensure_current(&index, &file)?;
        let split = if is_split {
            Some(Volumes::read(&file)?)
        } else {
            None
        };
        Ok(Self {
            path,
            index,
            file,
            split,
        })
    }

    fn stream(&self) -> Result<File, BatoError> {
        match &self.split {
            Some(volumes) => volumes.join(&self.path),
            None => Ok(self.file.try_clone()?),
        }
    }
}

enum Access {
    Edit,
    Read,
}

fn ensure_current(path: &Path, original: &File) -> Result<(), BatoError> {
    let old = same_file::Handle::from_file(original.try_clone()?)?;
    let current = same_file::Handle::from_file(open_regular_file(path)?)?;
    if old != current {
        return Err(invalid(
            "archive changed while being opened; retry the operation",
        ));
    }
    Ok(())
}

fn index_trust(magic: [u8; 4]) -> Option<IndexTrust> {
    match magic {
        indexed::PLAIN => Some(IndexTrust::CheckStream),
        indexed::ENCRYPTED => Some(IndexTrust::Authenticated),
        _ => None,
    }
}

pub fn is_managed_prefix(prefix: &[u8]) -> bool {
    prefix.starts_with(&WHOLE)
        || prefix.starts_with(&SELECTIVE)
        || prefix.starts_with(&indexed::PLAIN)
        || prefix.starts_with(&indexed::ENCRYPTED)
}

pub fn create(
    source: &Path,
    target: &Path,
    overwrite: Overwrite,
    features: &ArchiveOptions,
    options: &CompressionOptions,
) -> Result<(), BatoError> {
    create_with_job(source, target, overwrite, (features, options, None))
}

pub fn create_with_job(
    source: &Path,
    target: &Path,
    overwrite: Overwrite,
    (features, options, job): (&ArchiveOptions, &CompressionOptions, Option<&Job>),
) -> Result<(), BatoError> {
    let source = absolute_path(source)?;
    let target = absolute_path(target)?;
    if target.starts_with(&source) {
        return Err(invalid(
            "place managed archive output outside the source tree",
        ));
    }
    let pass = features
        .password_file
        .as_deref()
        .map(password)
        .transpose()?;
    let protection = match (pass.as_ref(), scopes(&features.protect)?) {
        (None, roots) if roots.is_empty() => Protection::Plain,
        (None, _) => return Err(invalid("--protect requires --password-file")),
        (Some(_), roots) if roots.is_empty() => Protection::Whole,
        (Some(_), roots) => Protection::Selected(roots),
    };
    if features.indexed {
        if features.split_size.is_some() || !features.protect.is_empty() || job.is_some() {
            return Err(invalid("--indexed cannot be combined with split, selective protection or checkpointed creation"));
        }
        return write_indexed(&source, &target, (overwrite, pass.as_ref()), options);
    }
    if let Some(limit) = features.split_size {
        let mut archive = tempfile()?;
        write_buffered(
            &source,
            &mut archive,
            (&protection, pass.as_ref()),
            (options, job),
        )?;
        Volumes::create(
            &target,
            limit,
            &mut archive,
            features.parity_volumes.max(u8::from(features.parity)),
        )
    } else {
        let staged = AtomicOutput::new(&target, overwrite, features.password_file.as_deref())?;
        write_buffered(
            &source,
            &mut staged.writer()?,
            (&protection, pass.as_ref()),
            (options, job),
        )?;
        staged.publish()?;
        Ok(())
    }
}

fn write_indexed(
    source: &Path,
    target: &Path,
    (overwrite, pass): (Overwrite, Option<&Password>),
    options: &CompressionOptions,
) -> Result<(), BatoError> {
    // Mirror the exclusion used by the other encrypted archive writers.
    if let Some(password) = pass {
        let budget = HostBudget::new(options.memory_budget, &options.execution.stats)?;
        let _metadata = budget.reserve(INVENTORY_METADATA)?;
        for (path, _) in inventory(source)?.files {
            check_password_file(&open_regular_file(&path)?, Some(password))?;
        }
    }
    let staged = AtomicOutput::new(target, overwrite, None)?;
    let mut output = BufWriter::new(staged.writer()?);
    indexed::write(source, &mut output, pass, options)?;
    output.flush()?;
    output.get_ref().sync_all()?;
    drop(output);
    staged.publish()?;
    Ok(())
}

fn write_buffered(
    source: &Path,
    file: &mut File,
    protection: (&Protection, Option<&Password>),
    options: (&CompressionOptions, Option<&Job>),
) -> Result<(), BatoError> {
    let mut output = BufWriter::new(file);
    write_envelope(source, &mut output, protection, options)?;
    output.flush()?;
    output.get_ref().sync_all()?;
    Ok(())
}

pub fn extract(
    source: &Path,
    destination: &Path,
    overwrite: Overwrite,
    access: (&ArchiveOptions, &DecompressionOptions),
) -> Result<(), BatoError> {
    extract_with_job(source, destination, overwrite, (access.0, access.1, None))
}

pub fn extract_with_job(
    source: &Path,
    destination: &Path,
    overwrite: Overwrite,
    (features, options, job): (&ArchiveOptions, &DecompressionOptions, Option<&Job>),
) -> Result<(), BatoError> {
    extract_members(
        source,
        destination,
        (overwrite, &MemberSelection::default()),
        (features, options, job),
    )
}

/// Extracts the requested members of a split or protected archive.
pub fn extract_members(
    source: &Path,
    destination: &Path,
    output: (Overwrite, &MemberSelection),
    (features, options, job): (&ArchiveOptions, &DecompressionOptions, Option<&Job>),
) -> Result<(), BatoError> {
    let archive = ArchiveFile::open(source, Access::Read)?;
    let destination = absolute_path(destination)?;
    if destination.starts_with(&archive.path) {
        return Err(invalid("extraction destination overlaps the archive"));
    }
    let pass = features
        .password_file
        .as_deref()
        .map(password)
        .transpose()?;
    extract_envelope(
        &mut BufReader::new(archive.stream()?),
        &destination,
        output,
        (pass.as_ref(), features.public_only),
        (options, job),
    )?;
    Ok(())
}

pub fn repair_split(source: &Path, target: &Path) -> Result<(), BatoError> {
    let archive = ArchiveFile::open(source, Access::Read)?;
    let target = absolute_path(target)?;
    if target.starts_with(&archive.path) {
        return Err(invalid(
            "repair destination must be outside the split archive",
        ));
    }
    let split = archive
        .split
        .ok_or_else(|| invalid("expected a split archive"))?;
    split.repair(&archive.path, &target)
}

pub fn edit(
    source: &Path,
    edit: Edit<'_>,
    overwrite: Overwrite,
    settings: (&ArchiveOptions, &CompressionOptions, &DecompressionOptions),
) -> Result<Option<io::Error>, BatoError> {
    let (features, compression, decompression) = settings;
    let archive = ArchiveFile::open(source, Access::Edit)?;
    let pass = features
        .password_file
        .as_deref()
        .map(password)
        .transpose()?;
    check_password_file(&archive.file, pass.as_ref())?;
    let stage = tempdir()?;
    let tree = stage.path().join("contents");
    let mut input = BufReader::new(archive.stream()?);
    let indexed = index_trust(read_number(&mut input)?).is_some();
    input.rewind()?;
    if indexed && !features.protect.is_empty() {
        return Err(invalid("indexed archives cannot use selective protection"));
    }
    let mut protection = extract_envelope(
        &mut input,
        &tree,
        (Overwrite::Refuse, &MemberSelection::default()),
        (pass.as_ref(), false),
        (decompression, None),
    )?;
    match edit {
        Edit::Add(paths) => {
            let identity = same_file::Handle::from_file(archive.file.try_clone()?)?;
            let guard = CopyGuard {
                archive: &identity,
                password: pass.as_ref(),
            };
            for path in paths {
                add_path(path, &tree, &archive.path, (overwrite, &guard))?;
            }
        }
        Edit::Remove(paths) => {
            let removals = paths
                .iter()
                .map(|path| member(path).map(PathBuf::from))
                .collect::<Result<BTreeSet<_>, _>>()?;
            let directory = Directory::open(&tree, IfMissing::Fail)?;
            for path in &removals {
                if path
                    .ancestors()
                    .skip(1)
                    .any(|parent| removals.contains(parent))
                {
                    continue;
                }
                let kind = member_kind(&directory, path)?;
                let path = tree.join(path);
                if kind == Kind::Directory {
                    fs::remove_dir_all(path)?;
                } else {
                    fs::remove_file(path)?;
                }
            }
            if let Protection::Selected(roots) = &mut protection {
                roots.retain(|path| !covered(path, &removals));
            }
        }
    }
    let new_roots = scopes(&features.protect)?;
    match &mut protection {
        Protection::Plain if pass.is_some() && new_roots.is_empty() => {
            protection = Protection::Whole
        }
        Protection::Plain if !new_roots.is_empty() && pass.is_some() => {
            protection = Protection::Selected(new_roots);
        }
        Protection::Selected(roots) => {
            roots.extend(new_roots);
            if roots.len() > MAX_SCOPES
                || roots.iter().map(|p| p.as_os_str().len()).sum::<usize>() > MAX_SCOPE_BYTES
            {
                return Err(invalid("too many protected paths"));
            }
        }
        _ if !new_roots.is_empty() => {
            return Err(invalid(
                "--protect requires a password and is unnecessary for a fully encrypted archive",
            ))
        }
        _ => {}
    }
    if let Some(split) = &archive.split {
        let mut updated = tempfile()?;
        write_buffered(
            &tree,
            &mut updated,
            (&protection, pass.as_ref()),
            (compression, None),
        )?;
        return split.replace(&archive.path, &mut updated, || {
            ensure_current(&archive.index, &archive.file)
        });
    }
    let staged = AtomicOutput::new(&archive.path, Overwrite::Replace, None)?;
    if indexed {
        let mut output = BufWriter::new(staged.writer()?);
        indexed::write(&tree, &mut output, pass.as_ref(), compression)?;
        output.flush()?;
        output.get_ref().sync_all()?;
    } else {
        write_buffered(
            &tree,
            &mut staged.writer()?,
            (&protection, pass.as_ref()),
            (compression, None),
        )?;
    }
    ensure_current(&archive.index, &archive.file)?;
    staged.publish()?;
    Ok(None)
}

struct CopyGuard<'a> {
    archive: &'a same_file::Handle,
    password: Option<&'a Password>,
}

fn add_path(
    source: &Path,
    tree: &Path,
    archive: &Path,
    settings: (Overwrite, &CopyGuard<'_>),
) -> Result<(), BatoError> {
    let source = absolute_path(source)?;
    if archive.starts_with(&source) || source.starts_with(archive) {
        return Err(invalid("added path overlaps the archive"));
    }
    let name = source
        .file_name()
        .ok_or_else(|| invalid("added path has no filename"))?;
    member(Path::new(name))?;
    let parent = source
        .parent()
        .ok_or_else(|| invalid("added path has no parent"))?;
    let parent = Directory::open(parent, IfMissing::Fail)?;
    let target = Directory::open(tree, IfMissing::Fail)?;
    let name = name
        .to_str()
        .ok_or_else(|| invalid("archive paths must be UTF-8"))?;
    copy_member(&parent, &target, name, settings)
}

fn copy_member(
    source: &Directory,
    target: &Directory,
    relative: &str,
    settings: (Overwrite, &CopyGuard<'_>),
) -> Result<(), BatoError> {
    // The relative path also bounds recursion to the archive's path limits.
    let relative = Path::new(relative);
    let archive_name = member(relative)?;
    let (overwrite, guard) = settings;
    let name = relative
        .file_name()
        .ok_or_else(|| invalid("missing member name"))?;
    match source.kind(name)? {
        Kind::Directory => {
            let source = source.child(name, IfMissing::Fail)?;
            let target = target.child(name, IfMissing::Create)?;
            for child in source.names()? {
                let child = child?;
                let child = child
                    .to_str()
                    .ok_or_else(|| invalid("archive paths must be UTF-8"))?;
                copy_member(
                    &source,
                    &target,
                    &format!("{archive_name}/{child}"),
                    settings,
                )?;
            }
        }
        Kind::Regular => {
            let mut input = source.open_file(Path::new(name))?;
            if &same_file::Handle::from_file(input.try_clone()?)? == guard.archive {
                return Err(invalid("cannot add an archive to itself"));
            }
            check_password_file(&input, guard.password)?;
            if target.file_target(name, overwrite)? {
                // This destination is inside the private, fully staged edit tree.
                fs::remove_file(target.path().join(name))?;
            }
            io::copy(&mut input, &mut target.create_file(Path::new(name))?)?;
        }
        Kind::Special => return Err(invalid("cannot add symlinks or special files")),
    }
    Ok(())
}

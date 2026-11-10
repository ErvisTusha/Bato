//! Durable checkpoints for seekable Bato jobs. Only complete segments commit.
use super::{
    archive::{restore_catalogue_details, stage_catalogue, Extracted, MemberSelection},
    frame::{
        compress_checkpointed, decompress_checkpointed, read_stream_header, SegmentCheckpoint,
    },
    safe_fs::{absolute_path, open_regular_file, AtomicOutput, Directory, IfMissing},
    types::{BatoError, CompressionOptions, DecompressionOptions, Overwrite, StreamKind},
};
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    fs::{self, File, OpenOptions},
    io::{self, BufReader, Read, Seek, Write},
    path::{Path, PathBuf},
};

mod sealed;

const MAGIC: &[u8; 4] = b"BAJ2";
const STATE_BYTES: usize = 4 + 32 + 8 + 8 + 4 + 32 + 8;

#[derive(Clone, Copy)]
pub enum Start {
    New,
    Resume,
}

pub struct Job {
    directory: Directory,
    _lock: File,
    remaining: Cell<Option<usize>>,
    pause_file: Option<PathBuf>,
    paused: Cell<bool>,
    key: Option<age::x25519::Identity>,
    progress: bool,
}

fn invalid(message: &str) -> BatoError {
    BatoError::InvalidOptions(message.into())
}

impl Job {
    pub fn open(
        path: &Path,
        start: Start,
        identity: [u8; 32],
        pause: (Option<usize>, Option<PathBuf>),
    ) -> Result<Self, BatoError> {
        let path = absolute_path(path)?;
        if matches!(start, Start::New) {
            let parent = path
                .parent()
                .ok_or_else(|| invalid("checkpoint needs a parent"))?;
            let parent = Directory::open(parent, IfMissing::Create)?;
            parent.verify()?;
            let mut builder = fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            builder.create(&path)?;
            #[cfg(unix)]
            parent.file().sync_all()?;
        }
        let directory = Directory::open(&path, IfMissing::Fail)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let metadata = directory.file().metadata()?;
            if metadata.mode() & 0o077 != 0 || metadata.uid() != effective_uid() {
                return Err(invalid(
                    "checkpoint directory must be private and owned by the current user",
                ));
            }
        }
        let lock = match start {
            Start::New => {
                let mut file = directory.create_file(Path::new("job"))?;
                file.write_all(MAGIC)?;
                file.write_all(&identity)?;
                file.sync_all()?;
                file
            }
            Start::Resume => directory.open_file(Path::new("job"))?,
        };
        lock.try_lock().map_err(io::Error::from)?;
        if matches!(start, Start::Resume) {
            let mut reader = &lock;
            let mut expected = [0; 36];
            reader.read_exact(&mut expected)?;
            if &expected[..4] != MAGIC || expected[4..] != identity || reader.read(&mut [0])? != 0 {
                return Err(invalid(
                    "checkpoint belongs to a different operation, path, or settings",
                ));
            }
        }
        Ok(Self {
            directory,
            _lock: lock,
            remaining: Cell::new(pause.0),
            pause_file: pause.1,
            paused: Cell::new(false),
            key: None,
            progress: false,
        })
    }

    pub fn enable_progress(&mut self) {
        self.progress = true;
    }

    pub fn protect(
        &mut self,
        password_file: &Path,
        options: (&CompressionOptions, &DecompressionOptions),
    ) -> Result<(), BatoError> {
        self.key = Some(sealed::open_key(&self.directory, password_file, options)?);
        Ok(())
    }

    fn key(&self, section: Section) -> Result<Option<&age::x25519::Identity>, BatoError> {
        match section {
            Section::Private => {
                Ok(Some(self.key.as_ref().ok_or_else(|| {
                    invalid("private checkpoint requires a password")
                })?))
            }
            _ => Ok(None),
        }
    }

    pub fn paused(&self) -> bool {
        self.paused.get()
    }

    /// Removes only names Bato owns. Retain unrelated directory entries.
    pub fn complete(self) -> Result<(), BatoError> {
        self.directory.verify()?;
        for name in ["plain", "public", "private"] {
            for ext in ["state", "data"] {
                remove_if_present(&self.directory.path().join(format!("{name}.{ext}")))?;
            }
        }
        let path = self.directory.path().to_path_buf();
        remove_if_present(&path.join(sealed::KEY))?;
        // Delete the marker last, while holding the job lock.
        remove_if_present(&path.join("job"))?;
        drop(self);
        fs::remove_dir(path)?;
        Ok(())
    }

    pub fn encode(
        &self,
        section: Section,
        input: &mut File,
        output: &mut dyn Write,
        settings: (StreamKind, &CompressionOptions),
    ) -> Result<(), BatoError> {
        let (kind, options) = settings;
        let (mut part, resume) = self.prepare(section, input, Direction::Encode)?;
        compress_checkpointed(
            input,
            &mut part.data,
            options,
            kind,
            (resume, &mut |file, state| {
                self.save(
                    section,
                    file,
                    state,
                    (part.input, Direction::Encode, &mut part.sealed),
                )
            }),
        )?;
        part.data.file.rewind()?;
        io::copy(&mut part.data.file, output)?;
        Ok(())
    }

    pub fn encode_tree(
        &self,
        section: Section,
        source: &Path,
        output: &mut dyn Write,
        options: &CompressionOptions,
    ) -> Result<(), BatoError> {
        let mut catalogue = tempfile::tempfile()?;
        stage_catalogue(source, &mut catalogue, options)?;
        self.encode(
            section,
            &mut catalogue,
            output,
            (StreamKind::DirectoryArchive, options),
        )
    }

    pub fn decode(
        &self,
        section: Section,
        input: &mut File,
        options: &DecompressionOptions,
    ) -> Result<(File, StreamKind), BatoError> {
        input.rewind()?;
        let kind = read_stream_header(input)?.kind;
        let mut limits = options.clone();
        if kind == StreamKind::DirectoryArchive {
            limits.limits.max_output_bytes = limits
                .limits
                .max_output_bytes
                .saturating_add(limits.limits.max_metadata_bytes as u64)
                .saturating_add(5);
        }
        let (mut part, resume) = self.prepare(section, input, Direction::Decode)?;
        let mut source = BufReader::new(input);
        decompress_checkpointed(
            &mut source,
            &mut part.data,
            &limits,
            (resume, &mut |file, state| {
                self.save(
                    section,
                    file,
                    state,
                    (part.input, Direction::Decode, &mut part.sealed),
                )
            }),
        )?;
        part.data.file.rewind()?;
        Ok((part.data.file, kind))
    }

    pub(crate) fn extract_tree(
        &self,
        section: Section,
        input: &mut impl Read,
        destination: &Path,
        (overwrite, options, members): (Overwrite, &DecompressionOptions, &MemberSelection),
    ) -> Result<Extracted, BatoError> {
        // Copy through EOF first. For age readers this authenticates the full
        // envelope before any decrypted codec state is checkpointed.
        let mut stream = tempfile::tempfile()?;
        io::copy(input, &mut stream)?;
        let (mut raw, kind) = self.decode(section, &mut stream, options)?;
        if kind != StreamKind::DirectoryArchive {
            return Err(invalid("expected a directory archive"));
        }
        restore_catalogue_details(&mut raw, destination, overwrite, (options, members))
    }

    fn prepare(
        &self,
        section: Section,
        input: &mut File,
        direction: Direction,
    ) -> Result<(Part, Option<SegmentCheckpoint>), BatoError> {
        let input_hash = fingerprint(input)?;
        let name = section.name();
        let state_path = self.directory.path().join(format!("{name}.state"));
        let data_name = format!("{name}.data");
        let path = self.directory.path().join(&data_name);
        let key = self.key(section)?;
        let saved = match open_regular_file(&state_path) {
            Ok(file) => {
                let state = if let Some(key) = key {
                    if file.metadata()?.len() > 4096 {
                        return Err(invalid("invalid protected checkpoint state size"));
                    }
                    load(&mut sealed::decrypt(file, key)?, input_hash)?
                } else {
                    load(&mut &file, input_hash)?
                };
                Some(state)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e.into()),
        };
        let file = match self.directory.open_file(Path::new(&data_name)) {
            Ok(original) => reopen(&path, original)?,
            Err(e) if e.kind() == io::ErrorKind::NotFound && saved.is_none() => {
                drop(self.directory.create_file(Path::new(&data_name))?);
                reopen(&path, self.directory.open_file(Path::new(&data_name))?)?
            }
            Err(e) => return Err(e.into()),
        };
        self.directory.verify()?;
        let (mut data, mut sealed_data) = if key.is_some() {
            (
                DigestWriter {
                    file: tempfile::tempfile()?,
                    digest: Sha256::new(),
                },
                Some(sealed::Sealed { file, plain: 0 }),
            )
        } else {
            (
                DigestWriter {
                    file,
                    digest: Sha256::new(),
                },
                None,
            )
        };
        let mut length = 0;
        if let Some((state, expected, encrypted)) = saved {
            length = direction.output(state);
            if direction.input(state) > input.metadata()?.len() {
                return Err(invalid("checkpoint offset is outside the source"));
            }
            if let (Some(key), Some(sealed_data)) = (key, &mut sealed_data) {
                sealed::restore(&mut sealed_data.file, &mut data, (encrypted, length), key)?;
                sealed_data.file.set_len(encrypted)?;
                sealed_data.plain = length;
            } else if encrypted != 0
                || io::copy(&mut (&mut data.file).take(length), &mut data.digest)? != length
            {
                return Err(invalid("invalid checkpoint data length"));
            }
            if data.digest.clone().finalize()[..] != expected {
                return Err(invalid("checkpoint data failed SHA-256 verification"));
            }
        } else if let Some(sealed_data) = &sealed_data {
            sealed_data.file.set_len(0)?;
        }
        data.file.set_len(length)?;
        Ok((
            Part {
                input: input_hash,
                data,
                sealed: sealed_data,
            },
            saved.map(|(state, _, _)| state),
        ))
    }

    fn save(
        &self,
        section: Section,
        data: &mut DigestWriter,
        state: SegmentCheckpoint,
        (input, direction, sealed_data): ([u8; 32], Direction, &mut Option<sealed::Sealed>),
    ) -> Result<(), BatoError> {
        if data.file.stream_position()? != direction.output(state) {
            return Err(invalid("checkpoint output offset mismatch"));
        }
        let key = self.key(section)?;
        let encrypted = match (key, sealed_data) {
            (Some(key), Some(sealed_data)) => {
                sealed_data.append(&mut data.file, direction.output(state), key)?
            }
            _ => {
                data.file.sync_all()?;
                0
            }
        };
        self.directory.verify()?;
        let staged = AtomicOutput::new(
            &self
                .directory
                .path()
                .join(format!("{}.state", section.name())),
            Overwrite::Replace,
            None,
        )?;
        let mut state_file = staged.writer()?;
        let mut record = Vec::with_capacity(STATE_BYTES);
        record.extend_from_slice(MAGIC);
        record.extend_from_slice(&input);
        record.extend_from_slice(&state.decoded.to_le_bytes());
        record.extend_from_slice(&state.encoded.to_le_bytes());
        record.extend_from_slice(&state.crc32c.to_le_bytes());
        record.extend_from_slice(&data.digest.clone().finalize());
        record.extend_from_slice(&encrypted.to_le_bytes());
        if let Some(key) = key {
            let mut encrypted = sealed::encrypt(&mut state_file, key)?;
            encrypted.write_all(&record)?;
            encrypted.finish()?;
        } else {
            state_file.write_all(&record)?;
        }
        state_file.sync_all()?;
        drop(state_file);
        staged.publish()?;
        #[cfg(unix)]
        self.directory.file().sync_all()?;
        if self.progress {
            eprintln!(
                "{} {}: committed {} input bytes, {} output bytes",
                direction.label(),
                section.name(),
                direction.input(state),
                direction.output(state)
            );
        }
        let reached_limit = self.remaining.get().is_some_and(|remaining| {
            self.remaining.set(Some(remaining - 1));
            remaining == 1
        });
        let requested = self
            .pause_file
            .as_ref()
            .map(|path| path.try_exists())
            .transpose()?
            .unwrap_or(false);
        if reached_limit || requested {
            self.paused.set(true);
            return Err(BatoError::Io(io::Error::other(
                "paused at a verified segment boundary",
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub enum Section {
    Plain,
    Public,
    Private,
}

impl Section {
    fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Public => "public",
            Self::Private => "private",
        }
    }
}

#[derive(Clone, Copy)]
enum Direction {
    Encode,
    Decode,
}

impl Direction {
    fn label(self) -> &'static str {
        match self {
            Self::Encode => "compress",
            Self::Decode => "extract",
        }
    }

    fn output(self, state: SegmentCheckpoint) -> u64 {
        match self {
            Self::Encode => state.encoded,
            Self::Decode => state.decoded,
        }
    }
    fn input(self, state: SegmentCheckpoint) -> u64 {
        match self {
            Self::Encode => state.decoded,
            Self::Decode => state.encoded,
        }
    }
}

struct Part {
    input: [u8; 32],
    data: DigestWriter,
    sealed: Option<sealed::Sealed>,
}

struct DigestWriter {
    file: File,
    digest: Sha256,
}

impl Write for DigestWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let n = self.file.write(data)?;
        self.digest.update(&data[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

fn load(
    file: &mut impl Read,
    input: [u8; 32],
) -> Result<(SegmentCheckpoint, [u8; 32], u64), BatoError> {
    let mut bytes = [0; STATE_BYTES];
    file.read_exact(&mut bytes)?;
    if &bytes[..4] != MAGIC || bytes[4..36] != input || file.read(&mut [0])? != 0 {
        return Err(invalid(
            "checkpoint input has changed or its state is invalid",
        ));
    }
    let mut data = &bytes[36..];
    let mut read_u64 = || -> io::Result<u64> {
        let mut value = [0; 8];
        data.read_exact(&mut value)?;
        Ok(u64::from_le_bytes(value))
    };
    let decoded = read_u64()?;
    let encoded = read_u64()?;
    let mut crc = [0; 4];
    let mut digest = [0; 32];
    data.read_exact(&mut crc)?;
    data.read_exact(&mut digest)?;
    let mut encrypted = [0; 8];
    data.read_exact(&mut encrypted)?;
    Ok((
        SegmentCheckpoint {
            decoded,
            encoded,
            crc32c: u32::from_le_bytes(crc),
        },
        digest,
        u64::from_le_bytes(encrypted),
    ))
}

pub(crate) fn fingerprint(input: &mut (impl Read + Seek)) -> io::Result<[u8; 32]> {
    input.rewind()?;
    let mut digest = Sha256::new();
    io::copy(input, &mut digest)?;
    input.rewind()?;
    Ok(digest.finalize().into())
}

fn reopen(path: &Path, original: File) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        options.custom_flags(windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let file = options.open(path)?;
    if same_file::Handle::from_file(original)? != same_file::Handle::from_file(file.try_clone()?)? {
        return Err(io::Error::other("checkpoint file changed while opening"));
    }
    Ok(file)
}

fn remove_if_present(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        result => result,
    }
}

#[cfg(unix)]
fn effective_uid() -> u32 {
    // SAFETY: geteuid takes no arguments and has no memory preconditions.
    unsafe { libc::geteuid() }
}

/// Report committed checkpoint offsets without acquiring the writer lock.
pub fn status(path: &Path) -> Result<Vec<String>, BatoError> {
    let directory = Directory::open(path, IfMissing::Fail)?;
    let mut marker = directory.open_file(Path::new("job"))?;
    let mut identity = [0; 36];
    marker.read_exact(&mut identity)?;
    if &identity[..4] != MAGIC || marker.read(&mut [0])? != 0 {
        return Err(invalid("invalid checkpoint directory"));
    }
    let mut rows = Vec::new();
    for section in ["plain", "public", "private"] {
        let name = format!("{section}.state");
        let mut file = match directory.open_file(Path::new(&name)) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        if section == "private" {
            rows.push("private: committed checkpoint (encrypted)".into());
            continue;
        }
        let mut record = [0; STATE_BYTES];
        file.read_exact(&mut record)?;
        if &record[..4] != MAGIC || file.read(&mut [0])? != 0 {
            return Err(invalid("invalid checkpoint state"));
        }
        let mut decoded = [0; 8];
        let mut encoded = [0; 8];
        decoded.copy_from_slice(&record[36..44]);
        encoded.copy_from_slice(&record[44..52]);
        rows.push(format!(
            "{section}: {} decoded bytes, {} encoded bytes",
            u64::from_le_bytes(decoded),
            u64::from_le_bytes(encoded)
        ));
    }
    Ok(rows)
}

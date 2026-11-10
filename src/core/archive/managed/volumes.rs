//! Split archive storage. The manifest is the publication point for a generation.
use super::{invalid, read_number, require_eof, BatoError, Directory, IfMissing};
use crate::core::safe_fs::{absolute_path, open_regular_file, AtomicOutput, StagingDirectory};
use crate::Overwrite;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

mod multi;

const MULTI_MAGIC: &[u8; 4] = b"BAV3";
const MAGIC: &[u8; 4] = b"BAV1";
const PARITY_MAGIC: &[u8; 4] = b"BAV2";
const PARITY_FILE: &str = "parity";
const INDEX: &str = "index.bato";
const GENERATION_PREFIX: &str = "volumes-";
const GENERATION_SUFFIX_LEN: usize = 12;
const MAX_PARTS: u64 = 1_000_000;

pub(super) struct Volumes {
    limit: u64,
    bytes: u64,
    digest: [u8; 32],
    generation: String,
    hashes: Option<Vec<[u8; 32]>>,
    parity: u8,
}

impl Volumes {
    pub(super) fn index(root: &Path) -> PathBuf {
        root.join(INDEX)
    }

    pub(super) fn read(file: &File) -> Result<Self, BatoError> {
        let mut reader = BufReader::new(file);
        let magic = read_number::<4>(&mut reader)?;
        if &magic != MAGIC && &magic != PARITY_MAGIC && &magic != MULTI_MAGIC {
            return Err(invalid("invalid split archive index"));
        }
        let limit = u64::from_le_bytes(read_number(&mut reader)?);
        let bytes = u64::from_le_bytes(read_number(&mut reader)?);
        let count = count_parts(limit, bytes)?;
        let digest = read_number(&mut reader)?;
        let suffix = read_number::<GENERATION_SUFFIX_LEN>(&mut reader)?;
        let parity = if &magic == MULTI_MAGIC {
            let [parity] = read_number(&mut reader)?;
            multi::codec(count, parity)?;
            parity
        } else {
            u8::from(&magic == PARITY_MAGIC)
        };
        let hashes = if parity != 0 {
            let mut hashes = Vec::new();
            hashes
                .try_reserve_exact(count as usize + usize::from(parity))
                .map_err(|_| invalid("cannot allocate split parity index"))?;
            for _ in 0..count + u64::from(parity) {
                hashes.push(read_number(&mut reader)?);
            }
            Some(hashes)
        } else {
            None
        };
        require_eof(&mut reader)?;
        if !suffix.iter().all(u8::is_ascii_alphanumeric) {
            return Err(invalid("invalid split archive generation name"));
        }
        let suffix = String::from_utf8(suffix.to_vec())
            .map_err(|_| invalid("invalid split archive generation name"))?;
        Ok(Self {
            limit,
            bytes,
            digest,
            generation: format!("{GENERATION_PREFIX}{suffix}"),
            hashes,
            parity,
        })
    }

    pub(super) fn join(&self, root: &Path) -> Result<File, BatoError> {
        let root = Directory::open(root, IfMissing::Fail)?;
        let generation = root.child(self.generation.as_ref(), IfMissing::Fail)?;
        if let Some(hashes) = &self.hashes {
            return if self.parity > 1 {
                multi::join(self, &generation, hashes)
            } else {
                self.join_parity(&generation, hashes)
            };
        }
        let mut names = generation.names()?;
        for _ in 0..count_parts(self.limit, self.bytes)? {
            names
                .next()
                .transpose()?
                .ok_or_else(|| invalid("missing split volume"))?;
        }
        if names.next().transpose()?.is_some() {
            return Err(invalid("unexpected entry in split volume directory"));
        }
        let mut output = BufWriter::new(tempfile::tempfile()?);
        if !self.verify_data(&generation, &mut output)? {
            return Err(BatoError::CorruptStream(
                "split volume SHA-256 mismatch".into(),
            ));
        }
        output.flush()?;
        let mut output = output.into_inner().map_err(|e| e.into_error())?;
        output.rewind()?;
        Ok(output)
    }

    fn verify_data(
        &self,
        generation: &Directory,
        output: &mut impl Write,
    ) -> Result<bool, BatoError> {
        let digest = copy_parts(output, self.limit, self.bytes, |index, size, sink| {
            let part = generation.open_file(Path::new(&part_name(index)))?;
            if part.metadata()?.len() != size {
                return Err(invalid(format!("invalid size for split volume {index}")));
            }
            let mut part = BufReader::new(part);
            transfer(&mut part, sink, size)?;
            require_eof(&mut part)
        })?;
        Ok(digest == self.digest)
    }

    pub(super) fn health(&self, root: &Path) -> Result<serde_json::Value, BatoError> {
        let directory = Directory::open(root, IfMissing::Fail)?
            .child(self.generation.as_ref(), IfMissing::Fail)?;
        let count = count_parts(self.limit, self.bytes)?;
        let mut volumes = Vec::new();
        let mut bad_data = 0;
        let mut bad_parity = 0;
        for i in 0..count + u64::from(self.parity) {
            let data = i < count;
            let (name, size) = if data {
                (part_name(i + 1), self.part_size(i + 1))
            } else if self.parity == 1 {
                (PARITY_FILE.to_string(), self.limit.min(self.bytes))
            } else {
                (
                    format!("parity-{:02}", i - count + 1),
                    self.limit.min(self.bytes),
                )
            };
            let status = match directory.open_file(Path::new(&name)) {
                Err(error) if error.kind() == io::ErrorKind::NotFound => "missing",
                Err(error) => return Err(error.into()),
                Ok(mut file) if file.metadata()?.len() == size => match &self.hashes {
                    Some(hashes)
                        if crate::core::jobs::fingerprint(&mut file)? == hashes[i as usize] =>
                    {
                        "verified"
                    }
                    Some(_) => "damaged",
                    None => "size-checked",
                },
                Ok(_) => "damaged",
            };
            if matches!(status, "missing" | "damaged") {
                if data {
                    bad_data += 1;
                } else {
                    bad_parity += 1;
                }
                volumes.push(serde_json::json!({"name":name,"status":status}));
            }
        }
        let verification = if bad_data == 0 {
            self.verify_data(&directory, &mut io::sink())
        } else if bad_data + bad_parity <= usize::from(self.parity) {
            self.join(root).map(|_| true)
        } else {
            Ok(false)
        };
        let assembled = match verification {
            Ok(verified) => verified,
            Err(BatoError::CorruptStream(_) | BatoError::InvalidOptions(_)) => false,
            Err(error) => return Err(error),
        };
        let status = if !assembled {
            "unrecoverable"
        } else if bad_data != 0 {
            "repairable"
        } else if bad_parity != 0 {
            "degraded"
        } else {
            "healthy"
        };
        Ok(
            serde_json::json!({"status":status, "data_volumes":count, "parity_volumes":self.parity,
            "bad_data_volumes":bad_data, "bad_parity_volumes":bad_parity,
            "archive_sha256_verified":assembled, "per_volume_sha256":self.hashes.is_some(),
            "affected_volumes":volumes}),
        )
    }

    fn join_parity(&self, directory: &Directory, hashes: &[[u8; 32]]) -> Result<File, BatoError> {
        let count = count_parts(self.limit, self.bytes)?;
        let mut repaired = None;
        let mut output = BufWriter::new(tempfile::tempfile()?);
        let mut bad = None;
        for index in 1..=count {
            let size = self.part_size(index);
            let file = directory.open_file(Path::new(&part_name(index)));
            let valid = match file {
                Ok(mut file) => {
                    file.metadata()?.len() == size
                        && crate::core::jobs::fingerprint(&mut file)? == hashes[index as usize - 1]
                }
                Err(e) if e.kind() == io::ErrorKind::NotFound => false,
                Err(e) => return Err(e.into()),
            };
            if !valid && bad.replace(index).is_some() {
                return Err(invalid(
                    "one parity volume can repair only one missing or damaged data volume",
                ));
            }
        }
        if let Some(missing) = bad {
            let mut parity = directory.open_file(Path::new(PARITY_FILE))?;
            let parity_len = self.limit.min(self.bytes);
            if parity.metadata()?.len() != parity_len
                || crate::core::jobs::fingerprint(&mut parity)? != hashes[count as usize]
            {
                return Err(invalid("split parity volume failed verification"));
            }
            let mut restored = tempfile::tempfile()?;
            io::copy(&mut parity, &mut restored)?;
            for index in (1..=count).filter(|&index| index != missing) {
                xor_part(
                    &mut restored,
                    directory.open_file(Path::new(&part_name(index)))?,
                )?;
            }
            restored.set_len(self.part_size(missing))?;
            if crate::core::jobs::fingerprint(&mut restored)? != hashes[missing as usize - 1] {
                return Err(invalid("reconstructed volume failed verification"));
            }
            repaired = Some(restored);
        }
        let digest = copy_parts(&mut output, self.limit, self.bytes, |index, size, sink| {
            let file = if bad == Some(index) {
                repaired
                    .take()
                    .ok_or_else(|| invalid("missing reconstructed volume"))?
            } else {
                directory.open_file(Path::new(&part_name(index)))?
            };
            let mut part = BufReader::new(file);
            transfer(&mut part, sink, size)?;
            require_eof(&mut part)
        })?;
        if digest != self.digest {
            return Err(invalid("assembled archive failed SHA-256 verification"));
        }
        output.flush()?;
        let mut file = output.into_inner().map_err(|e| e.into_error())?;
        file.rewind()?;
        Ok(file)
    }

    fn part_size(&self, index: u64) -> u64 {
        (self.bytes - (index - 1) * self.limit).min(self.limit)
    }

    pub(super) fn repair(&self, source: &Path, target: &Path) -> Result<(), BatoError> {
        let mut archive = self.join(source)?;
        Self::create(target, self.limit, &mut archive, self.parity)
    }

    pub(super) fn create(
        root: &Path,
        limit: u64,
        file: &mut File,
        parity: u8,
    ) -> Result<(), BatoError> {
        let root = absolute_path(root)?;
        let parent = root
            .parent()
            .ok_or_else(|| invalid("split output has no parent"))?;
        let parent = Directory::open(parent, IfMissing::Create)?;
        let name = root
            .file_name()
            .ok_or_else(|| invalid("missing split output name"))?;
        let staged = StagingDirectory::new(parent)?;
        Self::write(
            staged.directory().path(),
            limit,
            file,
            (Overwrite::Refuse, parity),
            || Ok(()),
        )?;
        staged.publish_new(name)?;
        Ok(())
    }

    pub(super) fn replace(
        &self,
        root: &Path,
        file: &mut File,
        check_current: impl FnOnce() -> Result<(), BatoError>,
    ) -> Result<Option<io::Error>, BatoError> {
        Self::write(
            root,
            self.limit,
            file,
            (Overwrite::Replace, self.parity),
            check_current,
        )?;
        // The replacement is committed. Surface cleanup failure separately.
        Ok(fs::remove_dir_all(root.join(&self.generation)).err())
    }

    fn write(
        root: &Path,
        limit: u64,
        file: &mut File,
        (overwrite, parity): (Overwrite, u8),
        check_current: impl FnOnce() -> Result<(), BatoError>,
    ) -> Result<(), BatoError> {
        let root = Directory::open(root, IfMissing::Fail)?;
        let bytes = file.metadata()?.len();
        let count = count_parts(limit, bytes)?;
        if parity > 1 {
            multi::codec(count, parity)?;
        }
        file.rewind()?;
        let mut input = BufReader::new(file);
        let generation = tempfile::Builder::new()
            .prefix(GENERATION_PREFIX)
            .rand_bytes(GENERATION_SUFFIX_LEN)
            .tempdir_in(root.path())?;
        let directory = Directory::open(generation.path(), IfMissing::Fail)?;
        let digest = copy_parts(&mut io::sink(), limit, bytes, |index, size, sink| {
            let file = directory.create_file(Path::new(&part_name(index)))?;
            let mut part = BufWriter::new(file);
            transfer(&mut input, &mut Tee(&mut part, sink), size)?;
            part.flush()?;
            part.get_ref().sync_all()?;
            Ok(())
        })?;
        let hashes = match parity {
            0 => None,
            1 => Some(write_parity(&directory, limit, bytes)?),
            _ => Some(multi::write(&directory, limit, bytes, parity)?),
        };
        let name = directory
            .path()
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| invalid("invalid split generation directory"))?;
        let suffix = name
            .strip_prefix(GENERATION_PREFIX)
            .filter(|suffix| suffix.len() == GENERATION_SUFFIX_LEN)
            .ok_or_else(|| invalid("invalid split generation directory"))?;
        let staged = AtomicOutput::new(&Self::index(root.path()), overwrite, None)?;
        let mut manifest = staged.writer()?;
        manifest.write_all(match parity {
            0 => MAGIC,
            1 => PARITY_MAGIC,
            _ => MULTI_MAGIC,
        })?;
        manifest.write_all(&limit.to_le_bytes())?;
        manifest.write_all(&bytes.to_le_bytes())?;
        manifest.write_all(&digest)?;
        manifest.write_all(suffix.as_bytes())?;
        if parity > 1 {
            manifest.write_all(&[parity])?;
        }
        if let Some(hashes) = hashes {
            for hash in hashes {
                manifest.write_all(&hash)?;
            }
        }
        manifest.sync_all()?;
        directory.verify()?;
        #[cfg(unix)]
        directory.file().sync_all()?;
        root.verify()?;
        check_current()?;
        staged.publish()?;
        // Keep the generation referenced by the published index.
        let _published = generation.keep();
        #[cfg(unix)]
        root.file().sync_all()?;
        Ok(())
    }
}

fn write_parity(directory: &Directory, limit: u64, bytes: u64) -> Result<Vec<[u8; 32]>, BatoError> {
    let count = count_parts(limit, bytes)?;
    let mut hashes = Vec::new();
    hashes
        .try_reserve_exact(count as usize + 1)
        .map_err(|_| invalid("cannot allocate split parity index"))?;
    let mut parity = tempfile::tempfile()?;
    parity.set_len(limit.min(bytes))?;
    for index in 1..=count {
        let mut part = directory.open_file(Path::new(&part_name(index)))?;
        hashes.push(crate::core::jobs::fingerprint(&mut part)?);
        xor_part(&mut parity, part)?;
    }
    hashes.push(crate::core::jobs::fingerprint(&mut parity)?);
    let mut output = directory.create_file(Path::new(PARITY_FILE))?;
    io::copy(&mut parity, &mut output)?;
    output.sync_all()?;
    Ok(hashes)
}

fn xor_part(parity: &mut File, mut part: File) -> Result<(), BatoError> {
    parity.rewind()?;
    let mut source = [0; 64 * 1024];
    let mut target = [0; 64 * 1024];
    loop {
        let n = part.read(&mut source)?;
        if n == 0 {
            return Ok(());
        }
        parity.read_exact(&mut target[..n])?;
        for (a, b) in target[..n].iter_mut().zip(&source[..n]) {
            *a ^= *b;
        }
        parity.seek(SeekFrom::Current(-(n as i64)))?;
        parity.write_all(&target[..n])?;
    }
}

fn count_parts(limit: u64, bytes: u64) -> Result<u64, BatoError> {
    if limit == 0 || bytes == 0 || bytes.div_ceil(limit) > MAX_PARTS {
        return Err(invalid(
            "split archive requires between 1 and 1000000 volumes",
        ));
    }
    Ok(bytes.div_ceil(limit))
}

fn part_name(index: u64) -> String {
    format!("part-{index:07}")
}

fn transfer(
    input: &mut impl Read,
    output: &mut (impl Write + ?Sized),
    size: u64,
) -> Result<(), BatoError> {
    if io::copy(&mut input.take(size), output)? != size {
        return Err(BatoError::CorruptStream("truncated split volume".into()));
    }
    Ok(())
}

fn copy_parts(
    writer: &mut impl Write,
    limit: u64,
    bytes: u64,
    mut copy: impl FnMut(u64, u64, &mut dyn Write) -> Result<(), BatoError>,
) -> Result<[u8; 32], BatoError> {
    let mut digest = Sha256::new();
    let mut sink = super::HashWriter {
        output: writer,
        digest: &mut digest,
    };
    let mut remaining = bytes;
    for index in 1..=count_parts(limit, bytes)? {
        let size = remaining.min(limit);
        copy(index, size, &mut sink)?;
        remaining -= size;
    }
    Ok(digest.finalize().into())
}

struct Tee<'a, W>(&'a mut W, &'a mut dyn Write);

impl<W: Write> Write for Tee<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.write_all(bytes)?;
        self.1.write_all(bytes)?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.0.flush()?;
        self.1.flush()
    }
}

pub(super) fn open_index(path: &Path) -> Result<(PathBuf, File), BatoError> {
    let directory = Directory::open(path, IfMissing::Fail)?;
    let index = Volumes::index(directory.path());
    let file = open_regular_file(&index)?;
    Ok((index, file))
}

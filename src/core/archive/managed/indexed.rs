//! Optional BAR2 index followed by the unchanged Bato stream.
//! Encryption provenance: age v1 authenticated chunk seeking, https://c2sp.org/age.
use super::{decode_section, invalid, read_number, CompressionOptions, DecompressionOptions, Job};
use crate::core::{
    archive::{create_solid_archive, inventory, MemberSelection},
    jobs::Section,
    recovery::{self, IndexTrust},
    types::{BatoError, Overwrite},
};
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
};

pub(super) const PLAIN: [u8; 4] = *b"BAI1";
pub(super) const ENCRYPTED: [u8; 4] = *b"BAE2";

pub(super) fn write(
    source: &Path,
    output: &mut impl Write,
    password: Option<&super::Password>,
    options: &CompressionOptions,
) -> Result<(), BatoError> {
    let mut raw = tempfile::tempfile()?;
    create_solid_archive(source, &mut raw, options)?;
    let mut index = tempfile::tempfile()?;
    let decode = DecompressionOptions {
        limits: crate::DecodeLimits {
            max_working_memory: options.memory_budget,
            ..Default::default()
        },
        execution: options.execution.clone(),
    };
    recovery::write_index_generated(&mut raw, &mut index, &decode)?;
    raw.rewind()?;
    if let Some(password) = password {
        output.write_all(&ENCRYPTED)?;
        let mut encrypted = super::encrypt(output, password, options)?;
        write_body(&mut encrypted, &mut index, &mut raw)?;
        encrypted.finish()?;
    } else {
        output.write_all(&PLAIN)?;
        write_body(output, &mut index, &mut raw)?;
    }
    Ok(())
}

fn write_body(output: &mut impl Write, index: &mut File, raw: &mut File) -> io::Result<()> {
    output.write_all(&index.metadata()?.len().to_le_bytes())?;
    io::copy(index, output)?;
    io::copy(raw, output)?;
    Ok(())
}

pub(super) enum Operation<'a> {
    Inspect,
    Extract {
        destination: &'a Path,
        overwrite: Overwrite,
        members: &'a MemberSelection,
        job: Option<&'a Job>,
    },
}

pub(super) fn read(
    input: &mut (impl Read + Seek),
    operation: Operation<'_>,
    protection: (Option<&super::Password>, IndexTrust),
    options: &DecompressionOptions,
) -> Result<inventory::Inventory, BatoError> {
    let (password, trust) = protection;
    if let IndexTrust::Authenticated = trust {
        let password = password.ok_or_else(|| invalid("this archive requires --password-file"))?;
        let bound = Cell::new(None);
        let mut encrypted = super::decrypt(input, &bound, password, options)?;
        return read_body(&mut encrypted, operation, trust, options);
    }
    read_body(input, operation, trust, options)
}

fn read_body(
    input: &mut (impl Read + Seek),
    operation: Operation<'_>,
    trust: IndexTrust,
    options: &DecompressionOptions,
) -> Result<inventory::Inventory, BatoError> {
    let length = u64::from_le_bytes(read_number(input)?);
    if !(55..=recovery::MAX_INDEX_BYTES).contains(&length) {
        return Err(invalid("invalid embedded index size"));
    }
    let mut index = tempfile::tempfile()?;
    if io::copy(&mut input.take(length), &mut index)? != length {
        return Err(invalid("truncated embedded index"));
    }
    let mut stream = Window::new(input)?;
    match operation {
        Operation::Inspect => {
            let expected = recovery::index_binding(index)?;
            let mut digest = Sha256::new();
            let entries = inventory::inspect(
                &mut super::HashReader {
                    input: &mut stream,
                    digest: &mut digest,
                },
                options,
            )?;
            if digest.finalize()[..] != expected {
                return Err(invalid("archive does not match its embedded index"));
            }
            Ok(entries)
        }
        Operation::Extract {
            destination,
            overwrite,
            members,
            job,
        } => {
            let paths = members.paths();
            if paths.is_empty() {
                recovery::index_binding(index)?;
                let section = match trust {
                    IndexTrust::Authenticated => Section::Private,
                    IndexTrust::CheckStream => Section::Plain,
                };
                decode_section(
                    &mut stream,
                    destination,
                    (section, overwrite, members),
                    (options, job),
                )?;
            } else {
                if job.is_some() {
                    return Err(invalid(
                        "indexed member extraction does not support checkpoints",
                    ));
                }
                recovery::extract_indexed_stream(
                    &mut stream,
                    (index, trust),
                    destination,
                    (&paths, overwrite, options),
                )?;
            }
            Ok(inventory::Inventory::new())
        }
    }
}

/// Presents the rest of the underlying stream as a zero-based file.
struct Window<R> {
    input: R,
    start: u64,
    length: u64,
    position: u64,
}

impl<R: Read + Seek> Window<R> {
    fn new(mut input: R) -> io::Result<Self> {
        let start = input.stream_position()?;
        let length = input
            .seek(SeekFrom::End(0))?
            .checked_sub(start)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "truncated embedded index")
            })?;
        input.seek(SeekFrom::Start(start))?;
        Ok(Self {
            input,
            start,
            length,
            position: 0,
        })
    }
}

impl<R: Read> Read for Window<R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let limit = (output.len() as u64).min(self.length.saturating_sub(self.position)) as usize;
        if limit == 0 {
            return Ok(0);
        }
        let count = self.input.read(&mut output[..limit])?;
        self.position += count as u64;
        Ok(count)
    }
}

impl<R: Seek> Seek for Window<R> {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let position = match from {
            SeekFrom::Start(position) => Some(position),
            SeekFrom::Current(delta) => self.position.checked_add_signed(delta),
            SeekFrom::End(delta) => self.length.checked_add_signed(delta),
        }
        .filter(|&position| position <= self.length)
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "seek outside indexed stream")
        })?;
        if position != self.position {
            self.input.seek(SeekFrom::Start(self.start + position))?;
            self.position = position;
        }
        Ok(position)
    }
}

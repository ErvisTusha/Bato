//! Password handling and authenticated encryption of compressed Bato streams.
//! Provenance: age v1, https://c2sp.org/age.
use super::{
    invalid, open_regular_file, BatoError, CompressionOptions, DecompressionOptions, HostBudget,
};
use age::{secrecy::SecretString, Decryptor, Encryptor, Identity};
use std::{
    cell::Cell,
    io::{self, Read, Seek, SeekFrom, Write},
    iter,
    path::Path,
};

const MAX_PASSWORD_BYTES: usize = 4096;
const MAX_AGE_HEADER_BYTES: u64 = 4096;
// scrypt r=8, p=1, N=2^17 uses 128 MiB, plus working space.
const SCRYPT_LOG_N: u8 = 17;
const SCRYPT_RESERVE: usize = 130 * 1024 * 1024;

pub(crate) struct Password {
    secret: SecretString,
    pub(super) file: same_file::Handle,
}

fn crypto(error: impl std::fmt::Display) -> BatoError {
    io::Error::other(format!("archive encryption: {error}")).into()
}

pub(crate) fn password(path: &Path) -> Result<Password, BatoError> {
    let file = open_regular_file(path)?;
    let mut text = String::new();
    (&file)
        .take(MAX_PASSWORD_BYTES as u64 + 1)
        .read_to_string(&mut text)?;
    if text.len() > MAX_PASSWORD_BYTES {
        return Err(invalid("password file exceeds 4096 bytes"));
    }
    if text.ends_with('\n') {
        text.pop();
        if text.ends_with('\r') {
            text.pop();
        }
    }
    if text.is_empty() {
        return Err(invalid("password file is empty"));
    }
    Ok(Password {
        secret: text.into(),
        file: same_file::Handle::from_file(file)?,
    })
}

pub(crate) fn encrypt<W: Write>(
    writer: W,
    pass: &Password,
    options: &CompressionOptions,
) -> Result<age::stream::StreamWriter<W>, BatoError> {
    let budget = HostBudget::new(options.memory_budget, &options.execution.stats)?;
    let _kdf = budget.reserve(SCRYPT_RESERVE)?;
    let mut recipient = age::scrypt::Recipient::new(pass.secret.clone());
    recipient.set_work_factor(SCRYPT_LOG_N);
    let encryptor = Encryptor::with_recipients(iter::once(&recipient as &dyn age::Recipient))
        .map_err(crypto)?;
    encryptor.wrap_output(writer).map_err(crypto)
}

pub(crate) struct HeaderLimit<'a, R> {
    input: R,
    remaining: &'a Cell<Option<u64>>,
}

impl<R: Read> Read for HeaderLimit<'_, R> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let Some(left) = self.remaining.get() else {
            return self.input.read(bytes);
        };
        let length = bytes.len().min(left as usize);
        let n = self.input.read(&mut bytes[..length])?;
        self.remaining.set(Some(left - n as u64));
        Ok(n)
    }
}

impl<R: Seek> Seek for HeaderLimit<'_, R> {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.input.seek(position)
    }
}

pub(crate) fn decrypt<'a, R: Read + 'a>(
    input: R,
    remaining: &'a Cell<Option<u64>>,
    pass: &Password,
    options: &DecompressionOptions,
) -> Result<age::stream::StreamReader<HeaderLimit<'a, R>>, BatoError> {
    let budget = HostBudget::new(options.limits.max_working_memory, &options.execution.stats)?;
    let _kdf = budget.reserve(SCRYPT_RESERVE)?;
    remaining.set(Some(MAX_AGE_HEADER_BYTES));
    let decryptor = Decryptor::new(HeaderLimit { input, remaining }).map_err(crypto)?;
    remaining.set(None);
    let mut identity = age::scrypt::Identity::new(pass.secret.clone());
    identity.set_max_work_factor(SCRYPT_LOG_N);
    decryptor
        .decrypt(iter::once(&identity as &dyn Identity))
        .map_err(crypto)
}

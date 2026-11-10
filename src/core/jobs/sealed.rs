//! Protected checkpoint segments use independent age records. The job's
//! X25519 identity is itself protected with the archive passphrase.
use super::{
    invalid, AtomicOutput, BatoError, CompressionOptions, DecompressionOptions, Directory,
    Overwrite,
};
use crate::core::archive::managed::protection;
use age::{
    secrecy::{ExposeSecret, SecretString},
    x25519, Decryptor, Encryptor, Identity, Recipient,
};
use std::{
    cell::Cell,
    fs::File,
    io::{self, Read, Seek, SeekFrom, Write},
    iter,
    path::Path,
};

pub(super) const KEY: &str = "private.key";
const MAX_KEY_FILE_BYTES: u64 = 4096;

pub(super) fn open_key(
    directory: &Directory,
    passphrase: &Path,
    (encode, decode): (&CompressionOptions, &DecompressionOptions),
) -> Result<x25519::Identity, BatoError> {
    let password = protection::password(passphrase)?;
    match directory.open_file(Path::new(KEY)) {
        Ok(input) => {
            if input.metadata()?.len() > MAX_KEY_FILE_BYTES {
                return Err(invalid("invalid protected checkpoint key size"));
            }
            let header = Cell::new(None);
            let mut input = protection::decrypt(input, &header, &password, decode)?;
            let mut text = String::new();
            (&mut input).take(256).read_to_string(&mut text)?;
            if text.len() >= 256 || input.read(&mut [0])? != 0 {
                return Err(invalid("invalid protected checkpoint key"));
            }
            let key: SecretString = text.into();
            key.expose_secret()
                .parse()
                .map_err(|_| invalid("invalid protected checkpoint key"))
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            if directory.path().join("private.state").try_exists()?
                || directory.path().join("private.data").try_exists()?
            {
                return Err(invalid("protected checkpoint key is missing"));
            }
            let identity = x25519::Identity::generate();
            let staged = AtomicOutput::new(&directory.path().join(KEY), Overwrite::Refuse, None)?;
            let mut writer = protection::encrypt(staged.writer()?, &password, encode)?;
            writer.write_all(identity.to_string().expose_secret().as_bytes())?;
            writer.finish()?.sync_all()?;
            staged.publish()?;
            #[cfg(unix)]
            directory.file().sync_all()?;
            Ok(identity)
        }
        Err(error) => Err(error.into()),
    }
}

pub(super) fn encrypt<W: Write>(
    output: W,
    identity: &x25519::Identity,
) -> Result<age::stream::StreamWriter<W>, BatoError> {
    let recipient = identity.to_public();
    let encoder = Encryptor::with_recipients(iter::once(&recipient as &dyn Recipient))
        .map_err(|error| io::Error::other(error.to_string()))?;
    Ok(encoder.wrap_output(output)?)
}

pub(super) fn decrypt<R: Read>(
    input: R,
    identity: &x25519::Identity,
) -> Result<impl Read + use<R>, BatoError> {
    let bounded = std::rc::Rc::new(Cell::new(Some(4096usize)));
    let decoder = Decryptor::new(KeyHeader {
        input,
        remaining: bounded.clone(),
    })
    .map_err(|error| io::Error::other(error.to_string()))?;
    bounded.set(None);
    decoder
        .decrypt(iter::once(identity as &dyn Identity))
        .map_err(|error| io::Error::other(error.to_string()).into())
}

struct KeyHeader<R> {
    input: R,
    remaining: std::rc::Rc<Cell<Option<usize>>>,
}

impl<R: Read> Read for KeyHeader<R> {
    fn read(&mut self, data: &mut [u8]) -> io::Result<usize> {
        if let Some(left) = self.remaining.get() {
            let size = data.len().min(left);
            let count = self.input.read(&mut data[..size])?;
            self.remaining.set(Some(left - count));
            Ok(count)
        } else {
            self.input.read(data)
        }
    }
}

pub(super) struct Sealed {
    pub(super) file: File,
    pub(super) plain: u64,
}

impl Sealed {
    pub(super) fn append(
        &mut self,
        raw: &mut File,
        end: u64,
        key: &x25519::Identity,
    ) -> Result<u64, BatoError> {
        let record = self.file.stream_position()?;
        self.file.write_all(&[0; 8])?;
        let mut writer = encrypt(&mut self.file, key)?;
        raw.seek(SeekFrom::Start(self.plain))?;
        let length = end - self.plain;
        if io::copy(&mut raw.take(length), &mut writer)? != length {
            return Err(invalid("truncated private checkpoint segment"));
        }
        let file = writer.finish()?;
        let next = file.stream_position()?;
        file.seek(SeekFrom::Start(record))?;
        file.write_all(&(next - record - 8).to_le_bytes())?;
        file.seek(SeekFrom::Start(next))?;
        file.sync_all()?;
        self.plain = end;
        Ok(next)
    }
}

pub(super) fn restore(
    input: &mut File,
    output: &mut impl Write,
    (encrypted, plain): (u64, u64),
    key: &x25519::Identity,
) -> Result<(), BatoError> {
    let mut left = plain;
    let mut input = input.take(encrypted);
    while input.limit() > 0 {
        let mut len = [0; 8];
        input.read_exact(&mut len)?;
        let len = u64::from_le_bytes(len);
        if len == 0 || len > input.limit() {
            return Err(invalid("invalid encrypted checkpoint segment length"));
        }
        let mut record = (&mut input).take(len);
        let count = {
            let reader = decrypt(&mut record, key)?;
            io::copy(&mut reader.take(left.saturating_add(1)), output)?
        };
        if count == 0 || count > left || record.limit() != 0 {
            return Err(invalid("invalid encrypted checkpoint segment"));
        }
        left -= count;
    }
    if left != 0 {
        return Err(invalid("truncated encrypted checkpoint data"));
    }
    Ok(())
}

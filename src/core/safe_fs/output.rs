//! Staged output: a temporary file or directory created next to its final
//! location and published (or removed) as a whole.
use super::super::types::Overwrite;
#[cfg(unix)]
use super::os_error;
use super::{component, lexical_absolute, open_regular_file, unsafe_path, Directory, IfMissing};
#[cfg(unix)]
use rustix::fs::{self as rfs, AtFlags, Mode, OFlags};
use std::ffi::OsStr;
#[cfg(windows)]
use std::fs;
use std::{
    ffi::OsString,
    fs::File,
    io,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

/// Attempts at creating a uniquely named temporary entry before giving up.
const CREATE_ATTEMPTS: u32 = 128;

/// A temporary name for one creation attempt. Uniqueness comes from the
/// process id, the clock and the attempt number; callers create the entry
/// exclusively and retry with the next attempt on a collision.
fn temporary_name(prefix: &str, attempt: u32) -> OsString {
    // A clock before 1970 only weakens uniqueness, which retries cover.
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{prefix}{:x}-{stamp:x}-{attempt:x}", std::process::id()).into()
}
pub struct AtomicOutput {
    dir: Directory,
    file: File,
    temporary: OsString,
    name: OsString,
    overwrite: Overwrite,
    published: bool,
}
impl AtomicOutput {
    pub fn new(dst: &Path, overwrite: Overwrite, src: Option<&Path>) -> io::Result<Self> {
        let dst = lexical_absolute(dst)?;
        let name = dst
            .file_name()
            .ok_or_else(|| unsafe_path("missing output filename"))?
            .to_os_string();
        let parent = dst
            .parent()
            .ok_or_else(|| unsafe_path("missing output directory"))?;
        let dir = Directory::open(parent, IfMissing::Create)?;
        // Type admission happens before same-file opening: a FIFO must never
        // block the CLI merely because it was passed as a forced destination.
        if dir.file_target(&name, overwrite)? {
            if let Some(src) = src {
                if let Ok(source) = open_regular_file(src) {
                    let target = dir.open_file(Path::new(&name))?;
                    if same_file::Handle::from_file(source)?
                        == same_file::Handle::from_file(target)?
                    {
                        return Err(unsafe_path("input and output refer to the same file"));
                    }
                }
            }
        }
        for attempt in 0..CREATE_ATTEMPTS {
            let temporary = temporary_name(".bato-", attempt);
            match dir.create_file(Path::new(&temporary)) {
                Ok(file) => {
                    return Ok(Self {
                        dir,
                        file,
                        temporary,
                        name,
                        overwrite,
                        published: false,
                    })
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "cannot create temporary output",
        ))
    }
    pub fn writer(&self) -> io::Result<File> {
        self.file.try_clone()
    }
    pub fn temporary_path(&self) -> PathBuf {
        self.dir.path.join(&self.temporary)
    }
    pub fn publish(mut self) -> io::Result<()> {
        self.dir
            .publish_file(&self.dir, &self.temporary, &self.name, self.overwrite)?;
        self.published = true;
        Ok(())
    }
}
impl Drop for AtomicOutput {
    fn drop(&mut self) {
        if !self.published {
            #[cfg(unix)]
            {
                if let (Ok(a), Ok(b)) = (
                    rfs::statat(self.dir.file(), &self.temporary, AtFlags::SYMLINK_NOFOLLOW),
                    rfs::fstat(&self.file),
                ) {
                    if a.st_dev == b.st_dev && a.st_ino == b.st_ino {
                        // Best-effort cleanup: Drop cannot report errors, and a
                        // leftover temporary file never replaces the output.
                        let _ = self.dir.remove_file(&self.temporary);
                    }
                }
            }
            #[cfg(not(unix))]
            if self.dir.verify().is_ok() {
                // Best-effort cleanup; see the Unix branch above.
                let _ = self.dir.remove_file(&self.temporary);
            }
        }
    }
}

pub(crate) struct StagingDirectory {
    parent: Directory,
    dir: Option<Directory>,
    name: OsString,
}
impl StagingDirectory {
    pub fn new(parent: Directory) -> io::Result<Self> {
        for attempt in 0..CREATE_ATTEMPTS {
            let name = temporary_name(".bato-stage-", attempt);
            parent.verify()?;
            #[cfg(unix)]
            let made =
                rfs::mkdirat(parent.file(), &name, Mode::from_raw_mode(0o700)).map_err(os_error);
            #[cfg(not(unix))]
            let made = fs::create_dir(parent.path.join(&name));
            match made {
                Ok(()) => {
                    return Ok(Self {
                        dir: Some(parent.child(&name, IfMissing::Fail)?),
                        parent,
                        name,
                    })
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "cannot create staging directory",
        ))
    }
    pub fn directory(&self) -> &Directory {
        self.dir.as_ref().unwrap()
    }

    /// Publishes a fully built tree as a new sibling without replacing a
    /// concurrently created destination.
    pub fn publish_new(mut self, name: &OsStr) -> io::Result<()> {
        component(name)?;
        self.parent.verify()?;
        self.directory().verify()?;
        #[cfg(unix)]
        rfs::renameat_with(
            self.parent.file(),
            &self.name,
            self.parent.file(),
            name,
            rfs::RenameFlags::NOREPLACE,
        )
        .map_err(os_error)?;
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            use windows_sys::Win32::Storage::FileSystem::MoveFileExW;
            let old: Vec<u16> = self
                .parent
                .path
                .join(&self.name)
                .as_os_str()
                .encode_wide()
                .chain([0])
                .collect();
            let new: Vec<u16> = self
                .parent
                .path
                .join(name)
                .as_os_str()
                .encode_wide()
                .chain([0])
                .collect();
            // The parent stays pinned. The staging directory's handles must
            // close before Windows will permit renaming the entry.
            drop(self.dir.take());
            // SAFETY: both buffers are NUL-terminated UTF-16 paths and live
            // for the call. Flags zero prohibit destination replacement.
            if unsafe { MoveFileExW(old.as_ptr(), new.as_ptr(), 0) } == 0 {
                return Err(io::Error::last_os_error());
            }
        }
        drop(self.dir.take());
        #[cfg(unix)]
        self.parent.file().sync_all()?;
        Ok(())
    }
}
#[cfg(unix)]
fn clear_directory(dir: &File) -> io::Result<()> {
    use std::os::unix::ffi::OsStrExt;
    for e in rfs::Dir::read_from(dir).map_err(os_error)? {
        let e = e.map_err(os_error)?;
        if matches!(e.file_name().to_bytes(), b"." | b"..") {
            continue;
        }
        let name = OsStr::from_bytes(e.file_name().to_bytes());
        let st = rfs::statat(dir, name, AtFlags::SYMLINK_NOFOLLOW).map_err(os_error)?;
        let flags = if st.st_mode & 0o170000 == 0o040000 {
            let child: File = rfs::openat(
                dir,
                name,
                OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC,
                Mode::empty(),
            )
            .map_err(os_error)?
            .into();
            clear_directory(&child)?;
            AtFlags::REMOVEDIR
        } else {
            AtFlags::empty()
        };
        rfs::unlinkat(dir, name, flags).map_err(os_error)?;
    }
    Ok(())
}
impl Drop for StagingDirectory {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(dir) = &self.dir {
            if let (Ok(a), Ok(b)) = (
                rfs::statat(self.parent.file(), &self.name, AtFlags::SYMLINK_NOFOLLOW),
                rfs::fstat(dir.file()),
            ) {
                if a.st_dev == b.st_dev && a.st_ino == b.st_ino {
                    // Best-effort cleanup: Drop cannot report errors, and an
                    // unpublished staging directory is never extracted output.
                    let _ = clear_directory(dir.file());
                    let _ = rfs::unlinkat(self.parent.file(), &self.name, AtFlags::REMOVEDIR);
                }
            }
        }
        #[cfg(not(unix))]
        if self.dir.as_ref().is_some_and(|dir| dir.verify().is_ok()) && self.parent.verify().is_ok()
        {
            let path = self.parent.path.join(&self.name);
            drop(self.dir.take());
            // Best-effort cleanup; see the Unix branch above.
            let _ = fs::remove_dir_all(path);
        }
    }
}

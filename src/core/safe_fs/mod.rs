//! Shared filesystem operations for CLI output and staged directory archives.
//! Unix operations stay relative to pinned descriptors. Windows pins every
//! ancestor without write/delete sharing and opens reparse points themselves.
use super::types::Overwrite;
#[cfg(unix)]
use rustix::fs::{self as rfs, AtFlags, Mode, OFlags, RenameFlags};
use std::{
    ffi::{OsStr, OsString},
    fs::{self, File},
    io,
    path::{Component, Path, PathBuf},
};

/// What opening a directory does when an entry is missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IfMissing {
    Fail,
    Create,
}
/// Whether a name must match the on-disk entry's spelling exactly. User-given
/// ancestors may differ in case on case-insensitive filesystems; names Bato
/// creates or extracts must not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Spelling {
    AsFound,
    Exact,
}

mod output;
pub use output::AtomicOutput;
pub(crate) use output::StagingDirectory;

fn unsafe_path(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
#[cfg(unix)]
fn os_error(e: rustix::io::Errno) -> io::Error {
    io::Error::from_raw_os_error(e.raw_os_error())
}

pub fn absolute_path(path: &Path) -> io::Result<PathBuf> {
    let full = lexical_absolute(path)?;
    #[cfg(target_os = "macos")]
    let (full, _) = system_alias(&full)?;
    Ok(full)
}
fn lexical_absolute(path: &Path) -> io::Result<PathBuf> {
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(unsafe_path("path contains parent traversal"));
    }
    let full = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    #[cfg(windows)]
    for c in full.components() {
        if let Component::Prefix(p) = c {
            use std::path::Prefix;
            if !matches!(
                p.kind(),
                Prefix::Disk(_)
                    | Prefix::UNC(_, _)
                    | Prefix::VerbatimDisk(_)
                    | Prefix::VerbatimUNC(_, _)
            ) {
                return Err(unsafe_path("device namespace is not a filesystem path"));
            }
        }
    }
    Ok(full
        .components()
        .filter(|c| !matches!(c, Component::CurDir))
        .collect())
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
struct SystemAlias {
    path: PathBuf,
    target: PathBuf,
    dev: u64,
    ino: u64,
}
#[cfg(target_os = "macos")]
fn system_alias(full: &Path) -> io::Result<(PathBuf, Option<SystemAlias>)> {
    use std::os::unix::fs::MetadataExt;
    for (alias, target) in [
        ("/tmp", "/private/tmp"),
        ("/var", "/private/var"),
        ("/etc", "/private/etc"),
    ] {
        if let Ok(rest) = full.strip_prefix(alias) {
            let m = fs::symlink_metadata(alias)?;
            if !m.file_type().is_symlink() {
                break;
            }
            let link = fs::read_link(alias)?;
            if m.uid() != 0
                || !(link == Path::new(target) || link == Path::new(target.trim_start_matches('/')))
            {
                return Err(unsafe_path("unsafe system directory alias"));
            }
            return Ok((
                Path::new(target).join(rest),
                Some(SystemAlias {
                    path: alias.into(),
                    target: link,
                    dev: m.dev(),
                    ino: m.ino(),
                }),
            ));
        }
    }
    Ok((full.to_path_buf(), None))
}

fn component(name: &OsStr) -> io::Result<()> {
    let mut components = Path::new(name).components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(unsafe_path("invalid filesystem name"));
    }
    #[cfg(windows)]
    windows_component(name)?;
    Ok(())
}

#[cfg(any(windows, test))]
fn windows_component(name: &OsStr) -> io::Result<()> {
    let name = name
        .to_str()
        .ok_or_else(|| unsafe_path("unrepresentable filename"))?;
    if name.is_empty()
        || name.ends_with(['.', ' '])
        || name
            .chars()
            .any(|c| c < ' ' || matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*'))
    {
        return Err(unsafe_path("unrepresentable Windows filename"));
    }
    let base = name
        .split('.')
        .next()
        .unwrap()
        .trim_end_matches(' ')
        .to_ascii_uppercase();
    let device = matches!(
        base.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$"
    ) || ["COM", "LPT"].iter().any(|prefix| {
        base.strip_prefix(prefix).is_some_and(|suffix| {
            matches!(
                suffix,
                "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
            )
        })
    });
    if device {
        Err(unsafe_path("reserved Windows device filename"))
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Directory,
    Regular,
    Special,
}
struct Pin {
    file: File,
    name: Option<OsString>,
}
pub struct Directory {
    path: PathBuf,
    pins: Vec<Pin>,
    #[cfg(target_os = "macos")]
    alias: Option<SystemAlias>,
}

impl Directory {
    pub fn open(path: &Path, missing: IfMissing) -> io::Result<Self> {
        let full = lexical_absolute(path)?;
        #[cfg(target_os = "macos")]
        let (full, alias) = system_alias(&full)?;
        #[cfg(unix)]
        let root = PathBuf::from("/");
        #[cfg(windows)]
        let root: PathBuf = full
            .components()
            .take_while(|c| matches!(c, Component::Prefix(_) | Component::RootDir))
            .collect();
        #[cfg(not(any(unix, windows)))]
        return Err(unsafe_path("unsupported filesystem platform"));
        #[cfg(any(unix, windows))]
        {
            let file = open_directory(None, &root, None, IfMissing::Fail)?;
            let mut dir = Self {
                path: root,
                pins: vec![Pin { file, name: None }],
                #[cfg(target_os = "macos")]
                alias,
            };
            for c in full.components() {
                if let Component::Normal(name) = c {
                    dir.append(name, missing, Spelling::AsFound)?;
                }
            }
            dir.verify()?;
            Ok(dir)
        }
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub(crate) fn file(&self) -> &File {
        &self.pins.last().unwrap().file
    }
    pub fn verify(&self) -> io::Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(alias) = &self.alias {
            use std::os::unix::fs::MetadataExt;
            let m = fs::symlink_metadata(&alias.path)?;
            if !m.file_type().is_symlink()
                || m.dev() != alias.dev
                || m.ino() != alias.ino
                || fs::read_link(&alias.path)? != alias.target
            {
                return Err(unsafe_path("system directory alias changed"));
            }
        }
        #[cfg(unix)]
        for pair in self.pins.windows(2) {
            let actual = rfs::statat(
                &pair[0].file,
                pair[1].name.as_ref().unwrap(),
                AtFlags::SYMLINK_NOFOLLOW,
            )
            .map_err(os_error)?;
            let pinned = rfs::fstat(&pair[1].file).map_err(os_error)?;
            if actual.st_mode & 0o170000 != 0o040000
                || actual.st_dev != pinned.st_dev
                || actual.st_ino != pinned.st_ino
            {
                return Err(unsafe_path("ancestor directory changed"));
            }
        }
        // Windows ancestor handles deny write/delete sharing, preventing
        // reparse-point conversion, removal, rename and substitution.
        #[cfg(windows)]
        for pin in &self.pins {
            checked_directory(&pin.file)?;
        }
        Ok(())
    }
    fn duplicate(&self) -> io::Result<Self> {
        let mut pins = Vec::new();
        pins.try_reserve_exact(self.pins.len())
            .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        for pin in &self.pins {
            pins.push(Pin {
                file: pin.file.try_clone()?,
                name: pin.name.clone(),
            });
        }
        Ok(Self {
            path: self.path.clone(),
            pins,
            #[cfg(target_os = "macos")]
            alias: self.alias.clone(),
        })
    }
    fn append(&mut self, name: &OsStr, missing: IfMissing, spelling: Spelling) -> io::Result<()> {
        component(name)?;
        self.verify()?;
        let path = self.path.join(name);
        let file = open_directory(Some(self.file()), &path, Some(name), missing)?;
        if spelling == Spelling::Exact {
            self.exact_name(name)?;
        }
        self.pins
            .try_reserve(1)
            .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        self.pins.push(Pin {
            file,
            name: Some(name.to_os_string()),
        });
        self.path = path;
        self.verify()
    }
    pub(crate) fn child(&self, name: &OsStr, missing: IfMissing) -> io::Result<Self> {
        let mut dir = self.duplicate()?;
        dir.append(name, missing, Spelling::Exact)?;
        Ok(dir)
    }
    pub(crate) fn descend(&self, relative: &Path, missing: IfMissing) -> io::Result<Self> {
        let mut dir = self.duplicate()?;
        for c in relative.components() {
            match c {
                Component::Normal(name) => dir.append(name, missing, Spelling::Exact)?,
                Component::CurDir => {}
                _ => return Err(unsafe_path("unsafe relative path")),
            }
        }
        Ok(dir)
    }
    fn parent_for(&self, relative: &Path, missing: IfMissing) -> io::Result<(Self, OsString)> {
        let name = relative
            .file_name()
            .ok_or_else(|| unsafe_path("missing filename"))?;
        component(name)?;
        let parent = self.descend(relative.parent().unwrap_or(Path::new("")), missing)?;
        Ok((parent, name.to_os_string()))
    }
    #[cfg(unix)]
    pub(crate) fn names(&self) -> io::Result<impl Iterator<Item = io::Result<OsString>>> {
        use std::os::unix::ffi::OsStrExt;
        self.verify()?;
        let entries = rfs::Dir::read_from(self.file()).map_err(os_error)?;
        Ok(entries.filter_map(|e| match e {
            Ok(e) if matches!(e.file_name().to_bytes(), b"." | b"..") => None,
            Ok(e) => Some(Ok(
                OsStr::from_bytes(e.file_name().to_bytes()).to_os_string()
            )),
            Err(e) => Some(Err(os_error(e))),
        }))
    }
    #[cfg(not(unix))]
    pub(crate) fn names(&self) -> io::Result<impl Iterator<Item = io::Result<OsString>>> {
        self.verify()?;
        Ok(fs::read_dir(&self.path)?.map(|e| e.map(|e| e.file_name())))
    }
    // Check the name actually stored by this filesystem. This detects case,
    // normalization and short-name aliases without inventing a case-fold rule.
    fn exact_name(&self, requested: &OsStr) -> io::Result<()> {
        for name in self.names()? {
            if name? == requested {
                return Ok(());
            }
        }
        Err(unsafe_path(
            "filesystem cannot preserve the requested filename",
        ))
    }
    pub(crate) fn kind(&self, name: &OsStr) -> io::Result<Kind> {
        component(name)?;
        #[cfg(unix)]
        {
            let st = rfs::statat(self.file(), name, AtFlags::SYMLINK_NOFOLLOW).map_err(os_error)?;
            Ok(match st.st_mode & 0o170000 {
                0o040000 => Kind::Directory,
                0o100000 => Kind::Regular,
                _ => Kind::Special,
            })
        }
        #[cfg(not(unix))]
        {
            let m = fs::symlink_metadata(self.path.join(name))?;
            Ok(if special_metadata(&m) {
                Kind::Special
            } else if m.is_dir() {
                Kind::Directory
            } else if m.is_file() {
                Kind::Regular
            } else {
                Kind::Special
            })
        }
    }
    pub(crate) fn file_target(&self, name: &OsStr, overwrite: Overwrite) -> io::Result<bool> {
        self.verify()?;
        match self.kind(name) {
            Ok(Kind::Regular) => {
                self.exact_name(name)?;
                if overwrite == Overwrite::Replace {
                    Ok(true)
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::AlreadyExists,
                        "output exists; use -f",
                    ))
                }
            }
            Ok(_) => Err(unsafe_path(
                "force only replaces regular files; refusing special output",
            )),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e),
        }
    }
    pub(crate) fn open_file(&self, relative: &Path) -> io::Result<File> {
        let (parent, name) = self.parent_for(relative, IfMissing::Fail)?;
        parent.verify()?;
        #[cfg(unix)]
        let file: File = rfs::openat(
            parent.file(),
            &name,
            OFlags::RDONLY | OFlags::NONBLOCK | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::empty(),
        )
        .map_err(os_error)?
        .into();
        #[cfg(not(unix))]
        let file = {
            use std::os::windows::fs::OpenOptionsExt;
            use windows_sys::Win32::Storage::FileSystem::FILE_FLAG_OPEN_REPARSE_POINT;
            fs::OpenOptions::new()
                .read(true)
                .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
                .open(parent.path.join(&name))?
        };
        let m = file.metadata()?;
        if !m.is_file() || special_metadata(&m) {
            return Err(unsafe_path("source is not a regular file"));
        }
        parent.exact_name(&name)?;
        parent.verify()?;
        Ok(file)
    }
    pub(crate) fn create_file(&self, relative: &Path) -> io::Result<File> {
        let (parent, name) = self.parent_for(relative, IfMissing::Create)?;
        parent.file_target(&name, Overwrite::Refuse)?;
        #[cfg(unix)]
        let file: File = rfs::openat(
            parent.file(),
            &name,
            OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::NOFOLLOW | OFlags::CLOEXEC,
            Mode::from_raw_mode(0o600),
        )
        .map_err(os_error)?
        .into();
        #[cfg(not(unix))]
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(parent.path.join(&name))?;
        parent.exact_name(&name)?;
        parent.verify()?;
        Ok(file)
    }
    pub(crate) fn publish_file(
        &self,
        source: &Self,
        old: &OsStr,
        name: &OsStr,
        overwrite: Overwrite,
    ) -> io::Result<()> {
        source.verify()?;
        self.file_target(name, overwrite)?;
        if source.kind(old)? != Kind::Regular {
            return Err(unsafe_path("staged file is not regular"));
        }
        #[cfg(unix)]
        rfs::renameat_with(
            source.file(),
            old,
            self.file(),
            name,
            match overwrite {
                Overwrite::Replace => RenameFlags::empty(),
                Overwrite::Refuse => RenameFlags::NOREPLACE,
            },
        )
        .map_err(os_error)?;
        #[cfg(not(unix))]
        if overwrite == Overwrite::Replace {
            fs::rename(source.path.join(old), self.path.join(name))?;
        } else {
            fs::hard_link(source.path.join(old), self.path.join(name))?;
            fs::remove_file(source.path.join(old))?;
        }
        Ok(())
    }
    fn remove_file(&self, name: &OsStr) -> io::Result<()> {
        #[cfg(unix)]
        return rfs::unlinkat(self.file(), name, AtFlags::empty()).map_err(os_error);
        #[cfg(not(unix))]
        fs::remove_file(self.path.join(name))
    }
}

#[cfg(unix)]
fn open_directory(
    parent: Option<&File>,
    _path: &Path,
    name: Option<&OsStr>,
    missing: IfMissing,
) -> io::Result<File> {
    let flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC;
    let open = || match parent {
        Some(parent) => rfs::openat(parent, name.unwrap(), flags, Mode::empty()),
        None => rfs::openat(rfs::CWD, "/", flags, Mode::empty()),
    };
    let fd = match open() {
        Ok(fd) => fd,
        Err(e) if missing == IfMissing::Create && e == rustix::io::Errno::NOENT => {
            rfs::mkdirat(parent.unwrap(), name.unwrap(), Mode::from_raw_mode(0o755))
                .map_err(os_error)?;
            open().map_err(os_error)?
        }
        Err(e) => return Err(os_error(e)),
    };
    Ok(fd.into())
}
#[cfg(windows)]
fn open_directory(
    _parent: Option<&File>,
    path: &Path,
    _name: Option<&OsStr>,
    missing: IfMissing,
) -> io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT, FILE_SHARE_READ,
    };
    let open = || {
        fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT)
            .open(path)
    };
    let file = match open() {
        Ok(file) => file,
        Err(e) if missing == IfMissing::Create && e.kind() == io::ErrorKind::NotFound => {
            fs::create_dir(path)?;
            open()?
        }
        Err(e) => return Err(e),
    };
    checked_directory(&file)?;
    Ok(file)
}
#[cfg(windows)]
fn checked_directory(file: &File) -> io::Result<()> {
    let m = file.metadata()?;
    if !m.is_dir() || special_metadata(&m) {
        Err(unsafe_path("directory is a reparse point or special file"))
    } else {
        Ok(())
    }
}
fn special_metadata(m: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        use windows_sys::Win32::Storage::FileSystem::{
            FILE_ATTRIBUTE_DEVICE, FILE_ATTRIBUTE_REPARSE_POINT,
        };
        m.file_attributes() & (FILE_ATTRIBUTE_DEVICE | FILE_ATTRIBUTE_REPARSE_POINT) != 0
    }
    #[cfg(not(windows))]
    {
        m.file_type().is_symlink()
    }
}
pub fn open_regular_file(path: &Path) -> io::Result<File> {
    let path = absolute_path(path)?;
    let parent = Directory::open(
        path.parent().ok_or_else(|| unsafe_path("missing parent"))?,
        IfMissing::Fail,
    )?;
    parent.open_file(Path::new(
        path.file_name()
            .ok_or_else(|| unsafe_path("missing filename"))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::io::Write;
    #[test]
    fn windows_names_reject_aliases_and_device_targets_without_conversion() {
        for name in [
            "NUL",
            "nul.txt",
            "LPT1.bin",
            "COM³.dat",
            "CONIN$",
            "AUX .dat",
            "trailing.",
            "trailing ",
            "a:b",
            "a\u{1}b",
        ] {
            assert!(windows_component(OsStr::new(name)).is_err(), "{name}");
        }
        for name in [
            "ordinary.name",
            "COM10",
            "...inside",
            "normal space",
            "未来",
        ] {
            assert!(windows_component(OsStr::new(name)).is_ok());
        }
    }
    #[cfg(unix)]
    #[test]
    fn ancestor_replacement_cannot_redirect_publication_or_cleanup() {
        let root = tempfile::tempdir().unwrap();
        let parent = root.path().join("parent");
        fs::create_dir(&parent).unwrap();
        let output = AtomicOutput::new(&parent.join("output"), Overwrite::Refuse, None).unwrap();
        output
            .writer()
            .unwrap()
            .write_all(b"verified bytes")
            .unwrap();
        fs::rename(&parent, root.path().join("original")).unwrap();
        fs::create_dir(&parent).unwrap();
        fs::write(parent.join("unrelated"), b"keep").unwrap();
        assert!(output.publish().is_err());
        assert!(!parent.join("output").exists());
        assert!(!root.path().join("original/output").exists());
        assert_eq!(fs::read(parent.join("unrelated")).unwrap(), b"keep");
        assert_eq!(
            fs::read_dir(root.path().join("original")).unwrap().count(),
            0
        );
    }
    #[cfg(unix)]
    #[test]
    fn staged_files_remain_anchored_when_an_ancestor_moves() {
        let root = tempfile::tempdir().unwrap();
        let parent = root.path().join("parent");
        fs::create_dir(&parent).unwrap();
        let stage =
            StagingDirectory::new(Directory::open(&parent, IfMissing::Fail).unwrap()).unwrap();
        stage
            .directory()
            .create_file(Path::new("file"))
            .unwrap()
            .write_all(b"staged")
            .unwrap();
        fs::rename(&parent, root.path().join("moved")).unwrap();
        fs::create_dir(&parent).unwrap();
        assert!(stage
            .directory()
            .create_file(Path::new("elsewhere"))
            .is_err());
        drop(stage);
        assert_eq!(fs::read_dir(root.path().join("moved")).unwrap().count(), 0);
        assert_eq!(fs::read_dir(parent).unwrap().count(), 0);
    }
    #[cfg(windows)]
    #[test]
    fn pinned_ancestor_denies_rename_until_the_operation_ends() {
        let root = tempfile::tempdir().unwrap();
        let parent = root.path().join("parent");
        fs::create_dir(&parent).unwrap();
        let output = AtomicOutput::new(&parent.join("output"), Overwrite::Refuse, None).unwrap();
        assert!(fs::rename(&parent, root.path().join("moved")).is_err());
        drop(output);
        fs::rename(&parent, root.path().join("moved")).unwrap();
    }
    #[cfg(windows)]
    #[test]
    fn reparse_ancestors_are_rejected_before_creating_output() {
        let root = tempfile::tempdir().unwrap();
        let real = root.path().join("real");
        fs::create_dir(&real).unwrap();
        let link = root.path().join("link");
        if let Err(e) = std::os::windows::fs::symlink_dir(&real, &link) {
            if e.kind() == io::ErrorKind::PermissionDenied {
                eprintln!("UNTESTED: Windows symlink fixture privilege unavailable");
                return;
            }
            panic!("cannot create reparse fixture: {e}");
        }
        assert!(Directory::open(&link, IfMissing::Fail).is_err());
        assert!(AtomicOutput::new(&link.join("output"), Overwrite::Refuse, None).is_err());
        assert_eq!(fs::read_dir(real).unwrap().count(), 0);
    }
}

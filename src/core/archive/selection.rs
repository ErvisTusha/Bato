//! Archive member validation and filtering for staged publication.
use super::{validate_path, BatoError, Directory, IfMissing, Kind, MAX_PATH_BYTES};
use std::{
    collections::BTreeSet,
    ffi::OsStr,
    path::{Path, PathBuf},
};

pub(super) fn member(path: &Path) -> Result<&str, BatoError> {
    let name = path
        .to_str()
        .ok_or_else(|| BatoError::InvalidOptions("archive paths must be UTF-8".into()))?;
    validate_path(name)?;
    if name.len() > MAX_PATH_BYTES {
        return Err(BatoError::InvalidOptions(
            "archive path exceeds length limit".into(),
        ));
    }
    Ok(name)
}

pub(super) fn member_kind(root: &Directory, path: &Path) -> Result<Kind, BatoError> {
    let parent = root.descend(path.parent().unwrap_or(Path::new("")), IfMissing::Fail)?;
    let name = path
        .file_name()
        .ok_or_else(|| BatoError::InvalidOptions("missing member name".into()))?;
    let kind = parent.kind(name)?;
    // Both operations check exact spelling on case-insensitive filesystems.
    match kind {
        Kind::Directory => {
            parent.child(name, IfMissing::Fail)?;
        }
        Kind::Regular => {
            parent.open_file(Path::new(name))?;
        }
        Kind::Special => {
            return Err(BatoError::InvalidOptions(
                "archive member must be a regular file or directory".into(),
            ))
        }
    }
    Ok(kind)
}

/// Archive-relative paths to publish. An empty selection restores the whole tree.
#[derive(Default)]
pub struct MemberSelection {
    roots: BTreeSet<PathBuf>,
}

impl MemberSelection {
    pub fn new(paths: &[PathBuf]) -> Result<Self, BatoError> {
        let roots = paths
            .iter()
            .map(|path| member(path).map(PathBuf::from))
            .collect::<Result<BTreeSet<_>, _>>()?;
        Ok(Self { roots })
    }

    pub(crate) fn paths(&self) -> Vec<PathBuf> {
        self.roots.iter().cloned().collect()
    }

    fn verify(&self, stage: &Directory) -> Result<(), BatoError> {
        for path in &self.roots {
            member_kind(stage, path).map_err(|error| {
                BatoError::InvalidOptions(format!("archive member {}: {error}", path.display()))
            })?;
        }
        Ok(())
    }
}

pub(super) struct PublishScope<'a> {
    stage_root: &'a Path,
    roots: &'a BTreeSet<PathBuf>,
    parents: BTreeSet<&'a Path>,
}

impl<'a> PublishScope<'a> {
    pub(super) fn new(
        stage: &'a Directory,
        selection: &'a MemberSelection,
    ) -> Result<Self, BatoError> {
        selection.verify(stage)?;
        let parents = selection
            .roots
            .iter()
            .flat_map(|path| path.ancestors().skip(1))
            .filter(|path| !path.as_os_str().is_empty())
            .collect();
        Ok(Self {
            stage_root: stage.path(),
            roots: &selection.roots,
            parents,
        })
    }

    pub(super) fn includes(&self, directory: &Directory, name: &OsStr) -> Result<bool, BatoError> {
        if self.roots.is_empty() {
            return Ok(true);
        }
        let parent = directory
            .path()
            .strip_prefix(self.stage_root)
            .map_err(|_| {
                BatoError::InvalidOptions("publication path is outside its staging tree".into())
            })?;
        let path = parent.join(name);
        Ok(self.parents.contains(path.as_path())
            || path
                .ancestors()
                .any(|ancestor| self.roots.contains(ancestor)))
    }
}

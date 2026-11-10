//! Verified inventory, archive comparison and volume health.
use super::{
    covered, decrypt, index_trust, indexed, invalid, password, read_number, read_scopes,
    require_eof, Access, ArchiveFile, ArchiveOptions, BatoError, DecompressionOptions, HashReader,
    SELECTIVE, WHOLE,
};
use crate::core::archive::inventory;
use sha2::{Digest, Sha256};
use std::{
    cell::Cell,
    collections::BTreeSet,
    fs,
    io::{BufReader, Read, Seek},
    path::Path,
};

/// Verify member bytes and collect paths and hashes without writing a restored tree.
pub fn inspect(
    source: &Path,
    features: &ArchiveOptions,
    options: &DecompressionOptions,
) -> Result<inventory::Inventory, BatoError> {
    let archive = ArchiveFile::open(source, Access::Read)?;
    let mut stream = BufReader::new(archive.stream()?);
    let magic = read_number(&mut stream)?;
    if let Some(trust) = index_trust(magic) {
        if features.public_only {
            return Err(invalid("indexed archives have no separate public section"));
        }
        let pass = features
            .password_file
            .as_deref()
            .map(password)
            .transpose()?;
        return indexed::read(
            &mut stream,
            indexed::Operation::Inspect,
            (pass.as_ref(), trust),
            options,
        );
    }
    if !matches!(magic, WHOLE | SELECTIVE) {
        if features.public_only {
            return Err(invalid(
                "--public-only requires a selectively protected archive",
            ));
        }
        stream.rewind()?;
        return inventory::inspect(&mut stream, options);
    }
    if magic == WHOLE && features.public_only {
        return Err(invalid("this archive has no public section"));
    }
    let mut public = inventory::Inventory::new();
    let mut digest = Sha256::new();
    if magic == SELECTIVE {
        let length = u64::from_le_bytes(read_number(&mut stream)?);
        let mut bounded = (&mut stream).take(length);
        public = inventory::inspect(
            &mut HashReader {
                input: &mut bounded,
                digest: &mut digest,
            },
            options,
        )?;
        if bounded.limit() != 0 {
            return Err(invalid("invalid public stream length"));
        }
    }
    if features.public_only {
        return Ok(public);
    }
    let password = features
        .password_file
        .as_deref()
        .ok_or_else(|| invalid("this archive requires --password-file"))?;
    let pass = super::password(password)?;
    let limit = Cell::new(None);
    let mut decrypted = decrypt(&mut stream, &limit, &pass, options)?;
    let scopes = if magic == SELECTIVE {
        if read_number::<32>(&mut decrypted)?[..] != digest.finalize()[..] {
            return Err(invalid("public archive SHA-256 mismatch"));
        }
        read_scopes(&mut decrypted)?
    } else {
        BTreeSet::new()
    };
    let mut remaining = options.clone();
    remaining.limits.max_archive_entries -= public.len();
    remaining.limits.max_metadata_bytes = remaining.limits.max_metadata_bytes.saturating_sub(
        public
            .keys()
            .map(|path| inventory::entry_cost(path.len()))
            .sum(),
    );
    remaining.limits.max_output_bytes -= public
        .values()
        .flatten()
        .map(|entry| entry.size)
        .sum::<u64>();
    let private = inventory::inspect(&mut decrypted, &remaining)?;
    require_eof(&mut decrypted)?;
    drop(decrypted);
    require_eof(&mut stream)?;
    if magic == WHOLE {
        return Ok(private);
    }
    for scope in &scopes {
        let name = scope
            .to_str()
            .ok_or_else(|| invalid("invalid protected scope"))?;
        if !private.contains_key(name) {
            return Err(invalid("protected scope is missing from archive"));
        }
    }
    if public.keys().any(|name| covered(Path::new(name), &scopes)) {
        return Err(invalid("protected member is in the public catalogue"));
    }
    for (name, entry) in private {
        let path = Path::new(&name);
        if !covered(path, &scopes)
            && !(entry.is_none() && scopes.iter().any(|scope| scope.starts_with(path)))
        {
            return Err(invalid("private member lies outside the protection scopes"));
        }
        match public.entry(name) {
            std::collections::btree_map::Entry::Vacant(slot) => {
                slot.insert(entry);
            }
            std::collections::btree_map::Entry::Occupied(slot)
                if slot.get().is_none() && entry.is_none() => {}
            _ => return Err(invalid("duplicate public and private archive member")),
        }
    }
    inventory::validate_tree(&public)?;
    Ok(public)
}

pub fn health(
    source: &Path,
    features: &ArchiveOptions,
    options: &DecompressionOptions,
) -> Result<serde_json::Value, BatoError> {
    if fs::symlink_metadata(source)?.is_dir() {
        let archive = ArchiveFile::open(source, Access::Read)?;
        let split = archive
            .split
            .ok_or_else(|| invalid("expected a split archive"))?;
        let mut report = split.health(&archive.path)?;
        report["scope"] = "split volumes".into();
        return Ok(report);
    }
    let scope = if features.public_only {
        "public section"
    } else {
        "archive"
    };
    Ok(match inspect(source, features, options) {
        Ok(entries) => serde_json::json!({
            "status":"healthy", "scope":scope, "verified_entries":entries.len()
        }),
        Err(error) => serde_json::json!({
            "status":"unverified", "scope":scope, "error":error.to_string()
        }),
    })
}

pub struct Change {
    pub path: String,
    pub status: &'static str,
}

pub fn diff(
    source: &Path,
    other: &Path,
    features: &ArchiveOptions,
    options: &DecompressionOptions,
) -> Result<Vec<Change>, BatoError> {
    let left = inspect(source, features, options)?;
    let right =
        if fs::symlink_metadata(other)?.is_dir() && !other.join("index.bato").try_exists()? {
            inventory::inspect_directory(other, options)?
        } else {
            inspect(other, features, options)?
        };
    let paths: BTreeSet<_> = left.keys().chain(right.keys()).collect();
    Ok(paths
        .into_iter()
        .filter_map(|path| {
            let status = match (left.get(path), right.get(path)) {
                (None, Some(_)) => "added",
                (Some(_), None) => "removed",
                (before, after) if before != after => "changed",
                _ => return None,
            };
            Some(Change {
                path: path.clone(),
                status,
            })
        })
        .collect())
}

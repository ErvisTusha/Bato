//! Stable grouping of small files with shared byte ranges.
//! Provenance: original file ordering based on byte windows, using Bato's
//! existing SHA-256 implementation.

use super::{FileDigest, INVENTORY_METADATA};
use crate::core::types::BatoError;
use sha2::{Digest, Sha256};
use std::{collections::HashMap, ops::Range, path::PathBuf};

const WINDOW: u64 = 2048;
const INSET: u64 = 1024;
const MIN_FILE: u64 = 8 * 1024;
const MAX_FILE: u64 = 30 * 1024;

#[derive(Clone, Copy)]
pub(super) enum Order {
    Path,
    SharedRanges,
}

pub(super) struct Sketch {
    hash: Sha256,
    ranges: [Range<u64>; 2],
}

impl Sketch {
    pub(super) fn new(size: u64, order: Order) -> Option<Self> {
        (matches!(order, Order::SharedRanges) && (MIN_FILE..=MAX_FILE).contains(&size)).then(|| {
            Self {
                hash: Sha256::new(),
                ranges: [INSET..INSET + WINDOW, size - INSET - WINDOW..size - INSET],
            }
        })
    }

    pub(super) fn update(&mut self, offset: u64, bytes: &[u8]) {
        for range in &self.ranges {
            let start = range.start.max(offset);
            let end = range.end.min(offset + bytes.len() as u64);
            if start < end {
                self.hash
                    .update(&bytes[(start - offset) as usize..(end - offset) as usize]);
            }
        }
    }

    pub(super) fn finish(self) -> [u8; 32] {
        self.hash.finalize().into()
    }
}

pub(super) struct Member {
    pub(super) path: (PathBuf, String),
    pub(super) digest: FileDigest,
    key: Option<[u8; 32]>,
    position: (usize, usize),
}

impl Member {
    pub(super) fn new(
        path: (PathBuf, String),
        (digest, key): (FileDigest, Option<[u8; 32]>),
        index: usize,
    ) -> Self {
        Self {
            path,
            digest,
            key,
            position: (index, index),
        }
    }
}

/// Pairs two distinct members whose sampled bytes agree. Unpaired members
/// retain their relative order. Whole-file deduplication runs after this.
pub(super) fn group(members: &mut [Member]) -> Result<(), BatoError> {
    let total = members
        .iter()
        .fold(0_u64, |bytes, member| bytes.saturating_add(member.digest.0));
    if !(128 * 1024..=256 * 1024).contains(&total) {
        return Ok(());
    }
    let mut leaders = HashMap::<[u8; 32], (usize, [u8; 32], bool)>::new();
    let mut paired = false;
    for member in members.iter_mut() {
        let Some(key) = member.key else { continue };
        if let Some((leader, digest, used)) = leaders.get_mut(&key) {
            if !*used && *digest != member.digest.1 {
                member.position.0 = *leader;
                *used = true;
                paired = true;
            }
        } else {
            leaders
                .try_reserve(1)
                .map_err(|_| BatoError::AllocationFailed {
                    bytes: INVENTORY_METADATA,
                })?;
            leaders.insert(key, (member.position.0, member.digest.1, false));
        }
    }
    if paired {
        members.sort_unstable_by_key(|member| member.position);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sketch_is_invariant_to_short_reads_and_body_edits() {
        let mut input = vec![31; 13_000];
        let mut single = Sketch::new(input.len() as u64, Order::SharedRanges).unwrap();
        single.update(0, &input);
        input[8000] = 42;
        input.insert(8001, 91);
        let mut streaming = Sketch::new(input.len() as u64, Order::SharedRanges).unwrap();
        for (index, part) in input.chunks(93).enumerate() {
            streaming.update((index * 93) as u64, part);
        }
        assert_eq!(single.finish(), streaming.finish());
    }

    #[test]
    fn pairs_shared_ranges_and_keeps_exact_duplicates_separate() {
        let mut members: Vec<_> = [(8, 1), (9, 2), (8, 1), (8, 3), (8, 4), (0, 5), (0, 6)]
            .into_iter()
            .enumerate()
            .map(|(index, (key, digest))| {
                Member::new(
                    (PathBuf::from(index.to_string()), index.to_string()),
                    (
                        (22 * 1024, [digest; 32], 0),
                        (key != 0).then_some([key; 32]),
                    ),
                    index,
                )
            })
            .collect();
        group(&mut members).unwrap();
        let actual: Vec<_> = members
            .iter()
            .map(|member| member.path.1.as_str())
            .collect();
        assert_eq!(actual, ["0", "3", "1", "2", "4", "5", "6"]);
    }
}

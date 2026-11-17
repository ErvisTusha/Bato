//! A stream's workers can clone the same exact trained predictor.
//! Provenance: Bato's existing same-segment primer snapshots.

use super::{
    descriptor::Descriptor,
    predictor::{self, Predictor},
    profile::{Step, Timing},
    segment::{prime, PRIMER},
    simd::Dispatch,
};
use crate::core::memory::{HostBudget, MemoryLease};
use std::sync::{Arc, OnceLock};

const MAX_SEGMENT: usize = 256 * 1024;
const OUTPUT_HEADROOM: usize = 256 * 1024 * 1024;

pub(crate) struct PrimerCache {
    template: OnceLock<Option<Predictor>>,
    length: usize,
    table_log2: u8,
    _memory: MemoryLease,
}

impl PrimerCache {
    pub(crate) fn open(
        descriptor: Descriptor,
        host: &Arc<HostBudget>,
        (workers, reserved): (usize, usize),
    ) -> Option<Self> {
        if workers < 2 || descriptor.segment_bytes() > MAX_SEGMENT {
            return None;
        }
        let length = descriptor.segment_bytes() + PRIMER.len();
        let bytes = predictor::memory_required(length, descriptor.table_log2);
        if host.available().saturating_sub(reserved) < bytes + OUTPUT_HEADROOM {
            return None;
        }
        Some(Self {
            template: OnceLock::new(),
            length,
            table_log2: descriptor.table_log2,
            _memory: host.reserve(bytes).ok()?,
        })
    }

    pub(super) fn supports(&self, length: usize, table_log2: u8) -> bool {
        // The match index and context table are the only model allocations
        // that depend on length. Their constructors round to powers of two
        // before clamping to a cap. Both shapes match in this length class.
        table_log2 == self.table_log2
            && length.next_power_of_two() == self.length.next_power_of_two()
    }

    pub(super) fn model(&self, timing: Timing<'_>) -> Option<Predictor> {
        self.template
            .get_or_init(|| {
                let mut model =
                    Predictor::new(self.length, self.table_log2, Dispatch::Native).ok()?;
                timing.time(Step::PrimerTraining, || prime(&mut model, PRIMER));
                Some(model)
            })
            .as_ref()
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        memory::HostBudget,
        types::{ExecutionStats, MemoryBudget},
    };

    #[test]
    fn a_shared_template_matches_fresh_training_for_its_length_class() {
        let descriptor = Descriptor {
            segment_log2: 16,
            chunk_log2: 14,
            table_log2: 14,
        };
        let stats = Arc::new(ExecutionStats::default());
        let host = HostBudget::new(MemoryBudget::Bytes(512 * 1024 * 1024), &stats).unwrap();
        let cache = PrimerCache::open(descriptor, &host, (2, 0)).unwrap();
        let length = PRIMER.len() + 43 * 1024;
        assert!(cache.supports(length, descriptor.table_log2));
        assert!(!cache.supports(length, descriptor.table_log2 + 1));
        assert!(!cache.supports(PRIMER.len() + 1, descriptor.table_log2));

        let mut fresh = Predictor::new(length, descriptor.table_log2, Dispatch::Native).unwrap();
        prime(&mut fresh, PRIMER);
        let mut cached = cache.model(Timing::off()).unwrap();
        let mut history = PRIMER.to_vec();
        for byte in b"The same primed predictions continue across all eight bits.".repeat(4) {
            for shift in (0..8).rev() {
                assert_eq!(fresh.probability(&history), cached.probability(&history));
                let bit = (byte >> shift) & 1;
                fresh.observe(bit);
                cached.observe(bit);
            }
            history.push(byte);
        }
        assert!(PrimerCache::open(descriptor, &host, (2, host.available())).is_none());
    }
}

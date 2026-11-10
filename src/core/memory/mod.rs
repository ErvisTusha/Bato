//! One startup allowance and shared, lifetime-based managed-memory admission.
//! Reservations conservatively cover Rust buffers and scratch, not process RSS.
use super::types::{BatoError, ExecutionStats};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

mod available;
use available::effective_available_memory;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryBudget {
    #[default]
    Auto,
    Bytes(usize),
}

impl std::str::FromStr for MemoryBudget {
    type Err = BatoError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        match value.parse::<usize>() {
            Ok(n) if n != 0 => Ok(Self::Bytes(n)),
            _ => Err(BatoError::InvalidOptions(
                "memory budget must be auto or a positive byte count".into(),
            )),
        }
    }
}

fn resolve(policy: MemoryBudget, available: usize) -> Result<usize, BatoError> {
    // Exact floor(9*available/10), without intermediate overflow or floats.
    let automatic = (available / 10) * 9 + ((available % 10) * 9) / 10;
    let limit = match policy {
        MemoryBudget::Auto => automatic,
        MemoryBudget::Bytes(0) => {
            return Err(BatoError::InvalidOptions(
                "memory budget must be positive".into(),
            ))
        }
        MemoryBudget::Bytes(n) => n.min(automatic),
    };
    if limit == 0 {
        return Err(BatoError::MemoryLimitExceeded {
            required: 1,
            limit: 0,
        });
    }
    Ok(limit)
}

impl MemoryBudget {
    /// Probe the host once. A caller can freeze this absolute allowance for
    /// subsequent paired runs by explicitly supplying `Bytes(resolved)`.
    pub fn resolve(self) -> Result<usize, BatoError> {
        resolve(self, effective_available_memory()?)
    }
}

#[derive(Debug)]
pub(crate) struct HostBudget {
    limit: usize,
    used: AtomicUsize,
    peak: AtomicUsize,
    stats: Arc<ExecutionStats>,
}

impl HostBudget {
    pub fn new(policy: MemoryBudget, stats: &Arc<ExecutionStats>) -> Result<Arc<Self>, BatoError> {
        let limit = policy.resolve()?;
        stats.update(|s| {
            *s = Default::default();
            s.memory_budget = limit;
        });
        Ok(Arc::new(Self {
            limit,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            stats: stats.clone(),
        }))
    }
    pub fn limit(&self) -> usize {
        self.limit
    }
    pub fn used(&self) -> usize {
        self.used.load(Ordering::Acquire)
    }
    pub fn available(&self) -> usize {
        self.limit - self.used()
    }
    fn charge(&self, bytes: usize) -> Result<(), BatoError> {
        let mut before = self.used.load(Ordering::Acquire);
        let total = loop {
            let total = before
                .checked_add(bytes)
                .filter(|&n| n <= self.limit)
                .ok_or(BatoError::MemoryLimitExceeded {
                    required: before.saturating_add(bytes),
                    limit: self.limit,
                })?;
            match self.used.compare_exchange_weak(
                before,
                total,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break total,
                Err(current) => before = current,
            }
        };
        if total > self.peak.fetch_max(total, Ordering::AcqRel) {
            self.stats
                .update(|s| s.managed_host_peak = s.managed_host_peak.max(total));
        }
        Ok(())
    }
    pub fn reserve(self: &Arc<Self>, bytes: usize) -> Result<MemoryLease, BatoError> {
        self.charge(bytes)?;
        Ok(MemoryLease {
            budget: self.clone(),
            bytes,
        })
    }
}

#[derive(Debug)]
pub(crate) struct MemoryLease {
    budget: Arc<HostBudget>,
    bytes: usize,
}
impl MemoryLease {
    pub fn grow(&mut self, total: usize) -> Result<(), BatoError> {
        if total > self.bytes {
            self.budget.charge(total - self.bytes)?;
            self.bytes = total;
        }
        Ok(())
    }
    fn resize(&mut self, total: usize) -> Result<(), BatoError> {
        if total < self.bytes {
            self.budget
                .used
                .fetch_sub(self.bytes - total, Ordering::AcqRel);
            self.bytes = total;
        } else {
            self.grow(total)?;
        }
        Ok(())
    }
}
impl Drop for MemoryLease {
    fn drop(&mut self) {
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

pub(crate) fn try_capacity<T>(size: usize) -> Result<Vec<T>, BatoError> {
    let allocation = size
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(BatoError::AllocationFailed { bytes: usize::MAX })?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(size)
        .map_err(|_| BatoError::AllocationFailed { bytes: allocation })?;
    Ok(values)
}
pub(crate) fn try_reserve<T>(values: &mut Vec<T>, additional: usize) -> Result<(), BatoError> {
    let bytes = values
        .len()
        .checked_add(additional)
        .and_then(|n| n.checked_mul(std::mem::size_of::<T>()))
        .ok_or(BatoError::AllocationFailed { bytes: usize::MAX })?;
    values
        .try_reserve(additional)
        .map_err(|_| BatoError::AllocationFailed { bytes })
}
pub(crate) fn try_filled<T: Clone>(size: usize, value: T) -> Result<Vec<T>, BatoError> {
    let mut values = try_capacity(size)?;
    values.resize(size, value);
    Ok(values)
}
pub(crate) fn try_copy(input: &[u8]) -> Result<Vec<u8>, BatoError> {
    let mut values = try_capacity(input.len())?;
    values.extend_from_slice(input);
    Ok(values)
}
pub(crate) fn try_buffer(size: usize) -> Result<Vec<u8>, BatoError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| BatoError::AllocationFailed { bytes: size })?;
    bytes.resize(size, 0);
    Ok(bytes)
}

pub(crate) struct ManagedVec {
    bytes: Vec<u8>,
    lease: MemoryLease,
}
impl ManagedVec {
    pub fn new(host: &Arc<HostBudget>) -> Result<Self, BatoError> {
        Ok(Self {
            bytes: Vec::new(),
            lease: host.reserve(0)?,
        })
    }
    pub fn into_vec(self) -> Vec<u8> {
        self.bytes
    }
}
impl std::io::Write for ManagedVec {
    fn write(&mut self, input: &[u8]) -> std::io::Result<usize> {
        let needed = self.bytes.len().checked_add(input.len()).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::OutOfMemory, "output size overflow")
        })?;
        if needed > self.bytes.capacity() {
            let old = self.bytes.capacity();
            let mut target = needed.max(old.checked_mul(2).unwrap_or(needed));
            let simultaneous = old.checked_add(target).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    "output allocation overflow",
                )
            })?;
            if self.lease.grow(simultaneous).is_err() {
                // Buffer capacity may shrink under admission pressure; this
                // does not alter codec effort, models, or encoded bytes.
                target = needed;
                let minimum = old.checked_add(target).ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::OutOfMemory,
                        "output allocation overflow",
                    )
                })?;
                self.lease.grow(minimum).map_err(std::io::Error::other)?;
            }
            if self
                .bytes
                .try_reserve_exact(target - self.bytes.len())
                .is_err()
            {
                self.lease.resize(old).map_err(std::io::Error::other)?;
                return Err(std::io::Error::new(
                    std::io::ErrorKind::OutOfMemory,
                    BatoError::AllocationFailed { bytes: target },
                ));
            }
            self.lease
                .resize(self.bytes.capacity())
                .map_err(std::io::Error::other)?;
        }
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn policy_uses_ninety_percent_and_lower_explicit_limits_without_overflow() {
        assert_eq!(resolve(MemoryBudget::Auto, 101).unwrap(), 90);
        assert_eq!(resolve(MemoryBudget::Bytes(45), 101).unwrap(), 45);
        assert_eq!(resolve(MemoryBudget::Bytes(1000), 101).unwrap(), 90);
        assert_eq!(
            resolve(MemoryBudget::Auto, usize::MAX).unwrap(),
            (usize::MAX / 10) * 9 + ((usize::MAX % 10) * 9) / 10
        );
        assert!(resolve(MemoryBudget::Auto, 0).is_err());
        assert!(resolve(MemoryBudget::Bytes(0), 101).is_err());
        for s in ["0", "-1", "invalid"] {
            assert!(s.parse::<MemoryBudget>().is_err());
        }
    }
    #[test]
    fn leases_share_one_allowance_and_release_only_at_end_of_lifetime() {
        let b = Arc::new(HostBudget {
            limit: 100,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            stats: Arc::default(),
        });
        let mut first = b.reserve(60).unwrap();
        let second = b.reserve(40).unwrap();
        assert!(first.grow(61).is_err());
        assert_eq!(b.used(), 100);
        drop(second);
        first.grow(80).unwrap();
        assert_eq!(b.used(), 80);
        assert!(b.reserve(usize::MAX).is_err());
        drop(first);
        assert_eq!(b.used(), 0);
        assert_eq!(b.stats.snapshot().managed_host_peak, 100);
    }
    #[test]
    fn concurrent_workers_cannot_each_spend_the_full_allowance() {
        let b = Arc::new(HostBudget {
            limit: 1000,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            stats: Arc::default(),
        });
        let start = Arc::new(std::sync::Barrier::new(8));
        let finish = Arc::new(std::sync::Barrier::new(8));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let (b, start, finish) = (b.clone(), start.clone(), finish.clone());
            workers.push(std::thread::spawn(move || {
                start.wait();
                let lease = b.reserve(200);
                finish.wait();
                assert!(b.used() <= 1000);
                lease.is_ok()
            }));
        }
        assert_eq!(
            workers
                .into_iter()
                .map(|w| w.join().unwrap())
                .filter(|&ok| ok)
                .count(),
            5
        );
        assert_eq!(b.used(), 0);
    }
    #[test]
    fn output_reallocation_accounts_for_old_and_new_buffers_together() {
        use std::io::Write;
        let b = Arc::new(HostBudget {
            limit: 115,
            used: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            stats: Arc::default(),
        });
        let mut output = ManagedVec::new(&b).unwrap();
        output.write_all(&[1; 40]).unwrap();
        output.write_all(&[2; 30]).unwrap();
        assert_eq!(b.used(), 70);
        assert_eq!(b.peak.load(Ordering::Relaxed), 110);
        let error = BatoError::from(output.write_all(&[3; 10]).unwrap_err());
        assert!(matches!(error, BatoError::MemoryLimitExceeded { .. }));
        assert_eq!(output.bytes.len(), 70);
        drop(output);
        assert_eq!(b.used(), 0);
    }
}

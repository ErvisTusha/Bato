//! Device admission for the archive byte filters.
//! Provenance: original scheduling; the CPU and both device backends
//! implement the same integer transforms.

use super::{
    cuda::{self, Cuda},
    wgpu_filters::WgpuFilters,
    Gpu, GpuError, NVIDIA_VENDOR,
};
use crate::core::{
    memory::try_filled,
    types::{BatoError, Compute, ExecutionStats, TransformId},
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex, MutexGuard, OnceLock, TryLockError,
};

/// Bound a single transfer and its temporary device buffers.
pub(super) const MAX_BYTES: usize = 64 * 1024 * 1024;
/// Small inputs rarely repay device startup and host transfers.
const AUTO_MIN_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy)]
pub(crate) enum Direction {
    Forward,
    Inverse,
}

#[derive(Default)]
struct DeviceState {
    cuda_error: Option<String>,
    wgpu_error: Option<String>,
    wgpu: Option<WgpuFilters>,
}

struct DeviceSlot {
    cuda: Option<i32>,
    adapter: Option<wgpu::Adapter>,
    // With no matching PCI ID, a wgpu NVIDIA adapter may be a device we
    // already have via CUDA. It is eligible after all CUDA devices fail.
    deferred: bool,
    state: Mutex<DeviceState>,
}

impl DeviceSlot {
    fn new(cuda: Option<i32>, adapter: Option<wgpu::Adapter>, deferred: bool) -> Self {
        Self {
            cuda,
            adapter,
            deferred,
            state: Mutex::default(),
        }
    }

    fn acquire(&self, compute: Compute) -> Result<Option<MutexGuard<'_, DeviceState>>, GpuError> {
        let result = if compute == Compute::Auto {
            self.state.try_lock()
        } else {
            self.state.lock().map_err(TryLockError::Poisoned)
        };
        match result {
            Ok(state) => Ok(Some(state)),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Poisoned(_)) => {
                Err(GpuError::Execution("device slot poisoned".into()))
            }
        }
    }
}

struct Devices {
    slots: Vec<DeviceSlot>,
    active_cuda: AtomicUsize,
    discovery_error: Option<String>,
}

fn pci_identity(value: &str) -> Option<[u32; 4]> {
    let mut fields = value.split([':', '.']);
    let mut identity = [0; 4];
    for field in &mut identity {
        *field = u32::from_str_radix(fields.next()?, 16).ok()?;
    }
    fields.next().is_none().then_some(identity)
}

impl Devices {
    fn discover(compute: Compute) -> Self {
        let mut discovery_error = None;
        let cuda = match compute {
            Compute::Auto | Compute::Cuda => Cuda::adapters().unwrap_or_else(|error| {
                discovery_error = Some(error.to_string());
                Vec::new()
            }),
            _ => Vec::new(),
        };
        let mut wgpu = if matches!(compute, Compute::Auto | Compute::Wgpu) {
            Gpu::adapters()
                .into_iter()
                .filter(Gpu::hardware_adapter)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let has_unknown_cuda = cuda
            .iter()
            .any(|item| pci_identity(&item.pci_bus_id).is_none());
        let mut slots = Vec::new();
        for device in &cuda {
            let identity = pci_identity(&device.pci_bus_id);
            // Identical adapter models can report the same PCI ID through
            // DX12. Pair only when both APIs identify a unique device.
            let paired = identity
                .filter(|identity| {
                    cuda.iter()
                        .filter(|candidate| pci_identity(&candidate.pci_bus_id) == Some(*identity))
                        .count()
                        == 1
                })
                .and_then(|identity| {
                    let mut matching = wgpu.iter().enumerate().filter(|(_, adapter)| {
                        let info = adapter.get_info();
                        info.vendor == NVIDIA_VENDOR
                            && pci_identity(&info.device_pci_bus_id) == Some(identity)
                    });
                    let (index, _) = matching.next()?;
                    matching.next().is_none().then_some(index)
                })
                .map(|index| wgpu.remove(index));
            slots.push(DeviceSlot::new(Some(device.ordinal), paired, false));
        }
        for adapter in wgpu {
            let info = adapter.get_info();
            let identity = pci_identity(&info.device_pci_bus_id);
            let deferred = !cuda.is_empty()
                && info.vendor == NVIDIA_VENDOR
                && (has_unknown_cuda
                    || identity.is_none()
                    || cuda
                        .iter()
                        .any(|item| pci_identity(&item.pci_bus_id) == identity));
            slots.push(DeviceSlot::new(None, Some(adapter), deferred));
        }
        Self {
            slots,
            active_cuda: AtomicUsize::new(cuda.len()),
            discovery_error,
        }
    }
}

/// One lazily opened slot per detected GPU, shared by the archive's workers.
/// The CPU path never opens a driver. Each device has at most one filter
/// transfer in flight, independently of the other devices.
#[derive(Default)]
pub(crate) struct FilterPool(OnceLock<Devices>);

#[cfg(test)]
impl FilterPool {
    pub(crate) fn wgpu_for_test() -> Option<Self> {
        let adapter = Gpu::adapters().into_iter().next()?;
        Some(Self(OnceLock::from(Devices {
            slots: vec![DeviceSlot::new(None, Some(adapter), false)],
            active_cuda: AtomicUsize::new(0),
            discovery_error: None,
        })))
    }
}

pub(crate) struct WorkerFilters<'a> {
    compute: Compute,
    pool: &'a FilterPool,
    stats: &'a ExecutionStats,
    worker: usize,
    // CUDA contexts and their resources stay on the thread that opened them.
    // Release the current context's resources before opening another ordinal.
    cuda: Option<(i32, Cuda)>,
}

impl<'a> WorkerFilters<'a> {
    pub(crate) fn new(
        pool: &'a FilterPool,
        (compute, stats): (Compute, &'a ExecutionStats),
        worker: usize,
    ) -> Self {
        Self {
            compute,
            pool,
            stats,
            worker,
            cuda: None,
        }
    }

    pub(crate) fn run(
        &mut self,
        input: &[u8],
        transform: TransformId,
        direction: Direction,
    ) -> Result<Option<Vec<u8>>, BatoError> {
        let minimum = match self.compute {
            Compute::Cpu => return Ok(None),
            Compute::Auto => AUTO_MIN_BYTES,
            Compute::Cuda | Compute::Wgpu => 1,
        };
        if !(minimum..=MAX_BYTES).contains(&input.len()) {
            return Ok(None);
        }
        let devices = self.pool.0.get_or_init(|| Devices::discover(self.compute));
        let mut reason = devices.discovery_error.clone();
        let mut output = None;
        let slot_count = devices.slots.len();
        for deferred in [false, true] {
            for offset in 0..slot_count {
                let index = (self.worker + offset) % slot_count;
                let slot = &devices.slots[index];
                if slot.deferred != deferred
                    || (deferred && devices.active_cuda.load(Ordering::Relaxed) != 0)
                {
                    continue;
                }
                let mut state = match slot.acquire(self.compute) {
                    Ok(Some(state)) => state,
                    Ok(None) => continue,
                    Err(error) => {
                        reason = Some(error.to_string());
                        continue;
                    }
                };
                if let Some(error) = state.wgpu_error.as_ref().or(state.cuda_error.as_ref()) {
                    reason = Some(error.clone());
                }
                let cuda = slot.cuda.filter(|_| state.cuda_error.is_none());
                let adapter = slot.adapter.as_ref().filter(|_| state.wgpu_error.is_none());
                if cuda.is_none() && adapter.is_none() {
                    continue;
                }
                let bytes = match &mut output {
                    Some(bytes) => bytes,
                    None => match try_filled(input.len(), 0u8) {
                        Ok(bytes) => output.insert(bytes),
                        Err(error) if self.compute == Compute::Auto => {
                            self.stats
                                .update(|stats| stats.gpu_fallback = error.to_string());
                            return Ok(None);
                        }
                        Err(error) => return Err(error),
                    },
                };
                if let Some(ordinal) = cuda {
                    match apply_cuda(
                        &mut self.cuda,
                        ordinal,
                        (input, bytes),
                        (transform, direction),
                    ) {
                        Ok((launches, device)) => {
                            self.stats.update(|stats| {
                                stats.cuda_filter_dispatches += launches;
                                stats.cuda_filter_bytes += input.len() as u64;
                                stats.cuda_device = device.clone();
                                if !stats.gpu_devices.contains(&device) {
                                    stats.gpu_devices.push(device);
                                }
                            });
                            return Ok(output);
                        }
                        Err(error) => {
                            if self.compute == Compute::Auto {
                                self.stats
                                    .update(|stats| stats.cuda_fallback = error.to_string());
                            }
                            state.cuda_error = Some(error.to_string());
                            devices.active_cuda.fetch_sub(1, Ordering::Relaxed);
                        }
                    }
                    reason.clone_from(&state.cuda_error);
                }
                if let Some(adapter) = adapter {
                    let result = match &state.wgpu {
                        Some(filter) => Ok(filter),
                        None => {
                            WgpuFilters::open(adapter).map(|filter| &*state.wgpu.insert(filter))
                        }
                    }
                    .and_then(|filter| filter.apply(input, bytes, (transform, direction)));
                    match result {
                        Ok(launches) => {
                            let info = adapter.get_info();
                            let device =
                                format!("{} ({:?}, wgpu #{index})", info.name, info.backend);
                            self.stats.update(|stats| {
                                stats.wgpu_filter_dispatches += launches;
                                stats.wgpu_filter_bytes += input.len() as u64;
                                if !stats.gpu_devices.contains(&device) {
                                    stats.gpu_devices.push(device);
                                }
                            });
                            return Ok(output);
                        }
                        Err(error) => {
                            state.wgpu = None;
                            state.wgpu_error = Some(error.to_string());
                        }
                    }
                    reason.clone_from(&state.wgpu_error);
                }
            }
        }
        if self.compute == Compute::Auto {
            if let Some(reason) = reason {
                self.stats.update(|stats| {
                    stats.gpu_fallback = reason;
                    if stats.cuda_fallback.is_empty() {
                        if let Some(error) = &devices.discovery_error {
                            stats.cuda_fallback.clone_from(error);
                        }
                    }
                });
            }
            return Ok(None);
        }
        let reason = reason.unwrap_or_else(|| GpuError::NoAdapter.to_string());
        match self.compute {
            Compute::Cuda => Err(BatoError::Cuda(reason)),
            _ => Err(BatoError::Wgpu(reason)),
        }
    }
}

fn apply_cuda(
    active: &mut Option<(i32, Cuda)>,
    ordinal: i32,
    (input, output): (&[u8], &mut [u8]),
    filter: (TransformId, Direction),
) -> Result<(u64, String), GpuError> {
    if active
        .as_ref()
        .is_none_or(|(current, _)| *current != ordinal)
    {
        // Drop before opening the replacement while this context is still current.
        *active = None;
        *active = Some((ordinal, Cuda::open_for_filters(ordinal)?));
    }
    let (_, cuda) = active
        .as_ref()
        .ok_or_else(|| GpuError::Device("CUDA session unavailable".into()))?;
    let launches = cuda::filters::apply(cuda, input, output, filter)?;
    Ok((launches, format!("{} #{}", cuda.description(), ordinal)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pci_ids_normalize_the_domain_and_hex_case() {
        assert_eq!(
            pci_identity("00000000:2A:00.0"),
            pci_identity("0000:2a:00.0")
        );
        assert_ne!(pci_identity("0001:2a:00.0"), pci_identity("0000:2a:00.0"));
        assert!(pci_identity("").is_none());
    }
}

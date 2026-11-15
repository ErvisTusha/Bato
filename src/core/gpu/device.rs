//! GPU device discovery, PCI-identity de-duplication and device admission.
//! Provenance: original Bato GPU scheduling, extracted without changing behavior.

use super::{
    cuda::Cuda,
    wgpu_backend::WgpuFilters,
    Gpu, GpuError, NVIDIA_VENDOR,
};
use crate::core::types::Compute;
use std::sync::{
    atomic::AtomicUsize,
    Mutex, MutexGuard, OnceLock, TryLockError,
};

#[derive(Default)]
pub(super) struct DeviceState {
    pub(super) cuda_error: Option<String>,
    pub(super) wgpu_error: Option<String>,
    pub(super) wgpu: Option<WgpuFilters>,
}

pub(super) struct DeviceSlot {
    pub(super) cuda: Option<i32>,
    pub(super) adapter: Option<wgpu::Adapter>,
    // With no matching PCI ID, a wgpu NVIDIA adapter may be a device we
    // already have via CUDA. It is eligible after all CUDA devices fail.
    pub(super) deferred: bool,
    pub(super) state: Mutex<DeviceState>,
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

    pub(super) fn acquire(&self, compute: Compute) -> Result<Option<MutexGuard<'_, DeviceState>>, GpuError> {
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

pub(super) struct Devices {
    pub(super) slots: Vec<DeviceSlot>,
    pub(super) active_cuda: AtomicUsize,
    pub(super) discovery_error: Option<String>,
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
    pub(super) fn discover(compute: Compute) -> Self {
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
pub(crate) struct FilterPool(pub(super) OnceLock<Devices>);

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

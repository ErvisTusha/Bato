//! Device discovery and synchronized admission for independently compiled
//! CUDA and wgpu backends. Native SDK plugins have their own discovery layer.
//! PCI identities prevent a CUDA device from also occupying a wgpu slot.

#[cfg(any(feature = "cuda", feature = "wgpu"))]
use super::GpuError;
use crate::core::types::Compute;
use std::sync::OnceLock;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use std::sync::{Mutex, MutexGuard, TryLockError};
#[cfg(feature = "cuda")]
use super::cuda::Cuda;
#[cfg(feature = "cuda")]
use std::sync::atomic::AtomicUsize;
#[cfg(feature = "wgpu")]
use super::backend::wgpu::{Gpu, WgpuFilters};
#[cfg(all(feature = "cuda", feature = "wgpu"))]
use super::backend::wgpu::NVIDIA_VENDOR;

#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[derive(Default)]
pub(super) struct DeviceState {
    #[cfg(feature = "cuda")]
    pub(super) cuda_error: Option<String>,
    #[cfg(feature = "wgpu")]
    pub(super) wgpu_error: Option<String>,
    #[cfg(feature = "wgpu")]
    pub(super) wgpu: Option<WgpuFilters>,
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(super) struct DeviceSlot {
    #[cfg(feature = "cuda")]
    pub(super) cuda: Option<i32>,
    #[cfg(feature = "wgpu")]
    pub(super) adapter: Option<wgpu::Adapter>,
    #[cfg(feature = "wgpu")]
    pub(super) deferred: bool,
    state: Mutex<DeviceState>,
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
impl DeviceSlot {
    pub(super) fn acquire(
        &self,
        compute: Compute,
    ) -> Result<Option<MutexGuard<'_, DeviceState>>, GpuError> {
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

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
pub(super) type DeviceSlot = ();

pub(super) struct Devices {
    pub(super) slots: Vec<DeviceSlot>,
    #[cfg(feature = "cuda")]
    pub(super) active_cuda: AtomicUsize,
    pub(super) discovery_error: Option<String>,
}

#[cfg(all(feature = "cuda", feature = "wgpu"))]
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
        #[cfg(feature = "cuda")]
        let mut discovery_error = None;
        #[cfg(not(feature = "cuda"))]
        let discovery_error = None;
        #[cfg(any(feature = "cuda", feature = "wgpu"))]
        let mut slots = Vec::new();
        #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
        let slots = Vec::new();

        #[cfg(feature = "cuda")]
        let cuda = if matches!(compute, Compute::Auto | Compute::Cuda) {
            Cuda::adapters().unwrap_or_else(|error| {
                discovery_error = Some(error.to_string());
                Vec::new()
            })
        } else {
            Vec::new()
        };

        #[cfg(all(feature = "cuda", feature = "wgpu"))]
        let mut wgpu_adapters = if matches!(compute, Compute::Auto | Compute::Wgpu) {
            Gpu::adapters()
                .into_iter()
                .filter(Gpu::hardware_adapter)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        #[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
        let wgpu_adapters = if matches!(compute, Compute::Auto | Compute::Wgpu) {
            Gpu::adapters()
                .into_iter()
                .filter(Gpu::hardware_adapter)
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        #[cfg(all(feature = "cuda", feature = "wgpu"))]
        let has_unknown_cuda = cuda
            .iter()
            .any(|item| pci_identity(&item.pci_bus_id).is_none());

        #[cfg(feature = "cuda")]
        for device in &cuda {
            #[cfg(feature = "wgpu")]
            let paired = {
                let identity = pci_identity(&device.pci_bus_id);
                identity
                    .filter(|identity| {
                        cuda.iter()
                            .filter(|candidate| {
                                pci_identity(&candidate.pci_bus_id) == Some(*identity)
                            })
                            .count() == 1
                    })
                    .and_then(|identity| {
                        let mut matching = wgpu_adapters.iter().enumerate().filter(|(_, adapter)| {
                            let info = adapter.get_info();
                            info.vendor == NVIDIA_VENDOR
                                && pci_identity(&info.device_pci_bus_id) == Some(identity)
                        });
                        let (index, _) = matching.next()?;
                        matching.next().is_none().then_some(index)
                    })
                    .map(|index| wgpu_adapters.remove(index))
            };
            slots.push(DeviceSlot {
                cuda: Some(device.ordinal),
                #[cfg(feature = "wgpu")]
                adapter: paired,
                #[cfg(feature = "wgpu")]
                deferred: false,
                state: Mutex::default(),
            });
        }

        #[cfg(feature = "wgpu")]
        for adapter in wgpu_adapters {
            #[cfg(feature = "cuda")]
            let deferred = {
                let info = adapter.get_info();
                let identity = pci_identity(&info.device_pci_bus_id);
                !cuda.is_empty()
                    && info.vendor == NVIDIA_VENDOR
                    && (has_unknown_cuda
                        || identity.is_none()
                        || cuda.iter().any(|item| pci_identity(&item.pci_bus_id) == identity))
            };
            #[cfg(not(feature = "cuda"))]
            let deferred = false;
            slots.push(DeviceSlot {
                #[cfg(feature = "cuda")]
                cuda: None,
                adapter: Some(adapter),
                deferred,
                state: Mutex::default(),
            });
        }

        #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
        let _ = compute;

        Self {
            slots,
            #[cfg(feature = "cuda")]
            active_cuda: AtomicUsize::new(cuda.len()),
            discovery_error,
        }
    }
}

/// One shared slot per physical GPU, lazily initialized by archive workers.
#[derive(Default)]
pub(crate) struct FilterPool(pub(super) OnceLock<Devices>);

#[cfg(all(test, feature = "wgpu"))]
impl FilterPool {
    pub(crate) fn wgpu_for_test() -> Option<Self> {
        let adapter = Gpu::adapters().into_iter().next()?;
        Some(Self(OnceLock::from(Devices {
            slots: vec![DeviceSlot {
                #[cfg(feature = "cuda")]
                cuda: None,
                adapter: Some(adapter),
                deferred: false,
                state: Mutex::default(),
            }],
            #[cfg(feature = "cuda")]
            active_cuda: AtomicUsize::new(0),
            discovery_error: None,
        })))
    }
}

#[cfg(all(test, feature = "cuda", feature = "wgpu"))]
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

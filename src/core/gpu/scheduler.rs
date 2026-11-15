//! Per-worker GPU backend selection and CPU fallback for archive byte filters.
//! CUDA and wgpu can compile independently. Native vendor plugins are
//! independently optional, without linking their SDKs into the Rust binary.

use super::{
    device::{Devices, FilterPool},
    filters::{Direction, AUTO_MIN_BYTES, MAX_BYTES},
    GpuError,
};
use crate::core::types::{BatoError, Compute, ExecutionStats, TransformId};
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use crate::core::memory::try_filled;
#[cfg(feature = "cuda")]
use super::cuda::{self, Cuda};
#[cfg(feature = "cuda")]
use std::sync::atomic::Ordering;
#[cfg(feature = "wgpu")]
use super::backend::wgpu::WgpuFilters;
#[cfg(any(feature = "hip", feature = "level-zero", feature = "metal"))]
use super::backend::native::{NativeKind, WorkerNative};

pub(crate) struct WorkerFilters<'a> {
    compute: Compute,
    pool: &'a FilterPool,
    stats: &'a ExecutionStats,
    worker: usize,
    // CUDA sessions remain on the thread that opened the driver context.
    #[cfg(feature = "cuda")]
    cuda: Option<(i32, Cuda)>,
    #[cfg(any(feature = "hip", feature = "level-zero", feature = "metal"))]
    native: Option<WorkerNative>,
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
            #[cfg(feature = "cuda")]
            cuda: None,
            #[cfg(any(feature = "hip", feature = "level-zero", feature = "metal"))]
            native: None,
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
            Compute::Cuda | Compute::Wgpu | Compute::Hip | Compute::LevelZero | Compute::Metal => {
                1
            }
        };
        if !(minimum..=MAX_BYTES).contains(&input.len()) {
            return Ok(None);
        }

        #[cfg(not(feature = "cuda"))]
        if self.compute == Compute::Cuda {
            return Err(BatoError::Cuda("CUDA backend was not compiled".into()));
        }
        #[cfg(not(feature = "wgpu"))]
        if self.compute == Compute::Wgpu {
            return Err(BatoError::Wgpu("wgpu backend was not compiled".into()));
        }

        #[cfg(any(feature = "hip", feature = "level-zero", feature = "metal"))]
        {
            let selected = match self.compute {
                Compute::Hip => Some(NativeKind::Hip),
                Compute::LevelZero => Some(NativeKind::LevelZero),
                Compute::Metal => Some(NativeKind::Metal),
                _ => None,
            };
            if let Some(kind) = selected {
                return self.try_native(input, transform, direction, Some(kind));
            }
        }
        #[cfg(not(any(feature = "hip", feature = "level-zero", feature = "metal")))]
        if matches!(self.compute, Compute::Hip | Compute::LevelZero | Compute::Metal) {
            return Err(BatoError::Native(
                "native backend feature was not compiled".into(),
            ));
        }

        let devices = self.pool.0.get_or_init(|| Devices::discover(self.compute));
        let discovery_error = devices.discovery_error.clone();

        #[cfg(any(feature = "hip", feature = "level-zero", feature = "metal"))]
        let native_count = if self.compute == Compute::Auto {
            self.native
                .get_or_insert_with(WorkerNative::discover)
                .device_count()
        } else {
            0
        };
        #[cfg(any(feature = "hip", feature = "level-zero", feature = "metal"))]
        let prefer_native = native_count > 0
            && self.worker % (native_count + devices.slots.len()) < native_count;
        #[cfg(any(feature = "hip", feature = "level-zero", feature = "metal"))]
        if prefer_native {
            if let Some(output) = self.try_native(input, transform, direction, None)? {
                return Ok(Some(output));
            }
        }

        #[cfg(any(feature = "cuda", feature = "wgpu"))]
        let mut reason = discovery_error.clone();
        #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
        let reason = discovery_error.clone();
        #[cfg(any(feature = "cuda", feature = "wgpu"))]
        {
            let mut output = None;
            let slot_count = devices.slots.len();
            for deferred in [false, true] {
                for offset in 0..slot_count {
                    let index = (self.worker + offset) % slot_count;
                    let slot = &devices.slots[index];

                    #[cfg(feature = "wgpu")]
                    if slot.deferred != deferred {
                        continue;
                    }
                    #[cfg(not(feature = "wgpu"))]
                    if deferred {
                        continue;
                    }
                    #[cfg(all(feature = "cuda", feature = "wgpu"))]
                    if deferred && devices.active_cuda.load(Ordering::Relaxed) != 0 {
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
                    #[cfg(feature = "cuda")]
                    if let Some(error) = state.cuda_error.as_ref() {
                        reason = Some(error.clone());
                    }
                    #[cfg(feature = "wgpu")]
                    if let Some(error) = state.wgpu_error.as_ref() {
                        reason = Some(error.clone());
                    }

                    #[cfg(feature = "cuda")]
                    let cuda = slot.cuda.filter(|_| state.cuda_error.is_none());
                    #[cfg(feature = "wgpu")]
                    let adapter = slot.adapter.as_ref().filter(|_| state.wgpu_error.is_none());
                    #[cfg(all(feature = "cuda", feature = "wgpu"))]
                    if cuda.is_none() && adapter.is_none() {
                        continue;
                    }
                    #[cfg(all(feature = "cuda", not(feature = "wgpu")))]
                    if cuda.is_none() {
                        continue;
                    }
                    #[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
                    if adapter.is_none() {
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

                    #[cfg(feature = "cuda")]
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

                    #[cfg(feature = "wgpu")]
                    if let Some(adapter) = adapter {
                        let result = match &state.wgpu {
                            Some(filter) => Ok(filter),
                            None => WgpuFilters::open(adapter)
                                .map(|filter| &*state.wgpu.insert(filter)),
                        }
                        .and_then(|filter| filter.apply(input, bytes, (transform, direction)));
                        match result {
                            Ok(launches) => {
                                let info = adapter.get_info();
                                let device = format!(
                                    "{} ({:?}, wgpu #{index})",
                                    info.name, info.backend
                                );
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
        }

        #[cfg(any(feature = "hip", feature = "level-zero", feature = "metal"))]
        if self.compute == Compute::Auto && !prefer_native && native_count > 0 {
            if let Some(output) = self.try_native(input, transform, direction, None)? {
                return Ok(Some(output));
            }
        }

        if self.compute == Compute::Auto {
            if let Some(reason) = reason {
                self.stats.update(|stats| {
                    stats.gpu_fallback = reason;
                    #[cfg(feature = "cuda")]
                    if stats.cuda_fallback.is_empty() {
                        if let Some(error) = &discovery_error {
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
            Compute::Wgpu => Err(BatoError::Wgpu(reason)),
            _ => Err(BatoError::Native(reason)),
        }
    }

    #[cfg(any(feature = "hip", feature = "level-zero", feature = "metal"))]
    fn try_native(
        &mut self,
        input: &[u8],
        transform: TransformId,
        direction: Direction,
        selected: Option<NativeKind>,
    ) -> Result<Option<Vec<u8>>, BatoError> {
        let native = self.native.get_or_insert_with(WorkerNative::discover);
        match native.run(self.worker, input, transform, direction, selected) {
            Ok(Some((output, device))) => {
                self.stats.update(|stats| {
                    stats.native_filter_dispatches += 1;
                    stats.native_filter_bytes += input.len() as u64;
                    stats.native_device = device.clone();
                    if !stats.gpu_devices.contains(&device) {
                        stats.gpu_devices.push(device);
                    }
                });
                Ok(Some(output))
            }
            Ok(None) if selected.is_none() => Ok(None),
            Ok(None) => Err(BatoError::Native("requested native GPU is unavailable".into())),
            Err(error) if selected.is_none() => {
                self.stats.update(|stats| stats.gpu_fallback = error.to_string());
                Ok(None)
            }
            Err(error) => Err(BatoError::Native(error.to_string())),
        }
    }
}

#[cfg(feature = "cuda")]
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
        // Drop resources before opening a different CUDA context.
        *active = None;
        *active = Some((ordinal, Cuda::open_for_filters(ordinal)?));
    }
    let (_, cuda) = active
        .as_ref()
        .ok_or_else(|| GpuError::Device("CUDA session unavailable".into()))?;
    let launches = cuda::filters::apply(cuda, input, output, filter)?;
    Ok((launches, format!("{} #{}", cuda.description(), ordinal)))
}

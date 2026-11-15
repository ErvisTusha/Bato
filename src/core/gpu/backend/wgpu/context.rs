//! Portable GPU device and compute helpers for the wgpu backend.
//! Vulkan, Metal and Direct3D 12 use one shared Rust implementation.

use super::super::super::GpuError;
#[cfg(feature = "cuda")]
use std::fmt;
use wgpu::util::DeviceExt;

/// NVIDIA PCI vendor ID used only to rank wgpu adapters.
pub(in crate::core::gpu) const NVIDIA_VENDOR: u32 = 0x10de;
/// Bytes in a packed 32-bit GPU word.
const WORD: usize = 4;

/// An opened compute device; clones share it.
#[derive(Clone)]
pub struct Gpu {
    pub(in crate::core::gpu) device: wgpu::Device,
    pub(in crate::core::gpu) queue: wgpu::Queue,
    pub(in crate::core::gpu) info: wgpu::AdapterInfo,
}

#[cfg(feature = "cuda")]
/// A compiled compute shader with its buffers bound.
pub(in crate::core::gpu) struct Kernel {
    pub(in crate::core::gpu) pipeline: wgpu::ComputePipeline,
    pub(in crate::core::gpu) bindings: wgpu::BindGroup,
}

/// Preference of an adapter, lower first: NVIDIA discrete, any discrete,
/// integrated, virtual, then software and unknown adapters.
fn rank(info: &wgpu::AdapterInfo) -> u8 {
    match (info.device_type, info.vendor) {
        (wgpu::DeviceType::DiscreteGpu, NVIDIA_VENDOR) => 0,
        (wgpu::DeviceType::DiscreteGpu, _) => 1,
        (wgpu::DeviceType::IntegratedGpu, _) => 2,
        (wgpu::DeviceType::VirtualGpu, _) => 3,
        (wgpu::DeviceType::Cpu, _) => 4,
        (wgpu::DeviceType::Other, _) => 5,
    }
}

/// The bytes of 32-bit values as a GPU buffer holds them.
pub(in crate::core::gpu) fn words(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_ne_bytes).collect()
}

impl Gpu {
    /// Opens the preferred adapter, or says why none can be used.
    pub fn probe() -> Result<Self, GpuError> {
        let mut adapters = Self::adapters();
        adapters.sort_by_key(|adapter| rank(&adapter.get_info()));
        let adapter = adapters.into_iter().next().ok_or(GpuError::NoAdapter)?;
        Self::open_adapter(&adapter)
    }

    pub(in crate::core::gpu) fn adapters() -> Vec<wgpu::Adapter> {
        // Enumerate one graphics API per OS so the same device is not
        // scheduled twice through two wgpu backends.
        let backends = if cfg!(target_vendor = "apple") {
            wgpu::Backends::METAL
        } else if cfg!(windows) {
            wgpu::Backends::DX12
        } else {
            wgpu::Backends::VULKAN
        };
        let enumerate = |backends| {
            let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
            descriptor.backends = backends;
            let instance = wgpu::Instance::new(descriptor);
            pollster::block_on(instance.enumerate_adapters(backends))
        };
        let adapters = enumerate(backends);
        if cfg!(windows) && !adapters.iter().any(Self::hardware_adapter) {
            return enumerate(wgpu::Backends::VULKAN);
        }
        adapters
    }

    pub(in crate::core::gpu) fn hardware_adapter(adapter: &wgpu::Adapter) -> bool {
        matches!(
            adapter.get_info().device_type,
            wgpu::DeviceType::DiscreteGpu
                | wgpu::DeviceType::IntegratedGpu
                | wgpu::DeviceType::VirtualGpu
        )
    }

    pub(in crate::core::gpu) fn open_adapter(adapter: &wgpu::Adapter) -> Result<Self, GpuError> {
        let info = adapter.get_info();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .map_err(|error| GpuError::Device(error.to_string()))?;
        Ok(Self {
            device,
            queue,
            info,
        })
    }

    /// Adapter name, kind and driver, for reports.
    pub fn description(&self) -> String {
        format!(
            "{} ({:?}, {} {})",
            self.info.name, self.info.device_type, self.info.driver, self.info.driver_info
        )
    }

    /// Submits no work and waits for the device: the cost of a call alone.
    pub fn idle_call(&self) -> Result<(), GpuError> {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        self.queue.submit([encoder.finish()]);
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|error| GpuError::Execution(error.to_string()))
    }

    /// Whether a storage buffer of `words` values and a dispatch of `groups`
    /// workgroups are within the device limits.
    pub(in crate::core::gpu) fn fits(&self, words: usize, groups: usize) -> bool {
        let limits = self.device.limits();
        let bytes = (words as u64).saturating_mul(WORD as u64);
        bytes <= limits.max_storage_buffer_binding_size
            && bytes <= limits.max_buffer_size
            && groups as u64 <= u64::from(limits.max_compute_workgroups_per_dimension)
    }

    /// Runs `build` and returns a validation error it raised on the device.
    #[cfg(feature = "cuda")]
    pub(in crate::core::gpu) fn validated<T>(
        &self,
        build: impl FnOnce() -> T,
    ) -> Result<T, GpuError> {
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let built = build();
        match pollster::block_on(scope.pop()) {
            Some(error) => Err(GpuError::Device(error.to_string())),
            None => Ok(built),
        }
    }

    /// A device buffer holding `contents`.
    pub(in crate::core::gpu) fn uploaded(
        &self,
        contents: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents,
                usage,
            })
    }

    /// A device buffer of `words` zero values.
    pub(in crate::core::gpu) fn zeroed(
        &self,
        words: usize,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (words * WORD) as u64,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Compiles `shader` and binds `buffers` to its bindings 0, 1, 2, ...
    #[cfg(feature = "cuda")]
    pub(in crate::core::gpu) fn kernel(&self, shader: &str, buffers: &[&wgpu::Buffer]) -> Kernel {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader.into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let entries: Vec<_> = buffers
            .iter()
            .zip(0..)
            .map(|(buffer, binding)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bindings = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Kernel { pipeline, bindings }
    }

    /// Commands that run `groups` workgroups of `kernel`, not yet submitted.
    #[cfg(feature = "cuda")]
    pub(in crate::core::gpu) fn pass(&self, kernel: &Kernel, groups: u32) -> wgpu::CommandEncoder {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &kernel.bindings, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder
    }

    /// Submits `encoder` followed by a copy of `source` to `staging`, waits
    /// for all submitted work and returns the copied values.
    #[cfg(feature = "cuda")]
    pub(in crate::core::gpu) fn read(
        &self,
        mut encoder: wgpu::CommandEncoder,
        source: &wgpu::Buffer,
        staging: &wgpu::Buffer,
    ) -> Result<Vec<u32>, GpuError> {
        let failed = |error: &dyn fmt::Display| GpuError::Execution(error.to_string());
        encoder.copy_buffer_to_buffer(source, 0, staging, 0, None);
        self.queue.submit([encoder.finish()]);
        let (mapped, result) = std::sync::mpsc::channel();
        staging.map_async(wgpu::MapMode::Read, .., move |outcome| {
            // The receiver is gone only when `read` already returned an error.
            let _ = mapped.send(outcome);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| failed(&error))?;
        result
            .recv()
            .map_err(|error| failed(&error))?
            .map_err(|error| failed(&error))?;
        let values = {
            let view = staging
                .get_mapped_range(..)
                .map_err(|error| failed(&error))?;
            view.as_chunks::<WORD>().0.iter()
                .map(|word| u32::from_ne_bytes([word[0], word[1], word[2], word[3]]))
                .collect()
        };
        staging.unmap();
        Ok(values)
    }
}

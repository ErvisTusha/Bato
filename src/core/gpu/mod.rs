//! GPU compute beside the CPU workers, with a CPU path that gives the same
//! integers when no GPU is present.
//! Provenance: original; the device is driven through the public WebGPU
//! compute model as implemented by the `wgpu` crate (MIT OR Apache-2.0).
//!
//! Nothing in the codec calls this module yet: archives do not depend on it.
//! It exists to measure, on real hardware, which integer workloads a GPU
//! finishes sooner than the CPU with identical results.

pub mod matvec;

use std::fmt;

/// PCI vendor identifier of NVIDIA, the first vendor this path is tested on.
const NVIDIA_VENDOR: u32 = 0x10de;

/// Why the GPU could not be used; the caller then runs on the CPU alone.
#[derive(Debug)]
pub enum GpuError {
    /// No Vulkan compute adapter is installed or visible.
    NoAdapter,
    /// The adapter refused to open a device.
    Device(String),
    /// The requested kernel geometry is invalid or exceeds a device limit.
    Shape(String),
    /// A submitted kernel did not return its results.
    Execution(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no Vulkan compute adapter"),
            Self::Device(reason) => write!(f, "GPU device unavailable: {reason}"),
            Self::Shape(reason) => write!(f, "invalid GPU kernel geometry: {reason}"),
            Self::Execution(reason) => write!(f, "GPU kernel failed: {reason}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// An opened compute device.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
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

impl Gpu {
    /// Opens the preferred adapter, or says why none can be used.
    pub fn probe() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let mut adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN));
        adapters.sort_by_key(|adapter| rank(&adapter.get_info()));
        let adapter = adapters.into_iter().next().ok_or(GpuError::NoAdapter)?;
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
}

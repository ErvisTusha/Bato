//! Bato GPU integration. The backends are independently enabled by Cargo
//! features and are coordinated by a device scheduler.
//! Compression formats and the CPU reference implementation are independent
//! of the selected accelerator.

mod backend;
mod device;
pub(crate) mod filters;
mod operations;
mod scheduler;

#[cfg(all(feature = "cuda", feature = "wgpu"))]
pub mod matvec;
#[cfg(all(feature = "cuda", feature = "wgpu"))]
pub mod streams;
#[cfg(feature = "cuda")]
pub use backend::cuda;
#[cfg(feature = "wgpu")]
pub use backend::wgpu::Gpu;
#[cfg(all(feature = "cuda", feature = "wgpu"))]
pub(in crate::core::gpu) use backend::wgpu::{words, Kernel};

use std::fmt;

/// Why the GPU could not be used; the caller then runs on the CPU alone.
#[derive(Debug)]
pub enum GpuError {
    /// No compute adapter is installed or visible.
    NoAdapter,
    /// The adapter refused to open a device or to build a kernel.
    Device(String),
    /// The requested kernel geometry is invalid or exceeds a device limit.
    Shape(String),
    /// A submitted kernel did not return its results.
    Execution(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no compatible GPU compute adapter"),
            Self::Device(reason) => write!(f, "GPU device unavailable: {reason}"),
            Self::Shape(reason) => write!(f, "invalid GPU kernel geometry: {reason}"),
            Self::Execution(reason) => write!(f, "GPU kernel failed: {reason}"),
        }
    }
}

impl std::error::Error for GpuError {}

#[cfg(all(test, feature = "wgpu"))]
/// Parses and validates a shader as the device would, without a device.
fn validate_shader(source: &str) {
    let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
}

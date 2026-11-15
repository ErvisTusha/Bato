//! Cross-platform wgpu implementation for Vulkan, Metal and Direct3D 12.
//! Its dependency and operations are absent from CUDA-only or native-only builds.

mod context;
mod filters;

pub use context::Gpu;
pub(in crate::core::gpu) use context::{words, NVIDIA_VENDOR};
#[cfg(feature = "cuda")]
pub(in crate::core::gpu) use context::Kernel;
pub(in crate::core::gpu) use filters::WgpuFilters;

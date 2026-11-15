//! Hardware-specific GPU backends. Each module is Cargo feature gated.
//! Archive scheduling and CPU fallback live in the sibling scheduler module.

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(feature = "hip")]
pub(in crate::core::gpu) mod hip;
#[cfg(feature = "level-zero")]
pub(in crate::core::gpu) mod level_zero;
#[cfg(feature = "metal")]
pub(in crate::core::gpu) mod metal;
#[cfg(any(feature = "hip", feature = "level-zero", feature = "metal"))]
pub(in crate::core::gpu) mod native;

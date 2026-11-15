pub mod archive;
pub mod checksum;
pub mod codec;
pub mod frame;
#[cfg(any(
    feature = "cuda",
    feature = "wgpu",
    feature = "hip",
    feature = "level-zero",
    feature = "metal"
))]
pub mod gpu;
#[cfg(not(any(
    feature = "cuda",
    feature = "wgpu",
    feature = "hip",
    feature = "level-zero",
    feature = "metal"
)))]
#[path = "gpu_cpu.rs"]
pub mod gpu;
pub mod jobs;
pub mod memory;
pub mod recovery;
pub mod safe_fs;
pub mod types;

mod cm;
pub use cm::profile::TimingReport;
#[cfg(test)]
mod diagnostics;

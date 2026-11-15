//! Portable byte-filter backend for Vulkan, Metal and Direct3D 12.
//! Provenance: original Bato WGSL filters, relocated without behavior changes.

mod filters;

pub(super) use filters::WgpuFilters;

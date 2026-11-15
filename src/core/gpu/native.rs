//! Optional native accelerator plugin ABI for AMD HIP, Intel Level Zero,
//! and Apple Metal. Native libraries are built with their platform SDKs.
//! Provenance: original Bato ABI; filters remain defined in core/cm/filters.rs.
//!
//! Untrusted hardware results are checked against Bato's exact CPU reference
//! before the archive can consume them. No native toolchain is needed by Cargo.

use super::{filters::Direction, GpuError};
use crate::core::{cm::filters, types::TransformId};
use libloading::Library;
use std::path::PathBuf;

const ABI_VERSION: u32 = 1;
type VersionFn = unsafe extern "C" fn() -> u32;
type CountFn = unsafe extern "C" fn() -> i32;
type ApplyFn = unsafe extern "C" fn(u32, *const u8, *mut u8, usize, u32, u32) -> i32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeKind {
    Hip,
    LevelZero,
    Metal,
}

impl NativeKind {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Hip => "HIP",
            Self::LevelZero => "Level Zero",
            Self::Metal => "Metal",
        }
    }

    fn variable(self) -> &'static str {
        match self {
            Self::Hip => "BATO_HIP_LIBRARY",
            Self::LevelZero => "BATO_LEVEL_ZERO_LIBRARY",
            Self::Metal => "BATO_METAL_LIBRARY",
        }
    }

    fn filename(self) -> &'static str {
        match self {
            Self::Hip => {
                if cfg!(windows) { "bato_hip.dll" } else { "libbato_hip.so" }
            }
            Self::LevelZero => {
                if cfg!(windows) { "bato_level_zero.dll" } else { "libbato_level_zero.so" }
            }
            Self::Metal => "libbato_metal.dylib",
        }
    }

    fn compiled() -> Vec<Self> {
        let mut built = Vec::new();
        #[cfg(feature = "hip")]
        built.push(Self::Hip);
        #[cfg(feature = "level-zero")]
        built.push(Self::LevelZero);
        #[cfg(feature = "metal")]
        built.push(Self::Metal);
        built
    }
}

struct Plugin {
    // Keep the dynamically loaded library alive for the function pointers.
    _library: Library,
    kind: NativeKind,
    devices: u32,
    apply: ApplyFn,
}

impl Plugin {
    fn open(kind: NativeKind) -> Result<Self, GpuError> {
        let path = match std::env::var_os(kind.variable()) {
            Some(value) => PathBuf::from(value),
            None => PathBuf::from(kind.filename()),
        };
        // SAFETY: loading a native library is explicitly opted into by the
        // Cargo feature; the ABI is checked before any device call.
        let library = unsafe { Library::new(&path) }
            .map_err(|e| GpuError::Device(format!("{}: {e}", path.display())))?;
        // SAFETY: the plugin ABI defines these exact symbol names and
        // signatures. They remain valid while _library is held by Plugin.
        let (version, count, apply) = unsafe {
            let version = *library
                .get::<VersionFn>(b"bato_gpu_abi_version\0")
                .map_err(|e| GpuError::Device(e.to_string()))?;
            let count = *library
                .get::<CountFn>(b"bato_gpu_device_count\0")
                .map_err(|e| GpuError::Device(e.to_string()))?;
            let apply = *library
                .get::<ApplyFn>(b"bato_gpu_filter\0")
                .map_err(|e| GpuError::Device(e.to_string()))?;
            (version, count, apply)
        };
        // SAFETY: pointers were resolved using the ABI above.
        let actual = unsafe { version() };
        if actual != ABI_VERSION {
            return Err(GpuError::Device(format!(
                "{} plugin ABI {actual}, expected {ABI_VERSION}", kind.label()
            )));
        }
        // SAFETY: the ABI states that device_count takes no arguments.
        let discovered = unsafe { count() };
        if discovered <= 0 {
            return Err(GpuError::Device(format!(
                "{} has no available native GPU devices", kind.label()
            )));
        }
        Ok(Self {
            _library: library,
            kind,
            devices: discovered as u32,
            apply,
        })
    }

    fn run(
        &self,
        ordinal: u32,
        input: &[u8],
        transform: TransformId,
        direction: Direction,
    ) -> Result<Vec<u8>, GpuError> {
        let mut output = crate::core::memory::try_filled(input.len(), 0u8)
            .map_err(|error| GpuError::Execution(error.to_string()))?;
        let inverse = u32::from(matches!(direction, Direction::Inverse));
        // SAFETY: input/output buffers remain valid and disjoint for the
        // entire synchronous call; ordinal is within device_count; filter
        // and direction use the documented ABI discriminants.
        let result = unsafe {
            (self.apply)(
                ordinal,
                input.as_ptr(),
                output.as_mut_ptr(),
                input.len(),
                transform as u32,
                inverse,
            )
        };
        if result != 0 {
            return Err(GpuError::Execution(format!(
                "{} GPU #{ordinal} filter returned error {result}", self.kind.label()
            )));
        }
        // Until hardware-specific validation is complete, verify every byte
        // against the canonical CPU implementation. Never trust an optional
        // plugin to silently change archive bytes or checksum semantics.
        let reference = match direction {
            Direction::Forward => filters::forward(input, transform),
            Direction::Inverse => {
                let mut data = crate::core::memory::try_copy(input)
                    .map_err(|error| GpuError::Execution(error.to_string()))?;
                filters::inverse(&mut data, transform).map(|()| data)
            }
        }
        .map_err(|error| GpuError::Execution(error.to_string()))?;
        if output != reference {
            return Err(GpuError::Execution(format!(
                "{} GPU #{ordinal} disagrees with exact CPU filter", self.kind.label()
            )));
        }
        Ok(output)
    }
}

/// Native plugins available to one archive worker. All plugin calls are
/// synchronous so they cannot outlive the input, output or loaded library.
#[derive(Default)]
pub(crate) struct WorkerNative {
    plugins: Vec<Plugin>,
    errors: Vec<String>,
}

impl WorkerNative {
    pub(crate) fn discover() -> Self {
        let mut found = Self::default();
        for kind in NativeKind::compiled() {
            match Plugin::open(kind) {
                Ok(plugin) => found.plugins.push(plugin),
                Err(error) => found.errors.push(error.to_string()),
            }
        }
        found
    }

    pub(crate) fn device_count(&self) -> usize {
        self.plugins.iter().map(|p| p.devices as usize).sum()
    }

    pub(crate) fn run(
        &self,
        worker: usize,
        input: &[u8],
        transform: TransformId,
        direction: Direction,
        selected: Option<NativeKind>,
    ) -> Result<Option<(Vec<u8>, String)>, GpuError> {
        let total = self.plugins.iter()
            .filter(|p| selected.is_none_or(|kind| p.kind == kind))
            .map(|p| p.devices as usize)
            .sum::<usize>();
        if total == 0 {
            return if selected.is_some() {
                Err(GpuError::Device(self.errors.join("; ")))
            } else {
                Ok(None)
            };
        }
        let mut target = worker % total;
        for plugin in &self.plugins {
            if selected.is_some_and(|kind| plugin.kind != kind) {
                continue;
            }
            if target >= plugin.devices as usize {
                target -= plugin.devices as usize;
                continue;
            }
            let result = plugin.run(target as u32, input, transform, direction)?;
            return Ok(Some((result, format!("{} #{target}", plugin.kind.label()))));
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_paths_are_stable() {
        assert_eq!(NativeKind::Hip.variable(), "BATO_HIP_LIBRARY");
        assert_eq!(NativeKind::LevelZero.label(), "Level Zero");
        assert_eq!(NativeKind::Metal.filename(), "libbato_metal.dylib");
    }

    #[test]
    fn plugin_abi_uses_byte_and_size_types() {
        assert_eq!(std::mem::size_of::<u32>(), 4);
        assert_eq!(ABI_VERSION, 1);
    }

    #[test]
    fn native_plugin_fixture_matches_cpu_filter_and_inverse() {
        // GitHub CI sets BATO_TEST_NATIVE_PLUGIN after building the reference
        // C fixture. Ordinary runs do not need any native library.
        if std::env::var_os("BATO_TEST_NATIVE_PLUGIN").is_none() {
            return;
        }
        let plugin = Plugin::open(NativeKind::Hip).unwrap();
        let sizes = [1, 2, 3, 4, 7, 63, 65, 4095, 4096, 4097, 8193];
        for size in sizes {
            let bytes = (0..size)
                .map(|i| ((i * 127 + i / 3 + 91) & 255) as u8)
                .collect::<Vec<_>>();
            for transform in [
                TransformId::Delta1,
                TransformId::Delta2,
                TransformId::Delta4,
                TransformId::Shuffle2,
                TransformId::Shuffle4,
            ] {
                let encoded = plugin
                    .run(0, &bytes, transform, Direction::Forward)
                    .unwrap();
                let restored = plugin
                    .run(0, &encoded, transform, Direction::Inverse)
                    .unwrap();
                assert_eq!(restored, bytes, "{transform:?}, input {size}");
            }
        }
    }
}

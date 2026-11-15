//! An NVIDIA device through the CUDA Driver API: context, kernels and memory.
//! Provenance: original; the calls are those of NVIDIA's public CUDA Driver
//! API reference. The kernels in `kernels.cu` are original and are loaded as
//! PTX text, so only the installed NVIDIA driver is needed, not the Toolkit.

mod driver;
pub(crate) mod filters;

use super::super::GpuError;
use driver::{Device, DevicePointer, Driver, Handle};
#[cfg(feature = "wgpu")]
use std::ptr::NonNull;
use std::{
    ffi::{c_char, c_int, c_uint, c_void, CStr},
    rc::Rc,
};

/// The compiled kernels as NUL-terminated PTX text.
const PTX: &str = concat!(include_str!("../../../../../native/cuda/kernels.ptx"), "\0");
/// Threads per block of every launch.
const BLOCK_THREADS: usize = 64;
/// Bytes kept of the driver's message when it rejects the PTX.
const JIT_LOG_BYTES: usize = 8192;
/// Bytes read of the device name.
const NAME_BYTES: usize = 128;

// A 32-bit kernel argument is passed as the low half of a 64-bit slot.
const _: () = assert!(cfg!(target_endian = "little"));

/// Integer types a device buffer holds: every bit pattern is a valid value.
///
/// # Safety
/// Implementors have no padding and no invalid bit patterns.
pub(in crate::core::gpu) unsafe trait Plain: Copy {}
// SAFETY: primitive integers have no padding and no invalid bit patterns.
unsafe impl Plain for i16 {}
// SAFETY: as above.
unsafe impl Plain for u8 {}
// SAFETY: as above.
unsafe impl Plain for i32 {}
// SAFETY: as above.
unsafe impl Plain for u32 {}

/// The retained context and loaded kernels of one device.
struct Session {
    driver: Driver,
    device: Device,
    /// Null until the kernels are loaded.
    module: Handle,
    name: String,
    memory: usize,
}

impl Drop for Session {
    fn drop(&mut self) {
        // SAFETY: `module` is null or the module this session loaded, and the
        // context was retained once by this session; both are released once.
        unsafe {
            if !self.module.is_null() {
                // Nothing can be done about a failed release while dropping.
                let _ = (self.driver.unload_module)(self.module);
            }
            // As above.
            let _ = (self.driver.release_context)(self.device);
        }
    }
}

/// An opened CUDA device; clones share it. It stays on the thread that opened
/// it, where its context is current.
#[derive(Clone)]
pub struct Cuda(Rc<Session>);

pub(in crate::core::gpu) struct Adapter {
    pub(in crate::core::gpu) ordinal: c_int,
    pub(in crate::core::gpu) pci_bus_id: String,
}

/// A kernel of the loaded module.
#[derive(Clone, Copy)]
pub(in crate::core::gpu) struct Function(Handle);

/// One argument of a kernel launch.
#[derive(Clone, Copy)]
pub(in crate::core::gpu) enum Argument {
    Memory(DevicePointer),
    Word(u32),
}

impl Cuda {
    /// Opens the first CUDA device and loads the kernels, or says why not.
    pub fn probe() -> Result<Self, GpuError> {
        Self::open(0, driver::SCHED_SPIN)
    }

    pub(in crate::core::gpu) fn open_for_filters(ordinal: c_int) -> Result<Self, GpuError> {
        Self::open(ordinal, driver::SCHED_BLOCKING_SYNC)
    }

    pub(in crate::core::gpu) fn adapters() -> Result<Vec<Adapter>, GpuError> {
        let driver = Driver::load()?;
        // SAFETY: `cuInit` takes only its flags.
        driver.check(unsafe { (driver.init)(0) }, "cuInit")?;
        let mut count: c_int = 0;
        // SAFETY: `count` is a valid place for the one value written.
        let status = unsafe { (driver.device_count)(&mut count) };
        driver.check(status, "cuDeviceGetCount")?;
        let mut adapters = Vec::new();
        for ordinal in 0..count {
            let mut device: Device = 0;
            // SAFETY: the ordinal is within the device count and the output is valid.
            driver.check(
                unsafe { (driver.device)(&mut device, ordinal) },
                "cuDeviceGet",
            )?;
            let mut bus = [0 as c_char; NAME_BYTES];
            // SAFETY: the buffer is one byte longer than the length passed to CUDA.
            let status = unsafe {
                (driver.device_pci_bus_id)(bus.as_mut_ptr(), (NAME_BYTES - 1) as c_int, device)
            };
            let pci_bus_id = if driver.check(status, "cuDeviceGetPCIBusId").is_ok() {
                // SAFETY: the final byte is zero regardless of what the driver wrote.
                unsafe { CStr::from_ptr(bus.as_ptr()) }
                    .to_string_lossy()
                    .into_owned()
            } else {
                // CUDA can still use the device. An unknown ID cannot deduplicate adapters.
                String::new()
            };
            adapters.push(Adapter {
                ordinal,
                pci_bus_id,
            });
        }
        if adapters.is_empty() {
            return Err(GpuError::Device("no CUDA device".into()));
        }
        Ok(adapters)
    }

    fn open(ordinal: c_int, schedule: c_uint) -> Result<Self, GpuError> {
        let driver = Driver::load()?;
        // SAFETY: `cuInit` takes only its flags.
        driver.check(unsafe { (driver.init)(0) }, "cuInit")?;
        let mut device: Device = 0;
        // SAFETY: `device` is valid; CUDA validates the ordinal.
        driver.check(
            unsafe { (driver.device)(&mut device, ordinal) },
            "cuDeviceGet",
        )?;
        let mut name = [0 as c_char; NAME_BYTES];
        // SAFETY: the driver writes at most the stated length; the last byte
        // is left out so the text always ends with a NUL.
        let status =
            unsafe { (driver.device_name)(name.as_mut_ptr(), (NAME_BYTES - 1) as c_int, device) };
        driver.check(status, "cuDeviceGetName")?;
        // SAFETY: `name` holds a NUL within its length, as arranged above.
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let mut memory = 0usize;
        // SAFETY: `memory` is a valid place for the one value written.
        let status = unsafe { (driver.device_memory)(&mut memory, device) };
        driver.check(status, "cuDeviceTotalMem")?;
        // The flag only chooses how waits are done, and a context that is
        // already active keeps its own flags, so a refusal is not an error.
        // SAFETY: the call takes the device handle and flags by value.
        let _ = unsafe { (driver.set_context_flags)(device, schedule) };
        let mut context: Handle = std::ptr::null_mut();
        // SAFETY: `context` is a valid place for the handle written.
        let status = unsafe { (driver.retain_context)(&mut context, device) };
        driver.check(status, "cuDevicePrimaryCtxRetain")?;
        // From here the session's `Drop` releases the context on any error.
        let mut session = Session {
            driver,
            device,
            module: std::ptr::null_mut(),
            name,
            memory,
        };
        let driver = &session.driver;
        // SAFETY: `context` is the live context retained above.
        driver.check(unsafe { (driver.set_current)(context) }, "cuCtxSetCurrent")?;
        let mut log = vec![0u8; JIT_LOG_BYTES];
        let mut options = [
            driver::JIT_ERROR_LOG_BUFFER,
            driver::JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        ];
        let mut values = [
            log.as_mut_ptr().cast::<c_void>(),
            std::ptr::without_provenance_mut(JIT_LOG_BYTES),
        ];
        let mut module: Handle = std::ptr::null_mut();
        // SAFETY: `PTX` is NUL-terminated text; the two options and their
        // values match in number and kind, and `log` outlives the call.
        let status = unsafe {
            (driver.load_module)(
                &mut module,
                PTX.as_ptr().cast(),
                options.len() as c_uint,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };
        if let Err(error) = driver.check(status, "cuModuleLoadDataEx") {
            let end = log.iter().position(|&byte| byte == 0).unwrap_or(log.len());
            return Err(GpuError::Device(format!(
                "{error}: {}",
                String::from_utf8_lossy(&log[..end]).trim()
            )));
        }
        session.module = module;
        Ok(Self(Rc::new(session)))
    }

    /// Device name and memory, for reports.
    pub fn description(&self) -> String {
        format!("{} (CUDA, {} MiB)", self.0.name, self.0.memory >> 20)
    }

    /// Launches a kernel that does nothing and waits for it: the cost of a
    /// call alone.
    pub fn idle_call(&self) -> Result<(), GpuError> {
        self.launch(self.function(c"empty")?, 1, &[])?;
        self.synchronize()
    }

    /// The kernel called `name` in the loaded module.
    pub(in crate::core::gpu) fn function(&self, name: &CStr) -> Result<Function, GpuError> {
        let mut function: Handle = std::ptr::null_mut();
        // SAFETY: `function` is a valid place, the module is loaded and
        // `name` is NUL-terminated.
        let status =
            unsafe { (self.0.driver.function)(&mut function, self.0.module, name.as_ptr()) };
        self.0.driver.check(status, "cuModuleGetFunction")?;
        Ok(Function(function))
    }

    /// Starts `function` on at least `threads` threads; the caller's kernel
    /// ignores the surplus. `arguments` must match the kernel's parameters.
    pub(in crate::core::gpu) fn launch(
        &self,
        function: Function,
        threads: usize,
        arguments: &[Argument],
    ) -> Result<(), GpuError> {
        let blocks = c_uint::try_from(threads.div_ceil(BLOCK_THREADS).max(1))
            .map_err(|_| GpuError::Shape(format!("{threads} threads in one launch")))?;
        let mut values: Vec<u64> = arguments
            .iter()
            .map(|argument| match argument {
                Argument::Memory(pointer) => *pointer,
                Argument::Word(word) => u64::from(*word),
            })
            .collect();
        let mut pointers: Vec<*mut c_void> = values
            .iter_mut()
            .map(|value| std::ptr::from_mut(value).cast())
            .collect();
        // SAFETY: `function` belongs to the loaded module; `pointers` holds
        // one pointer per kernel parameter, each to a live value at least as
        // large as that parameter; the device memory arguments are live
        // allocations of this context.
        let status = unsafe {
            (self.0.driver.launch)(
                function.0,
                blocks,
                1,
                1,
                BLOCK_THREADS as c_uint,
                1,
                1,
                0,
                std::ptr::null_mut(),
                pointers.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        self.0.driver.check(status, "cuLaunchKernel")
    }

    /// Waits until every launched kernel has finished.
    pub(in crate::core::gpu) fn synchronize(&self) -> Result<(), GpuError> {
        // SAFETY: the call takes no arguments and uses the current context.
        let status = unsafe { (self.0.driver.synchronize)() };
        self.0.driver.check(status, "cuCtxSynchronize")
    }
}

/// An allocation in device memory.
pub(in crate::core::gpu) struct Memory {
    cuda: Cuda,
    pointer: DevicePointer,
    bytes: usize,
}

impl Memory {
    /// Uninitialised device memory for `length` values of `T`.
    fn allocate<T: Plain>(cuda: &Cuda, length: usize) -> Result<Self, GpuError> {
        let bytes = length
            .checked_mul(std::mem::size_of::<T>())
            .filter(|&bytes| bytes > 0)
            .ok_or_else(|| GpuError::Shape(format!("device buffer of {length} values")))?;
        let mut pointer: DevicePointer = 0;
        // SAFETY: `pointer` is a valid place for the address written.
        let status = unsafe { (cuda.0.driver.allocate)(&mut pointer, bytes) };
        cuda.0.driver.check(status, "cuMemAlloc")?;
        Ok(Self {
            cuda: cuda.clone(),
            pointer,
            bytes,
        })
    }

    /// Device memory of `words` zero 32-bit values.
    #[cfg(feature = "wgpu")]
    pub(in crate::core::gpu) fn zeroed(cuda: &Cuda, words: usize) -> Result<Self, GpuError> {
        let memory = Self::allocate::<u32>(cuda, words)?;
        // SAFETY: the allocation holds exactly `words` 32-bit values.
        let status = unsafe { (cuda.0.driver.fill_words)(memory.pointer, 0, words) };
        cuda.0.driver.check(status, "cuMemsetD32")?;
        Ok(memory)
    }

    /// Device memory holding a copy of `values`.
    pub(in crate::core::gpu) fn uploaded<T: Plain>(
        cuda: &Cuda,
        values: &[T],
    ) -> Result<Self, GpuError> {
        let memory = Self::allocate::<T>(cuda, values.len())?;
        memory.write(values)?;
        Ok(memory)
    }

    fn check_size<T: Plain>(&self, length: usize) -> Result<(), GpuError> {
        if length.checked_mul(std::mem::size_of::<T>()) == Some(self.bytes) {
            return Ok(());
        }
        Err(GpuError::Shape(format!(
            "{length} values for a device buffer of {} bytes",
            self.bytes
        )))
    }

    /// Copies `values`, which must fill the allocation, to the device.
    pub(in crate::core::gpu) fn write<T: Plain>(&self, values: &[T]) -> Result<(), GpuError> {
        self.check_size::<T>(values.len())?;
        // SAFETY: `values` is readable for `bytes` bytes, the size of the
        // device allocation, as checked above.
        let status = unsafe {
            (self.cuda.0.driver.to_device)(self.pointer, values.as_ptr().cast(), self.bytes)
        };
        self.cuda.0.driver.check(status, "cuMemcpyHtoD")
    }

    /// Waits for running kernels and copies the whole allocation to `values`.
    pub(in crate::core::gpu) fn read<T: Plain>(&self, values: &mut [T]) -> Result<(), GpuError> {
        self.check_size::<T>(values.len())?;
        // SAFETY: `values` is writable for `bytes` bytes, as checked above,
        // and every bit pattern is a valid `T`.
        let status = unsafe {
            (self.cuda.0.driver.to_host)(values.as_mut_ptr().cast(), self.pointer, self.bytes)
        };
        self.cuda.0.driver.check(status, "cuMemcpyDtoH")
    }

    pub(in crate::core::gpu) fn argument(&self) -> Argument {
        Argument::Memory(self.pointer)
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        // Nothing can be done about a failed release while dropping.
        // SAFETY: `pointer` is the live allocation made in `allocate`.
        let _ = unsafe { (self.cuda.0.driver.free)(self.pointer) };
    }
}

#[cfg(feature = "wgpu")]
/// Page-locked host memory, which the driver copies to and from the device
/// without an intermediate buffer.
pub(in crate::core::gpu) struct Pinned<T: Plain> {
    cuda: Cuda,
    pointer: NonNull<T>,
    length: usize,
}

#[cfg(feature = "wgpu")]
impl<T: Plain> Pinned<T> {
    /// `length` zero values.
    pub(in crate::core::gpu) fn new(cuda: &Cuda, length: usize) -> Result<Self, GpuError> {
        let bytes = length
            .checked_mul(std::mem::size_of::<T>())
            .filter(|&bytes| bytes > 0)
            .ok_or_else(|| GpuError::Shape(format!("host buffer of {length} values")))?;
        let mut pointer: *mut c_void = std::ptr::null_mut();
        // SAFETY: `pointer` is a valid place for the address written.
        let status = unsafe { (cuda.0.driver.allocate_host)(&mut pointer, bytes) };
        cuda.0.driver.check(status, "cuMemAllocHost")?;
        let pointer = NonNull::new(pointer.cast::<T>())
            .ok_or_else(|| GpuError::Execution("cuMemAllocHost returned no memory".into()))?;
        // SAFETY: the driver returned `bytes` writable, page-aligned bytes;
        // zero is a valid `T`.
        unsafe { pointer.as_ptr().write_bytes(0, length) };
        Ok(Self {
            cuda: cuda.clone(),
            pointer,
            length,
        })
    }

    pub(in crate::core::gpu) fn as_slice(&self) -> &[T] {
        // SAFETY: the allocation holds `length` initialised values and lives
        // as long as `self`.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }

    pub(in crate::core::gpu) fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as in `as_slice`, and `&mut self` makes the access unique.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.length) }
    }
}

#[cfg(feature = "wgpu")]
impl<T: Plain> Drop for Pinned<T> {
    fn drop(&mut self) {
        // Nothing can be done about a failed release while dropping.
        // SAFETY: `pointer` is the live allocation made in `new`.
        let _ = unsafe { (self.cuda.0.driver.free_host)(self.pointer.as_ptr().cast()) };
    }
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    const SOURCE: &str = include_str!("../../../../../native/cuda/kernels.cu");

    #[test]
    fn ptx_is_generated_from_the_current_source() {
        let digest: String = Sha256::digest(SOURCE.as_bytes())
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(
            super::PTX.lines().next(),
            Some(format!("// source sha256: {digest}").as_str()),
            "regenerate kernels.ptx as the header of kernels.cu describes"
        );
        for entry in [
            "empty",
            "matvec",
            "streams",
            "filter_bytes",
            "filter_delta_local",
            "filter_delta_carries",
            "filter_delta_finish",
        ] {
            assert!(super::PTX.contains(&format!(".visible .entry {entry}(")));
        }
    }

    #[test]
    fn a_missing_driver_is_reported_not_fatal() {
        match super::Cuda::probe() {
            Ok(cuda) => {
                cuda.idle_call().unwrap();
                eprintln!("CUDA device: {}", cuda.description());
            }
            Err(reason) => eprintln!("CUDA not used on this machine: {reason}"),
        }
    }
}

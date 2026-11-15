//! The CUDA Driver API functions this crate calls, loaded at run time from the
//! library the NVIDIA driver installs.
//! Provenance: function names, argument types and constants follow NVIDIA's
//! public CUDA Driver API reference; no NVIDIA source or header is included.

use crate::core::gpu::GpuError;
use libloading::Library;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};

/// A `CUresult`; zero is success.
pub(super) type Status = c_int;
/// A `CUdevice` ordinal handle.
pub(super) type Device = c_int;
/// A `CUdeviceptr`: an address in device memory.
pub(super) type DevicePointer = u64;
/// An opaque context, module, function or stream handle.
pub(super) type Handle = *mut c_void;

const SUCCESS: Status = 0;
/// `CU_CTX_SCHED_SPIN`: wait for the device by spinning, the lowest latency.
pub(super) const SCHED_SPIN: c_uint = 1;
/// `CU_CTX_SCHED_BLOCKING_SYNC`: let CPU workers run while waiting for CUDA.
pub(super) const SCHED_BLOCKING_SYNC: c_uint = 4;
/// `CU_JIT_ERROR_LOG_BUFFER` and `CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES`.
pub(super) const JIT_ERROR_LOG_BUFFER: c_uint = 5;
pub(super) const JIT_ERROR_LOG_BUFFER_SIZE_BYTES: c_uint = 6;

/// File names of the driver library, in the order tried.
const LIBRARIES: [&str; 3] = ["libcuda.so.1", "libcuda.so", "nvcuda.dll"];

macro_rules! driver {
    ($($field:ident = $symbol:literal: fn($($argument:ty),*);)*) => {
        /// The loaded driver functions; they stay valid while `_library` lives.
        pub(super) struct Driver {
            $(pub(super) $field: unsafe extern "C" fn($($argument),*) -> Status,)*
            _library: Library,
        }

        impl Driver {
            /// Loads the driver library and every function, or says what is
            /// missing.
            pub(super) fn load() -> Result<Self, GpuError> {
                let library = open()?;
                Ok(Self {
                    $($field: {
                        // SAFETY: `$symbol` is the CUDA Driver API function of
                        // that name, whose C signature is the declared type.
                        let symbol = unsafe { library.get($symbol.as_bytes()) };
                        *symbol.map_err(|error| {
                            GpuError::Device(format!("CUDA driver lacks {}: {error}", $symbol))
                        })?
                    },)*
                    _library: library,
                })
            }
        }
    };
}

driver! {
    init = "cuInit": fn(c_uint);
    device_count = "cuDeviceGetCount": fn(*mut c_int);
    device = "cuDeviceGet": fn(*mut Device, c_int);
    device_name = "cuDeviceGetName": fn(*mut c_char, c_int, Device);
    device_memory = "cuDeviceTotalMem_v2": fn(*mut usize, Device);
    set_context_flags = "cuDevicePrimaryCtxSetFlags_v2": fn(Device, c_uint);
    retain_context = "cuDevicePrimaryCtxRetain": fn(*mut Handle, Device);
    release_context = "cuDevicePrimaryCtxRelease_v2": fn(Device);
    set_current = "cuCtxSetCurrent": fn(Handle);
    synchronize = "cuCtxSynchronize": fn();
    load_module = "cuModuleLoadDataEx":
        fn(*mut Handle, *const c_void, c_uint, *mut c_uint, *mut *mut c_void);
    unload_module = "cuModuleUnload": fn(Handle);
    function = "cuModuleGetFunction": fn(*mut Handle, Handle, *const c_char);
    allocate = "cuMemAlloc_v2": fn(*mut DevicePointer, usize);
    free = "cuMemFree_v2": fn(DevicePointer);
    to_device = "cuMemcpyHtoD_v2": fn(DevicePointer, *const c_void, usize);
    to_host = "cuMemcpyDtoH_v2": fn(*mut c_void, DevicePointer, usize);
    fill_words = "cuMemsetD32_v2": fn(DevicePointer, c_uint, usize);
    allocate_host = "cuMemAllocHost_v2": fn(*mut *mut c_void, usize);
    free_host = "cuMemFreeHost": fn(*mut c_void);
    launch = "cuLaunchKernel": fn(
        Handle, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, Handle,
        *mut *mut c_void, *mut *mut c_void
    );
    error_name = "cuGetErrorName": fn(Status, *mut *const c_char);
}

/// Opens the first driver library present.
fn open() -> Result<Library, GpuError> {
    for name in LIBRARIES {
        // SAFETY: loading the NVIDIA driver library runs only its own
        // initialisers; nothing of this process is exposed to it yet.
        if let Ok(library) = unsafe { Library::new(name) } {
            return Ok(library);
        }
    }
    Err(GpuError::Device(format!(
        "no CUDA driver library ({})",
        LIBRARIES.join(", ")
    )))
}

impl Driver {
    /// Turns a failed status into an error naming the call and the code.
    pub(super) fn check(&self, status: Status, call: &str) -> Result<(), GpuError> {
        if status == SUCCESS {
            return Ok(());
        }
        let mut name: *const c_char = std::ptr::null();
        // SAFETY: `name` is a valid place for the one pointer the call writes.
        let named = unsafe { (self.error_name)(status, &mut name) };
        let name = if named == SUCCESS && !name.is_null() {
            // SAFETY: on success the driver stored a pointer to one of its
            // static NUL-terminated strings.
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        } else {
            String::from("unknown error")
        };
        Err(GpuError::Execution(format!(
            "{call}: {name} (CUDA {status})"
        )))
    }
}

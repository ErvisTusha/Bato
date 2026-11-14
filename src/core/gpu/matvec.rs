//! Integer matrix-vector kernel: the CPU reference, the wgpu and CUDA kernels
//! and a runner that uses the CPU and a device at once. Provenance: original.
//!
//! Each output is `clamp(sum(weight * input) >> OUTPUT_SHIFT, ±OUTPUT_LIMIT)`
//! with the sum taken in 64 bits, the arithmetic of a mixer layer. Every path
//! returns the same integers, so a result never depends on whether a GPU was
//! present.

use super::{
    cuda::{Argument, Cuda, Function, Memory, Pinned},
    words, Gpu, GpuError, Kernel,
};

/// Fractional bits dropped from each sum; `matvec.wgsl` and `cuda/kernels.cu`
/// repeat the value.
pub const OUTPUT_SHIFT: u32 = 10;
/// Largest magnitude of an output; `matvec.wgsl` and `cuda/kernels.cu` repeat
/// the value.
pub const OUTPUT_LIMIT: i32 = 2047;
/// Rows per workgroup, the `@workgroup_size` of `matvec.wgsl`.
const WORKGROUP_ROWS: usize = 64;
const SHADER: &str = include_str!("matvec.wgsl");

/// Rows and columns of the weight matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}

impl Shape {
    fn check(self, weights: usize, inputs: usize) -> Result<(), GpuError> {
        let cells = self.rows.checked_mul(self.cols);
        if self.rows == 0 || self.cols == 0 || cells != Some(weights) || inputs != self.cols {
            return Err(GpuError::Shape(format!(
                "{} rows by {} columns with {weights} weights and {inputs} inputs",
                self.rows, self.cols
            )));
        }
        Ok(())
    }
}

/// One output of the kernel: the definition every other path must equal.
fn row_output(weights: &[i16], inputs: &[i16]) -> i32 {
    let sum: i64 = weights
        .iter()
        .zip(inputs)
        .map(|(&weight, &input)| i64::from(weight) * i64::from(input))
        .sum();
    (sum >> OUTPUT_SHIFT).clamp(i64::from(-OUTPUT_LIMIT), i64::from(OUTPUT_LIMIT)) as i32
}

/// Writes the outputs of the rows in `weights` to `outputs`, in order.
fn fill_rows(weights: &[i16], inputs: &[i16], outputs: &mut [i32]) {
    for (output, row) in outputs.iter_mut().zip(weights.chunks_exact(inputs.len())) {
        *output = row_output(row, inputs);
    }
}

/// Like `fill_rows`, on every available CPU.
fn fill_rows_threaded(weights: &[i16], inputs: &[i16], outputs: &mut [i32]) {
    if outputs.is_empty() {
        return;
    }
    let workers = std::thread::available_parallelism().map_or(1, usize::from);
    let rows = outputs.len().div_ceil(workers);
    std::thread::scope(|scope| {
        for (part, weights) in outputs
            .chunks_mut(rows)
            .zip(weights.chunks(rows * inputs.len()))
        {
            scope.spawn(move || fill_rows(weights, inputs, part));
        }
    });
}

/// The kernel on one CPU thread.
pub fn reference(weights: &[i16], inputs: &[i16], shape: Shape) -> Result<Vec<i32>, GpuError> {
    shape.check(weights.len(), inputs.len())?;
    let mut outputs = vec![0; shape.rows];
    fill_rows(weights, inputs, &mut outputs);
    Ok(outputs)
}

/// Widens 16-bit values to the 32-bit words of a GPU buffer.
fn widened(values: &[i16]) -> Vec<u8> {
    words(values.iter().map(|&value| i32::from(value) as u32))
}

/// The kernel on a GPU, holding its weights in device memory.
pub struct GpuMatVec {
    gpu: Gpu,
    shape: Shape,
    kernel: Kernel,
    inputs: wgpu::Buffer,
    outputs: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl GpuMatVec {
    /// Uploads `weights` and compiles the kernel for `shape`.
    pub fn new(gpu: &Gpu, weights: &[i16], shape: Shape) -> Result<Self, GpuError> {
        shape.check(weights.len(), shape.cols)?;
        if !gpu.fits(weights.len(), shape.rows.div_ceil(WORKGROUP_ROWS)) {
            return Err(GpuError::Shape(format!(
                "{} rows by {} columns exceeds the device limits",
                shape.rows, shape.cols
            )));
        }
        gpu.validated(|| {
            let weights = gpu.uploaded(&widened(weights), wgpu::BufferUsages::STORAGE);
            let inputs = gpu.zeroed(
                shape.cols,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let outputs = gpu.zeroed(
                shape.rows,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let params = gpu.uploaded(
                &words([shape.rows as u32, shape.cols as u32]),
                wgpu::BufferUsages::UNIFORM,
            );
            Self {
                gpu: gpu.clone(),
                shape,
                kernel: gpu.kernel(SHADER, &[&weights, &inputs, &outputs, &params]),
                staging: gpu.zeroed(
                    shape.rows,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                ),
                inputs,
                outputs,
            }
        })
    }

    /// Runs the kernel on `inputs` and reads the outputs back. Mutates the
    /// kernel's device buffers.
    pub fn run(&mut self, inputs: &[i16]) -> Result<Vec<i32>, GpuError> {
        if inputs.len() != self.shape.cols {
            return Err(GpuError::Shape(format!(
                "{} inputs for {} columns",
                inputs.len(),
                self.shape.cols
            )));
        }
        self.gpu
            .queue
            .write_buffer(&self.inputs, 0, &widened(inputs));
        let groups = self.shape.rows.div_ceil(WORKGROUP_ROWS) as u32;
        let pass = self.gpu.pass(&self.kernel, groups);
        let outputs = self.gpu.read(pass, &self.outputs, &self.staging)?;
        Ok(outputs.into_iter().map(|value| value as i32).collect())
    }
}

/// The kernel on a CUDA device, holding its weights in device memory.
pub struct CudaMatVec {
    cuda: Cuda,
    shape: Shape,
    function: Function,
    weights: Memory,
    inputs: Memory,
    outputs: Memory,
    host_inputs: Pinned<i16>,
    host_outputs: Pinned<i32>,
}

impl CudaMatVec {
    /// Uploads `weights` and prepares the kernel for `shape`.
    pub fn new(cuda: &Cuda, weights: &[i16], shape: Shape) -> Result<Self, GpuError> {
        shape.check(weights.len(), shape.cols)?;
        if shape.rows > u32::MAX as usize || shape.cols > u32::MAX as usize {
            return Err(GpuError::Shape(format!(
                "{} rows by {} columns exceeds the kernel's 32-bit counts",
                shape.rows, shape.cols
            )));
        }
        let host_inputs = Pinned::new(cuda, shape.cols)?;
        Ok(Self {
            cuda: cuda.clone(),
            shape,
            function: cuda.function(c"matvec")?,
            weights: Memory::uploaded(cuda, weights)?,
            inputs: Memory::uploaded(cuda, host_inputs.as_slice())?,
            outputs: Memory::zeroed(cuda, shape.rows)?,
            host_inputs,
            host_outputs: Pinned::new(cuda, shape.rows)?,
        })
    }

    /// Runs the kernel on `inputs` and reads the outputs back. Mutates the
    /// kernel's device and host buffers.
    pub fn run(&mut self, inputs: &[i16]) -> Result<&[i32], GpuError> {
        if inputs.len() != self.shape.cols {
            return Err(GpuError::Shape(format!(
                "{} inputs for {} columns",
                inputs.len(),
                self.shape.cols
            )));
        }
        self.host_inputs.as_mut_slice().copy_from_slice(inputs);
        self.inputs.write(self.host_inputs.as_slice())?;
        let arguments = [
            self.weights.argument(),
            self.inputs.argument(),
            self.outputs.argument(),
            Argument::Word(self.shape.rows as u32),
            Argument::Word(self.shape.cols as u32),
        ];
        self.cuda
            .launch(self.function, self.shape.rows, &arguments)?;
        self.outputs.read(self.host_outputs.as_mut_slice())?;
        Ok(self.host_outputs.as_slice())
    }
}

/// The device a hybrid run shares the rows with.
#[derive(Clone, Copy)]
pub enum Accelerator<'a> {
    None,
    Wgpu(&'a Gpu),
    Cuda(&'a Cuda),
}

/// A device kernel of either kind.
enum Device {
    Wgpu(Box<GpuMatVec>),
    Cuda(Box<CudaMatVec>),
}

impl Device {
    /// Runs the kernel and writes its outputs to `outputs`. Mutates the
    /// kernel's buffers.
    fn run_into(&mut self, inputs: &[i16], outputs: &mut [i32]) -> Result<(), GpuError> {
        let short = || GpuError::Execution("the device returned too few rows".into());
        match self {
            Self::Wgpu(kernel) => {
                let rows = kernel.run(inputs)?;
                if rows.len() != outputs.len() {
                    return Err(short());
                }
                outputs.copy_from_slice(&rows);
            }
            Self::Cuda(kernel) => {
                let rows = kernel.run(inputs)?;
                if rows.len() != outputs.len() {
                    return Err(short());
                }
                outputs.copy_from_slice(rows);
            }
        }
        Ok(())
    }
}

/// Which processors produced a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Processors {
    Cpu,
    CpuAndGpu,
}

/// The kernel on a device and the CPU at once: the device computes the first
/// `device_rows` rows while the CPU threads compute the rest. Without a
/// device, or when it fails, the CPU computes every row; the outputs are the
/// same.
pub struct HybridMatVec {
    weights: Vec<i16>,
    shape: Shape,
    device: Option<Device>,
    device_rows: usize,
}

impl HybridMatVec {
    /// Prepares the kernel; `device_rows` of the rows go to `accelerator`
    /// when there is one.
    pub fn new(
        accelerator: Accelerator<'_>,
        weights: Vec<i16>,
        shape: Shape,
        device_rows: usize,
    ) -> Result<Self, GpuError> {
        shape.check(weights.len(), shape.cols)?;
        let device_rows = device_rows.min(shape.rows);
        let part = Shape {
            rows: device_rows,
            cols: shape.cols,
        };
        let share = &weights[..device_rows * shape.cols];
        let device = match accelerator {
            _ if device_rows == 0 => None,
            Accelerator::None => None,
            Accelerator::Wgpu(gpu) => {
                Some(Device::Wgpu(Box::new(GpuMatVec::new(gpu, share, part)?)))
            }
            Accelerator::Cuda(cuda) => {
                Some(Device::Cuda(Box::new(CudaMatVec::new(cuda, share, part)?)))
            }
        };
        Ok(Self {
            device_rows: if device.is_some() { device_rows } else { 0 },
            weights,
            shape,
            device,
        })
    }

    /// Computes every output and says which processors took part. Mutates the
    /// device kernel's buffers.
    pub fn run(&mut self, inputs: &[i16]) -> Result<(Vec<i32>, Processors), GpuError> {
        self.shape.check(self.weights.len(), inputs.len())?;
        let mut outputs = vec![0; self.shape.rows];
        let split = self.device_rows * self.shape.cols;
        let (device_part, cpu_part) = outputs.split_at_mut(self.device_rows);
        let (device_weights, cpu_weights) = self.weights.split_at(split);
        let Some(device) = &mut self.device else {
            fill_rows_threaded(cpu_weights, inputs, cpu_part);
            return Ok((outputs, Processors::Cpu));
        };
        let from_device = std::thread::scope(|scope| {
            scope.spawn(|| fill_rows_threaded(cpu_weights, inputs, cpu_part));
            device.run_into(inputs, device_part)
        });
        if from_device.is_err() {
            // A lost or failing device is not an error of the computation:
            // the CPU computes the same rows.
            fill_rows_threaded(device_weights, inputs, device_part);
            return Ok((outputs, Processors::Cpu));
        }
        Ok((outputs, Processors::CpuAndGpu))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed-seed integer hash: reproducible test values.
    fn values(count: usize, seed: u64) -> Vec<i16> {
        (0..count as u64)
            .map(|i| {
                let mut x = (i + seed).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                x ^= x >> 29;
                x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                (x >> 40) as i16
            })
            .collect()
    }

    /// `matvec.wgsl` statement by statement, with its 32-bit operations.
    fn shader_row(weights: &[i16], inputs: &[i16]) -> i32 {
        let (mut low, mut high) = (0u32, 0i32);
        for (&weight, &input) in weights.iter().zip(inputs) {
            let product = i32::from(weight).wrapping_mul(i32::from(input));
            let next = low.wrapping_add(product as u32);
            high = high.wrapping_add((product >> 31) + i32::from(next < low));
            low = next;
        }
        let top = high >> OUTPUT_SHIFT;
        let shifted = (((high as u32) << (32 - OUTPUT_SHIFT)) | (low >> OUTPUT_SHIFT)) as i32;
        if (top == 0 && shifted >= 0) || (top == -1 && shifted < 0) {
            shifted.clamp(-OUTPUT_LIMIT, OUTPUT_LIMIT)
        } else if high < 0 {
            -OUTPUT_LIMIT
        } else {
            OUTPUT_LIMIT
        }
    }

    #[test]
    fn shader_arithmetic_equals_the_64_bit_reference() {
        let extremes = [i16::MIN, -1, 0, 1, i16::MAX];
        for &weight in &extremes {
            for &input in &extremes {
                for cols in [1, 2, 3, 64, 4096] {
                    let (weights, inputs) = (vec![weight; cols], vec![input; cols]);
                    assert_eq!(shader_row(&weights, &inputs), row_output(&weights, &inputs));
                }
            }
        }
        for seed in 0..2000 {
            let cols = 1 + (seed as usize % 97);
            let (weights, inputs) = (values(cols, seed), values(cols, seed + 7919));
            // Small weights exercise the unsaturated range.
            let small: Vec<i16> = weights.iter().map(|w| w >> (seed % 12)).collect();
            assert_eq!(shader_row(&small, &inputs), row_output(&small, &inputs));
            assert_eq!(shader_row(&weights, &inputs), row_output(&weights, &inputs));
        }
    }

    #[test]
    fn kernels_are_valid_and_repeat_the_constants() {
        super::super::validate_shader(SHADER);
        for constant in [
            format!("const OUTPUT_SHIFT: u32 = {OUTPUT_SHIFT}u;"),
            format!("const OUTPUT_LIMIT: i32 = {OUTPUT_LIMIT};"),
            format!("@workgroup_size({WORKGROUP_ROWS})"),
        ] {
            assert!(SHADER.contains(&constant), "{constant}");
        }
        let source = include_str!("cuda/kernels.cu");
        for constant in [
            format!("#define OUTPUT_SHIFT {OUTPUT_SHIFT}"),
            format!("#define OUTPUT_LIMIT {OUTPUT_LIMIT}"),
        ] {
            assert!(source.contains(&constant), "{constant}");
        }
    }

    #[test]
    fn invalid_geometry_is_rejected() {
        let shape = Shape { rows: 2, cols: 3 };
        assert!(reference(&[0; 5], &[0; 3], shape).is_err());
        assert!(reference(&[0; 6], &[0; 2], shape).is_err());
        assert!(reference(&[], &[], Shape { rows: 0, cols: 0 }).is_err());
        assert!(HybridMatVec::new(Accelerator::None, vec![0; 5], shape, 1).is_err());
    }

    #[test]
    fn every_path_returns_the_reference_outputs() {
        let shape = Shape {
            rows: 1000,
            cols: 64,
        };
        let weights = values(shape.rows * shape.cols, 1);
        let inputs: Vec<i16> = values(shape.cols, 2).iter().map(|x| x >> 4).collect();
        let expected = reference(&weights, &inputs, shape).unwrap();
        let mut cpu = HybridMatVec::new(Accelerator::None, weights.clone(), shape, 600).unwrap();
        assert_eq!(
            cpu.run(&inputs).unwrap(),
            (expected.clone(), Processors::Cpu)
        );
        let wgpu = Gpu::probe().inspect_err(|reason| eprintln!("wgpu parity not run: {reason}"));
        let cuda = Cuda::probe().inspect_err(|reason| eprintln!("CUDA parity not run: {reason}"));
        let accelerators = [
            wgpu.as_ref().map(Accelerator::Wgpu).ok(),
            cuda.as_ref().map(Accelerator::Cuda).ok(),
        ];
        for accelerator in accelerators.into_iter().flatten() {
            for device_rows in [1, 63, 64, 65, 600, 1000] {
                let mut hybrid =
                    HybridMatVec::new(accelerator, weights.clone(), shape, device_rows).unwrap();
                for _ in 0..2 {
                    assert_eq!(
                        hybrid.run(&inputs).unwrap(),
                        (expected.clone(), Processors::CpuAndGpu)
                    );
                }
            }
        }
    }

    #[test]
    fn cuda_kernel_saturates_like_the_reference() {
        let Ok(cuda) = Cuda::probe() else {
            return;
        };
        for (weight, input) in [(i16::MIN, i16::MIN), (i16::MAX, i16::MIN), (3, 7), (0, 0)] {
            let shape = Shape { rows: 5, cols: 300 };
            let weights = vec![weight; shape.rows * shape.cols];
            let inputs = vec![input; shape.cols];
            let mut kernel = CudaMatVec::new(&cuda, &weights, shape).unwrap();
            assert_eq!(
                kernel.run(&inputs).unwrap(),
                reference(&weights, &inputs, shape).unwrap()
            );
        }
    }
}

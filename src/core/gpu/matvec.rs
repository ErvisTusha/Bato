//! Integer matrix-vector kernel: the CPU reference, the GPU kernel and a
//! runner that uses both at once. Provenance: original.
//!
//! Each output is `clamp(sum(weight * input) >> OUTPUT_SHIFT, ±OUTPUT_LIMIT)`
//! with the sum taken in 64 bits, the arithmetic of a mixer layer. Every path
//! returns the same integers, so a result never depends on whether a GPU was
//! present.

use super::{Gpu, GpuError};
use wgpu::util::DeviceExt;

/// Fractional bits dropped from each sum; `matvec.wgsl` repeats the value.
pub const OUTPUT_SHIFT: u32 = 10;
/// Largest magnitude of an output; `matvec.wgsl` repeats the value.
pub const OUTPUT_LIMIT: i32 = 2047;
/// Rows per workgroup, the `@workgroup_size` of `matvec.wgsl`.
const WORKGROUP_ROWS: usize = 64;
/// Bytes of one 32-bit value in a GPU buffer.
const WORD: usize = 4;
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
fn words(values: &[i16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| i32::from(value).to_ne_bytes())
        .collect()
}

/// The kernel on a GPU, holding its weights in device memory.
pub struct GpuMatVec {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shape: Shape,
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    inputs: wgpu::Buffer,
    outputs: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl GpuMatVec {
    /// Uploads `weights` and compiles the kernel for `shape`.
    pub fn new(gpu: &Gpu, weights: &[i16], shape: Shape) -> Result<Self, GpuError> {
        shape.check(weights.len(), shape.cols)?;
        let limits = gpu.device.limits();
        let weight_bytes = (weights.len() as u64).saturating_mul(WORD as u64);
        let groups = shape.rows.div_ceil(WORKGROUP_ROWS) as u64;
        if weight_bytes > limits.max_storage_buffer_binding_size
            || weight_bytes > limits.max_buffer_size
            || groups > u64::from(limits.max_compute_workgroups_per_dimension)
        {
            return Err(GpuError::Shape(format!(
                "{} rows by {} columns exceeds the device limits",
                shape.rows, shape.cols
            )));
        }
        let device = gpu.device.clone();
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let storage = |label, contents: &[u8], usage| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let weight_buffer = storage("weights", &words(weights), wgpu::BufferUsages::STORAGE);
        let inputs = storage(
            "inputs",
            &vec![0; shape.cols * WORD],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let outputs = storage(
            "outputs",
            &vec![0; shape.rows * WORD],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let params: Vec<u8> = [shape.rows as u32, shape.cols as u32]
            .into_iter()
            .flat_map(u32::to_ne_bytes)
            .collect();
        let params = storage("params", &params, wgpu::BufferUsages::UNIFORM);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (shape.rows * WORD) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matvec"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matvec"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let buffers = [&weight_buffer, &inputs, &outputs, &params];
        let entries: Vec<_> = buffers
            .iter()
            .zip(0..)
            .map(|(buffer, binding)| wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matvec"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        if let Some(error) = pollster::block_on(scope.pop()) {
            return Err(GpuError::Device(error.to_string()));
        }
        Ok(Self {
            device,
            queue: gpu.queue.clone(),
            shape,
            pipeline,
            bindings,
            inputs,
            outputs,
            staging,
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
        let failed = |error: &dyn std::fmt::Display| GpuError::Execution(error.to_string());
        self.queue.write_buffer(&self.inputs, 0, &words(inputs));
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bindings, &[]);
            pass.dispatch_workgroups(self.shape.rows.div_ceil(WORKGROUP_ROWS) as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.outputs, 0, &self.staging, 0, None);
        self.queue.submit([encoder.finish()]);
        let (mapped, result) = std::sync::mpsc::channel();
        self.staging
            .map_async(wgpu::MapMode::Read, .., move |outcome| {
                // The receiver is gone only when `run` already returned an error.
                let _ = mapped.send(outcome);
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| failed(&error))?;
        result
            .recv()
            .map_err(|error| failed(&error))?
            .map_err(|error| failed(&error))?;
        let outputs = {
            let view = self
                .staging
                .get_mapped_range(..)
                .map_err(|error| failed(&error))?;
            view.chunks_exact(WORD)
                .map(|word| i32::from_ne_bytes([word[0], word[1], word[2], word[3]]))
                .collect()
        };
        self.staging.unmap();
        Ok(outputs)
    }
}

/// Which processors produced a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Processors {
    Cpu,
    CpuAndGpu,
}

/// The kernel on the GPU and the CPU at once: the GPU computes the first
/// `gpu_rows` rows while the CPU threads compute the rest. Without a GPU, or
/// when the GPU fails, the CPU computes every row; the outputs are the same.
pub struct HybridMatVec {
    weights: Vec<i16>,
    shape: Shape,
    gpu: Option<GpuMatVec>,
    gpu_rows: usize,
}

impl HybridMatVec {
    /// Prepares the kernel; `gpu_rows` of the rows go to `gpu` when present.
    pub fn new(
        gpu: Option<&Gpu>,
        weights: Vec<i16>,
        shape: Shape,
        gpu_rows: usize,
    ) -> Result<Self, GpuError> {
        shape.check(weights.len(), shape.cols)?;
        let gpu_rows = gpu_rows.min(shape.rows);
        let kernel = match gpu {
            Some(gpu) if gpu_rows > 0 => {
                let part = Shape {
                    rows: gpu_rows,
                    cols: shape.cols,
                };
                Some(GpuMatVec::new(
                    gpu,
                    &weights[..gpu_rows * shape.cols],
                    part,
                )?)
            }
            _ => None,
        };
        Ok(Self {
            weights,
            shape,
            gpu: kernel,
            gpu_rows: if gpu.is_some() { gpu_rows } else { 0 },
        })
    }

    /// Computes every output and says which processors took part. Mutates the
    /// GPU kernel's device buffers.
    pub fn run(&mut self, inputs: &[i16]) -> Result<(Vec<i32>, Processors), GpuError> {
        self.shape.check(self.weights.len(), inputs.len())?;
        let mut outputs = vec![0; self.shape.rows];
        let split = self.gpu_rows * self.shape.cols;
        let (gpu_part, cpu_part) = outputs.split_at_mut(self.gpu_rows);
        let (gpu_weights, cpu_weights) = self.weights.split_at(split);
        let Some(kernel) = &mut self.gpu else {
            fill_rows_threaded(cpu_weights, inputs, cpu_part);
            return Ok((outputs, Processors::Cpu));
        };
        let from_gpu = std::thread::scope(|scope| {
            scope.spawn(|| fill_rows_threaded(cpu_weights, inputs, cpu_part));
            kernel.run(inputs)
        });
        match from_gpu {
            Ok(rows) if rows.len() == gpu_part.len() => {
                gpu_part.copy_from_slice(&rows);
                Ok((outputs, Processors::CpuAndGpu))
            }
            // A lost or failing device is not an error of the computation:
            // the CPU computes the same rows.
            _ => {
                fill_rows_threaded(gpu_weights, inputs, gpu_part);
                Ok((outputs, Processors::Cpu))
            }
        }
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
    fn shader_is_valid_and_repeats_the_kernel_constants() {
        let module = wgpu::naga::front::wgsl::parse_str(SHADER).unwrap();
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        for constant in [
            format!("const OUTPUT_SHIFT: u32 = {OUTPUT_SHIFT}u;"),
            format!("const OUTPUT_LIMIT: i32 = {OUTPUT_LIMIT};"),
            format!("@workgroup_size({WORKGROUP_ROWS})"),
        ] {
            assert!(SHADER.contains(&constant), "{constant}");
        }
    }

    #[test]
    fn invalid_geometry_is_rejected() {
        let shape = Shape { rows: 2, cols: 3 };
        assert!(reference(&[0; 5], &[0; 3], shape).is_err());
        assert!(reference(&[0; 6], &[0; 2], shape).is_err());
        assert!(reference(&[], &[], Shape { rows: 0, cols: 0 }).is_err());
        assert!(HybridMatVec::new(None, vec![0; 5], shape, 1).is_err());
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
        let mut cpu = HybridMatVec::new(None, weights.clone(), shape, 600).unwrap();
        assert_eq!(
            cpu.run(&inputs).unwrap(),
            (expected.clone(), Processors::Cpu)
        );
        let gpu = match Gpu::probe() {
            Ok(gpu) => gpu,
            Err(reason) => {
                eprintln!("GPU parity not run on this machine: {reason}");
                return;
            }
        };
        for gpu_rows in [1, 63, 64, 65, 600, 1000] {
            let mut hybrid =
                HybridMatVec::new(Some(&gpu), weights.clone(), shape, gpu_rows).unwrap();
            for _ in 0..2 {
                assert_eq!(
                    hybrid.run(&inputs).unwrap(),
                    (expected.clone(), Processors::CpuAndGpu)
                );
            }
        }
    }
}

//! Integer implementations of the archive byte filters for wgpu.
//! Provenance: original packed-word port of the sibling CPU and CUDA filters.

use super::{filters::Direction, words, Gpu, GpuError};
use crate::core::types::TransformId;
use std::{sync::mpsc, time::Duration};
use wgpu::{
    BufferBindingType, BufferUsages, CommandEncoder, ComputePipeline, ErrorFilter, ShaderStages,
};

const SHADER: &str = include_str!("byte_filters.wgsl");
const THREADS: u32 = 64;
const TILE_BYTES: usize = 4096;
const WAIT_TIMEOUT: Duration = Duration::from_secs(60);

pub(super) struct WgpuFilters {
    gpu: Gpu,
    bindings: wgpu::BindGroupLayout,
    direct: ComputePipeline,
    delta_local: ComputePipeline,
    delta_carries: ComputePipeline,
    delta_finish: ComputePipeline,
}

fn checked<T>(device: &wgpu::Device, build: impl FnOnce() -> T) -> Result<T, GpuError> {
    let scopes = [
        ErrorFilter::OutOfMemory,
        ErrorFilter::Internal,
        ErrorFilter::Validation,
    ]
    .map(|filter| device.push_error_scope(filter));
    let outcome = build();
    let mut error = None;
    for scope in scopes.into_iter().rev() {
        if let Some(result) = pollster::block_on(scope.pop()) {
            error = Some(result);
        }
    }
    match error {
        Some(error) => Err(GpuError::Execution(error.to_string())),
        None => Ok(outcome),
    }
}

fn grid(groups: u32, limit: u32) -> Result<[u32; 2], GpuError> {
    let x = groups.min(limit);
    if x == 0 || groups.div_ceil(x) > limit {
        return Err(GpuError::Shape(format!(
            "wgpu filter needs {groups} workgroups"
        )));
    }
    Ok([x, groups.div_ceil(x)])
}

impl WgpuFilters {
    pub(super) fn open(adapter: &wgpu::Adapter) -> Result<Self, GpuError> {
        let gpu = Gpu::open_adapter(adapter)?;
        let (bindings, [direct, delta_local, delta_carries, delta_finish]) =
            checked(&gpu.device, || {
                let storage = |read_only| BufferBindingType::Storage { read_only };
                let entries = [
                    storage(true),
                    storage(false),
                    BufferBindingType::Uniform,
                    storage(false),
                ]
                .into_iter()
                .enumerate()
                .map(|(binding, ty)| wgpu::BindGroupLayoutEntry {
                    binding: binding as u32,
                    visibility: ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>();
                let bindings =
                    gpu.device
                        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                            label: Some("Bato byte filters"),
                            entries: &entries,
                        });
                let layout = gpu
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        bind_group_layouts: &[Some(&bindings)],
                        ..Default::default()
                    });
                let module = gpu
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Bato byte filters"),
                        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
                    });
                let pipelines =
                    ["direct", "delta_local", "delta_carries", "delta_finish"].map(|entry_point| {
                        gpu.device
                            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                                label: Some(entry_point),
                                layout: Some(&layout),
                                module: &module,
                                entry_point: Some(entry_point),
                                compilation_options: Default::default(),
                                cache: None,
                            })
                    });
                (bindings, pipelines)
            })?;
        Ok(Self {
            gpu,
            bindings,
            direct,
            delta_local,
            delta_carries,
            delta_finish,
        })
    }

    pub(super) fn apply(
        &self,
        input: &[u8],
        output: &mut [u8],
        (transform, direction): (TransformId, Direction),
    ) -> Result<u64, GpuError> {
        let size = input.len();
        let word_count = size.div_ceil(4);
        let limits = self.gpu.device.limits();
        let group_limit = limits.max_compute_workgroups_per_dimension;
        if size == 0
            || size > super::filters::MAX_BYTES
            || output.len() != size
            || !self.gpu.fits(word_count, 1)
        {
            return Err(GpuError::Shape(format!("wgpu filter of length {size}")));
        }
        let word_grid = grid((word_count as u32).div_ceil(THREADS), group_limit)?;
        let tile_count = size.div_ceil(TILE_BYTES);
        let tile_grid = grid(tile_count as u32, group_limit)?;
        let (kind, passes) = match (transform, direction) {
            (
                TransformId::Delta1 | TransformId::Delta2 | TransformId::Delta4,
                Direction::Inverse,
            ) => (transform as u32, 3),
            (TransformId::Shuffle2, Direction::Inverse) => (7, 1),
            (TransformId::Shuffle4, Direction::Inverse) => (8, 1),
            (
                TransformId::Delta1
                | TransformId::Delta2
                | TransformId::Delta4
                | TransformId::Shuffle2
                | TransformId::Shuffle4,
                Direction::Forward,
            ) => (transform as u32, 1),
            _ => {
                return Err(GpuError::Shape(format!(
                    "unsupported byte filter {transform:?}"
                )))
            }
        };
        checked(&self.gpu.device, || {
            let source = self.gpu.uploaded(input, BufferUsages::STORAGE);
            let result = self
                .gpu
                .zeroed(word_count, BufferUsages::STORAGE | BufferUsages::COPY_SRC);
            let parameters = self.gpu.uploaded(
                &words([size as u32, kind, word_grid[0], tile_grid[0]]),
                BufferUsages::UNIFORM,
            );
            let tiles = self.gpu.zeroed(tile_count, BufferUsages::STORAGE);
            let staging = self
                .gpu
                .zeroed(word_count, BufferUsages::COPY_DST | BufferUsages::MAP_READ);
            let entries =
                [&source, &result, &parameters, &tiles].map(|buffer| buffer.as_entire_binding());
            let bindings = self
                .gpu
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.bindings,
                    entries: &std::array::from_fn::<_, 4, _>(|index| wgpu::BindGroupEntry {
                        binding: index as u32,
                        resource: entries[index].clone(),
                    }),
                });
            let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
            let mut dispatch =
                |pipeline, groups| encode_pass(&mut encoder, pipeline, &bindings, groups);
            if passes == 1 {
                dispatch(&self.direct, word_grid);
            } else {
                dispatch(&self.delta_local, tile_grid);
                dispatch(&self.delta_carries, [1, 1]);
                dispatch(&self.delta_finish, word_grid);
            }
            encoder.copy_buffer_to_buffer(&result, 0, &staging, 0, None);
            let submission = self.gpu.queue.submit([encoder.finish()]);
            let (mapped, outcome) = mpsc::channel();
            staging.map_async(wgpu::MapMode::Read, .., move |status| {
                // The receiver may have gone away after a timeout or device loss.
                let _ = mapped.send(status);
            });
            let failed = |error: &dyn std::fmt::Display| GpuError::Execution(error.to_string());
            self.gpu
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: Some(WAIT_TIMEOUT),
                })
                .map_err(|error| failed(&error))?;
            outcome
                .recv_timeout(WAIT_TIMEOUT)
                .map_err(|error| failed(&error))?
                .map_err(|error| failed(&error))?;
            {
                let view = staging
                    .get_mapped_range(..)
                    .map_err(|error| failed(&error))?;
                output.copy_from_slice(&view[..size]);
            }
            staging.unmap();
            Ok(passes)
        })?
    }
}

fn encode_pass(
    encoder: &mut CommandEncoder,
    pipeline: &ComputePipeline,
    bindings: &wgpu::BindGroup,
    [x, y]: [u32; 2],
) {
    let mut pass = encoder.begin_compute_pass(&Default::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bindings, &[]);
    pass.dispatch_workgroups(x, y, 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_and_large_dispatch_grid_are_valid() {
        super::super::validate_shader(SHADER);
        assert_eq!(grid(65_536, 65_535).unwrap(), [65_535, 2]);
        assert!(grid(5, 2).is_err());
    }
}

//! Independent serial streams on the CPU and on the GPU. Provenance: original;
//! the hash is a generic multiply-xorshift finalizer.
//!
//! A stream is a chain of dependent table lookups: each step reads a cell at a
//! hashed position, adapts it and derives the next position from the value
//! read, as the lookups of a context model do. A whole stream runs inside the
//! device with no round trip per step, so this measures how fast a processor
//! advances serial model work and how many such streams it holds at once.

use super::{words, Gpu, GpuError, Kernel};

/// Value of a cell that has only seen one bits; `streams.wgsl` repeats it.
const CELL_ONE: u32 = 65535;
/// Adaptation rate of a cell as a shift; `streams.wgsl` repeats it.
const RATE_SHIFT: u32 = 4;
/// Streams per workgroup, the `@workgroup_size` of `streams.wgsl`.
const WORKGROUP_STREAMS: usize = 64;
/// Steps every stream takes in one GPU dispatch: short enough that a dispatch
/// ends well within a driver's watchdog time.
pub const DISPATCH_STEPS: u32 = 1 << 14;
/// Largest table of one stream as a power of two of 32-bit cells.
const MAX_CELLS_LOG2: u32 = 30;
const SHADER: &str = include_str!("streams.wgsl");

/// Number of streams and the size of each stream's table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub streams: usize,
    /// Cells of one table as a power of two.
    pub cells_log2: u32,
}

impl Layout {
    /// Cells of all tables together.
    fn cells(self) -> Result<usize, GpuError> {
        let invalid = || {
            GpuError::Shape(format!(
                "{} streams of 2^{} cells",
                self.streams, self.cells_log2
            ))
        };
        if self.streams == 0 || self.streams > u32::MAX as usize {
            return Err(invalid());
        }
        if !(1..=MAX_CELLS_LOG2).contains(&self.cells_log2) {
            return Err(invalid());
        }
        self.streams
            .checked_mul(1 << self.cells_log2)
            .ok_or_else(invalid)
    }
}

/// Multiply-xorshift hash of a 32-bit value.
fn mix(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x9e37_79b1);
    x ^= x >> 13;
    x = x.wrapping_mul(0x85eb_ca6b);
    x ^= x >> 16;
    x
}

/// First state of each stream.
fn seeds(streams: usize) -> Vec<u32> {
    (0..streams as u32).map(|stream| mix(stream + 1)).collect()
}

/// Advances one stream by `steps`, mutating its table, and returns its state.
fn advance_stream(table: &mut [u32], mut state: u32, steps: u32) -> u32 {
    let mask = table.len() - 1;
    for _ in 0..steps {
        let slot = mix(state) as usize & mask;
        let cell = table[slot];
        let goal = (state & 1) * CELL_ONE;
        let next = (cell as i32 + ((goal as i32 - cell as i32) >> RATE_SHIFT)) as u32;
        table[slot] = next;
        state = mix(state.wrapping_add(next).wrapping_add(1));
    }
    state
}

/// How many CPU threads advance the streams.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threads {
    One,
    All,
}

/// The streams on the CPU: the definition the GPU must equal.
pub struct CpuStreams {
    layout: Layout,
    table: Vec<u32>,
    states: Vec<u32>,
}

impl CpuStreams {
    pub fn new(layout: Layout) -> Result<Self, GpuError> {
        Ok(Self {
            layout,
            table: vec![0; layout.cells()?],
            states: seeds(layout.streams),
        })
    }

    /// Advances every stream by `steps`, mutating tables and states, and
    /// returns the states.
    pub fn advance(&mut self, steps: u32, threads: Threads) -> &[u32] {
        let workers = match threads {
            Threads::One => 1,
            Threads::All => std::thread::available_parallelism().map_or(1, usize::from),
        };
        let cells = 1 << self.layout.cells_log2;
        let share = self.layout.streams.div_ceil(workers);
        std::thread::scope(|scope| {
            for (tables, states) in self
                .table
                .chunks_mut(share * cells)
                .zip(self.states.chunks_mut(share))
            {
                scope.spawn(move || {
                    for (table, state) in tables.chunks_mut(cells).zip(states) {
                        *state = advance_stream(table, *state, steps);
                    }
                });
            }
        });
        &self.states
    }
}

/// The streams on a GPU, with tables and states in device memory.
pub struct GpuStreams {
    gpu: Gpu,
    layout: Layout,
    kernel: Kernel,
    states: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl GpuStreams {
    pub fn new(gpu: &Gpu, layout: Layout) -> Result<Self, GpuError> {
        let cells = layout.cells()?;
        if !gpu.fits(cells, layout.streams.div_ceil(WORKGROUP_STREAMS)) {
            return Err(GpuError::Shape(format!(
                "{} streams of 2^{} cells exceed the device limits",
                layout.streams, layout.cells_log2
            )));
        }
        gpu.validated(|| {
            let table = gpu.zeroed(cells, wgpu::BufferUsages::STORAGE);
            let states = gpu.uploaded(
                &words(seeds(layout.streams)),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let params = gpu.uploaded(
                &words([layout.streams as u32, layout.cells_log2, DISPATCH_STEPS]),
                wgpu::BufferUsages::UNIFORM,
            );
            Self {
                gpu: gpu.clone(),
                layout,
                kernel: gpu.kernel(SHADER, &[&table, &states, &params]),
                staging: gpu.zeroed(
                    layout.streams,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                ),
                states,
            }
        })
    }

    /// Advances every stream by `dispatches * DISPATCH_STEPS` steps, mutating
    /// the device tables and states, and returns the states.
    pub fn advance(&mut self, dispatches: u32) -> Result<Vec<u32>, GpuError> {
        let groups = self.layout.streams.div_ceil(WORKGROUP_STREAMS) as u32;
        for _ in 1..dispatches {
            let pass = self.gpu.pass(&self.kernel, groups);
            self.gpu.queue.submit([pass.finish()]);
        }
        let last = if dispatches == 0 {
            self.gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
        } else {
            self.gpu.pass(&self.kernel, groups)
        };
        self.gpu.read(last, &self.states, &self.staging)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_is_valid_and_repeats_the_kernel_constants() {
        super::super::validate_shader(SHADER);
        for constant in [
            format!("const CELL_ONE: u32 = {CELL_ONE}u;"),
            format!("const RATE_SHIFT: u32 = {RATE_SHIFT}u;"),
            format!("@workgroup_size({WORKGROUP_STREAMS})"),
        ] {
            assert!(SHADER.contains(&constant), "{constant}");
        }
    }

    #[test]
    fn invalid_layouts_are_rejected() {
        for (streams, cells_log2) in [(0, 4), (1, 0), (1, 31), (usize::MAX, 4)] {
            assert!(CpuStreams::new(Layout {
                streams,
                cells_log2
            })
            .is_err());
        }
    }

    #[test]
    fn streams_adapt_their_tables_and_ignore_the_thread_count() {
        let layout = Layout {
            streams: 37,
            cells_log2: 6,
        };
        let mut one = CpuStreams::new(layout).unwrap();
        let mut all = CpuStreams::new(layout).unwrap();
        let before = one.states.clone();
        for steps in [1, 1000, 4096] {
            assert_eq!(
                one.advance(steps, Threads::One),
                all.advance(steps, Threads::All)
            );
        }
        assert_eq!(one.table, all.table);
        assert!(one.states.iter().zip(&before).all(|(now, was)| now != was));
        assert!(one.table.iter().all(|&cell| cell <= CELL_ONE));
        assert!(one.table.iter().any(|&cell| cell > 0));
    }

    #[test]
    fn gpu_streams_equal_the_cpu_streams() {
        let gpu = match Gpu::probe() {
            Ok(gpu) => gpu,
            Err(reason) => {
                eprintln!("GPU parity not run on this machine: {reason}");
                return;
            }
        };
        for (streams, cells_log2) in [(1, 10), (65, 8), (200, 12)] {
            let layout = Layout {
                streams,
                cells_log2,
            };
            let mut cpu = CpuStreams::new(layout).unwrap();
            let mut device = GpuStreams::new(&gpu, layout).unwrap();
            assert_eq!(device.advance(0).unwrap(), cpu.states);
            for dispatches in [1, 3] {
                let expected = cpu.advance(dispatches * DISPATCH_STEPS, Threads::All);
                assert_eq!(device.advance(dispatches).unwrap(), expected);
            }
        }
    }
}

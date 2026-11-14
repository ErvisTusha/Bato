//! Measures two integer workloads on the CPU, through wgpu and through CUDA,
//! and checks that every path returns the same integers: a matrix-vector
//! kernel that makes one round trip per call, and whole serial streams that
//! run inside the device. This is a diagnostic for deciding which workloads a
//! GPU can take; it is not the canonical benchmark workflow and no archive
//! depends on it.
//!
//! Run with: `cargo run --release --example gpu_probe` (about two minutes).
use bato_engine::core::gpu::{
    cuda::Cuda,
    matvec::{reference, Accelerator, CudaMatVec, GpuMatVec, HybridMatVec, Processors, Shape},
    streams::{CpuStreams, CudaStreams, GpuStreams, Layout, Threads, DISPATCH_STEPS},
    Gpu, GpuError,
};
use std::time::{Duration, Instant};

type Outcome<T> = Result<T, Box<dyn std::error::Error>>;

/// Inputs of one mixer weight set.
const COLS: usize = 64;
/// Row counts measured: one weight set up to a large dense layer.
const ROWS: [usize; 7] = [1, 64, 1024, 16_384, 65_536, 262_144, 524_288];
/// Load applied before timing, so that the CPU and the GPU leave their
/// power-saving states.
const WARM_UP: Duration = Duration::from_millis(300);
/// Time over which calls are measured; the median call is reported.
const MEASURE: Duration = Duration::from_millis(300);
/// Fewest and most calls measured, whatever their duration.
const CALLS: (usize, usize) = (3, 100_000);
/// Stream layouts measured, as (table cells log2, streams): one stream and as
/// many as fill 128 MiB, the default largest wgpu buffer, for tables of
/// 32 MiB, 1 MiB and 64 KiB.
const LAYOUTS: [(u32, usize); 6] = [(23, 1), (23, 4), (18, 1), (18, 128), (14, 1), (14, 2048)];
/// Steps over all streams aimed at per timed call, as a power of two, and the
/// fewest and most dispatches that may take.
const CALL_STEPS_LOG2: u32 = 26;
const DISPATCHES: (u32, u32) = (4, 64);
/// Dispatches after which the stream states of every path are compared.
const PARITY_DISPATCHES: u32 = 2;

/// Fixed-seed integer hash: reproducible values.
fn values(count: usize, seed: u64) -> Vec<i16> {
    (0..count as u64)
        .map(|i| {
            let mut x = (i + seed).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            x ^= x >> 29;
            x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            (x >> 44) as i16 - 2048
        })
        .collect()
}

/// Median time of one `call`, measured back to back after a warm-up.
fn timed(mut call: impl FnMut() -> Result<(), GpuError>) -> Result<Duration, GpuError> {
    let warm = Instant::now();
    call()?;
    while warm.elapsed() < WARM_UP {
        call()?;
    }
    let mut times = Vec::new();
    let started = Instant::now();
    while times.len() < CALLS.0 || (started.elapsed() < MEASURE && times.len() < CALLS.1) {
        let call_started = Instant::now();
        call()?;
        times.push(call_started.elapsed());
    }
    times.sort();
    Ok(times[times.len() / 2])
}

fn micros(time: Duration) -> String {
    format!("{:.1}", time.as_secs_f64() * 1e6)
}

/// A measured time, or a dash for a device that is not present.
fn cell(time: Option<Duration>, format: impl Fn(Duration) -> String) -> String {
    time.map_or_else(|| String::from("-"), format)
}

/// The devices found on this machine.
struct Devices {
    wgpu: Option<Gpu>,
    cuda: Option<Cuda>,
}

impl Devices {
    fn probe() -> Outcome<Self> {
        let wgpu = match Gpu::probe() {
            Ok(gpu) => {
                println!("wgpu: {}", gpu.description());
                Some(gpu)
            }
            Err(reason) => {
                println!("wgpu: unavailable ({reason})");
                None
            }
        };
        let cuda = match Cuda::probe() {
            Ok(cuda) => {
                println!("cuda: {}", cuda.description());
                Some(cuda)
            }
            Err(reason) => {
                println!("cuda: unavailable ({reason})");
                None
            }
        };
        let devices = Self { wgpu, cuda };
        let wgpu_idle = match &devices.wgpu {
            Some(gpu) => Some(timed(|| gpu.idle_call())?),
            None => None,
        };
        let cuda_idle = match &devices.cuda {
            Some(cuda) => Some(timed(|| cuda.idle_call())?),
            None => None,
        };
        println!(
            "idle call, microseconds: wgpu {}, cuda {}",
            cell(wgpu_idle, micros),
            cell(cuda_idle, micros)
        );
        Ok(devices)
    }

    /// The device a hybrid run uses: CUDA when present, else wgpu.
    fn accelerator(&self) -> Accelerator<'_> {
        match (&self.cuda, &self.wgpu) {
            (Some(cuda), _) => Accelerator::Cuda(cuda),
            (None, Some(gpu)) => Accelerator::Wgpu(gpu),
            (None, None) => Accelerator::None,
        }
    }
}

/// Times the matrix-vector kernel: one round trip per call.
fn matrix_table(devices: &Devices) -> Outcome<()> {
    println!("matrix-vector kernel, microseconds per call");
    println!(
        "{:>8} {:>12} {:>10} {:>10} {:>10} {:>10} {:>9}  result",
        "rows", "cpu 1 thread", "cpu all", "wgpu", "cuda", "cpu+gpu", "gpu rows"
    );
    let inputs = values(COLS, 2);
    for rows in ROWS {
        let shape = Shape { rows, cols: COLS };
        let weights = values(rows * COLS, 1);
        let expected = reference(&weights, &inputs, shape)?;
        let single = timed(|| reference(&weights, &inputs, shape).map(|_| ()))?;
        let mut all_cpu = HybridMatVec::new(Accelerator::None, weights.clone(), shape, 0)?;
        let mut identical = all_cpu.run(&inputs)?.0 == expected;
        let cpu = timed(|| all_cpu.run(&inputs).map(|_| ()))?;
        let mut wgpu = None;
        if let Some(gpu) = &devices.wgpu {
            let mut kernel = GpuMatVec::new(gpu, &weights, shape)?;
            identical &= kernel.run(&inputs)? == expected;
            wgpu = Some(timed(|| kernel.run(&inputs).map(|_| ()))?);
        }
        let mut cuda = None;
        if let Some(device) = &devices.cuda {
            let mut kernel = CudaMatVec::new(device, &weights, shape)?;
            identical &= kernel.run(&inputs)? == expected;
            cuda = Some(timed(|| kernel.run(&inputs).map(|_| ()))?);
        }
        // Rows go to the device in proportion to the measured speeds.
        let (mut both, mut share) = (None, String::from("-"));
        if let Some(device) = cuda.or(wgpu) {
            let fraction = cpu.as_secs_f64() / (cpu.as_secs_f64() + device.as_secs_f64());
            let device_rows = ((rows as f64 * fraction).round() as usize).clamp(1, rows);
            let mut hybrid =
                HybridMatVec::new(devices.accelerator(), weights.clone(), shape, device_rows)?;
            let (outputs, processors) = hybrid.run(&inputs)?;
            identical &= outputs == expected;
            if processors != Processors::CpuAndGpu {
                return Err("the device failed during the hybrid run".into());
            }
            both = Some(timed(|| hybrid.run(&inputs).map(|_| ()))?);
            share = format!("{:.0}%", 100.0 * device_rows as f64 / rows as f64);
        }
        println!(
            "{rows:>8} {:>12} {:>10} {:>10} {:>10} {:>10} {share:>9}  {}",
            micros(single),
            micros(cpu),
            cell(wgpu, micros),
            cell(cuda, micros),
            cell(both, micros),
            if identical { "identical" } else { "MISMATCH" }
        );
        if !identical {
            return Err("a path returned different integers".into());
        }
    }
    Ok(())
}

/// Checks that every path reaches the same stream states.
fn streams_agree(devices: &Devices, layout: Layout) -> Outcome<bool> {
    let steps = PARITY_DISPATCHES * DISPATCH_STEPS;
    let expected = CpuStreams::new(layout)?
        .advance(steps, Threads::One)
        .to_vec();
    let mut identical = CpuStreams::new(layout)?.advance(steps, Threads::All) == expected;
    if let Some(gpu) = &devices.wgpu {
        identical &= GpuStreams::new(gpu, layout)?.advance(PARITY_DISPATCHES)? == expected;
    }
    if let Some(cuda) = &devices.cuda {
        identical &= CudaStreams::new(cuda, layout)?.advance(PARITY_DISPATCHES)? == expected;
    }
    Ok(identical)
}

/// Times whole serial streams inside each processor: no round trip per step.
fn stream_table(devices: &Devices) -> Outcome<()> {
    println!("serial streams, million steps per second over all streams (ns per step of a stream)");
    println!(
        "{:>10} {:>8} {:>14} {:>10} {:>16} {:>16}  result",
        "table KiB", "streams", "cpu 1 thread", "cpu all", "wgpu", "cuda"
    );
    for (cells_log2, streams) in LAYOUTS {
        let layout = Layout {
            streams,
            cells_log2,
        };
        let identical = streams_agree(devices, layout)?;
        let wanted = (1u64 << CALL_STEPS_LOG2) / (streams as u64 * u64::from(DISPATCH_STEPS));
        let dispatches = (wanted as u32).clamp(DISPATCHES.0, DISPATCHES.1);
        let steps = dispatches * DISPATCH_STEPS;
        let total = u64::from(steps) * streams as u64;
        let rate = |time: Duration| format!("{:.1}", total as f64 / time.as_secs_f64() / 1e6);
        // A device advances all streams at once; one CPU thread takes them
        // in turn, so its time per step is over all of them.
        let with_step = |time: Duration| {
            let step = time.as_secs_f64() * 1e9 / f64::from(steps);
            format!("{} ({step:.0})", rate(time))
        };
        let in_turn = |time: Duration| {
            let step = time.as_secs_f64() * 1e9 / total as f64;
            format!("{} ({step:.0})", rate(time))
        };
        let mut one = CpuStreams::new(layout)?;
        let single = timed(|| {
            one.advance(steps, Threads::One);
            Ok(())
        })?;
        let mut all = CpuStreams::new(layout)?;
        let threaded = timed(|| {
            all.advance(steps, Threads::All);
            Ok(())
        })?;
        let mut wgpu = None;
        if let Some(gpu) = &devices.wgpu {
            let mut device = GpuStreams::new(gpu, layout)?;
            wgpu = Some(timed(|| device.advance(dispatches).map(|_| ()))?);
        }
        let mut cuda = None;
        if let Some(cuda_device) = &devices.cuda {
            let mut device = CudaStreams::new(cuda_device, layout)?;
            cuda = Some(timed(|| device.advance(dispatches).map(|_| ()))?);
        }
        println!(
            "{:>10} {streams:>8} {:>14} {:>10} {:>16} {:>16}  {}",
            (4usize << cells_log2) / 1024,
            in_turn(single),
            rate(threaded),
            cell(wgpu, with_step),
            cell(cuda, with_step),
            if identical { "identical" } else { "MISMATCH" }
        );
        if !identical {
            return Err("a path returned different stream states".into());
        }
    }
    Ok(())
}

fn main() -> Outcome<()> {
    let devices = Devices::probe()?;
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    println!("cpu threads: {threads}");
    matrix_table(&devices)?;
    stream_table(&devices)
}

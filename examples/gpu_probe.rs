//! Measures two integer workloads on the CPU and on the GPU and checks that
//! every path returns the same integers: a matrix-vector kernel that makes one
//! round trip per call, and whole serial streams that run inside the device.
//! This is a diagnostic for deciding which workloads a GPU can take; it is not
//! the canonical benchmark workflow and no archive depends on it.
//!
//! Run with: `cargo run --release --example gpu_probe`
use bato_engine::core::gpu::{
    matvec::{reference, GpuMatVec, HybridMatVec, Processors, Shape},
    streams::{CpuStreams, GpuStreams, Layout, Threads, DISPATCH_STEPS},
    Gpu,
};
use std::time::{Duration, Instant};

/// Inputs of one mixer weight set.
const COLS: usize = 64;
/// Row counts measured: one weight set up to a large dense layer.
const ROWS: [usize; 7] = [1, 64, 1024, 16_384, 65_536, 262_144, 524_288];
/// Timed runs per matrix measurement; the median is reported.
const RUNS: usize = 15;
/// Stream layouts measured, as (table cells log2, streams): one stream and as
/// many as fill 128 MiB, the default largest buffer, for tables of 32 MiB,
/// 1 MiB and 64 KiB.
const LAYOUTS: [(u32, usize); 6] = [(23, 1), (23, 4), (18, 1), (18, 128), (14, 1), (14, 2048)];
/// GPU dispatches per timed stream run.
const STREAM_DISPATCHES: u32 = 4;
/// Timed runs per stream measurement; the median is reported.
const STREAM_RUNS: usize = 5;

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

/// Median time of `runs` calls after one untimed call.
fn median(runs: usize, mut run: impl FnMut()) -> Duration {
    run();
    let mut times: Vec<Duration> = (0..runs)
        .map(|_| {
            let started = Instant::now();
            run();
            started.elapsed()
        })
        .collect();
    times.sort();
    times[runs / 2]
}

fn micros(time: Duration) -> String {
    format!("{:.1}", time.as_secs_f64() * 1e6)
}

/// Times the matrix-vector kernel: one round trip per call.
fn matrix_table(gpu: Option<&Gpu>) -> Result<(), Box<dyn std::error::Error>> {
    println!("matrix-vector kernel, microseconds per call");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12}  result",
        "rows", "cpu 1 thread", "cpu all", "gpu only", "cpu+gpu"
    );
    let inputs = values(COLS, 2);
    for rows in ROWS {
        let shape = Shape { rows, cols: COLS };
        let weights = values(rows * COLS, 1);
        let expected = reference(&weights, &inputs, shape)?;
        let single = median(RUNS, || {
            reference(&weights, &inputs, shape).expect("validated shape");
        });
        let mut all_cpu = HybridMatVec::new(None, weights.clone(), shape, 0)?;
        let mut identical = all_cpu.run(&inputs)?.0 == expected;
        let cpu = median(RUNS, || {
            all_cpu.run(&inputs).expect("validated shape");
        });
        let (mut gpu_only, mut both) = (String::from("-"), String::from("-"));
        if let Some(gpu) = gpu {
            let mut kernel = GpuMatVec::new(gpu, &weights, shape)?;
            identical &= kernel.run(&inputs)? == expected;
            gpu_only = micros(median(RUNS, || {
                kernel.run(&inputs).expect("GPU kernel");
            }));
            // Half of the rows on each processor, at the same time.
            let mut hybrid = HybridMatVec::new(Some(gpu), weights.clone(), shape, rows / 2)?;
            let (outputs, processors) = hybrid.run(&inputs)?;
            identical &= outputs == expected;
            if rows > 1 && processors != Processors::CpuAndGpu {
                return Err("the GPU failed during the hybrid run".into());
            }
            both = micros(median(RUNS, || {
                hybrid.run(&inputs).expect("hybrid kernel");
            }));
        }
        println!(
            "{rows:>8} {:>12} {:>12} {gpu_only:>12} {both:>12}  {}",
            micros(single),
            micros(cpu),
            if identical { "identical" } else { "MISMATCH" }
        );
        if !identical {
            return Err("a path returned different integers".into());
        }
    }
    Ok(())
}

/// Million steps per second of `steps` taken in `time`.
fn rate(steps: u64, time: Duration) -> String {
    format!("{:.1}", steps as f64 / time.as_secs_f64() / 1e6)
}

/// Times whole serial streams inside each processor: no round trip per step.
fn stream_table(gpu: Option<&Gpu>) -> Result<(), Box<dyn std::error::Error>> {
    println!("serial streams, million steps per second over all streams");
    println!(
        "{:>10} {:>8} {:>12} {:>12} {:>12} {:>14}  result",
        "table KiB", "streams", "cpu 1 thread", "cpu all", "gpu", "gpu ns/step"
    );
    let steps = STREAM_DISPATCHES * DISPATCH_STEPS;
    for (cells_log2, streams) in LAYOUTS {
        let layout = Layout {
            streams,
            cells_log2,
        };
        let total = u64::from(steps) * streams as u64;
        let mut one = CpuStreams::new(layout)?;
        let single = median(STREAM_RUNS, || {
            one.advance(steps, Threads::One);
        });
        let mut all = CpuStreams::new(layout)?;
        let threaded = median(STREAM_RUNS, || {
            all.advance(steps, Threads::All);
        });
        // Every path has now advanced `STREAM_RUNS + 1` times.
        let expected = one.advance(0, Threads::One).to_vec();
        let mut identical = all.advance(0, Threads::All) == expected;
        let (mut device_rate, mut device_step) = (String::from("-"), String::from("-"));
        if let Some(gpu) = gpu {
            let mut device = GpuStreams::new(gpu, layout)?;
            let time = median(STREAM_RUNS, || {
                device.advance(STREAM_DISPATCHES).expect("GPU streams");
            });
            identical &= device.advance(0)? == expected;
            device_rate = rate(total, time);
            device_step = format!("{:.1}", time.as_secs_f64() * 1e9 / f64::from(steps));
        }
        println!(
            "{:>10} {streams:>8} {:>12} {:>12} {device_rate:>12} {device_step:>14}  {}",
            (4usize << cells_log2) / 1024,
            rate(total, single),
            rate(total, threaded),
            if identical { "identical" } else { "MISMATCH" }
        );
        if !identical {
            return Err("a path returned different stream states".into());
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = match Gpu::probe() {
        Ok(gpu) => {
            println!("gpu: {}", gpu.description());
            Some(gpu)
        }
        Err(reason) => {
            println!("gpu: unavailable ({reason}); every path below runs on the CPU");
            None
        }
    };
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    println!("cpu threads: {threads}");
    matrix_table(gpu.as_ref())?;
    stream_table(gpu.as_ref())
}

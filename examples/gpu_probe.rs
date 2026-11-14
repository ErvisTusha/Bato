//! Measures the integer matrix-vector kernel on the CPU, on the GPU and on
//! both at once, and checks that every path returns the same integers. This is
//! a diagnostic for deciding which workloads a GPU can take; it is not the
//! canonical benchmark workflow and no archive depends on it.
//!
//! Run with: `cargo run --release --example gpu_probe`
use bato_engine::core::gpu::{
    matvec::{reference, GpuMatVec, HybridMatVec, Processors, Shape},
    Gpu,
};
use std::time::{Duration, Instant};

/// Inputs of one mixer weight set.
const COLS: usize = 64;
/// Row counts measured: one weight set up to a large dense layer.
const ROWS: [usize; 7] = [1, 64, 1024, 16_384, 65_536, 262_144, 524_288];
/// Timed runs per measurement; the median is reported.
const RUNS: usize = 15;

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

/// Median time of `RUNS` calls after one untimed call.
fn median(mut run: impl FnMut()) -> Duration {
    run();
    let mut times: Vec<Duration> = (0..RUNS)
        .map(|_| {
            let started = Instant::now();
            run();
            started.elapsed()
        })
        .collect();
    times.sort();
    times[RUNS / 2]
}

fn micros(time: Duration) -> String {
    format!("{:.1}", time.as_secs_f64() * 1e6)
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
    println!("cpu threads: {threads}; times are medians of {RUNS} runs in microseconds");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12}  result",
        "rows", "cpu 1 thread", "cpu all", "gpu only", "cpu+gpu"
    );
    let inputs = values(COLS, 2);
    for rows in ROWS {
        let shape = Shape { rows, cols: COLS };
        let weights = values(rows * COLS, 1);
        let expected = reference(&weights, &inputs, shape)?;
        let single = median(|| {
            reference(&weights, &inputs, shape).expect("validated shape");
        });
        let mut all_cpu = HybridMatVec::new(None, weights.clone(), shape, 0)?;
        let mut identical = all_cpu.run(&inputs)?.0 == expected;
        let cpu = median(|| {
            all_cpu.run(&inputs).expect("validated shape");
        });
        let (mut gpu_only, mut both) = (String::from("-"), String::from("-"));
        if let Some(gpu) = &gpu {
            let mut kernel = GpuMatVec::new(gpu, &weights, shape)?;
            identical &= kernel.run(&inputs)? == expected;
            gpu_only = micros(median(|| {
                kernel.run(&inputs).expect("GPU kernel");
            }));
            // Half of the rows on each processor, at the same time.
            let mut hybrid = HybridMatVec::new(Some(gpu), weights.clone(), shape, rows / 2)?;
            let (outputs, processors) = hybrid.run(&inputs)?;
            identical &= outputs == expected;
            if rows > 1 && processors != Processors::CpuAndGpu {
                return Err("the GPU failed during the hybrid run".into());
            }
            both = micros(median(|| {
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

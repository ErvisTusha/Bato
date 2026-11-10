//! Memory admission and an ordered worker pool for independent segments.
use super::cm::{
    descriptor::Descriptor,
    profile::Step,
    segment::{self, Chunk, Coding, Direction, Trials},
};
use super::memory::{HostBudget, MemoryLease};
use super::{checksum::crc32c, types::*};
use std::{
    collections::BTreeMap,
    sync::{mpsc, Arc, Mutex},
    thread::JoinHandle,
};

/// Fixed host allowance for the pool's own bookkeeping and I/O buffers.
const HOST_RESERVE: usize = 64 * 1024 * 1024;
/// Stack size of every worker thread, charged to the memory budget.
const STACK_MEMORY: usize = 2 * 1024 * 1024;

/// Workers admitted for one stream and the host memory they reserve.
#[derive(Debug, Clone)]
pub struct Admission {
    pub workers: usize,
    pub host_memory: usize,
}

/// Admits as many segment workers as the requested thread count and the
/// memory budget allow, failing before any output if not even one fits.
pub(crate) fn admit(
    (descriptor, direction): (Descriptor, Direction),
    budget: usize,
    options: &ExecutionOptions,
) -> Result<Admission, BatoError> {
    options.validate()?;
    let per_worker = segment::memory_required(descriptor, direction)?.saturating_add(STACK_MEMORY);
    let desired = if options.threads == 0 {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
            .min(MAX_THREADS)
    } else {
        options.threads
    };
    let workers = budget
        .saturating_sub(HOST_RESERVE)
        .checked_div(per_worker)
        .unwrap_or(0)
        .min(desired);
    if workers == 0 {
        return Err(BatoError::MemoryLimitExceeded {
            required: HOST_RESERVE.saturating_add(per_worker),
            limit: budget,
        });
    }
    Ok(Admission {
        workers,
        host_memory: HOST_RESERVE + per_worker * workers,
    })
}

pub(crate) enum Work {
    Encode {
        raw: Vec<u8>,
        descriptor: Descriptor,
        trials: Trials,
    },
    Decode {
        chunks: Vec<Chunk>,
        descriptor: Descriptor,
        coding: Coding,
    },
}

#[derive(Debug)]
pub(crate) struct EncodedChunk {
    pub header: BlockHeader,
    pub bytes: Vec<u8>,
    pub crc: u32,
}

/// Length and CRC32C of a segment before any transform.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Original {
    pub length: u64,
    pub crc: u32,
}

#[derive(Debug)]
pub(crate) enum Output {
    Encoded(Vec<EncodedChunk>, Original),
    Decoded { bytes: Vec<u8>, crc: u32 },
}

#[derive(Debug)]
pub(crate) struct Completed {
    pub output: Output,
    pub memory: Option<MemoryLease>,
}

/// Frames one coded or raw chunk as a block of a segment using `coding`.
fn encoded_chunk(chunk: Chunk, coding: Coding) -> EncodedChunk {
    EncodedChunk {
        header: BlockHeader {
            block_type: chunk.kind,
            transform_id: coding.transform,
            primer: coding.primer,
            uncompressed_size: chunk.length,
            compressed_size: chunk.payload.len() as u32,
        },
        bytes: chunk.payload,
        crc: chunk.crc,
    }
}

fn execute(work: Work, stats: &ExecutionStats) -> Result<Output, BatoError> {
    match work {
        Work::Encode {
            mut raw,
            descriptor,
            trials,
        } => {
            let timing = stats.timing();
            let original = Original {
                length: raw.len() as u64,
                crc: timing.time(Step::Checksum, || crc32c(&raw)),
            };
            let (chunks, coding) = segment::encode_segment(&mut raw, (descriptor, trials), timing)?;
            stats.update(|s| s.cpu_blocks += chunks.len() as u64);
            Ok(Output::Encoded(
                chunks
                    .into_iter()
                    .map(|chunk| encoded_chunk(chunk, coding))
                    .collect(),
                original,
            ))
        }
        Work::Decode {
            chunks,
            descriptor,
            coding,
        } => {
            let timing = stats.timing();
            let max_output = descriptor.segment_bytes();
            let bytes = segment::decode_segment(&chunks, (descriptor, max_output), coding, timing)?;
            stats.update(|s| s.cpu_blocks += chunks.len() as u64);
            Ok(Output::Decoded {
                crc: timing.time(Step::Checksum, || crc32c(&bytes)),
                bytes,
            })
        }
    }
}

type Job = (u64, Work, MemoryLease);
type JobResult = (u64, Result<Completed, BatoError>);

/// Runs segments on worker threads and returns their results in submission
/// order. Each job holds a memory lease until its result is consumed.
pub(crate) struct Pool {
    jobs: Option<mpsc::Sender<Job>>,
    results: mpsc::Receiver<JobResult>,
    handles: Vec<JoinHandle<()>>,
    ready: BTreeMap<u64, Result<Completed, BatoError>>,
    next: u64,
    submitted: u64,
    pub admission: Admission,
    host: Arc<HostBudget>,
    job_memory: usize,
    parallel_trials_fit: bool,
    _infrastructure: MemoryLease,
}

impl Pool {
    pub(crate) fn new(
        (descriptor, direction): (Descriptor, Direction),
        host: Arc<HostBudget>,
        options: &ExecutionOptions,
    ) -> Result<Self, BatoError> {
        let admission = admit((descriptor, direction), host.available(), options)?;
        options.stats.update(|s| {
            s.cpu_workers = admission.workers;
            s.estimated_host_memory = host.used().saturating_add(admission.host_memory);
        });
        let infrastructure = host.reserve(HOST_RESERVE + STACK_MEMORY * admission.workers)?;
        let (jobs, incoming) = mpsc::channel::<Job>();
        let incoming = Arc::new(Mutex::new(incoming));
        let (finished, results) = mpsc::channel::<JobResult>();
        let mut handles = Vec::with_capacity(admission.workers);
        for _ in 0..admission.workers {
            let incoming = incoming.clone();
            let finished = finished.clone();
            let stats = options.stats.clone();
            handles.push(
                std::thread::Builder::new()
                    .stack_size(STACK_MEMORY)
                    .spawn(move || run_worker(&incoming, &finished, &stats))?,
            );
        }
        Ok(Self {
            jobs: Some(jobs),
            results,
            handles,
            ready: BTreeMap::new(),
            next: 0,
            submitted: 0,
            admission,
            host,
            job_memory: segment::memory_required(descriptor, direction)?,
            parallel_trials_fit: direction == Direction::Encode
                && segment::parallel_trials_fit(descriptor),
            _infrastructure: infrastructure,
        })
    }

    pub fn capacity(&self) -> usize {
        self.admission.workers
    }

    /// Two trials only run concurrently when the normal worker reservation
    /// covers both full predictor states. Otherwise the same compression
    /// candidates run sequentially with no extra memory requirement.
    pub(crate) fn trials(&self) -> Trials {
        if self.capacity() > 1 && self.parallel_trials_fit {
            Trials::Parallel
        } else {
            Trials::Sequential
        }
    }

    pub fn outstanding(&self) -> usize {
        (self.submitted - self.next) as usize
    }

    pub fn prepare_job(&self) -> Result<MemoryLease, BatoError> {
        if self.outstanding() >= self.capacity() {
            return Err(BatoError::CorruptStream("worker capacity exceeded".into()));
        }
        self.host.reserve(self.job_memory)
    }

    pub fn submit(&mut self, job: Work, lease: MemoryLease) -> Result<(), BatoError> {
        if self.outstanding() >= self.capacity() {
            return Err(BatoError::CorruptStream("worker capacity exceeded".into()));
        }
        self.jobs
            .as_ref()
            .ok_or_else(|| BatoError::CorruptStream("worker stopped".into()))?
            .send((self.submitted, job, lease))
            .map_err(|_| BatoError::CorruptStream("worker stopped".into()))?;
        self.submitted += 1;
        Ok(())
    }

    pub fn receive(&mut self) -> Result<Completed, BatoError> {
        loop {
            if let Some(result) = self.ready.remove(&self.next) {
                self.next += 1;
                return result;
            }
            let (id, result) = self
                .results
                .recv()
                .map_err(|_| BatoError::CorruptStream("worker stopped".into()))?;
            if id < self.next || id >= self.submitted || self.ready.insert(id, result).is_some() {
                return Err(BatoError::CorruptStream(
                    "duplicate or stale worker result".into(),
                ));
            }
        }
    }
}

fn run_worker(
    incoming: &Mutex<mpsc::Receiver<Job>>,
    finished: &mpsc::Sender<JobResult>,
    stats: &ExecutionStats,
) {
    loop {
        let Ok((id, work, lease)) = incoming.lock().unwrap().recv() else {
            return;
        };
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(work, stats)))
                .unwrap_or_else(|_| Err(BatoError::CorruptStream("worker panicked".into())))
                .map(|output| Completed {
                    output,
                    memory: Some(lease),
                });
        if finished.send((id, result)).is_err() {
            return;
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.jobs.take();
        for handle in self.handles.drain(..) {
            // A panicking worker already reported its error through `results`.
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> Descriptor {
        Descriptor {
            segment_log2: 16,
            chunk_log2: 14,
            table_log2: 16,
        }
    }

    #[test]
    fn parallel_trials_respect_the_existing_worker_budget() {
        // Small geometries keep their original admission floor and execute
        // the same candidate search sequentially if two models do not fit.
        assert!(!segment::parallel_trials_fit(descriptor()));
        assert!(segment::parallel_trials_fit(Descriptor::default()));
    }

    #[test]
    fn admission_scales_workers_with_memory_and_rejects_too_little() {
        let options = ExecutionOptions {
            threads: 8,
            ..Default::default()
        };
        let per = segment::memory_required(descriptor(), Direction::Encode).unwrap() + STACK_MEMORY;
        let three = admit(
            (descriptor(), Direction::Encode),
            HOST_RESERVE + 3 * per,
            &options,
        )
        .unwrap();
        assert_eq!(three.workers, 3);
        assert_eq!(three.host_memory, HOST_RESERVE + 3 * per);
        assert!(matches!(
            admit(
                (descriptor(), Direction::Encode),
                HOST_RESERVE + per - 1,
                &options
            ),
            Err(BatoError::MemoryLimitExceeded { .. })
        ));
        let capped = admit((descriptor(), Direction::Encode), usize::MAX, &options).unwrap();
        assert_eq!(capped.workers, 8);
        let default_worker =
            segment::memory_required(Descriptor::default(), Direction::Encode).unwrap();
        assert!(default_worker <= 1536 * 1024 * 1024, "{default_worker}");
    }

    #[test]
    fn pool_returns_results_in_submission_order_and_enforces_capacity() {
        let options = ExecutionOptions {
            threads: 3,
            ..Default::default()
        };
        let host = HostBudget::new(MemoryBudget::Bytes(usize::MAX / 2), &options.stats).unwrap();
        let mut pool = Pool::new((descriptor(), Direction::Encode), host, &options).unwrap();
        let inputs: Vec<Vec<u8>> = (0..3u8)
            .map(|i| vec![i; 1000 + usize::from(i) * 2000])
            .collect();
        for raw in &inputs {
            let lease = pool.prepare_job().unwrap();
            pool.submit(
                Work::Encode {
                    raw: raw.clone(),
                    descriptor: descriptor(),
                    trials: Trials::Sequential,
                },
                lease,
            )
            .unwrap();
        }
        assert!(pool.prepare_job().is_err());
        for raw in &inputs {
            let Output::Encoded(chunks, original) = pool.receive().unwrap().output else {
                panic!("expected encoded output");
            };
            let total: u32 = chunks.iter().map(|c| c.header.uncompressed_size).sum();
            assert_eq!(total as usize, raw.len());
            assert_eq!(original.length as usize, raw.len());
        }
        assert_eq!(pool.outstanding(), 0);
    }
}

//! Optional step timing for performance work. Provenance: original design.
//!
//! Timing only observes: no coding decision reads a clock, so archives are
//! identical with and without it. Whole-segment steps are timed exactly. The
//! per-bit model steps are timed by lapping one byte in `SAMPLE_PERIOD` and
//! scaling by the bytes seen, with the cost of the clock calls subtracted.

use std::{
    fmt,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

/// One in this many modelled bytes is lapped step by step. A prime, so that
/// periodic work (scans every 32 KiB, chunk boundaries) is not always sampled or
/// never sampled, which would scale its cost by the period or to zero.
const SAMPLE_PERIOD: u64 = 61;
/// Clock calls used to estimate the cost of one lap.
const CALIBRATION_CALLS: u32 = 4096;

/// A named step or group of steps whose time is reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Step {
    TransformSelection,
    FastCandidates,
    DeflateTransform,
    PrimerTraining,
    ModelTrial,
    ModelDecode,
    InverseTransform,
    Checksum,
    Coder,
    ByteContexts,
    ByteMatch,
    ByteLag,
    ByteRows,
    ByteRecords,
    CalibrationSetup,
    Prefetch,
    ContextMaps,
    ExtraInputs,
    FirstMixers,
    Experts,
    FinalMixers,
    Isse,
    Calibration,
    TrainCalibration,
    TrainMixers,
    TrainExperts,
    TrainMaps,
    ByteEnd,
}

const STEPS: usize = Step::ByteEnd as usize + 1;
const FIRST_MODEL_STEP: usize = Step::Coder as usize;

impl Step {
    const ALL: [Step; STEPS] = [
        Step::TransformSelection,
        Step::FastCandidates,
        Step::DeflateTransform,
        Step::PrimerTraining,
        Step::ModelTrial,
        Step::ModelDecode,
        Step::InverseTransform,
        Step::Checksum,
        Step::Coder,
        Step::ByteContexts,
        Step::ByteMatch,
        Step::ByteLag,
        Step::ByteRows,
        Step::ByteRecords,
        Step::CalibrationSetup,
        Step::Prefetch,
        Step::ContextMaps,
        Step::ExtraInputs,
        Step::FirstMixers,
        Step::Experts,
        Step::FinalMixers,
        Step::Isse,
        Step::Calibration,
        Step::TrainCalibration,
        Step::TrainMixers,
        Step::TrainExperts,
        Step::TrainMaps,
        Step::ByteEnd,
    ];

    fn name(self) -> &'static str {
        match self {
            Step::TransformSelection => "x86 transform selection",
            Step::FastCandidates => "fast codec candidates",
            Step::DeflateTransform => "deflate scan and transform",
            Step::PrimerTraining => "primer training",
            Step::ModelTrial => "model encoding trials (cpu)",
            Step::ModelDecode => "model decoding",
            Step::InverseTransform => "inverse transforms",
            Step::Checksum => "checksums",
            Step::Coder => "arithmetic coder and loop",
            Step::ByteContexts => "byte start: order, word, sparse hashes",
            Step::ByteMatch => "byte start: match model",
            Step::ByteLag => "byte start: lag expert",
            Step::ByteRows => "byte start: row expert",
            Step::ByteRecords => "byte start: record and neighbour models",
            Step::CalibrationSetup => "calibration contexts and prefetch",
            Step::Prefetch => "context table prefetch",
            Step::ContextMaps => "main context maps",
            Step::ExtraInputs => "order-0, run, match, record inputs",
            Step::FirstMixers => "first-layer mixers",
            Step::Experts => "expert maps and mixers",
            Step::FinalMixers => "final mixers",
            Step::Isse => "ISSE chain",
            Step::Calibration => "calibration (APM) stages",
            Step::TrainCalibration => "train: APM and ISSE",
            Step::TrainMixers => "train: first and final mixers",
            Step::TrainExperts => "train: experts",
            Step::TrainMaps => "train: maps, order 0, match",
            Step::ByteEnd => "byte-end model updates",
        }
    }

    fn is_model_step(self) -> bool {
        self as usize >= FIRST_MODEL_STEP
    }
}

/// Counts of bits, sampled bytes, in which an optional model took part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Activity {
    Bits,
    MatchPredicting,
    Records,
    TextExpert,
    SparseExpert,
    LagExpert,
    RowExpert,
}

const ACTIVITIES: usize = Activity::RowExpert as usize + 1;

impl Activity {
    const ALL: [Activity; ACTIVITIES] = [
        Activity::Bits,
        Activity::MatchPredicting,
        Activity::Records,
        Activity::TextExpert,
        Activity::SparseExpert,
        Activity::LagExpert,
        Activity::RowExpert,
    ];

    fn name(self) -> &'static str {
        match self {
            Activity::Bits => "sampled bits",
            Activity::MatchPredicting => "match model predicting",
            Activity::Records => "record model active",
            Activity::TextExpert => "text expert active",
            Activity::SparseExpert => "sparse expert active",
            Activity::LagExpert => "lag expert active",
            Activity::RowExpert => "row expert active",
        }
    }
}

/// What the bytes of the finished archive pay for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum Part {
    StreamHeader,
    BlockHeaders,
    BlockChecksums,
    Footer,
    Coded,
    Stored,
}

const PARTS: usize = Part::Stored as usize + 1;

impl Part {
    const ALL: [Part; PARTS] = [
        Part::StreamHeader,
        Part::BlockHeaders,
        Part::BlockChecksums,
        Part::Footer,
        Part::Coded,
        Part::Stored,
    ];

    fn name(self) -> &'static str {
        match self {
            Part::StreamHeader => "stream header",
            Part::BlockHeaders => "block headers",
            Part::BlockChecksums => "block checksums (4 bytes each)",
            Part::Footer => "footer (length and checksum)",
            Part::Coded => "model-coded payload",
            Part::Stored => "stored payload (not modelled)",
        }
    }
}

/// Per-thread accumulator the predictor laps while a sampled byte is coded.
pub(super) struct Meter {
    last: Instant,
    sampled: bool,
    bytes: u64,
    sampled_bytes: u64,
    nanos: [u64; STEPS],
    laps: [u64; STEPS],
    activity: [u64; ACTIVITIES],
}

impl Meter {
    pub(super) fn new() -> Self {
        Self {
            last: Instant::now(),
            sampled: true,
            bytes: 0,
            sampled_bytes: 0,
            nanos: [0; STEPS],
            laps: [0; STEPS],
            activity: [0; ACTIVITIES],
        }
    }

    /// Attributes the time since the previous lap to `step`.
    #[inline]
    pub(super) fn lap(&mut self, step: Step) {
        if !self.sampled {
            return;
        }
        let now = Instant::now();
        self.nanos[step as usize] += nanos(now - self.last);
        self.laps[step as usize] += 1;
        self.last = now;
    }

    /// Whether the current byte is being lapped.
    #[inline]
    pub(super) fn sampled(&self) -> bool {
        self.sampled
    }

    /// Counts one sampled bit in which `activity` was (`yes`) or was not on.
    #[inline]
    pub(super) fn count(&mut self, activity: Activity, yes: bool) {
        if self.sampled {
            self.activity[activity as usize] += u64::from(yes);
        }
    }

    /// Called after each modelled byte: chooses whether the next one is lapped.
    pub(super) fn end_byte(&mut self) {
        self.bytes += 1;
        self.sampled = self.bytes.is_multiple_of(SAMPLE_PERIOD);
        if self.sampled {
            self.sampled_bytes += 1;
            self.last = Instant::now();
        }
    }
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Mean cost in nanoseconds of the clock call that one lap performs.
fn lap_cost() -> u64 {
    let start = Instant::now();
    for _ in 0..CALIBRATION_CALLS {
        std::hint::black_box(Instant::now());
    }
    nanos(start.elapsed()) / u64::from(CALIBRATION_CALLS)
}

#[derive(Debug, Default)]
struct Totals {
    parts: [u64; PARTS],
    nanos: [u64; STEPS],
    modelled_bytes: u64,
    activity: [u64; ACTIVITIES],
}

/// Collects the timing of every segment of one operation.
#[derive(Debug, Default)]
pub struct TimingSink {
    enabled: AtomicBool,
    totals: Mutex<Totals>,
}

impl TimingSink {
    pub(crate) fn enable(&self) {
        self.enabled.store(true, Ordering::Relaxed);
    }

    pub(crate) fn handle(&self) -> Timing<'_> {
        Timing(self.enabled.load(Ordering::Relaxed).then_some(self))
    }

    fn add_part(&self, part: Part, bytes: u64) {
        if let Ok(mut totals) = self.totals.lock() {
            totals.parts[part as usize] += bytes;
        }
    }

    fn add(&self, step: Step, elapsed: Duration) {
        if let Ok(mut totals) = self.totals.lock() {
            totals.nanos[step as usize] += nanos(elapsed);
        }
    }

    /// Adds a finished meter, scaling its sampled steps to all its bytes.
    fn merge(&self, meter: &Meter) {
        let cost = lap_cost();
        let Ok(mut totals) = self.totals.lock() else {
            return;
        };
        let sampled = meter.sampled_bytes.max(1);
        for step in Step::ALL.into_iter().filter(|step| step.is_model_step()) {
            let i = step as usize;
            let net = meter.nanos[i].saturating_sub(meter.laps[i] * cost);
            let scaled = u128::from(net) * u128::from(meter.bytes) / u128::from(sampled);
            totals.nanos[i] += u64::try_from(scaled).unwrap_or(u64::MAX);
        }
        for (total, seen) in totals.activity.iter_mut().zip(meter.activity) {
            *total += seen;
        }
        totals.modelled_bytes += meter.bytes;
    }

    /// The report so far, or `None` when timing was never enabled.
    pub fn report(&self) -> Option<TimingReport> {
        if !self.enabled.load(Ordering::Relaxed) {
            return None;
        }
        let totals = self.totals.lock().ok()?;
        Some(TimingReport {
            parts: totals.parts,
            nanos: totals.nanos,
            modelled_bytes: totals.modelled_bytes,
            activity: totals.activity,
        })
    }
}

/// A cheap handle that times whole steps when timing is enabled.
#[derive(Clone, Copy)]
pub(crate) struct Timing<'a>(Option<&'a TimingSink>);

impl Timing<'_> {
    #[cfg(test)]
    pub(crate) fn off() -> Self {
        Self(None)
    }

    pub(crate) fn enabled(self) -> bool {
        self.0.is_some()
    }

    pub(crate) fn time<T>(self, step: Step, work: impl FnOnce() -> T) -> T {
        let Some(sink) = self.0 else {
            return work();
        };
        let start = Instant::now();
        let value = work();
        sink.add(step, start.elapsed());
        value
    }

    /// Starts a step that spans several statements; see `stop`.
    pub(crate) fn start(self) -> Option<Instant> {
        self.0.map(|_| Instant::now())
    }

    pub(crate) fn stop(self, step: Step, started: Option<Instant>) {
        if let (Some(sink), Some(started)) = (self.0, started) {
            sink.add(step, started.elapsed());
        }
    }

    /// Counts `bytes` of the archive as paying for `part`.
    pub(crate) fn part(self, part: Part, bytes: usize) {
        if let Some(sink) = self.0 {
            sink.add_part(part, bytes as u64);
        }
    }

    pub(super) fn merge(self, meter: &Meter) {
        if let Some(sink) = self.0 {
            sink.merge(meter);
        }
    }
}

/// Times and activity of one operation, for printing.
#[derive(Clone, Debug)]
pub struct TimingReport {
    parts: [u64; PARTS],
    nanos: [u64; STEPS],
    modelled_bytes: u64,
    activity: [u64; ACTIVITIES],
}

impl TimingReport {
    fn write_rows(
        &self,
        f: &mut fmt::Formatter<'_>,
        steps: impl Iterator<Item = Step>,
        total: u64,
    ) -> fmt::Result {
        let per_byte = |n: u64| n.checked_div(self.modelled_bytes).unwrap_or(0);
        for step in steps {
            let n = self.nanos[step as usize];
            if n == 0 {
                continue;
            }
            writeln!(
                f,
                "{:<40} {:>10} {:>6}% {:>9}",
                step.name(),
                n / 1_000_000,
                (u128::from(n) * 100 / u128::from(total.max(1))) as u64,
                per_byte(n)
            )?;
        }
        Ok(())
    }
}

impl fmt::Display for TimingReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let whole = |step: &Step| !step.is_model_step();
        let total: u64 = Step::ALL
            .iter()
            .filter(|step| whole(step))
            .map(|&step| self.nanos[step as usize])
            .sum();
        let archive: u64 = self.parts.iter().sum();
        if archive > 0 {
            writeln!(f, "archive bytes by purpose, {archive} in all")?;
            for part in Part::ALL {
                let n = self.parts[part as usize];
                if n > 0 {
                    writeln!(
                        f,
                        "  {:<38} {:>10} {:>6}%",
                        part.name(),
                        n,
                        (u128::from(n) * 100 / u128::from(archive)) as u64
                    )?;
                }
            }
        }
        writeln!(f, "step timing, cpu time summed over all threads")?;
        writeln!(
            f,
            "{:<40} {:>10} {:>7} {:>9}",
            "step", "ms", "share", "ns/byte"
        )?;
        self.write_rows(f, Step::ALL.into_iter().filter(whole), total)?;
        writeln!(f, "{:<40} {:>10}", "total", total / 1_000_000)?;
        let modelled: u64 = Step::ALL
            .iter()
            .filter(|step| step.is_model_step())
            .map(|&step| self.nanos[step as usize])
            .sum();
        if modelled > 0 {
            writeln!(
                f,
                "model breakdown, sampled 1 byte in {SAMPLE_PERIOD} and scaled, {} bytes",
                self.modelled_bytes
            )?;
            self.write_rows(
                f,
                Step::ALL.into_iter().filter(|s| s.is_model_step()),
                modelled,
            )?;
            writeln!(f, "{:<40} {:>10}", "sampled estimate", modelled / 1_000_000)?;
        }
        let bits = self.activity[Activity::Bits as usize];
        if bits > 0 {
            writeln!(f, "activity of optional models, share of sampled bits")?;
            for activity in Activity::ALL.into_iter().skip(1) {
                let seen = self.activity[activity as usize];
                writeln!(
                    f,
                    "  {:<38} {:>6}%",
                    activity.name(),
                    (u128::from(seen) * 100 / u128::from(bits)) as u64
                )?;
            }
        }
        Ok(())
    }
}

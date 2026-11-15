//! Runs the predictor's three stages on separate threads when every bit is
//! known in advance, as it is for an encoder. Provenance: original; a
//! three-stage software pipeline over batches of bits, connected by bounded
//! `std::sync::mpsc` channels. Each stage sees the same bits in the same
//! order as in the serial path, and no stage reads another's later output,
//! so every probability equals the serial one and so does the coded output.

use std::sync::mpsc::{sync_channel, Receiver, SyncSender};

use super::{Calibration, Front, Mixed, Mixing, Predictor, Signals};
use crate::core::{cm::simd::Dispatch, memory::try_filled, types::BatoError};

/// Bits per batch: large enough to amortise a channel hand-off, small enough
/// to stay in the second-level cache.
const BATCH_BITS: usize = 256;
/// Batches circulating on each link; more lets a fast stage run further ahead.
const BATCHES: usize = 3;
/// Bits ahead of the one being calibrated whose APM rows are prefetched.
const PREFETCH_AHEAD: usize = 4;
/// Stack of each stage thread, in bytes; the stages keep their state on the heap.
const STAGE_STACK: usize = 1 << 20;

/// Whether an encoding run spreads the model over threads. The coded bytes
/// are identical either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::core::cm) enum Stages {
    Serial,
    Pipelined,
}

/// Heap and stacks a pipelined run adds to its model's own memory, in bytes.
pub(in crate::core::cm) fn memory_required() -> usize {
    BATCHES * BATCH_BITS * (std::mem::size_of::<Signals>() + std::mem::size_of::<Mixed>())
        + 2 * STAGE_STACK
}

struct SignalBatch {
    items: Vec<Signals>,
    len: usize,
}

struct MixedBatch {
    items: Vec<Mixed>,
    len: usize,
}

fn stopped() -> BatoError {
    BatoError::CorruptStream("model pipeline stopped".into())
}

/// The calibration stage fed by the threads before it, driven by the caller.
pub(in crate::core::cm) struct Staged {
    calibration: Calibration,
    full: Receiver<MixedBatch>,
    empty: SyncSender<MixedBatch>,
    current: Option<MixedBatch>,
    at: usize,
}

impl Staged {
    fn probability(&mut self) -> Result<u16, BatoError> {
        if self
            .current
            .as_ref()
            .is_none_or(|batch| self.at == batch.len)
        {
            if let Some(done) = self.current.take() {
                // The mixing thread has already ended after its last batch.
                let _ = self.empty.send(done);
            }
            self.current = Some(self.full.recv().map_err(|_| stopped())?);
            self.at = 0;
        }
        let batch = self.current.as_ref().ok_or_else(stopped)?;
        if let Some(ahead) = batch.items[..batch.len].get(self.at + PREFETCH_AHEAD) {
            self.calibration.prefetch(&ahead.calibration, ahead.half);
        }
        Ok(self.calibration.predict(&batch.items[self.at], &mut None))
    }

    fn observe(&mut self, bit: u8) {
        self.calibration.observe(bit, &mut None);
        self.at += 1;
    }
}

/// What a coder drives: the whole model on its thread, or the last stage of
/// a pipeline.
pub(in crate::core::cm) enum Engine<'a> {
    Serial(&'a mut Predictor),
    Staged(&'a mut Staged),
}

impl Engine<'_> {
    /// Probability of the next bit; `history` is every byte before it, which a
    /// pipeline's stages already hold.
    pub(in crate::core::cm) fn probability(&mut self, history: &[u8]) -> Result<u16, BatoError> {
        match self {
            Engine::Serial(model) => Ok(model.probability(history)),
            Engine::Staged(staged) => staged.probability(),
        }
    }

    pub(in crate::core::cm) fn observe(&mut self, bit: u8) {
        match self {
            Engine::Serial(model) => model.observe(bit),
            Engine::Staged(staged) => staged.observe(bit),
        }
    }
}

/// The bit at `index` of the bytes of `history` from `start` on, most
/// significant first.
#[inline(always)]
fn bit_at(history: &[u8], start: usize, index: usize) -> u8 {
    (history[start + index / 8] >> (7 - index % 8)) & 1
}

struct FrontJob<'a> {
    front: Front,
    history: &'a [u8],
    start: usize,
    full: SyncSender<SignalBatch>,
    empty: Receiver<SignalBatch>,
}

#[inline(always)]
fn front_loop(job: FrontJob<'_>) {
    let FrontJob {
        mut front,
        history,
        start,
        full,
        empty,
    } = job;
    let bits = (history.len() - start) * 8;
    let mut done = 0;
    while done < bits {
        let Ok(mut batch) = empty.recv() else {
            return;
        };
        batch.len = BATCH_BITS.min(bits - done);
        for (offset, signals) in batch.items[..batch.len].iter_mut().enumerate() {
            let index = done + offset;
            let bit = bit_at(history, start, index);
            front.predict_main(&history[..start + index / 8], signals, &mut None);
            front.predict_experts(signals, &mut None);
            front.prefetch_after(bit);
            front.observe(bit, &mut None);
        }
        done += batch.len;
        if full.send(batch).is_err() {
            return;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn front_avx2(job: FrontJob<'_>) {
    front_loop(job);
}

struct MixingJob<'a> {
    mixing: Mixing,
    history: &'a [u8],
    start: usize,
    from_front: Receiver<SignalBatch>,
    to_front: SyncSender<SignalBatch>,
    full: SyncSender<MixedBatch>,
    empty: Receiver<MixedBatch>,
}

#[inline(always)]
fn mixing_loop(job: MixingJob<'_>) {
    let MixingJob {
        mut mixing,
        history,
        start,
        from_front,
        to_front,
        full,
        empty,
    } = job;
    let mut done = 0;
    while let Ok(batch) = from_front.recv() {
        let Ok(mut out) = empty.recv() else {
            return;
        };
        out.len = batch.len;
        for (offset, (signals, mixed)) in batch.items[..batch.len]
            .iter()
            .zip(&mut out.items)
            .enumerate()
        {
            mixing.predict_first(signals, &mut None);
            *mixed = mixing.predict_rest(signals, &mut None);
            mixing.observe(signals, bit_at(history, start, done + offset), &mut None);
        }
        done += batch.len;
        if full.send(out).is_err() {
            return;
        }
        // The front thread has ended once it has produced its last batch.
        let _ = to_front.send(batch);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mixing_avx2(job: MixingJob<'_>) {
    mixing_loop(job);
}

fn run_front(job: FrontJob<'_>, dispatch: Dispatch) {
    #[cfg(target_arch = "x86_64")]
    if dispatch.is_resolved_avx2() {
        // SAFETY: `dispatch` was resolved via `is_x86_feature_detected!("avx2")`.
        unsafe { front_avx2(job) };
        return;
    }
    let _ = dispatch; // Only x86-64 selects a vector path.
    front_loop(job);
}

fn run_mixing(job: MixingJob<'_>, dispatch: Dispatch) {
    #[cfg(target_arch = "x86_64")]
    if dispatch.is_resolved_avx2() {
        // SAFETY: `dispatch` was resolved via `is_x86_feature_detected!("avx2")`.
        unsafe { mixing_avx2(job) };
        return;
    }
    let _ = dispatch; // Only x86-64 selects a vector path.
    mixing_loop(job);
}

/// Runs `coder` over the stages of `model` on three threads. `history` holds
/// the bytes the model has already seen followed by the `history.len() - start`
/// bytes the coder will code, whose bits the stages read ahead of it. The
/// coder may stop early; the stages then end on their own.
pub(in crate::core::cm) fn run_staged<R>(
    model: Predictor,
    (history, start): (&[u8], usize),
    coder: impl FnOnce(&mut Engine<'_>) -> R,
) -> Result<R, BatoError> {
    let Predictor {
        front,
        mixing,
        calibration,
        dispatch,
        ..
    } = model;
    let (signals_full, from_front) = sync_channel::<SignalBatch>(BATCHES);
    let (to_front, signals_empty) = sync_channel::<SignalBatch>(BATCHES);
    let (mixed_full, from_mixing) = sync_channel::<MixedBatch>(BATCHES);
    let (to_mixing, mixed_empty) = sync_channel::<MixedBatch>(BATCHES);
    for _ in 0..BATCHES {
        // Each channel holds BATCHES batches, so these sends cannot block.
        to_front
            .send(SignalBatch {
                items: try_filled(BATCH_BITS, Signals::new())?,
                len: 0,
            })
            .map_err(|_| stopped())?;
        to_mixing
            .send(MixedBatch {
                items: try_filled(BATCH_BITS, Mixed::new())?,
                len: 0,
            })
            .map_err(|_| stopped())?;
    }
    std::thread::scope(|scope| {
        let stage = || std::thread::Builder::new().stack_size(STAGE_STACK);
        let front_job = FrontJob {
            front,
            history,
            start,
            full: signals_full,
            empty: signals_empty,
        };
        stage().spawn_scoped(scope, move || run_front(front_job, dispatch))?;
        let mixing_job = MixingJob {
            mixing,
            history,
            start,
            from_front,
            to_front,
            full: mixed_full,
            empty: mixed_empty,
        };
        stage().spawn_scoped(scope, move || run_mixing(mixing_job, dispatch))?;
        let mut staged = Staged {
            calibration,
            full: from_mixing,
            empty: to_mixing,
            current: None,
            at: 0,
        };
        let result = coder(&mut Engine::Staged(&mut staged));
        // Closing the links ends stages the coder stopped waiting for.
        drop(staged);
        Ok(result)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Vec<u8>, usize) {
        let bytes = include_bytes!("../primer.bin")[..30_000].to_vec();
        (bytes, 1_500)
    }

    fn trained() -> (Predictor, Vec<u8>, usize) {
        let (history, start) = fixture();
        let mut model = Predictor::new(history.len(), 18, Dispatch::Native).unwrap();
        for (at, &byte) in history[..start].iter().enumerate() {
            model.observe_byte(byte, &history[..at]);
        }
        (model, history, start)
    }

    #[test]
    fn staged_probabilities_equal_serial_ones() {
        let (model, history, start) = trained();
        let mut serial = model.clone();
        let mut expected = Vec::new();
        for (at, &byte) in history.iter().enumerate().skip(start) {
            for shift in (0..8).rev() {
                expected.push(serial.probability(&history[..at]));
                serial.observe((byte >> shift) & 1);
            }
        }
        let staged = run_staged(model, (&history, start), |engine| {
            let mut seen = Vec::new();
            for (at, &byte) in history.iter().enumerate().skip(start) {
                for shift in (0..8).rev() {
                    seen.push(engine.probability(&history[..at]).unwrap());
                    engine.observe((byte >> shift) & 1);
                }
            }
            seen
        })
        .unwrap();
        assert_eq!(staged.len(), expected.len());
        assert!(staged == expected, "pipelined probabilities differ");
    }

    #[test]
    fn a_coder_that_stops_early_ends_the_stage_threads() {
        let (model, history, start) = trained();
        let stopped = run_staged(model, (&history, start), |engine| {
            let mut taken = 0;
            for (at, &byte) in history.iter().enumerate().skip(start).take(40) {
                for shift in (0..8).rev() {
                    engine.probability(&history[..at]).unwrap();
                    engine.observe((byte >> shift) & 1);
                    taken += 1;
                }
            }
            taken
        })
        .unwrap();
        assert_eq!(stopped, 320);
    }
}

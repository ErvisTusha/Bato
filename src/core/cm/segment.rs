//! Original segment-persistent coding with independently flushed chunks.
//! Provenance: Bato's supplied chunk/segment design and the original sibling
//! coder and predictor; no third-party implementation is incorporated.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::{
    coder::{Decoder, Encoder},
    deflate::container,
    descriptor::Descriptor,
    fast, filters, fragments, lz,
    predictor::{self, pipeline, Engine, Predictor, Stages},
    profile::{Step, Timing},
    raw_noise,
    simd::Dispatch,
    spatial, varint, x86,
};
use crate::core::{
    checksum::crc32c,
    gpu::filters::WorkerFilters,
    memory::{try_capacity, try_copy},
    types::{BatoError, BlockType, Primer, TransformId, DEFLATE_EXPANSION},
};

/// Built-in text and code every segment's model reads, without coding it,
/// before the first byte, so small inputs start from useful statistics.
pub(super) const PRIMER: &[u8] = include_bytes!("primer.bin");

/// The primer bytes a segment's model reads, if it reads them.
fn primer_bytes(primer: Primer) -> &'static [u8] {
    match primer {
        Primer::Read => PRIMER,
        Primer::Skipped => &[],
    }
}

/// Trains a fresh model on the segment's primer, exactly as coding would.
fn prime(model: &mut Predictor, primer: &[u8]) {
    for (at, &byte) in primer.iter().enumerate() {
        model.observe_byte(byte, &primer[..at]);
    }
}

/// Largest segment for which the encoder also tries coding without the
/// primer, which helps data unlike the primer's text and code.
const UNPRIMED_TRIAL_BYTES: usize = 1 << 24;

/// How a CM segment was prepared before and during coding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Coding {
    pub(crate) transform: TransformId,
    pub(crate) primer: Primer,
}

/// Settings of one model run over a segment.
#[derive(Clone, Copy)]
struct Run<'a> {
    primer: Primer,
    dispatch: Dispatch,
    budget: Option<usize>,
    incumbent: Option<&'a AtomicUsize>,
    /// Largest input the run accepts: the segment size, or for deflate-
    /// transformed bytes `DEFLATE_EXPANSION` times that.
    max_bytes: usize,
    timing: Timing<'a>,
    stages: Stages,
}

/// Fixed per-worker allowance for small bookkeeping allocations, in bytes.
const WORKER_SLACK: usize = 4 * 1024 * 1024;

#[derive(Clone, Debug)]
pub(crate) struct Chunk {
    pub(crate) kind: BlockType,
    pub(crate) length: u32,
    pub(crate) payload: Vec<u8>,
    pub(crate) crc: u32,
}

/// Largest input whose primed model is kept for the segment's other
/// candidates of the same length: above it the primer is a negligible share
/// of the coding time and a second model a large share of the memory.
const SNAPSHOT_BYTES: usize = 1 << 21;

/// A model trained on the primer, kept so that further candidates of the
/// same length copy it instead of training again. Copying gives the state
/// training would, so the archive does not depend on it.
struct Snapshot {
    model: Option<std::sync::Arc<(usize, Predictor)>>,
    wanted: bool,
}

impl Snapshot {
    fn new(wanted: bool) -> Self {
        Self {
            model: None,
            wanted,
        }
    }
    fn shared(model: Option<std::sync::Arc<(usize, Predictor)>>) -> Self {
        Self {
            model,
            wanted: false,
        }
    }
}

/// Whether the encoder may run its independent model trials on several
/// threads. The archive is the same either way; only the time differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Trials {
    Sequential,
    /// `reservations` is how many worker memory reservations the segment may
    /// spread its concurrent models over: its own, plus those of workers that
    /// will never receive a job because this is the stream's last segment.
    Parallel {
        reservations: usize,
    },
}

impl Trials {
    fn is_parallel(self) -> bool {
        matches!(self, Trials::Parallel { .. })
    }

    /// Worker reservations available to this segment's concurrent models.
    fn reservations(self) -> usize {
        match self {
            Trials::Sequential => 1,
            Trials::Parallel { reservations } => reservations,
        }
    }

    /// What one of `ways` concurrent callers may use of these reservations.
    fn shared(self, ways: usize) -> Trials {
        match self {
            Trials::Sequential => Trials::Sequential,
            Trials::Parallel { reservations } => Trials::Parallel {
                reservations: (reservations / ways).max(1),
            },
        }
    }

    /// Whether each of `models` concurrent model runs may spread over the
    /// threads of the stages it has reservations for.
    fn stages(self, models: usize) -> Stages {
        if self.is_parallel() && self.reservations() / models >= STAGE_THREADS {
            Stages::Pipelined
        } else {
            Stages::Serial
        }
    }
}

/// Threads of a pipelined model run: the data-driven stage, the mixing stage
/// and the calibrating coder.
const STAGE_THREADS: usize = 3;

/// Which side of the codec a worker serves; decoding never builds the
/// candidate packets that dominate the encoder's independent phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Direction {
    Encode,
    Decode,
}

/// Segment-sized buffers the encoder's independent phase may hold at once:
/// digit groups, candidate packets, skeleton, input and output, in units of
/// the segment length including geometric `Vec` capacity growth.
const ENCODE_INDEPENDENT_SEGMENTS: usize = 18;

/// Segment-sized buffers `fast::decode` plus the inverse transform may hold
/// at once, in units of the segment length `L`. Derivation for the numeric
/// path, the largest: the skeleton is at most `2L`; the owned low, high and
/// zero streams total at most `L/2 + (2L + 2) + L/2 = 3L + 2`; one more
/// stream's token buffer plus its output is transient, at most `2 * 2L`
/// (high stream, `2L + 2` bytes each); the output is `L`; the received
/// payload is at most `L`. That sums to `2 + 3 + 4 + 1 + 1 = 11`; one more
/// `L` covers the inverse filter copy. A test measures the real peak.
const DECODE_FAST_SEGMENTS: usize = 12;

pub(crate) fn memory_required(
    descriptor: Descriptor,
    direction: Direction,
) -> Result<usize, BatoError> {
    descriptor.validate()?;
    let segment = descriptor.segment_bytes();
    let chunk = descriptor.chunk_bytes();
    let independent_segments = match direction {
        Direction::Encode => ENCODE_INDEPENDENT_SEGMENTS,
        Direction::Decode => DECODE_FAST_SEGMENTS,
    };
    let independent = segment.checked_mul(independent_segments).ok_or_else(|| {
        BatoError::InvalidOptions("segment working set exceeds address space".into())
    })?;
    // Buffers around the model, in segments. A deflate-transformed segment
    // holds up to DEFLATE_EXPANSION times the segment size: the encoder keeps
    // the raw segment, the inflated bytes and the model's copy of them; the
    // decoder the coded payloads and the decoded transformed bytes.
    let model_segments = match direction {
        Direction::Encode => 2 * DEFLATE_EXPANSION + 1,
        Direction::Decode => 2 * DEFLATE_EXPANSION,
    }
    .max(4);
    Ok(
        (predictor::memory_required(segment + PRIMER.len(), descriptor.table_log2)
            + model_segments * segment
            + PRIMER.len())
        // Independent candidate construction and the model's single-trial
        // phase do not overlap. Additional concurrent trials are admitted
        // only when this existing per-worker reservation also covers them.
        .max(independent)
            + 8 * 1024 * 1024
            + pipeline::memory_required()
            + (segment / chunk)
                * (std::mem::size_of::<Chunk>()
                    + std::mem::size_of::<crate::core::codec::EncodedChunk>())
            + 2 * chunk
            + x86::SELECTION_SCRATCH
            + WORKER_SLACK,
    )
}

/// Bytes needed to run primed and unprimed trials side by side, including
/// the largest modeled DEFLATE expansion for which both trials run.
pub(crate) fn parallel_trials_memory(descriptor: Descriptor) -> usize {
    let segment = descriptor.segment_bytes();
    let chunk = descriptor.chunk_bytes();
    let trial_input = segment
        .saturating_mul(DEFLATE_EXPANSION)
        .min(UNPRIMED_TRIAL_BYTES);
    let per_trial = predictor::memory_required(trial_input + PRIMER.len(), descriptor.table_log2)
        .saturating_add(2 * trial_input + PRIMER.len() + 2 * chunk + WORKER_SLACK);
    2usize.saturating_mul(per_trial).saturating_add(4 * segment)
}

/// Whether `models` predictors for inputs of `length` bytes fit in the
/// segment's `reservations` worker reservations at once, each with its
/// history and output.
fn concurrent_models_fit(
    descriptor: Descriptor,
    length: usize,
    (models, reservations): (usize, usize),
) -> bool {
    let Ok(reserved) = memory_required(descriptor, Direction::Encode) else {
        return false;
    };
    let per_model = predictor::memory_required(length + PRIMER.len(), descriptor.table_log2)
        .saturating_add(3 * length + PRIMER.len() + WORKER_SLACK + pipeline::memory_required());
    reserved.saturating_mul(reservations) >= models.saturating_mul(per_model)
}

/// Encodes one segment, first applying the reversible x86 address transform
/// in place when its byte statistics favor it. `raw` is restored before
/// returning, and chunk CRCs always cover the original bytes.
pub(crate) fn encode_segment(
    raw: &mut [u8],
    (descriptor, trials): (Descriptor, Trials),
    timing: Timing<'_>,
    mut accelerator: Option<&mut WorkerFilters<'_>>,
) -> Result<(Vec<Chunk>, Coding), BatoError> {
    descriptor.validate()?;
    if raw.len() > descriptor.segment_bytes() {
        return Err(BatoError::InvalidOptions("segment exceeds geometry".into()));
    }
    let plain = |transform| Coding {
        transform,
        primer: Primer::Read,
    };
    if raw.is_empty() {
        return Ok((Vec::new(), plain(TransformId::None)));
    }
    if let Some(period) = timing.time(Step::FastCandidates, || {
        raw_noise::periodic(raw, descriptor.chunk_bytes())
    }) {
        let chunk = Chunk {
            kind: BlockType::Lz,
            length: raw.len() as u32,
            payload: lz::encode_period(raw, period)?,
            crc: crc32c(raw),
        };
        return Ok((vec![chunk], plain(TransformId::None)));
    }
    let x86 = timing.time(Step::TransformSelection, || x86::select(raw))?;
    let independent = timing.time(Step::FastCandidates, || {
        fast::best(raw, x86.map(x86::Sites::transform), &mut accelerator)
    })?;
    let original_crc = crc32c(raw);
    let selected_filter = independent.transform;
    let coding = plain(selected_filter);
    let cost = independent.cost();
    let chunks = vec![Chunk {
        kind: independent.kind,
        length: raw.len() as u32,
        payload: independent.payload,
        crc: original_crc,
    }];
    // Repeat uses one payload byte and no payload-length field. Even an
    // empty coded payload needs that byte for its length. The deflate
    // transform only expands the input and can only add framing cost.
    if independent.kind == BlockType::Repeat {
        return Ok((chunks, coding));
    }
    // Keep complete framed costs throughout: a winning deflate transform is
    // not proof that the raw CM candidate (or the x86 candidate) is worse.
    let mut winner = (cost, chunks, coding);
    if let Some(tape) = timing.time(Step::FastCandidates, || fragments::commands(raw))? {
        let packed = x86::select(&tape)?
            .map(|sites| {
                encode_fragment_packet(
                    (&tape, sites.transform()),
                    (descriptor, trials),
                    winner.0,
                    timing,
                )
            })
            .transpose()?
            .flatten();
        for (kind, payload) in [(BlockType::LzHuffman, packed), (BlockType::Lz, Some(tape))] {
            let Some(payload) = payload else { continue };
            let cost = crate::core::frame::block_cost(kind, raw.len(), payload.len());
            if cost < winner.0 {
                winner = (
                    cost,
                    vec![Chunk {
                        kind,
                        length: raw.len() as u32,
                        payload,
                        crc: original_crc,
                    }],
                    plain(TransformId::None),
                );
            }
        }
    }
    if let Some(payload) = spatial::encode(raw, winner.0.min(raw.len() - 1))? {
        let chunk = Chunk {
            kind: BlockType::Huffman,
            length: raw.len() as u32,
            payload,
            crc: original_crc,
        };
        let spatial_cost = chunks_cost(std::slice::from_ref(&chunk));
        if spatial_cost < winner.0 {
            winner = (spatial_cost, vec![chunk], plain(TransformId::Spatial));
        }
    }
    let minimum = |len: usize| -> usize {
        (0..len)
            .step_by(descriptor.chunk_bytes())
            .map(|start| {
                crate::core::frame::block_cost(
                    BlockType::Raw,
                    (len - start).min(descriptor.chunk_bytes()),
                    0,
                )
            })
            .sum()
    };
    let inflated = timing.time(Step::DeflateTransform, || {
        container::forward(
            raw,
            descriptor.segment_bytes() * DEFLATE_EXPANSION,
            descriptor.chunk_bytes(),
        )
    });
    if let Some(mut inflated) = inflated {
        // Always compare transformed and original representations on their
        // final byte costs. Do not stop just because this candidate wins.
        if minimum(inflated.len()) < winner.0 {
            let sites = timing.time(Step::TransformSelection, || x86::select(&inflated))?;
            if let Some(sites) = sites {
                x86::forward(&mut inflated, sites);
            }
            if let Some((cost, primer, chunks)) = model_candidate(
                &inflated,
                (
                    descriptor,
                    trials,
                    descriptor.segment_bytes() * DEFLATE_EXPANSION,
                ),
                winner.0,
                (timing, &mut Snapshot::new(false)),
            )? {
                winner = (
                    cost,
                    chunks,
                    Coding {
                        transform: x86::Sites::container_transform(sites),
                        primer,
                    },
                );
            }
        }
    }
    if winner.1[0].kind == BlockType::Stored && raw_noise::uniform(raw) {
        return Ok((winner.1, winner.2));
    }
    if let Some(chunk) =
        encode_cold_tail(raw, (descriptor, trials), (winner.0, original_crc), timing)?
    {
        return Ok((vec![chunk], plain(TransformId::None)));
    }
    let min_frame = minimum(raw.len());
    // The x86, filter and plain candidates have the same length and need
    // nothing from each other, so on a worker whose reservation holds all
    // their models they run at once. Each is accepted only below the cost so
    // far and ties go to the earlier one, exactly as when run in turn.
    let filter = Some(selected_filter).filter(|filter| filters::CANDIDATES.contains(filter));
    let transforms: Vec<TransformId> = x86
        .map(x86::Sites::transform)
        .into_iter()
        .chain(filter)
        .chain([TransformId::None])
        .collect();
    let reservations = trials.reservations();
    if trials.is_parallel()
        && transforms.len() > 1
        && min_frame < winner.0
        && concurrent_models_fit(descriptor, raw.len(), (2 * transforms.len(), reservations))
    {
        let inputs = transforms
            .iter()
            .map(|&transform| filters::forward_on(raw, transform, &mut accelerator))
            .collect::<Result<Vec<_>, _>>()?;
        let limit = winner.0;
        let shared = (raw.len() <= SNAPSHOT_BYTES
            && concurrent_models_fit(
                descriptor,
                raw.len(),
                (2 * transforms.len() + 1, reservations),
            ))
        .then(|| {
            let total = PRIMER.len() + raw.len();
            Predictor::new(total, descriptor.table_log2, Dispatch::Native).map(|mut model| {
                timing.time(Step::PrimerTraining, || prime(&mut model, PRIMER));
                std::sync::Arc::new((total, model))
            })
        })
        .transpose()?;
        let each = trials.shared(transforms.len());
        let outcomes: Vec<_> = std::thread::scope(|scope| {
            let runs: Vec<_> = inputs
                .iter()
                .map(|input| {
                    let shared = shared.clone();
                    scope.spawn(move || {
                        let mut snapshot = Snapshot::shared(shared);
                        model_candidate(
                            input,
                            (descriptor, each, descriptor.segment_bytes()),
                            limit,
                            (timing, &mut snapshot),
                        )
                    })
                })
                .collect();
            runs.into_iter().map(|run| run.join()).collect()
        });
        for (outcome, &transform) in outcomes.into_iter().zip(&transforms) {
            let outcome = outcome
                .map_err(|_| BatoError::CorruptStream("model trial thread panicked".into()))?;
            let Some((cost, primer, mut chunks)) = outcome?.filter(|(cost, ..)| *cost < winner.0)
            else {
                continue;
            };
            for (chunk, bytes) in chunks.iter_mut().zip(raw.chunks(descriptor.chunk_bytes())) {
                chunk.crc = crc32c(bytes);
            }
            winner = (cost, chunks, Coding { transform, primer });
        }
        return Ok((winner.1, winner.2));
    }
    // Two or more primed candidates of this length follow: train once.
    let mut snapshot = Snapshot::new(
        raw.len() <= SNAPSHOT_BYTES
            && concurrent_models_fit(descriptor, raw.len(), (2, reservations))
            && (x86.is_some() || filters::CANDIDATES.contains(&selected_filter)),
    );
    if min_frame < winner.0 {
        if let Some(sites) = x86 {
            x86::forward(raw, sites);
            let candidate = model_candidate(
                raw,
                (descriptor, trials, descriptor.segment_bytes()),
                winner.0,
                (timing, &mut snapshot),
            );
            // Even when coding errors, leave the caller's input intact.
            x86::inverse(raw, sites);
            if let Some((cost, primer, mut chunks)) = candidate? {
                for (chunk, bytes) in chunks.iter_mut().zip(raw.chunks(descriptor.chunk_bytes())) {
                    chunk.crc = crc32c(bytes);
                }
                winner = (
                    cost,
                    chunks,
                    Coding {
                        transform: sites.transform(),
                        primer,
                    },
                );
            }
        }
    }
    // The independent codec has already compared every delta/shuffle
    // representation by exact byte cost. If one of those transformations won
    // that inexpensive competition, also give it a CM coding trial. This
    // avoids five additional full CM passes on every segment while testing
    // one promising representation that was previously CM-inaccessible.
    if filters::CANDIDATES.contains(&selected_filter) && min_frame < winner.0 {
        let transformed = filters::forward_on(raw, selected_filter, &mut accelerator)?;
        if let Some((cost, primer, mut chunks)) = model_candidate(
            &transformed,
            (descriptor, trials, descriptor.segment_bytes()),
            winner.0,
            (timing, &mut snapshot),
        )? {
            for (chunk, bytes) in chunks.iter_mut().zip(raw.chunks(descriptor.chunk_bytes())) {
                chunk.crc = crc32c(bytes);
            }
            winner = (
                cost,
                chunks,
                Coding {
                    transform: selected_filter,
                    primer,
                },
            );
        }
    }
    if min_frame < winner.0 {
        if let Some((cost, primer, chunks)) = model_candidate(
            raw,
            (descriptor, trials, descriptor.segment_bytes()),
            winner.0,
            (timing, &mut snapshot),
        )? {
            winner = (
                cost,
                chunks,
                Coding {
                    transform: TransformId::None,
                    primer,
                },
            );
        }
    }
    Ok((winner.1, winner.2))
}

/// Codes `input` with the model, with and without the primer where both are
/// tried, and returns the cheapest result that costs less than `limit`; on a
/// tie the primed run wins. The runs are independent, so they execute on two
/// threads: a run needs no result of the other, and a run that costs `limit`
/// or more is discarded either way, so the output does not depend on timing.
fn model_candidate(
    input: &[u8],
    (descriptor, trials, max_bytes): (Descriptor, Trials, usize),
    limit: usize,
    (timing, snapshot): (Timing<'_>, &mut Snapshot),
) -> Result<Option<(usize, Primer, Vec<Chunk>)>, BatoError> {
    // Primed and unprimed trials run side by side when both are tried.
    let models = if input.len() > UNPRIMED_TRIAL_BYTES {
        1
    } else {
        2
    };
    let stages = trials.stages(models);
    let run = |primer,
               (cap, incumbent),
               snapshot: Option<&mut Snapshot>|
     -> Result<Option<(usize, Primer, Vec<Chunk>)>, BatoError> {
        let run = Run {
            primer,
            dispatch: Dispatch::Native,
            budget: Some(cap),
            incumbent,
            max_bytes,
            timing,
            stages,
        };
        let chunks = encode_model(input, descriptor, run, snapshot)?;
        Ok(chunks.map(|chunks| (chunks_cost(&chunks), primer, chunks)))
    };
    let outcomes: Vec<Option<(usize, Primer, Vec<Chunk>)>> = if input.len() > UNPRIMED_TRIAL_BYTES {
        vec![run(Primer::Read, (limit, None), Some(snapshot))?]
    } else if trials.is_parallel() {
        let incumbent = AtomicUsize::new(limit);
        let (primed, unprimed) = std::thread::scope(|scope| {
            let unprimed = scope.spawn(|| {
                let result = run(Primer::Skipped, (limit, Some(&incumbent)), None)?;
                if let Some((cost, ..)) = &result {
                    incumbent.fetch_min(*cost, Ordering::Relaxed);
                }
                Ok::<_, BatoError>(result)
            });
            let primed = run(Primer::Read, (limit, Some(&incumbent)), Some(snapshot));
            if let Ok(Some((cost, ..))) = &primed {
                incumbent.fetch_min(*cost, Ordering::Relaxed);
            }
            (primed, unprimed.join())
        });
        let unprimed =
            unprimed.map_err(|_| BatoError::CorruptStream("model trial thread panicked".into()))?;
        vec![primed?, unprimed?]
    } else {
        let primed = run(Primer::Read, (limit, None), Some(snapshot))?;
        // The unprimed trial can only win by being strictly cheaper. Passing
        // this proven bound avoids modeling its remaining bytes after it can
        // no longer win; it does not change the winner or archive format.
        let cap = primed.as_ref().map_or(limit, |(cost, ..)| *cost);
        let unprimed = run(Primer::Skipped, (cap, None), None)?;
        vec![primed, unprimed]
    };
    let mut cheapest: Option<(usize, Primer, Vec<Chunk>)> = None;
    for trial in outcomes.into_iter().flatten() {
        if cheapest.as_ref().is_none_or(|(best, ..)| trial.0 < *best) && trial.0 < limit {
            cheapest = Some(trial);
        }
    }
    Ok(cheapest)
}

/// Framed size of a chunk list.
fn chunks_cost(chunks: &[Chunk]) -> usize {
    chunks
        .iter()
        .map(|c| crate::core::frame::block_cost(c.kind, c.length as usize, c.payload.len()))
        .sum()
}

/// A zero token-length field distinguishes this packet from static Huffman.
fn encode_fragment_packet(
    (tape, transform): (&[u8], TransformId),
    (descriptor, trials): (Descriptor, Trials),
    limit: usize,
    timing: Timing<'_>,
) -> Result<Option<Vec<u8>>, BatoError> {
    let transformed;
    let input = if transform == TransformId::None {
        tape
    } else {
        transformed = filters::forward(tape, transform)?;
        &transformed
    };
    let Some((cost, primer, chunks)) = model_candidate(
        input,
        (descriptor, trials, descriptor.segment_bytes()),
        limit,
        (timing, &mut Snapshot::new(false)),
    )?
    else {
        return Ok(None);
    };
    let mut packet = try_capacity(cost + 3 + varint::len(tape.len() as u64))?;
    packet.push(0);
    varint::push(&mut packet, tape.len());
    packet.push(u8::from(primer == Primer::Skipped));
    packet.push(transform as u8);
    for chunk in chunks {
        let length = if chunk.kind == BlockType::Raw {
            0
        } else {
            chunk.payload.len()
        };
        varint::push(&mut packet, length);
        packet.extend_from_slice(&chunk.payload);
    }
    Ok(Some(packet))
}

fn encode_cold_tail(
    raw: &[u8],
    (descriptor, trials): (Descriptor, Trials),
    (limit, crc): (usize, u32),
    timing: Timing<'_>,
) -> Result<Option<Chunk>, BatoError> {
    let Some(prefix) = raw_noise::cold_prefix(raw, descriptor.chunk_bytes()) else {
        return Ok(None);
    };
    let overhead = 3 + varint::len(prefix as u64);
    let Some(budget) = limit.checked_sub(prefix + overhead) else {
        return Ok(None);
    };
    let Some((_, primer, chunks)) = model_candidate(
        &raw[prefix..],
        (descriptor, trials, descriptor.chunk_bytes()),
        budget,
        (timing, &mut Snapshot::new(false)),
    )?
    else {
        return Ok(None);
    };
    let [tail] = chunks.as_slice() else {
        return Ok(None);
    };
    let packed = prefix + overhead + tail.payload.len();
    let kind = BlockType::LzHuffman;
    if tail.kind != BlockType::Coded
        || tail.payload.is_empty()
        || crate::core::frame::compact_block_cost(kind, raw.len(), packed, descriptor) >= limit
    {
        return Ok(None);
    }
    let mut payload = try_capacity(packed)?;
    payload.extend_from_slice(&[0, 0, u8::from(primer == Primer::Skipped)]);
    varint::push(&mut payload, prefix);
    payload.extend_from_slice(&raw[..prefix]);
    payload.extend_from_slice(&tail.payload);
    Ok(Some(Chunk {
        kind,
        length: raw.len() as u32,
        payload,
        crc,
    }))
}

fn decode_cold_tail(
    packet: &[u8],
    descriptor: Descriptor,
    length: usize,
    timing: Timing<'_>,
) -> Result<Vec<u8>, BatoError> {
    let invalid = || BatoError::CorruptStream("invalid stored-prefix packet".into());
    let primer = match packet.first() {
        Some(0) => Primer::Read,
        Some(1) => Primer::Skipped,
        _ => return Err(invalid()),
    };
    let mut at = 1;
    let prefix = varint::take(packet, &mut at)?;
    let tail = length.checked_sub(prefix).ok_or_else(invalid)?;
    let chunk = descriptor.chunk_bytes();
    if prefix < 3 * chunk || !prefix.is_multiple_of(chunk) || tail == 0 || tail > chunk {
        return Err(invalid());
    }
    let end = at.checked_add(prefix).ok_or_else(invalid)?;
    let stored = packet.get(at..end).ok_or_else(invalid)?;
    let coded = &packet[end..];
    if coded.is_empty() || coded.len() >= tail {
        return Err(invalid());
    }
    let block = Chunk {
        kind: BlockType::Coded,
        length: tail as u32,
        payload: try_copy(coded)?,
        crc: 0,
    };
    let decoded = decode_chunks(&[block], descriptor, tail, (primer, timing))?;
    let mut output = try_capacity(length)?;
    output.extend_from_slice(stored);
    output.extend_from_slice(&decoded);
    Ok(output)
}

fn decode_fragment_packet(
    packet: &[u8],
    descriptor: Descriptor,
    length: usize,
    timing: Timing<'_>,
) -> Result<Vec<u8>, BatoError> {
    let invalid = || BatoError::CorruptStream("invalid context-mixed fragments".into());
    let mut at = 0;
    let token_length = varint::take(packet, &mut at)?;
    if token_length == 0 || token_length >= length {
        return Err(invalid());
    }
    let primer = match packet.get(at) {
        Some(0) => Primer::Read,
        Some(1) => Primer::Skipped,
        _ => return Err(invalid()),
    };
    let transform = TransformId::from_u8(*packet.get(at + 1).ok_or_else(invalid)?)?;
    if transform != TransformId::None && x86::Sites::of_transform(transform).is_none() {
        return Err(invalid());
    }
    at += 2;
    let size = descriptor.chunk_bytes();
    let mut chunks = try_capacity(token_length.div_ceil(size))?;
    for position in (0..token_length).step_by(size) {
        let n = size.min(token_length - position);
        let coded = varint::take(packet, &mut at)?;
        if coded >= n {
            return Err(invalid());
        }
        let (kind, length) = if coded == 0 {
            (BlockType::Raw, n)
        } else {
            (BlockType::Coded, coded)
        };
        let end = at.checked_add(length).ok_or_else(invalid)?;
        let payload = try_copy(packet.get(at..end).ok_or_else(invalid)?)?;
        chunks.push(Chunk {
            kind,
            length: n as u32,
            payload,
            crc: 0,
        });
        at = end;
    }
    if at != packet.len() {
        return Err(invalid());
    }
    let mut tape = decode_chunks(&chunks, descriptor, token_length, (primer, timing))?;
    timing.time(Step::InverseTransform, || {
        filters::inverse(&mut tape, transform)?;
        lz::decode(&tape, length)
    })
}

/// Decodes one segment and reverses its declared transform. Chunk CRCs are
/// verified on the original bytes.
pub(crate) fn decode_segment(
    chunks: &[Chunk],
    (descriptor, max_output): (Descriptor, usize),
    coding: Coding,
    timing: Timing<'_>,
    accelerator: Option<&mut WorkerFilters<'_>>,
) -> Result<Vec<u8>, BatoError> {
    descriptor.validate()?;
    if coding.transform == TransformId::Spatial {
        let [chunk] = chunks else {
            return Err(BatoError::CorruptStream("invalid spatial segment".into()));
        };
        let length = chunk.length as usize;
        if chunk.kind != BlockType::Huffman || length > max_output.min(descriptor.segment_bytes()) {
            return Err(BatoError::CorruptStream(
                "invalid spatial segment geometry".into(),
            ));
        }
        let output = timing.time(Step::ModelDecode, || {
            spatial::decode(&chunk.payload, length)
        })?;
        timing.time(Step::Checksum, || verify_chunks(chunks, &output))?;
        return Ok(output);
    }
    if let Some(sites) = x86::Sites::of_container(coding.transform) {
        if chunks.iter().any(|chunk| chunk.kind.independent()) {
            return Err(BatoError::CorruptStream(
                "deflate transform on an independent segment".into(),
            ));
        }
        let mut transformed = decode_chunks(
            chunks,
            descriptor,
            max_output.saturating_mul(DEFLATE_EXPANSION),
            (coding.primer, timing),
        )?;
        timing.time(Step::Checksum, || verify_chunks(chunks, &transformed))?;
        return timing.time(Step::InverseTransform, || {
            if let Some(sites) = sites {
                x86::inverse(&mut transformed, sites);
            }
            container::inverse(&transformed, max_output.min(descriptor.segment_bytes()))
        });
    }
    let mut output = if chunks.len() == 1 && chunks[0].kind.independent() {
        let chunk = &chunks[0];
        let n = chunk.length as usize;
        if n == 0 || n > descriptor.segment_bytes() || n > max_output {
            return Err(BatoError::DecompressionLimitExceeded {
                size: n,
                limit: max_output.min(descriptor.segment_bytes()),
            });
        }
        if chunk.kind == BlockType::LzHuffman && chunk.payload.first() == Some(&0) {
            if coding.transform != TransformId::None || coding.primer != Primer::Read {
                return Err(BatoError::CorruptStream(
                    "invalid fragment block flags".into(),
                ));
            }
            let output = if let Some(tail) = chunk.payload.strip_prefix(&[0, 0]) {
                decode_cold_tail(tail, descriptor, n, timing)?
            } else {
                decode_fragment_packet(&chunk.payload[1..], descriptor, n, timing)?
            };
            timing.time(Step::Checksum, || verify_chunks(chunks, &output))?;
            return Ok(output);
        }
        fast::decode(chunk.kind, &chunk.payload, n)?
    } else {
        decode_chunks(chunks, descriptor, max_output, (coding.primer, timing))?
    };
    timing.time(Step::InverseTransform, || {
        filters::inverse_on(&mut output, coding.transform, accelerator)
    })?;
    timing.time(Step::Checksum, || verify_chunks(chunks, &output))?;
    Ok(output)
}

/// Checks every chunk's CRC32C against its slice of the original bytes.
fn verify_chunks(chunks: &[Chunk], output: &[u8]) -> Result<(), BatoError> {
    let mut start = 0;
    for chunk in chunks {
        let end = start + chunk.length as usize;
        let computed = crc32c(&output[start..end]);
        if computed != chunk.crc {
            return Err(BatoError::ChecksumMismatch {
                expected: chunk.crc,
                computed,
            });
        }
        start = end;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn encode(raw: &[u8], descriptor: Descriptor) -> Result<Vec<Chunk>, BatoError> {
    encode_with_dispatch(raw, descriptor, Dispatch::Native)
}
#[cfg(test)]
fn encode_with_dispatch(
    raw: &[u8],
    descriptor: Descriptor,
    dispatch: Dispatch,
) -> Result<Vec<Chunk>, BatoError> {
    let run = Run {
        primer: Primer::Read,
        dispatch,
        budget: None,
        incumbent: None,
        max_bytes: descriptor.segment_bytes(),
        timing: Timing::off(),
        stages: Stages::Serial,
    };
    encode_model(raw, descriptor, run, None)?
        .ok_or_else(|| BatoError::CorruptStream("unbounded model encoding aborted".into()))
}
fn encode_model(
    raw: &[u8],
    descriptor: Descriptor,
    run: Run<'_>,
    snapshot: Option<&mut Snapshot>,
) -> Result<Option<Vec<Chunk>>, BatoError> {
    let primer = primer_bytes(run.primer);
    descriptor.validate()?;
    if raw.len() > run.max_bytes {
        return Err(BatoError::InvalidOptions(
            "CM segment input exceeds its declared geometry".into(),
        ));
    }
    if raw.is_empty() {
        return Ok(Some(Vec::new()));
    }
    let total = primer.len() + raw.len();
    let kept = snapshot
        .as_ref()
        .and_then(|snapshot| snapshot.model.as_ref())
        .filter(|arc| arc.0 == total)
        .map(|arc| arc.1.clone());
    let mut model = match kept {
        Some(model) => model,
        None => {
            let mut model = Predictor::new(total, descriptor.table_log2, run.dispatch)?;
            run.timing
                .time(Step::PrimerTraining, || prime(&mut model, primer));
            if let Some(snapshot) = snapshot.filter(|snapshot| snapshot.wanted) {
                snapshot.model = Some(std::sync::Arc::new((total, model.clone())));
            }
            model
        }
    };
    let mut history = try_capacity(total)?;
    history.extend_from_slice(primer);
    history.extend_from_slice(raw);
    let geometry = (&history[..], primer.len());
    let cap = || match run.incumbent {
        Some(incumbent) => {
            // The primed trial wins equal costs. The static limit is always strict.
            let best = incumbent.load(Ordering::Relaxed);
            Some(
                run.budget
                    .unwrap_or(usize::MAX)
                    .min(best.saturating_add(usize::from(run.primer == Primer::Read))),
            )
        }
        None => run.budget,
    };
    if run.stages == Stages::Pipelined {
        // The stage threads carry no step timer; `--timing` runs serially.
        return run.timing.time(Step::ModelTrial, || {
            pipeline::run_staged(model, geometry, |engine| {
                encode_chunks(raw, geometry, descriptor, (engine, &cap))
            })?
        });
    }
    if run.timing.enabled() {
        model.start_meter();
    }
    let coded = run.timing.time(Step::ModelTrial, || {
        encode_chunks(
            raw,
            geometry,
            descriptor,
            (&mut Engine::Serial(&mut model), &cap),
        )
    });
    model.finish_meter(run.timing);
    coded
}

fn encode_chunks(
    raw: &[u8],
    (history, primed): (&[u8], usize),
    descriptor: Descriptor,
    (model, budget): (&mut Engine<'_>, &impl Fn() -> Option<usize>),
) -> Result<Option<Vec<Chunk>>, BatoError> {
    let size = descriptor.chunk_bytes();
    let mut output = try_capacity(raw.len().div_ceil(size))?;
    let mut position = 0;
    let mut spent = 0usize;
    let mut minimum: usize = raw
        .chunks(size)
        .map(|b| crate::core::frame::block_cost(BlockType::Raw, b.len(), 0))
        .sum();
    for bytes in raw.chunks(size) {
        let base = crate::core::frame::block_cost(BlockType::Raw, bytes.len(), 0);
        let allowed = budget().map_or(bytes.len() - 1, |cap| {
            cap.saturating_sub(spent + minimum).min(bytes.len() - 1)
        });
        let mut coder = Encoder::new(allowed)?;
        for (index, &byte) in bytes.iter().enumerate() {
            // Already emitted bytes cannot be removed. The raw fallback for
            // this chunk is at least as large, including its framing.
            if index.is_multiple_of(64)
                && budget().is_some_and(|cap| spent + minimum + coder.emitted() >= cap)
            {
                return Ok(None);
            }
            for shift in (0..8).rev() {
                let bit = (byte >> shift) & 1;
                let probability = model.probability(&history[..primed + position])?;
                if !coder.exceeded() {
                    coder.bit(bit, probability)?;
                }
                model.observe(bit);
                if coder.exceeded() && allowed < bytes.len() - 1 {
                    return Ok(None);
                }
            }
            position += 1;
        }
        let payload = coder.finish();
        let chunk = Chunk {
            kind: if payload.is_some() {
                BlockType::Coded
            } else {
                BlockType::Raw
            },
            length: bytes.len() as u32,
            payload: match payload {
                Some(payload) => payload,
                None => try_copy(bytes)?,
            },
            crc: crc32c(bytes),
        };
        minimum -= base;
        spent += crate::core::frame::block_cost(chunk.kind, bytes.len(), chunk.payload.len());
        if budget().is_some_and(|cap| spent + minimum >= cap) {
            return Ok(None);
        }
        output.push(chunk);
    }
    Ok(Some(output))
}

/// Validates geometry and limits; replays only history needed by coded chunks.
/// The result is still transformed when the segment declares a transform.
fn decode_chunks(
    chunks: &[Chunk],
    descriptor: Descriptor,
    max_output: usize,
    (primer, timing): (Primer, Timing<'_>),
) -> Result<Vec<u8>, BatoError> {
    let primer = primer_bytes(primer);
    descriptor.validate()?;
    let mut total = 0usize;
    for (index, chunk) in chunks.iter().enumerate() {
        let length = chunk.length as usize;
        if !matches!(chunk.kind, BlockType::Raw | BlockType::Coded)
            || length == 0
            || length > descriptor.chunk_bytes()
            || (index + 1 != chunks.len() && length != descriptor.chunk_bytes())
            || (chunk.kind == BlockType::Coded && chunk.payload.len() >= length)
            || (chunk.kind != BlockType::Coded && chunk.payload.len() != length)
        {
            return Err(BatoError::CorruptStream(
                "invalid CM chunk geometry or coded length".into(),
            ));
        }
        total = total
            .checked_add(length)
            .ok_or_else(|| BatoError::CorruptStream("CM output counter overflow".into()))?;
        if total > max_output {
            return Err(BatoError::DecompressionLimitExceeded {
                size: total,
                limit: max_output,
            });
        }
    }
    if total == 0 {
        return Ok(Vec::new());
    }
    let Some(last_coded) = chunks
        .iter()
        .rposition(|chunk| chunk.kind == BlockType::Coded)
    else {
        let mut output = try_capacity(total)?;
        for chunk in chunks {
            output.extend_from_slice(&chunk.payload);
        }
        return Ok(output);
    };
    let mut output = try_capacity(primer.len() + total)?;
    let mut model = Predictor::new(
        primer.len() + total,
        descriptor.table_log2,
        Dispatch::Native,
    )?;
    timing.time(Step::PrimerTraining, || prime(&mut model, primer));
    if timing.enabled() {
        model.start_meter();
    }
    output.extend_from_slice(primer);
    let started = timing.start();
    for (index, chunk) in chunks.iter().enumerate() {
        if chunk.kind == BlockType::Coded {
            let mut coder = Decoder::new(&chunk.payload);
            for _ in 0..chunk.length {
                let mut byte = 0u8;
                for _ in 0..8 {
                    let bit = coder.bit(model.probability(&output))?;
                    model.observe(bit);
                    byte = (byte << 1) | bit;
                }
                output.push(byte);
            }
            coder.finish()?;
        } else if index > last_coded {
            // No later coded chunk consumes predictor state before the next
            // independent segment. Preserve the bytes without replaying it.
            output.extend_from_slice(&chunk.payload);
        } else {
            for &byte in &chunk.payload {
                model.observe_byte(byte, &output);
                output.push(byte);
            }
        }
    }
    timing.stop(Step::ModelDecode, started);
    model.finish_meter(timing);
    output.drain(..primer.len());
    Ok(output)
}

#[cfg(test)]
mod tests {

    #[test]
    fn fragment_packets_restore_and_validate_nested_lengths() {
        let seed: Vec<u8> = (0u32..3000)
            .flat_map(|n| n.wrapping_mul(0x9117_4f2d).to_le_bytes())
            .collect();
        let mut input = seed.clone();
        input.extend_from_slice(b"an edit");
        input.extend_from_slice(&seed[7..]);
        input.extend_from_slice(b"a second edit");
        input.extend_from_slice(&seed);
        let descriptor = Descriptor::default();
        let tape = fragments::commands(&input).unwrap().unwrap();
        let payload = encode_fragment_packet(
            (&tape, TransformId::None),
            (descriptor, Trials::Sequential),
            input.len(),
            Timing::off(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(payload[0], 0);
        let decoded =
            decode_fragment_packet(&payload[1..], descriptor, input.len(), Timing::off()).unwrap();
        assert_eq!(decoded, input);
        assert!(decode_fragment_packet(
            &payload[1..payload.len() - 1],
            descriptor,
            input.len(),
            Timing::off()
        )
        .is_err());
        assert!(
            decode_fragment_packet(&payload[1..], descriptor, tape.len(), Timing::off()).is_err()
        );
    }

    use super::*;

    fn decode(
        chunks: &[Chunk],
        descriptor: Descriptor,
        max_output: usize,
    ) -> Result<Vec<u8>, BatoError> {
        decode_segment(
            chunks,
            (descriptor, max_output),
            Coding {
                transform: TransformId::None,
                primer: Primer::Read,
            },
            Timing::off(),
            None,
        )
    }
    use sha2::{Digest, Sha256};

    fn geometry() -> Descriptor {
        Descriptor {
            segment_log2: 16,
            chunk_log2: 14,
            table_log2: 16,
        }
    }

    fn fixture(length: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        for i in 0..length.div_ceil(32) {
            bytes.extend_from_slice(&Sha256::digest((i as u64).to_le_bytes()));
        }
        bytes.truncate(length);
        bytes
    }

    #[test]
    fn edge_sizes_and_all_single_byte_values_roundtrip() {
        let descriptor = geometry();
        let c = descriptor.chunk_bytes();
        let s = descriptor.segment_bytes();
        for length in [0, 1, 2, 255, 4095, 4096, 4097, c - 1, c, c + 1, s - 1, s] {
            let raw = fixture(length);
            let chunks = encode(&raw, descriptor).unwrap();
            assert_eq!(decode(&chunks, descriptor, length).unwrap(), raw);
            assert!(chunks.iter().all(|c| c.payload.len() <= c.length as usize));
            assert!(chunks
                .iter()
                .filter(|c| c.kind == BlockType::Coded)
                .all(|c| c.payload.len() < c.length as usize));
        }
        let mut snapshot = Snapshot::new(true);
        let run = Run {
            primer: Primer::Read,
            dispatch: Dispatch::Native,
            budget: None,
            incumbent: None,
            max_bytes: descriptor.segment_bytes(),
            timing: Timing::off(),
            stages: Stages::Serial,
        };
        for byte in 0..=255u8 {
            let chunks = encode_model(&[byte], descriptor, run, Some(&mut snapshot))
                .unwrap()
                .unwrap();
            assert_eq!(decode(&chunks, descriptor, 1).unwrap(), [byte]);
        }
        assert!(encode(&vec![0; s + 1], descriptor).is_err());
    }

    #[test]
    fn shared_trial_bound_keeps_the_sequential_winner() {
        let descriptor = geometry();
        let raw = b"integer context mixer reference text\n".repeat(1200);
        let trial = |trials| {
            model_candidate(
                &raw,
                (descriptor, trials, descriptor.segment_bytes()),
                raw.len(),
                (Timing::off(), &mut Snapshot::new(false)),
            )
            .unwrap()
            .unwrap()
        };
        let (cost, primer, chunks) = trial(Trials::Sequential);
        let (other_cost, other_primer, other_chunks) = trial(Trials::Parallel { reservations: 2 });
        assert_eq!((cost, primer), (other_cost, other_primer));
        assert_eq!(
            chunks
                .iter()
                .map(|c| c.payload.as_slice())
                .collect::<Vec<_>>(),
            other_chunks
                .iter()
                .map(|c| c.payload.as_slice())
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn unprimed_segments_round_trip_and_differ_from_primed_ones() {
        let descriptor = geometry();
        let raw = b"integer context mixer reference text\n".repeat(100);
        let encode_with = |primer| {
            let run = Run {
                primer,
                dispatch: Dispatch::Native,
                budget: None,
                incumbent: None,
                max_bytes: descriptor.segment_bytes(),
                timing: Timing::off(),
                stages: Stages::Serial,
            };
            encode_model(&raw, descriptor, run, None).unwrap().unwrap()
        };
        let (primed, unprimed) = (encode_with(Primer::Read), encode_with(Primer::Skipped));
        assert_ne!(primed[0].payload, unprimed[0].payload);
        assert_eq!(
            decode_chunks(
                &unprimed,
                descriptor,
                raw.len(),
                (Primer::Skipped, Timing::off())
            )
            .unwrap(),
            raw
        );
        assert_eq!(
            decode_chunks(
                &primed,
                descriptor,
                raw.len(),
                (Primer::Read, Timing::off())
            )
            .unwrap(),
            raw
        );
    }

    #[test]
    fn deflate_transform_round_trips_and_is_checked_on_both_sides() {
        let descriptor = Descriptor {
            segment_log2: 20,
            chunk_log2: 16,
            table_log2: 16,
        };
        let mut raw = b"archive member header and name\n".repeat(3);
        raw.extend_from_slice(include_bytes!("deflate/fixtures/l6.deflate"));
        raw.extend_from_slice(b"\x00 separator \xff");
        raw.extend_from_slice(include_bytes!("deflate/fixtures/l9.deflate"));
        let original = raw.clone();
        let (chunks, coding) = encode_segment(
            &mut raw,
            (descriptor, Trials::Parallel { reservations: 1 }),
            Timing::off(),
            None,
        )
        .unwrap();
        assert_eq!(raw, original, "encoding restores its input");
        assert_eq!(coding.transform, TransformId::Deflate);
        let coded: usize = chunks.iter().map(|chunk| chunk.length as usize).sum();
        assert!(coded > original.len(), "chunks hold the inflated bytes");
        let decoded =
            decode_segment(&chunks, (descriptor, 1 << 20), coding, Timing::off(), None).unwrap();
        assert_eq!(decoded, original);
        let mut damaged = chunks.clone();
        damaged[0].crc ^= 1;
        assert!(
            decode_segment(&damaged, (descriptor, 1 << 20), coding, Timing::off(), None).is_err()
        );
        let independent = Coding {
            transform: TransformId::Deflate,
            primer: Primer::Read,
        };
        let stored = [Chunk {
            kind: BlockType::Stored,
            length: 3,
            payload: vec![1, 2, 3],
            crc: crc32c(&[1, 2, 3]),
        }];
        assert!(decode_segment(
            &stored,
            (descriptor, 1 << 20),
            independent,
            Timing::off(),
            None
        )
        .is_err());
    }

    #[test]
    fn winning_segment_never_exceeds_raw_cm_or_independent_cost() {
        let descriptor = geometry();
        let mut raw = b"catalogue metadata and recurring text\\n".repeat(128);
        raw.extend_from_slice(include_bytes!("deflate/fixtures/l6.deflate"));
        let original = raw.clone();
        let independent = fast::best(&raw, None, &mut None).unwrap().cost();
        let raw_model = model_candidate(
            &raw,
            (descriptor, Trials::Sequential, descriptor.segment_bytes()),
            usize::MAX,
            (Timing::off(), &mut Snapshot::new(false)),
        )
        .unwrap()
        .unwrap()
        .0;

        let (chunks, coding) = encode_segment(
            &mut raw,
            (descriptor, Trials::Sequential),
            Timing::off(),
            None,
        )
        .unwrap();
        assert_eq!(raw, original, "input is unchanged after candidate trials");
        assert!(chunks_cost(&chunks) <= independent);
        assert!(chunks_cost(&chunks) <= raw_model);
        assert_eq!(
            decode_segment(
                &chunks,
                (descriptor, descriptor.segment_bytes()),
                coding,
                Timing::off(),
                None
            )
            .unwrap(),
            original,
        );
    }

    #[test]
    fn primer_is_the_documented_format_constant() {
        use sha2::{Digest, Sha256};
        let digest: String = Sha256::digest(PRIMER)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        assert_eq!(PRIMER.len(), 104_049);
        assert_eq!(
            digest,
            "add44b9a887b2d00ac973cf83b35f5f78ed2a42b265cdd22cd463ea63ee3bb20"
        );
    }

    #[test]
    fn numeric_and_text_decode_peaks_stay_within_the_decode_bound() {
        let descriptor = Descriptor {
            segment_log2: 20,
            chunk_log2: 18,
            table_log2: 16,
        };
        let numeric: Vec<u8> = (0..120_000u64)
            .flat_map(|i| {
                format!("{},{}.{:03}\n", i * 7 + i % 5, i * 31 % 999_983, i % 997).into_bytes()
            })
            .take(descriptor.segment_bytes())
            .collect();
        let bound = memory_required(descriptor, Direction::Decode).unwrap();
        for raw in [numeric, fixture(descriptor.segment_bytes())] {
            let chunks = encode(&raw, descriptor).unwrap();
            let (decoded, observed) = crate::core::diagnostics::capture(|| {
                decode(&chunks, descriptor, raw.len()).unwrap()
            });
            assert_eq!(decoded, raw);
            assert!(
                observed.heap.peak <= bound,
                "{} > {bound}",
                observed.heap.peak
            );
        }
        assert!(bound <= memory_required(descriptor, Direction::Encode).unwrap());
    }

    #[test]
    fn deflate_transformed_segment_peaks_stay_within_the_bounds() {
        let descriptor = Descriptor {
            segment_log2: 16,
            chunk_log2: 14,
            table_log2: 16,
        };
        let (first, second) = (
            include_bytes!("deflate/fixtures/l6.deflate"),
            include_bytes!("deflate/fixtures/l9.deflate"),
        );
        let mut raw = b"member header ".to_vec();
        raw.extend_from_slice(first);
        raw.extend_from_slice(b" next ");
        raw.extend_from_slice(second);
        assert!(raw.len() <= descriptor.segment_bytes());
        let (encoded, observed) = crate::core::diagnostics::capture(|| {
            encode_segment(
                &mut raw.clone(),
                (descriptor, Trials::Sequential),
                Timing::off(),
                None,
            )
            .unwrap()
        });
        let (chunks, coding) = encoded;
        assert_eq!(coding.transform, TransformId::Deflate);
        let transformed: usize = chunks.iter().map(|chunk| chunk.length as usize).sum();
        assert!(transformed > descriptor.segment_bytes(), "{transformed}");
        let bound = memory_required(descriptor, Direction::Encode).unwrap();
        assert!(
            observed.heap.peak <= bound,
            "{} > {bound}",
            observed.heap.peak
        );
        let (decoded, observed) = crate::core::diagnostics::capture(|| {
            decode_segment(
                &chunks,
                (descriptor, descriptor.segment_bytes()),
                coding,
                Timing::off(),
                None,
            )
            .unwrap()
        });
        assert_eq!(decoded, raw);
        let bound = memory_required(descriptor, Direction::Decode).unwrap();
        assert!(
            observed.heap.peak <= bound,
            "{} > {bound}",
            observed.heap.peak
        );
    }

    #[test]
    fn deflate_segments_beyond_the_expansion_limit_are_rejected() {
        let descriptor = Descriptor {
            segment_log2: 16,
            chunk_log2: 12,
            table_log2: 16,
        };
        let chunk = descriptor.chunk_bytes();
        let stored = |count: usize| -> Vec<Chunk> {
            (0..count)
                .map(|_| Chunk {
                    kind: BlockType::Raw,
                    length: chunk as u32,
                    payload: vec![0; chunk],
                    crc: crc32c(&vec![0; chunk]),
                })
                .collect()
        };
        let coding = Coding {
            transform: TransformId::Deflate,
            primer: Primer::Read,
        };
        let limit = DEFLATE_EXPANSION * descriptor.segment_bytes() / chunk;
        let decode = |count| {
            decode_segment(
                &stored(count),
                (descriptor, descriptor.segment_bytes()),
                coding,
                Timing::off(),
                None,
            )
        };
        assert!(matches!(
            decode(limit + 1),
            Err(BatoError::DecompressionLimitExceeded { .. })
        ));
        // Within the limit the chunks are accepted by the size check; they
        // are not a valid container, which is a different error.
        assert!(!matches!(
            decode(limit),
            Err(BatoError::DecompressionLimitExceeded { .. })
        ));
    }

    #[test]
    fn all_raw_decode_allocates_only_the_output() {
        let raw = fixture(32768);
        let chunks = vec![Chunk {
            kind: BlockType::Raw,
            length: raw.len() as u32,
            payload: raw.clone(),
            crc: crc32c(&raw),
        }];
        let (_, observed) = crate::core::diagnostics::capture(|| {
            assert_eq!(
                decode(
                    &chunks,
                    Descriptor {
                        chunk_log2: 16,
                        ..geometry()
                    },
                    raw.len()
                )
                .unwrap(),
                raw
            );
        });
        assert!(
            observed.heap.peak <= raw.len() + 1024,
            "{}",
            observed.heap.peak
        );
    }

    #[test]
    fn raw_chunks_update_the_continuing_model_exactly_once() {
        let descriptor = geometry();
        let c = descriptor.chunk_bytes();
        let raw: Vec<u8> = b"segment history continues across independent coder flushes\n"
            .iter()
            .copied()
            .cycle()
            .take(c * 3 + 7)
            .collect();
        let mut chunks = encode(&raw, descriptor).unwrap();
        assert!(chunks.iter().any(|c| c.kind == BlockType::Coded));
        chunks[0].kind = BlockType::Raw;
        chunks[0].payload = raw[..c].to_vec();
        assert_eq!(decode(&chunks, descriptor, raw.len()).unwrap(), raw);
        let mut mixed = fixture(c);
        mixed.extend_from_slice(&vec![0; c]);
        let chunks = encode(&mixed, descriptor).unwrap();
        assert!(chunks.iter().any(|c| c.kind == BlockType::Coded));
        assert!(chunks.iter().any(|c| c.kind == BlockType::Raw));
        assert_eq!(decode(&chunks, descriptor, mixed.len()).unwrap(), mixed);
    }

    #[test]
    fn malformed_geometry_crc_and_output_limits_are_rejected() {
        let descriptor = geometry();
        let raw = b"bounded segment coding\n".repeat(2000);
        let chunks = encode(&raw, descriptor).unwrap();
        let (limited, observed) =
            crate::core::diagnostics::capture(|| decode(&chunks, descriptor, 1));
        assert!(matches!(
            limited,
            Err(BatoError::DecompressionLimitExceeded { .. })
        ));
        assert_eq!(observed.heap.peak, 0);
        let mut broken = chunks.clone();
        broken[0].crc ^= 1;
        assert!(decode(&broken, descriptor, raw.len()).is_err());
        let mut broken = chunks.clone();
        broken[0].length -= 1;
        assert!(decode(&broken, descriptor, raw.len()).is_err());
        let mut broken = chunks.clone();
        broken[0].kind = BlockType::Raw;
        assert!(decode(&broken, descriptor, raw.len()).is_err());
        let mut broken = chunks.clone();
        broken[0].payload.push(0);
        assert!(decode(&broken, descriptor, raw.len()).is_err());
        assert!(
            memory_required(Descriptor::default(), Direction::Encode).unwrap() < 1536 * 1024 * 1024
        );
        assert!(
            memory_required(Descriptor::default(), Direction::Decode).unwrap()
                < 1024 * 1024 * 1024 - 64 * 1024 * 1024
        );
    }
    #[test]
    fn scalar_and_vector_dispatch_emit_identical_chunk_bytes() {
        let raw = b"complete segment scalar and vector parity\n".repeat(1200);
        let vector = encode_with_dispatch(&raw, geometry(), Dispatch::Native).unwrap();
        let scalar = encode_with_dispatch(&raw, geometry(), Dispatch::Scalar).unwrap();
        assert_eq!(vector.len(), scalar.len());
        for (a, b) in vector.iter().zip(&scalar) {
            assert_eq!((a.kind, a.length, a.crc), (b.kind, b.length, b.crc));
            assert_eq!(a.payload, b.payload);
        }
        assert_eq!(decode(&scalar, geometry(), raw.len()).unwrap(), raw);
    }

    #[test]
    fn scalar_and_vector_bytes_and_raw_observation_agree() {
        let descriptor = Descriptor { ..geometry() };
        let c = descriptor.chunk_bytes();
        let mut raw = fixture(c);
        raw.extend_from_slice(&b"original causal integer model\n".repeat(1100));
        let vector = encode_with_dispatch(&raw, descriptor, Dispatch::Native).unwrap();
        let scalar = encode_with_dispatch(&raw, descriptor, Dispatch::Scalar).unwrap();
        assert_eq!(vector.len(), scalar.len());
        for (a, b) in vector.iter().zip(&scalar) {
            assert_eq!((a.kind, a.length, a.crc), (b.kind, b.length, b.crc));
            assert_eq!(a.payload, b.payload);
        }
        assert!(vector.iter().any(|c| c.kind == BlockType::Raw));
        assert!(vector.iter().any(|c| c.kind == BlockType::Coded));
        assert_eq!(decode(&vector, descriptor, raw.len()).unwrap(), raw);
        let mut mixed = vector;
        let index = mixed
            .iter()
            .position(|chunk| chunk.kind == BlockType::Coded)
            .unwrap();
        assert!(index + 1 < mixed.len());
        mixed[index].kind = BlockType::Raw;
        mixed[index].payload = raw[index * c..index * c + mixed[index].length as usize].to_vec();
        assert_eq!(decode(&mixed, descriptor, raw.len()).unwrap(), raw);
    }
}

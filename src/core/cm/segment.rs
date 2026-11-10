//! Original segment-persistent coding with independently flushed chunks.
//! Provenance: Bato's supplied chunk/segment design and the original sibling
//! coder and predictor; no third-party implementation is incorporated.

use super::{
    coder::{Decoder, Encoder},
    deflate::container,
    descriptor::Descriptor,
    fast, filters,
    predictor::{self, Predictor},
    profile::{Step, Timing},
    simd::Dispatch,
    x86,
};
use crate::core::{
    checksum::crc32c,
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
const UNPRIMED_TRIAL_BYTES: usize = 1 << 21;

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
    /// Largest input the run accepts: the segment size, or for deflate-
    /// transformed bytes `DEFLATE_EXPANSION` times that.
    max_bytes: usize,
    timing: Timing<'a>,
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

/// Whether the encoder may run its independent model trials on two threads.
/// The archive is the same either way; only the time differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Trials {
    Sequential,
    Parallel,
}

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
            + (segment / chunk)
                * (std::mem::size_of::<Chunk>()
                    + std::mem::size_of::<crate::core::codec::EncodedChunk>())
            + 2 * chunk
            + x86::SELECTION_SCRATCH
            + WORKER_SLACK,
    )
}

/// Parallel primed/unprimed trials share a worker's existing reservation.
/// If two predictors do not fit, run the same two candidates sequentially.
/// This retains low-memory configurations and does not change coded bytes.
pub(crate) fn parallel_trials_fit(descriptor: Descriptor) -> bool {
    let Ok(reserved) = memory_required(descriptor, Direction::Encode) else {
        return false;
    };
    let segment = descriptor.segment_bytes();
    let chunk = descriptor.chunk_bytes();
    let trial_input = segment
        .saturating_mul(DEFLATE_EXPANSION)
        .min(UNPRIMED_TRIAL_BYTES);
    let per_trial = predictor::memory_required(trial_input + PRIMER.len(), descriptor.table_log2)
        .saturating_add(2 * trial_input + PRIMER.len() + 2 * chunk + WORKER_SLACK);
    let required = 2usize.saturating_mul(per_trial).saturating_add(4 * segment);
    reserved >= required
}

/// Encodes one segment, first applying the reversible x86 address transform
/// in place when its byte statistics favor it. `raw` is restored before
/// returning, and chunk CRCs always cover the original bytes.
pub(crate) fn encode_segment(
    raw: &mut [u8],
    (descriptor, trials): (Descriptor, Trials),
    timing: Timing<'_>,
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
    let x86 = timing.time(Step::TransformSelection, || x86::select(raw))?;
    let independent = timing.time(Step::FastCandidates, || {
        fast::best(raw, x86.map(x86::Sites::transform))
    })?;
    let original_crc = crc32c(raw);
    let selected_filter = independent.transform;
    // Keep complete framed costs throughout: a winning deflate transform is
    // not proof that the raw CM candidate (or the x86 candidate) is worse.
    let mut winner = (
        independent.cost(),
        vec![Chunk {
            kind: independent.kind,
            length: raw.len() as u32,
            payload: independent.payload,
            crc: original_crc,
        }],
        plain(independent.transform),
    );
    let minimum = |len: usize| -> usize {
        (0..len)
            .step_by(descriptor.chunk_bytes())
            .map(|start| {
                crate::core::frame::block_cost(
                    BlockType::Coded,
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
                timing,
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
    let min_frame = minimum(raw.len());
    if min_frame < winner.0 {
        if let Some(sites) = x86 {
            x86::forward(raw, sites);
            let candidate = model_candidate(
                raw,
                (descriptor, trials, descriptor.segment_bytes()),
                winner.0,
                timing,
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
        let transformed = filters::forward(raw, selected_filter)?;
        if let Some((cost, primer, mut chunks)) = model_candidate(
            &transformed,
            (descriptor, trials, descriptor.segment_bytes()),
            winner.0,
            timing,
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
            timing,
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
    timing: Timing<'_>,
) -> Result<Option<(usize, Primer, Vec<Chunk>)>, BatoError> {
    let run = |primer, cap| -> Result<Option<(usize, Primer, Vec<Chunk>)>, BatoError> {
        let run = Run {
            primer,
            dispatch: Dispatch::Native,
            budget: Some(cap),
            max_bytes,
            timing,
        };
        let chunks = encode_model(input, descriptor, run)?;
        Ok(chunks.map(|chunks| (chunks_cost(&chunks), primer, chunks)))
    };
    let outcomes: Vec<Option<(usize, Primer, Vec<Chunk>)>> = if input.len() > UNPRIMED_TRIAL_BYTES {
        vec![run(Primer::Read, limit)?]
    } else if trials == Trials::Parallel {
        let (primed, unprimed) = std::thread::scope(|scope| {
            let unprimed = scope.spawn(|| run(Primer::Skipped, limit));
            (run(Primer::Read, limit), unprimed.join())
        });
        let unprimed =
            unprimed.map_err(|_| BatoError::CorruptStream("model trial thread panicked".into()))?;
        vec![primed?, unprimed?]
    } else {
        let primed = run(Primer::Read, limit)?;
        // The unprimed trial can only win by being strictly cheaper. Passing
        // this proven bound avoids modeling its remaining bytes after it can
        // no longer win; it does not change the winner or archive format.
        let cap = primed.as_ref().map_or(limit, |(cost, ..)| *cost);
        let unprimed = run(Primer::Skipped, cap)?;
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

/// Decodes one segment and reverses its declared transform. Chunk CRCs are
/// verified on the original bytes.
pub(crate) fn decode_segment(
    chunks: &[Chunk],
    (descriptor, max_output): (Descriptor, usize),
    coding: Coding,
    timing: Timing<'_>,
) -> Result<Vec<u8>, BatoError> {
    descriptor.validate()?;
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
        fast::decode(chunk.kind, &chunk.payload, n)?
    } else {
        decode_chunks(chunks, descriptor, max_output, (coding.primer, timing))?
    };
    timing.time(Step::InverseTransform, || {
        filters::inverse(&mut output, coding.transform)
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
        max_bytes: descriptor.segment_bytes(),
        timing: Timing::off(),
    };
    encode_model(raw, descriptor, run)?
        .ok_or_else(|| BatoError::CorruptStream("unbounded model encoding aborted".into()))
}
fn encode_model(
    raw: &[u8],
    descriptor: Descriptor,
    run: Run<'_>,
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
    let mut model = Predictor::new(
        primer.len() + raw.len(),
        descriptor.table_log2,
        run.dispatch,
    )?;
    let mut history = try_capacity(primer.len() + raw.len())?;
    history.extend_from_slice(primer);
    history.extend_from_slice(raw);
    run.timing
        .time(Step::PrimerTraining, || prime(&mut model, primer));
    if run.timing.enabled() {
        model.start_meter();
    }
    let coded = run.timing.time(Step::ModelTrial, || {
        encode_chunks(
            raw,
            (&history, primer.len()),
            descriptor,
            (&mut model, run.budget),
        )
    });
    model.finish_meter(run.timing);
    coded
}

fn encode_chunks(
    raw: &[u8],
    (history, primed): (&[u8], usize),
    descriptor: Descriptor,
    (model, budget): (&mut Predictor, Option<usize>),
) -> Result<Option<Vec<Chunk>>, BatoError> {
    let size = descriptor.chunk_bytes();
    let mut output = try_capacity(raw.len().div_ceil(size))?;
    let mut position = 0;
    let mut spent = 0usize;
    let mut minimum: usize = raw
        .chunks(size)
        .map(|b| crate::core::frame::block_cost(BlockType::Coded, b.len(), 0))
        .sum();
    for bytes in raw.chunks(size) {
        let base = crate::core::frame::block_cost(BlockType::Coded, bytes.len(), 0);
        let allowed = budget.map_or(bytes.len() - 1, |cap| {
            cap.saturating_sub(spent + minimum).min(bytes.len() - 1)
        });
        let mut coder = Encoder::new(allowed)?;
        for &byte in bytes {
            for shift in (0..8).rev() {
                let bit = (byte >> shift) & 1;
                let probability = model.probability(&history[..primed + position]);
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
        if budget.is_some_and(|cap| spent + minimum >= cap) {
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
        for byte in 0..=255u8 {
            let chunks = encode(&[byte], descriptor).unwrap();
            assert_eq!(decode(&chunks, descriptor, 1).unwrap(), [byte]);
        }
        assert!(encode(&vec![0; s + 1], descriptor).is_err());
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
                max_bytes: descriptor.segment_bytes(),
                timing: Timing::off(),
            };
            encode_model(&raw, descriptor, run).unwrap().unwrap()
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
        let (chunks, coding) =
            encode_segment(&mut raw, (descriptor, Trials::Parallel), Timing::off()).unwrap();
        assert_eq!(raw, original, "encoding restores its input");
        assert_eq!(coding.transform, TransformId::Deflate);
        let coded: usize = chunks.iter().map(|chunk| chunk.length as usize).sum();
        assert!(coded > original.len(), "chunks hold the inflated bytes");
        let decoded =
            decode_segment(&chunks, (descriptor, 1 << 20), coding, Timing::off()).unwrap();
        assert_eq!(decoded, original);
        let mut damaged = chunks.clone();
        damaged[0].crc ^= 1;
        assert!(decode_segment(&damaged, (descriptor, 1 << 20), coding, Timing::off()).is_err());
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
        assert!(
            decode_segment(&stored, (descriptor, 1 << 20), independent, Timing::off()).is_err()
        );
    }

    #[test]
    fn winning_segment_never_exceeds_raw_cm_or_independent_cost() {
        let descriptor = geometry();
        let mut raw = b"catalogue metadata and recurring text\\n".repeat(128);
        raw.extend_from_slice(include_bytes!("deflate/fixtures/l6.deflate"));
        let original = raw.clone();
        let independent = fast::best(&raw, None).unwrap().cost();
        let raw_model = model_candidate(
            &raw,
            (descriptor, Trials::Sequential, descriptor.segment_bytes()),
            usize::MAX,
            Timing::off(),
        )
        .unwrap()
        .unwrap()
        .0;

        let (chunks, coding) =
            encode_segment(&mut raw, (descriptor, Trials::Sequential), Timing::off()).unwrap();
        assert_eq!(raw, original, "input is unchanged after candidate trials");
        assert!(chunks_cost(&chunks) <= independent);
        assert!(chunks_cost(&chunks) <= raw_model);
        assert_eq!(
            decode_segment(
                &chunks,
                (descriptor, descriptor.segment_bytes()),
                coding,
                Timing::off()
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

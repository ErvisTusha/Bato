//! Original reversible digit-run separation and contextual integer prediction.
//! Provenance: generic delta prediction and LZ77; static ANS is described in
//! ans.rs. No JSON parser, file signature, schema, or third-party code is used.
use super::{ans, fast, varint};
use crate::core::{
    memory::try_capacity,
    types::{BatoError, BlockType},
};
const CONTEXTS: usize = 256;
const CONTEXT_BYTES: usize = 8;
const MAX_DIGITS: usize = 19;
#[derive(Default)]
struct Group {
    values: Vec<u64>,
    zeroes: Vec<u8>,
}
fn context(bytes: &[u8], at: usize) -> usize {
    let mut begin = at;
    while begin > 0 && at - begin < CONTEXT_BYTES && !bytes[begin - 1].is_ascii_digit() {
        begin -= 1;
    }
    let mut h = 2166136261u32;
    for &b in &bytes[begin..at] {
        h = (h ^ u32::from(b)).wrapping_mul(16777619);
    }
    h as usize & (CONTEXTS - 1)
}
fn end_of_number(bytes: &[u8], at: usize) -> usize {
    let mut end = at + 1;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    end
}
fn value(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0, |n, &b| n * 10 + u64::from(b - b'0'))
}
fn digits(mut n: u64) -> usize {
    let mut count = 1;
    while n >= 10 {
        count += 1;
        n /= 10;
    }
    count
}
fn predict(mode: u8, last: u64, before: u64) -> u64 {
    match mode {
        1 => last,
        2 => last.wrapping_mul(2).wrapping_sub(before),
        _ => 0,
    }
}
fn fold(n: u64) -> u64 {
    (n << 1) ^ (0u64.wrapping_sub(n >> 63))
}
fn unfold(n: u64) -> u64 {
    (n >> 1) ^ (0u64.wrapping_sub(n & 1))
}

pub(super) fn encode(raw: &[u8]) -> Result<Option<Vec<u8>>, BatoError> {
    let mut counts = [0usize; CONTEXTS];
    let mut at = 0;
    let mut skel_len = 0;
    while at < raw.len() {
        if raw[at].is_ascii_digit() {
            let end = end_of_number(raw, at);
            if end - at <= MAX_DIGITS {
                counts[context(raw, at)] += 1;
                skel_len += 2;
            } else {
                skel_len += end - at;
            }
            at = end;
        } else {
            skel_len += if raw[at] == 0 { 2 } else { 1 };
            at += 1;
        }
    }
    if counts.iter().sum::<usize>() < 2 {
        return Ok(None);
    }
    let mut groups: [Group; CONTEXTS] = std::array::from_fn(|_| Group::default());
    for (g, &n) in groups.iter_mut().zip(&counts) {
        g.values = try_capacity(n)?;
        g.zeroes = try_capacity(n)?;
    }
    let mut skeleton = try_capacity(skel_len)?;
    at = 0;
    while at < raw.len() {
        if !raw[at].is_ascii_digit() {
            let b = raw[at];
            skeleton.push(b);
            if b == 0 {
                skeleton.push(0);
            }
            at += 1;
            continue;
        }
        let end = end_of_number(raw, at);
        if end - at > MAX_DIGITS {
            skeleton.extend_from_slice(&raw[at..end]);
            at = end;
            continue;
        }
        let n = value(&raw[at..end]);
        let g = &mut groups[context(raw, at)];
        g.values.push(n);
        g.zeroes.push((end - at - digits(n)) as u8);
        skeleton.extend_from_slice(&[0, 1]);
        at = end;
    }
    let skel = fast::unfiltered(&skeleton)?;
    let mut out = Vec::new();
    varint::push(&mut out, skeleton.len());
    out.push(skel.kind as u8);
    varint::push(&mut out, skel.payload.len());
    out.extend_from_slice(&skel.payload);
    drop(skeleton);
    drop(skel);
    varint::push(&mut out, counts.iter().filter(|&&n| n > 0).count());
    for (id, g) in groups.iter().enumerate() {
        if g.values.is_empty() {
            continue;
        }
        let mut selected: Option<(u8, Vec<u8>)> = None;
        for mode in 0..=2 {
            let mut low = try_capacity(g.values.len())?;
            let mut high = Vec::new();
            let (mut last, mut before) = (0u64, 0u64);
            for &n in &g.values {
                let residual = n.wrapping_sub(predict(mode, last, before));
                let coded = if mode == 0 { n } else { fold(residual) };
                low.push((coded & 127) as u8);
                varint::write(&mut high, coded >> 7)?;
                before = last;
                last = n;
            }
            let mut data = ans::packet(&low)?;
            data.extend_from_slice(&ans::packet(&high)?);
            if selected
                .as_ref()
                .is_none_or(|(_, best)| data.len() < best.len())
            {
                selected = Some((mode, data));
            }
        }
        let (mode, data) = selected.ok_or_else(invalid)?;
        out.extend_from_slice(&[id as u8, mode]);
        out.extend_from_slice(&data);
        out.extend_from_slice(&ans::packet(&g.zeroes)?);
    }
    Ok(Some(out))
}
struct Restore<'a> {
    mode: u8,
    low: ans::Part<'a>,
    high: ans::Part<'a>,
    zeroes: ans::Part<'a>,
    last: u64,
    before: u64,
}
impl Restore<'_> {
    fn append(&mut self, out: &mut Vec<u8>, limit: usize) -> Result<(), BatoError> {
        let low = self.low.byte()?;
        if low >= 128 {
            return Err(invalid());
        }
        let high = self.high.integer()?;
        if high > u64::MAX >> 7 {
            return Err(invalid());
        }
        let coded = (high << 7) | u64::from(low);
        let n = if self.mode == 0 {
            coded
        } else {
            predict(self.mode, self.last, self.before).wrapping_add(unfold(coded))
        };
        self.before = self.last;
        self.last = n;
        let zeroes = self.zeroes.byte()? as usize;
        let mut text = [0u8; 20];
        let mut at = text.len();
        let mut v = n;
        loop {
            at -= 1;
            text[at] = b'0' + (v % 10) as u8;
            v /= 10;
            if v == 0 {
                break;
            }
        }
        let len = zeroes + text.len() - at;
        if len > MAX_DIGITS || len > limit.saturating_sub(out.len()) {
            return Err(invalid());
        }
        out.resize(out.len() + zeroes, b'0');
        out.extend_from_slice(&text[at..]);
        Ok(())
    }
    fn finish(&self) -> Result<(), BatoError> {
        self.low.finish()?;
        self.high.finish()?;
        self.zeroes.finish()
    }
}
pub(super) fn decode(payload: &[u8], length: usize) -> Result<Vec<u8>, BatoError> {
    let mut at = 0;
    let n = varint::take(payload, &mut at)?;
    if n > length.saturating_mul(2) {
        return Err(invalid());
    }
    let &kind = payload.get(at).ok_or_else(invalid)?;
    at += 1;
    let kind = BlockType::from_u8(kind)?;
    if !matches!(
        kind,
        BlockType::Stored
            | BlockType::Lz
            | BlockType::Huffman
            | BlockType::LzHuffman
            | BlockType::Repeat
    ) {
        return Err(invalid());
    }
    let c = varint::take(payload, &mut at)?;
    let end = at.checked_add(c).ok_or_else(invalid)?;
    let skeleton = fast::decode(kind, payload.get(at..end).ok_or_else(invalid)?, n)?;
    at = end;
    let count = varint::take(payload, &mut at)?;
    if !(1..=CONTEXTS).contains(&count) {
        return Err(invalid());
    }
    let mut groups: [Option<Restore<'_>>; CONTEXTS] = std::array::from_fn(|_| None);
    let mut previous = None;
    let mut total = 0usize;
    let mut high_total = 0usize;
    let maximum_numbers = length.div_ceil(2);
    let maximum_high = length.saturating_mul(2).saturating_add(2);
    for _ in 0..count {
        let &id = payload.get(at).ok_or_else(invalid)?;
        let &mode = payload.get(at + 1).ok_or_else(invalid)?;
        at += 2;
        if previous.is_some_and(|p| id <= p) || mode > 2 {
            return Err(invalid());
        }
        previous = Some(id);
        let low = ans::Part::parse(payload, &mut at, maximum_numbers.saturating_sub(total))?;
        let tokens = low.remaining();
        if tokens == 0 {
            return Err(invalid());
        }
        total += tokens;
        // A residual depends on at most three neighboring 1..19 digit
        // values. Summed high groups fit in twice the original byte count;
        // cap the aggregate before expanding any auxiliary LZ packets.
        let high = ans::Part::parse(
            payload,
            &mut at,
            tokens
                .saturating_mul(9)
                .min(maximum_high.saturating_sub(high_total)),
        )?;
        high_total += high.remaining();
        let zeroes = ans::Part::parse(payload, &mut at, tokens)?;
        if high.remaining() < tokens || zeroes.remaining() != tokens {
            return Err(invalid());
        }
        groups[id as usize] = Some(Restore {
            mode,
            low,
            high,
            zeroes,
            last: 0,
            before: 0,
        });
    }
    if at != payload.len() {
        return Err(invalid());
    }
    let mut out = try_capacity(length)?;
    let mut at = 0;
    while at < skeleton.len() {
        let end = skeleton[at..]
            .iter()
            .position(|&b| b == 0)
            .map_or(skeleton.len(), |n| at + n);
        if end - at > length.saturating_sub(out.len()) {
            return Err(invalid());
        }
        out.extend_from_slice(&skeleton[at..end]);
        at = end;
        if at == skeleton.len() {
            break;
        }
        let &escape = skeleton.get(at + 1).ok_or_else(invalid)?;
        at += 2;
        match escape {
            0 if out.len() < length => out.push(0),
            1 => {
                let id = context(&out, out.len());
                groups[id]
                    .as_mut()
                    .ok_or_else(invalid)?
                    .append(&mut out, length)?;
            }
            _ => return Err(invalid()),
        }
    }
    if out.len() != length {
        return Err(invalid());
    }
    for group in groups.iter().flatten() {
        group.finish()?;
    }
    Ok(out)
}
fn invalid() -> BatoError {
    BatoError::CorruptStream("invalid numeric segment".into())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn digit_runs_are_bytes_not_a_file_format() {
        for raw in [
            b"0a0000b12\x2e34c-9999d9999999999999999999e10000000000000000000".repeat(500),
            b"invalid\xff\x00123\x000001 x0\x01".repeat(500),
            (0..=255u8).cycle().take(10000).collect(),
        ] {
            let z = encode(&raw).unwrap().unwrap();
            assert_eq!(decode(&z, raw.len()).unwrap(), raw);
            assert!(decode(&z, raw.len() - 1).is_err());
            let mut more = z.clone();
            more.push(0);
            assert!(decode(&more, raw.len()).is_err());
            for cut in 0..z.len().min(100) {
                assert!(decode(&z[..cut], raw.len()).is_err());
            }
        }
    }
    #[test]
    fn candidate_allocation_stays_within_phase_bound() {
        let mut state = 0x192753a4u64;
        let random: Vec<u8> = (0..65536)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect();
        for raw in [
            b"x9".repeat(32768),
            b"n=9999999999999999999 small=0\n".repeat(3000),
            random,
        ] {
            let (result, stats) = crate::core::diagnostics::capture(|| encode(&raw));
            result.unwrap();
            assert!(stats.heap.peak < 18 * raw.len() + 8 * 1024 * 1024);
        }
    }
    #[test]
    fn varying_integer_widths_cover_all_predictors() {
        let mut state = 0x92745fe1u64;
        let values: Vec<u64> = (0..2000)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state % 10u64.pow(1 + i % 19)
            })
            .collect();
        let raw = values
            .iter()
            .map(|v| format!("\nrow={v}"))
            .collect::<String>()
            .into_bytes();
        let skeleton = b"\nrow=\0\x01".repeat(values.len());
        let skel = fast::unfiltered(&skeleton).unwrap();
        for mode in 0..=2 {
            let mut low = Vec::new();
            let mut high = Vec::new();
            let (mut last, mut before) = (0u64, 0u64);
            for &n in &values {
                let v = if mode == 0 {
                    n
                } else {
                    fold(n.wrapping_sub(predict(mode, last, before)))
                };
                low.push((v & 127) as u8);
                varint::write(&mut high, v >> 7).unwrap();
                before = last;
                last = n;
            }
            let mut encoded = Vec::new();
            varint::push(&mut encoded, skeleton.len());
            encoded.push(skel.kind as u8);
            varint::push(&mut encoded, skel.payload.len());
            encoded.extend_from_slice(&skel.payload);
            varint::push(&mut encoded, 1);
            encoded.extend_from_slice(&[context(b"\nrow=", 5) as u8, mode]);
            encoded.extend(ans::packet(&low).unwrap());
            encoded.extend(ans::packet(&high).unwrap());
            encoded.extend(ans::packet(&vec![0; values.len()]).unwrap());
            assert_eq!(decode(&encoded, raw.len()).unwrap(), raw);
        }
    }
    #[test]
    fn original_decimal_prediction_and_unsigned_wrap() {
        for n in [0, 1, u64::MAX, 1 << 63, 1 << 62] {
            assert_eq!(unfold(fold(n)), n);
        }
        let raw = (0..10000)
            .map(|i| format!("row={i} latency={} zeros=00000 end\n", (i * 379) % 401))
            .collect::<String>()
            .into_bytes();
        let z = encode(&raw).unwrap().unwrap();
        assert_eq!(decode(&z, raw.len()).unwrap(), raw);
    }
}

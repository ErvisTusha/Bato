//! Bato-next: an experimental from-scratch compressor.
//!
//! Method: each block of up to 16 MiB is block-sorted (Burrows-Wheeler
//! transform), then the sorted bytes are coded bit by bit with a small
//! context-mixing model and a binary arithmetic coder. Block sorting moves
//! the expensive work to the packer; the unpacker runs a light model and one
//! inverse transform, and blocks are independent, so they run on all cores.
//!
//! Provenance: the Burrows-Wheeler transform (Burrows and Wheeler, 1994),
//! suffix sorting by prefix doubling (Manber and Myers, 1990), logistic
//! mixing and adaptive probability maps as described in Mahoney's Data
//! Compression Explained. The model, formats and Rust are original.
//! Integer arithmetic only, so archives are identical on every machine.

use std::io::{Read, Write};

const MAGIC: &[u8; 4] = b"BNX1";
const BLOCK: usize = 1 << 24;

// ---------- checksum (CRC-32, IEEE polynomial) ----------

fn crc32(bytes: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *slot = c;
    }
    !bytes
        .iter()
        .fold(!0u32, |c, &b| table[((c ^ u32::from(b)) & 255) as usize] ^ (c >> 8))
}

// ---------- block sorting ----------

/// Suffix array by prefix doubling; a suffix that ends sorts before any
/// longer one, as if a smallest sentinel followed the text.
fn suffix_array(text: &[u8]) -> Vec<u32> {
    let n = text.len();
    let mut sa: Vec<u32> = (0..n as u32).collect();
    sa.sort_unstable_by_key(|&i| text[i as usize]);
    let mut rank = vec![0u32; n];
    let mut next = vec![0u32; n];
    let mut groups = 0u32;
    for j in 0..n {
        if j > 0 && text[sa[j] as usize] != text[sa[j - 1] as usize] {
            groups = j as u32;
        }
        rank[sa[j] as usize] = groups;
    }
    let mut k = 1usize;
    loop {
        let key = |rank: &[u32], i: u32| -> u32 {
            let at = i as usize + k;
            if at < n {
                rank[at] + 1
            } else {
                0
            }
        };
        let mut start = 0;
        while start < n {
            let group = rank[sa[start] as usize];
            let mut end = start + 1;
            while end < n && rank[sa[end] as usize] == group {
                end += 1;
            }
            if end - start > 1 {
                sa[start..end].sort_unstable_by_key(|&i| key(&rank, i));
            }
            start = end;
        }
        // New ranks: a group's rank is the index of its first member.
        let mut first = 0usize;
        for j in 0..n {
            if j > 0
                && (rank[sa[j] as usize] != rank[sa[j - 1] as usize]
                    || key(&rank, sa[j]) != key(&rank, sa[j - 1]))
            {
                first = j;
            }
            next[sa[j] as usize] = first as u32;
        }
        let sorted = (0..n).all(|j| next[sa[j] as usize] == j as u32);
        std::mem::swap(&mut rank, &mut next);
        if sorted || k >= n {
            return sa;
        }
        k *= 2;
    }
}

/// The last column of the sorted suffixes, without the sentinel, and the row
/// at which the sentinel would stand.
fn forward_bwt(text: &[u8]) -> (Vec<u8>, u32) {
    let sa = suffix_array(text);
    let mut last = Vec::with_capacity(text.len());
    let mut primary = 0u32;
    last.push(text[text.len() - 1]);
    for (row, &start) in sa.iter().enumerate() {
        if start == 0 {
            primary = row as u32 + 1;
        } else {
            last.push(text[start as usize - 1]);
        }
    }
    (last, primary)
}

fn inverse_bwt(last: &[u8], primary: u32) -> Option<Vec<u8>> {
    let n = last.len();
    let primary = primary as usize;
    if primary == 0 || primary > n {
        return None;
    }
    let mut counts = [0u32; 256];
    for &b in last {
        counts[usize::from(b)] += 1;
    }
    let mut total = 1u32; // row 0 is the sentinel's own suffix
    for count in &mut counts {
        let c = *count;
        *count = total;
        total += c;
    }
    // Row r of the last column: bytes before `primary` sit at r, later ones
    // at r - 1; row `primary` holds the sentinel.
    let mut step = vec![0u32; n + 1];
    for (i, &b) in last.iter().enumerate() {
        let row = if i < primary { i } else { i + 1 };
        step[row] = counts[usize::from(b)];
        counts[usize::from(b)] += 1;
    }
    let mut text = vec![0u8; n];
    let mut row = 0usize;
    for at in (0..n).rev() {
        if row == primary {
            return None;
        }
        text[at] = last[if row < primary { row } else { row - 1 }];
        row = step[row] as usize;
    }
    (row == primary).then_some(text)
}

// ---------- model ----------

const SQUASH: [i32; 33] = [
    1, 2, 4, 6, 10, 17, 27, 45, 74, 120, 194, 311, 488, 747, 1102, 1546, 2048, 2550, 2994, 3349,
    3608, 3785, 3902, 3976, 4022, 4051, 4069, 4079, 4086, 4090, 4092, 4094, 4095,
];

fn squash(d: i32) -> i32 {
    let d = d.clamp(-2047, 2047);
    let w = d & 127;
    let i = ((d >> 7) + 16) as usize;
    (SQUASH[i] * (128 - w) + SQUASH[i + 1] * w + 64) >> 7
}

const INPUTS: usize = 11;

struct Model {
    stretch: Vec<i16>,
    order0: Vec<u32>,
    order1: Vec<u32>,
    order2: Vec<u32>,
    runs: Vec<u32>,
    order3: Vec<u32>,
    weights_a: Vec<i32>,
    weights_b: Vec<i32>,
    apm_a: Vec<u16>,
    apm_b: Vec<u16>,
    c1: usize,
    c2: usize,
    c3: usize,
    run: usize,
    slots: [usize; 5],
    inputs: [i32; INPUTS],
    sets: [usize; 2],
    mixed: [i32; 2],
    apm_at: [usize; 2],
    apm_weight: [i32; 2],
    stretched: i32,
}

/// Two probabilities per context in one word: a fast one (high half) and a
/// slow one (low half), both scaled to 16 bits.
const FRESH: u32 = 0x8000_0000;
/// Largest observation count a context keeps; its rate settles at 1/LIMIT.
const LIMIT: u32 = 8;

impl Model {
    fn new() -> Self {
        let mut stretch = vec![0i16; 4096];
        let mut filled = 0usize;
        for d in -2047..=2047 {
            let v = squash(d) as usize;
            for slot in &mut stretch[filled..=v] {
                *slot = d as i16;
            }
            filled = v + 1;
        }
        for slot in &mut stretch[filled..] {
            *slot = 2047;
        }
        let apm = |contexts: usize| -> Vec<u16> {
            (0..contexts * 33)
                .map(|i| (squash((i as i32 % 33 - 16) * 128) * 16) as u16)
                .collect()
        };
        Self {
            stretch,
            order0: vec![FRESH; 256],
            order1: vec![FRESH; 1 << 16],
            order2: vec![FRESH; 1 << 24],
            runs: vec![FRESH; 1 << 19],
            order3: vec![FRESH; 1 << 22],
            weights_a: vec![6553; 256 * INPUTS],
            weights_b: vec![6553; 2048 * INPUTS],
            apm_a: apm(1 << 16),
            apm_b: apm(1 << 11),
            c1: 0,
            c2: 0,
            c3: 0,
            run: 0,
            slots: [0; 5],
            inputs: [0; INPUTS],
            sets: [0; 2],
            mixed: [0; 2],
            apm_at: [0; 2],
            apm_weight: [0; 2],
            stretched: 0,
        }
    }

    /// Run length of the previous byte in eight classes.
    fn run_class(&self) -> usize {
        match self.run {
            0..=3 => self.run,
            4..=7 => 4,
            8..=15 => 5,
            16..=63 => 6,
            _ => 7,
        }
    }

    /// Probability (12 bits) that the next bit is one, after `partial`.
    fn predict(&mut self, partial: usize) -> i32 {
        let run = self.run_class();
        let hash3 = (self.c3.wrapping_mul(0x9e5) ^ self.c2.wrapping_mul(0x3d) ^ self.c1) & 0x3fff;
        self.slots = [
            partial,
            self.c1 << 8 | partial,
            (self.c2 << 8 | self.c1) << 8 | partial,
            (run << 8 | self.c1) << 8 | partial,
            hash3 << 8 | partial,
        ];
        let tables = [&self.order0, &self.order1, &self.order2, &self.runs, &self.order3];
        for (i, (table, &slot)) in tables.iter().zip(&self.slots).enumerate() {
            let word = table[slot];
            let p = (word >> 20) as i32;
            self.inputs[2 * i] = i32::from(self.stretch[p as usize]);
            // Confidence grows with the observations behind the estimate.
            let seen = (word & 1023).min(LIMIT) as i32;
            self.inputs[2 * i + 1] = (p - 2048) * seen.min(32) >> 6;
        }
        self.inputs[INPUTS - 1] = 256;
        self.sets = [partial * INPUTS, (run << 8 | partial) * INPUTS];
        let dot = |weights: &[i32], inputs: &[i32; INPUTS]| -> i32 {
            let sum: i64 = weights
                .iter()
                .zip(inputs)
                .map(|(&w, &x)| i64::from(w) * i64::from(x))
                .sum();
            (sum >> 16) as i32
        };
        self.mixed = [
            squash(dot(&self.weights_a[self.sets[0]..self.sets[0] + INPUTS], &self.inputs)),
            squash(dot(&self.weights_b[self.sets[1]..self.sets[1] + INPUTS], &self.inputs)),
        ];
        let mixed = (self.mixed[0] + self.mixed[1] + 1) >> 1;
        self.stretched = i32::from(self.stretch[mixed as usize]);
        let position = self.stretched + 2048;
        let (low, weight) = ((position >> 7) as usize, position & 127);
        self.apm_weight = [128 - weight, weight];
        self.apm_at = [
            (self.c1 << 8 | partial) * 33 + low,
            (run << 8 | partial) * 33 + low,
        ];
        let refine = |table: &[u16], at: usize, weight: [i32; 2]| -> i32 {
            (i32::from(table[at]) * weight[0] + i32::from(table[at + 1]) * weight[1]) >> 11
        };
        let a = refine(&self.apm_a, self.apm_at[0], self.apm_weight);
        let b = refine(&self.apm_b, self.apm_at[1], self.apm_weight);
        ((2 * mixed + a + b + 2) >> 2).clamp(1, 4095)
    }

    fn update(&mut self, bit: usize) {
        let target = (bit as u32) * 0x3f_ffff;
        let tables = [
            &mut self.order0,
            &mut self.order1,
            &mut self.order2,
            &mut self.runs,
            &mut self.order3,
        ];
        for (table, &slot) in tables.into_iter().zip(&self.slots) {
            let word = table[slot];
            let (p, seen) = ((word >> 10) as i64, word & 1023);
            // Rate 1 / (seen + 1.5): an average at first, a slow tracker later.
            let p = p + (((target as i64) - p) * 2 / (2 * i64::from(seen) + 3));
            table[slot] = (p as u32) << 10 | (seen + 1).min(LIMIT);
        }
        for (weights, (&set, &mixed)) in [&mut self.weights_a, &mut self.weights_b]
            .into_iter()
            .zip(self.sets.iter().zip(&self.mixed))
        {
            let error = (((bit as i32) << 12) - mixed) * 6;
            for (w, &x) in weights[set..set + INPUTS].iter_mut().zip(&self.inputs) {
                *w += (x * error + 0x8000) >> 16;
            }
        }
        let goal = (bit as i32) * 65535;
        for (table, &at) in [&mut self.apm_a, &mut self.apm_b].into_iter().zip(&self.apm_at) {
            for (offset, &weight) in self.apm_weight.iter().enumerate() {
                let old = i32::from(table[at + offset]);
                table[at + offset] = (old + (((goal - old) * weight) >> 13)) as u16;
            }
        }
    }

    fn end_byte(&mut self, byte: usize) {
        self.run = if byte == self.c1 { self.run + 1 } else { 0 };
        self.c3 = self.c2;
        self.c2 = self.c1;
        self.c1 = byte;
    }
}

// ---------- arithmetic coder ----------

fn encode_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut model = Model::new();
    let (mut low, mut high) = (0u32, u32::MAX);
    let mut out = Vec::with_capacity(bytes.len() / 3 + 16);
    for &byte in bytes {
        let mut partial = 1usize;
        for shift in (0..8).rev() {
            let bit = usize::from(byte >> shift) & 1;
            let p = model.predict(partial) as u64;
            let middle = low + ((u64::from(high - low) * p) >> 12) as u32;
            if bit == 1 {
                high = middle;
            } else {
                low = middle + 1;
            }
            while (low ^ high) & 0xff00_0000 == 0 {
                out.push((high >> 24) as u8);
                low <<= 8;
                high = high << 8 | 255;
            }
            model.update(bit);
            partial = partial << 1 | bit;
        }
        model.end_byte(usize::from(byte));
    }
    out.extend_from_slice(&low.to_be_bytes());
    out
}

fn decode_bytes(payload: &[u8], length: usize) -> Vec<u8> {
    let mut model = Model::new();
    let (mut low, mut high) = (0u32, u32::MAX);
    let mut at = 0usize;
    let mut take = || {
        let byte = payload.get(at).copied().unwrap_or(0);
        at += 1;
        u32::from(byte)
    };
    let mut code = 0u32;
    for _ in 0..4 {
        code = code << 8 | take();
    }
    let mut out = Vec::with_capacity(length);
    for _ in 0..length {
        let mut partial = 1usize;
        for _ in 0..8 {
            let p = model.predict(partial) as u64;
            let middle = low + ((u64::from(high - low) * p) >> 12) as u32;
            let bit = usize::from(code <= middle);
            if bit == 1 {
                high = middle;
            } else {
                low = middle + 1;
            }
            while (low ^ high) & 0xff00_0000 == 0 {
                low <<= 8;
                high = high << 8 | 255;
                code = code << 8 | take();
            }
            model.update(bit);
            partial = partial << 1 | bit;
        }
        model.end_byte(partial & 255);
        out.push(partial as u8);
    }
    out
}

// ---------- reversible filters ----------

/// A filter applied to a block before sorting. The packer tries several and
/// keeps the one that gives the smallest block; the unpacker inverts the one
/// the block declares.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Filter {
    None,
    /// Each byte minus the byte `distance` back.
    Subtract(u32),
    /// Each byte exclusive-or the byte `distance` back (bit images).
    Xor(u32),
    /// Relative x86 call/jump targets made absolute.
    Calls,
}

impl Filter {
    fn code(self) -> (u8, u32) {
        match self {
            Self::None => (0, 0),
            Self::Subtract(distance) => (1, distance),
            Self::Xor(distance) => (2, distance),
            Self::Calls => (3, 0),
        }
    }

    fn from_code(kind: u8, distance: u32) -> Option<Self> {
        match (kind, distance) {
            (0, 0) => Some(Self::None),
            (1, 1..) => Some(Self::Subtract(distance)),
            (2, 1..) => Some(Self::Xor(distance)),
            (3, 0) => Some(Self::Calls),
            _ => None,
        }
    }

    fn call_site(bytes: &[u8], i: usize) -> bool {
        bytes[i] & 0xfe == 0xe8 && bytes[i + 4].wrapping_add(1) & 0xfe == 0
    }

    fn shift_call(bytes: &mut [u8], i: usize, forward: bool) {
        let value = u32::from(bytes[i + 1]) | u32::from(bytes[i + 2]) << 8 | u32::from(bytes[i + 3]) << 16;
        let at = i as u32;
        let value = if forward { value.wrapping_add(at) } else { value.wrapping_sub(at) };
        bytes[i + 1] = value as u8;
        bytes[i + 2] = (value >> 8) as u8;
        bytes[i + 3] = (value >> 16) as u8;
    }

    fn forward(self, block: &[u8]) -> Vec<u8> {
        let mut out = block.to_vec();
        match self {
            Self::None => {}
            Self::Subtract(distance) => {
                let d = distance as usize;
                for i in d..block.len() {
                    out[i] = block[i].wrapping_sub(block[i - d]);
                }
            }
            Self::Xor(distance) => {
                let d = distance as usize;
                for i in d..block.len() {
                    out[i] = block[i] ^ block[i - d];
                }
            }
            // Descending, so each test sees the bytes the inverse will see.
            Self::Calls => {
                for i in (0..block.len().saturating_sub(4)).rev() {
                    if Self::call_site(&out, i) {
                        Self::shift_call(&mut out, i, true);
                    }
                }
            }
        }
        out
    }

    fn inverse(self, block: &mut [u8]) {
        match self {
            Self::None => {}
            Self::Subtract(distance) => {
                let d = distance as usize;
                for i in d..block.len() {
                    block[i] = block[i].wrapping_add(block[i - d]);
                }
            }
            Self::Xor(distance) => {
                let d = distance as usize;
                for i in d..block.len() {
                    block[i] ^= block[i - d];
                }
            }
            Self::Calls => {
                for i in 0..block.len().saturating_sub(4) {
                    if Self::call_site(block, i) {
                        Self::shift_call(block, i, false);
                    }
                }
            }
        }
    }
}

/// The distance, up to 4096, at which sampled bytes most often equal the
/// earlier byte: the record length or image row of the block, if it has one.
fn period(block: &[u8]) -> Option<u32> {
    const MAX: usize = 4096;
    if block.len() < 4 * MAX {
        return None;
    }
    let mut equal = vec![0u32; MAX + 1];
    let step = (block.len() / 4096).max(1);
    for i in (MAX..block.len()).step_by(step) {
        let current = block[i];
        for (distance, count) in equal.iter_mut().enumerate().skip(5) {
            *count += u32::from(block[i - distance] == current);
        }
    }
    let best = *equal.iter().max()?;
    // Multiples of the period score alike; the shortest is the period.
    let distance = equal.iter().position(|&count| count * 16 >= best * 15)?;
    (best >= 64).then_some(distance as u32)
}

// ---------- container ----------

fn pack_block(block: &[u8]) -> Vec<u8> {
    let mut filters = vec![
        Filter::None,
        Filter::Calls,
        Filter::Subtract(1),
        Filter::Subtract(2),
        Filter::Subtract(3),
        Filter::Subtract(4),
    ];
    if let Some(distance) = period(block) {
        filters.extend([Filter::Subtract(distance), Filter::Xor(distance)]);
    }
    // Every filter is tried in full; the smallest coded block wins and the
    // first of equals is kept.
    let (filter, primary, coded) = filters
        .into_iter()
        .map(|filter| {
            let (last, primary) = forward_bwt(&filter.forward(block));
            (filter, primary, encode_bytes(&last))
        })
        .min_by_key(|(_, _, coded)| coded.len())
        .expect("the filter list is not empty");
    let mut out = Vec::with_capacity(coded.len() + 22);
    out.extend_from_slice(&(block.len() as u32).to_le_bytes());
    out.extend_from_slice(&crc32(block).to_le_bytes());
    if coded.len() < block.len() {
        out.push(1);
        let (kind, distance) = filter.code();
        out.push(kind);
        out.extend_from_slice(&distance.to_le_bytes());
        out.extend_from_slice(&primary.to_le_bytes());
        out.extend_from_slice(&(coded.len() as u32).to_le_bytes());
        out.extend_from_slice(&coded);
    } else {
        out.push(0); // stored: data that does not shrink
        out.extend_from_slice(block);
    }
    out
}

fn word(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

/// Splits the archive into its blocks: (length, checksum, filter and sort
/// row if sorted, payload).
#[allow(clippy::type_complexity)]
fn blocks(archive: &[u8]) -> Option<Vec<(usize, u32, Option<(Filter, u32)>, &[u8])>> {
    let mut found = Vec::new();
    let mut at = 4;
    while at < archive.len() {
        let length = word(archive, at)? as usize;
        let checksum = word(archive, at + 4)?;
        let kind = *archive.get(at + 8)?;
        at += 9;
        if length == 0 || length > BLOCK {
            return None;
        }
        let (sorted, size) = match kind {
            0 => (None, length),
            1 => {
                let filter = Filter::from_code(*archive.get(at)?, word(archive, at + 1)?)?;
                let header = (Some((filter, word(archive, at + 5)?)), word(archive, at + 9)? as usize);
                at += 13;
                header
            }
            _ => return None,
        };
        found.push((length, checksum, sorted, archive.get(at..at + size)?));
        at += size;
    }
    Some(found)
}

fn unpack_block(
    (length, checksum, sorted, payload): (usize, u32, Option<(Filter, u32)>, &[u8]),
) -> Option<Vec<u8>> {
    let block = match sorted {
        None => payload.to_vec(),
        Some((filter, primary)) => {
            let mut block = inverse_bwt(&decode_bytes(payload, length), primary)?;
            filter.inverse(&mut block);
            block
        }
    };
    (crc32(&block) == checksum).then_some(block)
}

/// Runs `work` over `items` on all cores, keeping the order.
fn parallel<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<(usize, R)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads.min(items.len()))
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(item) = items.get(i) else {
                            return done;
                        };
                        done.push((i, work(item)));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("worker panicked"))
            .collect()
    });
    results.sort_unstable_by_key(|&(i, _)| i);
    results.into_iter().map(|(_, r)| r).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 || !matches!(args[1].as_str(), "c" | "d") {
        eprintln!("usage: bato-next c|d INPUT OUTPUT");
        std::process::exit(2);
    }
    let mut input = Vec::new();
    std::fs::File::open(&args[2])
        .and_then(|mut file| file.read_to_end(&mut input))
        .unwrap_or_else(|error| {
            eprintln!("cannot read {}: {error}", args[2]);
            std::process::exit(1);
        });
    let output = if args[1] == "c" {
        let pieces: Vec<&[u8]> = input.chunks(BLOCK).collect();
        let mut out = MAGIC.to_vec();
        for packed in parallel(&pieces, |piece| pack_block(piece)) {
            out.extend_from_slice(&packed);
        }
        out
    } else {
        let restored = (input.get(..4) == Some(MAGIC.as_slice()))
            .then(|| blocks(&input))
            .flatten()
            .and_then(|list| {
                parallel(&list, |&block| unpack_block(block))
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
            });
        let Some(parts) = restored else {
            eprintln!("corrupt or unsupported archive");
            std::process::exit(1);
        };
        parts.concat()
    };
    std::fs::File::create(&args[3])
        .and_then(|mut file| file.write_all(&output))
        .unwrap_or_else(|error| {
            eprintln!("cannot write {}: {error}", args[3]);
            std::process::exit(1);
        });
}

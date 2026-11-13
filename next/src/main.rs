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

const INPUTS: usize = 15;
/// Observation limits per table: quick trackers, then two slow averages of
/// the order-1 and order-2 contexts for stable regions.
const LIMITS: [u32; 7] = [LIMIT, LIMIT, LIMIT, LIMIT, LIMIT, 255, 255];

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
    slots: [usize; 7],
    slow1: Vec<u32>,
    slow2: Vec<u32>,
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
            slow1: vec![FRESH; 1 << 16],
            slow2: vec![FRESH; 1 << 24],
            weights_a: vec![6553; 256 * INPUTS],
            weights_b: vec![6553; 2048 * INPUTS],
            apm_a: apm(1 << 16),
            apm_b: apm(1 << 11),
            c1: 0,
            c2: 0,
            c3: 0,
            run: 0,
            slots: [0; 7],
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
            self.c1 << 8 | partial,
            (self.c2 << 8 | self.c1) << 8 | partial,
        ];
        let tables = [
            &self.order0,
            &self.order1,
            &self.order2,
            &self.runs,
            &self.order3,
            &self.slow1,
            &self.slow2,
        ];
        for (i, (table, &slot)) in tables.iter().zip(&self.slots).enumerate() {
            let word = table[slot];
            let p = (word >> 20) as i32;
            self.inputs[2 * i] = i32::from(self.stretch[p as usize]);
            // Confidence grows with the observations behind the estimate.
            let seen = (word & 1023) as i32;
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
            &mut self.slow1,
            &mut self.slow2,
        ];
        for ((table, &slot), limit) in tables.into_iter().zip(&self.slots).zip(LIMITS) {
            let word = table[slot];
            let (p, seen) = ((word >> 10) as i64, word & 1023);
            // Rate 1 / (seen + 1.5): an average at first, a slow tracker later.
            let p = p + (((target as i64) - p) * 2 / (2 * i64::from(seen) + 3));
            table[slot] = (p as u32) << 10 | (seen + 1).min(limit);
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

// ---------- match coding ----------
//
// The second method for a block: repeats are coded as (length, distance)
// pairs and the rest as literals, all through adaptive bit models and the
// arithmetic coder. It suits program code and mixed data, where block
// sorting is weak. Provenance: Lempel-Ziv 1977 parsing with lazy matching
// and the general literal/match/repeat-distance modelling described for
// LZMA in public documentation; the models, parser and Rust are original.

const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = MIN_MATCH + 271;
const HASH_BITS: u32 = 20;
/// Earlier positions examined per match search.
const CHAIN: usize = 256;
/// Adaptation shift of the bit models.
const RATE: u32 = 5;
/// Marks a match-coded block where a sorted block stores its row.
const MATCH_CODED: u32 = u32::MAX;

struct BitModels {
    is_match: [[u16; 4]; 2],
    is_repeat: u16,
    /// Which of the four last distances a repeat uses.
    repeat_index: [u16; 4],
    length_choice: [[u16; 2]; 2],
    length_low: [[u16; 8]; 2],
    length_mid: [[u16; 8]; 2],
    length_high: [[u16; 256]; 2],
    slot: [[u16; 64]; 4],
    special: Vec<u16>,
    align: [u16; 16],
}

impl BitModels {
    fn new() -> Self {
        Self {
            is_match: [[2048; 4]; 2],
            is_repeat: 2048,
            repeat_index: [2048; 4],
            length_choice: [[2048; 2]; 2],
            length_low: [[2048; 8]; 2],
            length_mid: [[2048; 8]; 2],
            length_high: [[2048; 256]; 2],
            slot: [[2048; 64]; 4],
            special: vec![2048; 64 * 16],
            align: [2048; 16],
        }
    }
}

/// Literal bits predicted by mixing three views: the previous byte, the two
/// previous bytes, and the byte the last match distance points at (while it
/// still agrees with the bits coded so far). Match coders usually code
/// literals from one context; mixing several is what context-mixing
/// compressors do, applied here only to the literals.
struct Literals {
    stretch: Vec<i16>,
    order1: Vec<u16>,
    order2: Vec<u16>,
    guessed: Vec<u16>,
    order3: Vec<u16>,
    aligned: Vec<u16>,
    weights: Vec<i32>,
    slots: [usize; 5],
    inputs: [i32; 6],
    set: usize,
    mixed: i32,
}

impl Literals {
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
        Self {
            stretch,
            order1: vec![2048; 1 << 16],
            order2: vec![2048; 1 << 22],
            guessed: vec![2048; 1 << 17],
            order3: vec![2048; 1 << 22],
            aligned: vec![2048; 1 << 18],
            weights: vec![1 << 14; 2048 * 6],
            slots: [0; 5],
            inputs: [0; 6],
            set: 0,
            mixed: 2048,
        }
    }

    /// Probability (12 bits) of a one after the `node` bits of this literal.
    /// `guess` is the expected byte, if the coded bits still agree with it.
    fn predict(&mut self, (c1, c2, c3, at): (usize, usize, usize, usize), guess: Option<u8>, node: usize) -> u32 {
        let pair = (c2.wrapping_mul(0x2f0b) ^ c1) & 0x3fff;
        let (agrees, expected) = match guess {
            Some(byte) => (1, usize::from(byte)),
            None => (0, 0),
        };
        let triple = (c3.wrapping_mul(0x9e5b) ^ c2.wrapping_mul(0x2f0b) ^ c1) & 0x3fff;
        self.slots = [
            c1 << 8 | node,
            pair << 8 | node,
            (agrees << 8 | expected) << 8 | node,
            triple << 8 | node,
            ((at & 3) << 8 | c1) << 8 | node,
        ];
        let tables = [&self.order1, &self.order2, &self.guessed, &self.order3, &self.aligned];
        for (input, (table, &slot)) in self.inputs.iter_mut().zip(tables.iter().zip(&self.slots)) {
            *input = i32::from(self.stretch[usize::from(table[slot])]);
        }
        self.inputs[5] = 256;
        self.set = (((at & 3) << 1 | agrees) << 8 | node) * 6;
        let dot: i64 = self.weights[self.set..self.set + 6]
            .iter()
            .zip(&self.inputs)
            .map(|(&w, &x)| i64::from(w) * i64::from(x))
            .sum();
        self.mixed = squash((dot >> 16) as i32).clamp(1, 4095);
        self.mixed as u32
    }

    fn update(&mut self, bit: u32) {
        let target = bit as i32 * 4095;
        for (table, &slot) in [
            &mut self.order1,
            &mut self.order2,
            &mut self.guessed,
            &mut self.order3,
            &mut self.aligned,
        ]
        .into_iter()
            .zip(&self.slots)
        {
            let p = i32::from(table[slot]);
            table[slot] = (p + ((target - p) >> 4)).clamp(1, 4095) as u16;
        }
        let error = ((bit as i32) << 12) - self.mixed;
        for (w, &x) in self.weights[self.set..self.set + 6].iter_mut().zip(&self.inputs) {
            *w += (x * error * 6 + 0x8000) >> 16;
        }
    }
}

/// The expected byte while its leading bits equal the bits in `node`.
fn agreeing(expected: Option<u8>, node: usize, coded: u32) -> Option<u8> {
    expected.filter(|&byte| usize::from(byte) >> (8 - coded) == node & ((1 << coded) - 1))
}

/// Distance code: values below 4 are their own slot; above, the slot holds
/// the top two bits and the count of bits that follow.
fn distance_slot(x: u32) -> u32 {
    if x < 4 {
        return x;
    }
    let top = 31 - x.leading_zeros();
    2 * top + ((x >> (top - 1)) & 1)
}

struct Encoder {
    low: u32,
    high: u32,
    out: Vec<u8>,
}

impl Encoder {
    fn code(&mut self, bit: u32, p: u32) {
        let middle = self.low + ((u64::from(self.high - self.low) * u64::from(p)) >> 12) as u32;
        if bit == 1 {
            self.high = middle;
        } else {
            self.low = middle + 1;
        }
        while (self.low ^ self.high) & 0xff00_0000 == 0 {
            self.out.push((self.high >> 24) as u8);
            self.low <<= 8;
            self.high = self.high << 8 | 255;
        }
    }

    fn bit(&mut self, model: &mut u16, bit: u32) {
        self.code(bit, u32::from(*model));
        if bit == 1 {
            *model += (4096 - *model) >> RATE;
        } else {
            *model -= *model >> RATE;
        }
    }

    fn tree(&mut self, models: &mut [u16], bits: u32, value: u32) {
        let mut node = 1usize;
        for shift in (0..bits).rev() {
            let bit = (value >> shift) & 1;
            self.bit(&mut models[node], bit);
            node = node << 1 | bit as usize;
        }
    }

    fn plain(&mut self, bits: u32, value: u32) {
        for shift in (0..bits).rev() {
            self.code((value >> shift) & 1, 2048);
        }
    }
}

struct Decoder<'a> {
    low: u32,
    high: u32,
    code: u32,
    payload: &'a [u8],
    at: usize,
}

impl<'a> Decoder<'a> {
    fn new(payload: &'a [u8]) -> Self {
        let mut decoder = Self {
            low: 0,
            high: u32::MAX,
            code: 0,
            payload,
            at: 0,
        };
        for _ in 0..4 {
            decoder.code = decoder.code << 8 | decoder.take();
        }
        decoder
    }

    fn take(&mut self) -> u32 {
        let byte = self.payload.get(self.at).copied().unwrap_or(0);
        self.at += 1;
        u32::from(byte)
    }

    fn code(&mut self, p: u32) -> u32 {
        let middle = self.low + ((u64::from(self.high - self.low) * u64::from(p)) >> 12) as u32;
        let bit = u32::from(self.code <= middle);
        if bit == 1 {
            self.high = middle;
        } else {
            self.low = middle + 1;
        }
        while (self.low ^ self.high) & 0xff00_0000 == 0 {
            self.low <<= 8;
            self.high = self.high << 8 | 255;
            self.code = self.code << 8 | self.take();
        }
        bit
    }

    fn bit(&mut self, model: &mut u16) -> u32 {
        let bit = self.code(u32::from(*model));
        if bit == 1 {
            *model += (4096 - *model) >> RATE;
        } else {
            *model -= *model >> RATE;
        }
        bit
    }

    fn tree(&mut self, models: &mut [u16], bits: u32) -> u32 {
        let mut node = 1usize;
        for _ in 0..bits {
            node = node << 1 | self.bit(&mut models[node]) as usize;
        }
        node as u32 - (1 << bits)
    }

    fn plain(&mut self, bits: u32) -> u32 {
        (0..bits).fold(0, |value, _| value << 1 | self.code(2048))
    }
}

/// Chained hash table over four-byte prefixes of the block.
struct Matches<'a> {
    bytes: &'a [u8],
    head: Vec<u32>,
    previous: Vec<u32>,
    inserted: usize,
}

impl<'a> Matches<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            head: vec![u32::MAX; 1 << HASH_BITS],
            previous: vec![u32::MAX; bytes.len()],
            inserted: 0,
        }
    }

    fn slot(&self, at: usize) -> usize {
        let key = u32::from(self.bytes[at])
            | u32::from(self.bytes[at + 1]) << 8
            | u32::from(self.bytes[at + 2]) << 16;
        (key.wrapping_mul(0x9e37_79b1) >> (32 - HASH_BITS)) as usize
    }

    /// Makes every position below `end` findable.
    fn insert_until(&mut self, end: usize) {
        while self.inserted < end && self.inserted + 3 <= self.bytes.len() {
            let slot = self.slot(self.inserted);
            self.previous[self.inserted] = self.head[slot];
            self.head[slot] = self.inserted as u32;
            self.inserted += 1;
        }
        self.inserted = self.inserted.max(end.min(self.bytes.len()));
    }

    fn common(&self, a: usize, b: usize) -> usize {
        let limit = (self.bytes.len() - b).min(MAX_MATCH);
        (0..limit)
            .find(|&k| self.bytes[a + k] != self.bytes[b + k])
            .unwrap_or(limit)
    }

    /// Longest earlier match for position `at` (length, distance); the
    /// nearest wins ties. `repeats` are the last four distances used.
    fn best(&mut self, at: usize, repeats: [usize; 4]) -> (usize, usize) {
        self.insert_until(at);
        let mut best = (0usize, 0usize);
        for repeat in repeats {
            if repeat != 0 && repeat <= at {
                let length = self.common(at - repeat, at);
                if length >= MIN_MATCH && length > best.0 {
                    best = (length, repeat);
                }
            }
        }
        if at + 3 > self.bytes.len() {
            return best;
        }
        let mut candidate = self.head[self.slot(at)];
        for _ in 0..CHAIN {
            if candidate == u32::MAX {
                break;
            }
            let from = candidate as usize;
            let length = self.common(from, at);
            // A repeat of the last distance is cheaper: keep it unless the
            // new match is clearly longer.
            let needed = if best.0 > 0 && repeats.contains(&best.1) {
                best.0 + 2
            } else {
                best.0 + 1
            };
            if length >= needed.max(MIN_MATCH) {
                best = (length, at - from);
            }
            candidate = self.previous[from];
        }
        // A short match far away costs more than its literals.
        let reach = match best.0 {
            3 => 1 << 12,
            4 => 1 << 17,
            _ => usize::MAX,
        };
        if best.1 > reach && !repeats.contains(&best.1) {
            return (0, 0);
        }
        best
    }
}

fn match_encode(bytes: &[u8]) -> Vec<u8> {
    let mut models = BitModels::new();
    let mut coder = Encoder {
        low: 0,
        high: u32::MAX,
        out: Vec::with_capacity(bytes.len() / 3 + 16),
    };
    let mut matches = Matches::new(bytes);
    let mut literals = Literals::new();
    let (mut at, mut repeats, mut after_match) = (0usize, [0usize; 4], 0usize);
    while at < bytes.len() {
        let (mut length, distance) = matches.best(at, repeats);
        if length >= MIN_MATCH && at + 1 < bytes.len() {
            // Lazy step: a longer match one byte later beats this one.
            let (next, _) = matches.best(at + 1, repeats);
            if next > length {
                length = 0;
            }
        }
        if length < MIN_MATCH {
            coder.bit(&mut models.is_match[after_match][at & 3], 0);
            let back = |k: usize| if at >= k { usize::from(bytes[at - k]) } else { 0 };
            let before = (back(1), back(2), back(3), at);
            let expected = (repeats[0] != 0).then(|| bytes[at - repeats[0]]);
            let mut node = 1usize;
            for coded in 0..8u32 {
                let bit = u32::from(bytes[at] >> (7 - coded)) & 1;
                let p = literals.predict(before, agreeing(expected, node, coded), node);
                coder.code(bit, p);
                literals.update(bit);
                node = node << 1 | bit as usize;
            }
            after_match = 0;
            at += 1;
            continue;
        }
        coder.bit(&mut models.is_match[after_match][at & 3], 1);
        let index = repeats.iter().position(|&repeat| repeat == distance);
        let repeated = usize::from(index.is_some());
        coder.bit(&mut models.is_repeat, repeated as u32);
        let index = index.unwrap_or(3);
        if repeated == 1 {
            coder.tree(&mut models.repeat_index, 2, index as u32);
        }
        let value = (length - MIN_MATCH) as u32;
        let choice = &mut models.length_choice[repeated];
        if value < 8 {
            coder.bit(&mut choice[0], 0);
            coder.tree(&mut models.length_low[repeated], 3, value);
        } else if value < 16 {
            coder.bit(&mut choice[0], 1);
            coder.bit(&mut choice[1], 0);
            coder.tree(&mut models.length_mid[repeated], 3, value - 8);
        } else {
            coder.bit(&mut choice[0], 1);
            coder.bit(&mut choice[1], 1);
            coder.tree(&mut models.length_high[repeated], 8, value - 16);
        }
        if repeated == 0 {
            let x = distance as u32 - 1;
            let slot = distance_slot(x);
            coder.tree(&mut models.slot[(value as usize).min(3)], 6, slot);
            if slot >= 4 {
                let extra = (slot >> 1) - 1;
                let rest = x - ((2 | (slot & 1)) << extra);
                if extra <= 4 {
                    let base = slot as usize * 16;
                    coder.tree(&mut models.special[base..base + 16], extra, rest);
                } else {
                    coder.plain(extra - 4, rest >> 4);
                    coder.tree(&mut models.align, 4, rest & 15);
                }
            }
        }
        // The distance moves to the front of the last four.
        repeats.copy_within(0..index, 1);
        repeats[0] = distance;
        after_match = 1;
        at += length;
    }
    coder.out.extend_from_slice(&coder.low.to_be_bytes());
    coder.out
}

fn match_decode(payload: &[u8], length: usize) -> Option<Vec<u8>> {
    let mut models = BitModels::new();
    let mut coder = Decoder::new(payload);
    let mut literals = Literals::new();
    let mut out: Vec<u8> = Vec::with_capacity(length);
    let (mut repeats, mut after_match) = ([0usize; 4], 0usize);
    while out.len() < length {
        if coder.bit(&mut models.is_match[after_match][out.len() & 3]) == 0 {
            let n = out.len();
            let back = |k: usize| if n >= k { usize::from(out[n - k]) } else { 0 };
            let before = (back(1), back(2), back(3), n);
            let expected = (repeats[0] != 0).then(|| out[n - repeats[0]]);
            let mut node = 1usize;
            for coded in 0..8u32 {
                let p = literals.predict(before, agreeing(expected, node, coded), node);
                let bit = coder.code(p);
                literals.update(bit);
                node = node << 1 | bit as usize;
            }
            let byte = node as u8;
            out.push(byte);
            after_match = 0;
            continue;
        }
        let repeated = coder.bit(&mut models.is_repeat) as usize;
        let index = if repeated == 1 {
            coder.tree(&mut models.repeat_index, 2) as usize
        } else {
            3
        };
        let mut repeat = repeats[index];
        let choice = &mut models.length_choice[repeated];
        let value = if coder.bit(&mut choice[0]) == 0 {
            coder.tree(&mut models.length_low[repeated], 3)
        } else if coder.bit(&mut choice[1]) == 0 {
            8 + coder.tree(&mut models.length_mid[repeated], 3)
        } else {
            16 + coder.tree(&mut models.length_high[repeated], 8)
        };
        if repeated == 0 {
            let slot = coder.tree(&mut models.slot[(value as usize).min(3)], 6);
            let x = if slot < 4 {
                slot
            } else {
                let extra = (slot >> 1) - 1;
                let rest = if extra <= 4 {
                    let base = slot as usize * 16;
                    coder.tree(&mut models.special[base..base + 16], extra)
                } else {
                    let high = coder.plain(extra - 4);
                    high << 4 | coder.tree(&mut models.align, 4)
                };
                ((2 | (slot & 1)) << extra).checked_add(rest)?
            };
            repeat = x as usize + 1;
        }
        let count = value as usize + MIN_MATCH;
        if repeat == 0 || repeat > out.len() || out.len() + count > length {
            return None;
        }
        repeats.copy_within(0..index, 1);
        repeats[0] = repeat;
        for _ in 0..count {
            out.push(out[out.len() - repeat]);
        }
        after_match = 1;
    }
    Some(out)
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
    /// Records of `distance` bytes stored column by column.
    Columns(u32),
    /// `Subtract(distance)`, then `Columns(distance)`.
    ColumnDeltas(u32),
}

impl Filter {
    fn code(self) -> (u8, u32) {
        match self {
            Self::None => (0, 0),
            Self::Subtract(distance) => (1, distance),
            Self::Xor(distance) => (2, distance),
            Self::Calls => (3, 0),
            Self::Columns(distance) => (4, distance),
            Self::ColumnDeltas(distance) => (5, distance),
        }
    }

    fn from_code(kind: u8, distance: u32) -> Option<Self> {
        match (kind, distance) {
            (0, 0) => Some(Self::None),
            (1, 1..) => Some(Self::Subtract(distance)),
            (2, 1..) => Some(Self::Xor(distance)),
            (3, 0) => Some(Self::Calls),
            (4, 2..) => Some(Self::Columns(distance)),
            (5, 2..) => Some(Self::ColumnDeltas(distance)),
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

    /// Position in the column order of byte `i` of `len` bytes in records of
    /// `d`: column by column, each column in record order.
    fn column_order(len: usize, d: usize) -> impl Iterator<Item = usize> {
        (0..d).flat_map(move |column| (column..len).step_by(d))
    }

    fn forward(self, block: &[u8]) -> Vec<u8> {
        let mut out = block.to_vec();
        match self {
            Self::None => {}
            Self::Columns(distance) => {
                for (slot, from) in out.iter_mut().zip(Self::column_order(block.len(), distance as usize)) {
                    *slot = block[from];
                }
            }
            Self::ColumnDeltas(distance) => {
                return Self::Columns(distance).forward(&Self::Subtract(distance).forward(block));
            }
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
            Self::Columns(distance) => {
                let columns = block.to_vec();
                for (&byte, to) in columns.iter().zip(Self::column_order(block.len(), distance as usize)) {
                    block[to] = byte;
                }
            }
            Self::ColumnDeltas(distance) => {
                Self::Columns(distance).inverse(block);
                Self::Subtract(distance).inverse(block);
            }
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

// ---------- far repeats ----------

/// Shortest repeat replaced by a reference; shorter ones are left to the
/// block sort, which handles them better inside one block.
const MIN_REPEAT: usize = 2048;
/// Bytes hashed to find a repeat.
const WINDOW: usize = 32;
/// Every this many positions enter the table.
const SAMPLE: usize = 4;
const TABLE_BITS: u32 = 24;
/// Distance beyond which a repeat's source lies outside the reach of the
/// block that holds it, and the shortest repeat replaced at that distance.
const FAR_DISTANCE: usize = 1 << 20;
const FAR_REPEAT: usize = 40;

fn put_number(out: &mut Vec<u8>, mut value: u64) {
    while value >= 128 {
        out.push(value as u8 | 128);
        value >>= 7;
    }
    out.push(value as u8);
}

fn take_number(bytes: &[u8], at: &mut usize) -> Option<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *bytes.get(*at)?;
        *at += 1;
        value |= u64::from(byte & 127) << shift;
        if byte < 128 {
            return Some(value);
        }
    }
    None
}

/// Replaces repeats of at least `MIN_REPEAT` bytes, at any distance in the
/// whole input, by references. The result is the stream the blocks are cut
/// from: the input length, the operation list (literal count, repeat
/// length, distance; a zero repeat length ends it) and the literal bytes.
fn remove_repeats(input: &[u8]) -> Vec<u8> {
    const MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut table = vec![u32::MAX; 1 << TABLE_BITS];
    let slot = |hash: u64| (hash >> (64 - TABLE_BITS)) as usize;
    let mut power = 1u64;
    for _ in 1..WINDOW {
        power = power.wrapping_mul(MULTIPLIER);
    }
    let (mut ops, mut literals) = (Vec::new(), Vec::new());
    let mut literal_start = 0usize;
    let mut hash = 0u64;
    let mut hashed_from = usize::MAX; // start of the window `hash` covers
    let mut i = 0usize;
    while i + WINDOW <= input.len() {
        if hashed_from != i {
            hash = input[i..i + WINDOW]
                .iter()
                .fold(0u64, |h, &b| h.wrapping_mul(MULTIPLIER).wrapping_add(u64::from(b) + 1));
            hashed_from = i;
        }
        let candidate = table[slot(hash)];
        if i % SAMPLE == 0 {
            table[slot(hash)] = i as u32;
        }
        let mut length = 0usize;
        if candidate != u32::MAX {
            let from = candidate as usize;
            while i + length < input.len() && input[from + length] == input[i + length] {
                length += 1;
            }
        }
        // A source too far back for the block sort to see is worth a
        // reference even when the repeat is short.
        let far = candidate != u32::MAX && i - candidate as usize >= FAR_DISTANCE;
        if length >= MIN_REPEAT || (far && length >= FAR_REPEAT) {
            put_number(&mut ops, (i - literal_start) as u64);
            put_number(&mut ops, length as u64);
            put_number(&mut ops, (i - candidate as usize) as u64);
            literals.extend_from_slice(&input[literal_start..i]);
            i += length;
            literal_start = i;
            continue;
        }
        // Slide the window one byte.
        if i + WINDOW < input.len() {
            hash = hash
                .wrapping_sub((u64::from(input[i]) + 1).wrapping_mul(power))
                .wrapping_mul(MULTIPLIER)
                .wrapping_add(u64::from(input[i + WINDOW]) + 1);
            hashed_from = i + 1;
        }
        i += 1;
    }
    put_number(&mut ops, (input.len() - literal_start) as u64);
    put_number(&mut ops, 0);
    literals.extend_from_slice(&input[literal_start..]);
    let mut stream = Vec::with_capacity(literals.len() + ops.len() + 20);
    put_number(&mut stream, input.len() as u64);
    put_number(&mut stream, ops.len() as u64);
    stream.extend_from_slice(&ops);
    stream.extend_from_slice(&literals);
    stream
}

fn restore_repeats(stream: &[u8]) -> Option<Vec<u8>> {
    let mut at = 0usize;
    let total = usize::try_from(take_number(stream, &mut at)?).ok()?;
    let ops_length = usize::try_from(take_number(stream, &mut at)?).ok()?;
    let ops = stream.get(at..at.checked_add(ops_length)?)?;
    let mut literals = stream.get(at + ops_length..)?;
    if total > literals.len().checked_mul(1 << 20)? + (1 << 20) {
        return None;
    }
    let mut out = Vec::with_capacity(total);
    let mut op = 0usize;
    loop {
        let count = usize::try_from(take_number(ops, &mut op)?).ok()?;
        let length = usize::try_from(take_number(ops, &mut op)?).ok()?;
        let (now, rest) = literals.split_at_checked(count)?;
        out.extend_from_slice(now);
        literals = rest;
        if length == 0 {
            break;
        }
        let distance = usize::try_from(take_number(ops, &mut op)?).ok()?;
        if distance == 0 || distance > out.len() || out.len() + length > total {
            return None;
        }
        for _ in 0..length {
            out.push(out[out.len() - distance]);
        }
    }
    (out.len() == total && literals.is_empty() && op == ops.len()).then_some(out)
}

// ---------- measurement ----------

/// Stages of packing a block, timed when `NEXT_TRACE` is set.
#[derive(Clone, Copy)]
enum Stage {
    Filter,
    Sort,
    SortedCoding,
    MatchCoding,
}

/// Running totals for the `NEXT_TRACE` report: microseconds per stage (summed
/// over threads) and input and output bytes per method.
struct Stats;

static STAGE_MICROS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];
static METHOD_BYTES: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

impl Stats {
    fn enabled() -> bool {
        std::env::var_os("NEXT_TRACE").is_some()
    }

    fn time<R>(stage: Stage, work: impl FnOnce() -> R) -> R {
        let start = std::time::Instant::now();
        let result = work();
        STAGE_MICROS[stage as usize]
            .fetch_add(start.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
        result
    }

    fn note(label: &str, filter: Filter, size: usize) {
        if Self::enabled() {
            eprintln!("  trial {label} {:?} -> {size}", filter.code());
        }
    }

    fn chosen(matched: bool, input: usize, output: usize) {
        let at = 2 * usize::from(matched);
        METHOD_BYTES[at].fetch_add(input as u64, std::sync::atomic::Ordering::Relaxed);
        METHOD_BYTES[at + 1].fetch_add(output as u64, std::sync::atomic::Ordering::Relaxed);
    }

    fn report() {
        if !Self::enabled() {
            return;
        }
        let load = |cell: &std::sync::atomic::AtomicU64| cell.load(std::sync::atomic::Ordering::Relaxed);
        let names = ["filters", "sorting", "coding sorted bytes", "match coding"];
        for (name, cell) in names.iter().zip(&STAGE_MICROS) {
            eprintln!("stage {name:20} {:8.2} s (all threads)", load(cell) as f64 / 1e6);
        }
        eprintln!("blocks sorted:      {} -> {} bytes", load(&METHOD_BYTES[0]), load(&METHOD_BYTES[1]));
        eprintln!("blocks match-coded: {} -> {} bytes", load(&METHOD_BYTES[2]), load(&METHOD_BYTES[3]));
    }
}

/// Up to three 256 KiB slices of a block, for quick estimates.
fn sample(block: &[u8]) -> Vec<u8> {
    const SLICE: usize = 1 << 18;
    if block.len() <= 4 * SLICE {
        return block.to_vec();
    }
    let starts = [0, block.len() / 2 - SLICE / 2, block.len() - SLICE];
    starts.iter().flat_map(|&start| &block[start..start + SLICE]).copied().collect()
}

/// Bits needed for `bytes` when each byte is coded from the counts of bytes
/// that followed the same previous byte: a cheap stand-in for a full trial.
fn estimate(bytes: &[u8]) -> u64 {
    let bytes = if bytes.len() > 1 << 20 { sample(bytes) } else { bytes.to_vec() };
    let mut counts = vec![0u32; 1 << 16];
    for pair in bytes.windows(2) {
        counts[usize::from(pair[0]) << 8 | usize::from(pair[1])] += 1;
    }
    // log2 in 1/16 bit steps, integer only.
    let log2 = |value: u32| -> u64 {
        let top = 31 - value.max(1).leading_zeros();
        let fraction = if top >= 4 { (value >> (top - 4)) & 15 } else { (value << (4 - top)) & 15 };
        u64::from(top * 16 + fraction)
    };
    counts
        .chunks(256)
        .map(|row| {
            let total: u32 = row.iter().sum();
            row.iter()
                .filter(|&&count| count > 0)
                .map(|&count| u64::from(count) * (log2(total) - log2(count)))
                .sum::<u64>()
        })
        .sum()
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
        if distance >= 2 {
            filters.extend([Filter::Columns(distance), Filter::ColumnDeltas(distance)]);
        }
    }
    // A quick estimate (bits per byte given the previous byte, on samples)
    // picks which filters deserve a full trial: the plain block and the call
    // filter always, and the three best of the others when they beat plain.
    let plain = estimate(block);
    let mut scored: Vec<(u64, Filter)> = filters[2..]
        .iter()
        .map(|&filter| (estimate(&filter.forward(&sample(block))), filter))
        .filter(|&(score, _)| score < plain)
        .collect();
    scored.sort_by_key(|&(score, _)| score);
    let trials = filters[..2].iter().copied().chain(scored.iter().take(3).map(|&(_, filter)| filter));
    // Each chosen filter is tried in full with block sorting; the smallest
    // coded block wins and the first of equals is kept.
    let mut best: Option<(Filter, u32, Vec<u8>)> = None;
    let mut keep = |candidate: (Filter, u32, Vec<u8>)| {
        let label = if candidate.1 == MATCH_CODED { "match" } else { "sort" };
        Stats::note(label, candidate.0, candidate.2.len());
        if best.as_ref().is_none_or(|(_, _, coded)| candidate.2.len() < coded.len()) {
            best = Some(candidate);
        }
    };
    let mut sorted_sizes = [usize::MAX; 2];
    for (index, filter) in trials.enumerate() {
        let filtered = Stats::time(Stage::Filter, || filter.forward(block));
        let (last, primary) = Stats::time(Stage::Sort, || forward_bwt(&filtered));
        let coded = Stats::time(Stage::SortedCoding, || encode_bytes(&last));
        if index < 2 {
            sorted_sizes[index] = coded.len();
        }
        keep((filter, primary, coded));
    }
    // The match coder, after whichever of plain and the call filter sorted
    // smaller.
    let filter = if sorted_sizes[1] < sorted_sizes[0] { Filter::Calls } else { Filter::None };
    let coded = Stats::time(Stage::MatchCoding, || match_encode(&filter.forward(block)));
    keep((filter, MATCH_CODED, coded));
    let (filter, primary, coded) = best.expect("at least the plain block was tried");
    Stats::chosen(primary == MATCH_CODED, block.len(), coded.len());
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
            let mut block = if primary == MATCH_CODED {
                match_decode(payload, length)?
            } else {
                inverse_bwt(&decode_bytes(payload, length), primary)?
            };
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
        let stream = remove_repeats(&input);
        let pieces: Vec<&[u8]> = stream.chunks(BLOCK).collect();
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
        let Some(output) = restored.and_then(|parts| restore_repeats(&parts.concat())) else {
            eprintln!("corrupt or unsupported archive");
            std::process::exit(1);
        };
        output
    };
    Stats::report();
    std::fs::File::create(&args[3])
        .and_then(|mut file| file.write_all(&output))
        .unwrap_or_else(|error| {
            eprintln!("cannot write {}: {error}", args[3]);
            std::process::exit(1);
        });
}

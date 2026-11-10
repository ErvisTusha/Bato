//! Reversible relative-to-absolute address transform for x86 operands.
//! Provenance: the general E8/E9 relative-address filtering idea used by
//! executable preprocessors (BCJ-style), extended to the RIP-relative
//! displacements of x86-64 as documented in the public Intel and AMD
//! instruction-format manuals (ModRM `mod = 00`, `r/m = 101`). The segment
//! reach uses the interval bijection publicly documented for the LZX E8 call
//! translation. The descending/ascending formulation, the opcode set, the
//! selection statistics and the Rust are original.
//!
//! A site is a position `i` with a 32-bit little-endian operand in bytes
//! `i + 1..=i + 4` that is in reach (below) and is either
//! * a call or jump: byte `i` is `0xE8` or `0xE9`; or
//! * (`Class::CallsAndRelative` only) a RIP-relative operand: byte `i` is a
//!   ModRM byte `mod = 00, r/m = 101` and byte `i - 1` is one of the
//!   `RELATIVE_OPCODES`.
//!
//! `Reach::Near`: byte `i + 4` is `0x00` or `0xFF`; the transform adds `i` to
//! the low 24 bits and leaves byte `i + 4` alone. `Reach::Segment`: the signed
//! operand `v` lies in `[-i, W)`, `W` being the larger of the byte count and
//! 16 MiB. That interval holds exactly the relative and the absolute forms of
//! the targets `[0, W)`, and the transform maps it onto itself: `v + i` when
//! that is below `W`, otherwise `v - W`. Either way the reach test gives the
//! same answer before and after the rewrite.
//!
//! The forward transform visits positions in descending order and the inverse
//! in ascending order. When either pass tests position `i`,
//! every site above `i` is in its transformed state and every site below is
//! in its original state, so both passes see the same bytes and make the same
//! decision. That makes the pair exactly invertible for every input,
//! including overlapping sites.
//!
//! Selection follows byte statistics only (no name, signature or header):
//! repeated targets become identical absolute values, while their relative
//! operands differ. A site class is used when its absolute values repeat
//! clearly more often than its relative operands. In 32-bit code the same
//! ModRM form holds absolute addresses, which the statistic rejects. The
//! segment reach is used when it finds clearly more repeated call targets
//! than the near reach, as in executables larger than 16 MiB (where the near
//! reach alone may not qualify at all); elsewhere its wider interval only
//! admits more false sites.

use crate::core::{
    memory::try_filled,
    types::{BatoError, TransformId},
};

/// Which instructions' operands are sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Class {
    /// Calls and jumps.
    Calls,
    /// Calls, jumps and RIP-relative displacements.
    CallsAndRelative,
}

/// How far from a site its target may lie.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reach {
    /// Within 16 MiB either way.
    Near,
    /// Anywhere in the rewritten bytes.
    Segment,
}

/// The operand sites a transformed segment rewrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sites {
    pub(crate) class: Class,
    pub(crate) reach: Reach,
}

const fn sites(class: Class, reach: Reach) -> Sites {
    Sites { class, reach }
}

/// Every site set with its transform on plain segment bytes and on a deflate
/// container.
const TRANSFORMS: [(Sites, TransformId, TransformId); 4] = {
    use TransformId as T;
    [
        (
            sites(Class::Calls, Reach::Near),
            T::X86Calls,
            T::DeflateX86Calls,
        ),
        (
            sites(Class::CallsAndRelative, Reach::Near),
            T::X86Relative,
            T::DeflateX86Relative,
        ),
        (
            sites(Class::Calls, Reach::Segment),
            T::X86CallsSegment,
            T::DeflateX86CallsSegment,
        ),
        (
            sites(Class::CallsAndRelative, Reach::Segment),
            T::X86RelativeSegment,
            T::DeflateX86RelativeSegment,
        ),
    ]
};

impl Sites {
    fn row(self) -> (Sites, TransformId, TransformId) {
        TRANSFORMS[(self.class == Class::CallsAndRelative) as usize
            + 2 * (self.reach == Reach::Segment) as usize]
    }

    /// The transform that rewrites these sites in the segment's bytes.
    pub(crate) fn transform(self) -> TransformId {
        self.row().1
    }

    /// The sites `transform` rewrites in the segment's bytes, if it is an
    /// x86 transform.
    pub(crate) fn of_transform(transform: TransformId) -> Option<Self> {
        let found = TRANSFORMS.iter().find(|row| row.1 == transform);
        found.map(|row| row.0)
    }

    /// The transform of a deflate container rewritten at `sites`.
    pub(crate) fn container_transform(sites: Option<Self>) -> TransformId {
        sites.map_or(TransformId::Deflate, |sites| sites.row().2)
    }

    /// The sites rewritten inside the deflate container of `transform`
    /// (inner `None`: none), or `None` when it is not a container.
    pub(crate) fn of_container(transform: TransformId) -> Option<Option<Self>> {
        if transform == TransformId::Deflate {
            return Some(None);
        }
        let found = TRANSFORMS.iter().find(|row| row.2 == transform);
        found.map(|row| Some(row.0))
    }
}

/// Opcode bytes that commonly precede a RIP-relative ModRM byte: integer
/// moves, LEA, arithmetic, compares and tests, indirect calls, and the
/// second byte of common `0F`-map moves, extensions and scalar/packed
/// floating-point operations. Chosen by measurement on x86-64 executables:
/// admitting every preceding byte loses most of the gain to false sites.
const RELATIVE_OPCODES: [u8; 58] = [
    0x01, 0x03, 0x09, 0x0b, 0x10, 0x11, 0x13, 0x1b, 0x21, 0x23, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d,
    0x2e, 0x2f, 0x31, 0x33, 0x38, 0x39, 0x3a, 0x3b, 0x51, 0x54, 0x57, 0x58, 0x59, 0x5c, 0x5e, 0x63,
    0x6e, 0x6f, 0x7e, 0x7f, 0x80, 0x81, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x8b, 0x8d,
    0xaf, 0xb6, 0xb7, 0xbe, 0xbf, 0xc6, 0xc7, 0xd6, 0xef, 0xff,
];

/// One bit per byte value: set for `RELATIVE_OPCODES`.
const RELATIVE_OPCODE_BITS: [u32; 8] = {
    let mut bits = [0u32; 8];
    let mut k = 0;
    while k < RELATIVE_OPCODES.len() {
        bits[(RELATIVE_OPCODES[k] >> 5) as usize] |= 1 << (RELATIVE_OPCODES[k] & 31);
        k += 1;
    }
    bits
};

#[inline]
fn value(bytes: &[u8], i: usize) -> i64 {
    i64::from(i32::from_le_bytes([
        bytes[i + 1],
        bytes[i + 2],
        bytes[i + 3],
        bytes[i + 4],
    ]))
}

/// One past the largest absolute target of the segment reach: targets may
/// lie past the bytes (zero-initialised data, page alignment of sections).
#[inline]
fn window(len: usize) -> i64 {
    len.max(1 << 24) as i64
}

#[inline]
fn in_reach(bytes: &[u8], i: usize, reach: Reach) -> bool {
    match reach {
        Reach::Near => bytes[i + 4].wrapping_add(1) & 0xfe == 0,
        Reach::Segment => {
            let v = value(bytes, i);
            v >= -(i as i64) && v < window(bytes.len())
        }
    }
}

#[inline]
fn call_opcode(bytes: &[u8], i: usize) -> bool {
    bytes[i] & 0xfe == 0xe8
}

#[inline]
fn relative_opcode(bytes: &[u8], i: usize) -> bool {
    if bytes[i] & 0xc7 != 0x05 || i == 0 {
        return false;
    }
    let opcode = bytes[i - 1];
    RELATIVE_OPCODE_BITS[(opcode >> 5) as usize] >> (opcode & 31) & 1 != 0
}

#[inline]
fn site(bytes: &[u8], i: usize, sites: Sites) -> bool {
    (call_opcode(bytes, i) || (sites.class == Class::CallsAndRelative && relative_opcode(bytes, i)))
        && in_reach(bytes, i, sites.reach)
}

#[inline]
fn store(bytes: &mut [u8], i: usize, v: i64) {
    bytes[i + 1..i + 5].copy_from_slice(&(v as i32).to_le_bytes());
}

/// The absolute form of the relative operand `v` at position `at`.
#[inline]
fn absolute(v: i64, at: i64, (reach, window): (Reach, i64)) -> i64 {
    match reach {
        Reach::Near => (v & !0xff_ffff) | ((v + at) & 0xff_ffff),
        Reach::Segment if v + at < window => v + at,
        Reach::Segment => v - window,
    }
}

/// The inverse of `absolute`.
#[inline]
fn relative(v: i64, at: i64, (reach, window): (Reach, i64)) -> i64 {
    match reach {
        Reach::Near => (v & !0xff_ffff) | ((v - at) & 0xff_ffff),
        Reach::Segment if v >= 0 => v - at,
        Reach::Segment => v + window,
    }
}

/// Rewrites the operands of `sites` in `bytes` to absolute values, in place.
pub(crate) fn forward(bytes: &mut [u8], sites: Sites) {
    let reach = (sites.reach, window(bytes.len()));
    for i in (0..bytes.len().saturating_sub(4)).rev() {
        if site(bytes, i, sites) {
            store(bytes, i, absolute(value(bytes, i), i as i64, reach));
        }
    }
}

/// Restores the bytes `forward` rewrote, in place.
pub(crate) fn inverse(bytes: &mut [u8], sites: Sites) {
    let reach = (sites.reach, window(bytes.len()));
    for i in 0..bytes.len().saturating_sub(4) {
        if site(bytes, i, sites) {
            store(bytes, i, relative(value(bytes, i), i as i64, reach));
        }
    }
}

/// Bytes of scratch used by `select` (two 2^24-bit occupancy sets).
pub(crate) const SELECTION_SCRATCH: usize = 2 << 21;

/// Fewest repeated absolute values for a site class to count as code.
const MIN_REPEATS: u64 = 32;

/// Repeated absolute values must exceed repeated relative operands by this
/// ratio (numerator, denominator) for a class to be rewritten.
const ABSOLUTE_MARGIN: (u64, u64) = (3, 2);

/// The segment reach must find this ratio (numerator, denominator) of the
/// near reach's repeated call targets to be used. Measured: +11% on a 64 MiB
/// slice of a 120 MB executable, none on trees of executables under 16 MiB.
const SEGMENT_MARGIN: (u64, u64) = (17, 16);

/// Low operand bits the occupancy sets distinguish.
const SET_MASK: u32 = 0xff_ffff;

/// Counts of repeated operand values over the positions `is_site` accepts,
/// as absolute values and as stored. Clears both occupancy sets first.
fn repeats(
    bytes: &[u8],
    is_site: impl Fn(usize) -> bool,
    (absolute, relative): (&mut [u8], &mut [u8]),
) -> (u64, u64) {
    absolute.fill(0);
    relative.fill(0);
    let seen = |set: &mut [u8], value: u32| {
        let (index, bit) = ((value >> 3) as usize, 1u8 << (value & 7));
        let repeated = set[index] & bit != 0;
        set[index] |= bit;
        u64::from(repeated)
    };
    let (mut absolute_repeats, mut relative_repeats) = (0u64, 0u64);
    for i in (0..bytes.len() - 4).filter(|&i| is_site(i)) {
        let value = value(bytes, i) as u32;
        relative_repeats += seen(relative, value & SET_MASK);
        absolute_repeats += seen(absolute, value.wrapping_add(i as u32) & SET_MASK);
    }
    (absolute_repeats, relative_repeats)
}

fn favours_absolute((absolute, relative): (u64, u64)) -> bool {
    absolute >= MIN_REPEATS && ABSOLUTE_MARGIN.1 * absolute > ABSOLUTE_MARGIN.0 * relative
}

/// The sites worth rewriting in `bytes`, if any. RIP-relative sites are only
/// used together with call sites: code that has the former has the latter.
pub(crate) fn select(bytes: &[u8]) -> Result<Option<Sites>, BatoError> {
    if bytes.len() < 5 {
        return Ok(None);
    }
    // With fewer than MIN_REPEATS + 1 candidate calls/jumps there cannot be
    // MIN_REPEATS repeated targets in either reach. This exact prerequisite
    // avoids allocating and clearing 4 MiB of occupancy sets on most
    // non-executable data without changing transform decisions.
    let mut possible = 0u64;
    for i in 0..bytes.len() - 4 {
        if call_opcode(bytes, i)
            && (in_reach(bytes, i, Reach::Near) || in_reach(bytes, i, Reach::Segment))
        {
            possible += 1;
            if possible > MIN_REPEATS {
                break;
            }
        }
    }
    if possible <= MIN_REPEATS {
        return Ok(None);
    }
    let mut absolute = try_filled(1 << 21, 0u8)?;
    let mut relative = try_filled(1 << 21, 0u8)?;
    let mut count =
        |is_site: &dyn Fn(usize) -> bool| repeats(bytes, is_site, (&mut absolute, &mut relative));
    let calls = |reach| move |i| call_opcode(bytes, i) && in_reach(bytes, i, reach);
    let near = count(&calls(Reach::Near));
    let segment = count(&calls(Reach::Segment));
    let (reach, chosen) = if SEGMENT_MARGIN.1 * segment.0 > SEGMENT_MARGIN.0 * near.0 {
        (Reach::Segment, segment)
    } else {
        (Reach::Near, near)
    };
    if !favours_absolute(chosen) {
        return Ok(None);
    }
    let loads = count(&|i| relative_opcode(bytes, i) && in_reach(bytes, i, reach));
    let class = if favours_absolute(loads) {
        Class::CallsAndRelative
    } else {
        Class::Calls
    };
    Ok(Some(Sites { class, reach }))
}

#[cfg(test)]
mod tests {
    use super::super::hash;
    use super::*;

    const CALLS: Sites = Sites {
        class: Class::Calls,
        reach: Reach::Near,
    };
    const RELATIVE: Sites = Sites {
        class: Class::CallsAndRelative,
        reach: Reach::Near,
    };

    fn segment(sites: Sites) -> Sites {
        Sites {
            reach: Reach::Segment,
            ..sites
        }
    }

    fn operand(bytes: &[u8], i: usize) -> u32 {
        value(bytes, i) as u32 & SET_MASK
    }

    fn roundtrip(original: &[u8]) {
        let mut bytes = original.to_vec();
        for sites in [CALLS, RELATIVE, segment(CALLS), segment(RELATIVE)] {
            forward(&mut bytes, sites);
            inverse(&mut bytes, sites);
            assert_eq!(bytes, original);
        }
    }

    #[test]
    fn too_few_sites_skip_occupancy_allocation() {
        for count in [0, 1, 16, MIN_REPEATS as usize] {
            let mut raw = vec![0x90; 4096];
            for i in 0..count {
                let at = i * 8;
                raw[at] = 0xe8;
                raw[at + 1..at + 5].copy_from_slice(&0i32.to_le_bytes());
            }
            let (selected, observed) = crate::core::diagnostics::capture(|| select(&raw).unwrap());
            assert!(selected.is_none());
            assert_eq!(observed.heap.peak, 0);
        }
    }

    #[test]
    fn transform_is_exactly_invertible_for_dense_overlapping_sites() {
        for length in 0..64 {
            roundtrip(&vec![0xe8; length]);
            roundtrip(&vec![0xe9; length]);
            let alternating: Vec<u8> = (0..length)
                .map(|i| [0xe8, 0xff, 0xe9, 0x00][i % 4])
                .collect();
            roundtrip(&alternating);
        }
        // Pseudo-random bytes drawn mostly from site/condition values so that
        // overlapping operands rewrite later condition bytes.
        for seed in 0..256u64 {
            let bytes: Vec<u8> = (0..4096u64)
                .map(|i| {
                    let r = hash::mix(seed << 32 | i);
                    [0xe8, 0xe9, 0x00, 0xff, 0x8b, 0x05, 0x8d, 0x0d, r as u8]
                        [(r >> 40) as usize % 9]
                })
                .collect();
            roundtrip(&bytes);
        }
    }

    #[test]
    fn operands_at_the_window_edges_map_one_to_one() {
        let window = window(0) as i32;
        for at in [0usize, 1, 9] {
            for v in [
                -(at as i32) - 1,
                -(at as i32),
                -1,
                0,
                window - at as i32 - 1,
            ]
            .into_iter()
            .chain([window - at as i32, window - 1, window])
            {
                let mut bytes = vec![0x90; at];
                bytes.push(0xe8);
                bytes.extend_from_slice(&v.to_le_bytes());
                let original = bytes.clone();
                forward(&mut bytes, segment(CALLS));
                let inside = v >= -(at as i32) && v < window;
                assert_eq!(site(&bytes, at, segment(CALLS)), inside);
                inverse(&mut bytes, segment(CALLS));
                assert_eq!(bytes, original);
            }
        }
    }

    #[test]
    fn forward_rewrites_relative_calls_to_shared_absolute_targets() {
        let mut code = Vec::new();
        for i in 0..64u32 {
            let at = code.len() as u32;
            code.push(0xe8);
            code.extend_from_slice(&(0x1000u32.wrapping_sub(at)).to_le_bytes());
            code.extend_from_slice(&[0x90, (i as u8) & 0x7f]);
        }
        assert_eq!(select(&code).unwrap(), Some(CALLS));
        let mut transformed = code.clone();
        forward(&mut transformed, CALLS);
        let targets: std::collections::HashSet<u32> =
            (0..64).map(|k| operand(&transformed, k * 7)).collect();
        assert_eq!(targets.len(), 1);
        roundtrip(&code);
    }

    /// Code with shared call targets and `lea rax, [rip + d]` loads of one
    /// address (`relative`) or `mov eax, [d]` loads of one absolute address.
    fn loads(relative: bool) -> Vec<u8> {
        let mut code = Vec::new();
        for i in 0..256u32 {
            let at = code.len() as u32;
            code.push(0xe8);
            code.extend_from_slice(&(0x9000u32.wrapping_sub(at)).to_le_bytes());
            let at = code.len() as u32 + 2;
            let operand = if relative {
                0x4000u32.wrapping_sub(at)
            } else {
                0x4000 + (i & 3)
            };
            code.extend_from_slice(&[0x48, 0x8d, 0x05]);
            code.extend_from_slice(&operand.to_le_bytes());
            code.push(0x90);
        }
        code
    }

    #[test]
    fn relative_operands_are_rewritten_only_where_they_are_relative() {
        let code = loads(true);
        assert_eq!(select(&code).unwrap(), Some(RELATIVE));
        let mut transformed = code.clone();
        forward(&mut transformed, RELATIVE);
        let targets: std::collections::HashSet<u32> = (0..256)
            .map(|k| operand(&transformed, k * 13 + 7))
            .collect();
        assert_eq!(targets.len(), 1);
        roundtrip(&code);
        // The same instruction form with absolute addresses, as in 32-bit code.
        assert_eq!(select(&loads(false)).unwrap(), Some(CALLS));
    }

    #[test]
    fn far_targets_select_the_segment_reach() {
        // Calls from 20 MiB up reach a few targets near the start: more
        // than 16 MiB away, so only the segment reach sees them at all.
        let mut code = vec![0u8; 20 << 20];
        for k in 0..4096u32 {
            let at = code.len() as u32;
            code.push(0xe8);
            code.extend_from_slice(&(0x1000 + (k & 15) * 64).wrapping_sub(at).to_le_bytes());
            code.extend_from_slice(&[0x90, k as u8 & 0x7f]);
        }
        assert_eq!(select(&code).unwrap(), Some(segment(CALLS)));
    }

    #[test]
    fn text_and_random_bytes_are_not_selected() {
        let text = b"plain text without call sites\n".repeat(500);
        assert_eq!(select(&text).unwrap(), None);
        let random: Vec<u8> = (0..1 << 16).map(|i| hash::mix(i) as u8).collect();
        assert_eq!(select(&random).unwrap(), None);
        assert_eq!(select(&[0xe8, 0, 0, 0]).unwrap(), None);
    }
}

//! Bounded byte-distribution checks for exact repeats and stored prefixes.
//! Provenance: Pearson's chi-squared goodness-of-fit statistic applied to
//! byte values, lagged differences, and paired high and low nibbles.

const WINDOW: usize = 8192;
const CHI_SQUARED_LIMIT: u64 = 380;
const MIN_SAMPLE: usize = 4 * WINDOW;
const LAGS: [usize; 11] = [1, 2, 4, 8, 16, 31, 257, 512, 1024, 2048, 4096];

fn uniform_bins(bins: [u32; 256], count: usize) -> bool {
    let count = count as u64;
    let sum: u64 = bins.into_iter().map(|n| u64::from(n).pow(2)).sum();
    sum * 256 <= count * (count + CHI_SQUARED_LIMIT)
}

fn uniform_window(bytes: &[u8; WINDOW]) -> bool {
    let mut bins = [0; 256];
    for &byte in bytes {
        bins[usize::from(byte)] += 1;
    }
    if !uniform_bins(bins, bytes.len()) {
        return false;
    }
    for lag in LAGS {
        let (mut differences, mut high, mut low) = ([0; 256], [0; 256], [0; 256]);
        for (&a, &b) in bytes.iter().zip(&bytes[lag..]) {
            differences[usize::from(b.wrapping_sub(a))] += 1;
            high[usize::from((a & 0xf0) | (b >> 4))] += 1;
            low[usize::from((a << 4) | (b & 15))] += 1;
        }
        let count = bytes.len() - lag;
        if ![differences, high, low]
            .into_iter()
            .all(|bins| uniform_bins(bins, count))
        {
            return false;
        }
    }
    true
}

pub(super) fn uniform(raw: &[u8]) -> bool {
    let (windows, tail) = raw.as_chunks::<WINDOW>();
    raw.len() >= MIN_SAMPLE && tail.is_empty() && windows.iter().all(uniform_window)
}

pub(super) fn periodic(raw: &[u8], chunk: usize) -> Option<usize> {
    let period = chunk.max(MIN_SAMPLE);
    (raw.len() / 2 >= period
        && raw[period..] == raw[..raw.len() - period]
        && uniform(&raw[..period]))
    .then_some(period)
}

pub(super) fn cold_prefix(raw: &[u8], chunk: usize) -> Option<usize> {
    if raw.len() / chunk < 4 {
        return None;
    }
    let prefix = (raw.len() - 1) / chunk * chunk;
    let suffix = &raw[prefix..];
    (suffix.len() >= MIN_SAMPLE && !uniform(suffix) && uniform(&raw[..prefix])).then_some(prefix)
}

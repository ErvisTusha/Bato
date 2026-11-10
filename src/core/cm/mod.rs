//! Original integer CM segment engine under development.
//! Provenance: independently implemented from arithmetic-coding and logistic
//! mixing descriptions in Mahoney, Data Compression Explained. No PAQ/CMIX
//! implementation, model table, or third-party compressor is incorporated.

mod adapt;
mod ans;
mod apm;
mod coder;
mod context_map;
mod deflate;
pub(crate) mod descriptor;
mod fast;
mod filters;
mod hash;
mod huffman;
mod isse;
mod logistic;
mod lz;
mod mixer;
mod models;
mod numbers;
mod predictor;
pub mod profile;
pub(crate) mod segment;
mod simd;
mod state;
mod table;
pub(crate) mod varint;
mod x86;

/// Bits of the current byte already coded. A prefix holds them below a
/// leading 1 marker, so prefixes 1..=255 give depths 0..=7.
#[inline]
fn coded_bits(prefix: u16) -> u32 {
    15 - prefix.leading_zeros()
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    #[test]
    fn cm_sources_are_integer_only() {
        fn inspect(path: &Path) {
            for entry in fs::read_dir(path).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    inspect(&path);
                } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
                    let source = fs::read_to_string(&path).unwrap();
                    let code: String = source
                        .lines()
                        .map(|line| line.split("//").next().unwrap())
                        .collect::<Vec<_>>()
                        .join("\n");
                    let forbidden = [format!("f{}", 32), format!("f{}", 64)];
                    for token in code.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
                        assert!(
                            !forbidden.iter().any(|kind| token == kind
                                || (token.starts_with(|c: char| c.is_ascii_digit())
                                    && token.ends_with(kind))),
                            "floating type in {}",
                            path.display()
                        );
                        if token.starts_with(|c: char| c.is_ascii_digit())
                            && !["0x", "0b", "0o"]
                                .iter()
                                .any(|prefix| token.starts_with(prefix))
                        {
                            let suffix =
                                token.trim_start_matches(|c: char| c.is_ascii_digit() || c == '_');
                            assert!(
                                !suffix.starts_with(['e', 'E']),
                                "floating exponent in {}",
                                path.display()
                            );
                        }
                    }
                    let bytes = code.as_bytes();
                    for i in 1..bytes.len().saturating_sub(1) {
                        assert!(
                            !(bytes[i] == b'.'
                                && bytes[i - 1].is_ascii_digit()
                                && bytes[i + 1].is_ascii_digit()),
                            "floating literal in {}",
                            path.display()
                        );
                    }
                }
            }
        }
        inspect(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src/core/cm"));
    }
}

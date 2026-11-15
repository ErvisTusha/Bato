//! Embed vendor-native GPU libraries into the OS-specific Bato executable.
//! No vendor SDK is needed by Rust itself; release tooling supplies the already
//! compiled native libraries through BATO_BUNDLE_*_LIBRARY environment values.
//! A missing library does not prevent the portable CUDA/wgpu/CPU build.

use std::{
    env,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

struct Backend {
    feature: &'static str,
    source: &'static str,
    output: &'static str,
    cfg: &'static str,
}

const BACKENDS: &[Backend] = &[
    Backend {
        feature: "CARGO_FEATURE_HIP",
        source: "BATO_BUNDLE_HIP_LIBRARY",
        output: "BATO_EMBEDDED_HIP_LIBRARY",
        cfg: "bato_embedded_hip",
    },
    Backend {
        feature: "CARGO_FEATURE_LEVEL_ZERO",
        source: "BATO_BUNDLE_LEVEL_ZERO_LIBRARY",
        output: "BATO_EMBEDDED_LEVEL_ZERO_LIBRARY",
        cfg: "bato_embedded_level_zero",
    },
    Backend {
        feature: "CARGO_FEATURE_METAL",
        source: "BATO_BUNDLE_METAL_LIBRARY",
        output: "BATO_EMBEDDED_METAL_LIBRARY",
        cfg: "bato_embedded_metal",
    },
];

fn validate_binary(path: &Path, os: &str) {
    let mut file = File::open(path).unwrap_or_else(|error| {
        panic!("cannot open native GPU library {}: {error}", path.display())
    });
    let mut signature = [0_u8; 4];
    file.read_exact(&mut signature).unwrap_or_else(|error| {
        panic!("native GPU library {} is empty or unreadable: {error}", path.display())
    });
    let valid = match os {
        "linux" => signature == *b"\x7fELF",
        "windows" => signature[..2] == *b"MZ",
        "macos" => matches!(
            signature,
            [0xcf, 0xfa, 0xed, 0xfe] | [0xfe, 0xed, 0xfa, 0xcf]
                | [0xca, 0xfe, 0xba, 0xbe] | [0xbe, 0xba, 0xfe, 0xca]
        ),
        other => panic!("unsupported native GPU library target OS: {other}"),
    };
    assert!(
        valid,
        "native GPU library {} is not a {} shared library",
        path.display(),
        os
    );
}

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("Cargo target OS is required");
    let mut embedded = Vec::new();
    for backend in BACKENDS {
        println!("cargo:rustc-check-cfg=cfg({})", backend.cfg);
        println!("cargo:rerun-if-env-changed={}", backend.source);
        let Some(path) = env::var_os(backend.source) else {
            continue;
        };
        assert!(
            env::var_os(backend.feature).is_some(),
            "{} was supplied but the corresponding Cargo feature is disabled",
            backend.source
        );
        let file = PathBuf::from(path).canonicalize().unwrap_or_else(|error| {
            panic!("cannot resolve {} native GPU library: {error}", backend.source)
        });
        validate_binary(&file, &target_os);
        println!("cargo:rerun-if-changed={}", file.display());
        println!("cargo:rustc-env={}={}", backend.output, file.display());
        println!("cargo:rustc-cfg={}", backend.cfg);
        embedded.push(backend.cfg);
    }

    println!("cargo:rerun-if-env-changed=BATO_REQUIRE_BUNDLED_NATIVE");
    if env::var_os("BATO_REQUIRE_BUNDLED_NATIVE").is_some() {
        let required: &[&str] = match target_os.as_str() {
            "macos" => &["bato_embedded_metal"],
            "linux" | "windows" => &["bato_embedded_hip", "bato_embedded_level_zero"],
            _ => &[],
        };
        for cfg in required {
            assert!(
                embedded.contains(cfg),
                "universal native GPU release requires {cfg}; supply the vendor SDK-built library"
            );
        }
    }

    println!(
        "cargo:rustc-env=BATO_BUNDLED_NATIVE_BACKENDS={}",
        if embedded.is_empty() {
            "none".to_string()
        } else {
            embedded
                .iter()
                .map(|cfg| cfg.trim_start_matches("bato_embedded_"))
                .collect::<Vec<_>>()
                .join(",")
        }
    );
    println!(
        "cargo:warning=Bato GPU bundle: {} native provider(s) embedded for {target_os}; \
         CUDA and wgpu remain linked by the selected features",
        embedded.len()
    );
}

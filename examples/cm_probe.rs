//! Diagnostic exact sizes on complete frozen development files. This is not
//! the canonical timing/acceptance workflow; use benchmark_suite.py for gates.
use bato_engine::*;
use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args_os().skip(1);
    let mut paths = Vec::new();
    let mut output = None;
    let mut source = String::from("uncommitted");
    while let Some(arg) = args.next() {
        if arg == "--output" {
            output = Some(PathBuf::from(args.next().ok_or("--output needs a path")?));
        } else if arg == "--source-revision" {
            source = args
                .next()
                .ok_or("--source-revision needs an identity")?
                .into_string()
                .map_err(|_| "source identity must be UTF-8")?;
        } else {
            paths.push(PathBuf::from(arg));
        }
    }
    if paths.is_empty() {
        return Err("supply complete frozen development files".into());
    }
    let cm = Geometry::default();
    let mut rows = Vec::new();
    let mut total = 0usize;
    for path in paths {
        let bytes = fs::read(&path)?;
        if bytes.len() > 2 * 1024 * 1024 {
            return Err("a probe file exceeds the 2 MiB feedback cap".into());
        }
        total = total
            .checked_add(bytes.len())
            .ok_or("probe size overflow")?;
        if total > 8 * 1024 * 1024 {
            return Err("the probe exceeds the 8 MiB feedback cap".into());
        }
        let options = CompressionOptions {
            geometry: cm,
            execution: ExecutionOptions {
                threads: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let encoded = compress_with_options(&bytes, &options)?;
        let decoded = decompress_with_options(
            &encoded,
            &DecompressionOptions {
                execution: ExecutionOptions {
                    threads: 1,
                    ..Default::default()
                },
                ..Default::default()
            },
        )?;
        if decoded != bytes {
            return Err("probe failed exact restored-byte verification".into());
        }
        rows.push(serde_json::json!({"path":path,"input_bytes":bytes.len(),
            "input_sha256":format!("{:x}",Sha256::digest(&bytes)),"encoded_bytes":encoded.len(),
            "encoded_sha256":format!("{:x}",Sha256::digest(&encoded)),"verified":true}));
        eprintln!(
            "verified {}: {} -> {} bytes",
            path.display(),
            bytes.len(),
            encoded.len()
        );
    }
    let report = serde_json::json!({"diagnostic_only":true,"source_revision":source,
        "format":format!("{}.{}", bato_engine::VERSION_MAJOR, bato_engine::VERSION_MINOR),"segment_size":cm.segment_size,
        "chunk_size":cm.chunk_size,"table_log2":cm.table_log2,"files":rows});
    let path = output.unwrap_or_else(|| {
        PathBuf::from(format!(
            "artifacts/probes/format-{}.{}.json",
            bato_engine::VERSION_MAJOR,
            bato_engine::VERSION_MINOR
        ))
    });
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, serde_json::to_string_pretty(&report)? + "\n")?;
    println!("{}", path.display());
    Ok(())
}

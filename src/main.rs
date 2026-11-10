//! Bato command-line interface: parses arguments, calls the library and
//! prints results.
use bato_engine::core::archive::{
    extract_solid_archive_members,
    managed::{self, ArchiveOptions, Edit},
    restore_catalogue_members, MemberSelection,
};
use bato_engine::core::jobs::{Job, Section, Start};
use bato_engine::core::recovery;
use bato_engine::core::safe_fs::{absolute_path, open_regular_file, AtomicOutput};
use bato_engine::{
    compress_stream, create_solid_archive_excluding, decompress_stream_with_options,
    extract_solid_archive_with_execution, read_stream_header, read_stream_layout,
    CompressionOptions, DecodeLimits, DecompressionOptions, ExecutionOptions, ExecutionStats,
    Geometry, MemoryBudget, Overwrite, StreamHeader, StreamKind, VERSION_MAJOR, VERSION_MINOR,
};
use sha2::{Digest, Sha256};
use std::{
    env,
    ffi::OsString,
    fs::{self, File},
    io::{self, BufRead, BufReader, BufWriter, Cursor, Read, Seek, Write},
    path::{Path, PathBuf},
    process,
};
use tempfile::tempdir;

/// Suffix appended to compressed output names.
const BATO_SUFFIX: &str = ".bato";
/// Extension recognized (and removed) when choosing a decoded output name.
const BATO_EXTENSION: &str = "bato";
/// Command-line spelling of standard input or output.
const STDIO: &str = "-";
/// Bytes of the fixed stream header read to classify an input.
const CLASSIFY_BYTES: usize = 7;

/// What one invocation does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Compress,
    Archive,
    Decompress,
    Extract,
    Test,
    Info,
    Add,
    Remove,
    RepairSplit,
    RecoveryIndex,
    Recover,
    List,
    Diff,
    Health,
    JobStatus,
}

enum OutputTarget {
    Default,
    Stdout,
    Path(PathBuf),
}
impl OutputTarget {
    fn path(&self) -> Option<&Path> {
        match self {
            Self::Path(path) => Some(path),
            _ => None,
        }
    }
}

/// Where integer option values may be zero.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Zero {
    Allowed,
    Rejected,
}

/// A fully parsed command line.
struct Command {
    mode: Mode,
    /// `None` reads standard input.
    input: Option<PathBuf>,
    output: OutputTarget,
    overwrite: Overwrite,
    compression: CompressionOptions,
    decompression: DecompressionOptions,
    execution: ExecutionOptions,
    print_stats: bool,
    archive: ArchiveOptions,
    members: Vec<PathBuf>,
    selection: MemberSelection,
    checkpoint: Option<Checkpoint>,
    pause_after: Option<usize>,
    pause_file: Option<PathBuf>,
    recovery_index: Option<PathBuf>,
    json: bool,
    progress: bool,
}

struct Checkpoint {
    path: PathBuf,
    start: Start,
}

fn die(s: impl std::fmt::Display) -> ! {
    eprintln!("Error: {s}");
    process::exit(1)
}
fn help() {
    eprintln!(
        "bato {} (format {}.{}) - Pure Rust Bato lossless compression engine",
        env!("CARGO_PKG_VERSION"),
        VERSION_MAJOR,
        VERSION_MINOR
    );
    eprintln!("Usage: bato [OPTIONS] [FILE]");
    eprintln!("Options: -c -d -t -i archive extract -o PATH -f --segment-size BYTES --chunk-size BYTES --threads N --memory-budget auto|BYTES --stats --timing");
    eprintln!("Archive commands: archive DIRECTORY, extract ARCHIVE [MEMBER...], add ARCHIVE PATH..., remove ARCHIVE MEMBER...");
    eprintln!("Archive options: --split-size BYTES --password-file FILE --protect MEMBER (repeatable) --public-only");
    eprintln!("Split recovery: archive --split-size BYTES [--parity | --parity-volumes N], repair-split ARCHIVE -o PATH");
    eprintln!(
        "Salvage: recovery-index ARCHIVE -o INDEX, recover ARCHIVE --index INDEX -o DIRECTORY"
    );
    eprintln!("List: list ARCHIVE [--json] [--password-file FILE | --public-only]");
    eprintln!("Inspection: diff ARCHIVE ARCHIVE_OR_DIRECTORY [--json], health ARCHIVE [--json]");
    eprintln!("Embedded index: archive DIRECTORY --indexed [--password-file FILE]");
    eprintln!("Indexed extraction: extract ARCHIVE MEMBER... [--index INDEX] -o DIRECTORY");
    eprintln!("Job progress: --progress with --checkpoint or --resume, job-status DIRECTORY");
    eprintln!("Checkpointing: --checkpoint DIR or --resume DIR, with optional --pause-file FILE or --pause-after-segments N");
    eprintln!("Automatic resources: all available CPUs and 90% of usable RAM.");
}
fn value(args: &[OsString], i: &mut usize, flag: &str) -> OsString {
    *i += 1;
    args.get(*i)
        .cloned()
        .unwrap_or_else(|| die(format!("{flag} requires a value")))
}
fn integer(args: &[OsString], i: &mut usize, flag: &str, zero: Zero) -> usize {
    let n = value(args, i, flag)
        .to_str()
        .and_then(|x| x.parse().ok())
        .unwrap_or_else(|| die(format!("{flag} requires an integer")));
    if zero == Zero::Rejected && n == 0 {
        die(format!("{flag} must be positive"))
    }
    n
}
fn append_suffix(src: &Path, suffix: &str) -> PathBuf {
    let mut name = src.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}
fn decoded_path(src: &Path, kind: StreamKind) -> PathBuf {
    if src.extension().and_then(|s| s.to_str()) == Some(BATO_EXTENSION) {
        return src.with_extension("");
    }
    append_suffix(
        src,
        match kind {
            StreamKind::DirectoryArchive => ".extracted",
            StreamKind::Bytes => ".out",
        },
    )
}

/// Parses the arguments, or returns `None` after printing help or the version.
fn parse(args: &[OsString]) -> Option<Command> {
    let mut mode = Mode::Compress;
    let mut input = None;
    let mut output = OutputTarget::Default;
    let mut overwrite = Overwrite::Refuse;
    let mut geometry = Geometry::default();
    let mut budget = MemoryBudget::Auto;
    let mut threads = 0;
    let mut print_stats = false;
    let mut print_timing = false;
    let mut archive = ArchiveOptions::default();
    let mut members = Vec::new();
    let mut checkpoint = None;
    let mut pause_after = None;
    let mut pause_file = None;
    let mut recovery_index = None;
    let mut positional = false;
    let mut json = false;
    let mut progress = false;
    let mut i = 1;
    while i < args.len() {
        let x = args[i].to_str().unwrap_or("");
        match if positional { "" } else { x } {
            "-h" | "--help" => {
                help();
                return None;
            }
            "-V" | "--version" => {
                println!(
                    "bato {} (format {}.{})",
                    env!("CARGO_PKG_VERSION"),
                    VERSION_MAJOR,
                    VERSION_MINOR
                );
                return None;
            }
            "--" => positional = true,
            "archive" if input.is_none() && mode == Mode::Compress => mode = Mode::Archive,
            "add" if input.is_none() && mode == Mode::Compress => mode = Mode::Add,
            "remove" if input.is_none() && mode == Mode::Compress => mode = Mode::Remove,
            "repair-split" if input.is_none() && mode == Mode::Compress => mode = Mode::RepairSplit,
            "extract" if input.is_none() && mode == Mode::Compress => mode = Mode::Extract,
            "recovery-index" if input.is_none() && mode == Mode::Compress => {
                mode = Mode::RecoveryIndex
            }
            "recover" if input.is_none() && mode == Mode::Compress => mode = Mode::Recover,
            "-c" | "--compress" => mode = Mode::Compress,
            "-d" | "--decompress" => mode = Mode::Decompress,
            "-t" | "--test" => mode = Mode::Test,
            "-i" | "--info" => mode = Mode::Info,
            "-f" | "--force" => overwrite = Overwrite::Replace,
            "--segment-size" => geometry.segment_size = integer(args, &mut i, x, Zero::Rejected),
            "--chunk-size" => geometry.chunk_size = integer(args, &mut i, x, Zero::Rejected),
            "--threads" => threads = integer(args, &mut i, x, Zero::Allowed),
            "--memory-budget" => {
                budget = value(args, &mut i, x)
                    .to_str()
                    .unwrap_or_else(|| die("memory budget is not UTF-8"))
                    .parse::<MemoryBudget>()
                    .unwrap_or_else(|e| die(e))
            }
            "list" if input.is_none() && mode == Mode::Compress => mode = Mode::List,
            "diff" if input.is_none() && mode == Mode::Compress => mode = Mode::Diff,
            "health" if input.is_none() && mode == Mode::Compress => mode = Mode::Health,
            "job-status" if input.is_none() && mode == Mode::Compress => mode = Mode::JobStatus,
            "--json" => json = true,
            "--indexed" => archive.indexed = true,
            "--progress" => progress = true,
            "--stats" => print_stats = true,
            "--timing" => print_timing = true,
            "--checkpoint" | "--resume" => {
                let path = value(args, &mut i, x).into();
                let start = if x == "--resume" {
                    Start::Resume
                } else {
                    Start::New
                };
                if checkpoint.replace(Checkpoint { path, start }).is_some() {
                    die("choose one of --checkpoint and --resume");
                }
            }
            "--pause-after-segments" => {
                pause_after = Some(integer(args, &mut i, x, Zero::Rejected))
            }
            "--pause-file" => pause_file = Some(value(args, &mut i, x).into()),
            "--password-file" => archive.password_file = Some(value(args, &mut i, x).into()),
            "--protect" => archive.protect.push(value(args, &mut i, x).into()),
            "--public-only" => archive.public_only = true,
            "--parity" => archive.parity = true,
            "--parity-volumes" => {
                let count = integer(args, &mut i, x, Zero::Rejected);
                if !(2..=16).contains(&count) {
                    die("--parity-volumes requires a count from 2 to 16");
                }
                archive.parity_volumes = count as u8;
            }
            "--index" => recovery_index = Some(value(args, &mut i, x).into()),
            "--split-size" => {
                archive.split_size = Some(integer(args, &mut i, x, Zero::Rejected) as u64)
            }
            "-o" | "--output" => {
                let v = value(args, &mut i, x);
                output = if v == STDIO {
                    OutputTarget::Stdout
                } else {
                    OutputTarget::Path(PathBuf::from(v))
                };
            }
            _ if !positional && x.starts_with('-') && x != STDIO => {
                die(format!("unrecognized option '{x}'"))
            }
            _ => {
                let path = PathBuf::from(&args[i]);
                if input.is_none() {
                    input = Some(path);
                } else {
                    members.push(path);
                }
            }
        }
        i += 1;
    }
    let editing = matches!(mode, Mode::Add | Mode::Remove);
    let decoding = matches!(
        mode,
        Mode::Decompress
            | Mode::Extract
            | Mode::Test
            | Mode::Info
            | Mode::List
            | Mode::Diff
            | Mode::Health
    );
    if (editing || matches!(mode, Mode::Archive | Mode::Extract)) && input.is_none() {
        die("input path required")
    }
    let stdin = input.as_deref() == Some(Path::new(STDIO));
    if (mode == Mode::Archive || editing) && stdin {
        die("archive operations require a filesystem path")
    }
    if editing {
        if members.is_empty() || !matches!(output, OutputTarget::Default) {
            die("add/remove require one or more paths and edit the archive in place");
        }
    } else if mode == Mode::Diff {
        if members.len() != 1 {
            die("diff requires an archive and one archive or directory to compare");
        }
    } else if mode != Mode::Extract && !members.is_empty() {
        die("multiple input paths");
    }
    if (archive.parity || archive.parity_volumes != 0) && archive.split_size.is_none() {
        die("parity requires --split-size");
    }
    if archive.parity && archive.parity_volumes != 0 {
        die("choose --parity or --parity-volumes");
    }
    if archive.split_size.is_some() && mode != Mode::Archive {
        die("--split-size requires archive");
    }
    if !archive.protect.is_empty() && !matches!(mode, Mode::Archive | Mode::Add) {
        die("--protect requires archive or add");
    }
    if archive.public_only && (!decoding || archive.password_file.is_some()) {
        die("--public-only requires extraction, testing or inspection without a password");
    }
    if archive.password_file.is_some() && mode == Mode::Compress {
        die("use the archive command for password protection");
    }
    if (archive.password_file.is_some() || archive.public_only) && (input.is_none() || stdin) {
        die("password and public-only options require a filesystem path");
    }
    if (pause_after.is_some() || pause_file.is_some()) && checkpoint.is_none() {
        die("pausing requires --checkpoint or --resume");
    }
    if checkpoint.is_some()
        && (input.is_none()
            || stdin
            || matches!(output, OutputTarget::Stdout)
            || !matches!(
                mode,
                Mode::Compress | Mode::Archive | Mode::Decompress | Mode::Extract
            ))
    {
        die("checkpointing requires compression or extraction with filesystem paths");
    }
    if mode == Mode::Recover && recovery_index.is_none() {
        die("recover requires --index PATH");
    }
    if recovery_index.is_some()
        && mode != Mode::Recover
        && (mode != Mode::Extract
            || members.is_empty()
            || output.path().is_none()
            || checkpoint.is_some()
            || archive.password_file.is_some()
            || archive.public_only)
    {
        die("--index requires recover or plain extract with selected members and -o PATH");
    }
    if json && !matches!(mode, Mode::List | Mode::Diff | Mode::Health) {
        die("--json requires list, diff or health");
    }
    if progress && checkpoint.is_none() {
        die("--progress requires --checkpoint or --resume");
    }
    if archive.indexed
        && (mode != Mode::Archive
            || checkpoint.is_some()
            || archive.split_size.is_some()
            || !archive.protect.is_empty())
    {
        die("--indexed requires archive without splitting, selective protection or checkpointing");
    }
    if matches!(
        mode,
        Mode::List | Mode::Diff | Mode::Health | Mode::JobStatus
    ) && (input.is_none() || stdin || !matches!(output, OutputTarget::Default))
    {
        die("inspection commands require a filesystem input");
    }
    if matches!(
        mode,
        Mode::Recover | Mode::RecoveryIndex | Mode::RepairSplit
    ) && (input.is_none()
        || stdin
        || output.path().is_none()
        || archive.password_file.is_some()
        || (mode != Mode::RecoveryIndex && overwrite == Overwrite::Replace))
    {
        die("recovery requires a filesystem input and a new output path");
    }
    if stdin {
        input = None;
    }
    let execution = ExecutionOptions {
        threads,
        stats: std::sync::Arc::new(if print_timing {
            ExecutionStats::with_timing()
        } else {
            ExecutionStats::default()
        }),
    };
    let selection = if mode == Mode::Extract {
        MemberSelection::new(&members).unwrap_or_else(|error| die(error))
    } else {
        MemberSelection::default()
    };
    Some(Command {
        mode,
        input,
        output,
        overwrite,
        compression: CompressionOptions {
            geometry,
            memory_budget: budget,
            execution: execution.clone(),
        },
        decompression: DecompressionOptions {
            limits: DecodeLimits {
                max_working_memory: budget,
                ..Default::default()
            },
            execution: execution.clone(),
        },
        execution,
        print_stats,
        archive,
        members,
        selection,
        checkpoint,
        pause_after,
        pause_file,
        recovery_index,
        json,
        progress,
    })
}

/// The input file, or standard input when no path was given.
fn open_input(input: Option<&Path>) -> Box<dyn Read> {
    match input {
        Some(path) => Box::new(BufReader::new(
            open_regular_file(path).unwrap_or_else(|e| die(e)),
        )),
        None => Box::new(BufReader::new(io::stdin())),
    }
}

/// A classified input: its header and a reader replaying the whole stream.
type Classified<R> = (StreamHeader, io::Chain<Cursor<[u8; CLASSIFY_BYTES]>, R>);

/// Replays only the fixed header. Classification never decodes user bytes,
/// allocates a second codec pool, or depends on the input filename.
fn classify_input<R: Read>(mut reader: R) -> Result<Classified<R>, String> {
    let mut prefix = [0; CLASSIFY_BYTES];
    reader.read_exact(&mut prefix).map_err(|e| e.to_string())?;
    let header = read_stream_header(&mut Cursor::new(prefix)).map_err(|e| e.to_string())?;
    Ok((header, Cursor::new(prefix).chain(reader)))
}

/// Writes through a staged file that replaces `dst` only after `f` succeeds.
fn atomic<F>(dst: &Path, command: &Command, f: F) -> Result<(), String>
where
    F: FnOnce(&mut BufWriter<File>, &Path) -> Result<(), String>,
{
    let staged = AtomicOutput::new(dst, command.overwrite, command.input.as_deref())
        .map_err(|e| e.to_string())?;
    let mut w = BufWriter::new(staged.writer().map_err(|e| e.to_string())?);
    f(&mut w, &staged.temporary_path())?;
    w.flush().map_err(|e| e.to_string())?;
    drop(w);
    staged.publish().map_err(|e| e.to_string())
}

fn compress(command: &Command) -> Result<(), String> {
    let options = &command.compression;
    let input = command.input.as_deref();
    let to_stdout = match &command.output {
        OutputTarget::Stdout => true,
        OutputTarget::Default => input.is_none(),
        OutputTarget::Path(_) => false,
    };
    if command.mode == Mode::Archive && matches!(command.output, OutputTarget::Stdout) {
        return Err("archive output cannot be stdout".into());
    }
    if command.mode == Mode::Compress && to_stdout {
        let mut stdout = BufWriter::new(io::stdout());
        return compress_stream(&mut open_input(input), &mut stdout, options)
            .map(|_| ())
            .map_err(|e| e.to_string());
    }
    let dst = command
        .output
        .path()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| append_suffix(input.unwrap_or(Path::new(STDIO)), BATO_SUFFIX));
    if let Some(src) = input.filter(|_| {
        command.archive.split_size.is_some()
            || command.archive.password_file.is_some()
            || !command.archive.protect.is_empty()
            || command.archive.indexed
    }) {
        return managed::create(src, &dst, command.overwrite, &command.archive, options)
            .map_err(|e| e.to_string());
    }
    match input {
        Some(src) if command.mode == Mode::Archive || src.is_dir() => {
            atomic(&dst, command, |w, temporary| {
                let excluded = [dst.clone(), temporary.to_path_buf()];
                create_solid_archive_excluding(src, w, options, &excluded)
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            })
        }
        _ => {
            let mut reader = open_input(input);
            atomic(&dst, command, |w, _| {
                compress_stream(&mut reader, w, options)
                    .map(|_| ())
                    .map_err(|e| e.to_string())
            })
        }
    }
}

fn decode(command: &Command, input: impl Read) -> Result<(), String> {
    let options = &command.decompression;
    let src = command.input.as_deref();
    let (header, mut reader) = classify_input(input)?;
    if command.mode == Mode::Extract && header.kind != StreamKind::DirectoryArchive {
        return Err("byte stream is not a directory archive".into());
    }
    let destination = match (&command.output, src) {
        (OutputTarget::Path(path), _) => Some(path.clone()),
        (OutputTarget::Default, Some(src)) => Some(decoded_path(src, header.kind)),
        (OutputTarget::Stdout, _) | (OutputTarget::Default, None) => None,
    };
    match (header.kind, destination) {
        (StreamKind::DirectoryArchive, None) => {
            Err("directory archive requires an output directory; use -o PATH".into())
        }
        (StreamKind::DirectoryArchive, Some(dst)) => extract_solid_archive_members(
            &mut reader,
            &dst,
            command.overwrite,
            (options, &command.selection),
        )
        .map_err(|e| e.to_string()),
        (StreamKind::Bytes, None) => {
            decompress_stream_with_options(&mut reader, &mut BufWriter::new(io::stdout()), options)
                .map(|_| ())
                .map_err(|e| e.to_string())
        }
        (StreamKind::Bytes, Some(dst)) => atomic(&dst, command, |w, _| {
            decompress_stream_with_options(&mut reader, w, options)
                .map(|_| ())
                .map_err(|e| e.to_string())
        }),
    }
}

/// Decodes and verifies everything without publishing any output.
fn test_input<R: Read>(reader: R, options: &DecompressionOptions) -> Result<(), String> {
    let (header, mut reader) = classify_input(reader)?;
    match header.kind {
        StreamKind::DirectoryArchive => {
            let sandbox = tempdir().map_err(|e| e.to_string())?;
            extract_solid_archive_with_execution(
                &mut reader,
                sandbox.path().join("extract"),
                Overwrite::Refuse,
                options,
            )
            .map(|_| ())
            .map_err(|e| e.to_string())
        }
        StreamKind::Bytes => decompress_stream_with_options(&mut reader, &mut io::sink(), options)
            .map(|_| ())
            .map_err(|e| e.to_string()),
    }
}

/// Prints the stream layout, then verifies the stream like `--test`.
fn info(command: &Command, src: &Path, mut reader: BufReader<File>) -> Result<(), String> {
    let layout = read_stream_layout(&mut reader).map_err(|e| e.to_string())?;
    let kind = match layout.header.kind {
        StreamKind::DirectoryArchive => "directory archive",
        StreamKind::Bytes => "byte stream",
    };
    println!(
        "Stream File: {}\nFormat: Bato {}.{}\nKind: {kind}",
        src.display(),
        layout.header.version_major,
        layout.header.version_minor,
    );
    let geometry = layout.geometry;
    println!(
        "Geometry: segment_size={}, chunk_size={}, table_log2={}",
        geometry.segment_size, geometry.chunk_size, geometry.table_log2
    );
    for block in &layout.blocks {
        println!(
            "Block: type={:?}, {} -> {} bytes",
            block.block_type, block.uncompressed_size, block.compressed_size
        );
    }
    reader.rewind().map_err(|e| e.to_string())?;
    test_input(reader, &command.decompression)
}

fn read_managed(command: &Command, src: &Path) -> Result<(), String> {
    if matches!(command.mode, Mode::Decompress | Mode::Extract) {
        let destination = match &command.output {
            OutputTarget::Path(path) => path.clone(),
            OutputTarget::Default => decoded_path(src, StreamKind::DirectoryArchive),
            OutputTarget::Stdout => {
                return Err("directory archive requires an output directory".into())
            }
        };
        return managed::extract_members(
            src,
            &destination,
            (command.overwrite, &command.selection),
            (&command.archive, &command.decompression, None),
        )
        .map_err(|error| error.to_string());
    }
    managed::inspect(src, &command.archive, &command.decompression)
        .map_err(|error| error.to_string())?;
    if command.mode == Mode::Info {
        println!(
            "Archive File: {}\nKind: managed directory archive\nVerified: {}",
            src.display(),
            if command.archive.public_only {
                "public section"
            } else {
                "all sections"
            }
        );
    }
    Ok(())
}

fn read_command(command: &Command) -> Result<(), String> {
    let Some(src) = command.input.as_deref() else {
        let input = open_input(None);
        return match command.mode {
            Mode::Info => Err("--info requires input".into()),
            Mode::Test => test_input(input, &command.decompression),
            _ => decode(command, input),
        };
    };
    if fs::symlink_metadata(src)
        .map_err(|e| e.to_string())?
        .is_dir()
    {
        return read_managed(command, src);
    }
    let mut input = BufReader::new(open_regular_file(src).map_err(|e| e.to_string())?);
    if managed::is_managed_prefix(input.fill_buf().map_err(|e| e.to_string())?) {
        return read_managed(command, src);
    }
    if command.archive.public_only || command.archive.password_file.is_some() {
        return Err("archive has no password protection envelope".into());
    }
    match command.mode {
        Mode::Info => info(command, src, input),
        Mode::Test => test_input(input, &command.decompression),
        _ => decode(command, input),
    }
}

fn checkpointed(command: &Command, checkpoint: &Checkpoint) -> Result<(), String> {
    let src = absolute_path(command.input.as_deref().ok_or("input required")?)
        .map_err(|e| e.to_string())?;
    let job_path = absolute_path(&checkpoint.path).map_err(|e| e.to_string())?;
    let encoding = matches!(command.mode, Mode::Compress | Mode::Archive);
    let directory = fs::symlink_metadata(&src)
        .map_err(|e| e.to_string())?
        .is_dir();
    let (mut file, kind, managed_input) = if encoding || directory {
        (None, StreamKind::DirectoryArchive, directory)
    } else {
        let mut file = open_regular_file(&src).map_err(|e| e.to_string())?;
        let mut prefix = [0; 4];
        file.read_exact(&mut prefix).map_err(|e| e.to_string())?;
        file.rewind().map_err(|e| e.to_string())?;
        let managed = managed::is_managed_prefix(&prefix);
        let kind = if managed {
            StreamKind::DirectoryArchive
        } else {
            read_stream_header(&mut file)
                .map_err(|e| e.to_string())?
                .kind
        };
        (Some(file), kind, managed)
    };
    if !encoding
        && !managed_input
        && (command.archive.public_only || command.archive.password_file.is_some())
    {
        return Err("archive has no password protection envelope".into());
    }
    if command.mode == Mode::Extract && kind != StreamKind::DirectoryArchive {
        return Err("byte stream is not a directory archive".into());
    }
    let default = if encoding {
        append_suffix(&src, BATO_SUFFIX)
    } else {
        decoded_path(&src, kind)
    };
    let dst =
        absolute_path(command.output.path().unwrap_or(&default)).map_err(|e| e.to_string())?;
    if [(&src, &job_path), (&dst, &job_path)]
        .iter()
        .any(|(a, b)| a.starts_with(b) || b.starts_with(a))
        || (encoding && directory && dst.starts_with(&src))
        || src == dst
    {
        return Err("input, output, and checkpoint must be separate paths".into());
    }
    let mut identity = Sha256::new();
    identity.update(format!(
        "{:?}\n{src:?}\n{dst:?}\n{:?}\n{:?}\n{:?}",
        command.mode,
        command.archive,
        command.compression.geometry,
        command.decompression.limits.max_output_bytes,
    ));
    if command.mode == Mode::Extract && !command.members.is_empty() {
        identity.update(format!("\n{:?}", command.members));
    }
    let identity = identity.finalize().into();
    let mut job = Job::open(
        &job_path,
        checkpoint.start,
        identity,
        (command.pause_after, command.pause_file.clone()),
    )
    .map_err(|e| e.to_string())?;
    if command.progress {
        job.enable_progress();
        eprintln!(
            "{}: preparing input for checkpointed job",
            if encoding { "compress" } else { "extract" }
        );
    }
    if let Some(password) = &command.archive.password_file {
        job.protect(password, (&command.compression, &command.decompression))
            .map_err(|e| e.to_string())?;
    }
    let result = if encoding && (directory || command.mode == Mode::Archive) {
        managed::create_with_job(
            &src,
            &dst,
            command.overwrite,
            (&command.archive, &command.compression, Some(&job)),
        )
        .map_err(|e| e.to_string())
    } else if !encoding && managed_input {
        managed::extract_members(
            &src,
            &dst,
            (command.overwrite, &command.selection),
            (&command.archive, &command.decompression, Some(&job)),
        )
        .map_err(|e| e.to_string())
    } else if encoding {
        let mut source = open_regular_file(&src).map_err(|e| e.to_string())?;
        let mut staged = tempfile::tempfile().map_err(|e| e.to_string())?;
        io::copy(&mut source, &mut staged).map_err(|e| e.to_string())?;
        atomic(&dst, command, |output, _| {
            job.encode(
                Section::Plain,
                &mut staged,
                output,
                (StreamKind::Bytes, &command.compression),
            )
            .map_err(|e| e.to_string())
        })
    } else {
        let mut input = file.take().ok_or("input required")?;
        input.rewind().map_err(|e| e.to_string())?;
        let mut staged = tempfile::tempfile().map_err(|e| e.to_string())?;
        io::copy(&mut input, &mut staged).map_err(|e| e.to_string())?;
        job.decode(Section::Plain, &mut staged, &command.decompression)
            .map_err(|e| e.to_string())
            .and_then(|(mut raw, kind)| match kind {
                StreamKind::DirectoryArchive => restore_catalogue_members(
                    &mut raw,
                    &dst,
                    command.overwrite,
                    (&command.decompression, &command.selection),
                )
                .map_err(|e| e.to_string()),
                StreamKind::Bytes => atomic(&dst, command, |output, _| {
                    io::copy(&mut raw, output)
                        .map(|_| ())
                        .map_err(|e| e.to_string())
                }),
            })
    };
    if job.paused() {
        eprintln!("Paused. Resume with --resume {:?}", job_path);
        return Ok(());
    }
    result?;
    if let Err(error) = job.complete() {
        eprintln!("Warning: output completed, but checkpoint cleanup failed: {error}");
    }
    Ok(())
}

fn list_archive(command: &Command) -> Result<(), String> {
    let source = command.input.as_deref().ok_or("archive input required")?;
    let inventory = managed::inspect(source, &command.archive, &command.decompression)
        .map_err(|error| error.to_string())?;
    let entries = inventory
        .into_iter()
        .map(|(path, entry)| (path, entry.map(|file| file.size)));
    if command.json {
        let items: Vec<_> = entries.map(|(path, size)| serde_json::json!({
            "path": path, "size": size, "type": if size.is_some() { "file" } else { "directory" }
        })).collect();
        println!(
            "{}",
            serde_json::to_string(&items).map_err(|error| error.to_string())?
        );
    } else {
        for (path, size) in entries {
            match size {
                Some(size) => println!("file\t{size}\t{path:?}"),
                None => println!("directory\t-\t{path:?}"),
            }
        }
    }
    Ok(())
}

fn diff_archive(command: &Command) -> Result<(), String> {
    let source = command.input.as_deref().ok_or("archive input required")?;
    let other = command.members.first().ok_or("comparison path required")?;
    let changes = managed::diff(source, other, &command.archive, &command.decompression)
        .map_err(|error| error.to_string())?;
    if command.json {
        println!(
            "{}",
            serde_json::to_string(
                &changes
                    .iter()
                    .map(|change| serde_json::json!({"path":change.path, "status":change.status}))
                    .collect::<Vec<_>>()
            )
            .map_err(|error| error.to_string())?
        );
    } else {
        for change in changes {
            println!("{}\t{:?}", change.status, change.path);
        }
    }
    Ok(())
}

fn archive_health(command: &Command) -> Result<(), String> {
    let source = command.input.as_deref().ok_or("archive input required")?;
    let report = managed::health(source, &command.archive, &command.decompression)
        .map_err(|error| error.to_string())?;
    if command.json {
        println!("{report}");
    } else {
        println!("{report:#}");
    }
    if matches!(
        report["status"].as_str(),
        Some("unrecoverable" | "unverified")
    ) {
        return Err("archive integrity verification failed".into());
    }
    Ok(())
}

fn dispatch(command: &Command) -> Result<(), String> {
    if let Some(checkpoint) = &command.checkpoint {
        return checkpointed(command, checkpoint);
    }
    if let (Mode::Extract, Some(index)) = (command.mode, &command.recovery_index) {
        let decoded = recovery::extract_indexed(
            command.input.as_deref().ok_or("archive input required")?,
            index,
            command
                .output
                .path()
                .ok_or("indexed extraction destination required")?,
            (&command.members, command.overwrite, &command.decompression),
        )
        .map_err(|error| error.to_string())?;
        if command.print_stats {
            eprintln!("Indexed extraction decoded {decoded} segments");
        }
        return Ok(());
    }
    match command.mode {
        Mode::List => list_archive(command),
        Mode::Diff => diff_archive(command),
        Mode::Health => archive_health(command),
        Mode::JobStatus => {
            let rows = bato_engine::core::jobs::status(
                command.input.as_deref().ok_or("job directory required")?,
            )
            .map_err(|error| error.to_string())?;
            if rows.is_empty() {
                println!("No committed segments yet.");
            }
            for row in rows {
                println!("{row}");
            }
            Ok(())
        }
        Mode::Compress | Mode::Archive => compress(command),
        Mode::Decompress | Mode::Extract | Mode::Test | Mode::Info => read_command(command),
        Mode::RecoveryIndex => recovery::create_index(
            command.input.as_deref().ok_or("archive input required")?,
            command.output.path().ok_or("index destination required")?,
            command.overwrite,
            &command.decompression,
        )
        .map_err(|e| e.to_string()),
        Mode::Recover => {
            let report = recovery::salvage(
                command.input.as_deref().ok_or("archive input required")?,
                command
                    .recovery_index
                    .as_deref()
                    .ok_or("recovery index required")?,
                command
                    .output
                    .path()
                    .ok_or("recovery destination required")?,
                &command.decompression,
            )
            .map_err(|e| e.to_string())?;
            println!(
                "Recovered {} verified files or segments. Missing {}. Damaged segments {}.",
                report.recovered, report.missing, report.damaged_segments
            );
            Ok(())
        }
        Mode::RepairSplit => managed::repair_split(
            command
                .input
                .as_deref()
                .ok_or("split archive input required")?,
            command
                .output
                .path()
                .ok_or("repair-split requires -o PATH")?,
        )
        .map_err(|e| e.to_string()),
        Mode::Add | Mode::Remove => {
            let edit = if command.mode == Mode::Add {
                Edit::Add(&command.members)
            } else {
                Edit::Remove(&command.members)
            };
            let src = command.input.as_deref().ok_or("archive path required")?;
            if let Some(warning) = managed::edit(
                src,
                edit,
                command.overwrite,
                (
                    &command.archive,
                    &command.compression,
                    &command.decompression,
                ),
            )
            .map_err(|e| e.to_string())?
            {
                eprintln!("Warning: archive updated, but old split volumes could not be removed: {warning}");
            }
            Ok(())
        }
    }
}

fn main() {
    let args: Vec<OsString> = env::args_os().collect();
    if args.len() == 1 {
        help();
        return;
    }
    let Some(command) = parse(&args) else {
        return;
    };
    dispatch(&command).unwrap_or_else(|e| die(e));
    if command.print_stats {
        eprintln!("execution stats: {:?}", command.execution.stats.snapshot());
    }
    if let Some(report) = command.execution.stats.timing_report() {
        eprint!("{report}");
    }
}

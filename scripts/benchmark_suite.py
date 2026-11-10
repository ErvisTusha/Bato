#!/usr/bin/env python3
"""Verified compression benchmarks with explicit profiles and CPU allocations."""
import argparse
import sys
import sqlite3
from bato_tools.archive_ops import run_archive_ops
from bato_tools.database import DEFAULT_DATABASE, prepare_database
from bato_tools.store import MeasurementStore
from bato_tools.benchmark import run_suite
from bato_tools.common import PROJECT_ROOT, ProcessFailure


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return value


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("Value must be nonnegative")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--corpus", default="multi_corpus")
    parser.add_argument("-t", "--target", action="append", default=[], help="Repeatable relative path, glob, or substring")
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--origin", action="append", choices=["real", "synthetic", "local"], help="Default: real")
    parser.add_argument("--include-local", action="store_true", help="Explicitly validate and include untracked input files for this run")
    parser.add_argument("--include-duplicates", action="store_true")
    parser.add_argument("--mode", choices=["quick", "rigorous"], default="quick")
    parser.add_argument("--suite", choices=["corpus", "stress", "full"], default="corpus", help="Full: corpus plus explicitly generated stress cases")
    parser.add_argument("--track", choices=["native", "common-tar", "both"], default="both")
    parser.add_argument("--profile", choices=["fast", "default", "high"], default="high")
    parser.add_argument("--codec-tuning", help="JSON compression parameters for zip/7z/rar/xz; recorded in codec identities")
    parser.add_argument("--nncp-profile", choices=["default","enwik8","enwik9","lstm","lstm_fast"], default="default",
                        help="NNCP model profile; default is generic, enwik profiles are text-specific")
    parser.add_argument("--nncp-cuda", action="store_true", help="Explicit NNCP CUDA execution (separate hardware track)")
    parser.add_argument("--tool-check-timeout", type=positive, default=30,
                        help="Seconds per optional codec preflight; CMIX/NNCP may need a much larger allowance")
    parser.add_argument("--cpu-limit", type=positive, help="CPU allocation; default: all affinity/quota-available CPUs")
    parser.add_argument("--threads", type=positive, help="Worker limit; default: CPU allocation (Bato limit 256)")
    parser.add_argument("--bato-modes", default="cpu", help="Comma-separated cpu1,cpu (one worker, all allocated workers); default: cpu")
    parser.add_argument("--bato-segment-size", type=positive, default=64 * 2**20,
                        help="CM segment bytes: power of two from 64 KiB to 1 GiB")
    parser.add_argument("--bato-memory-budget", type=positive, help="Managed Bato allowance; default: 90%% of usable RAM, frozen for the run")
    parser.add_argument("--bato-chunk-size", type=positive, default=256 * 1024,
                        help="Bato chunk bytes: power of two from 4 KiB to 1 MiB; recorded in the codec identity")
    parser.add_argument("--campaign-manifest", help="Explicit whole-file and whole-directory workload list")
    parser.add_argument("--frozen-workloads", help="Reject changes from a saved --list-workloads JSON manifest")
    parser.add_argument("--allow-unavailable", action="store_true", help="Record missing/failed tools and measure only ready codecs")
    parser.add_argument("--list-workloads", action="store_true", help="Print the manifest without generating fixtures, checking tools, or measuring")
    parser.add_argument("--workload", choices=["files", "directories", "both"], default="both")
    parser.add_argument("--bato-bin", help="Path to the Rust Bato executable")
    parser.add_argument("--only-bato", action="store_true")
    parser.add_argument("--codecs", help="Comma-separated bato,kgb,paq8px,paq8l,zpaq,cmix,nncp,rar,7z,xz,zstd,gzip,lz4,zip,bzip2; cmix/nncp require opt-in")
    parser.add_argument('--ensure-competitors', action='store_true',
                        help='Measure only missing compatible competitor cases; reference original verified trials')
    parser.add_argument("--check-tools", action="store_true", help="Check selected tools at extreme settings without a corpus or measurements")
    parser.add_argument("--repetitions", type=positive)
    parser.add_argument("--warmups", type=nonnegative)
    parser.add_argument("--large-repetitions", type=positive, help="Override repetitions for >1 MiB or >10,000 files (default: 1)")
    parser.add_argument("--large-warmups", type=nonnegative, help="Override warmups for large cases (default: 0)")
    parser.add_argument("--timeout", type=positive, default=600, help="Seconds per preparation, compression, decompression, and verification phase")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=positive, default=50, help="Quick sample file cap")
    parser.add_argument("--sample-bytes", type=positive, default=256 * 1024 * 1024, help="Quick sample payload cap")
    parser.add_argument("--max-archives", type=positive, default=10, help="Quick sample directory cap, independent of file sampling")
    parser.add_argument("--max-file-bytes", type=positive, default=4 * 2**30)
    parser.add_argument("--max-directory-bytes", type=positive, default=4 * 2**30)
    parser.add_argument("--max-directory-files", type=positive, default=100000)
    parser.add_argument("--memory-budget", type=positive, help="Per-process address-space cap; default: effective RAM (managed Bato allowance is separately limited to 90%% of usable RAM)")
    parser.add_argument("--label", help="Human build/run label; binary hashes remain authoritative")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts/benchmarks"))
    parser.add_argument("--scratch-dir", help="Directory for isolated temporary workspaces")
    parser.add_argument("--checkpoint-dir", help="Mirror committed database and reports after workloads, outside timed trials")
    parser.add_argument("--keep-archives", help="Retain the first verified measured archive per case outside timed phases")
    parser.add_argument("--resume", help="Run ID from this output directory; requires matching settings and inputs")
    parser.add_argument("--archive-operations", action="store_true")
    parser.add_argument("--archive-operation-format", choices=("indexed", "plain"), default="indexed")
    parser.add_argument("--bato-revision", help="Git revision declared for the operation benchmark binary")
    parser.add_argument("--baseline-bato-bin")
    parser.add_argument("--baseline-bato-revision")
    parser.add_argument('--database', default=str(DEFAULT_DATABASE), help='Canonical SQLite measurement database')
    parser.add_argument('--init-db', action='store_true', help='Initialize database only; do not discover or benchmark inputs')
    args = parser.parse_args(argv)
    try:
        if args.ensure_competitors and (not args.codecs or any(c.strip().startswith('bato') for c in args.codecs.split(',')) or args.only_bato):
            parser.error('--ensure-competitors requires an explicit competitor-only --codecs list')
        if args.init_db:
            from bato_tools.runner import host_measurement_lock
            from pathlib import Path
            with host_measurement_lock():
                if Path(args.database).resolve() == DEFAULT_DATABASE.resolve():
                    prepare_database(args.database)
                Path(args.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
                store = MeasurementStore(Path(args.database).expanduser())
                try:
                    if store.connection.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                        raise ValueError('Database integrity check failed')
                finally:
                    store.close()
            print(f'Database initialized: {args.database}')
            return 0
        return run_archive_ops(args) if args.archive_operations else run_suite(args)
    except KeyboardInterrupt:
        message = "Interrupted. Completed trials are preserved in the database."
        if not args.archive_operations:
            message += " Resume with the same settings and --resume RUN_ID."
        print(message, file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, sqlite3.Error, MemoryError, ProcessFailure) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

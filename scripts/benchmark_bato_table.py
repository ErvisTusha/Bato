#!/usr/bin/env python3
"""Compare two Bato revisions and print a table after each verified workload.

Rebuild and use the local release binary for the branch (see --no-rebuild) and
report actual CUDA, native HIP/Level Zero/Metal and wgpu filter dispatches.
"""
import argparse
import contextlib
import fcntl
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import benchmark_suite
from bato_tools.common import PROJECT_ROOT, atomic_json, sha256, snapshot
from bato_tools.measurement import FAILURES
from bato_tools.resources import available_cpu_count, memory_state, resolve_memory_budget
from bato_tools.table_store import DEFAULT_TABLE_DATABASE, VERIFIED, TableStore, condition, hardware_identity
from bato_tools.workloads import identify

CORPUS = Path('/home/et/data-research/corpus/real')
MAIN = 'origin/main'
BRANCH = 'origin/claude/focused-faraday-t2c712'
LOCAL_BRANCH_BIN = PROJECT_ROOT / 'target' / 'release' / 'bato'
MIN_DIR = 20_000_000
MAX_DIR = 50_000_000
MAX_FILE = 5_000_000
CPU_COMPUTE = ('cpu', 'legacy')
DISPATCH_KEYS = ('cuda_filter_dispatches', 'wgpu_filter_dispatches', 'native_filter_dispatches')
COLUMN_WIDTHS = (35, 15, 33, 33, 37, 11)
TABLE_WIDTH = sum(COLUMN_WIDTHS) + len(COLUMN_WIDTHS) - 1
# Row keys of the three measured variants, in column order.
MAIN_CPU, BRANCH_CPU, BRANCH_GPU = 'main CPU', 'branch CPU', 'branch GPU'


def select_inputs(root, rng=None):
    """Sample five files and two complete directories from the corpus."""
    rng = rng or random.Random()
    files = []
    directories = []
    sizes = {}
    # Accumulate each directory once, including nested files and invalid entries.
    for parent, subdirs, names in os.walk(root, topdown=False, followlinks=False):
        directory = Path(parent)
        total = 0
        valid = True
        for name in names:
            path = directory / name
            if path.is_symlink() or not path.is_file():
                valid = False
                continue
            size = path.stat().st_size
            total += size
            if 100_000 <= size < MAX_FILE:
                files.append(path)
        for name in subdirs:
            child = directory / name
            if child.is_symlink() or child not in sizes or sizes[child] is None:
                valid = False
            else:
                total += sizes[child]
        sizes[directory] = total if valid else None
        if directory != root and valid and MIN_DIR <= total <= MAX_DIR:
            directories.append(directory)
    if len(files) < 5 or len(directories) < 2:
        raise ValueError('Need at least five usable files and two complete 20–50 MB directories in the corpus')
    # Sorting makes a supplied seed reproducible regardless of filesystem order.
    return rng.sample(sorted(files), 5), rng.sample(sorted(directories), 2)


def prepare_workloads(root, seed, inputs=()):
    """Identify the named inputs, or a seeded sample of the corpus when none are named."""
    if inputs:
        paths = [Path(path).resolve() for path in inputs]
    else:
        files, directories = select_inputs(root, random.Random(seed))
        paths = files + directories
    workloads = []
    for path in paths:
        item = {'target': str(path), 'path': str(path), 'type': 'directory_archive' if path.is_dir() else 'single_file',
                'category': 'directory' if path.is_dir() else path.suffix.lower(), 'origin': 'real',
                'snapshot': snapshot(path), 'scenario': 'corpus'}
        workloads.append(identify(item))
    return workloads


def cached_build(path, revision):
    """Trust a cached binary only when its saved identity matches its bytes."""
    try:
        metadata = json.loads((path / 'build.json').read_text())
        binary = path / 'bato'
        if metadata['revision'] != revision or sha256(binary) != metadata['sha256']:
            return None
        return {**metadata, 'binary': str(binary)}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def import_previous_build(root, revision, destination):
    """Reuse completed earlier runs only after verifying their recorded hash."""
    for manifest in sorted(root.glob('*/builds.json'), reverse=True):
        try:
            builds = json.loads(manifest.read_text())['builds']
            for build in builds.values():
                binary = Path(build['binary'])
                if build['revision'] != revision or sha256(binary) != build['sha256']:
                    continue
                shutil.copy2(binary, destination / 'bato')
                atomic_json(destination / 'build.json', {key: build[key] for key in ('revision', 'sha256', 'version')})
                return True
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return False


def build_revision(revision, root):
    sha = subprocess.check_output(['git', 'rev-parse', '--verify', revision + '^{commit}'], cwd=PROJECT_ROOT, text=True).strip()
    cache = root / 'build-cache'
    cache.mkdir(parents=True, exist_ok=True)
    # Serialize builders and publish a complete binary and metadata together.
    with (cache / f'{sha}.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        destination = cache / sha
        found = cached_build(destination, sha)
        if found:
            print(f'Reusing build {sha[:12]}', flush=True)
            return found
        with tempfile.TemporaryDirectory(prefix=f'.{sha}-', dir=cache) as temp:
            stage = Path(temp)
            if not import_previous_build(root, sha, stage):
                print(f'Building {sha[:12]}', flush=True)
                source = stage / 'source'
                source.mkdir()
                with tempfile.TemporaryFile() as archive:
                    subprocess.run(['git', 'archive', '--format=tar', sha], cwd=PROJECT_ROOT, stdout=archive, check=True)
                    archive.seek(0)
                    subprocess.run(['tar', '-xf', '-', '-C', str(source)], stdin=archive, check=True)
                subprocess.run(['cargo', 'build', '--release', '--bin', 'bato', '--locked'], cwd=source, check=True)
                binary = source / 'target' / 'release' / 'bato'
                version = subprocess.check_output([str(binary), '--version'], text=True).strip()
                shutil.copy2(binary, stage / 'bato')
                atomic_json(stage / 'build.json', {'revision': sha, 'sha256': sha256(stage / 'bato'), 'version': version})
                shutil.rmtree(source)
            else:
                print(f'Reusing verified previous build {sha[:12]}', flush=True)
            if destination.exists():
                shutil.rmtree(destination)
            stage.rename(destination)
        found = cached_build(destination, sha)
        if not found:
            raise ValueError(f'Cached build failed verification: {sha}')
        return found


def rebuild_local_binary():
    """Bring target/release/bato up to date; cargo does nothing when it is fresh."""
    print('Updating local release binary (cargo build --release --bin bato)', flush=True)
    subprocess.run(['cargo', 'build', '--release', '--bin', 'bato', '--locked'], cwd=PROJECT_ROOT, check=True)


def run_case(workload, variant, output, run):
    """Measure one variant of one input through the suite's verified workflow."""
    args, resources = run['args'], run['resources']
    build, compute = variant['build'], variant['compute']
    run_output = output / 'cases' / f'{workload["workload_id"]}-{variant["name"]}'
    run_output.mkdir(parents=True)
    manifest = run_output / 'manifest.json'
    atomic_json(manifest, {'schema_version': 4, 'workloads': [workload]})
    command = ['--corpus', str(args.corpus),
               '--campaign-manifest', str(manifest), '--frozen-workloads', str(manifest), '--track', 'native',
               '--mode', 'rigorous', '--codecs', 'bato', '--bato-bin', build['binary'], '--bato-modes', 'cpu',
               '--cpu-limit', str(resources['cpus']), '--threads', str(resources['cpus']),
               '--memory-budget', str(resources['memory']), '--bato-memory-budget', str(resources['managed']),
               '--warmups', str(args.warmups), '--repetitions', str(args.repetitions),
               '--large-warmups', str(args.warmups), '--large-repetitions', str(args.repetitions),
               '--timeout', str(args.timeout), '--output-dir', str(run_output), '--scratch-dir', str(output / 'scratch'),
               '--label', f'{variant["name"]} {build["revision"]}', '--bato-compute', compute]
    if args.bato_segment_size is not None:
        command.extend(['--bato-segment-size', str(args.bato_segment_size)])
    if args.bato_chunk_size is not None:
        command.extend(['--bato-chunk-size', str(args.bato_chunk_size)])
    rows = []
    diagnostics = io.StringIO()
    with contextlib.redirect_stdout(diagnostics), contextlib.redirect_stderr(diagnostics):
        status = benchmark_suite.main(command, result_rows=rows)
    if status == 130:
        raise KeyboardInterrupt
    if len(rows) != 1:
        detail = diagnostics.getvalue().strip() if status else f'expected one result, got {len(rows)}'
        return {'status': 'FAIL', 'error': detail or f'benchmark exited {status}'}
    row = rows[0]
    if status and row['status'] not in FAILURES:
        return {'status': 'FAIL', 'error': diagnostics.getvalue().strip() or f'benchmark exited {status}'}
    if compute not in CPU_COMPUTE and row['verified_lossless']:
        row['gpu_phases'] = ''.join(
            phase if row.get(f'{prefix}_execution_stats') and all(
                sum((stats or {}).get(key, 0) for key in DISPATCH_KEYS) > 0
                for stats in row[f'{prefix}_execution_stats']
            ) else '-'
            for phase, prefix in [('E', 'comp'), ('D', 'decomp')]
        )
    return row


def compute_for(build):
    """CPU compute flag for a build: binaries without --compute take the legacy path."""
    return 'cpu' if build.get('gpu_compute_modes') else 'legacy'


def build_capabilities(build):
    help_result = subprocess.run([build['binary'], '--help'], capture_output=True, text=True, check=True)
    help_text = help_result.stdout + help_result.stderr
    found = re.search(r'--compute\s+([a-z0-9|-]+)', help_text)
    modes = set(found.group(1).split('|')) if found else set()
    return {**build, 'gpu_compute_modes': sorted(modes),
            'cuda_filters': 'cuda' in modes, 'wgpu_filters': 'wgpu' in modes}


def table_line(values):
    return ' '.join(
        f'{value:{"<" if index == 0 else ">"}{width}}'
        for index, (value, width) in enumerate(zip(values, COLUMN_WIDTHS, strict=True))
    )


def print_table_header(gpu_compute):
    print('\nSizes are in decimal KB (1 KB = 1,000 bytes); times are compression/decompression seconds (medians).')
    print('Elapsed = time spent on this input across variants, including warmups and verification.')
    print('* marks a phase with at least one GPU dispatch in every measured trial.')
    print('= marks a stored result reused from the database, not measured in this run (dates listed below).')
    print(table_line(['Input', 'Original KB', 'Main CPU', 'Branch CPU', f'Branch {gpu_compute} (GPU + CPU)', 'Elapsed']))
    print('-' * TABLE_WIDTH, flush=True)


def format_duration(seconds):
    return f'{seconds:.2f} S'


def print_table_row(workload, rows, *, elapsed=None, refresh=False, final=True):
    def cell(row):
        if row.get('status') != VERIFIED:
            return row.get('status', 'FAIL')
        phases = row.get('gpu_phases', '')
        comp_mark = '*' if 'E' in phases else ''
        dec_mark = '*' if 'D' in phases else ''
        comp = f'{row["comp_time_s"]:.2f}'.replace('.', ',')
        dec = f'{row["decomp_time_s"]:.2f}'.replace('.', ',')
        reused = '=' if row.get('stored_at') else ''
        return f'{reused}{row["comp_size"] / 1000:,.2f} KB {comp}{comp_mark}/{dec}{dec_mark}'
    label = Path(workload['target']).name
    label_width = COLUMN_WIDTHS[0]
    if len(label) > label_width:
        label = label[:label_width - 3] + '...'
    prefix = '\r\033[2K' if refresh else ''
    line = table_line([
        label, f'{workload["snapshot"]["size"] / 1000:,.2f}',
        *(cell(rows[key]) for key in (MAIN_CPU, BRANCH_CPU, BRANCH_GPU)),
        format_duration(elapsed) if elapsed is not None else '-',
    ])
    print(prefix + line, end='\n' if final or not refresh else '', flush=True)


def format_usage(row, phase):
    trials = row['trials']
    child_cpu_times = [trial.get(f'{phase}_child_cpu_s') for trial in trials]
    wall = sum(trial[f'{phase}_time_s'] for trial in trials)
    cpu = 'unavailable'
    if wall > 0 and all(seconds is not None for seconds in child_cpu_times):
        cpu = f'{sum(child_cpu_times) / wall:.2f} logical CPUs on average'
    backend = row['effective_settings']['compute']
    total = sum((trial.get(f'{phase}_execution_stats') or {}).get(key, 0)
                for trial in trials for key in DISPATCH_KEYS)
    devices = sorted({device for trial in trials
                      for device in (trial.get(f'{phase}_execution_stats') or {}).get('gpu_devices', [])})
    return f'CPU {cpu}, {backend} GPU launches {total}, devices {devices}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=CORPUS)
    parser.add_argument('--seed', type=int, help='Random selection seed (printed for reproducibility)')
    parser.add_argument('--input', type=Path, action='append', default=[], metavar='PATH', dest='inputs',
                        help='Benchmark this file or directory instead of a random sample (repeatable)')
    parser.add_argument('--database', type=Path, default=DEFAULT_TABLE_DATABASE,
                        help=f'SQLite store of results; a stored result for the same binary, input, settings '
                             f'and hardware is reused instead of measured again (default: {DEFAULT_TABLE_DATABASE})')
    parser.add_argument('--refresh', action='store_true',
                        help='Measure every variant again even when the database holds a matching result')
    parser.add_argument('--gpu-compute', default='auto', metavar='MODE',
                        help='Compute mode of the third column, e.g. auto, cuda or wgpu (default: auto)')
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'artifacts' / 'bato-table-benchmark',
                        help='Directory for cached builds')
    parser.add_argument('--scratch', type=Path, default=Path(tempfile.gettempdir()) / 'bato-table-scratch')
    parser.add_argument('--main-revision', default=MAIN)
    branch_source = parser.add_mutually_exclusive_group()
    branch_source.add_argument('--branch-revision', help=f'Build this Git revision instead of using the local binary (fallback: {BRANCH})')
    branch_source.add_argument('--branch-bin', type=Path, help='Use this branch executable (default: target/release/bato when present)')
    parser.add_argument('--no-rebuild', action='store_true',
                        help='Use the existing target/release/bato without running cargo build first')
    parser.add_argument('--warmups', type=int, default=0, help='Warmup runs per variant (default: 0)')
    parser.add_argument('--repetitions', type=int, default=1, help='Measured runs per variant (default: 1)')
    parser.add_argument('--bato-segment-size', type=benchmark_suite.positive, metavar='BYTES',
                        help='Test a specific segment size across all variants (suite default: 64 MiB)')
    parser.add_argument('--bato-chunk-size', type=benchmark_suite.positive, metavar='BYTES',
                        help='Test a specific chunk size, at most the segment size (suite default: 256 KiB)')
    parser.add_argument('--show-usage', action='store_true',
                        help='Show average CPU usage and GPU launches across measured runs')
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1 or args.timeout < 1:
        parser.error('warmups must be nonnegative; repetitions and timeout must be positive')
    args.corpus = args.corpus.resolve()
    args.scratch = args.scratch.resolve()
    args.output = args.output.resolve()
    args.database = args.database.resolve()
    if any(path.is_relative_to(args.corpus) for path in (args.output, args.scratch, args.database)):
        parser.error('output, scratch and database must be outside the corpus')
    args.scratch.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix='bato-table-', dir=args.scratch) as temporary, \
            contextlib.closing(TableStore(args.database)) as store:
        status = run_table(args, Path(temporary), store)
    print(f'Benchmark elapsed: {format_duration(perf_counter() - started)}', flush=True)
    return status


def select_branch(args, output):
    """Copy the supplied or local branch binary, or build the requested revision."""
    branch_bin = args.branch_bin
    if branch_bin is None and args.branch_revision is None:
        if not args.no_rebuild:
            rebuild_local_binary()
        if LOCAL_BRANCH_BIN.is_file():
            branch_bin = LOCAL_BRANCH_BIN
    if branch_bin is None:
        return build_revision(args.branch_revision or BRANCH, args.output)
    supplied = branch_bin.resolve()
    digest = sha256(supplied)
    binary = output / 'binaries' / 'branch-bato'
    binary.parent.mkdir()
    shutil.copy2(supplied, binary)
    if sha256(binary) != digest:
        raise ValueError(f'Branch binary changed while copying: {supplied}')
    print(f'Using local branch binary {supplied}', flush=True)
    return {'binary': str(binary), 'source_binary': str(supplied),
            'revision': 'local-binary', 'sha256': digest,
            'version': subprocess.check_output([str(binary), '--version'], text=True).strip()}


def measure_or_reuse(workload, variant, store, run):
    """Reuse a stored verified result for identical conditions; otherwise measure and store it."""
    args = run['args']
    wanted = condition(workload, variant['build'], variant['compute'], run['settings'])
    found = None if args.refresh else store.lookup(wanted, args.warmups, args.repetitions)
    if found:
        result, stored_at = found
        return {**result, 'stored_at': stored_at}
    result = run_case(workload, variant, run['output'], run)
    store.save(wanted, variant['build'], result, (args.warmups, args.repetitions))
    return result


def run_table(args, output, store):
    if args.inputs:
        print(f'Inputs: {len(args.inputs)} named', flush=True)
    else:
        args.seed = args.seed if args.seed is not None else random.SystemRandom().getrandbits(64)
        print(f'Selection seed: {args.seed}', flush=True)
    workloads = prepare_workloads(args.corpus, args.seed, args.inputs)
    main = build_capabilities(build_revision(args.main_revision, args.output))
    branch = build_capabilities(select_branch(args, output))
    modes = branch['gpu_compute_modes']
    print(f'Branch compute modes: {", ".join(modes) if modes else "legacy CPU-only"}', flush=True)
    gpu_supported = args.gpu_compute in modes and args.gpu_compute not in CPU_COMPUTE
    variants = [{'label': MAIN_CPU, 'name': 'main', 'build': main, 'compute': compute_for(main)},
                {'label': BRANCH_CPU, 'name': 'branch', 'build': branch, 'compute': compute_for(branch)}]
    if gpu_supported:
        variants.append({'label': BRANCH_GPU, 'name': args.gpu_compute, 'build': branch, 'compute': args.gpu_compute})
    memory = resolve_memory_budget()
    resources = {'cpus': min(available_cpu_count(), 256), 'memory': memory,
                 'managed': max(1, min(memory, memory_state()['available_ram_bytes']) * 9 // 10)}
    # The managed budget follows free RAM at start-up, so it is recorded with each
    # result but kept out of the reuse key; the outer memory cap is part of the key.
    settings = {'cpus': resources['cpus'], 'memory': memory, 'segment_size': args.bato_segment_size,
                'chunk_size': args.bato_chunk_size, 'hardware': hardware_identity()}
    run = {'args': args, 'resources': resources, 'settings': settings, 'output': output}
    failures = []
    reused = []
    print(f'Per variant: {args.warmups} warmup(s), {args.repetitions} measured repetition(s).')
    print(f'Result database: {args.database}')
    if args.bato_segment_size is not None or args.bato_chunk_size is not None:
        segment = args.bato_segment_size or 'suite default'
        chunk = args.bato_chunk_size or 'suite default'
        print(f'Geometry override (bytes): segment {segment}, chunk {chunk}.')
    if args.show_usage:
        print(f'CPU allocation: {resources["cpus"]} logical CPUs.')
    print_table_header(args.gpu_compute)
    refresh = sys.stdout.isatty() and shutil.get_terminal_size().columns >= TABLE_WIDTH
    for workload in workloads:
        started = perf_counter()
        rows = {
            MAIN_CPU: {'status': 'PENDING'},
            BRANCH_CPU: {'status': 'PENDING'},
            BRANCH_GPU: {'status': 'PENDING'} if gpu_supported else {
                'status': 'UNSUPPORTED', 'error': f'Selected branch binary has no {args.gpu_compute} compute option'},
        }
        print_table_row(workload, rows, refresh=refresh, final=False)
        for index, variant in enumerate(variants, 1):
            label = variant['label']
            rows[label] = measure_or_reuse(workload, variant, store, run)
            print_table_row(workload, rows, elapsed=perf_counter() - started,
                            refresh=refresh, final=index == len(variants))
            if rows[label].get('stored_at'):
                reused.append(f'{Path(workload["target"]).name} ({label}): measured {rows[label]["stored_at"]}')
            if rows[label]['status'] != VERIFIED:
                error = rows[label].get('error') or rows[label]['status']
                failures.append(f'{workload["target"]} ({label}): {error}')
        if args.show_usage:
            for variant in variants:
                row = rows[variant['label']]
                if row.get('trials') and row['status'] == VERIFIED:
                    comp = format_usage(row, 'comp')
                    dec = format_usage(row, 'decomp')
                    print(f'  {variant["label"]} usage | compression: {comp} | decompression: {dec}', flush=True)
    if reused:
        print('Reused stored results (=), not measured in this run:')
        for line in reused:
            print(f'  {line}')
    for failure in failures:
        print(failure, file=sys.stderr)
    return int(not gpu_supported or bool(failures))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted.', file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(2)

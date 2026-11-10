"""One bounded benchmark phase; codec children remain in this worker's process group."""
import json
import os
import resource
import sys
import tarfile
import tempfile
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bato_tools.codecs import Codec
from bato_tools.common import ProcessFailure, atomic_json, make_tar, snapshot, sha256, extract_tar
from bato_tools.resources import ResourceLimit, limit_address_space
from bato_tools.workloads import materialize


IO_FIELDS = ('rchar', 'wchar', 'syscr', 'syscw', 'read_bytes', 'write_bytes', 'cancelled_write_bytes')


def io_counters():
    """Linux workflow counters, including waited-for children, not device energy.

    Reading self is independent of the externally visible PID namespace.
    Storage-layer writes are charged at dirtying time; cancellations remain
    separate, and neither counter proves physical SSD/HDD traffic.
    """
    try:
        fields = dict(line.split(':', 1) for line in Path('/proc/self/io').read_text().splitlines())
        return {name: int(fields[name]) for name in IO_FIELDS}
    except (OSError, KeyError, ValueError):
        return None


def io_delta(before, after):
    if before is None or after is None or any(after[k] < before[k] for k in IO_FIELDS):
        return {f'io_{k}': None for k in IO_FIELDS}
    return {f'io_{k}': after[k] - before[k] for k in IO_FIELDS}


def execute(spec):
    limit_address_space(spec.get('memory_limit'))
    os.environ['BATO_BENCH_STAGE_WORKER'] = '1'
    action = spec['action']
    if action == 'prepare':
        result = materialize(spec['workload'], spec['destination'])
        if spec.get('tar_path'):
            packing_started = time.perf_counter()
            make_tar(spec['destination'], spec['tar_path'], result['snapshot'])
            result['tar_packaging_s'] = time.perf_counter() - packing_started
            result['tar_snapshot'] = snapshot(spec['tar_path'])
            if snapshot(spec['destination']) != result['snapshot']:
                raise ValueError('Input changed while preparing the common tar')
        return result
    if action == 'verify':
        if snapshot(spec['source']) != spec['expected']:
            raise ValueError('Restored contents or relative paths differ')
        if spec.get('tar_directory_snapshot') is not None:
            expected = spec['tar_directory_snapshot']
            with tempfile.TemporaryDirectory(prefix='verify-tar-', dir=Path(spec['source']).parent) as temporary:
                restored = Path(temporary)/'tree'
                extract_tar(spec['source'], restored, expected['size'])
                if snapshot(restored) != expected:
                    raise ValueError('Restored tar files or directories differ')
        return {'verified_lossless': True}
    if action == 'compress':
        # Check membership and sizes outside measured workflow time. Hashes are
        # checked during staging and full output verification; an unlisted file
        # must never silently enter a frozen directory benchmark.
        expected = spec['expected']
        source = Path(spec['source'])
        if spec['directory']:
            from bato_tools.common import regular_files, regular_directories
            actual = sorted((p.relative_to(source).as_posix(), p.stat().st_size) for p in regular_files(source))
            frozen = sorted((e['path'], e['size']) for e in expected['files'])
            if actual != frozen or regular_directories(source) != expected['directories']:
                raise ValueError('Frozen input inventory changed before compression')
        elif source.stat().st_size != expected['size']:
            raise ValueError('Frozen input size changed before compression')
    own_before, child_before = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    io_before = io_counters()
    measured = Codec(**spec['codec']).workflow(action, spec['source'], spec['output'], spec['work'],
                                             spec['timeout'] + 60, spec['expected'], spec['directory'])
    own_after, child_after = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    io_after = io_counters()
    cpu = lambda usage: usage.ru_utime + usage.ru_stime
    result = asdict(measured)
    result.update(io_delta(io_before, io_after))
    result['io_scope'] = 'isolated_workflow_linux_self_and_waited_children' if io_before is not None and io_after is not None else 'unavailable'
    result['physical_device_bytes'] = None
    result.update(workflow_cpu_s=cpu(own_after) - cpu(own_before) + cpu(child_after) - cpu(child_before),
                  worker_peak_rss_kib=own_after.ru_maxrss,
                  cpu_scope='isolated_workflow_including_container_and_children',
                  worker_rss_scope='isolated_worker_lifetime_including_imports')
    if action == 'compress':
        output = Path(spec['output'])
        if not output.is_file() or not output.stat().st_size:
            raise ValueError('Compressor produced no archive')
        result['archive_sha256'] = sha256(output)
    return result


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    started = time.monotonic()
    try:
        result = {'status': 'PASS', **execute(spec)}
    except (MemoryError, ResourceLimit) as error:
        result = {'status': 'RESOURCE_LIMIT', 'error': str(error) or 'Address-space budget exceeded'}
    except (OSError, ValueError, ProcessFailure, tarfile.TarError, zipfile.BadZipFile, EOFError) as error:
        status = 'FAIL'
        if isinstance(error, ProcessFailure):
            status = 'TIMEOUT' if error.timed_out else 'FAIL'
        if spec.get('memory_limit') and any(marker in str(error).lower() for marker in ['cannot allocate memory', 'out of memory', 'failed to allocate', 'memoryerror', 'bad_alloc', 'memory allocation']):
            status = 'RESOURCE_LIMIT'
        result = {'status': status, 'error': str(error)}
        if isinstance(error, ProcessFailure):
            result.update(failed_command=error.command, returncode=error.returncode)
    result.update(stage=spec['action'], phase_elapsed_s=time.monotonic() - started)
    atomic_json(sys.argv[2], result)


if __name__ == '__main__':
    main()

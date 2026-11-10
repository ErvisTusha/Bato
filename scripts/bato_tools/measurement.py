"""Verified round trips and aggregates with explicit timing/resource scopes."""
from __future__ import annotations

import json
import shutil
import statistics
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path

from .common import ProcessFailure, atomic_json, digest_json, run_process, sha256, snapshot, utc_now, extract_tar
from .resources import ResourceLimit
from .workloads import identify

WORKER = Path(__file__).with_name('stage_worker.py')
FAILURES = {'FAIL', 'TIMEOUT', 'RESOURCE_LIMIT', 'UNSUPPORTED'}


def run_stage(spec, work, timeout, memory_limit=None, label=''):
    work = Path(work)
    action = spec['action']
    request, response = work / f'{action}-request.json', work / f'{action}-response.json'
    atomic_json(request, {**spec, 'timeout': timeout, 'memory_limit': memory_limit})
    response.unlink(missing_ok=True)
    try:
        usage = run_process([sys.executable, str(WORKER), str(request), str(response)], work / f'collect-{action}', timeout,
                            progress=lambda elapsed: print(f'{label} | {action} | {elapsed:.0f}s elapsed', flush=True), monitor_tree=True)
    except ProcessFailure as error:
        memory_failure = memory_limit and any(marker in str(error).lower() for marker in ['cannot allocate memory', 'out of memory', 'memoryerror', 'failed to allocate', 'bad_alloc', 'memory allocation'])
        return {'status': 'TIMEOUT' if error.timed_out else 'RESOURCE_LIMIT' if memory_failure else 'FAIL', 'stage': action, 'error': str(error),
                'failed_command': error.command, 'returncode': error.returncode}
    if not response.is_file():
        return {'status': 'FAIL', 'stage': action, 'error': 'Phase produced no result'}
    result = json.loads(response.read_text())
    if action in {'compress', 'decompress'}:
        result['sampled_tree_peak_rss_kib'] = usage.sampled_tree_peak_rss_kib
        result['tree_rss_scope'] = '100ms_process_tree_samples_lower_bound' if usage.sampled_tree_peak_rss_kib is not None else 'unavailable'
    return result


def base_record(codec, workload, repetitions, warmups):
    workload = identify(workload) if 'workload_id' not in workload else workload
    track = workload.get('track', 'native' if workload['type'] == 'directory_archive' else 'direct')
    expected = workload.get('snapshot', {}) or {}
    return {'schema_version': 4, 'case_id': digest_json([workload['workload_id'], track, codec.identifier]),
            'workload_id': workload['workload_id'], 'target': workload['target'], 'type': workload['type'],
            'track': track, 'category': workload['category'], 'origin': workload['origin'],
            'scenario': workload.get('scenario', 'corpus'), 'size_band': workload['size_band'], 'file_count': workload['file_count'],
            'raw_size': workload['logical_payload_bytes'], 'logical_payload_bytes': workload['logical_payload_bytes'],
            'input_bytes_scope': 'encoder_input_including_prepared_container_or_bato_catalog',
            'input_sha256': expected.get('sha256'), 'metadata_policy': workload['metadata_policy'],
            'codec': codec.identifier, 'compressor': codec.name, 'effective_settings': codec.settings(),
            'compression_profile': 'extreme' if codec.profile == 'high' else codec.profile, 'cache_origin': 'fresh', 'verified_lossless': False, 'status': 'PENDING',
            'repetitions': repetitions, 'warmups': warmups, 'measured_at': utc_now(),
            'verification_scope': 'file_bytes_paths_and_all_directories' if workload['type'] == 'directory_archive' else 'file_bytes'}


def measure_trial(codec, workload, timeout, scratch=None, isolated=False, memory_limit=None, retain_archive=None):
    expected = workload['snapshot']
    directory = workload['type'] == 'directory_archive' and workload.get('track', 'native') != 'common-tar'
    source = workload.get('tar_path') if workload.get('track') == 'common-tar' else workload['path']
    codec_expected = workload['tar_snapshot'] if workload.get('track') == 'common-tar' else expected
    stage = 'compress'
    try:
        with tempfile.TemporaryDirectory(prefix='trial-', dir=scratch) as temporary:
            work = Path(temporary)
            archive, restored = work / ('compressed' + codec.archive_suffix(directory)), work / 'restored'
            metrics = []
            for stage, input_path, output_path in [('compress', source, archive), ('decompress', archive, restored)]:
                if isolated:
                    result = run_stage({'action': stage, 'codec': asdict(codec), 'source': str(input_path), 'output': str(output_path),
                                        'work': str(work), 'expected': codec_expected, 'directory': directory}, work, timeout, memory_limit,
                                       f"{workload['target']} | {codec.name}")
                    if result['status'] != 'PASS':
                        return result
                else:
                    result = asdict(codec.workflow(stage, input_path, output_path, work, timeout, codec_expected, directory))
                metrics.append(result)
                if stage == 'compress' and (not archive.is_file() or not archive.stat().st_size):
                    raise ValueError('Compressor produced no archive')
            stage = 'verify'
            if isolated:
                verification = run_stage({'action': stage, 'source': str(restored), 'expected': codec_expected,
                                          'tar_directory_snapshot': expected if workload.get('track') == 'common-tar' else None}, work, timeout, memory_limit)
                if verification['status'] != 'PASS':
                    return verification
            elif snapshot(restored) != codec_expected:
                raise ValueError('Restored contents or relative paths differ')
            elif workload.get('track') == 'common-tar':
                restored_tree = work/'verified-tree'
                extract_tar(restored, restored_tree, expected['size'])
                if snapshot(restored_tree) != expected:
                    raise ValueError('Restored tar files or directories differ')
            c, d = metrics
            result = {'status': 'PASS', 'verified_lossless': True, 'measured_at': utc_now(),
                      'comp_size': archive.stat().st_size, 'archive_sha256': c.get('archive_sha256') or sha256(archive),
                      'input_bytes': c['input_bytes'] if c.get('input_bytes') is not None else codec_expected['size'],
                      'input_bytes_scope': 'encoder_input_including_prepared_container_or_bato_catalog',
                      'commands': [c['command'], d['command']]}
            result['cpu_scope'] = 'isolated_workflow_including_container_and_children' if isolated else 'child_only'
            for prefix, metric in [('comp', c), ('decomp', d)]:
                for source_key, target_key in [('wall_s', 'time_s'), ('cpu_s', 'child_cpu_s'), ('peak_rss_kib', 'peak_rss_kib'),
                                               ('workflow_cpu_s', 'cpu_s'), ('worker_peak_rss_kib', 'worker_peak_rss_kib'),
                                               ('sampled_tree_peak_rss_kib', 'sampled_tree_peak_rss_kib'), ('container_time_s', 'container_time_s')]:
                    result[f'{prefix}_{target_key}'] = metric.get(source_key)
                for field in ['execution_stats', 'gpu_vram_peak_bytes', 'energy_joules', 'io_rchar', 'io_wchar', 'io_syscr', 'io_syscw', 'io_read_bytes', 'io_write_bytes', 'io_cancelled_write_bytes', 'physical_device_bytes']:
                    result[f'{prefix}_{field}'] = metric.get(field)
                result[f'{prefix}_io_scope'] = metric.get('io_scope', 'unavailable')
            if retain_archive is not None:
                stage = 'retain'
                retained = Path(retain_archive)
                retained.parent.mkdir(parents=True, exist_ok=True)
                pending = retained.with_name(retained.name + '.pending')
                try:
                    shutil.copyfile(archive, pending)
                    if sha256(pending) != result['archive_sha256']:
                        raise ValueError('Retained archive differs from verified archive')
                    pending.replace(retained)
                finally:
                    pending.unlink(missing_ok=True)
                result['retained_archive'] = str(retained)
            return result
    except (OSError, ValueError, ProcessFailure, tarfile.TarError, zipfile.BadZipFile, EOFError, MemoryError) as error:
        status = 'RESOURCE_LIMIT' if isinstance(error, (MemoryError, ResourceLimit)) else 'TIMEOUT' if isinstance(error, ProcessFailure) and error.timed_out else 'FAIL'
        result = {'status': status, 'stage': stage, 'error': str(error)}
        if isinstance(error, ProcessFailure):
            result.update(failed_command=error.command, returncode=error.returncode)
        return result


def aggregate(base, observations):
    measured = [t for t in observations if t.get('phase', 'measured') == 'measured' and t['status'] == 'PASS']
    failed = next((t for t in observations if t['status'] in FAILURES), None)
    result = {**base, 'trials': measured, 'completed_trials': len(measured)}
    if failed:
        result.update({k: failed[k] for k in ['status', 'stage', 'error', 'failed_command', 'returncode'] if k in failed})
        result['verified_lossless'] = False
        return result
    if len(measured) < base['repetitions']:
        result.update(status='RUNNING', verified_lossless=False)
        return result
    c_times, d_times = [t['comp_time_s'] for t in measured], [t['decomp_time_s'] for t in measured]
    c_time, d_time = statistics.median(c_times), statistics.median(d_times)
    sizes = [t['comp_size'] for t in measured]
    size = sizes[0] if len(set(sizes)) == 1 else statistics.median(sizes)
    result.update(status='PASS & VERIFIED', verified_lossless=True, comp_size=size, comp_size_min=min(sizes), comp_size_max=max(sizes),
                  size_deterministic=len(set(sizes)) == 1 if len(sizes)>1 else None, input_bytes=measured[0]['input_bytes'], ratio=base['raw_size'] / size,
                  comp_time_s=c_time, decomp_time_s=d_time, comp_mb_s=base['raw_size'] / 2**20 / c_time,
                  decomp_mb_s=base['raw_size'] / 2**20 / d_time, speed_unit='MiB/s',
                  comp_latency_ms=c_time * 1000, decomp_latency_ms=d_time * 1000,
                  comp_time_stdev_s=statistics.stdev(c_times) if len(measured) > 1 else None,
                  decomp_time_stdev_s=statistics.stdev(d_times) if len(measured) > 1 else None,
                  comp_time_mad_s=statistics.median(abs(t-c_time) for t in c_times) if len(measured)>1 else None,
                  decomp_time_mad_s=statistics.median(abs(t-d_time) for t in d_times) if len(measured)>1 else None,
                  timing_quality='repeated' if len(measured) > 1 else 'single_observation',
                  cpu_scope=measured[0].get('cpu_scope', 'child_only'),
                  tree_rss_scope='100ms_process_tree_samples_lower_bound' if any(t.get('comp_sampled_tree_peak_rss_kib') is not None for t in measured) else None,
                  worker_rss_scope='isolated_worker_lifetime_including_imports' if any(t.get('comp_worker_peak_rss_kib') is not None for t in measured) else None)
    for prefix in ['comp', 'decomp']:
        for field in ['cpu_s', 'child_cpu_s', 'peak_rss_kib', 'worker_peak_rss_kib', 'sampled_tree_peak_rss_kib', 'container_time_s']:
            values = [t[f'{prefix}_{field}'] for t in measured if t.get(f'{prefix}_{field}') is not None]
            result[f'{prefix}_{field}'] = (max(values) if 'rss' in field else statistics.median(values)) if values else None
        for field in ['gpu_vram_peak_bytes', 'energy_joules', 'io_rchar', 'io_wchar', 'io_syscr', 'io_syscw', 'io_read_bytes', 'io_write_bytes', 'io_cancelled_write_bytes', 'physical_device_bytes']:
            values = [t[f'{prefix}_{field}'] for t in measured if t.get(f'{prefix}_{field}') is not None]
            result[f'{prefix}_{field}'] = (max(values) if 'peak' in field else statistics.median(values)) if len(values) == len(measured) else None
        result[f'{prefix}_execution_stats'] = [t.get(f'{prefix}_execution_stats') for t in measured]
        scopes = {t.get(f'{prefix}_io_scope', 'unavailable') for t in measured}
        result[f'{prefix}_io_scope'] = scopes.pop() if len(scopes) == 1 else 'mixed'
    result['peak_rss_kib'] = max((result[f'{p}_peak_rss_kib'] for p in ['comp','decomp'] if result[f'{p}_peak_rss_kib'] is not None), default=None)
    return result


def measure(codec, workload, repetitions, warmups, timeout, scratch=None):
    """Direct adapter test API; suite runs use isolated, checkpointed phases instead."""
    base = base_record(codec, workload, repetitions, warmups)
    if not codec.program:
        return {**base, 'status': 'SKIPPED', 'error': 'Codec executable unavailable'}
    if snapshot(workload['path']) != workload['snapshot']:
        return {**base, 'status': 'FAIL', 'error': 'Input changed after selection'}
    trials = []
    for number in range(warmups + repetitions):
        result = {**measure_trial(codec, workload, timeout, scratch), 'phase': 'warmup' if number < warmups else 'measured', 'number': number if number < warmups else number-warmups}
        trials.append(result)
        if result['status'] in FAILURES:
            break
    record = aggregate(base, trials)
    if snapshot(workload['path']) != workload['snapshot']:
        record.update(status='FAIL', verified_lossless=False, error='Input changed during measurement')
    return record

"""Versioned inventories and lazy, bounded, deterministic stress fixtures."""
from __future__ import annotations

import fnmatch
import os
import random
import shutil
import wave
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

from .common import digest_json, regular_files, regular_directories, safe_member, snapshot
from .corpus import read_index
from .validation import validate_file

GENERATOR_VERSION = 'bato-benchmark-stress-v1'
METADATA_POLICY = 'normalized-v1'
NORMALIZED_TIME = 315532800  # 1980-01-01, representable in ZIP as well.
CHUNK_BYTES = 4 * 2**20


def matches(name, targets):
    return not targets or any(t.lower() in name.lower() or fnmatch.fnmatch(name.lower(), t.lower()) for t in targets)


def size_band(size):
    return 'empty' if size == 0 else 'tiny' if size < 4096 else 'small' if size < 2**20 else 'medium' if size < 16 * 2**20 else 'large' if size <= 256 * 2**20 else 'huge'


def discover(root, targets=(), categories=(), origins=('real',), include_local=False, include_duplicates=False):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('Corpus directory does not exist')
    index = read_index(root)
    candidates = regular_files(root, exclude_state=True) if include_local else [safe_member(root, n) for n in sorted(index['files'])]
    workloads, excluded, seen = [], [], {}
    for path in candidates:
        name = path.relative_to(root).as_posix()
        meta = index['files'].get(name, {})
        category, origin = meta.get('category', path.parent.name), meta.get('origin', 'local')
        if not matches(name, targets) or (categories and category not in categories):
            continue
        if origin not in origins and not (include_local and origin == 'local'):
            excluded.append({'target': name, 'reason': f'origin={origin}'})
            continue
        try:
            validate_file(path)
            current = snapshot(path)
            if meta and (current['sha256'] != meta.get('sha256') or current['size'] != meta.get('size')):
                raise ValueError('Content no longer matches the corpus index')
            if current['sha256'] in seen and not include_duplicates:
                excluded.append({'target': name, 'reason': 'duplicate', 'same_as': seen[current['sha256']]})
                continue
            seen[current['sha256']] = name
            workloads.append({'target': name, 'path': str(path), 'type': 'single_file', 'category': category,
                              'origin': origin, 'asset': meta.get('asset'), 'snapshot': current, 'scenario': 'corpus'})
        except (OSError, ValueError, EOFError, zipfile.BadZipFile, ET.ParseError, wave.Error) as error:
            excluded.append({'target': name, 'reason': 'invalid', 'error': str(error)})
    return workloads, excluded


def quick_sample(workloads, seed, max_files=50, max_bytes=256 * 2**20, max_item_bytes=16 * 2**20):
    groups = defaultdict(list)
    for item in workloads:
        size = payload_size(item)
        if size <= max_item_bytes:
            groups[(item['category'], size_band(size))].append(item)
    randomizer = random.Random(seed)
    queues = []
    for key in sorted(groups):
        choices = sorted(groups[key], key=lambda item: item['target'])
        randomizer.shuffle(choices)
        queues.append(deque(choices))
    selected, total = [], 0
    while queues and len(selected) < max_files:
        remaining = []
        for choices in queues:
            item = choices.popleft()
            size = payload_size(item)
            if total + size <= max_bytes and len(selected) < max_files:
                selected.append(item)
                total += size
            if choices:
                remaining.append(choices)
        queues = remaining
    return sorted(selected, key=lambda item: item['target'])


def directory_workloads(root, files, limit=4 * 2**30, max_files=100000, targets=()):
    """Whole validated asset/collection trees, independent of the single-file sample."""
    root = Path(root).resolve()
    by_asset, candidates = defaultdict(list), set()
    for item in files:
        if item.get('asset'):
            by_asset[item['asset']].append(Path(item['path']))
        else:
            candidates.add(Path(item['path']).parent)
    for paths in by_asset.values():
        candidates.add(Path(os.path.commonpath([str(p.parent) for p in paths])))
    for target in targets:
        path = safe_member(root, target)
        if path.is_dir():
            candidates.add(path)
    # A local nested tree has no asset metadata: include its complete root.
    if files and not any(item.get('asset') for item in files):
        candidates.add(root)
    result, excluded = [], []
    for path in sorted(candidates):
        name = path.relative_to(root).as_posix()
        if targets and not matches(name, targets) and not any(matches(item['target'], targets) for item in files if Path(item['path']).is_relative_to(path)):
            continue
        selected = [item for item in files if Path(item['path']).is_relative_to(path)]
        if not selected:
            continue
        try:
            actual = {str(p) for p in regular_files(path, exclude_state=path == root)}
            if actual != {item['path'] for item in selected}:
                excluded.append({'target': name, 'reason': 'directory contains files outside selection'})
                continue
            # Construct a sorted content inventory without reading all payloads again.
            entries = sorted([{'path': Path(item['path']).relative_to(path).as_posix(), **item['snapshot']} for item in selected], key=lambda e: e['path'])
            size = sum(e['size'] for e in entries)
            if size > limit or len(entries) > max_files:
                excluded.append({'target': name, 'reason': 'directory workload limits', 'size': size, 'file_count': len(entries)})
                continue
            origins, categories = {i['origin'] for i in selected}, {i['category'] for i in selected}
            directories = regular_directories(path)
            result.append({'target': name, 'path': str(path), 'type': 'directory_archive',
                           'category': next(iter(categories)) if len(categories) == 1 else 'mixed',
                           'origin': next(iter(origins)) if len(origins) == 1 else 'mixed',
                           'origin_counts': {o: sum(i['origin'] == o for i in selected) for o in sorted(origins)},
                           'snapshot': {'size': size, 'sha256': digest_json({'files': entries, 'directories': directories}),
                                        'files': entries, 'directories': directories}, 'scenario': 'corpus'})
        except (OSError, ValueError) as error:
            excluded.append({'target': name, 'reason': 'invalid', 'error': str(error)})
    # Default collection roots must not double-count nested descendants.
    if not targets:
        result = [item for item in result if not any(Path(item['path']).is_relative_to(other['path']) and item['path'] != other['path'] for other in result)]
    return result, excluded


def payload_size(item):
    return item['snapshot']['size'] if item.get('snapshot') else item['recipe']['file_size'] * item['recipe']['count']


def file_count(item):
    return len(item['snapshot'].get('files', [])) if item.get('snapshot') and item['type'] == 'directory_archive' else item.get('recipe', {}).get('count', 1)


def identify(item):
    recipe = {k: v for k, v in item.get('recipe', {}).items() if k != 'source_path'}
    identity = {'type': item['type'], 'origin': item['origin'], 'category': item['category'], 'target': item['target'], 'scenario': item.get('scenario', 'corpus'),
                'metadata_policy': METADATA_POLICY, 'snapshot': item.get('snapshot'), 'recipe': recipe,
                'filename': Path(item['target']).name if item['type'] == 'single_file' else None}
    return {**item, 'workload_id': digest_json(identity), 'size_band': size_band(payload_size(item)),
            'file_count': file_count(item), 'logical_payload_bytes': payload_size(item), 'metadata_policy': METADATA_POLICY}


def stress_workloads(seed=42, real_files=()):
    result = []

    def add(target, content, size, count=1, layout='flat', scenario='generated', directory=False, **extra):
        category = 'duplicates' if scenario in {'exact_duplicates', 'near_duplicates', 'mixed_duplicates', 'derived_duplicates'} else 'text' if content == 'text' else 'structured' if content == 'records' else 'binary'
        result.append({'target': 'stress/' + target, 'type': 'directory_archive' if directory else 'single_file',
                       'category': category, 'origin': 'synthetic', 'scenario': scenario,
                       'recipe': {'generator': GENERATOR_VERSION, 'seed': seed, 'content': content,
                                  'file_size': size, 'count': count, 'layout': layout, 'scenario': scenario, **extra}})

    sizes = sorted({0, 1, 64, 4096, 65536, 2**20, 16 * 2**20, 64 * 2**20, 256 * 2**20, 2**30, 4 * 2**30,
                    *[boundary + delta for boundary in [128 * 1024, 256 * 1024] for delta in [-1, 0, 1]]})
    for content in ['zeros', 'text', 'records', 'random']:
        for size in sizes:
            add(f'files/{content}/{size}.bin', content, size)
    for layout in ['flat', 'nested']:
        for content in ['text', 'random']:
            for count in [1, 10, 100, 1000, 10000, 100000]:
                add(f'directories/{layout}/{content}/{count}-files', content, 1024, count, layout, directory=True)
    for content in ['zeros', 'text', 'records', 'random']:
        for size in [256 * 2**20, 2**30, 4 * 2**30]:
            add(f'large-directories/{content}/{size}', content, 2**20, size // 2**20, directory=True)
    for content in ['text', 'random']:
        for size in [1024, 65536, 131072, 262144, 2**20]:
            add(f'duplicates/exact/{content}/100x{size}', content, size, 100, scenario='exact_duplicates', directory=True)
        for scenario in ['near_duplicates', 'mixed_duplicates']:
            add(f'duplicates/{scenario}/{content}/100x65536', content, 65536, 100, scenario=scenario, directory=True)
    representatives = {}
    for item in sorted(real_files, key=lambda row: row['target']):
        if 0 < payload_size(item) <= 2**20:
            representatives.setdefault(item['category'], item)
    for category, item in representatives.items():
        add(f'duplicates/derived/{category}/100-copies', 'source', payload_size(item), 100,
            scenario='derived_duplicates', directory=True, source_path=item['path'], source_sha256=item['snapshot']['sha256'],
            source_origin=item['origin'], source_target=item['target'])
    return result


def _write_generated(path, recipe, index):
    scenario, content = recipe['scenario'], recipe['content']
    shared = scenario in {'exact_duplicates', 'near_duplicates', 'derived_duplicates'} or (scenario == 'mixed_duplicates' and index < recipe['count'] // 2)
    seed = recipe['seed'] if shared else recipe['seed'] + index
    randomizer = random.Random(seed)
    if content == 'source':
        shutil.copyfile(recipe['source_path'], path)
        if snapshot(path) != {'size': recipe['file_size'], 'sha256': recipe['source_sha256']}:
            raise ValueError('Duplicate source changed after workload selection')
        return
    line = (f'Compression benchmark record {seed}: Tiranë 日本語.\n' if content == 'text' else f'{{"id":{seed},"active":true,"value":12345}}\n').encode()
    offset, size = 0, recipe['file_size']
    with path.open('wb') as output:
        while offset < size:
            length = min(CHUNK_BYTES, size - offset)
            block = bytes(length) if content == 'zeros' else randomizer.randbytes(length) if content == 'random' else (line * (length // len(line) + 1))[:length]
            if scenario == 'near_duplicates':
                block = bytearray(block)
                start = (index % 100 - offset) % 100
                for position in range(start, length, 100):
                    block[position] ^= (index % 255) + 1
            output.write(block)
            offset += length


def materialize(item, destination):
    """Use real copies, never hard links, so normalization cannot mutate source inodes."""
    destination = Path(destination)
    directory = item['type'] == 'directory_archive'
    if directory:
        destination.mkdir(parents=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
    if item.get('recipe'):
        recipe = item['recipe']
        for index in range(recipe['count']):
            name = f'{index:06d}.bin'
            if recipe['layout'] == 'nested':
                name = f'{index // 10000:02d}/{index // 100 % 100:02d}/' + name
            path = safe_member(destination, name) if directory else destination
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_generated(path, recipe, index)
    else:
        source = Path(item['path'])
        if directory:
            for name in item['snapshot'].get('directories', []):
                safe_member(destination, name).mkdir(parents=True, exist_ok=True)
        for entry in item['snapshot'].get('files', []) if directory else [{'path': '', **item['snapshot']}]:
            path = safe_member(destination, entry['path']) if directory else destination
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / entry['path'] if directory else source, path)
    for path in regular_files(destination):
        path.chmod(0o644)
        os.utime(path, (NORMALIZED_TIME, NORMALIZED_TIME))
    if directory:
        for parent, _, _ in os.walk(destination, topdown=False):
            Path(parent).chmod(0o755)
            os.utime(parent, (NORMALIZED_TIME, NORMALIZED_TIME))
    current = snapshot(destination)
    if item.get('snapshot') and current != item['snapshot']:
        raise ValueError('Input changed during staging')
    entries = current.get('files', [current])
    unique = {entry['sha256']: entry['size'] for entry in entries}
    return {'snapshot': current, 'unique_content_bytes': sum(unique.values()),
            'duplicate_byte_fraction': 1 - sum(unique.values()) / current['size'] if current['size'] else None}

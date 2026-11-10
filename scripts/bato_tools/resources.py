"""Linux resource admission and explicitly sampled process-tree accounting."""
from __future__ import annotations

import os
import json
import sqlite3
import resource
import shutil
from pathlib import Path


class ResourceLimit(ValueError):
    pass


def memory_state():
    values = {}
    if Path('/proc/meminfo').exists():
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in {'MemTotal', 'MemAvailable', 'SwapTotal'}:
                values[key] = int(value.split()[0]) * 1024
    limits, remaining = [], []
    # Resolve the calling process's cgroup, including delegated/container hierarchies.
    base = Path('/sys/fs/cgroup')
    roots = {base}
    if Path('/proc/self/cgroup').exists():
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            hierarchy, controllers, name = line.split(':', 2)
            if hierarchy == '0':
                leaf = base.joinpath(name.lstrip('/')).resolve()
                if leaf.is_relative_to(base):
                    roots.update(p for p in [leaf, *leaf.parents] if p.is_relative_to(base))
    for root in roots:
        cap, used = root / 'memory.max', root / 'memory.current'
        if cap.is_file() and (value := cap.read_text().strip()).isdigit():
            limit = int(value)
            limits.append(limit)
            if used.is_file():
                remaining.append(max(0, limit - int(used.read_text().strip())))
    effective = min([values.get('MemTotal', 2**30), *limits])
    available = min(values.get('MemAvailable', effective), effective)
    if remaining:
        available = min(available, *remaining)
    return {'total_ram_bytes': values.get('MemTotal'), 'available_ram_bytes': available,
            'swap_bytes': values.get('SwapTotal'), 'effective_memory_bytes': effective}


def available_cpu_count():
    """Respect affinity and container CPU quotas, rather than host CPU count."""
    count = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    base = Path('/sys/fs/cgroup')
    roots = {base}
    membership = Path('/proc/self/cgroup')
    if membership.is_file():
        for line in membership.read_text().splitlines():
            hierarchy, _, name = line.split(':', 2)
            if hierarchy == '0':
                leaf = base.joinpath(name.lstrip('/')).resolve()
                if leaf.is_relative_to(base):
                    roots.update(p for p in [leaf, *leaf.parents] if p.is_relative_to(base))
    for root in roots:
        quota = root/'cpu.max'
        if quota.is_file():
            value, period = quota.read_text().split()
            if value != 'max' and int(period) > 0:
                count = min(count, max(1, (int(value) + int(period) - 1)//int(period)))
    return max(1, count)


def resolve_memory_budget(explicit=None):
    # Address space includes mappings and runtime overhead. Keep the outer cap
    # at effective RAM; the separate managed allowance retains 10% headroom.
    return explicit if explicit is not None else max(1, memory_state()['effective_memory_bytes'])


def configure_resources(args):
    """Freeze automatic defaults once; recover them unchanged on resume."""
    prior = {}
    resume = getattr(args, 'resume', None)
    if resume:
        if Path(resume).name != resume or resume in {'.', '..'}:
            raise ValueError('--resume requires a run ID, not a path')
        from .database import database_path
        database = database_path(args)
        if database.is_file():
            # Read canonical metadata even if recoverable reports are absent.
            with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as connection:
                row = connection.execute('SELECT record FROM runs WHERE run=?', (resume,)).fetchone()
                if row:
                    prior = json.loads(row[0])
        else:
            path = Path(args.output_dir)/'runs'/resume/'run.json'
            if path.is_file():
                prior = json.loads(path.read_text())
    if getattr(args, 'cpu_limit', None) is None:
        args.cpu_limit = prior.get('cpu_limit', available_cpu_count())
    if getattr(args, 'threads', None) is None:
        args.threads = prior.get('requested_threads', min(args.cpu_limit, 256))
    state = memory_state()
    if getattr(args, 'memory_budget', None) is None:
        args.memory_budget = prior.get('effective_memory_budget') or max(1, state['effective_memory_bytes'])
    if getattr(args, 'bato_memory_budget', None) is None:
        policy = prior.get('resource_policy', {})
        args.bato_memory_budget = (policy['bato_managed_memory_budget']
            if 'bato_managed_memory_budget' in policy
            else max(1, min(state['available_ram_bytes'], args.memory_budget)*9//10))


def limit_address_space(limit):
    if limit:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        effective = min(limit, hard) if hard != resource.RLIM_INFINITY else limit
        resource.setrlimit(resource.RLIMIT_AS, (effective, hard))


def admit_disk(scratch, size, count=1):
    # Conservative simultaneous staging + tar + archive + restore budgets, including tiny-file headers.
    required = 5 * (size + count * 8192) + 64 * 2**20
    free = shutil.disk_usage(scratch).free
    if required > free:
        raise ResourceLimit(f'Scratch disk requires {required} bytes; only {free} bytes available')
    return required


class ProcessTreeMonitor:
    """100 ms samples are a lower bound, not an exact simultaneous process-tree peak."""
    def __init__(self):
        self.peak_kib = None
        # Some sandboxes expose the host /proc while subprocess APIs return
        # namespace PIDs. Sampling /proc/<child> there measures another process.
        self.available = proc_pids_match()

    def sample(self, pid):
        if not self.available:
            return
        pending, seen, total = [pid], set(), 0
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            try:
                status = Path(f'/proc/{current}/status').read_text()
                for line in status.splitlines():
                    if line.startswith('VmRSS:'):
                        total += int(line.split()[1])
                # Include children launched by threads, rather than only the main task.
                for children in Path(f'/proc/{current}/task').glob('*/children'):
                    pending.extend(int(value) for value in children.read_text().split())
            except (OSError, ValueError):
                continue
        if seen and total:
            self.peak_kib = max(self.peak_kib or 0, total)


def proc_pids_match():
    try:
        return int(Path('/proc/self/stat').read_text().split()[0]) == os.getpid()
    except (OSError, ValueError):
        return False

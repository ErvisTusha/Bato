"""Seeded selection of the copied, whole-file test corpus."""
import os
import random
from pathlib import Path

from .common import PROJECT_ROOT, sha256
from .test_corpus import CorpusSample

DEFAULT_COPIED_CORPUS = PROJECT_ROOT / 'data-research' / 'corpus'
EXTENSIONS = ('.deb', '.gz', '.zip', '.cs', '.proto', '.xz', '.wasm', '.el',
              '.exe', '.tgz', '.txt', '.json', '.woff', '.gif', '.mp3', '.rar',
              '.woff2', '.cc', '.hh', '.ml')
FILES_PER_EXTENSION = 10


def select_samples(root, mode, seed):
    """Select whole files reproducibly; their hashes are checked again when used."""
    root = Path(root).expanduser().resolve(strict=True)
    groups = {extension: [] for extension in EXTENSIONS}
    real = root / 'real'
    if not real.is_dir() or real.is_symlink():
        raise ValueError(f'Missing regular corpus directory: {real}')
    def fail(error):
        raise error
    for directory, dirs, files in os.walk(real, followlinks=False, onerror=fail):
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise ValueError(f'Corpus symlinks are not test inputs: {Path(directory) / name}')
        for name in files:
            path = Path(directory) / name
            extension = path.suffix.lower()
            if extension not in groups:
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'Corpus input is not a regular file: {path}')
            groups[extension].append((path.stat().st_size, path.relative_to(root).as_posix(), path))
    rng = random.Random(seed)
    samples = []
    for extension in EXTENSIONS:
        entries = sorted(groups[extension])
        if len(entries) != FILES_PER_EXTENSION:
            raise ValueError(f'Expected {FILES_PER_EXTENSION} {extension} files; found {len(entries)}')
        if mode == 'fast':
            chosen = [rng.choice(entries[:5]), rng.choice(entries[5:])]
        elif mode == 'normal':
            chosen = rng.sample(entries, 5)
        elif mode == 'full':
            chosen = entries
        else:
            raise ValueError(f'Unknown sample mode: {mode}')
        for size, relative, path in sorted(chosen):
            sample = CorpusSample(root, relative, size, sha256(path))
            sample.verified_path()
            samples.append(sample)
    return samples

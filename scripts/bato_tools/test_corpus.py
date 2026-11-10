"""Deterministic, verified real-file inputs for the CLI correctness suite."""
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import json
import re

from .common import sha256

DEFAULT_TEST_CORPUS = Path.home() / 'data-research' / 'corpus'
MINIMUM_EXTENSIONS = ('txt', 'pdf', 'jpg', 'png', 'mp4')
FILES_PER_EXTENSION = 2


@dataclass(frozen=True)
class CorpusSample:
    root: Path
    relative_path: str
    size: int
    digest: str

    def verified_path(self):
        """Revalidate paths and contents immediately before and after testing."""
        relative = PurePosixPath(self.relative_path)
        if relative.is_absolute() or '..' in relative.parts or '\\' in self.relative_path:
            raise ValueError(f'Unsafe corpus path: {self.relative_path}')
        path = self.root
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError(f'Corpus symlinks are not test inputs: {self.relative_path}')
        if not path.is_file() or path.stat().st_size != self.size or sha256(path) != self.digest:
            raise ValueError(f'Corpus file differs from verified inventory: {self.relative_path}')
        return path


def minimum_samples(root):
    """Select whole files by size/path, excluding duplicate content identities."""
    root = Path(root).expanduser().resolve(strict=True)
    with (root / '.corpus.json').open(encoding='utf-8') as stream:
        inventory = json.load(stream)
    if not isinstance(inventory, dict):
        raise ValueError('Corpus inventory must be an object')
    files = inventory.get('files')
    if not isinstance(files, dict):
        raise ValueError('Corpus inventory must contain a files object')
    groups = {extension: [] for extension in MINIMUM_EXTENSIONS}
    for name, entry in files.items():
        relative = PurePosixPath(name)
        extension = relative.suffix.lower().lstrip('.')
        if not relative.parts or relative.parts[0] != 'real' or extension not in groups:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f'Invalid corpus inventory entry: {name}')
        size, digest = entry.get('size'), entry.get('sha256')
        if type(size) is not int or size < 0 or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError(f'Invalid corpus inventory entry: {name}')
        groups[extension].append(CorpusSample(root, name, size, digest))
    selected, seen = [], set()
    for extension, candidates in groups.items():
        accepted = 0
        for sample in sorted(candidates, key=lambda item: (item.size, item.relative_path)):
            if sample.digest in seen:
                continue
            sample.verified_path()
            selected.append(sample)
            seen.add(sample.digest)
            accepted += 1
            if accepted == FILES_PER_EXTENSION:
                break
        if accepted < FILES_PER_EXTENSION:
            raise ValueError(f'Corpus needs {FILES_PER_EXTENSION} distinct verified .{extension} files; found {accepted}')
    return selected

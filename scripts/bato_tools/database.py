"""Database locations and verified relocation of canonical benchmark history."""
from contextlib import closing
from pathlib import Path
import os
import sqlite3
import tempfile

from .common import PROJECT_ROOT

DEFAULT_DATABASE = Path.home() / 'data-research' / 'measurements.sqlite3'
LEGACY_DATABASE = PROJECT_ROOT / 'artifacts' / 'benchmarks' / 'measurements.sqlite3'


def database_path(args):
    return Path(getattr(args, 'database', None) or Path(args.output_dir) / 'measurements.sqlite3').resolve()


def database_dump(connection):
    return '\n'.join(connection.iterdump())


def prepare_database(destination, source=None):
    """Relocate only the old default, under the caller's host measurement lock."""
    destination = Path(destination).resolve()
    source = Path(source or LEGACY_DATABASE).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source == destination or not source.is_file():
        return
    with closing(sqlite3.connect(source.as_uri() + '?mode=rw', uri=True, timeout=0)) as original:
        # Fail immediately if another SQLite writer is active, including a foreign process.
        original.execute('BEGIN IMMEDIATE')
        original.rollback()
        if destination.exists():
            with closing(sqlite3.connect(destination.as_uri() + '?mode=ro', uri=True)) as existing:
                if database_dump(original) != database_dump(existing):
                    raise ValueError(f'Database conflict: preserving both {source} and {destination}; select --database explicitly')
            return
        fd, temporary = tempfile.mkstemp(prefix='.measurements-', suffix='.sqlite3', dir=destination.parent)
        os.close(fd)
        try:
            with closing(sqlite3.connect(temporary)) as copied:
                original.backup(copied)
                if copied.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                    raise ValueError('Relocated database failed integrity check')
                if database_dump(original) != database_dump(copied):
                    raise ValueError('Relocated database differs from original records')
            # Retain a write reservation through publication and unlink.
            original.execute('BEGIN IMMEDIATE')
            with closing(sqlite3.connect(temporary)) as copied:
                if database_dump(original) != database_dump(copied):
                    raise ValueError('Original database changed during relocation')
            os.link(temporary, destination)  # Atomic publication refuses to overwrite a competing file.
            source.unlink()
            original.rollback()
        finally:
            Path(temporary).unlink(missing_ok=True)

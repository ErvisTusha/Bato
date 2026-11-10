"""Competitor-first gate for the real-file CLI correctness suite."""
from pathlib import Path

from .common import atomic_json, digest_json
from .database import DEFAULT_DATABASE, prepare_database
from .runner import host_measurement_lock
from .store import MeasurementStore
from .workloads import identify

COMPETITORS = ('zip', 'gzip', '7z', 'rar', 'paq8px', 'cmix', 'nncp')


def ensure_test_baselines(samples, options):
    """Register original bytes, freeze selection, then fill missing baseline cases."""
    import benchmark_suite
    output = Path(options.benchmark_output).expanduser().resolve()
    database = Path(options.database).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    workloads = []
    with host_measurement_lock():
        if database == DEFAULT_DATABASE.resolve():
            prepare_database(database)
        database.parent.mkdir(parents=True, exist_ok=True)
        store = MeasurementStore(database)
        try:
            for sample in samples:
                store.register_input(sample)
                workloads.append(identify({'target':sample.relative_path, 'path':str(sample.verified_path()),
                    'type':'single_file', 'origin':'real', 'category':Path(sample.relative_path).parent.name,
                    'snapshot':{'size':sample.size,'sha256':sample.digest}, 'scenario':'corpus'}))
        finally:
            store.close()
    frozen = {'schema_version':4,'workloads':workloads, 'sample_mode':options.sample_mode, 'seed':options.seed,
              'coverage':'partial: copied corpus only'}
    manifest = output / 'selections' / (digest_json(frozen) + '.json')
    atomic_json(manifest, frozen)
    print('SQLite originals verified. Ensuring competitor baselines before Bato tests.',flush=True)
    return benchmark_suite.main([
        '--corpus',str(samples[0].root), '--database',str(database), '--output-dir',str(output),
        '--scratch-dir',str(Path(options.scratch_dir).expanduser().resolve()), '--campaign-manifest',str(manifest),
        '--frozen-workloads',str(manifest), '--workload','files', '--track','native',
        '--mode','rigorous', '--warmups','1', '--repetitions','5', '--large-warmups','1',
        '--large-repetitions','5', '--profile','high', '--seed',str(options.seed),
        '--label',f'partial copied-corpus {options.sample_mode}', '--allow-unavailable',
        '--timeout',str(options.competitor_timeout), '--codecs',','.join(getattr(options, 'competitors', None) or COMPETITORS),
        '--ensure-competitors'])

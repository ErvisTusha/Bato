"""Reuse original verified competitor trials without copying observations."""
from dataclasses import dataclass
import json

from .store import validate_case_records

CONDITION_FIELDS = ('environment', 'effective_memory_budget', 'resource_policy', 'cpu_limit',
                    'requested_threads', 'compression_profile', 'mode', 'suite', 'track', 'seed', 'timeout')
CASE_FIELDS = ('workload_id', 'track', 'codec', 'input_sha256', 'raw_size', 'binary_identity',
               'effective_settings', 'resource_policy', 'repetitions', 'warmups')


@dataclass(frozen=True)
class BaselineRequest:
    metadata: dict
    case: dict


def compatible(request, metadata, case):
    """Require both input identity and exact measurement conditions."""
    if metadata.get('schema_version') != 4:
        return False
    if case.get('status') != 'PASS & VERIFIED' or not case.get('verified_lossless') or case.get('cache_origin') != 'fresh':
        return False
    if not isinstance(case.get('comp_size'), (int, float)) or case['comp_size'] < 0:
        return False
    if any(request.metadata.get(k) is None or request.metadata[k] != metadata.get(k) for k in CONDITION_FIELDS):
        return False
    return all(request.case.get(k) is not None and request.case[k] == case.get(k) for k in CASE_FIELDS)


def find_baseline(store, request):
    # Deterministic first matching original observation, never the smallest archive.
    rows = store.connection.execute('''SELECT c.run,c.id,c.record,r.record FROM cases c
        JOIN runs r ON r.run=c.run WHERE c.run<>? AND c.id=? ORDER BY c.run,c.id''',
        (request.metadata['run_id'], request.case['case_id']))
    for run, case_id, encoded_case, encoded_meta in rows:
        case, meta = json.loads(encoded_case), json.loads(encoded_meta)
        if not compatible(request, meta, case):
            continue
        try:
            validate_case_records([case], {case_id:store.trials(run,case_id)})
        except ValueError:
            continue  # An incomplete or inconsistent historical case is not a baseline.
        return {'run_id':run, 'case_id':case_id, 'measured_at':case.get('measured_at'),
                'expected_case':request.case}
    return None


def validate_references(store, metadata):
    if metadata is None:
        return  # Legacy validation callers can supply cases without run metadata.
    for case_id, reference in metadata.get('baseline_references', {}).items():
        original_meta = store.run(reference['run_id'])
        case = next((c for c in store.cases(reference['run_id']) if c['case_id']==reference['case_id']), None)
        request = BaselineRequest(metadata, reference['expected_case'])
        if case is None or original_meta is None or case_id != case['case_id'] or not compatible(request,original_meta,case):
            raise ValueError('Referenced competitor baseline changed or is missing: ' + case_id)
        validate_case_records([case], {case_id:store.trials(reference['run_id'],case_id)})

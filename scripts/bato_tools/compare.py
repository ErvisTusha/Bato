"""Read-only matching of stored runs; binary changes are the subject of comparison."""
import json
import sqlite3
import uuid
from collections import defaultdict
from pathlib import Path

from .common import atomic_json, digest_json
from .reporting import atomic_csv
from .store import expected_case_ids, validate_case_records


BATO_SETTING_FLAGS = {
    'segment_size': '--segment-size', 'chunk_size': '--chunk-size',
}


def validate_bato_setting_changes(allowed=()):
    """Allow declared model changes, never hardware/resource/input changes."""
    if isinstance(allowed, str):
        allowed = allowed.split(',')
    fields = set(allowed)
    if fields - BATO_SETTING_FLAGS.keys():
        raise ValueError('Allowed Bato setting changes: ' + ', '.join(BATO_SETTING_FLAGS))
    return sorted(fields)


def normalized_bato_settings(settings, allowed):
    """Remove only named fields and their exact generated CLI flag/value pairs."""
    flags = {BATO_SETTING_FLAGS[field] for field in allowed}
    normalized = {key: value for key, value in settings.items() if key not in allowed}
    for key in ('file_options', 'directory_options'):
        if key not in normalized:
            continue
        options, kept, index = normalized[key], [], 0
        while index < len(options):
            option = options[index]
            if option in flags and index + 1 < len(options):
                index += 2
            else:
                kept.append(option)
                index += 1
        normalized[key] = kept
    return normalized


def metric_outcome(before, after, metric):
    """Conservative observed-range rule, not a statistical confidence interval.

    Exact bytes have no timing tolerance. Timed wins/losses require five fresh
    trials, a warmup, disjoint observed ranges, and >3% median improvement.
    Sampled tree RSS never establishes a true peak-memory or efficiency win.
    """
    a, b = before.get(metric), after.get(metric)
    if a is None or b is None:
        return 'unknown'
    if metric == 'comp_size':
        if before.get('size_deterministic') is False or after.get('size_deterministic') is False:
            return 'uncertain'
        return 'win' if b < a else 'loss' if b > a else 'tie'
    if 'sampled_tree' in metric or 'gpu_vram' in metric or 'energy' in metric or 'physical_device' in metric:
        return 'unknown'
    if any(row.get('cache_origin') != 'fresh' or len(row.get('trials', [])) < 5 or row.get('warmups', 0) < 1 for row in [before, after]):
        return 'uncertain'
    av = [t.get(metric) for t in before['trials']]
    bv = [t.get(metric) for t in after['trials']]
    if any(v is None for v in av + bv):
        return 'unknown'
    floor = 0.001 if metric.endswith('time_s') else 0.0001 if metric.endswith('cpu_s') else 0
    if abs(a - b) <= max(abs(a) * .03, floor):
        return 'tie'
    return 'win' if max(bv) < min(av) else 'loss' if min(bv) > max(av) else 'uncertain'


def compare_competitors(metadata, records, bato='bato-v1'):
    """Keep one named Bato configuration and compare every workload/track."""
    selected = [r for r in records if r['codec'] == bato]
    if not selected:
        raise ValueError(f'Bato configuration absent from run: {bato}')
    rows = []
    metrics = ['comp_size', 'comp_time_s', 'decomp_time_s', 'comp_cpu_s', 'decomp_cpu_s',
               'comp_sampled_tree_peak_rss_kib', 'decomp_sampled_tree_peak_rss_kib',
               'comp_gpu_vram_peak_bytes', 'decomp_gpu_vram_peak_bytes', 'comp_energy_joules', 'decomp_energy_joules',
               'comp_io_read_bytes', 'decomp_io_read_bytes', 'comp_io_write_bytes', 'decomp_io_write_bytes',
               'comp_physical_device_bytes', 'decomp_physical_device_bytes']
    for z in selected:
        for peer in records:
            if peer['codec'].startswith('bato') or (peer['workload_id'], peer['track']) != (z['workload_id'], z['track']):
                continue
            valid = (z.get('verified_lossless') and peer.get('verified_lossless') and
                     all(z.get(k) == peer.get(k) for k in ['input_sha256', 'codec_input_sha256', 'metadata_policy', 'raw_size', 'compression_profile']))
            outcomes = {metric: metric_outcome(peer, z, metric) if valid else 'unknown' for metric in metrics}
            rows.append({'target': z['target'], 'workload_id': z['workload_id'], 'track': z['track'], 'bato': bato, 'competitor': peer['codec'],
                         'profile': z['compression_profile'], 'verified_comparable': bool(valid),
                         'outcomes': outcomes, 'bato_values': {k: z.get(k) for k in metrics},
                         'competitor_values': {k: peer.get(k) for k in metrics},
                         'bato_trials': len(z.get('trials', [])), 'competitor_trials': len(peer.get('trials', [])),
                         'all_metric_win': all(v == 'win' for v in outcomes.values())})
    unavailable = [t for t in metadata.get('tool_preflight', {}).get('tools', []) if t['status'] != 'READY']
    return {'run_id': metadata['run_id'], 'bato_configuration': bato, 'comparisons': rows,
            'unavailable': unavailable, 'overall_goal_complete': False,
            'uncertainty_rule': 'Five measured trials plus warmup; >3% median difference and disjoint observed ranges. Ranges are not confidence intervals.',
            'resource_limitations': 'Process-tree RAM is a sampled lower bound; VRAM, energy, and physical-device bytes are unknown without reliable instruments. Linux workflow I/O counts storage-layer reads and page dirtying, with cancellations separate. A GPU cannot use strictly less VRAM than a CPU-only competitor using zero VRAM.'}


def load_run(root, run_id, database=None):
    if Path(run_id).name != run_id or run_id in {'.','..'}:
        raise ValueError('Comparison requires run IDs, not paths')
    database = Path(database) if database is not None else Path(root)/'measurements.sqlite3'
    if database.is_file():
        connection = sqlite3.connect(f'file:{database.resolve()}?mode=ro',uri=True)
        try:
            integrity = connection.execute('PRAGMA integrity_check').fetchall()
            if integrity != [('ok',)]:
                raise ValueError(f'Benchmark database integrity check failed: {integrity[:3]}')
            meta = connection.execute('SELECT record FROM runs WHERE run=?',(run_id,)).fetchone()
            if meta:
                metadata = json.loads(meta[0])
                records = [json.loads(row[0]) for row in connection.execute('SELECT record FROM cases WHERE run=? ORDER BY id',(run_id,))]
                expected = expected_case_ids(metadata)
                trials = {row['case_id']: [json.loads(t[0]) for t in connection.execute(
                    'SELECT record FROM trials WHERE run=? AND id=? ORDER BY phase DESC,number', (run_id, row['case_id']))]
                          for row in records}
                validate_case_records(records, trials, expected, complete=metadata.get('status') == 'complete')
                return metadata,records
        finally:
            connection.close()
    directory = Path(root)/'runs'/run_id
    metadata = json.loads((directory/'run.json').read_text())
    if metadata.get('schema_version') == 4:
        raise ValueError('Schema v4 comparison requires the canonical measurements.sqlite3, not report projections')
    return metadata,json.loads((directory/'results.json').read_text())


def compare_runs(baseline_meta, baseline, candidate_meta, candidate, allow_bato_setting_change=()):
    allowed = validate_bato_setting_changes(allow_bato_setting_change)
    if baseline_meta.get('schema_version') != 4 or candidate_meta.get('schema_version') != 4:
        raise ValueError('Comparable runs require schema v4. Legacy reports remain readable but need fresh measurements with normalized inputs/tracks.')
    key = lambda row:(row['workload_id'],row['track'],row['codec'])
    previous = {key(row):row for row in baseline}
    current = {key(row):row for row in candidate}
    rows, exclusions = [], []
    timing_compatible = (bool(baseline_meta.get('environment')) and bool(baseline_meta.get('effective_memory_budget')) and
                         bool(baseline_meta.get('resource_policy')) and
                         baseline_meta.get('environment') == candidate_meta.get('environment') and
                         baseline_meta.get('effective_memory_budget') == candidate_meta.get('effective_memory_budget') and
                         baseline_meta.get('resource_policy') == candidate_meta.get('resource_policy'))
    for item in sorted(set(previous)|set(current)):
        a,b = previous.get(item),current.get(item)
        name = (b or a)['target']
        reason = None
        if not a or not b:
            reason = 'Workload/codec absent from one run'
        elif not a.get('verified_lossless') or not b.get('verified_lossless'):
            reason = f"Unverified or incomplete result: {a['status']} / {b['status']}"
        elif not all(a.get(field) and b.get(field) for field in ['input_sha256','codec_input_sha256','metadata_policy','effective_settings']):
            reason = 'Required input or settings identity is missing'
        elif any(a.get(field) != b.get(field) for field in ['input_sha256','codec_input_sha256','metadata_policy','origin','category','scenario','type','size_band','raw_size']):
            reason = 'Input bytes, metadata policy, or workload grouping changed'
        elif a['effective_settings'] != b['effective_settings']:
            if not item[2].startswith('bato') or normalized_bato_settings(a['effective_settings'], allowed) != normalized_bato_settings(b['effective_settings'], allowed):
                reason = 'Codec settings changed outside the declared Bato model settings'
        if reason:
            exclusions.append({'target':name,'workload_id':item[0],'track':item[1],'codec':item[2],'reason':reason})
            continue
        delta = b['comp_size']-a['comp_size']
        row = {'target':name, 'workload_id':item[0], 'track':item[1], 'codec':item[2],
               'origin':b['origin'],'category':b['category'],'scenario':b['scenario'],'type':b['type'],'size_band':b['size_band'],
               'baseline_size_bytes':a['comp_size'],'candidate_size_bytes':b['comp_size'],'size_delta_bytes':delta,
               'size_improvement_percent':-delta/a['comp_size']*100 if a['comp_size'] else None,
               'size_outcome':'smaller' if delta<0 else 'larger' if delta>0 else 'tie',
               'size_variable':a.get('size_deterministic') is False or b.get('size_deterministic') is False,
               'baseline_timing_quality':a.get('timing_quality'),'candidate_timing_quality':b.get('timing_quality'),
               'timing_comparable':timing_compatible and a.get('cache_origin')=='fresh' and b.get('cache_origin')=='fresh' and a.get('timing_quality')=='repeated' and b.get('timing_quality')=='repeated'}
        if item[2].startswith('bato'):
            row['bato_setting_changes'] = {field: {'baseline': a['effective_settings'].get(field),
                                                     'candidate': b['effective_settings'].get(field)}
                                             for field in allowed if a['effective_settings'].get(field) != b['effective_settings'].get(field)}
        for prefix in ['comp','decomp']:
            row[f'baseline_{prefix}_time_s'] = a.get(f'{prefix}_time_s')
            row[f'candidate_{prefix}_time_s'] = b.get(f'{prefix}_time_s')
            row[f'{prefix}_time_change_percent'] = (b[f'{prefix}_time_s']/a[f'{prefix}_time_s']-1)*100 if row['timing_comparable'] and a.get(f'{prefix}_time_s') else None
            for metric in ['cpu_s','peak_rss_kib','worker_peak_rss_kib','sampled_tree_peak_rss_kib', 'io_read_bytes', 'io_write_bytes', 'io_cancelled_write_bytes']:
                before, after = a.get(f'{prefix}_{metric}'), b.get(f'{prefix}_{metric}')
                row[f'baseline_{prefix}_{metric}'] = before
                row[f'candidate_{prefix}_{metric}'] = after
                row[f'{prefix}_{metric}_delta'] = after-before if row['timing_comparable'] and before is not None and after is not None else None
        for side, record in [('baseline',a),('candidate',b)]:
            row[f'{side}_size_min_bytes'] = record.get('comp_size_min', record['comp_size'])
            row[f'{side}_size_max_bytes'] = record.get('comp_size_max', record['comp_size'])
            row[f'{side}_binary_identity'] = record.get('binary_identity')
        if row['size_variable'] and not (row['candidate_size_max_bytes'] < row['baseline_size_min_bytes'] or row['candidate_size_min_bytes'] > row['baseline_size_max_bytes']):
            row['size_outcome'] = 'variable_overlap'
        rows.append(row)
        row['metric_outcomes'] = {metric: metric_outcome(a, b, metric) if metric == 'comp_size' or row['timing_comparable'] else 'unknown'
                                  for metric in ['comp_size', 'comp_time_s', 'decomp_time_s', 'comp_cpu_s', 'decomp_cpu_s',
                                                 'comp_peak_rss_kib', 'decomp_peak_rss_kib', 'comp_sampled_tree_peak_rss_kib', 'decomp_sampled_tree_peak_rss_kib',
                                                 'comp_io_read_bytes', 'decomp_io_read_bytes', 'comp_io_write_bytes', 'decomp_io_write_bytes', 'comp_physical_device_bytes', 'decomp_physical_device_bytes']}
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[dimension] for dimension in ['origin','type','track','category','scenario','size_band','codec'])].append(row)
    summaries = []
    for group, cases in sorted(groups.items()):
        before = sum(row['baseline_size_bytes'] for row in cases)
        after = sum(row['candidate_size_bytes'] for row in cases)
        summaries.append({**dict(zip(['origin','type','track','category','scenario','size_band','codec'],group)),
                          'matched_cases':len(cases),'baseline_bytes':before,'candidate_bytes':after,
                          'size_improvement_percent':(before-after)/before*100 if before else None,
                          **{outcome:sum(row['size_outcome']==outcome for row in cases) for outcome in ['smaller','larger','tie','variable_overlap']}})
    return {'schema_version':4,'baseline_run':baseline_meta['run_id'],'candidate_run':candidate_meta['run_id'],
            'baseline_label':baseline_meta.get('label'),'candidate_label':candidate_meta.get('label'),
            'matched_cases':len(rows),'intended_cases':len(set(previous)|set(current)),
            'complete_coverage':not exclusions and baseline_meta.get('status')=='complete' and candidate_meta.get('status')=='complete',
            'timing_conditions_match':timing_compatible,'allowed_bato_setting_changes':allowed,
            'comparisons':rows,'summary':summaries,'exclusions':exclusions}


def write_comparison(root, baseline, candidate, competitors=None, allow_bato_setting_change=(), database=None):
    a,arows = load_run(root,baseline,database)
    b,brows = load_run(root,candidate,database)
    result = compare_runs(a,arows,b,brows,allow_bato_setting_change)
    if competitors:
        result['competitors'] = compare_competitors(b,brows,competitors)
    directory = Path(root)/'comparisons'/f'{baseline}__{candidate}-{uuid.uuid4().hex[:8]}'
    directory.mkdir(parents=True)
    atomic_json(directory/'comparison.json',result)
    atomic_csv(directory/'comparison.csv',result['comparisons'])
    atomic_csv(directory/'summary.csv',result['summary'])
    atomic_csv(directory/'exclusions.csv',result['exclusions'])
    lines = ['# Benchmark comparison','',f'Baseline: `{baseline}`; candidate: `{candidate}`.',
             f"Build labels: {a.get('label') or 'unspecified'} / {b.get('label') or 'unspecified'}.",
             'Declared Bato setting changes: ' + (', '.join(result['allowed_bato_setting_changes']) or 'none') + '.',
             '',f"Matched cases: {result['matched_cases']} / {result['intended_cases']}; complete coverage: {result['complete_coverage']}.",
             'Positive size improvement means fewer compressed bytes. Timing and resource deltas require matching conditions and repeated fresh trials.',
             'Variable output sizes have recorded ranges; overlapping ranges do not establish an improvement.',
             '', '| Target | Track | Codec | Baseline bytes | Candidate bytes | Size | Compression | Decompression | Compression CPU | Decompression CPU |', '| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |']
    for row in result['comparisons']:
        name = row['target'].replace('|','\\|').replace('\n',' ')
        outcome = 'overlapping ranges' if row['size_outcome']=='variable_overlap' else row['size_outcome']
        metrics = row['metric_outcomes']
        lines.append(f"| {name} | {row['track']} | {row['codec']} | {row['baseline_size_bytes']:,} | {row['candidate_size_bytes']:,} | {outcome} | {metrics['comp_time_s']} | {metrics['decomp_time_s']} | {metrics['comp_cpu_s']} | {metrics['decomp_cpu_s']} |")
    if competitors:
        peer = result['competitors']
        atomic_json(directory/'competitors.json',peer)
        lines.extend(['', f'## Competitors versus {competitors}', '', peer['uncertainty_rule'], '', peer['resource_limitations'], '',
                      '| Target | Track | Competitor | Size | Compression | Decompression | Compression CPU | Decompression CPU | RAM / VRAM / energy |',
                      '| --- | --- | --- | --- | --- | --- | --- | --- | --- |'])
        for row in peer['comparisons']:
            m = row['outcomes']
            name = row['target'].replace('|','\\|').replace('\n',' ')
            lines.append(f"| {name} | {row['track']} | {row['competitor']} | {m['comp_size']} | {m['comp_time_s']} | {m['decomp_time_s']} | {m['comp_cpu_s']} | {m['decomp_cpu_s']} | unknown |")
        lines.extend(['', 'Unavailable competitors or configurations (absence never counts as a win):'])
        lines.extend(f"- {t['name']}: {t['status']}; {t.get('error','unspecified')}" for t in peer['unavailable'])
    lines.extend(['','Excluded cases: '+str(len(result['exclusions'])), 'See exclusions.csv for missing, failed, or incompatible cases.'])
    (directory/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return directory,result

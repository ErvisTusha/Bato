"""Comparable size rankings and recoverable JSON/CSV projections."""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from .common import atomic_json
from .measurement import FAILURES

DIMENSIONS = ('type', 'track', 'origin', 'category', 'scenario', 'size_band')


def comparable_key(row):
    return tuple(row.get(k, 'unspecified') for k in DIMENSIONS)


def summarize(records):
    groups, shared_by_group = defaultdict(list), defaultdict(dict)
    for row in records:
        key = comparable_key(row)
        groups[(*key, row['codec'])].append(row)
        if row['status'] != 'SKIPPED':
            successes = shared_by_group[key].setdefault(row['codec'], set())
            if row.get('verified_lossless'):
                successes.add(row.get('workload_id', row['target']))
    shared = {key: set.intersection(*codecs.values()) for key, codecs in shared_by_group.items()}
    result = []
    for key, group in sorted(groups.items()):
        passing = [r for r in group if r.get('verified_lossless') and r.get('workload_id', r['target']) in shared.get(key[:-1], set())]
        fresh = [r for r in passing if r.get('cache_origin', 'fresh') == 'fresh']
        raw, compressed = sum(r['raw_size'] for r in passing), sum(r['comp_size'] for r in passing)
        c_time, d_time = sum(r['comp_time_s'] for r in fresh), sum(r['decomp_time_s'] for r in fresh)
        timed_raw = sum(r['raw_size'] for r in fresh)
        row = dict(zip((*DIMENSIONS, 'codec'), key))
        row.update(attempted=len(group), passed=sum(bool(r.get('verified_lossless')) for r in group),
                   failed=sum(r['status'] in FAILURES for r in group), skipped=sum(r['status'] == 'SKIPPED' for r in group),
                   pending=sum(r['status'] in {'PENDING','RUNNING'} for r in group), common_workloads=len(passing),
                   coverage_fraction=len(passing) / len(group), raw_size=raw, compressed_bytes=compressed,
                   weighted_ratio=raw / compressed if compressed else None,
                   comp_mib_s=timed_raw / 2**20 / c_time if c_time else None,
                   decomp_mib_s=timed_raw / 2**20 / d_time if d_time else None,
                   cached_records=sum(r.get('cache_origin') == 'baseline' for r in group))
        for prefix in ['comp', 'decomp']:
            for metric in ['cpu_s','peak_rss_kib','worker_peak_rss_kib','sampled_tree_peak_rss_kib']:
                values = [r[f'{prefix}_{metric}'] for r in fresh if r.get(f'{prefix}_{metric}') is not None]
                row[f'{prefix}_{metric}'] = (max(values) if 'rss' in metric else sum(values)) if values else None
        result.append(row)
    return result


def leaderboards(records, metadata):
    groups = defaultdict(list)
    for row in records:
        groups[comparable_key(row)].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        by_codec, by_case = defaultdict(list), defaultdict(list)
        for row in rows:
            by_codec[row['codec']].append(row)
            by_case[row.get('workload_id', row['target'])].append(row)
        codec_count = len(by_codec)
        common = {case for case, cases in by_case.items() if len(cases) == codec_count and all(r.get('verified_lossless') for r in cases)}
        rankings = []
        for codec, cases in sorted(by_codec.items()):
            passing = [r for r in cases if r.get('workload_id',r['target']) in common]
            wins = ties = 0
            for row in passing:
                peers = by_case[row.get('workload_id',row['target'])]
                best = min(p['comp_size'] for p in peers)
                if row['comp_size'] == best:
                    if sum(p['comp_size'] == best for p in peers) > 1:
                        ties += 1
                    else:
                        wins += 1
            rankings.append({'codec':codec, 'compressed_bytes':sum(r['comp_size'] for r in passing),
                             'common_workloads':len(passing), 'intended_workloads':len(by_case),
                             'passed':sum(bool(r.get('verified_lossless')) for r in cases),
                             'failures':sum(r['status'] in FAILURES for r in cases), 'wins':wins,'ties':ties,
                             'losses':len(passing)-wins-ties,
                             'variable_size_cases':sum(r.get('size_deterministic') is False for r in passing)})
        rankings.sort(key=lambda r:(r['compressed_bytes'],r['codec']))
        complete = bool(common) and len(common) == len(by_case) and all(not r['variable_size_cases'] for r in rankings)
        # Quick sampling and cached observations are exploratory, not full-corpus winner evidence.
        eligible = complete and metadata.get('status') == 'complete' and metadata.get('mode') == 'rigorous' and not any(r.get('cache_origin') == 'baseline' for r in rows)
        best = rankings[0]['compressed_bytes'] if rankings else None
        output.append({**dict(zip(DIMENSIONS, key)),
                       'complete_coverage':complete, 'size_ranking_eligible':eligible,
                       'all_workload_size_winners':[r['codec'] for r in rankings if r['wins'] == len(by_case)] if eligible else [],
                       'winner_claim_eligible':False, 'winners':[],
                       'ranking':rankings})
    return output


def atomic_lines(path, records):
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        for row in records:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def atomic_csv(path, records, fields=None):
    fields = fields or sorted({k for r in records for k in r if k != 'trials'})
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in record.items() if k in fields})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def export(run_dir, records, metadata, observations=()):
    run_dir = Path(run_dir)
    summaries, boards = summarize(records), leaderboards(records,metadata)
    atomic_json(run_dir/'results.json',records)
    atomic_lines(run_dir/'results.jsonl',records)
    atomic_json(run_dir/'summary.json',summaries)
    atomic_json(run_dir/'leaderboards.json',boards)
    atomic_json(run_dir/'run.json',metadata)
    atomic_csv(run_dir/'results.csv',records)
    atomic_csv(run_dir/'summary.csv',summaries)
    fields = ['target','type','track','origin','category','scenario','size_band','file_count','compressor','codec',
              'compression_profile','effective_settings','status','verified_lossless','raw_size','input_bytes',
              'comp_size','comp_size_min','comp_size_max','ratio','comp_mb_s','decomp_mb_s',
              'comp_cpu_s','decomp_cpu_s','comp_peak_rss_kib','decomp_peak_rss_kib',
              'comp_worker_peak_rss_kib','decomp_worker_peak_rss_kib','comp_sampled_tree_peak_rss_kib','decomp_sampled_tree_peak_rss_kib',
              'peak_rss_kib','repetitions','completed_trials','timing_quality','cache_origin','stage','error']
    atomic_csv(run_dir/'archive_comparison.csv',sorted([r for r in records if r['type']=='directory_archive'],key=lambda r:(r['target'],r.get('track','native'),r.get('comp_size',float('inf')))),fields)
    atomic_lines(run_dir/'trials.jsonl',observations)
    atomic_csv(run_dir/'trials.csv',observations)
    text = ['# Compression benchmark', '', f"Run: `{metadata.get('run_id','unspecified')}`; status: **{metadata.get('status','unspecified')}**.",
            '', 'Size rankings use matching verified workloads. CPU includes container work; sampled tree RAM is a lower bound.',
            'Quick/cached or incomplete comparisons cannot establish an overall winner.', '']
    for board in boards:
        text.extend([f"## {' / '.join(board[dimension] for dimension in DIMENSIONS)}",'',
                     '| Codec | Compressed bytes | Common / intended | Wins / ties / losses | Failures |',
                     '| --- | ---: | ---: | ---: | ---: |'])
        for row in board['ranking']:
            text.append(f"| {row['codec']} | {row['compressed_bytes']:,} | {row['common_workloads']} / {row['intended_workloads']} | {row['wins']} / {row['ties']} / {row['losses']} | {row['failures']} |")
        text.extend(['', 'Winners: ' + ', '.join(board['winners']) if board['winners'] else 'Overall winner: not established.', ''])
    temporary = run_dir/'.report.md.tmp'
    temporary.write_text('\n'.join(text),encoding='utf-8')
    os.replace(temporary,run_dir/'report.md')


def print_archive_summary(records):
    rows = [row for row in summarize(records) if row['type']=='directory_archive']
    if not rows:
        return
    print('Archive comparison: matching verified cases; tracks/origins/scenarios remain separate')
    print(f"{'Track':<11} {'Origin':<10} {'Category':<12} {'Codec':<10} {'Cases':>7} {'Ratio':>8} {'Failures':>9} {'Cached':>7}")
    for row in rows:
        ratio = f"{row['weighted_ratio']:.3f}x" if row['weighted_ratio'] is not None else 'n/a'
        print(f"{row['track']:<11} {row['origin']:<10} {row['category']:<12} {row['codec']:<10} {row['common_workloads']:>3}/{row['attempted']:<3} {ratio:>8} {row['failed']:>9} {row['cached_records']:>7}")

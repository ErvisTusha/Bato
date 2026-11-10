#!/usr/bin/env python3
"""Compare stored schema-v4 measurements without running compressors."""
import argparse
import sqlite3
import sys
from bato_tools.database import DEFAULT_DATABASE
from bato_tools.archive_ops import compare_operations
from bato_tools.common import PROJECT_ROOT
from bato_tools.common import atomic_json
from bato_tools.compare import compare_competitors, load_run, write_comparison, validate_bato_setting_changes


def reference_competitors(candidate_meta, candidate_records, reference_meta, reference_records, bato):
    """Read original peer trials; never copy observations into another run."""
    fields = ['environment', 'effective_memory_budget', 'resource_policy',
              'cpu_limit', 'requested_threads', 'compression_profile',
              'mode', 'suite', 'track', 'seed', 'timeout']
    if candidate_meta.get('schema_version') != 4 or reference_meta.get('schema_version') != 4:
        raise ValueError('Reference competitors require canonical schema-v4 runs')
    changed = [field for field in fields if candidate_meta.get(field) is None or
               candidate_meta.get(field) != reference_meta.get(field)]
    if changed:
        raise ValueError('Reference competitor conditions differ or are missing: ' + ', '.join(changed))
    def accepted(record):
        return bool(record.get('verified_lossless') and record.get('cache_origin') == 'fresh'
                    and record.get('status') == 'PASS & VERIFIED'
                    and record.get('warmups', 0) >= 1 and len(record.get('trials', [])) >= 5)
    selected = [dict(r, verified_lossless=accepted(r)) for r in candidate_records if r['codec'] == bato]
    peers = [dict(r, verified_lossless=accepted(r)) for r in reference_records if not r['codec'].startswith('bato')]
    policies = {(r['workload_id'], r['track']): (r['warmups'], len(r['trials']))
                for r in selected if r['verified_lossless']}
    for peer in peers:
        key = (peer['workload_id'], peer['track'])
        if peer['verified_lossless'] and key in policies and policies[key] != (peer['warmups'], len(peer['trials'])):
            raise ValueError('Reference competitor trial policy differs: ' + peer['target'])
    codecs = [c['id'] for c in reference_meta.get('codecs', []) if not c['id'].startswith('bato')]
    if not codecs:
        raise ValueError('Reference run has no competitor inventory')
    result = compare_competitors(reference_meta, selected + peers, bato)
    available = {(r['workload_id'], r['track'], r['codec']) for r in peers}
    result.update(run_id=candidate_meta['run_id'], competitor_run_id=reference_meta['run_id'],
                  competitor_run_status=reference_meta.get('status'), conditions_match=True,
                  matched_condition_fields=fields,
                  missing_reference_cases=[{'target':r['target'], 'track':r['track'], 'competitor':codec}
                                           for r in selected for codec in codecs
                                           if (r['workload_id'], r['track'], codec) not in available],
                  trial_provenance='Read from original canonical runs; no copied, cached, or invented observations',
                  timing_limitations='Sequential runs and uncontrolled filesystem cache; observed ranges are not confidence intervals')
    by_key = {(r['workload_id'], r['track'], r['codec']):r for r in peers}
    for row in result['comparisons']:
        peer = by_key[(row['workload_id'], row['track'], row['competitor'])]
        row.update(bato_run_id=candidate_meta['run_id'], competitor_run_id=reference_meta['run_id'],
                   competitor_status=peer.get('status'), competitor_error=peer.get('error'))
    return result


def reference_campaigns(candidate_meta, candidate_records, references, bato):
    """Join disjoint original campaigns without selecting the best repeated case."""
    if not references:
        raise ValueError('At least one reference competitor run is required')
    run_ids, observed, inventory, combined = [], set(), {}, None
    for metadata, records in references:
        run_id = metadata['run_id']
        if run_id in run_ids:
            raise ValueError('Duplicate reference run: ' + run_id)
        run_ids.append(run_id)
        for codec in metadata.get('codecs', []):
            if codec['id'].startswith('bato'):
                continue
            previous = inventory.setdefault(codec['id'], codec)
            if previous != codec:
                raise ValueError('Reference competitor identity changed: ' + codec['id'])
        for row in records:
            if row['codec'].startswith('bato'):
                continue
            key = (row['workload_id'], row['track'], row['codec'])
            if key in observed:
                raise ValueError('Overlapping reference competitor case: ' + str(key))
            observed.add(key)
        part = reference_competitors(candidate_meta, candidate_records, metadata, records, bato)
        if combined is None:
            combined = part
        else:
            combined['comparisons'].extend(part['comparisons'])
            combined['unavailable'].extend(part['unavailable'])
    combined.pop('competitor_run_id', None)
    combined.pop('competitor_run_status', None)
    combined['competitor_run_ids'] = run_ids
    combined['competitor_runs'] = [{'run_id':metadata['run_id'], 'status':metadata.get('status')}
                                   for metadata, _ in references]
    combined['missing_reference_cases'] = [
        {'target':row['target'], 'track':row['track'], 'competitor':codec}
        for row in candidate_records if row['codec'] == bato for codec in sorted(inventory)
        if (row['workload_id'], row['track'], codec) not in observed]
    return combined


def write_reference_competitors(root, directory, result, candidate, references, bato, database=None):
    candidate_meta, candidate_records = load_run(root, candidate, database)
    if isinstance(references, str):
        references = [references]
    peers = reference_campaigns(candidate_meta, candidate_records,
                                [load_run(root, reference, database) for reference in references], bato)
    result['competitors'] = peers
    atomic_json(directory/'comparison.json', result)
    atomic_json(directory/'competitors.json', peers)
    lines = ['', f'## Reference competitors versus {bato}', '',
             'Original competitor runs: ' + ', '.join(f'`{reference}`' for reference in references) + '.',
             peers['uncertainty_rule'], peers['resource_limitations'], peers['timing_limitations'], '',
             '| Target | Track | Competitor | Size | Compression | Decompression | Compression CPU | Decompression CPU | RAM / VRAM / energy |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in peers['comparisons']:
        name = row['target'].replace('|', '\\|').replace('\n', ' ')
        m = row['outcomes']
        lines.append(f'| {name} | {row["track"]} | {row["competitor"]} | {m["comp_size"]} | {m["comp_time_s"]} | {m["decomp_time_s"]} | {m["comp_cpu_s"]} | {m["decomp_cpu_s"]} | unknown |')
    lines.extend(['', f'Missing reference cases: {len(peers["missing_reference_cases"])}. Absence never counts as a win.',
                  'Unavailable competitors or configurations:'])
    lines.extend(f'- {t["name"]}: {t["status"]}; {t.get("error", "unspecified")}' for t in peers['unavailable'])
    with (directory/'comparison.md').open('a', encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',required=True)
    parser.add_argument('--candidate',required=True)
    parser.add_argument('--competitors', help='Compare candidate peers against this one Bato codec ID, e.g. bato-v1 or bato-v1-cpu8')
    parser.add_argument('--competitor-run', action='append', default=[],
                        help='Repeat for disjoint canonical competitor runs; requires --competitors and identical conditions')
    parser.add_argument('--allow-bato-setting-change', action='append', default=[],
                        help='Explicit model fields allowed to change, comma-separated: segment_size,chunk_size')
    parser.add_argument('--output-dir',default=str(PROJECT_ROOT/'artifacts/benchmarks'),help='Benchmark history directory')
    parser.add_argument("--archive-operations", action="store_true")
    parser.add_argument('--database', default=str(DEFAULT_DATABASE), help='Canonical SQLite measurement database')
    args = parser.parse_args(argv)
    if args.competitor_run and not args.competitors:
        parser.error('--competitor-run requires one declared Bato configuration with --competitors')
    try:
        if args.archive_operations:
            return compare_operations(args.output_dir, args.baseline, args.candidate, args.database)
        allowed = validate_bato_setting_changes(
            field.strip() for value in args.allow_bato_setting_change for field in value.split(','))
        directory,result = write_comparison(args.output_dir,args.baseline,args.candidate,
                                            None if args.competitor_run else args.competitors,
                                            allow_bato_setting_change=allowed, database=args.database)
        if args.competitor_run:
            write_reference_competitors(args.output_dir,directory,result,args.candidate,args.competitor_run,args.competitors,args.database)
        print(f"Compared {result['matched_cases']} / {result['intended_cases']} cases. Report: {directory/'comparison.md'}")
        return 0
    except (OSError,ValueError,KeyError,sqlite3.Error) as error:
        print(f'Error: {error}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

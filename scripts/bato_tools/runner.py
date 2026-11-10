"""Sequential, isolated, trial-checkpointed benchmarks with versioned history."""
from __future__ import annotations

import contextlib
import fcntl
import json
import shutil
import sqlite3
import os
import platform
import random
import resource
import subprocess
import tempfile
import uuid
from pathlib import Path

from .codecs import check_tools
from .common import PROJECT_ROOT, atomic_json, digest_json, sha256, utc_now
from .measurement import FAILURES, aggregate, base_record, measure_trial, run_stage
from .reporting import export, print_archive_summary
from .resources import ResourceLimit, admit_disk, memory_state, resolve_memory_budget, proc_pids_match, configure_resources
from .database import DEFAULT_DATABASE, database_path, prepare_database
from .store import MeasurementStore, SCHEMA_VERSION, expected_case_ids
from .workloads import (GENERATOR_VERSION, METADATA_POLICY, directory_workloads, discover, file_count,
                        identify, matches, payload_size, quick_sample, stress_workloads)


def environment(scratch=None):
    cpu_model, cores = 'unknown', set()
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.is_file():
        for block in cpuinfo.read_text().split('\n\n'):
            data = {key.strip():value.strip() for key,value in (line.split(':',1) for line in block.splitlines() if ':' in line)}
            cpu_model = data.get('model name',cpu_model)
            if 'physical id' in data and 'core id' in data:
                cores.add((data['physical id'],data['core id']))
    sources = list(Path(__file__).parent.glob('*.py')) + [PROJECT_ROOT/'scripts/benchmark_suite.py']
    governor = Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')
    host_file = Path('/etc/machine-id')
    state = memory_state()
    return {'os':platform.platform(), 'architecture':platform.machine(), 'python':platform.python_version(),
            'cpu_model':cpu_model, 'physical_cores':len(cores) or None,'logical_cpus':os.cpu_count(),
            'affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            'host_id_sha256':sha256(host_file) if host_file.is_file() else None,
            'governor':governor.read_text().strip() if governor.is_file() else None,
            'total_ram_bytes':state['total_ram_bytes'],'effective_memory_bytes':state['effective_memory_bytes'],
            'scratch_device':Path(scratch).stat().st_dev if scratch else None,
            'harness_sha256':digest_json({p.name:sha256(p) for p in sources}),
            'cpu_quota':Path('/sys/fs/cgroup/cpu.max').read_text().strip() if Path('/sys/fs/cgroup/cpu.max').is_file() else None,
            'gpu_devices':sorted(str(p) for p in Path('/dev/dri').glob('render*')),
            'energy_measurement':'unavailable', 'gpu_vram_measurement':'unavailable',
            'process_tree_ram_measurement':'100ms samples lower bound' if proc_pids_match() else 'unavailable: /proc PID namespace mismatch',
            'metric_contract':'isolated-workflow-v3-io-full-tar-verification', 'filesystem_io_measurement':'Linux /proc/self/io including waited-for children; page dirtying and cancellations, not physical-device bytes',
            'physical_device_measurement':'unavailable', 'filesystem_cache':'uncontrolled; input prehashed before measurement'}


def checkout_identity():
    def git(*arguments):
        result = subprocess.run(['git',*arguments],cwd=PROJECT_ROOT,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=10)
        return result.stdout if result.returncode == 0 else None
    revision, diff, status = git('rev-parse','HEAD'), git('diff','HEAD'), git('status','--porcelain','--untracked-files=normal')
    return {'checkout_revision':revision.decode().strip() if revision else None,
            'checkout_dirty':bool(status),'checkout_diff_sha256':digest_json(diff.hex()) if diff is not None else None,
            'binary_build_revision':'unknown; checkout identity does not establish binary provenance'}


def repetitions_for(args, workload):
    large = payload_size(workload) > 2**20 or file_count(workload) > 10000
    repetitions = args.repetitions if args.repetitions is not None else 1 if large or args.mode=='quick' else 5
    warmups = args.warmups if args.warmups is not None else 0 if large or args.mode=='quick' else 1
    if large:
        repetitions = args.large_repetitions if args.large_repetitions is not None else repetitions
        warmups = args.large_warmups if args.large_warmups is not None else warmups
    return repetitions,warmups


def select_workloads(args):
    if getattr(args, 'campaign_manifest', None):
        from .common import snapshot
        selected = json.loads(Path(args.campaign_manifest).read_text())['workloads']
        for item in selected:
            if snapshot(item['path']) != item['snapshot']:
                raise ValueError('Campaign input changed: ' + item['target'])
        if getattr(args, 'frozen_workloads', None):
            frozen = json.loads(Path(args.frozen_workloads).read_text())
            if frozen['workloads'] != selected:
                raise ValueError('Campaign differs from frozen manifest')
        return selected, [], {'selected': {'workloads': len(selected), 'payload_bytes': sum(payload_size(i) for i in selected)}}
    root = Path(args.corpus).resolve()
    inventory, exclusions, workloads = [], [], []
    if args.suite != 'stress':
        inventory, exclusions = discover(root,args.target,args.category,args.origin or ['real'],args.include_local,True)
        if args.workload != 'directories':
            seen = {}
            for item in inventory:
                digest = item['snapshot']['sha256']
                if digest in seen and not args.include_duplicates:
                    exclusions.append({'target':item['target'],'reason':'duplicate single-file case','same_as':seen[digest]})
                else:
                    seen[digest] = item['target']
                    workloads.append(item)
        if args.workload != 'files':
            directories, excluded = directory_workloads(root,inventory,args.max_directory_bytes,args.max_directory_files,args.target)
            workloads.extend(directories)
            exclusions.extend(excluded)
    if args.suite in {'stress','full'}:
        for item in stress_workloads(args.seed, inventory if args.suite=='full' else ()):
            if matches(item['target'],args.target) and (not args.category or item['category'] in args.category):
                if args.workload=='both' or (args.workload=='files') == (item['type']=='single_file'):
                    workloads.append(item)
    bounded = []
    for item in workloads:
        limit = args.max_directory_bytes if item['type']=='directory_archive' else args.max_file_bytes
        if payload_size(item)>limit or file_count(item)>args.max_directory_files:
            exclusions.append({'target':item['target'],'reason':'workload limits','size':payload_size(item),'file_count':file_count(item)})
        else:
            bounded.append(item)
    workloads = bounded
    intended = {'workloads':len(workloads),'payload_bytes':sum(payload_size(i) for i in workloads)}
    if args.mode=='quick':
        files = quick_sample([i for i in workloads if i['type']=='single_file'],args.seed,args.max_files,args.sample_bytes)
        directories = quick_sample([i for i in workloads if i['type']=='directory_archive'],args.seed,args.max_archives,args.sample_bytes,args.max_directory_bytes)
        names = {i['target'] for i in files+directories}
        exclusions.extend({'target':i['target'],'reason':'quick sampling limits'} for i in workloads if i['target'] not in names)
        workloads = files+directories
    if not workloads:
        raise ValueError('No valid workloads selected. Select a corpus/origin, include local inputs, or use --suite stress.')
    identified = [identify(item) for item in workloads]
    if getattr(args, 'frozen_workloads', None):
        frozen = json.loads(Path(args.frozen_workloads).read_text())
        if frozen.get('schema_version') != 4 or frozen.get('workloads') != identified:
            raise ValueError('Selected workload paths, hashes, or settings differ from the frozen manifest')
    ids = [item['workload_id'] for item in identified]
    if len(ids)!=len(set(ids)):
        raise ValueError('Selected workload identities collide')
    return identified,exclusions,{'available':intended,'selected':{'workloads':len(identified),'payload_bytes':sum(payload_size(i) for i in identified)}}


@contextlib.contextmanager
def host_measurement_lock():
    runtime = Path(os.environ.get('XDG_RUNTIME_DIR',tempfile.gettempdir()))
    path = runtime/f'bato-benchmark-{os.getuid()}.lock'
    fd = os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'a') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another benchmark is measuring on this host; try again when it finishes') from error
        yield


def _run_suite(args, codecs, tool_report):
    destination = database_path(args)
    if destination == DEFAULT_DATABASE.resolve():
        prepare_database(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    store = MeasurementStore(destination)
    try:
        return _run_stored_suite(args, codecs, tool_report, store)
    finally:
        store.close()


def _run_stored_suite(args, codecs, tool_report, store):
    workloads,excluded,selection = select_workloads(args)
    output, scratch = Path(args.output_dir).resolve(), Path(args.scratch_dir).resolve()
    env = environment(scratch)
    identities = {c.identifier:c.identity() for c in codecs}
    policy = {'memory_budget':args.memory_budget if args.memory_budget else 'auto:effective RAM address-space cap',
              'bato_managed_memory_budget':getattr(args, 'bato_memory_budget', None),
              'memory_enforcement':'per-process RLIMIT_AS inherited by codec children',
              'disk_policy':'5x payload and headers + 64 MiB reserve','process_tree_sampling_ms':100}
    if not any(codec.identifier.startswith('bato') for codec in codecs):
        policy.pop('bato_managed_memory_budget')
    settings = {'schema_version':SCHEMA_VERSION,'compression_profile':getattr(args,'profile','high'),'suite':args.suite,'mode':args.mode,'track':args.track,
                'cpu_limit':getattr(args,'cpu_limit',1),'requested_threads':getattr(args,'threads',1),
                'frozen_workloads_sha256':sha256(args.frozen_workloads) if getattr(args,'frozen_workloads',None) else None,
                'environment':env,'resource_policy':policy,'seed':args.seed,'timeout':args.timeout,
                'codecs':list(identities.values()),'generator_version':GENERATOR_VERSION,'metadata_policy':METADATA_POLICY,
                'corpus':str(Path(args.corpus).resolve()) if args.suite!='stress' else None,
                'workloads':workloads,'trial_policy':{item['workload_id']:repetitions_for(args,item) for item in workloads}}
    if getattr(args, 'ensure_competitors', False):
        settings['ensure_competitors'] = True
    fingerprint = digest_json(settings)
    run_dir = output/'runs'/args.run_id
    metadata = store.run(args.run_id) if args.resume else None
    if args.resume and (not metadata or metadata.get('schema_version')!=4 or metadata['fingerprint']!=fingerprint):
        raise ValueError('Resume requires schema v4 canonical measurements, matching inputs, binaries, harness, settings, and host')
    if metadata is None:
        metadata = {'run_id':args.run_id,'started_at':utc_now(),'fingerprint':fingerprint,**settings,'label':args.label,**checkout_identity()}
    else:
        metadata['resumed_at'] = utc_now()
    # Keep the resolved cap stable across partial-run recovery; available RAM is volatile.
    budget = metadata.get('effective_memory_budget') or resolve_memory_budget(args.memory_budget)
    metadata.update(status='running',exclusions=excluded,selection=selection,tool_preflight=tool_report,
                    effective_memory_budget=budget,resources_observed=memory_state())
    metadata.setdefault('baseline_references', {})
    store.save_run(args.run_id,metadata)
    atomic_json(run_dir/'workloads.json',workloads)
    records = {row['case_id']:row for row in store.cases(args.run_id)}
    plans = []
    for item in workloads:
        tracks = ['direct'] if item['type']=='single_file' else ['native','common-tar'] if args.track=='both' else [args.track]
        for track in tracks:
            for codec in codecs:
                reps,warmups = repetitions_for(args,item)
                base = base_record(codec,{**item,'track':track},reps,warmups)
                base.update(binary_identity=identities[codec.identifier],resource_policy=policy)
                if base['case_id'] in metadata['baseline_references']:
                    continue
                if getattr(args, 'ensure_competitors', False):
                    from .baselines import BaselineRequest, find_baseline
                    reference = find_baseline(store,BaselineRequest(metadata,base))
                    if reference:
                        metadata['baseline_references'][base['case_id']] = reference
                        print(f"Reusing {codec.identifier}: {item['target']} from run {reference['run_id']} ({reference['measured_at']})",flush=True)
                        continue
                case = records.get(base['case_id'])
                if case is None or case['status'] in {'PENDING','RUNNING'}:
                    case = aggregate({**base,**(case or {})},store.trials(args.run_id,base['case_id']))
                    store.save_case(args.run_id,case)
                records[base['case_id']] = case
                plans.append((item,track,codec,base['case_id']))

    store.save_run(args.run_id,metadata)

    def project():
        cases = store.cases(args.run_id)
        observations = [dict(t,run_id=args.run_id,case_id=row['case_id']) for row in cases for t in store.trials(args.run_id,row['case_id'])]
        export(run_dir,cases,metadata,observations)
        checkpoint = getattr(args, 'checkpoint_dir', None)
        if checkpoint:
            # Runs and scratch can live outside a synchronized workspace.
            # Mirror only committed evidence between workload measurements.
            mirror = Path(checkpoint).resolve()
            if mirror == output or mirror.is_relative_to(run_dir) or mirror/'measurements.sqlite3' == store.path.resolve():
                raise ValueError('Checkpoint directory must be separate from live reports')
            mirror.mkdir(parents=True, exist_ok=True)
            temporary = mirror/'measurements.sqlite3.pending'
            destination = sqlite3.connect(temporary)
            try:
                store.connection.backup(destination)
            finally:
                destination.close()
            os.replace(temporary, mirror/'measurements.sqlite3')
            shutil.copytree(run_dir, mirror/'runs'/args.run_id, dirs_exist_ok=True)


    def commit(case_id, observation):
        store.save_trial(args.run_id,case_id,observation['phase'],observation['number'],observation)
        case = aggregate(records[case_id],store.trials(args.run_id,case_id))
        records[case_id] = case
        store.save_case(args.run_id,case)
        with (run_dir/'trials.jsonl').open('a',encoding='utf-8') as journal:
            journal.write(json.dumps(dict(observation,run_id=args.run_id,case_id=case_id),sort_keys=True)+'\n')
            journal.flush()
            os.fsync(journal.fileno())

    try:
        project()
        for position,item in enumerate(workloads,1):
            cases = [p for p in plans if p[0]['workload_id']==item['workload_id']]
            active = [p for p in cases if records[p[3]]['status'] in {'PENDING','RUNNING'}]
            if not active:
                continue
            print(f"Workload {position}/{len(workloads)}: {item['target']} | {payload_size(item)} bytes | {file_count(item)} files",flush=True)
            try:
                admit_disk(scratch,payload_size(item),file_count(item))
                with tempfile.TemporaryDirectory(prefix='workload-',dir=scratch) as temporary:
                    workspace = Path(temporary)
                    source = workspace/'input'/('tree' if item['type']=='directory_archive' else Path(item['target']).name)
                    tar_path = workspace/'payload.tar' if any(p[1]=='common-tar' for p in cases) else None
                    prepared = run_stage({'action':'prepare','workload':item,'destination':str(source),'tar_path':str(tar_path) if tar_path else None},workspace,args.timeout,budget,item['target'])
                    if prepared['status']!='PASS':
                        for _,_,_,case_id in active:
                            commit(case_id,{**prepared,'phase':'prepare','number':0})
                        continue
                    staged = {**item,'path':str(source),'snapshot':prepared['snapshot'],
                              'tar_path':str(tar_path) if tar_path else None,'tar_snapshot':prepared.get('tar_snapshot')}
                    store.save_workload(args.run_id,{**item,'snapshot':prepared['snapshot'],'tar_snapshot':prepared.get('tar_snapshot')})
                    for _,track,codec,case_id in active:
                        row = records[case_id]
                        row.update(tar_packaging_s=prepared.get('tar_packaging_s'), preparation_s=prepared.get('phase_elapsed_s'), input_sha256=prepared['snapshot']['sha256'],
                                   codec_input_sha256=prepared['tar_snapshot']['sha256'] if track=='common-tar' else prepared['snapshot']['sha256'],
                                   duplicate_byte_fraction=prepared['duplicate_byte_fraction'],unique_content_bytes=prepared['unique_content_bytes'])
                        store.save_case(args.run_id,row)
                    order = list(active)
                    random.Random(digest_json([args.seed,item['workload_id']])).shuffle(order)
                    for phase in ['warmup','measured']:
                        total = max(records[p[3]]['warmups' if phase=='warmup' else 'repetitions'] for p in order)
                        for number in range(total):
                            rotated = order[number % len(order):]+order[:number % len(order)]
                            for _,track,codec,case_id in rotated:
                                row = records[case_id]
                                if row['status'] not in {'PENDING','RUNNING'} or number>=row['warmups' if phase=='warmup' else 'repetitions']:
                                    continue
                                if any(t['phase']==phase and t['number']==number for t in store.trials(args.run_id,case_id)):
                                    continue
                                print(f"{item['target']} | {track} | {codec.name} | {phase} {number+1}/{row['warmups' if phase=='warmup' else 'repetitions']}",flush=True)
                                retained = None
                                if getattr(args, 'keep_archives', None) and phase == 'measured' and number == 0:
                                    directory = item['type'] == 'directory_archive' and track != 'common-tar'
                                    retained = Path(args.keep_archives).resolve()/args.run_id/(case_id + codec.archive_suffix(directory))
                                retention = {'retain_archive':retained} if retained is not None else {}
                                trial = measure_trial(codec,{**staged,'track':track},args.timeout,scratch,True,budget,**retention)
                                commit(case_id,{**trial,'phase':phase,'number':number})
                                print(f"{item['target']} | {track} | {codec.name} | {records[case_id]['status']} | {len(records[case_id]['trials'])} measured trials",flush=True)
            except (ResourceLimit,OSError,ValueError) as error:
                for _,_,_,case_id in active:
                    if records[case_id]['status'] in {'PENDING','RUNNING'}:
                        commit(case_id,{'status':'RESOURCE_LIMIT' if isinstance(error,ResourceLimit) else 'FAIL','stage':'prepare','error':str(error),'phase':'prepare','number':0})
            project()
        metadata['invalid_inputs'] = sum(e['reason']=='invalid' for e in excluded)
        planned_case_ids = expected_case_ids(metadata) or {case_id for _, _, _, case_id in plans}
        store.validate(args.run_id, planned_case_ids)
        final = store.cases(args.run_id)
        failed = metadata['invalid_inputs'] or any(r['status'] in FAILURES or r['status'] in {'PENDING','RUNNING'} for r in final)
        unavailable = any(t['status'] != 'READY' for t in tool_report.get('tools', []))
        metadata['status'] = 'failed' if failed else 'partial' if unavailable else 'complete'
        if unavailable:
            print('Partial coverage: one or more requested tools were unavailable or failed preflight.', flush=True)
        metadata['finished_at'] = utc_now()
    except KeyboardInterrupt:
        metadata.update(status='interrupted',interrupted_at=utc_now())
        raise
    except Exception as error:
        metadata.update(status='failed',error=str(error),finished_at=utc_now())
        raise
    finally:
        store.save_run(args.run_id,metadata)
        project()
    atomic_json(output/'latest.json',{'run_id':args.run_id,'path':str(run_dir),'status':metadata['status']})
    print_archive_summary(final)
    print(f'Results: {run_dir}')
    return 0 if metadata['status']=='complete' else 1


def run_suite(args):
    configure_resources(args)
    original = os.sched_getaffinity(0) if hasattr(os, 'sched_getaffinity') else None
    count, threads = getattr(args, 'cpu_limit', 1), getattr(args, 'threads', 1)
    if threads > count:
        raise ValueError('Worker count must fit the CPU allocation')
    if original is not None:
        if count > len(original):
            raise ValueError('CPU allocation exceeds available affinity')
        os.sched_setaffinity(0, sorted(original)[:count])
    elif count != 1:
        raise ValueError('This platform cannot enforce equal CPU allocations')
    try:
        return _run_allocated_suite(args)
    finally:
        if original is not None:
            os.sched_setaffinity(0, original)


def _run_allocated_suite(args):
    if args.list_workloads:
        workloads,excluded,selection = select_workloads(args)
        print(json.dumps({'schema_version':4,'workloads':workloads,'exclusions':excluded,'selection':selection},indent=2))
        return 0
    with host_measurement_lock():
        old_limit = resource.getrlimit(resource.RLIMIT_AS)
        from .resources import limit_address_space
        try:
            limit_address_space(resolve_memory_budget(args.memory_budget))
            preflight_workload = 'both' if args.workload=='directories' and args.track in {'both','common-tar'} else args.workload
            codecs,report = check_tools(args.codecs,args.only_bato,args.bato_bin,preflight_workload,min(getattr(args,'tool_check_timeout',30),args.timeout),
                allow_unavailable=getattr(args,'allow_unavailable',False),profile=getattr(args,'profile','high'),
                threads=getattr(args,'threads',1),bato_modes=getattr(args,'bato_modes','cpu1'),
                memory_budget=getattr(args,'bato_memory_budget',256 * 2**20),
                chunk_size=getattr(args,'bato_chunk_size',256 * 1024),
                segment_size=getattr(args,'bato_segment_size',64 * 2**20),
                codec_tuning=json.loads(Path(args.codec_tuning).read_text()) if getattr(args,'codec_tuning',None) else None,
                nncp_profile=getattr(args,'nncp_profile','default'),nncp_cuda=getattr(args,'nncp_cuda',False))
        finally:
            resource.setrlimit(resource.RLIMIT_AS,old_limit)
        if report['status']!='ready':
            return 2
        if args.check_tools:
            return 0
        root,output = Path(args.corpus).resolve(),Path(args.output_dir).resolve()
        if args.suite!='stress' and not root.is_dir():
            raise ValueError('Corpus directory does not exist')
        args.scratch_dir = str(Path(args.scratch_dir).resolve() if args.scratch_dir else output/'scratch')
        if args.suite!='stress' and (output.is_relative_to(root) or Path(args.scratch_dir).is_relative_to(root)):
            raise ValueError('Benchmark output and scratch directories must be outside the corpus')
        if getattr(args, 'checkpoint_dir', None) and Path(args.checkpoint_dir).resolve().is_relative_to(root):
            raise ValueError('Checkpoint directory must be outside the corpus')
        if getattr(args, 'keep_archives', None) and Path(args.keep_archives).resolve().is_relative_to(root):
            raise ValueError('Retained archives must be outside the corpus')
        args.run_id = args.resume or utc_now().replace(':','-')+'-'+uuid.uuid4().hex[:8]
        if Path(args.run_id).name!=args.run_id or args.run_id in {'.','..'}:
            raise ValueError('--resume requires a run ID, not a path')
        run_dir = output/'runs'/args.run_id
        if args.resume:
            if not run_dir.is_dir():
                raise ValueError('Unknown resume run ID')
            if database_path(args) == DEFAULT_DATABASE.resolve():
                prepare_database(database_path(args))
            if not database_path(args).is_file():
                raise ValueError('Resume requires schema v4 and its canonical measurements.sqlite3; legacy runs are preserved')
            from .compare import load_run
            previous, _ = load_run(output,args.run_id,database_path(args))
            if previous.get('schema_version')!=4:
                raise ValueError('Resume requires schema v4 and its canonical measurements.sqlite3; legacy runs are preserved')
        run_dir.mkdir(parents=True,exist_ok=True)
        Path(args.scratch_dir).mkdir(parents=True,exist_ok=True)
        with contextlib.ExitStack() as stack:
            locks = [(run_dir/'run.lock',fcntl.LOCK_EX)]
            if args.suite!='stress':
                locks.append((root/'.corpus.lock',fcntl.LOCK_SH))
            for path,mode in locks:
                stream = stack.enter_context(path.open('a'))
                try:
                    fcntl.flock(stream,mode|fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError('Corpus is being modified or this run is already active') from error
            return _run_suite(args,codecs,report)

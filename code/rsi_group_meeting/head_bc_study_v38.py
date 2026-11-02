"""CPU-only registration for a finite, stronger fixed BC diagnostic. Never a method gain claim."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, hashlib, json, shutil, time

ROOT=Path('F:/robot').resolve()
W=ROOT/'rsi_group_meeting/v38/full_teacher_head_bc'
OLD=ROOT/'rsi_group_meeting/v32/repair_learning_dev'
CROSS=ROOT/'rsi_group_meeting/v35/cross_context_diagnostic'
PRIOR=ROOT/'rsi_group_meeting/v36/head_uniform_parameterization'
REVIEW=ROOT/'rsi_group_meeting/v36/root_completed_review_recovered_analysis'
FIT=ROOT/'rsi_group_meeting/v30/research_agent/offline_teacher_fit_diagnostic'
SOURCES=['head_bc_runtime_v38.py','head_bc_study_v38.py','train_head_bc_v38.py',
         'head_bc_worker_v38.py','evaluate_head_bc_v38.py','analyze_head_bc_v38.py',
         'score_head_bc_offline_v38.py','run_head_bc_sequence_v38.py']

def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as stream:json.dump(x,stream,ensure_ascii=False,indent=2)
def status(stage,**items):
    file=W/'status.working.json'
    file.write_text(json.dumps(dict(updated_utc=now(),stage=stage,work_demonstrated=False,**items),ensure_ascii=False,indent=2),'utf8')
    file.replace(W/'status.json')

def ready():
    old=read(OLD/'registration.json');prior=read(PRIOR/'registration.json')
    review=read(REVIEW/'review.json');result=read(PRIOR/'analysis/head_readout.json')
    offline=read(PRIOR/'offline_fit/summary.json')
    assert len(review['rows'])==len(result['rows'])==12
    assert review['registration_sha256']==sha(PRIOR/'registration.json')
    assert review['frozen_analysis_sha256']==sha(PRIOR/'analysis/head_readout.json')
    assert review['offline_fit_sha256']==sha(PRIOR/'offline_fit/summary.json')
    assert offline['new_head_density_forwards']==3291 and sum(offline['old_actor_rows_reused'].values())==9873
    assert sha(PRIOR/'offline_fit/head_per_action_fit.jsonl')==offline['new_fit_file_sha256']
    assert read(PRIOR/'evaluation_complete.json')['physical_episodes']==12
    assert read(PRIOR/'evaluation_worker_close.json')['closed']
    assert read(PRIOR/'training_complete.json')['optimizer_steps']==20
    return old,prior

def training_pools():
    """All3163 cached teacher actions, including63 starts; unchanged ordered128 retention."""
    import numpy as np
    membership=read(FIT/'diagnostic_membership.json')
    teachers=[copy.deepcopy(x) for x in membership if x['scope']=='teacher']
    retention=copy.deepcopy(read(OLD/'training_membership.json')['selected_targets']['retention'])
    assert len(teachers)==3163 and len(retention)==128
    identities=[(Path(x['file']).as_posix(),x['action_index']) for x in teachers]
    assert len(set(identities))==3163
    by_teacher={}
    for row in teachers:
        assert row['feature_field']=='spatial_features' and row['action_index']>=0
        by_teacher.setdefault(row['teacher_key'],[]).append(row)
    cache=read(OLD/'cache_complete.json');by_cache={x['key']:x for x in cache['teachers']}
    assert set(by_teacher)==set(by_cache) and len(by_teacher)==7
    for key,rows in by_teacher.items():
        receipt=by_cache[key];file=ROOT/rows[0]['file']
        assert sha(file)==receipt['sha256']
        assert all(x['file']==rows[0]['file'] for x in rows)
        assert [x['action_index'] for x in rows]==list(range(receipt['controls']))
        with np.load(file) as z:
            assert z['spatial_features'].shape==(receipt['controls'],4,768) and z['spatial_features'].dtype==np.float32
            assert z['actions'].shape==(receipt['controls'],7)
    assert sum(x['action_index']<9 for x in teachers)==63
    assert len([x for x in membership if x['scope']=='retention'])==128
    # Retention order is the old training order, not the differently sorted offline order.
    assert {(Path(x['file']).as_posix(),x['action_index']) for x in retention}=={(Path(x['file']).as_posix(),x['action_index']) for x in membership if x['scope']=='retention'}
    return dict(full_teacher=teachers,retention=retention)

def sampling_plan(pools):
    import numpy as np
    rng=np.random.default_rng(20265450);steps=[];counts={k:np.zeros(len(v),dtype=np.int64) for k,v in pools.items()}
    for step in range(1,2001):
        row=dict(step=step)
        for pool in ['full_teacher','retention']:
            row[pool]=rng.integers(0,len(pools[pool]),size=16).tolist()
            np.add.at(counts[pool],row[pool],1)
        steps.append(row)
    coverage=dict(teacher_pool=3163,retention_pool=128,teacher_presentations=int(counts['full_teacher'].sum()),
                  retention_presentations=int(counts['retention'].sum()),teacher_distinct=int((counts['full_teacher']>0).sum()),
                  retention_distinct=int((counts['retention']>0).sum()),teacher_min_presentations=int(counts['full_teacher'].min()),
                  teacher_max_presentations=int(counts['full_teacher'].max()),
                  early_history_targets=63,early_history_presentations=sum(int(counts['full_teacher'][i]) for i,r in enumerate(pools['full_teacher']) if r['action_index']<9),
                  servo_presentations=sum(int(counts['full_teacher'][i]) for i,r in enumerate(pools['full_teacher']) if r['stage']=='servo'),
                  teacher_target_counts=counts['full_teacher'].tolist(),retention_target_counts=counts['retention'].tolist())
    assert coverage['teacher_presentations']==coverage['retention_presentations']==32000
    assert coverage['teacher_distinct']==3163 and coverage['retention_distinct']==128
    assert coverage['early_history_presentations']==647 and coverage['servo_presentations']==4196
    return steps,coverage

def register():
    assert Path(__file__).resolve()==ROOT/'rsi_group_meeting/head_bc_study_v38.py','Preparation directory is not executable registration; root must review/copy first.'
    began=time.perf_counter();old,prior=ready();assert not W.exists(),'Preserve any old preparation/attempt.'
    pools=training_pools();plan,coverage=sampling_plan(pools)
    sources={};inputs={};copies=[]
    def bind(path,expected=None,source=False):
        path=Path(path).resolve();dest=sources if source else inputs;key=rel(path)
        if key in dest:
            if expected is not None:assert dest[key]==expected,str(path)
            return dest[key]
        value=sha(path)
        if expected is not None:assert value==expected,str(path)
        dest[key]=value
        return value
    for name in SOURCES:bind(ROOT/'rsi_group_meeting'/name,source=True)
    for name,value in prior['implementation_sha256'].items():bind(ROOT/name,value,True)
    for name,value in prior['input_files_sha256'].items():bind(ROOT/name,value)
    for file in [PRIOR/'registration.json',PRIOR/'analysis_frozen_before_head_outcomes.json',PRIOR/'fixed_offline_membership.json',
                 PRIOR/'analysis/head_readout.json',PRIOR/'analysis/episode_rows.csv',PRIOR/'offline_fit/summary.json',
                 PRIOR/'offline_fit/head_per_action_fit.jsonl',PRIOR/'trained_actors.json',PRIOR/'training_receipt.json',
                 PRIOR/'training_complete.json',PRIOR/'evaluation_complete.json',PRIOR/'evaluation_worker_close.json',
                 REVIEW/'review.json',REVIEW/'cost_increment.json',ROOT/'rsi_group_meeting/review_head_completion_recovered_v36.py',
                 OLD/'cache_complete.json',FIT/'diagnostic_membership.json']:
        bind(file)
    bind(old['parent']['file'],old['parent']['sha256'])
    fit_sources=read(FIT/'protocol.json')['sources_sha256']
    for row in pools['full_teacher']+pools['retention']:
        file=ROOT/row['file'];expected=fit_sources.get(rel(file))
        # Prior registration binds this exact file even if the old path used Windows separators.
        assert rel(file) in prior['input_files_sha256'];bind(file,prior['input_files_sha256'][rel(file)])
        if expected is not None:assert inputs[rel(file)]==expected
    def reference(folder,key):
        file=folder/'results'/f'{key}.json';r=read(file);bind(file)
        for path,h in [(file.with_suffix('.npz'),r['trace_sha256']),(folder/r['events'],r['events_sha256']),(folder/r['video'],r['video_sha256']),(folder/r['initial_sensors'],r['initial_sensors_sha256'])]:bind(path,h)
        return dict(result=rel(file),initial_sensors=rel(folder/r['initial_sensors']))
    cases=[]
    for previous in prior['cases']:
        case=copy.deepcopy(previous);key=previous['episode']['key']
        case['references']['head20_bc']=reference(PRIOR/'evaluation',key)
        case['episode']['key']=key.replace('__head_uniform_bc','__full_teacher_head_bc')
        case['episode']['actor']='full_teacher_head_bc';assert set(case['references'])=={'parent','uniform_bc','local_bc','head20_bc'}
        cases.append(case)
    assert len(cases)==12 and all(sum(c['episode']['context']['partition']==part for c in cases)==4 for part in ['training_context','test_context','source_regression'])
    parity=copy.deepcopy(prior['parity']);assert len(parity)==4
    W.mkdir(parents=True)
    def cp(src,dst):
        bind(src);dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copy2(src,dst)
        assert sha(src)==sha(dst);copies.append(dict(source=rel(src),copy=rel(dst),sha256=sha(src)))
    for i,item in enumerate(parity):
        dst=W/'parity_inputs'/f'{i:02d}.npz';cp(Path(item['observation_file']),dst);item['observation_file']=str(dst)
    for src in sorted((PRIOR/'evaluation/canonical').iterdir()):cp(src,W/'evaluation/canonical'/src.name)
    save(W/'fixed_training_membership.json',dict(selected_targets=pools,teacher_pool_size=3163,retention_pool_size=128,
                                               teacher_data_source='All7 original cached successful v31 teachers; no v35 or newer teachers.',
                                               old_retention_membership_sha256=sha(OLD/'training_membership.json'),
                                               full_teacher_membership_source_sha256=sha(FIT/'diagnostic_membership.json')))
    save(W/'fixed_training_sampling.json',plan);save(W/'fixed_training_coverage.json',coverage)
    save(W/'fixed_offline_membership.json',read(FIT/'diagnostic_membership.json'))
    p=dict(registered_utc=now(),scope='One fixed stronger full-teacher BC diagnostic, not an innovation/M/RSI or independent generalization experiment. All12 development cases were previously exposed.',
           parent=old['parent'],frozen_pretrained_parameters_sha256=old['frozen_pretrained_parameters_sha256'],public_sensors=old['public_sensors'],
           evaluation_primary='Official complete original task success by520.',evaluation_extension='By900 is the same-trajectory extension, never new samples; late-only success is not a primary repair.',
           implementation_sha256=sources,input_files_sha256=inputs,copies=copies,
           fixed_training_membership_sha256=sha(W/'fixed_training_membership.json'),fixed_training_sampling_sha256=sha(W/'fixed_training_sampling.json'),
           fixed_training_coverage_sha256=sha(W/'fixed_training_coverage.json'),fixed_offline_membership_sha256=sha(W/'fixed_offline_membership.json'),
           method=copy.deepcopy(prior['method']),
           training=dict(optimizer='AdamW',learning_rate=1e-4,weight_decay=0,betas=[.9,.999],eps=1e-8,gradient_clip=1.,joint_KL_coefficient=.001,
                         steps=2000,teacher_per_step=16,retention_per_step=16,rng_seed=20265450,teacher_presentations=32000,retention_presentations=32000,
                         sampling='Uniform with replacement over3163 action targets, then original128 retention; exact NumPy RNG sequence frozen before new queries.',
                         history='Unpadded real pre-action features[max(0,index-9):index+1], lengths1..10; all63 index0..8 targets included.',
                         checkpoint='Only final step2000; no intermediate candidates, early stop or best-checkpoint selection.',
                         initialization='Original author pi0 plus new zero-B head; never continue any step20 candidate.',
                         data='All3163 original teacher targets and unchanged128 retention; uniform over actions, not equal per trajectory. Full coverage and optimization budget change together.'),
           parity=parity,parity_rule=prior['parity_rule'],cases=cases,
           budget=dict(new_training_runs=1,new_optimizer_steps=2000,loss_presentations=64000,teacher_presentations=32000,retention_presentations=32000,
                       training_distribution_forwards=128000,parity_queries=8,parity_encodings=8,parity_action_samples=8,parity_physical_controls=0,
                       parity_numpy_records=8,new_teacher_episodes=0,new_training_feature_encoders=0,evaluation_episodes=12,
                       maximum_evaluation_controls=10800,evaluation_settling_controls=120,maximum_evaluation_feature_packets=10800,
                       additional_model_calls_for_packet_preservation=0,offline_new_head_density_forwards=3291,
                       offline_old_actor_forwards=0,offline_old_rows_reused=13164,offline_encoders_or_samples_or_controls=0),
           old_fit_per_action_file=prior['old_fit_per_action_file'],old_fit_summary_file=prior['old_fit_summary_file'],
           old_head_fit_per_action_file=rel(PRIOR/'offline_fit/head_per_action_fit.jsonl'),old_head_fit_summary_file=rel(PRIOR/'offline_fit/summary.json'),
           old_head_analysis_file=rel(PRIOR/'analysis/head_readout.json'),old_head_review_file=rel(REVIEW/'review.json'),
           old_cross_analysis_file=prior['old_cross_analysis_file'],old_cross_review_file=prior['old_cross_review_file'],
           original_student_analysis_file=prior['original_student_analysis_file'],
           limitations=['A stronger standard BC baseline, not a new repair method, M or RSI. No ordinaryRL run fabricated from teacher actions.',
                        'The start is author source90 pretraining; old Long parent7/90 may reflect missing target-task adaptation, not repair of a previously competent full-task system. A gain is first a supervised adaptation baseline.',
                        'Full3163 supervision and2000 steps change jointly versus old128/20; not single-factor causality or equal-budget superiority.',
                        'No frozen-sigma claim: shared head may alter means/scales/logits. Density/mean-fit is not task success or proof of a unique failure mechanism.',
                        'Retention/KL/representation constraints remain; a negative result does not reject BC or establish a theoretical supervision ceiling.',
                        'All12 contexts are previously exposed development; training-context success never independent generalization/work.',
                        'Old teacher failure/late success, U/L outcomes and local source harm remain visible; zero-B parity does not imply trained-policy parity.',
                        'Uniform action sampling weights longer teachers more; correlated frames are not independent robot trials.'],
           stopping='Exactly2000 steps, all12 evaluations, one frozen analysis, one3291-new-head offline pass. Preserve all attempts/prefixes/numerical failures; no auto retry or LR/epoch/rank/sigma/window search.',
           registration_cpu_wall_seconds=time.perf_counter()-began,work_demonstrated=False)
    save(W/'registration.json',p)
    save(W/'analysis_frozen_before_head_outcomes.json',dict(frozen_utc=now(),registration_sha256=sha(W/'registration.json'),
         analyzer_sha256=sha(ROOT/'rsi_group_meeting/analyze_head_bc_v38.py'),offline_scorer_sha256=sha(ROOT/'rsi_group_meeting/score_head_bc_offline_v38.py'),new_outcome_files_at_freeze=0))
    print('Registered finite full-teacher BC; no model, training or control started.')

def verify():
    p=read(W/'registration.json')
    for name,value in {**p['implementation_sha256'],**p['input_files_sha256']}.items():assert sha(ROOT/name)==value,name
    for item in p['copies']:assert sha(ROOT/item['source'])==sha(ROOT/item['copy'])==item['sha256']
    for field,file in [('fixed_training_membership_sha256','fixed_training_membership.json'),('fixed_training_sampling_sha256','fixed_training_sampling.json'),
                       ('fixed_training_coverage_sha256','fixed_training_coverage.json'),('fixed_offline_membership_sha256','fixed_offline_membership.json')]:assert sha(W/file)==p[field]
    freeze=read(W/'analysis_frozen_before_head_outcomes.json');assert freeze['registration_sha256']==sha(W/'registration.json')
    assert freeze['analyzer_sha256']==sha(ROOT/'rsi_group_meeting/analyze_head_bc_v38.py')
    assert freeze['offline_scorer_sha256']==sha(ROOT/'rsi_group_meeting/score_head_bc_offline_v38.py')
    return p

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','verify']);args=parser.parse_args()
    register() if args.command=='register' else verify()

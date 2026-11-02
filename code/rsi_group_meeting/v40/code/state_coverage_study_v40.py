"""CPU registration for one finite standard BC/replay-state support control. Not M/RSI."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, copy, hashlib, json, shutil, time

ROOT=Path('F:/robot').resolve()
CODE=ROOT/'rsi_group_meeting/v40/code'
W=ROOT/'rsi_group_meeting/v40/state_coverage_kl_control'
PRIOR=ROOT/'rsi_group_meeting/v38/full_teacher_head_bc'
OLD=ROOT/'rsi_group_meeting/v32/repair_learning_dev'
V36=ROOT/'rsi_group_meeting/v36/head_uniform_parameterization'
REVIEW=ROOT/'rsi_group_meeting/v38/root_completed_review'
CP=Path('C:/Users/c312_/.codex/visualizations/2026/10/14/01a138a6-cbdb-7480-aa7c-23a07b672490/coverage_control_preparation_20261102')
PROPOSAL_FILES={
 'membership':CP/'root/unregistered_membership_revision2.json',
 'future_conditions':CP/'root/unregistered_future_conditions_candidate.json',
 'exposure_receipt':CP/'exposure/exposure_review_receipt.json',
 'exposure_readout':CP/'exposure/sixteen_condition_readout.json',
 'exposure_sources':CP/'exposure/source_metadata_hash_manifest.json',
 'packet_manifest':CP/'research/all_1447_packet_manifest.json',
 'packet_receipt':CP/'research/packet_manifest_receipt.json',
}
SOURCES=['state_coverage_study_v40.py','state_coverage_runtime_v40.py','train_state_coverage_v40.py',
 'worker_state_coverage_v40.py','evaluate_state_coverage_v40.py','audit_state_coverage_exposure_v40.py','analyze_state_coverage_v40.py','run_state_coverage_sequence_v40.py']
ARMS=['oldA_KL','coverage_KL']
TRAINING_SEED=20265500
EVALUATION_SEED=20265510

def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as stream:json.dump(x,stream,ensure_ascii=False,indent=2)
def status(stage,**items):
    W.mkdir(parents=True,exist_ok=True);file=W/'status.working.json'
    file.write_text(json.dumps(dict(updated_utc=now(),stage=stage,work_demonstrated=False,**items),ensure_ascii=False,indent=2),'utf8')
    file.replace(W/'status.json')

def ready():
    """Read closed evidence only. No output creation, model import, encoding or physics."""
    prior=read(PRIOR/'registration.json');old=read(OLD/'registration.json')
    review=read(REVIEW/'review.json');offline=read(PRIOR/'offline_fit/summary.json')
    assert len(review['rows'])==12 and review['registration_sha256']==sha(PRIOR/'registration.json')
    assert review['frozen_analysis_sha256']==sha(PRIOR/'analysis/head_readout.json')
    assert review['offline_fit_sha256']==sha(PRIOR/'offline_fit/summary.json')
    assert read(PRIOR/'owner_complete.json')['all_four_commands_returned_zero']
    assert offline['new_head_density_forwards']==3291 and offline['old_total_rows_reused']==13164
    assert read(PRIOR/'training_complete.json')['optimizer_steps']==2000
    assert read(PRIOR/'evaluation_complete.json')['physical_episodes']==12
    assert read(PRIOR/'evaluation_worker_close.json')['closed']
    proposal={k:read(p) for k,p in PROPOSAL_FILES.items()}
    assert proposal['exposure_receipt']['candidate_sha256']==sha(PROPOSAL_FILES['future_conditions'])
    assert not proposal['exposure_receipt']['hash_mismatches_against_previous_index']
    assert not proposal['exposure_receipt']['parse_or_read_errors']
    assert len(proposal['exposure_readout'])==16
    assert all(not r['actual_completed_metadata_count'] and not r['probe_or_choice_count'] and not r['attempt_only_count'] for r in proposal['exposure_readout'])
    return dict(prior=prior,old=old,proposal=proposal)

def prepare_registration_inputs():
    """Pure CPU, returns proposed frozen objects. register() alone writes newW."""
    import numpy as np
    q=ready();prior=q['prior'];proposal=q['proposal'];m=proposal['membership']
    teacher=copy.deepcopy(read(PRIOR/'fixed_training_membership.json')['selected_targets']['full_teacher'])
    oldA=copy.deepcopy(m['control_old_A128']);old64=copy.deepcopy(m['coverage_old_A64'])
    new64=copy.deepcopy(m['coverage_cached_source64'])
    assert len(teacher)==3163 and len(oldA)==128 and len(old64)==len(new64)==64
    original=read(PRIOR/'fixed_training_membership.json')['selected_targets']['retention']
    def identity(r):return (rel(ROOT/r['file']),r['feature_field'],r['action_index'])
    assert [identity(r) for r in oldA]==[identity(r) for r in original]
    assert all(r['action_index']>=9 and r['feature_field']=='frozen_spatial_features' for r in oldA)
    # Keep historical metadata if proposal rows use a simplified representation.
    original_by_identity={identity(r):r for r in original}
    oldA=[dict(copy.deepcopy(original_by_identity[identity(r)]),**r) for r in oldA]
    old64=[dict(copy.deepcopy(original_by_identity[identity(r)]),**r) for r in old64]
    assert len({(r['file'],r['action_index']) for r in oldA})==128
    assert {(r['file'],r['action_index']) for r in old64}.issubset({(r['file'],r['action_index']) for r in oldA})
    packet_rows={r['packet']:r for r in proposal['packet_manifest']['rows']}
    assert len(packet_rows)==1447
    cache_hashes={}
    def checked(path,expected):
        path=Path(path).resolve();key=rel(path)
        if key not in cache_hashes:cache_hashes[key]=sha(path)
        assert cache_hashes[key]==expected,key
        return key
    def old_entry(r):
        out=copy.deepcopy(r);out['kind']='old_A_state';out['file']=rel(ROOT/r['file'])
        out['history_length']=10;out['labels_used']='No retention action target; common-parent distribution only.'
        return out
    new=[]
    for r in new64:
        i=r['action_index_zero_based'];paths=r['history_packet_paths']
        assert i>=9 and len(paths)==r['history_length']==10
        assert [int(Path(p).stem) for p in paths]==list(range(i-9,i+1))
        assert all(Path(p).parent==Path(r['packet']).parent for p in paths)
        assert paths[-1]==r['packet'] and r['feature_field']=='spatial_feature'
        histories=[]
        for j,path in enumerate(paths):
            row=packet_rows[path];assert row['case_key']==r['case_key'] and row['action_index_zero_based']==i-9+j
            key=checked(path,row['packet_sha256']);checked(row['receipt'],row['receipt_sha256'])
            histories.append(dict(file=key,sha256=row['packet_sha256']))
        new.append(dict(kind='cached_source_state',case_key=r['case_key'],action_index=i,history_length=10,
          feature_field='spatial_feature',history_packets=histories,labels_used='No action/proposal/goal label; public frozen-state history for common-parent KL only.'))
    assert len({(r['case_key'],r['action_index']) for r in new})==64
    cases_by_source={r['case_key'] for r in new};assert len(cases_by_source)==4
    assert all(sum(r['case_key']==key for r in new)==16 for key in cases_by_source)
    arms={'oldA_KL':[old_entry(r) for r in oldA], 'coverage_KL':[old_entry(r) for r in old64]+new}
    # Shape/finite/public-case/history checks on every actually used source packet, never action labels.
    for rows in arms.values():
        for row in rows:
            if row['kind']=='cached_source_state':
                for packet in row['history_packets']:
                    with np.load(ROOT/packet['file'],allow_pickle=False) as z:
                        f=z['spatial_feature'];assert f.shape==(4,768) and f.dtype==np.float32 and np.isfinite(f).all()
                        assert z['query_phase'].item()=='case' and z['actor'].item()=='head_uniform_bc'
            else:
                file=ROOT/row['file'];expected=prior['input_files_sha256'][rel(file)];checked(file,expected)
                with np.load(file,allow_pickle=False) as z:
                    f=z['frozen_spatial_features'];i=row['action_index'];assert f.dtype==np.float32 and f.shape[1:]==(4,768)
                    assert 9<=i<len(f) and f[i-9:i+1].shape==(10,4,768) and np.isfinite(f[i-9:i+1]).all()
    teacher_files={}
    for row in teacher:teacher_files.setdefault(row['file'],[]).append(row)
    assert len(teacher_files)==7 and sum(r['action_index']<9 for r in teacher)==63
    for file,rows in teacher_files.items():
        path=ROOT/file;checked(path,prior['input_files_sha256'][rel(path)])
        with np.load(path,allow_pickle=False) as z:
            f=z['spatial_features'];a=z['actions'];assert f.dtype==np.float32 and f.shape[1:]==(4,768) and a.shape==(len(f),7)
            assert [r['action_index'] for r in rows]==list(range(len(f)))
    rng=np.random.default_rng(TRAINING_SEED);sampling=[];tc=np.zeros(3163,dtype=np.int64);rc=np.zeros(128,dtype=np.int64)
    for step in range(1,2001):
        td=rng.integers(0,3163,size=16).tolist();rd=rng.integers(0,128,size=16).tolist()
        sampling.append(dict(step=step,full_teacher=td,retention=rd));np.add.at(tc,td,1);np.add.at(rc,rd,1)
    coverage=dict(teacher_pool=3163,retention_pool_per_arm=128,teacher_presentations_per_arm=int(tc.sum()),
      retention_state_presentations_per_arm=int(rc.sum()),teacher_distinct=int((tc>0).sum()),retention_distinct=int((rc>0).sum()),
      teacher_counts=tc.tolist(),retention_index_counts=rc.tolist(),same_plan_for_both_arms=True,
      early_teacher_target_count=63,early_teacher_presentations=sum(int(tc[i]) for i,r in enumerate(teacher) if r['action_index']<9),
      note='Index draws match across arms; physical-state identities differ at retention indices. Correlated states are not independent tasks.')
    assert tc.sum()==rc.sum()==32000 and (rc>0).all()
    coverage['teacher_min_presentations']=int(tc.min());coverage['teacher_max_presentations']=int(tc.max())
    coverage['unpresented_teacher_target_indices']=[int(i) for i in np.flatnonzero(tc==0)]
    coverage['uniform_replacement_does_not_guarantee_full_pool_presentation']=True
    conditions=[];cases=[]
    raw=proposal['future_conditions']['conditions'];assert len(raw)==16
    exposure={(r['suite'],r['task_id'],r['perturbation'],r['state_index']) for r in proposal['exposure_readout']}
    for item in raw:
        s=copy.deepcopy(item['setting']);state=item['state_index'];task=s['task_id'];suite=s['suite'];pert=s['perturbation']
        assert (suite,task,pert,state) in exposure and pert=='original'
        for field,hfield in [('bddl','bddl_sha256'),('init_file','init_sha256')]:checked(ROOT/s[field],s[hfield])
        part='Long' if suite=='libero_10' else ('source_coverage' if task in [9,34] else 'old_A')
        cap=900 if suite=='libero_10' else 520
        context_id=f'{suite}_task{task}_{pert}_init{state}_noise{EVALUATION_SEED}'
        c=dict(context_id=context_id,suite=suite,setting=s,state_index=state,partition=part,policy_seed=EVALUATION_SEED,max_task_controls=cap)
        conditions.append(c)
        for actor in ['parent']+ARMS:cases.append(dict(key=f'{context_id}__{actor}',context=copy.deepcopy(c),actor=actor))
    assert len(cases)==48 and len({c['context_id'] for c in conditions})==16
    assert sum(c['max_task_controls'] for c in conditions)*3==29520
    assert {p:sum(c['partition']==p for c in conditions) for p in ['old_A','source_coverage','Long']}==dict(old_A=8,source_coverage=4,Long=4)
    return dict(**q,pools=dict(full_teacher=teacher,retention_by_arm=arms),sampling=sampling,coverage=coverage,
      conditions=conditions,cases=cases,cache_hashes=cache_hashes,parity=copy.deepcopy(prior['parity']))

def register():
    assert Path(__file__).resolve()==CODE/'state_coverage_study_v40.py','Preparation-only copy; root must review/install.'
    began=time.perf_counter();q=prepare_registration_inputs();assert not W.exists(),'Never overwrite a previous registration/attempt.'
    sources={};inputs={};copies=[]
    def bind(path,expected=None,source=False):
        path=Path(path).resolve();key=rel(path);dest=sources if source else inputs
        value=sha(path)
        if expected is not None:assert value==expected,key
        if key in dest:assert dest[key]==value
        dest[key]=value;return value
    for name in SOURCES:bind(CODE/name,source=True)
    # Bind all immutable author/runtime/backend sources already used by closedv38, without changing them.
    for path,value in q['prior']['implementation_sha256'].items():bind(ROOT/path,value,True)
    for path,value in q['prior']['input_files_sha256'].items():bind(ROOT/path,value)
    for path,value in q['cache_hashes'].items():bind(ROOT/path,value)
    for f in [PRIOR/'registration.json',PRIOR/'owner_complete.json',PRIOR/'analysis/head_readout.json',PRIOR/'offline_fit/summary.json',
              REVIEW/'review.json',REVIEW/'cost_increment.json',PRIOR/'fixed_training_membership.json',V36/'registration.json',
              V36/'evaluation_worker_metadata.json',V36/'trained_actors.json',V36/'evaluation_query_receipts.jsonl']:
        bind(f)
    W.mkdir(parents=True)
    external=[]
    for key,src in PROPOSAL_FILES.items():
        dst=W/'frozen_proposals'/f'{key}.json';dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        assert sha(src)==sha(dst);bind(dst);external.append(dict(source=str(src),copy=rel(dst),sha256=sha(src)))
    parity=q['parity'];assert len(parity)==4
    for i,item in enumerate(parity):
        src=Path(item['observation_file']);bind(src);dst=W/'parity_inputs'/f'{i:02d}.npz';dst.parent.mkdir(parents=True,exist_ok=True)
        assert not dst.exists();shutil.copy2(src,dst);assert sha(src)==sha(dst)
        copies.append(dict(source=rel(src),copy=rel(dst),sha256=sha(src)));item['observation_file']=str(dst)
    for file,obj in [('fixed_training_membership.json',q['pools']),('fixed_training_sampling.json',q['sampling']),
      ('fixed_training_coverage.json',q['coverage']),('fixed_future_conditions.json',q['conditions'])]:save(W/file,obj)
    p=dict(registered_utc=now(),scope='Finite standard BC/replay-state-support gate. Not learnedimprover, innovation, RSI, fullbenchmark or guaranteed independence.',
      parent=q['old']['parent'],frozen_pretrained_parameters_sha256=q['old']['frozen_pretrained_parameters_sha256'],
      public_sensors=q['old']['public_sensors'],arms=ARMS,implementation_sha256=sources,input_files_sha256=inputs,
      external_proposal_copies=external,copies=copies,method=copy.deepcopy(q['prior']['method']),
      fixed_training_membership_sha256=sha(W/'fixed_training_membership.json'),fixed_training_sampling_sha256=sha(W/'fixed_training_sampling.json'),
      fixed_training_coverage_sha256=sha(W/'fixed_training_coverage.json'),fixed_future_conditions_sha256=sha(W/'fixed_future_conditions.json'),
      conditions=q['conditions'],cases=q['cases'],evaluation_policy_seed=EVALUATION_SEED,parity=parity,parity_rule=q['prior']['parity_rule'],
      evaluation_primary='Official complete-task success by520 for allconditions.',evaluation_extension='Long alone may continue same trajectory to900; not a new trial or substitute primary endpoint.',
      training=dict(arms=ARMS,optimizer='AdamW',learning_rate=1e-4,weight_decay=0,betas=[.9,.999],eps=1e-8,gradient_clip=1.,
        optimizer_steps_per_arm=2000,teacher_per_step=16,retention_per_step=16,rng_seed=TRAINING_SEED,initialization_seed=77210,
        teacher_loss='(action NLL +.001 KL_joint(current||pi0))/32 per teacher target',
        retention_loss='1.0 KL_joint(current||pi0)/32 per retention state; no retention action NLL or action labels read.',
        reference='Original authorpi0, not trainedhead and not another arm; scale0 exactly recovers this commonparent.',
        initialization='Independent originalauthor+samezeroB head eacharm; never continue v38 or firstarm.',
        history='Teacher real1..10; bothretention sets exactly10 real past/current features, no future/padding.',
        checkpoint='Only finalstep2000 each; no earlystop, bestcheckpoint, addedepochs or coefficient search.',
        objective_direction='Current-to-parent auxiliary-component/action jointKL upperbound, not exact marginalGMMKL.'),
      budget=dict(new_training_runs=2,total_optimizer_steps=4000,optimizer_steps_per_arm=2000,
        teacher_presentations_per_arm=32000,retention_state_presentations_per_arm=32000,
        distribution_forwards_per_arm=128000,total_distribution_forwards=256000,retention_action_labels_read=0,
        parity_queries_per_arm=8,total_parity_queries=16,total_parity_encodings=16,total_parity_action_samples=16,parity_physical_controls=0,
        new_teacher_acquisition=0,new_training_encoders=0,evaluation_episodes=48,maximum_evaluation_controls=29520,
        evaluation_settling_controls=480,maximum_evaluation_feature_packets=29520,additional_model_calls_for_packet_preservation=0),
      limitations=['Onlyretentionstatecollectiondiffersbetweennewarms; task and behaviorstatesupport change together, not puretaskID causality.',
        'BothnewarmsdropoldretentionactionNLL; gainsversusv38cannotbeuniquelyattributedtostatecoverage.',
        'Coveragearmremoves64oldAstates; all8oldAtasksmustbecheckedforadditionalharms.',
        'Current16conditionshavenolocalexactexecutionfoundinauditedscope,buttaskhistory/assetmetadataareknown; authorpretraining/externalhistoryunknown.',
        'Stateframesarecorrelated; no sample inflation or novelmethod/RSSreadiness claim.',
        'Allhistoricalv38gains/harmsandcostspreserved; no ordinaryRL control fabricated.'],
      stopping='Twofixed2000steparms, all48 fixedcases, frozenanalysisonce. Onanyfailure preserve files/tensors/optimizer/prefix/cost and stop; no automaticretry.',
      registration_cpu_wall_seconds=time.perf_counter()-began,work_demonstrated=False)
    save(W/'registration.json',p)
    save(W/'analysis_frozen_before_outcomes.json',dict(frozen_utc=now(),registration_sha256=sha(W/'registration.json'),
      analyzer_sha256=sha(CODE/'analyze_state_coverage_v40.py'),new_physical_outcome_files_at_freeze=0))
    print('Registered fixedstatecoverageKLcontrol; no model/training/physics executed.')

def verify():
    p=read(W/'registration.json')
    for path,h in {**p['implementation_sha256'],**p['input_files_sha256']}.items():assert sha(ROOT/path)==h,path
    for item in p['copies']:assert sha(ROOT/item['source'])==sha(ROOT/item['copy'])==item['sha256']
    for item in p['external_proposal_copies']:assert sha(ROOT/item['copy'])==item['sha256']
    for file,key in [('fixed_training_membership.json','fixed_training_membership_sha256'),('fixed_training_sampling.json','fixed_training_sampling_sha256'),
                     ('fixed_training_coverage.json','fixed_training_coverage_sha256'),('fixed_future_conditions.json','fixed_future_conditions_sha256')]:assert sha(W/file)==p[key]
    freeze=read(W/'analysis_frozen_before_outcomes.json');assert freeze['registration_sha256']==sha(W/'registration.json')
    assert freeze['analyzer_sha256']==sha(CODE/'analyze_state_coverage_v40.py')
    return p

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','verify']);args=parser.parse_args()
    register() if args.command=='register' else verify()

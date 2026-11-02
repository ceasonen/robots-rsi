"""CPU-only, create-exclusive aggregation of two real fixed training arms.

Never imports torch/native policy/simulator or executes the scientific analyzer.
The original failed owner remains failed. No true aggregation may run from C.
"""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, math, shutil, sys, time, traceback
import numpy as np

ROOT=Path('F:/robot').resolve()
W=ROOT/'rsi_group_meeting/v40/state_coverage_kl_control'
R=W/'recovery_namespace_guard_20261102'
RCODE=ROOT/'rsi_group_meeting/v40/recovery_code'
REG_SHA='e72b0d1d33729ff4c846aa20541efeddbbd2fc78bd9e2b4696b2c4601004940e'
INITIAL='90fd44d0931e906df4c3b16207e118a5678730c273440446855ca433f5c990c9'
OLD_ACTOR_SHA='953958bb31a0c8b5cde73978426c95ccaa88ec20ac06a40ac867885fc4b33111'
TRAIN_SHA='5b057f08ee8487f5e7f27becdd7d4222733f88d650c0a3a93e61fd5de675f84e'
BODY_SHA='b9bede856a9235b8525783918e15356a4d6be564fc860c93799fc6c977f109f9'
ARMS=('oldA_KL','coverage_KL')
GLOBALS=('trained_actors.json','zero_B_functional_parity_complete.json','training_receipt.json','training_complete.json')

class AuditError(RuntimeError): pass
def require(condition,message):
    if not condition: raise AuditError(message)
def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text('utf8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def finite(value,label,nonnegative=False):
    require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value),label)
    if nonnegative:require(value>=0,label+' nonnegative')
    return value
def same(left,right,label):
    left,right=np.asarray(left),np.asarray(right)
    require(left.shape==right.shape and left.dtype==right.dtype,label+' shape/dtype')
    require(np.array_equal(left,right,equal_nan=True) if left.dtype.kind in 'fc' else np.array_equal(left,right),label+' exact values')
def confined(root,path,label):
    root=Path(root).resolve();path=Path(path)
    require(path.resolve().is_relative_to(root),label+' path confinement')
    require(not path.is_symlink() and not (hasattr(path,'is_junction') and path.is_junction()),label+' no linked path')
    return path.resolve()
def files_under(folder):
    folder=Path(folder).resolve();rows={}
    for p in folder.rglob('*'):
        confined(folder,p,'arm artifact')
        if p.is_file():rows[p.relative_to(folder).as_posix()]=p
    return rows
def expected_arm_files():
    names={'native_metadata.json','initial_identity.json','zero_B_functional_parity_complete.json','training_sampling.json','training_receipt.json','training_complete.json'}
    for i in range(1,2001):
        names.add(f'training_metrics/step{i:04d}.json');names.add(f'training_sampling_steps/step{i:04d}.json')
    for phase in ('plain_author','zero_head'):
        for i in range(4):
            names.add(f'parity_attempts/{phase}/{i:02d}.json')
            for suffix in ('json','npz'):names.add(f'parity_records/{phase}/{i:02d}.{suffix}')
    return names

def validate_sampling(planned,coverage,members):
    require(len(planned)==2000 and len(members['full_teacher'])==3163,'fixed2000/3163 teacher pool')
    require(set(members['retention_by_arm'])==set(ARMS) and all(len(members['retention_by_arm'][a])==128 for a in ARMS),'fixed two128 retention pools')
    rng=np.random.default_rng(20265500);tc=np.zeros(3163,dtype=np.int64);rc=np.zeros(128,dtype=np.int64)
    for i,row in enumerate(planned,1):
        require(row['step']==i,'sampling step ordinal')
        for key,n,c in [('full_teacher',3163,tc),('retention',128,rc)]:
            draw=rng.integers(0,n,size=16).tolist();require(row[key]==draw,'frozen seed20265500 draw');np.add.at(c,draw,1)
    require(tc.sum()==rc.sum()==32000 and int((tc>0).sum())==3162 and tc[1549]==0 and (rc>0).all(),'fixed real draw coverage, no resampling')
    require(coverage['teacher_counts']==tc.tolist() and coverage['retention_index_counts']==rc.tolist(),'saved draw histogram')
    require(coverage['teacher_distinct']==3162 and coverage['retention_distinct']==128,'saved distinct histogram')
    require(all(r['history_length']==10 for a in ARMS for r in members['retention_by_arm'][a]),'retention real10 history metadata')
    return dict(teacher_pool=3163,teacher_distinct_actually_presented_per_arm=3162,teacher_unpresented_indices=[1549],teacher_presentations_per_arm=32000,retention_presentations_per_arm=32000)

def validate_arm(folder,arm,p,planned,sampling_sha,actor_parent,expected_initial=INITIAL):
    """Only saved receipts/NPZ parity and checkpoint bytes; never load a model."""
    folder=Path(folder).resolve();actual=files_under(folder)
    require(expected_arm_files().issubset(actual),'all required arm artifacts '+arm)
    require(len(list((folder/'training_metrics').glob('step*.json')))==len(list((folder/'training_sampling_steps').glob('step*.json')))==2000,'exact2000 metric/draw files '+arm)
    require(len(list((folder/'parity_attempts').rglob('*.json')))==len(list((folder/'parity_records').rglob('*.npz')))==len(list((folder/'parity_records').rglob('*.json')))==8,'exact eight saved parity attempts/records '+arm)
    item=read(folder/'training_receipt.json');done=read(folder/'training_complete.json')
    identity=read(folder/'initial_identity.json');native=read(folder/'native_metadata.json');parity=read(folder/'zero_B_functional_parity_complete.json')
    frozen=p['frozen_pretrained_parameters_sha256']
    require(item['adapter_kind']=='head_input_rank72_alpha144' and item['trainable_parameters']==73728,'head layout '+arm)
    require(item['steps']==done['steps']==2000 and done['arm']==arm,'actual2000 arm closure')
    require(item['teacher_presentations']==item['retention_state_presentations']==32000 and item['retention_action_labels_read']==0,'actual arm presentations')
    require(item['completed_loss_presentations']==done['completed_loss_presentations']==64000 and item['completed_distribution_forwards']==done['completed_distribution_forwards']==128000,'actual arm loss/forward closure')
    require(item['initial_all_parameter_digest']==done['initial_all_parameter_digest']==identity['initial_all_parameter_digest']==parity['initial_all_parameter_digest']==expected_initial,'same initial all-parameter digest')
    require(item['frozen_parameters_sha256']==done['frozen_parameters_sha256']==identity['author_named_parameter_sha256']==native['author_named_parameter_sha256']==parity['author_named_parameter_sha256']==frozen,'same frozen author digest')
    require(identity['zero_B'] is True and identity['A_seed']==77210 and identity['trainable_parameters']==73728 and identity['reference']=='authorpi0','original zeroB layout')
    actor=confined(actor_parent,item['file'],'actual actor')
    require(actor.name==f'{arm}_step2000.pt' and actor.is_file() and sha(actor)==item['sha256'],'actual final actor bytes')
    require(isinstance(item['all_parameter_digest'],str) and len(item['all_parameter_digest'])==64,'recorded trained parameter digest')
    require(item['sampling_sha256']==sampling_sha and read(folder/'training_sampling.json')==planned,'same complete sample plan')
    metrics=item['metrics'];require(len(metrics)==2000,'all2000 saved metrics')
    for i,(metric,draw) in enumerate(zip(metrics,planned),1):
        require(metric==read(folder/'training_metrics'/f'step{i:04d}.json'),'metric file/receipt identity')
        require(draw==read(folder/'training_sampling_steps'/f'step{i:04d}.json'),'step sample file/plan identity')
        require(metric['arm']==arm and metric['step']==i and metric['teacher_presentations']==metric['retention_state_presentations']==16,'step arm/presentation')
        require(metric['teacher_joint_kl_coefficient']==.001 and metric['retention_joint_kl_coefficient']==1. and metric['retention_action_labels_read']==0,'unchanged objective coefficients')
        require(metric['completed_loss_presentations']==i*32 and metric['completed_distribution_forwards']==i*64,'step completed counters')
        require(metric['gradient_dtype']=='float32','gradient dtype')
        for key in ('loss','teacher_nll','teacher_current_to_parent_joint_kl','retention_current_to_parent_joint_kl','gradient_norm_before_clip'):finite(metric[key],'finite '+key)
    finite(item['training_loop_wall_seconds'],'arm loop seconds',True)
    require(done['parity_queries']==8 and parity['arm']==arm and parity['total_queries']==parity['total_encoders']==parity['total_action_samples']==8,'eight real parity calls')
    require(parity['physical_controls']==0 and parity['plain_queries']==parity['head_zero_queries']==4 and parity['all_GMM_action_feature_first_linear_values_and_dtype_exact'] is True,'per-arm zeroB parity closure')
    parity_seconds=0.
    require(len(p['parity'])==4,'four frozen engineering inputs')
    for i,probe in enumerate(p['parity']):
        values=[]
        for phase in ('plain_author','zero_head'):
            f=folder/'parity_records'/phase/f'{i:02d}.npz';info=read(f.with_suffix('.json'));attempt=read(folder/'parity_attempts'/phase/f'{i:02d}.json')
            require(info['arm']==arm and info['phase']==phase and info['index']==i and sha(f)==info['file_sha256'],'actual parity record identity')
            require(info['query_encoders']==info['query_heads']==info['action_samples']==1 and info['physical_controls']==0,'actual query accounting')
            require(attempt['context']==probe['context'] and attempt['physical_controls']==0,'parity attempt input')
            parity_seconds+=finite(info['query_seconds'],'parity seconds',True)
            with np.load(f,allow_pickle=False) as z:values.append({k:z[k].copy() for k in z.files})
        require(set(values[0])==set(values[1]),'parity field sets')
        require(set(values[0])=={'action','log_probability','feature','means','scales','logits','first_linear_output','first_linear_dtype'},'full saved GMM/first-linear parity fields')
        for record in values:
            for key,value in record.items():
                if value.dtype.kind in 'fc':require(np.isfinite(value).all(),'finite engineering parity '+key)
            require(record['feature'].shape==(4,768) and record['feature'].dtype==np.float32,'parity public feature schema')
            require(record['first_linear_output'].dtype==np.float16,'parity native first-linear array dtype')
            require(record['means'].shape[-2:]==record['scales'].shape[-2:]==(5,7) and (record['scales']>0).all(),'finite positive GMM scale layout')
        for key in values[0]:same(values[0][key],values[1][key],'zeroB parity '+key)
        same(values[0]['action'],np.asarray(probe['expected_action'],dtype=np.float32),'plain parent proposal')
        require(abs(float(values[0]['log_probability'])-probe['expected_log_probability'])<=1e-5,'plain parent density')
        require(str(values[0]['first_linear_dtype'].item())=='torch.float16','native output dtype')
    hashes={name:sha(f) for name,f in actual.items()}
    return dict(actor=item,parity=parity,done=done,files=actual,hashes=hashes,parity_query_seconds=parity_seconds)

def check_failure(failure,phase):
    require(failure['arm']=='coverage_KL' and failure['step']==0 and failure['completed_arms']==['oldA_KL'],'recorded namespace failure location')
    for key in ('loss_presentations_attempted','loss_presentations_completed','distribution_forwards_attempted','distribution_forwards_completed','parity_attempted_current_arm','parity_completed_current_arm'):require(failure[key]==0,'second arm was unstarted '+key)
    require('native_model_namespaces' in failure['traceback'] and 'dedicated policy process' in failure['error'],'specific namespace guard failure')
    require(phase['return_code']==1 and abs(phase['external_command_wall_seconds']-3183.9016433)<1e-6,'old failed command timing retained')
    return phase['external_command_wall_seconds']

def validate_projection(proof,addendum,source):
    """Independently hash the original AST chunks; no runner/model execution."""
    tree=ast.parse(source);main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    outer=next(n for n in main.body if isinstance(n,ast.With));tr=next(n for n in outer.body if isinstance(n,ast.Try))
    loop=next(n for n in tr.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='arm')
    chunks={'module_preamble':[n for n in tree.body if n is not main and not isinstance(n,ast.If)],'main_scientific_preamble':main.body[1:main.body.index(outer)],'with_initialization_preamble':outer.body[:outer.body.index(tr)],'try_scientific_preamble_and_helpers':tr.body[:tr.body.index(loop)],'per_arm_body':loop.body,'exception_handlers':tr.handlers,'finally':tr.finalbody}
    require(set(proof['unchanged_chunks'])==set(chunks),'exact unchanged AST chunk set')
    for key,nodes in chunks.items():
        h=hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest();row=proof['unchanged_chunks'][key]
        require(row['AST_exact'] is True and row['original_sha256']==row['recovery_sha256']==h,'actual frozen AST chunk '+key)
    require(proof['per_arm_body_AST_sha256']==addendum['per_arm_body_AST_sha256']==BODY_SHA,'same frozen per-arm AST')
    require(proof['frozen_training_source_sha256']==addendum['frozen_training_source_sha256']==TRAIN_SHA,'same frozen training source')
    require(proof['original_cross_arm_tensor_equal_branch_executed'] is False and proof['declared_main_arm_schedule']==['coverage_KL'] and proof['original_global_ARMS_for_preload']==list(ARMS),'truthful one-arm execution scope')
    require(proof['new_initial_digest_assignment']=='initial_digest=_RECOVERY_EXPECTED_INITIAL_DIGEST','original initial identity assert invoked')

def verify_addendum(root,w,r,codefile):
    root,w,r=Path(root).resolve(),Path(w).resolve(),Path(r).resolve()
    a=read(r/'recovery_addendum_frozen_before_remaining_arm.json')
    require(a['registration_sha256']==REG_SHA==sha(w/'registration.json'),'original registration identity')
    require(a['scientific_changes']==[] and a['remaining_arm']=='coverage_KL','no scientific changes in recovery addendum')
    require(a['frozen_training_source_sha256']==TRAIN_SHA and a['per_arm_body_AST_sha256']==BODY_SHA,'frozen scientific body/source')
    require(a['original_training_failure_sha256']==sha(w/'training_failure.json') and a['original_owner_phase_sha256']==sha(w/'owner_phase_timings/two_fixed_trainings.json'),'frozen original failed attempt')
    require(set(a['recovery_sources_sha256'])=={'run_remaining_coverage_namespace_v40.py','aggregate_recovered_trainings_v40.py','run_namespace_recovery_sequence_v40.py'},'exact three recovery sources')
    code=root/'rsi_group_meeting/v40/recovery_code'
    for name,h in a['recovery_sources_sha256'].items():require(sha(confined(code,code/name,'recovery source'))==h,'frozen recovery source '+name)
    require(sha(codefile)==a['recovery_sources_sha256'][Path(codefile).name],'actual aggregator source identity')
    preserved=a['preserved_original_evidence_sha256'];require(isinstance(preserved,dict),'preserved original hash map')
    for path,h in preserved.items():require(sha(confined(root,root/path,'preserved original evidence'))==h,'preserved original evidence '+path)
    p=read(w/'registration.json')
    for path,h in {**p['implementation_sha256'],**p['input_files_sha256']}.items():require(sha(confined(root,root/path,'frozen original source/input'))==h,'frozen source/input '+path)
    require(sha(root/'rsi_group_meeting/v40/code/train_state_coverage_v40.py')==TRAIN_SHA,'frozen trainer source')
    freeze=read(w/'analysis_frozen_before_outcomes.json');require(freeze['registration_sha256']==REG_SHA and freeze['new_physical_outcome_files_at_freeze']==0,'original analysis freeze')
    require(freeze['analyzer_sha256']==sha(root/'rsi_group_meeting/v40/code/analyze_state_coverage_v40.py'),'unchanged frozen analyzer')
    for name,key in [('fixed_training_membership.json','fixed_training_membership_sha256'),('fixed_training_sampling.json','fixed_training_sampling_sha256'),('fixed_training_coverage.json','fixed_training_coverage_sha256'),('fixed_future_conditions.json','fixed_future_conditions_sha256')]:require(sha(w/name)==p[key],'fixed input '+name)
    require(len(p['cases'])==48 and len(p['conditions'])==16 and p['evaluation_policy_seed']==20265510,'unchanged future evaluation')
    return a,p

def validate_all(root,w,r,codefile):
    root,w,r=Path(root).resolve(),Path(w).resolve(),Path(r).resolve();t=r/'remaining_arm'
    a,p=verify_addendum(root,w,r,codefile);preserved=a['preserved_original_evidence_sha256']
    for name in ('training_failure.json','owner_phase_timings/two_fixed_trainings.json','owner_status.json'):
        require((w/name).relative_to(root).as_posix() in preserved,'old failure/owner evidence explicitly frozen '+name)
    require(not (w/'owner_complete.json').exists(),'original owner remains failed')
    require(not (w/'arms/coverage_KL').exists() and all(not (w/n).exists() for n in GLOBALS),'no canonical coverage/global overwrite')
    require(not (w/'evaluation_started.json').exists() and not (w/'analysis_attempt.json').exists() and not (w/'analysis').exists(),'no early evaluation/analysis')
    require(not list((w/'evaluation/results').glob('*')),'no target outcomes before truthful aggregation')
    failure=read(w/'training_failure.json');phase=read(w/'owner_phase_timings/two_fixed_trainings.json');old_seconds=check_failure(failure,phase)
    require(read(w/'owner_status.json')['stage']=='stopped_requires_audit_no_automatic_retry','old owner failure status retained')
    require(not (t/'training_failure.json').exists(),'remaining arm has no unresolved failure')
    require(not (r/'remaining_process_failure_requires_audit.json').exists(),'remaining process has no unresolved infrastructure failure')
    returned=read(r/'remaining_process_command_returned.json')
    require(returned['honest_remaining_arm_training_runs']==1 and returned['original_global_training_closed'] is False and returned['independent_aggregation_required'] is True,'one remaining process closed honestly')
    remaining_phase_file=r/'owner_phase_timings/remaining_coverage_training.json';remaining_phase=read(remaining_phase_file)
    require(remaining_phase['return_code']==0,'remaining training command returned zero')
    finite(remaining_phase['external_command_wall_seconds'],'remaining external command wall',True)
    planned=read(w/'fixed_training_sampling.json');coverage=read(w/'fixed_training_coverage.json');members=read(w/'fixed_training_membership.json')
    sampling=validate_sampling(planned,coverage,members)
    # The remaining runner keeps byte-identical fixed inputs, never new membership.
    for name in ('registration.json','fixed_training_membership.json','fixed_training_sampling.json','fixed_training_coverage.json'):
        require(sha(t/name)==sha(w/name),'remaining frozen input copy '+name)
    old=validate_arm(w/'arms/oldA_KL','oldA_KL',p,planned,p['fixed_training_sampling_sha256'],w/'actors')
    require(old['actor']['sha256']==OLD_ACTOR_SHA,'first arm checkpoint immutable')
    old_keys={f.relative_to(root).as_posix() for f in old['files'].values()}
    preserved_arm={k for k in preserved if (root/k).resolve().is_relative_to(w/'arms/oldA_KL')}
    require(old_keys==preserved_arm,'all actual original arm files frozen, no missing/extra')
    require(Path(old['actor']['file']).relative_to(root).as_posix() in preserved,'first actor explicitly frozen')
    new=validate_arm(t/'arms/coverage_KL','coverage_KL',p,planned,p['fixed_training_sampling_sha256'],t/'actors')
    end=read(t/'one_remaining_arm_complete.json');one=read(t/'one_remaining_arm_receipt.json')
    addendum_sha=sha(r/'recovery_addendum_frozen_before_remaining_arm.json')
    for record in (one,end):
        require(record['recovery_addendum_sha256']==addendum_sha,'remaining execution frozen addendum')
        require(record['original_first_arm_retrained'] is False and record['original_first_arm_queries_or_forwards_repeated']==0,'no duplicate original arm work')
        require(record['original_cross_process_tensor_equal_executed'] is False and record['cross_process_initial_full_parameter_digest_checked_by_unchanged_original_else_assert'] is True,'truthful cross-process identity checks')
        require(record['original_global_completion_markers_written'] is False,'remaining runner did not fabricate global completion')
    require(end['registration_sha256']==REG_SHA and end['arm']=='coverage_KL','remaining arm closure registration')
    for k,v in {'training_runs':1,'optimizer_steps':2000,'completed_loss_presentations':64000,'completed_distribution_forwards':128000,'parity_queries':8}.items():require(end[k]==v,'remaining-only complete '+k)
    require(end['initial_all_parameter_digest']==INITIAL and end['actor_file_sha256']==new['actor']['sha256'],'remaining closure actor identity')
    require(one['actor']==new['actor'] and one['sampling_sha256']==p['fixed_training_sampling_sha256'],'remaining-only real actor receipt')
    require(one['per_arm_body_AST_sha256']==BODY_SHA,'unchanged scientific per-arm body proof')
    require(one['registration_sha256']==REG_SHA and one['arm']=='coverage_KL','remaining-only receipt registration')
    require(one['initial_all_parameter_digest']==INITIAL,'remaining-only initial digest')
    finite(one['within_attempt_elapsed_seconds'],'remaining one-attempt interval',True)
    proof=read(t/'executed_AST_projection.json')
    validate_projection(proof,a,(root/'rsi_group_meeting/v40/code/train_state_coverage_v40.py').read_text('utf8'))
    # Re-read protected hashes immediately before handing off a mutation plan.
    for name,h in old['hashes'].items():require(sha(old['files'][name])==h,'old arm unchanged during audit')
    recovery_files={str(f):sha(f) for f in (t/'one_remaining_arm_receipt.json',t/'one_remaining_arm_complete.json',t/'executed_AST_projection.json',Path(new['actor']['file']),r/'remaining_process_command_returned.json',remaining_phase_file)}
    return dict(root=root,w=w,r=r,t=t,addendum=a,p=p,old=old,new=new,one=one,end=end,sampling=sampling,old_failed_external_seconds=old_seconds,remaining_external_seconds=remaining_phase['external_command_wall_seconds'],remaining_phase_file=remaining_phase_file,original_failure_sha256=sha(w/'training_failure.json'),original_phase_sha256=sha(w/'owner_phase_timings/two_fixed_trainings.json'),addendum_sha256=addendum_sha,recovery_files_sha256=recovery_files)

def copy_arm_create_exclusive(source,destination,expected_hashes):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    require(not destination.exists(),'canonical remaining-arm destination must not exist')
    require(set(files_under(source))==set(expected_hashes),'entire source inventory before copy')
    for name,h in expected_hashes.items():require(sha(confined(source,source/name,'copy source'))==h,'all source hashes before any copy')
    destination.parent.mkdir(parents=True,exist_ok=True);destination.mkdir(exist_ok=False);copied=[]
    for name,h in sorted(expected_hashes.items()):
        src=confined(source,source/name,'copy source');dst=confined(destination,destination/name,'copy destination')
        require(sha(src)==h,'source stayed immutable before copying');dst.parent.mkdir(parents=True,exist_ok=True)
        with src.open('rb') as f,dst.open('xb') as out:shutil.copyfileobj(f,out)
        require(sha(dst)==h,'copied bytes identical');copied.append(dict(source=str(src),canonical=str(dst),sha256=h))
    require(set(files_under(destination))==set(expected_hashes),'copied inventory complete')
    return copied

def build_outputs(bundle,copy_manifest_file,copy_manifest_sha):
    """Pure construction from validated real receipts. No invented single-attempt time."""
    actors={ARMS[0]:bundle['old']['actor'],ARMS[1]:bundle['new']['actor']}
    require(set(actors)==set(ARMS),'exact two actual actors')
    totals=dict(training_runs=len(actors),total_optimizer_steps=sum(v['steps'] for v in actors.values()),total_distribution_forwards=sum(v['completed_distribution_forwards'] for v in actors.values()),total_loss_presentations=sum(v['completed_loss_presentations'] for v in actors.values()),total_teacher_presentations=sum(v['teacher_presentations'] for v in actors.values()),total_retention_state_presentations=sum(v['retention_state_presentations'] for v in actors.values()),parity_queries=sum(bundle[k]['parity']['total_queries'] for k in ('old','new')))
    require(totals==dict(training_runs=2,total_optimizer_steps=4000,total_distribution_forwards=256000,total_loss_presentations=128000,total_teacher_presentations=64000,total_retention_state_presentations=64000,parity_queries=16),'sum of real per-arm completed work')
    parity=dict(completed_utc=now(),arms={ARMS[0]:bundle['old']['parity'],ARMS[1]:bundle['new']['parity']},total_queries=16,total_encoders=16,total_action_samples=16,physical_controls=0,initial_all_parameter_digest=INITIAL,both_initial_layout_parameter_digests_equal=True,
        cross_arm_identity_evidence='Same fixed named layout/shape/dtype and all-parameter byte digests, plus each arm actual4plain+4zeroB functional parity. Original same-process cross-arm torch.equal was not executed across these two attempts.')
    provenance=dict(derived_after_registered_infrastructure_recovery=True,registration_sha256=REG_SHA,recovery_addendum_sha256=bundle['addendum_sha256'],original_training_failure_sha256=bundle['original_failure_sha256'],original_failed_phase_sha256=bundle['original_phase_sha256'],original_failed_training_command_return_code=1,original_failed_training_command_external_seconds=bundle['old_failed_external_seconds'],original_failed_command_contains_completed_oldA=True,
        recovery_training_command_external_seconds=bundle['remaining_external_seconds'],recovery_training_phase_file=str(bundle['remaining_phase_file']),recovery_training_phase_sha256=sha(bundle['remaining_phase_file']),
        recovery_one_remaining_receipt=str(bundle['t']/'one_remaining_arm_receipt.json'),recovery_one_remaining_complete=str(bundle['t']/'one_remaining_arm_complete.json'),recovery_one_remaining_receipt_sha256=sha(bundle['t']/'one_remaining_arm_receipt.json'),recovery_one_remaining_complete_sha256=sha(bundle['t']/'one_remaining_arm_complete.json'),copy_manifest_file=str(copy_manifest_file),copy_manifest_sha256=copy_manifest_sha,
        optimization_runs_completed=2,original_owner_returned_zero=False,original_owner_successful_completion=None,original_owner_whole_wall_seconds=None,whole_project_wall_seconds=None,
        timing='Two disjoint command attempts. The original failed3183.9016433s contains completed oldA loop/startup/parity/failed next initialization; never add its nested arm-loop time. Recovery command has its own external timing. No single successful original command/owner interval is fabricated.')
    receipt=dict(completed_utc=now(),arms=actors,initial_all_parameter_digest=INITIAL,sampling_sha256=bundle['p']['fixed_training_sampling_sha256'],optimizer_steps_per_arm=2000,distribution_forwards_per_arm=128000,retention_action_labels_read=0,within_attempt_elapsed_seconds=None,precise_whole_command_wall_seconds=None,**totals,**provenance)
    complete=dict(completed_utc=now(),optimizer_steps_per_arm=2000,retention_action_labels_read=0,initial_all_parameter_digest=INITIAL,within_attempt_elapsed_seconds=None,precise_end_to_end_command_wall_seconds=None,**totals,**provenance)
    return {'trained_actors.json':actors,'zero_B_functional_parity_complete.json':parity,'training_receipt.json':receipt,'training_complete.json':complete}

def publish_globals_create_exclusive(w,outputs):
    w=Path(w);require(set(outputs)==set(GLOBALS),'four truthful global outputs only')
    require(all(not (w/name).exists() for name in GLOBALS),'all global destinations absent before publication')
    # Complete is intentionally last, after all supporting receipts.
    for name in GLOBALS:save(w/name,outputs[name])

def main():
    require(Path(__file__).resolve()==RCODE/'aggregate_recovered_trainings_v40.py','Preparation only; root must independently review/install/freeze.')
    sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
    from local_run_lock_v4 import exclusive
    began=time.perf_counter()
    with exclusive('state_coverage_v40'):
        require(not (R/'aggregation_attempt.json').exists() and not (R/'aggregation_failure_requires_audit.json').exists(),'one aggregation attempt; never blind retry')
        save(R/'aggregation_attempt.json',dict(started_utc=now(),actual_python_pid=__import__('os').getpid(),implementation_sha256=sha(__file__),registration_sha256=sha(W/'registration.json'),automatic_retry=False,scope='CPU validation/copy/derived aggregation only; no model loading or scientific analysis.'))
        try:
            b=validate_all(ROOT,W,R,Path(__file__))
            copied=copy_arm_create_exclusive(b['t']/'arms/coverage_KL',W/'arms/coverage_KL',b['new']['hashes'])
            copyfile=R/'coverage_canonical_copy_manifest.json';save(copyfile,dict(created_utc=now(),registration_sha256=REG_SHA,addendum_sha256=b['addendum_sha256'],files=copied,raw_recovery_originals_retained=True,actor_file_field_unchanged=True))
            # No original arm/failure/owner evidence may change during copying.
            verify_addendum(ROOT,W,R,Path(__file__))
            for file,h in b['recovery_files_sha256'].items():require(sha(file)==h,'remaining closure/checkpoint stayed immutable')
            outputs=build_outputs(b,copyfile,sha(copyfile));publish_globals_create_exclusive(W,outputs)
            verify_addendum(ROOT,W,R,Path(__file__))
            for file,h in b['recovery_files_sha256'].items():require(sha(file)==h,'remaining evidence unchanged after publication')
            save(R/'aggregation_complete.json',dict(completed_utc=now(),registration_sha256=REG_SHA,addendum_sha256=b['addendum_sha256'],global_outputs_sha256={name:sha(W/name) for name in GLOBALS},canonical_copy_manifest_sha256=sha(copyfile),actual_training_runs=2,optimizer_steps=4000,total_distribution_forwards=256000,total_loss_presentations=128000,total_parity_queries=16,original_owner_remains_failed=True,within_aggregation_attempt_seconds=time.perf_counter()-began,new_model_gradient_encoder_or_simulator_calls=0))
            print('Truthful two-arm aggregation complete; original owner remains failed; no policy outcomes analyzed.')
        except BaseException as error:
            save(R/'aggregation_failure_requires_audit.json',dict(failed_utc=now(),error=repr(error),traceback=traceback.format_exc(),within_attempt_elapsed_seconds=time.perf_counter()-began,automatic_retry=False,partial_copy_and_outputs_must_be_preserved=True,new_model_gradient_encoder_or_simulator_calls=0))
            raise
if __name__=='__main__':main()

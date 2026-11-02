"""CPU registration and immutable data bindings for one standard BC baseline."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,copy,shutil,time
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'rsi_group_meeting/v36/head_uniform_parameterization'
OLD=ROOT/'rsi_group_meeting/v32/repair_learning_dev';CROSS=ROOT/'rsi_group_meeting/v35/cross_context_diagnostic'
D0=ROOT/'rsi_group_meeting/v30/language_schedule_dev';FIT=ROOT/'rsi_group_meeting/v30/research_agent/offline_teacher_fit_diagnostic'
SOURCES=['head_bc_runtime_v36.py','head_bc_study_v36.py','train_head_bc_v36.py','head_bc_worker_v36.py','evaluate_head_bc_v36.py','analyze_head_bc_v36.py','score_head_bc_offline_v36.py']
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as stream:json.dump(x,stream,ensure_ascii=False,indent=2)
def status(stage,**items):
    file=W/'status.working.json';file.write_text(json.dumps(dict(updated_utc=now(),stage=stage,work_demonstrated=False,**items),ensure_ascii=False,indent=2),'utf8');file.replace(W/'status.json')
def ready():
    c=read(CROSS/'analysis/cross_context_readout.json');r=read(ROOT/'rsi_group_meeting/v35/root_completed_review/review.json')
    assert len(c['rows'])==12 and r['frozen_analysis_sha256']==sha(CROSS/'analysis/cross_context_readout.json')
    assert read(CROSS/'collection_complete.json')['physical_episodes']==12
    f=read(FIT/'execution/summary.json');assert f['density_head_forwards_completed']==9873
    return read(OLD/'registration.json'),read(CROSS/'registration.json')
def register():
    began=time.perf_counter();old,cross=ready();assert not W.exists(),'Preserve any earlier preparation/attempt.'
    members=read(OLD/'training_membership.json');fit=read(FIT/'protocol.json')
    assert members['N']==128 and len(members['selected_targets']['uniform_bc'])==len(members['selected_targets']['retention'])==128
    sources={};inputs={};copies=[]
    def bind(path,expected=None,source=False):
        path=Path(path).resolve();value=sha(path)
        if expected is not None:assert value==expected,str(path)
        (sources if source else inputs)[rel(path)]=value
    for name in SOURCES:bind(ROOT/'rsi_group_meeting'/name,source=True)
    for plan in [old,cross]:
        for name,value in plan['implementation_sha256'].items():bind(ROOT/name,value,True)
    for name,value in fit['sources_sha256'].items():
        if name.startswith('external/DMPEL/') or name.endswith('asset_receipt.json') or name=='data/models/dmpel_author_source90/config.json':bind(ROOT/name,value,True)
    for file in [OLD/'registration.json',OLD/'training_membership.json',OLD/'trained_actors.json',OLD/'analysis/repair_readout.json',CROSS/'registration.json',CROSS/'analysis/cross_context_readout.json',ROOT/'rsi_group_meeting/v35/root_completed_review/review.json',ROOT/'rsi_group_meeting/v35/root_completed_review/cost_increment.json',FIT/'protocol.json',FIT/'diagnostic_membership.json',FIT/'execution/summary.json',FIT/'execution/per_action_fit.jsonl']:
        bind(file)
    bind(old['parent']['file'],old['parent']['sha256'])
    for row in members['selected_targets']['uniform_bc']+members['selected_targets']['retention']:bind(ROOT/row['file'],members['files_sha256'][row['file']])
    # Offline scoring uses every old diagnostic row, including targets not sampled for training.
    for row in read(FIT/'diagnostic_membership.json'):bind(ROOT/row['file'],fit['sources_sha256'][row['file']])
    def reference(folder,key):
        file=folder/'results'/f'{key}.json';r=read(file);bind(file)
        for path,h in [(file.with_suffix('.npz'),r['trace_sha256']),(folder/r['events'],r['events_sha256']),(folder/r['video'],r['video_sha256']),(folder/r['initial_sensors'],r['initial_sensors_sha256'])]:bind(path,h)
        return dict(result=rel(file),initial_sensors=rel(folder/r['initial_sensors']))
    cases=[];parity=[]
    for previous in cross['reused_parent_train']:
        c=copy.deepcopy(previous['context']);c['partition']='training_context';refs={'parent':dict(result=previous['result'],initial_sensors=previous['initial_sensors'])}
        r=read(ROOT/previous['result']);reference((ROOT/previous['result']).parent.parent,(ROOT/previous['result']).stem)
        for actor in ['uniform_bc','local_bc']:refs[actor]=reference(CROSS/'student',c['context_id']+'__'+actor)
        first=json.loads(((ROOT/previous['result']).parent.parent/r['events']).read_text('utf8').splitlines()[0])
        parity.append(dict(context=c,source_initial=previous['initial_sensors'],expected_action=first['action'],expected_log_probability=first['log_probability'],legacy_logp_abs_tolerance=1e-5))
        cases.append(dict(episode=dict(key=c['context_id']+'__head_uniform_bc',context=c,actor='head_uniform_bc'),references=refs))
    for original in old['evaluation_contexts']:
        c=copy.deepcopy(original);c['policy_seed']=old['evaluation_policy_seed'];c['partition']='test_context' if original['partition']=='Long_development' else 'source_regression'
        refs={actor:reference(OLD/'evaluation',c['context_id']+'__'+actor) for actor in ['parent','uniform_bc','local_bc']}
        cases.append(dict(episode=dict(key=c['context_id']+'__head_uniform_bc',context=c,actor='head_uniform_bc'),references=refs))
    assert len(cases)==12 and len(parity)==4
    for case in cases:
        s=case['episode']['context']['setting'];bind(ROOT/s['bddl'],s['bddl_sha256']);bind(ROOT/s['init_file'],s['init_sha256'])
    bind(ROOT/'rsi_group_meeting/v4/xvla/protocol.json')
    W.mkdir(parents=True)
    def cp(src,dst):
        bind(src);dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copy2(src,dst);assert sha(src)==sha(dst)
        copies.append(dict(source=rel(src),copy=rel(dst),sha256=sha(src)))
    for i,item in enumerate(parity):
        dst=W/'parity_inputs'/f'{i:02d}.npz';cp(ROOT/item['source_initial'],dst);item['observation_file']=str(dst)
    for folder in [D0/'canonical',OLD/'evaluation/canonical']:
        for src in sorted(folder.iterdir()):cp(src,W/'evaluation/canonical'/src.name)
    save(W/'fixed_training_membership.json',dict(selected_targets={'uniform_bc':members['selected_targets']['uniform_bc'],'retention':members['selected_targets']['retention']},N=128,original_membership_sha256=sha(OLD/'training_membership.json')))
    save(W/'fixed_offline_membership.json',read(FIT/'diagnostic_membership.json'))
    p=dict(registered_utc=now(),scope='One standard head-uniform BC parameterization control; no improver/M/R1/R2/RSI claim. All evaluation contexts are exposed development.',
           parent=old['parent'],frozen_pretrained_parameters_sha256=old['frozen_pretrained_parameters_sha256'],public_sensors=old['public_sensors'],
           evaluation_primary='Officialcompleteoriginaltasksuccessby520.',evaluation_extension='By900onthesametrajectoryisanextension/sensitivity, neveranewsampleset; late-onlysuccessisnotaby520repair.',
           implementation_sha256=sources,input_files_sha256=inputs,copies=copies,fixed_training_membership_sha256=sha(W/'fixed_training_membership.json'),fixed_offline_membership_sha256=sha(W/'fixed_offline_membership.json'),
           method=dict(module='policy_head.share.0',input_size=768,output_size=256,rank=72,alpha=144,scale=2.,trainable_parameters=73728,A_initialization='kaiming_uniform_a=sqrt5,torchseed77210',B_initialization='ExactF32zero',merge='CastscaledLoRAresidualtobaseLinearoutputdtypebeforeaddition; nativeF16headactivationdtypeispreserved, LoRAparameters/gradientsF32.',
                       source_initialization='Sameoriginalauthorcheckpointonly. OldtemporaladapterptisreadonCPUsolelytoproveallsixBzero; neverloadedintoheadlayout. Sameauthornamedparametershashandfunctionalparity, notcross-layoutallparameterhash.'),
           training=dict(optimizer='AdamW',learning_rate=1e-4,weight_decay=0,betas=[.9,.999],eps=1e-8,gradient_clip=1.,joint_KL_coefficient=.001,steps=20,teacher_per_step=16,retention_per_step=16,rng_seed=20265450,teacher_presentations=320,retention_presentations=320,checkpoint='Onlyfinalstep20,nointermediatecandidatecheckpoints',data='Exactlyoriginaluniform128+retention128, sameorderanddraws; no newv35teacherdata.'),
           parity=parity,parity_rule='SameNativeinstancebeforeandafterheadinstallation:4rawfirstobservations×2. Originalnativeplainpolicyprobesmustmatcholdzero-temporalπ0actionsbitwiseandlegacylogp<=1e-5. Newzero-BmustmatcheachplainfirstLinearoutputdtype/values,allGMMmeans/scales/logits,spatialfeature,action/logpexactly. Savebothsidesbeforeassert; no thresholdrelaxation.',
           cases=cases,budget=dict(new_training_runs=1,new_optimizer_steps=20,loss_presentations=640,training_distribution_forwards=1280,parity_queries=8,parity_encodings=8,parity_action_samples=8,parity_physical_controls=0,parity_numpy_records=8,new_teacher_episodes=0,evaluation_episodes=12,maximum_evaluation_controls=10800,evaluation_settling_controls=120,maximum_evaluation_feature_packets=10800,additional_model_calls_for_packet_preservation=0,offline_new_head_density_forwards=3291,offline_old_actor_forwards=0,offline_encoders_or_samples_or_controls=0),
           old_fit_per_action_file=rel(FIT/'execution/per_action_fit.jsonl'),old_fit_summary_file=rel(FIT/'execution/summary.json'),
           old_cross_analysis_file=rel(CROSS/'analysis/cross_context_readout.json'),old_cross_review_file=rel(ROOT/'rsi_group_meeting/v35/root_completed_review/review.json'),
           original_student_analysis_file=rel(OLD/'analysis/repair_readout.json'),
           limitations=['Location,rank,initialactiveBgradientcountandfunctiongeometrychange together. EqualparametercountandpresentationsarenotequalFLOPs orfunctionalstep. alpha144onlymatchesα/r2, notoutputdrift.',
                        'Firstsharedheadlayerchangesmeans,sigmasandcategoricallogits. low_eval_noiseFalsefixestherule, notlearnedsigmavalues.',
                        'Anegative20stepresultdoesnotrejectallBC; apos itiveresultisastandardbaselinegain, notinnovation orRSI.',
                        'Trainingcontextpositivesnevergeneralization/work. Priororiginaltest0/4,localsourceharmandteacher3/4withA-stagefailureand623-step latesuccessremainvisible.',
                        'No ordinaryRL update/controlisfabricatedfromprivilegedteacheractions.'],
           stopping='Finite20steps+all12evaluation+one3291offlinefit. Numericalorinfra errorspreserveattempt/metrics/prefixandstoptoreview; noautomaticretry/checkpointselection/LR/alpha/rank/epochsearch.',
           registration_cpu_wall_seconds=time.perf_counter()-began,work_demonstrated=False)
    save(W/'registration.json',p)
    save(W/'analysis_frozen_before_head_outcomes.json',dict(frozen_utc=now(),registration_sha256=sha(W/'registration.json'),analyzer_sha256=sha(ROOT/'rsi_group_meeting/analyze_head_bc_v36.py'),offline_scorer_sha256=sha(ROOT/'rsi_group_meeting/score_head_bc_offline_v36.py'),new_outcome_files_at_freeze=0))
    print('Registeredonehead-uniformbaseline;notrainingormodelstarted.')
def verify():
    p=read(W/'registration.json')
    for name,value in {**p['implementation_sha256'],**p['input_files_sha256']}.items():assert sha(ROOT/name)==value,name
    for item in p['copies']:assert sha(ROOT/item['source'])==sha(ROOT/item['copy'])==item['sha256']
    assert sha(W/'fixed_training_membership.json')==p['fixed_training_membership_sha256'] and sha(W/'fixed_offline_membership.json')==p['fixed_offline_membership_sha256']
    freeze=read(W/'analysis_frozen_before_head_outcomes.json');assert freeze['registration_sha256']==sha(W/'registration.json')
    assert freeze['analyzer_sha256']==sha(ROOT/'rsi_group_meeting/analyze_head_bc_v36.py') and freeze['offline_scorer_sha256']==sha(ROOT/'rsi_group_meeting/score_head_bc_offline_v36.py')
    return p
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register']);parser.parse_args();register()

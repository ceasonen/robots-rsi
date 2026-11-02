"""One descriptive audit after all48 fixed cases. No model/simulator imports.
Pure validate_episode_payload/summarize accept CPU fixtures; main refuses preparation paths.
"""
from pathlib import Path
from collections import defaultdict
import csv,json,math,sys,time,traceback
import numpy as np

ACTORS=('parent','oldA_KL','coverage_KL')
PARTITIONS={'old_A':8,'source_coverage':4,'Long':4}

class AuditError(RuntimeError):pass

def require(condition,message):
    if not condition:raise AuditError(message)

def boolean(value,label):
    require(isinstance(value,(bool,np.bool_)),label+' must be a bool')
    return bool(value)

def same(a,b,label):
    aa,bb=np.asarray(a),np.asarray(b)
    require(aa.dtype==bb.dtype and aa.shape==bb.shape,label+' dtype/shape')
    equal=np.array_equal(aa,bb,equal_nan=True) if aa.dtype.kind in 'fc' else np.array_equal(aa,bb)
    require(equal,label+' values')

def finite_number(value,label):
    require(isinstance(value,(int,float)) and math.isfinite(value) and value>=0,label+' nonnegative finite')
    return float(value)

def validate_episode_payload(e,r,a,initial,events,queries,public_sensors,array_hash):
    """Validate one completed trial, including policy failure before its first action.
    queries are the paired attempt arguments/completed answers, in query order.
    Infrastructure exceptions have no completed trial and cannot be supplied here.
    """
    c=e['context'];cap=c['max_task_controls'];part=c['partition'];actor=e['actor']
    require(actor in ACTORS and part in PARTITIONS,'actor/partition')
    require(cap==(900 if part=='Long' else 520),'fixed horizon')
    require(c['policy_seed']==20265510,'execution seed')
    require(r['episode']==e,'result episode identity')
    require(r['maximum_task_controls']==cap and r['physical_episodes']==1,'trial scope')
    require(r['privileged_controller'] is False and r['original_joint_instruction'] is True,'public joint-instruction deployment')
    numeric=boolean(r['policy_numerical_failure'],'policy failure')
    physical=boolean(r['physics_instability_terminated'],'physics instability')
    require(boolean(r['instability_terminated'],'retained instability')==(numeric or physical),'instability union')
    actions=a['actions'];n=len(actions)
    require(actions.shape==(n,7) and actions.dtype==np.float32 and np.isfinite(actions).all(),'executed actions')
    require(0<=n<=cap and r['executed_physics_steps']==r['attempted_physics_steps']==n,'completed physical controls')
    require(r['settling_controls']==10,'reset settling')
    require(r['policy_queries']==len(queries)==n+int(numeric)<=cap,'query/control distinction')
    require(len(events)==len(queries),'event/query coverage')
    require([v['type'] for v in events]==['action']*n+(['query_failure'] if numeric else []),'terminal failure event order')
    goals=a['diagnostic_subgoals'];warnings=a['native_warning_counts'];states=a['privileged_sim_states_for_replay_only']
    require(goals.ndim==2 and goals.shape[0]==n and goals.shape[1]>0 and goals.dtype==np.bool_,'goals shape')
    require(warnings.ndim==2 and warnings.shape[0]==n and np.issubdtype(warnings.dtype,np.integer),'warning shape')
    require(states.ndim==2 and states.shape[0]==n,'full-state shape')
    require(a['robot_states'].ndim==2 and len(a['robot_states'])==n,'robot-state shape')
    require(a['official_success'].shape==(n,) and a['official_success'].dtype==np.bool_,'official-success array')
    same(a['initial_sim_state'],initial['initial_sim_state'],'initial simstate')
    same(a['initial_goal_predicates'],initial['initial_goal_predicates'],'initial goals')
    require(r['initial_all_goals_true']==bool(a['initial_goal_predicates'].all()),'initial goal disclosure')
    for key in public_sensors:require(key in initial,'missing public initial sensor '+key)
    samples=0;seconds=0.;packet_seconds=0.
    for i,(event,pair) in enumerate(zip(events,queries)):
        args,q=pair['arguments'],pair['answer']
        require(args['query_case']==e['key'] and args['query_step']==i and args['query_phase']=='case','query index')
        require(event['step']==i+1 and event['query_step']==i,'event index')
        require(q['actor']==event['actor']==actor,'actor alignment')
        require(args['language']==event['language']==c['setting']['language'],'language alignment')
        require(q['history_length']==event['history_length']==min(i+1,10),'real past-only history')
        require(q['decode_mode']=='learned','unchanged sampling rule')
        require(q['public_pre_action_sensor_sha256']==event['pre_action_sensor_sha256'],'public input hash')
        if i==0:require(event['pre_action_sensor_sha256']==array_hash({k:initial[k] for k in public_sensors}),'first public input')
        require(q['query_seconds']==event['query_seconds'],'query timing')
        seconds+=finite_number(q['query_seconds'],'query seconds');packet_seconds+=finite_number(q['feature_packet_io_seconds'],'packet seconds')
        samples+=int(boolean(q['action_sample_completed'],'sample completion'))
        qfail=boolean(q['policy_numerical_failure'],'query numerical failure')
        if i<n:
            require(not qfail and q['action_sample_completed'] is True,'executed query must be finite sampled proposal')
            qa=np.asarray(q['action'],dtype=np.float32);ea=np.asarray(event['action'],dtype=np.float32)
            require(qa.shape==(7,) and np.isfinite(qa).all(),'finite proposal')
            same(qa,actions[i],'query/trace action');same(ea,actions[i],'event/trace action')
            require(math.isfinite(q['log_probability']) and q['log_probability']==event['log_probability'],'sample density')
            require(event['policy_numerical_failure'] is False,'action event numerical status')
            require(event['goal_predicates']==goals[i].tolist() and event['warning_counts']==warnings[i].tolist(),'predicate/warning alignment')
            require(event['post_state_sha256']==array_hash({'sim':states[i]}),'post-action state hash')
            truth=bool(goals[i].all()) and not boolean(event['instability'],'action instability')
            require(boolean(event['official_success'],'event success')==bool(a['official_success'][i])==truth,'full-task success predicate')
        else:
            require(numeric and i==n and qfail,'only one final failed query')
            require(event['physical_action_executed'] is False and event['official_success'] is False,'no fabricated failed action/success')
            require(event['policy_numerical_failure'] is True and isinstance(q['numerical_failure_reason'],str) and q['numerical_failure_reason'],'failure reason')
            require(event['numerical_failure_reason']==q['numerical_failure_reason'],'failure-reason equality')
            require(r['terminal_query_failure']==q,'terminal query preservation')
    require(physical==any(bool(v['instability']) for v in events[:n]),'physics flag from action events only')
    require(not numeric or not a['official_success'].any(),'success must stop before a later failed query')
    require(n>0 or numeric,'zero-action completed trial needs retained numerical failure')
    require(samples==r['action_samples_completed']<=len(queries),'sampling distinct from query attempts')
    s520=bool(a['official_success'][:520].any());scap=bool(a['official_success'].any())
    require(boolean(r['success_by520'],'primary success')==s520 and boolean(r['success_by_cap'],'cap success')==scap,'success endpoints')
    if part=='Long':require(boolean(r['success_by900'],'Long extension')==scap,'same-trajectory Long900')
    else:require(r['success_by900'] is None,'source900 is unobserved/null, not zero')
    if scap:require(bool(a['official_success'][-1]) and not a['official_success'][:-1].any(),'stop at first full success')
    require(abs(r['inference_seconds']-seconds)<=max(1e-9,1e-12*seconds),'query timer total')
    finite_number(r['wall_seconds'],'episode wall')
    return dict(key=e['key'],context_id=c['context_id'],partition=part,actor=actor,policy_seed=c['policy_seed'],
        success520=s520,success900=scap if part=='Long' else None,success_by_cap=scap,
        policy_numerical_failure=numeric,physics_instability=physical,instability_retained=numeric or physical,
        task_controls=n,policy_queries=len(queries),action_samples_completed=samples,settling_controls=10,
        inference_seconds=seconds,episode_wall_seconds=r['wall_seconds'],packet_io_seconds_included_in_wall=packet_seconds,
        initial_goals=a['initial_goal_predicates'].tolist(),terminal_goals=(goals[-1] if n else a['initial_goal_predicates']).tolist(),
        initial_all_goals_true=r['initial_all_goals_true'])

def paired_comparison(candidate,baseline):
    require(set(candidate)==set(baseline),'paired comparison identities')
    ids=sorted(candidate);wins=sum(candidate[k] and not baseline[k] for k in ids);harms=sum(not candidate[k] and baseline[k] for k in ids)
    positives=sum(baseline.values());negatives=len(ids)-positives
    return dict(conditions=len(ids),repairs=int(wins),harms=int(harms),success_count_difference=int(wins-harms),
        baseline_success_opportunities=int(positives),baseline_failure_opportunities=int(negatives),
        harm_rate_given_baseline_success=harms/positives if positives else None,
        repair_rate_given_baseline_failure=wins/negatives if negatives else None,
        harm_information='finite observed comparison, not a safety guarantee' if positives else 'UNINFORMATIVE_FOR_HARM: baseline has zero successes; zero harms is not preservation evidence')

def summarize(rows):
    """No task-IID or frame-IID p-values; scopes remain separate."""
    require(len(rows)==48 and len({r['key'] for r in rows})==48,'all48 distinct rows')
    groups=defaultdict(dict)
    for row in rows:
        require(row['actor'] not in groups[row['context_id']],'duplicate actor/context')
        groups[row['context_id']][row['actor']]=row
    require(len(groups)==16 and all(set(g)==set(ACTORS) for g in groups.values()),'three actors per context')
    output={}
    for part,count in PARTITIONS.items():
        ids=sorted(k for k,v in groups.items() if v['parent']['partition']==part)
        require(len(ids)==count,'scope count '+part)
        partout={}
        for name,field in [('by520','success520')]+([('by900_same_trajectories','success900')] if part=='Long' else []):
            values={actor:{k:boolean(groups[k][actor][field],field) for k in ids} for actor in ACTORS}
            partout[name]=dict(actors={actor:dict(successes=sum(v.values()),conditions=count,success_rate=sum(v.values())/count,context_ids=ids,outcomes=[v[k] for k in ids]) for actor,v in values.items()},
                comparisons={'oldA_KL_vs_parent':paired_comparison(values['oldA_KL'],values['parent']),
                    'coverage_KL_vs_parent':paired_comparison(values['coverage_KL'],values['parent']),
                    'coverage_KL_vs_oldA_KL':paired_comparison(values['coverage_KL'],values['oldA_KL'])})
        if part!='Long':partout['by900_same_trajectories']=None
        output[part]=partout
    return output

def audit_training(root,w,p,read,sha):
    """Audit saved training bookkeeping without importing torch or loading weights."""
    complete=read(w/'training_complete.json');receipt=read(w/'training_receipt.json');actors=read(w/'trained_actors.json')
    require(set(actors)==set(ACTORS[1:]),'two trained actors only')
    expected={'training_runs':2,'optimizer_steps_per_arm':2000,'total_optimizer_steps':4000,'total_distribution_forwards':256000,
        'total_loss_presentations':128000,'total_teacher_presentations':64000,'total_retention_state_presentations':64000,'retention_action_labels_read':0,'parity_queries':16}
    for k,v in expected.items():require(complete[k]==v,'training complete '+k)
    require(complete['registration_sha256']==sha(w/'registration.json'),'training registration')
    require(receipt['sampling_sha256']==p['fixed_training_sampling_sha256']==sha(w/'fixed_training_sampling.json'),'common sampling hash')
    require(receipt['arms']==actors,'global actor receipts')
    planned=read(w/'fixed_training_sampling.json');coverage=read(w/'fixed_training_coverage.json');members=read(w/'fixed_training_membership.json')
    require(len(members['full_teacher'])==3163 and all(len(members['retention_by_arm'][a])==128 for a in ACTORS[1:]),'fixed pools, not guaranteed full draw coverage')
    rng=np.random.default_rng(20265500);tc=np.zeros(3163,dtype=np.int64);rc=np.zeros(128,dtype=np.int64)
    require(len(planned)==2000,'fixed2000 sample steps')
    for i,row in enumerate(planned,1):
        require(row['step']==i,'sampling step ordinal')
        for key,size,counts in [('full_teacher',3163,tc),('retention',128,rc)]:
            draw=rng.integers(0,size,size=16).tolist();require(row[key]==draw,'frozen RNG plan');np.add.at(counts,draw,1)
    require(tc.sum()==rc.sum()==32000 and coverage['teacher_counts']==tc.tolist() and coverage['retention_index_counts']==rc.tolist(),'actual presentation histogram')
    require(coverage['teacher_distinct']==int((tc>0).sum()) and coverage['retention_distinct']==int((rc>0).sum()),'actual distinct targets, zero counts permitted')
    # A pool of3163 is not proof that all3163 were sampled with replacement.
    parity=read(w/'zero_B_functional_parity_complete.json')
    require(parity['total_queries']==parity['total_encoders']==parity['total_action_samples']==16 and parity['physical_controls']==0,'sixteen actual engineering queries')
    require(parity['both_initial_layout_parameter_digests_equal'] is True,'cross-arm initial identity')
    initial=receipt['initial_all_parameter_digest'];require(initial==complete['initial_all_parameter_digest']==parity['initial_all_parameter_digest'],'common initial parameter digest')
    parity_seconds=0.
    for arm in ACTORS[1:]:
        item=actors[arm];folder=w/'arms'/arm
        require(sha(item['file'])==item['sha256'] and item['steps']==2000 and item['trainable_parameters']==73728,'final actor '+arm)
        require(item['initial_all_parameter_digest']==initial and item['frozen_parameters_sha256']==p['frozen_pretrained_parameters_sha256'],'initial/frozen digest '+arm)
        require(item['teacher_presentations']==item['retention_state_presentations']==32000 and item['retention_action_labels_read']==0,'arm presentations')
        require(item['completed_loss_presentations']==64000 and item['completed_distribution_forwards']==128000,'arm completed forwards')
        require(item['sampling_sha256']==receipt['sampling_sha256'] and read(folder/'training_sampling.json')==planned,'same index draws each arm')
        require(read(folder/'training_receipt.json')==item,'per-arm receipt')
        arm_complete=read(folder/'training_complete.json');require(arm_complete['steps']==2000 and arm_complete['completed_distribution_forwards']==128000 and arm_complete['parity_queries']==8,'per-arm closure')
        metrics=item['metrics'];require(len(metrics)==2000,'all step metrics')
        require(len(list((folder/'training_metrics').glob('step*.json')))==len(list((folder/'training_sampling_steps').glob('step*.json')))==2000,'all step files')
        for i,(metric,draw) in enumerate(zip(metrics,planned),1):
            require(metric==read(folder/'training_metrics'/f'step{i:04d}.json') and draw==read(folder/'training_sampling_steps'/f'step{i:04d}.json'),'immutable metric/draw receipt')
            require(metric['arm']==arm and metric['step']==i and metric['teacher_presentations']==metric['retention_state_presentations']==16,'metric arm/step counts')
            require(metric['teacher_joint_kl_coefficient']==.001 and metric['retention_joint_kl_coefficient']==1. and metric['retention_action_labels_read']==0,'loss coefficients/labels')
            require(metric['completed_loss_presentations']==i*32 and metric['completed_distribution_forwards']==i*64,'cumulative training counts')
            for k in ['loss','teacher_nll','teacher_current_to_parent_joint_kl','retention_current_to_parent_joint_kl','gradient_norm_before_clip']:require(math.isfinite(metric[k]),'finite metrics')
        ap=read(folder/'zero_B_functional_parity_complete.json');require(ap==parity['arms'][arm] and ap['total_queries']==8 and ap['all_GMM_action_feature_first_linear_values_and_dtype_exact'],'actual per-arm parity')
        require(ap['initial_all_parameter_digest']==initial,'parity initial digest')
        for i,probe in enumerate(p['parity']):
            values=[]
            for phase in ['plain_author','zero_head']:
                file=folder/'parity_records'/phase/f'{i:02d}.npz';info=read(file.with_suffix('.json'));require(sha(file)==info['file_sha256'],'parity NPZ hash')
                with np.load(file,allow_pickle=False) as z:values.append({k:z[k].copy() for k in z.files})
                parity_seconds+=finite_number(info['query_seconds'],'parity time')
            require(set(values[0])==set(values[1]),'parity field set')
            for k in values[0]:same(values[0][k],values[1][k],'zeroB parity '+k)
            same(values[0]['action'],np.asarray(probe['expected_action'],dtype=np.float32),'plain parent sample')
            require(abs(float(values[0]['log_probability'])-probe['expected_log_probability'])<=1e-5,'plain density check')
    return actors,initial,dict(teacher_distinct_actually_presented=int((tc>0).sum()),teacher_never_presented=int((tc==0).sum()),teacher_min_presentations=int(tc.min()),teacher_max_presentations=int(tc.max()),retention_index_distinct=int((rc>0).sum()),teacher_counts=tc.tolist(),retention_index_counts=rc.tolist(),parity_query_seconds=parity_seconds)

def audit_query_packets(root,w,read,sha,array_hash):
    records=[json.loads(x) for x in (w/'evaluation_query_receipts.jsonl').read_text('utf8').splitlines()]
    attempts=[x for x in records if x['phase']=='attempt'];ends=[x for x in records if x['phase']=='completed']
    require(len(records)==2*len(attempts)==2*len(ends),'complete query attempt/answer accounting')
    require([x['ordinal'] for x in attempts]==[x['ordinal'] for x in ends]==list(range(1,len(ends)+1)),'query ordinals')
    require(all(records[2*i]['phase']=='attempt' and records[2*i+1]['phase']=='completed' for i in range(len(ends))),'alternating query journal')
    grouped=defaultdict(list);packets=[]
    for start,end in zip(attempts,ends):
        args,q=start['arguments'],end['answer'];require(start['case']==end['case']==args['query_case'],'query case identity')
        require(Path(args['observation_file']).resolve().is_relative_to(w),'public observation path confinement')
        file=Path(q['feature_packet_file']).resolve();receipt=Path(q['feature_receipt_file']).resolve()
        require(file==Path(args['feature_packet_file']).resolve() and file.is_relative_to(w/'feature_packets') and receipt==file.with_suffix('.json'),'packet path confinement')
        require(sha(file)==q['feature_packet_sha256'] and sha(receipt)==q['feature_receipt_sha256'],'packet/receipt hash')
        info=read(receipt);require(info['feature_packet_sha256']==q['feature_packet_sha256'] and info['additional_encoder_or_head_or_sampling_calls']==0,'no preservation model call')
        for k in ['actor','history_length','policy_numerical_failure','numerical_failure_reason','action_sample_completed','public_pre_action_sensor_sha256']:
            require(info[k]==q[k],'packet receipt '+k)
        require(info['case_key']==args['query_case'] and info['action_index']==args['query_step'] and info['language']==args['language'],'packet context/step/language')
        require(info['query_seconds_before_packet_io']==q['query_seconds'] and info['packet_write_and_hash_seconds']<=q['feature_packet_io_seconds'],'packet timer boundaries')
        with np.load(file,allow_pickle=False) as z:
            for k,v in [('actor',q['actor']),('case_key',args['query_case']),('language',args['language']),('query_phase','case'),('decode_mode','learned')]:require(str(z[k].item())==v,'packet scalar '+k)
            require(int(z['action_index'])==args['query_step'] and int(z['step'])==args['query_step']+1 and int(z['history_length'])==q['history_length'],'packet index/history')
            require(bool(z['policy_numerical_failure'])==q['policy_numerical_failure'],'packet failure flag')
            if q['policy_numerical_failure']:
                public={k:z['failure_public_'+k] for k in read(w/'registration.json')['public_sensors']}
                require(array_hash(public)==q['public_pre_action_sensor_sha256'],'failed query public inputs retained')
                require(bool(q['numerical_failure_reason']),'specific numerical failure reason')
            else:
                require(info['feature_shape']==[4,768] and info['feature_dtype']=='float32','feature schema')
                require(z['spatial_feature'].shape==(4,768) and z['spatial_feature'].dtype==np.float32 and np.isfinite(z['spatial_feature']).all(),'preserved existing spatial feature')
                same(z['proposal_action'],np.asarray(q['action'],dtype=np.float32),'packet/query sample')
                require(float(z['proposal_log_probability'])==q['log_probability']==info['proposal_log_probability'],'packet proposal density')
        packets.append(file);grouped[end['case']].append(dict(arguments=args,answer=q))
    require(len(set(packets))==len(packets)==len(list((w/'feature_packets').rglob('*.npz')))==len(list((w/'feature_packets').rglob('*.json'))),'no missing/orphan/duplicate packet')
    return grouped,len(ends)

def validate_exposure_receipt(receipt,registration_sha256,source_metadata_sha256):
    require(receipt['registration_sha256']==registration_sha256 and receipt['target_conditions']==16,'exposure refresh registration/size')
    require(receipt['exact_actual_or_attempt_matches']==[] and receipt['changed_baseline_files']==[] and receipt['read_or_parse_errors']==[],'no refresh conflicts/errors')
    require(receipt['new_model_or_encoder_or_gradient_or_reset_or_action_calls']==0 and receipt['outcome_fields_selected'] is False,'metadata-only refresh')
    require(receipt['source_metadata_sha256']==source_metadata_sha256,'refresh manifest hash')


def main():
    from state_coverage_study_v40 import ROOT,W,read,sha,save,status,verify,now
    require(Path(__file__).resolve()==ROOT/'rsi_group_meeting/v40/code/analyze_state_coverage_v40.py','Preparation source cannot run a real analysis; root must review/install.')
    sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
    from paired_observation_v4 import array_hash
    began=time.perf_counter();p=verify();regsha=sha(W/'registration.json');complete=read(W/'evaluation_complete.json');close=read(W/'evaluation_worker_close.json')
    require(complete['registration_sha256']==regsha and complete['physical_episodes']==48 and complete['settling_controls']==480,'all48 evaluation closure before analysis')
    require(close['closed'] is True,'worker close ACK')
    require(not (W/'analysis').exists() and not (W/'analysis_attempt.json').exists(),'One frozen analysis attempt, never blind retry.')
    save(W/'analysis_attempt.json',dict(started_utc=now(),registration_sha256=regsha,analyzer_sha256=sha(__file__),automatic_retry=False))
    try:
        require(len(p['conditions'])==16 and len(p['cases'])==48,'fixed study size')
        expected_order=[(c['context_id'],a) for c in p['conditions'] for a in ACTORS]
        require([(e['context']['context_id'],e['actor']) for e in p['cases']]==expected_order,'condition/actor order')
        conditions={c['context_id']:c for c in p['conditions']}
        require(all(e['context']==conditions[e['context']['context_id']] for e in p['cases']),'entire frozen context identity')
        require(sum(c['max_task_controls'] for c in p['conditions'])*3==29520,'physical control cap')
        refresh=read(W/'pre_evaluation_exposure_refresh/receipt.json')
        refresh_file=W/'pre_evaluation_exposure_refresh/source_metadata_hashes.json'
        validate_exposure_receipt(refresh,regsha,sha(refresh_file))
        refresh_rows=read(refresh_file);baseline=read(W/'frozen_proposals/exposure_sources.json')
        require(refresh['files_checked']==len(refresh_rows)>=len(baseline),'refresh covers baseline and additions')
        require(refresh['new_metadata_files']==sum(not x['baseline'] for x in refresh_rows),'refresh new metadata count')
        actors,initial_digest,coverage=audit_training(ROOT,W,p,read,sha)
        metadata=read(W/'evaluation_worker_metadata.json')['metadata']
        expected=dict(parent=dict(p['parent'],all_parameter_digest=initial_digest),**actors)
        digests={k:v['all_parameter_digest'] for k,v in expected.items()}
        require(metadata['frozen_pretrained_sha256']==p['frozen_pretrained_parameters_sha256'] and metadata['parent_all_parameter_digest']==initial_digest==metadata['initial_all_parameter_digest'],'worker common/frozen identity')
        require(metadata['actor_parameter_digests']==close['actor_parameter_digests']==digests,'all three complete parameter digests')
        require(close['frozen_sha256']==p['frozen_pretrained_parameters_sha256'] and close['all_parameter_digest']==digests[close['actor']],'worker final parameter digest')
        for actor in expected.values():require(sha(actor['file'])==actor['sha256'],'bound actor file')
        grouped,queries=audit_query_packets(ROOT,W,read,sha,array_hash)
        require(set(grouped)=={e['key'] for e in p['cases']},'every case has query receipts')
        rows=[];parent_initial={};parent_result={}
        for e in p['cases']:
            file=W/'evaluation/results'/f"{e['key']}.json";r=read(file);load=r['actor_load_receipt'];which=expected[e['actor']]
            require(r['registration_sha256']==regsha and r['actor_file_sha256']==load['sha256']==which['sha256'] and load['all_parameter_digest']==which['all_parameter_digest'] and load['actor']==e['actor'],'loaded actor identity')
            for path,h in [(file.with_suffix('.npz'),r['trace_sha256']),(W/'evaluation'/r['events'],r['events_sha256']),(W/'evaluation'/r['video'],r['video_sha256']),(W/'evaluation'/r['initial_sensors'],r['initial_sensors_sha256'])]:require(Path(path).resolve().is_relative_to(W/'evaluation') and sha(path)==h,'trial artifact path/hash')
            with np.load(file.with_suffix('.npz'),allow_pickle=False) as z:trace={k:z[k].copy() for k in z.files}
            with np.load(W/'evaluation'/r['initial_sensors'],allow_pickle=False) as z:initial={k:z[k].copy() for k in z.files}
            events=[json.loads(x) for x in (W/'evaluation'/r['events']).read_text('utf8').splitlines()]
            guard=read(W/'initial_input_guards'/f"{e['key']}.json")
            require(all(guard[k] is True for k in ['before_policy_or_task_action','public_arrays_equal','full_state_equal','initial_goals_equal','model_hash_equal']),'initial guard')
            cid=e['context']['context_id']
            if e['actor']=='parent':parent_initial[cid]=initial;parent_result[cid]=(r,sha(file))
            else:
                require(cid in parent_initial and guard['parent_result_sha256']==parent_result[cid][1],'prior parent input anchor')
                for key in list(p['public_sensors'])+['initial_sim_state','initial_goal_predicates']:same(initial[key],parent_initial[cid][key],'paired initial '+key)
                for key in ['model_sha256','full_state_sha256']:require(r['paired_observation'][key]==parent_result[cid][0]['paired_observation'][key],'paired native state/model')
            row=validate_episode_payload(e,r,trace,initial,events,grouped[e['key']],p['public_sensors'],array_hash)
            row['result_sha256']=sha(file);rows.append(row)
        require(len(list((W/'evaluation/results').glob('*.json')))==len(list((W/'evaluation/results').glob('*.npz')))==48,'all48 only, no extra policy outcomes')
        controls=[json.loads(x) for x in (W/'evaluation_policy_control_receipts.jsonl').read_text('utf8').splitlines()]
        require([x['arguments']['actor'] for x in controls if x['command']=='load_actor']==[e['actor'] for e in p['cases']],'load every frozen actor')
        require([x['arguments']['seed'] for x in controls if x['command']=='reset']==[20265510]*48,'one policy reset per fixed case')
        require(len(controls)==96,'only48 load and48 policy-reset receipts')
        for e in p['cases']:
            one=[x for x in controls if x['case']==e['key']]
            require([x['command'] for x in one]==['load_actor','reset'],'per-case load/reset order')
            loaded,reset=one[0]['answer'],one[1]['answer'];which=expected[e['actor']]
            require(loaded['actor']==e['actor'] and loaded['sha256']==which['sha256'] and loaded['all_parameter_digest']==which['all_parameter_digest'],'control load receipt')
            require(reset['actor']==e['actor'] and reset['policy_seed']==20265510 and reset['history_length']==0 and reset['physical_resets']==0,'control reset receipt')
        require(sum(r['policy_queries'] for r in rows)==queries<=29520 and sum(r['task_controls'] for r in rows)<=29520,'total controls and model-query cap')
        panels=summarize(rows);prior_file=ROOT/'rsi_group_meeting/v38/root_completed_review/cost_increment.json';prior=read(prior_file)
        cost=dict(new_simulated_trial_rows=48,new_training_runs=2,total_optimizer_steps=4000,total_loss_presentations=128000,total_training_distribution_forwards=256000,
            actual_teacher_draw_coverage=coverage,new_retention_behavior_action_labels=0,new_teacher_acquisition_controls=0,parity_queries=16,parity_physical_controls=0,
            new_task_controls=sum(r['task_controls'] for r in rows),new_settling_controls=480,evaluation_policy_query_attempts=queries,
            evaluation_action_samples_completed=sum(r['action_samples_completed'] for r in rows),evaluation_policy_numerical_failures=sum(r['policy_numerical_failure'] for r in rows),
            evaluation_physics_instabilities=sum(r['physics_instability'] for r in rows),evaluation_query_seconds=sum(r['inference_seconds'] for r in rows),
            evaluation_episode_wall_sum_not_end_to_end_seconds=sum(r['episode_wall_seconds'] for r in rows),packet_io_seconds_already_inside_episode_wall=sum(r['packet_io_seconds_included_in_wall'] for r in rows),
            evaluation_registered_run_elapsed_seconds=complete['within_attempt_elapsed_seconds'],prior_ledger_file=str(prior_file),prior_ledger_sha256=sha(prior_file),
            cumulative_scoped_trial_rows=prior['cumulative_scoped_completed_trial_rows']+48,cumulative_scoped_task_controls=prior['cumulative_scoped_task_controls']+sum(r['task_controls'] for r in rows),
            cumulative_scoped_settling_controls=prior['cumulative_scoped_settling_controls']+480,known_HD_and_teacher_cache_replays_separate=prior['known_HD_and_teacher_cache_replays_separate'],
            whole_project_end_to_end_seconds=None,owner_wall_not_yet_closed=None,
            pre_evaluation_exposure_refresh_cpu_io_seconds=refresh['observed_cpu_io_seconds'],refresh_CPU_already_inside_evaluation_external_interval=True,
            historical_costs='Continue referenced ledger, including B480child+40parent, originaltraining/912bank/78interruptions, teacher/cache/probes/CPU. No historical costs erased and no repeated counting.',
            timing='Owner, external phase and within-episode timers are nested, not additive. Current owner may close only after this analysis; unknown end-to-end stays null.')
        out=dict(analyzed_utc=now(),registration_sha256=regsha,rows=rows,panels=panels,costs=cost,
            primary='Complete task by520, scopes separate',extension='Long only by900 on same trajectory; source extension is null',
            interpretation='Finite ordinary BC/state-support comparison, not pure task coverage, learned feedback, safety certification or RSS readiness.',
            limitations=p['limitations']+['Single fixed training seed and executionseed; correlated task conditions are not IID samples.','Parent-zero-success groups are uninformative for harms.','Coverage gains cannot conceal oldA harms or replace Long520 with late-only success.','Both objectives differ from historicalv38; no coverage-only attribution against that old actor.'],
            p_values=None,work_demonstrated=False,root_independent_review_required=True,
            analysis_within_attempt_seconds=time.perf_counter()-began,
            input_manifest={str(file.relative_to(ROOT)):sha(file) for file in [W/'registration.json',W/'analysis_frozen_before_outcomes.json',W/'training_complete.json',W/'training_receipt.json',W/'trained_actors.json',W/'evaluation_complete.json',W/'evaluation_worker_metadata.json',W/'evaluation_worker_close.json',W/'evaluation_query_receipts.jsonl',W/'evaluation_policy_control_receipts.jsonl',W/'pre_evaluation_exposure_refresh/receipt.json',refresh_file,prior_file]})
        folder=W/'analysis';folder.mkdir();save(folder/'coverage_readout.json',out)
        with (folder/'episode_rows.csv').open('x',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        verify();status('frozen_analysis_complete_requires_root_review',completed=48,expected=48)
        print(json.dumps(dict(panels=panels,work_demonstrated=False),allow_nan=False))
    except BaseException as error:
        save(W/'analysis_failure_requires_audit.json',dict(failed_utc=now(),error=repr(error),traceback=traceback.format_exc(),within_attempt_elapsed_seconds=time.perf_counter()-began,completed_case_validations=len(locals().get('rows',[])),automatic_retry=False,partial_outputs_must_be_preserved=True))
        raise

if __name__=='__main__':main()

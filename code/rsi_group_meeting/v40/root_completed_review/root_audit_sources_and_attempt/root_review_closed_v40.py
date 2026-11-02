"""Independent closed48 raw audit. Does not import or execute the frozen analyzer."""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
import hashlib,json,math,subprocess,sys,time
import numpy as np
ROOT=Path('F:/robot');W=ROOT/'rsi_group_meeting/v40/state_coverage_kl_control';R=W/'recovery_namespace_guard_20261102';OUT=ROOT/'rsi_group_meeting/v40/root_completed_review'
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def same(a,b):assert a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b)
def main():
    began=time.perf_counter();assert not OUT.exists()
    owner=read(R/'owner_complete.json');assert owner['all_four_recovery_commands_returned_zero'] and all(v==0 for v in owner['return_codes'].values())
    assert not (W/'owner_complete.json').exists() and read(W/'owner_phase_timings/two_fixed_trainings.json')['return_code']==1
    command="$ErrorActionPreference='Stop';$v40Live=@(Get-CimInstance Win32_Process -ErrorAction Stop -Filter \"Name = 'python.exe'\" | Where-Object { $_.CommandLine -match 'run_namespace_recovery_sequence_v40.py|evaluate_state_coverage_v40.py|worker_state_coverage_v40.py|analyze_state_coverage_v40.py' } | Select-Object ProcessId,CommandLine);[PSCustomObject]@{query_succeeded=$true;processes=$v40Live}|ConvertTo-Json -Depth 4"
    queried=subprocess.run(['powershell','-NoProfile','-Command',command],text=True,capture_output=True,check=True);identity=json.loads(queried.stdout);assert identity['query_succeeded'] and identity['processes']==[]
    p=read(W/'registration.json');regsha=sha(W/'registration.json');assert regsha=='e72b0d1d33729ff4c846aa20541efeddbbd2fc78bd9e2b4696b2c4601004940e'
    hashes={}
    def checked(file,expected=None):
        f=Path(file).resolve();assert f.is_relative_to(ROOT.resolve());h=sha(f)
        if expected is not None:assert h==expected,str(f)
        hashes[str(f)]=h;return h
    for path,h in {**p['implementation_sha256'],**p['input_files_sha256']}.items():checked(ROOT/path,h)
    checked(W/'analysis_frozen_before_outcomes.json','83e0dd91bf75091276582231381460c9592f063142e6d63964133b64774c42fc')
    addendum=read(R/'recovery_addendum_frozen_before_remaining_arm.json');checked(R/'recovery_addendum_frozen_before_remaining_arm.json','776012ec5c66c8052265f9c6ab054cd7b9010bb59b25569bb62f826af220690c')
    for n,h in addendum['recovery_sources_sha256'].items():checked(ROOT/'rsi_group_meeting/v40/recovery_code'/n,h)
    for n,h in addendum['preserved_original_evidence_sha256'].items():checked(ROOT/n,h)
    aggregate=read(R/'aggregation_complete.json')
    for n,h in aggregate['global_outputs_sha256'].items():checked(W/n,h)
    checked(R/'coverage_canonical_copy_manifest.json',aggregate['canonical_copy_manifest_sha256'])
    copyrows=read(R/'coverage_canonical_copy_manifest.json')['files'];assert len(copyrows)==4030
    for row in copyrows:checked(row['source'],row['sha256']);checked(row['canonical'],row['sha256'])
    train=read(W/'training_complete.json');assert train['training_runs']==2 and train['total_optimizer_steps']==4000 and train['total_distribution_forwards']==256000 and train['total_loss_presentations']==128000 and train['parity_queries']==16 and train['within_attempt_elapsed_seconds'] is None
    actors=read(W/'trained_actors.json')
    for v in actors.values():checked(v['file'],v['sha256'])
    complete=read(W/'evaluation_complete.json');close=read(W/'evaluation_worker_close.json');assert complete['physical_episodes']==48 and complete['settling_controls']==480 and close['closed']
    analysis=read(W/'analysis/coverage_readout.json');assert len(analysis['rows'])==48 and analysis['registration_sha256']==regsha
    for n,h in analysis['input_manifest'].items():checked(ROOT/n,h)
    assert len(list((W/'evaluation/results').glob('*.json')))==len(list((W/'evaluation/results').glob('*.npz')))==48
    journals=[json.loads(v) for v in (W/'evaluation_query_receipts.jsonl').read_text('utf8').splitlines()];assert len(journals)%2==0
    queries=defaultdict(list)
    for index in range(0,len(journals),2):
        start,end=journals[index:index+2];ordinal=index//2+1
        assert start['phase']=='attempt' and end['phase']=='completed' and start['ordinal']==end['ordinal']==ordinal and start['case']==end['case']
        args=start['arguments'];q=end['answer'];assert args['query_case']==start['case']
        checked(q['feature_packet_file'],q['feature_packet_sha256']);checked(q['feature_receipt_file'],q['feature_receipt_sha256'])
        info=read(q['feature_receipt_file']);assert info['case_key']==args['query_case'] and info['action_index']==args['query_step'] and info['additional_encoder_or_head_or_sampling_calls']==0
        assert info['feature_packet_sha256']==q['feature_packet_sha256'] and args['feature_packet_file']==q['feature_packet_file']
        queries[start['case']].append((args,q))
    assert len(queries)==48 and len(journals)//2==len(list((W/'feature_packets').rglob('*.npz')))==len(list((W/'feature_packets').rglob('*.json')))
    sys.path.insert(0,str(ROOT/'rsi_group_meeting'));from paired_observation_v4 import array_hash
    rows=[];initials={};rows_by_key={v['key']:v for v in analysis['rows']};groups=defaultdict(dict)
    for e in p['cases']:
        key=e['key'];c=e['context'];f=W/'evaluation/results'/f'{key}.json';result=read(f);checked(f)
        assert result['episode']==e and result['registration_sha256']==regsha
        trace=f.with_suffix('.npz');initial=W/'evaluation'/result['initial_sensors'];events=W/'evaluation'/result['events'];video=W/'evaluation'/result['video']
        for file,h in [(trace,result['trace_sha256']),(initial,result['initial_sensors_sha256']),(events,result['events_sha256']),(video,result['video_sha256'])]:checked(file,h)
        with np.load(trace,allow_pickle=False) as z:t={n:z[n].copy() for n in z.files}
        with np.load(initial,allow_pickle=False) as z:ini={n:z[n].copy() for n in z.files}
        n=len(t['actions']);assert n==result['executed_physics_steps']==result['attempted_physics_steps']<=c['max_task_controls']
        assert result['policy_queries']==len(queries[key])==n and result['action_samples_completed']==n
        assert not result['policy_numerical_failure'] and not result['physics_instability_terminated'] and not result['instability_terminated']
        assert t['actions'].shape==(n,7) and t['actions'].dtype==np.float32 and np.isfinite(t['actions']).all()
        g=t['diagnostic_subgoals'];official=t['official_success'];assert g.dtype==official.dtype==np.bool_ and len(g)==len(official)==n
        assert np.array_equal(official,g.all(axis=1))
        s520=bool(official[:520].any());scap=bool(official.any());s900=scap if c['partition']=='Long' else None
        assert (s520,s900,scap)==(result['success_by520'],result['success_by900'],result['success_by_cap'])
        if scap:assert official[-1] and not official[:-1].any()
        same(t['initial_sim_state'],ini['initial_sim_state']);same(t['initial_goal_predicates'],ini['initial_goal_predicates'])
        cid=c['context_id'];guard=read(W/'initial_input_guards'/f'{key}.json');checked(W/'initial_input_guards'/f'{key}.json')
        assert all(guard[k] is True for k in ['before_policy_or_task_action','public_arrays_equal','full_state_equal','initial_goals_equal','model_hash_equal'])
        if e['actor']=='parent':initials[cid]=(ini,result,sha(f))
        else:
            parent,pr,ph=initials[cid];assert guard['parent_result_sha256']==ph
            for name in p['public_sensors']+['initial_sim_state','initial_goal_predicates']:same(ini[name],parent[name])
            for name in ['model_sha256','full_state_sha256']:assert result['paired_observation'][name]==pr['paired_observation'][name]
        expected=p['parent'] if e['actor']=='parent' else actors[e['actor']]
        assert result['actor_file_sha256']==expected['sha256'] and result['actor_load_receipt']['all_parameter_digest']==close['actor_parameter_digests'][e['actor']]
        es=[json.loads(line) for line in events.read_text('utf8').splitlines()];assert len(es)==n
        inference=packet=0.
        for i,(event,(args,q)) in enumerate(zip(es,queries[key])):
            assert args['query_step']==i and event['query_step']==i and event['step']==i+1 and event['type']=='action'
            assert event['actor']==q['actor']==e['actor'] and args['language']==event['language']==c['setting']['language']
            assert event['history_length']==q['history_length']==min(i+1,10) and q['decode_mode']=='learned'
            same(np.asarray(event['action'],dtype=np.float32),t['actions'][i]);same(np.asarray(q['action'],dtype=np.float32),t['actions'][i])
            assert event['goal_predicates']==g[i].tolist() and event['official_success']==bool(official[i]) and not event['instability'] and not q['policy_numerical_failure']
            assert event['warning_counts']==t['native_warning_counts'][i].tolist()
            assert event['post_state_sha256']==array_hash({'sim':t['privileged_sim_states_for_replay_only'][i]})
            assert q['public_pre_action_sensor_sha256']==event['pre_action_sensor_sha256'] and q['action_sample_completed']
            assert event['log_probability']==q['log_probability'] and math.isfinite(q['log_probability'])
            if i==0:assert event['pre_action_sensor_sha256']==array_hash({k:ini[k] for k in p['public_sensors']})
            inference+=q['query_seconds'];packet+=q['feature_packet_io_seconds']
        assert math.isclose(inference,result['inference_seconds'],abs_tol=1e-8)
        ar=rows_by_key[key];assert ar['success520']==s520 and ar['success900']==s900 and ar['task_controls']==n and ar['result_sha256']==sha(f)
        hit=np.flatnonzero(official);first=int(hit[0]+1) if len(hit) else None
        row=dict(key=key,context_id=cid,partition=c['partition'],actor=e['actor'],success520=s520,success900=s900,first_complete_task_control=first,task_controls=n,settling_controls=10,policy_queries=n,inference_seconds=inference,packet_io_seconds_included_in_wall=packet,episode_wall_seconds=result['wall_seconds'],initial_goals=ini['initial_goal_predicates'].tolist(),terminal_goals=g[-1].tolist(),result_sha256=sha(f),video_sha256=result['video_sha256'])
        rows.append(row);groups[cid][e['actor']]=row
    controls=read_lines(W/'evaluation_policy_control_receipts.jsonl');assert len(controls)==96
    assert [v['arguments']['actor'] for v in controls if v['command']=='load_actor']==[e['actor'] for e in p['cases']]
    assert [v['arguments']['seed'] for v in controls if v['command']=='reset']==[20265510]*48
    totals={}
    for part in ['old_A','source_coverage','Long']:
        selected=[v for v in rows if v['partition']==part];counts={actor:sum(v['success520'] for v in selected if v['actor']==actor) for actor in ['parent','oldA_KL','coverage_KL']};totals[part]={'by520':counts}
        for actor in ['oldA_KL','coverage_KL']:
            pairs=[g for g in groups.values() if g['parent']['partition']==part];totals[part][actor+'_vs_parent']={'repairs':sum(g[actor]['success520'] and not g['parent']['success520'] for g in pairs),'harms':sum(g['parent']['success520'] and not g[actor]['success520'] for g in pairs)}
        assert counts=={a:v['successes'] for a,v in analysis['panels'][part]['by520']['actors'].items()}
        if part=='Long':totals[part]['by900']={a:sum(v['success900'] for v in selected if v['actor']==a) for a in counts}
    priorfile=ROOT/'rsi_group_meeting/v38/root_completed_review/cost_increment.json';prior=read(priorfile)
    phases={f.stem:read(f) for f in (R/'owner_phase_timings').glob('*.json')};assert len(phases)==4 and all(x['return_code']==0 for x in phases.values())
    task=sum(v['task_controls'] for v in rows);assert task==analysis['costs']['new_task_controls']==18069 and sum(v['settling_controls'] for v in rows)==480
    cost=dict(previous_ledger_file=str(priorfile),previous_ledger_sha256=sha(priorfile),new_physical_trial_rows=48,new_task_controls=task,new_settling_controls=480,new_training_runs=2,new_optimizer_steps=4000,new_teacher_loss_presentations=64000,new_retention_state_presentations=64000,new_training_distribution_forwards=256000,new_parity_queries_encodings_samples=16,parity_controls=0,new_offline_head_density_forwards=0,new_teacher_acquisition_controls=0,original_failed_training_command_external_seconds=read(W/'owner_phase_timings/two_fixed_trainings.json')['external_command_wall_seconds'],recovery_phase_external_seconds={k:v['external_command_wall_seconds'] for k,v in phases.items()},recovery_owner_observed_sequence_wall_seconds=owner['recovery_owner_observed_sequence_wall_seconds'],evaluation_query_seconds=sum(v['inference_seconds'] for v in rows),packet_IO_seconds_included_in_episode_wall=sum(v['packet_io_seconds_included_in_wall'] for v in rows),episode_wall_sum_not_whole_sequence=sum(v['episode_wall_seconds'] for v in rows),cumulative_scoped_completed_trial_rows=prior['cumulative_scoped_completed_trial_rows']+48,cumulative_scoped_task_controls=prior['cumulative_scoped_task_controls']+task,cumulative_scoped_settling_controls=prior['cumulative_scoped_settling_controls']+480,known_HD_and_teacher_cache_replays_separate=prior['known_HD_and_teacher_cache_replays_separate'],completed_new_development_episodes=140,completed_new_training_runs=6,whole_project_or_human_wall_seconds=None,accounting='Original failed training command and recovery owner are disjoint intervals. Recovery owner contains its four phase timers; query/packetIO/episode/loop times are nested and not additive. Historical acquisition/B labels/cache/interruptions remain in previous ledgers; preparation/visual CPU separate, unknown human/global wall null.')
    checked(R/'owner_complete.json');checked(W/'evaluation_complete.json');checked(W/'evaluation_worker_close.json');checked(W/'analysis/coverage_readout.json')
    for f in (R/'owner_phase_timings').glob('*.json'):checked(f)
    assert 'torch' not in sys.modules
    OUT.mkdir();save(OUT/'artifact_hashes.json',hashes);save(OUT/'cost_increment.json',cost)
    save(OUT/'review.json',dict(reviewed_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=regsha,frozen_analysis_sha256=sha(W/'analysis/coverage_readout.json'),recovery_addendum_sha256=sha(R/'recovery_addendum_frozen_before_remaining_arm.json'),implementation_sha256=sha(Path(__file__)),actual_process_query=identity,original_owner_successful=False,recovery_owner_four_phases_returned_zero=True,raw_rows=rows,summary=totals,all48_raw_success_initial_three_actor_inputs_event_query_action_hashes_checked=True,query_attempts_and_samples=len(journals)//2,file_hashes_checked=len(hashes),artifact_manifest_sha256=sha(OUT/'artifact_hashes.json'),cost_increment_sha256=sha(OUT/'cost_increment.json'),root_visual_review_pending=True,work_demonstrated=False,CPU_observed_seconds=time.perf_counter()-began,new_model_gradient_simulator_calls=0))
    print(json.dumps({'review':str(OUT/'review.json'),'raw_summary':totals,'task_controls':task,'files_hashed':len(hashes),'CPU_seconds':time.perf_counter()-began,'visual_pending':True}))
def read_lines(p):return [json.loads(v) for v in p.read_text('utf8').splitlines()]
if __name__=='__main__':main()

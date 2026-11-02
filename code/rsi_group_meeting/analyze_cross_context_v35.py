"""One descriptive analysis afterall12. No new task success rule or significance."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,csv,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'rsi_group_meeting/v35/cross_context_diagnostic'
sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    from cross_context_diag_v35 import verify,SENSORS,OLD,status
    from paired_observation_v4 import array_hash
    from analyze_handoff_factorial_v31 import servo_expected
    p=verify();regsha=sha(W/'registration.json');complete=read(W/'collection_complete.json')
    assert complete['physical_episodes']==12 and complete['registration_sha256']==regsha
    assert not (W/'analysis').exists(),'Preserve earlieranalysis; noautomaticreanalysis.'
    with (W/'analysis_attempt.json').open('x',encoding='utf8') as stream:
        json.dump(dict(started_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=regsha,analyzer_sha256=sha(__file__),failure_requires_audit_not_blind_retry=True),stream,indent=2)
    freeze=read(W/'analysis_frozen_before_cross_outcomes.json');assert freeze['registration_sha256']==regsha and freeze['analyzer_sha256']==sha(__file__)
    parity=read(W/'parity_complete.json');assert parity['queries']==4 and parity['all_actions_bitwise_equal'] and parity['physical_controls']==parity['physical_resets']==0
    for i,item in enumerate(p['parity']):
        q=read(W/'parity_receipts'/f'{i:02d}.json');np.testing.assert_array_equal(np.asarray(q['action'],dtype=np.float32),np.asarray(item['expected_action'],dtype=np.float32))
        assert abs(q['log_probability']-item['expected_log_probability'])<=1e-5
    ack=read(W/'worker_close_receipt.json');assert ack['closed'] and ack['frozen_sha256']==p['frozen_pretrained_parameters_sha256']
    assert ack['all_parameter_digest']==p['actors'][ack['actor']]['all_parameter_digest']
    head=[json.loads(line) for line in (W/'head_query_receipts.jsonl').read_text('utf8').splitlines()]
    attempts=[r for r in head if r['phase']=='attempt'];finished=[r for r in head if r['phase']=='completed']
    assert len(attempts)==len(finished)<=10804
    assert [r['ordinal'] for r in attempts]==[r['ordinal'] for r in finished]==list(range(1,len(attempts)+1))
    by_case={};packet_files=[];packet_receipt_files=[]
    for request,response in zip(attempts,finished):
        args=request['arguments'];q=response['answer'];assert request['case']==response['case']==args['query_case']
        packet=Path(q['feature_packet_file']).resolve();receipt_path=Path(q['feature_receipt_file']).resolve()
        assert packet==Path(args['feature_packet_file']).resolve() and packet.is_relative_to(W/'feature_packets')
        assert receipt_path==packet.with_suffix('.json') and sha(packet)==q['feature_packet_sha256'] and sha(receipt_path)==q['feature_receipt_sha256']
        receipt=read(receipt_path);assert receipt['feature_packet_sha256']==q['feature_packet_sha256']
        assert receipt['feature_shape']==[4,768] and receipt['feature_dtype']=='float32'
        assert receipt['additional_encoder_or_head_or_sampling_calls']==0
        assert receipt['packet_write_and_hash_seconds']<=q['feature_packet_io_seconds'] and q['feature_packet_io_seconds']>=0
        assert receipt['query_seconds_before_packet_io']==q['query_seconds']
        with np.load(packet) as z:
            assert z['spatial_feature'].shape==(4,768) and z['spatial_feature'].dtype==np.float32 and np.isfinite(z['spatial_feature']).all()
            np.testing.assert_array_equal(z['proposal_action'],np.asarray(q['action'],dtype=np.float32))
            assert float(z['proposal_log_probability'])==q['log_probability']==receipt['proposal_log_probability']
            assert str(z['case_key'].item())==args['query_case']==receipt['case_key']
            assert int(z['action_index'])==args['query_step']==receipt['action_index'] and int(z['step'])==args['query_step']+1
            assert str(z['query_phase'].item())==args['query_phase']==receipt['query_phase']
            assert str(z['language'].item())==args['language']==receipt['language']
            assert str(z['actor'].item())==q['actor']==receipt['actor'] and int(z['history_length'])==q['history_length']==receipt['history_length']
            assert str(z['decode_mode'].item())==q['decode_mode']==receipt['decode_mode']=='learned'
            assert str(z['density_kind'].item())==receipt['density_kind']=='policy_proposal_not_servo_behavior'
            assert 'behavior_log_probability' not in z and 'actual_executed_action' not in z
        by_case.setdefault(request['case'],[]).append((args,q))
        packet_files.append(packet);packet_receipt_files.append(receipt_path)
    assert len(set(packet_files))==len(packet_files)==len(list((W/'feature_packets').rglob('*.npz')))
    assert len(set(packet_receipt_files))==len(packet_receipt_files)==len(list((W/'feature_packets').rglob('*.json')))
    for item in p['parity']:
        metadata=by_case[item['context']['context_id']];assert len(metadata)==1
        assert metadata[0][0]['query_phase']=='parity' and metadata[0][0]['query_step']==0
    rows=[]
    for case in p['cases']:
        e=case['episode'];is_student=case['kind']=='student_training_context';folder=W/('student' if is_student else 'teacher')
        file=folder/'results'/f"{e['key']}.json";r=read(file);assert r['registration_sha256']==regsha
        assert sha(file.with_suffix('.npz'))==r['trace_sha256'] and sha(folder/r['events'])==r['events_sha256']
        assert sha(folder/r['video'])==r['video_sha256'] and sha(folder/r['initial_sensors'])==r['initial_sensors_sha256']
        with np.load(file.with_suffix('.npz')) as z:a={k:z[k].copy() for k in z.files}
        with np.load(folder/r['initial_sensors']) as z:initial={k:z[k].copy() for k in z.files}
        with np.load(ROOT/case['reference']['initial_sensors']) as expected:
            for field in SENSORS+['initial_sim_state']:np.testing.assert_array_equal(initial[field],expected[field])
            if 'initial_goal_predicates' in expected:np.testing.assert_array_equal(initial['initial_goal_predicates'],expected['initial_goal_predicates'])
        guard=read(W/'initial_input_guards'/f"{e['key']}.json");assert guard['public_arrays_equal'] and guard['model_xml_hash_equal'] and guard['canonical_full_state_hash_equal'] and guard['before_any_case_policy_query_or_task_action']
        events=[json.loads(line) for line in (folder/r['events']).read_text('utf8').splitlines()];n=len(events)
        queries=by_case[e['key']];assert len(queries)==n
        assert n==len(a['actions'])==r['executed_physics_steps']==r['policy_queries'] and n<=900
        actor=e['actor'] if is_student else 'parent';item=p['actors'][actor]
        if is_student:
            assert r['actor_file_sha256']==r['actor_load_receipt']['sha256']==item['sha256']
            assert r['actor_load_receipt']['all_parameter_digest']==item['all_parameter_digest']
            assert not r['privileged_controller'] and r['original_joint_instruction']
        else:assert r['actor_sha256']==item['sha256'] and e['home'] and not e['low_noise']
        active=False;home_count=0;goal_loss=[0,0];trigger_pos=None;servo_error=0.0
        if not is_student:home=np.asarray(r['home_position']);home_rotation=np.asarray(r['home_rotation'])
        prev=a['initial_goal_predicates'].astype(bool)
        for index,event in enumerate(events):
            args,q=queries[index];assert args['query_step']==index and args['query_phase']=='case'
            assert args['language']==event['language'] and q['history_length']==event['history_length'] and q['actor']==actor
            assert q['query_seconds']==event['query_seconds']
            assert event['step']==index+1 and event['history_length']==min(index+1,10)
            np.testing.assert_array_equal(np.asarray(event['action'],dtype=np.float32),a['actions'][index])
            assert event['goal_predicates']==a['diagnostic_subgoals'][index].tolist() and event['warning_counts']==a['native_warning_counts'][index].tolist()
            assert event['post_state_sha256']==array_hash({'sim':a['privileged_sim_states_for_replay_only'][index]})
            assert event['official_success']==bool(a['official_success'][index])==bool(a['diagnostic_subgoals'][index].all() and not event['instability'])
            goal_loss=(np.asarray(goal_loss)+((prev)&~a['diagnostic_subgoals'][index].astype(bool))).tolist();prev=a['diagnostic_subgoals'][index].astype(bool)
            if is_student:
                assert event['actor']==actor and event['language']==e['context']['setting']['language']
                assert np.isfinite(event['log_probability'])
                np.testing.assert_array_equal(np.asarray(q['action'],dtype=np.float32),a['actions'][index]);assert q['log_probability']==event['log_probability']
            else:
                expected_stage=int(active);assert event['language']==e['context']['subcommands'][expected_stage]
                assert event['decode_mode']=='learned' and a['decoder_modes'][index]=='learned'
                servo=bool(active and home_count<60)
                assert event['action_source']==('servo' if servo else 'policy')==a['action_sources'][index]
                np.testing.assert_array_equal(np.asarray(event['proposal_action'],dtype=np.float32),a['proposal_actions'][index])
                np.testing.assert_array_equal(np.asarray(q['action'],dtype=np.float32),a['proposal_actions'][index]);assert q['log_probability']==event['proposal_log_probability']==float(a['proposal_log_probabilities'][index])
                if servo:
                    info=event['servo_proprioception'];assert event['behavior_log_probability'] is None and info is not None
                    if trigger_pos is None:trigger_pos=np.asarray(info['public_robot_site_position'])
                    z=max(home[2],trigger_pos[2]+.12)
                    expected_target=np.r_[trigger_pos[:2],z] if home_count<20 else np.r_[home[:2],z] if home_count<40 else home
                    np.testing.assert_array_equal(np.asarray(info['target_position']),expected_target)
                    np.testing.assert_array_equal(np.asarray(info['target_rotation']),home_rotation)
                    reconstructed=servo_expected(info);servo_error=max(servo_error,float(np.abs(a['actions'][index]-reconstructed).max()))
                    np.testing.assert_allclose(a['actions'][index],reconstructed,rtol=0,atol=1e-7);home_count+=1
                else:
                    assert event['servo_proprioception'] is None and event['behavior_log_probability']==event['proposal_log_probability'];np.testing.assert_array_equal(a['actions'][index],a['proposal_actions'][index])
                active=active or bool(a['diagnostic_subgoals'][index,0])
        if not is_student:assert home_count==r['servo_controls']<=60
        assert r['success_by900']==bool(a['official_success'].any()) and r['success_by520']==bool(a['official_success'][:520].any())
        np.testing.assert_allclose(r['inference_seconds'],sum(q['query_seconds'] for args,q in queries),rtol=1e-12,atol=1e-9)
        np.testing.assert_allclose(r['inference_seconds'],sum(event['query_seconds'] for event in events),rtol=1e-12,atol=1e-9)
        np.testing.assert_array_equal(initial['initial_sim_state'],a['initial_sim_state'])
        rows.append(dict(key=e['key'],kind=case['kind'],context=e['context']['context_id'],actor=actor,policy_seed=e['context']['policy_seed'],success520=r['success_by520'],success900=r['success_by900'],instability=r['instability_terminated'],task_controls=n,inference_seconds=r['inference_seconds'],episode_wall_seconds=r['wall_seconds'],feature_packets=n,feature_packet_io_seconds=sum(q['feature_packet_io_seconds'] for args,q in queries),servo_controls=home_count,servo_reconstruction_max_abs_error=servo_error,first_servo_trigger_position=trigger_pos.tolist() if trigger_pos is not None else None,initial_goals=a['initial_goal_predicates'].tolist(),terminal_goals=a['diagnostic_subgoals'][-1].tolist(),goal_true_to_false_counts=goal_loss,result_sha256=sha(file),initial_inputs_equal_reference=True))
    assert len(rows)==12 and sum(r['task_controls'] for r in rows)<=10800
    controls=[json.loads(line) for line in (W/'policy_control_receipts.jsonl').read_text('utf8').splitlines()]
    resets=[r for r in controls if r['command']=='reset'];loads=[r for r in controls if r['command']=='load_actor']
    assert [r['arguments']['seed'] for r in resets]==[x['context']['policy_seed'] for x in p['parity']]+[x['episode']['context']['policy_seed'] for x in p['cases']]
    assert [r['arguments']['actor'] for r in loads]==['parent']+[x['episode'].get('actor','parent') for x in p['cases']]
    assert len(attempts)==len(finished)==sum(r['task_controls'] for r in rows)+4
    assert [r['ordinal'] for r in attempts]==[r['ordinal'] for r in finished]==list(range(1,len(attempts)+1))
    original=read(OLD/'analysis/repair_readout.json');assert sha(OLD/'analysis/repair_readout.json')==p['original_main_analysis_sha256']
    train={a:dict(successes900=sum(r['success900'] for r in rows if r['kind']=='student_training_context' and r['actor']==a),successes520=sum(r['success520'] for r in rows if r['kind']=='student_training_context' and r['actor']==a),conditions=4,training_context_only_never_work=True) for a in ['uniform_bc','local_bc']}
    teacher_test=dict(successes900=sum(r['success900'] for r in rows if r['kind']=='teacher_test_context'),successes520=sum(r['success520'] for r in rows if r['kind']=='teacher_test_context'),conditions=4,privileged_fixed_teacher=True)
    reused=dict(parent_train=[dict(context=x['context']['context_id'],success900=read(ROOT/x['result'])['success_by900']) for x in p['reused_parent_train']],teacher_train=[dict(context=x['context']['context_id'],success900=read(ROOT/x['result'])['success_by900']) for x in p['reused_teacher_train']],original24_totals=original['totals'],original24_analysis_sha256=p['original_main_analysis_sha256'])
    out=dict(analyzed_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=regsha,rows=rows,students_training_context=train,fixed_teacher_test_context=teacher_test,reused_context_evidence=reused,
             costs=dict(new_physical_episodes=12,new_task_controls=sum(r['task_controls'] for r in rows),new_settling_controls=120,parity_only_queries=4,total_new_policy_queries=len(finished),total_new_spatial_encodings=len(finished),total_policy_action_samples_including_discarded_servo_proposals=len(finished),preserved_feature_packets=len(packet_files),additional_model_calls_for_feature_preservation=0,feature_packet_io_seconds_including_parity=sum(r['answer']['feature_packet_io_seconds'] for r in finished),physical_case_packet_io_seconds=sum(r['feature_packet_io_seconds'] for r in rows),parity_packet_io_seconds=sum(q['feature_packet_io_seconds'] for item in p['parity'] for args,q in by_case[item['context']['context_id']]),feature_io_accounting='CaseIOalreadyinepisodewall;parityIOalreadyinrunwall;neitherisaddedagain.',feature_packet_bytes=sum(f.stat().st_size for f in packet_files),feature_receipt_bytes=sum(f.stat().st_size for f in packet_receipt_files),new_training_steps=0,servo_controls=sum(r['servo_controls'] for r in rows),inference_seconds_including_parity=sum(r['answer']['query_seconds'] for r in finished),registered_run_elapsed_seconds=complete['run_elapsed_seconds'],whole_command_end_to_end_seconds=None,reused_old_episodes_historical_total_increment=0,old_cost_ledger_remains='OriginalA288+old912+fresh1197+allv30/v31/v32acquisition/training/replay/evaluation, withinterruptionoverheadkept andno subsetdoublecount.'),
             guards=dict(all_initial_public_arrays_paired=True,all_initial_canonical_state_hashes_paired=True,parity_actions_bitwise=True,all_event_action_state_language_history_warning_checks=True,all12_complete=True),
             limits=[p['comparison_limit'],'Allnewdataarepost-hocexploratorydevelopment, notindependentconfirmation.','Trainingcontextstudentpositivesarenotnewtargetgeneralization/work; originalLongtest0/4andlocalsourceharmmustremainvisible.','Offlinefitandfree-runningcontexttransferareseparate. NoLR/epoch/window/servo/noisesweepornewMtrainingisjustifiedbythisdiagnostic alone.'],work_demonstrated=False)
    folder=W/'analysis';folder.mkdir();(folder/'cross_context_readout.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),'utf8')
    with (folder/'episode_rows.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    status('analysis_complete',completed=12,current=None,kind=None,worker_pid=None,parity_completed=4)
    print(json.dumps(dict(students_training_context=train,fixed_teacher_test_context=teacher_test,work_demonstrated=False)))
if __name__=='__main__':main()

"""Descriptive analysis once after all12, retaining every failure and baseline."""
import sys,json,csv
from pathlib import Path
import numpy as np
from head_bc_study_v36 import ROOT,W,read,sha,save,status,verify,now
sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
def main():
    from paired_observation_v4 import array_hash
    p=verify();complete=read(W/'evaluation_complete.json');regsha=sha(W/'registration.json')
    assert complete['physical_episodes']==12 and complete['registration_sha256']==regsha and not (W/'analysis').exists()
    save(W/'analysis_attempt.json',dict(started_utc=now(),registration_sha256=regsha,analyzer_sha256=sha(__file__),automatic_retry=False))
    actor=read(W/'trained_actors.json')['head_uniform_bc'];assert sha(actor['file'])==actor['sha256'] and actor['steps']==20
    train=read(W/'training_receipt.json');assert train['sha256']==actor['sha256'] and train['frozen_parameters_sha256']==p['frozen_pretrained_parameters_sha256']
    assert len(train['metrics'])==20 and all(x['teacher_presentations']==x['retention_presentations']==16 for x in train['metrics'])
    parity=read(W/'zero_B_functional_parity_complete.json');assert parity['total_queries']==8 and parity['all_GMM_action_feature_first_linear_values_and_dtype_exact'] and parity['physical_controls']==0
    parity_seconds=0.
    for i,item in enumerate(p['parity']):
        arrays=[]
        for phase in ['plain_author','zero_head']:
            file=W/'parity_records'/phase/f'{i:02d}.npz';receipt=read(file.with_suffix('.json'));assert sha(file)==receipt['file_sha256'];parity_seconds+=receipt['query_seconds']
            with np.load(file) as z:arrays.append({k:z[k].copy() for k in z.files})
        for field in arrays[0]:assert arrays[0][field].dtype==arrays[1][field].dtype;np.testing.assert_array_equal(arrays[0][field],arrays[1][field])
        np.testing.assert_array_equal(arrays[0]['action'],np.asarray(item['expected_action'],dtype=np.float32));assert abs(float(arrays[0]['log_probability'])-item['expected_log_probability'])<=1e-5
    membership=read(W/'fixed_training_membership.json');sampling=read(W/'training_sampling.json');rng=np.random.default_rng(20265450)
    assert len(sampling)==20
    for step in sampling:
        for pool in ['uniform_bc','retention']:assert step[pool]==rng.integers(0,128,size=16).tolist()
    metadata=read(W/'evaluation_worker_metadata.json')['metadata'];assert metadata['frozen_pretrained_sha256']==p['frozen_pretrained_parameters_sha256'] and metadata['initial_all_parameter_digest']==actor['all_parameter_digest']
    close=read(W/'evaluation_worker_close.json');assert close['closed'] and close['frozen_sha256']==p['frozen_pretrained_parameters_sha256'] and close['all_parameter_digest']==actor['all_parameter_digest']
    head=[json.loads(line) for line in (W/'evaluation_query_receipts.jsonl').read_text('utf8').splitlines()]
    starts=[x for x in head if x['phase']=='attempt'];ends=[x for x in head if x['phase']=='completed']
    assert len(starts)==len(ends)<=10800 and [x['ordinal'] for x in starts]==[x['ordinal'] for x in ends]==list(range(1,len(starts)+1))
    by_case={};packets=[]
    for start,end in zip(starts,ends):
        args=start['arguments'];q=end['answer'];assert start['case']==end['case']==args['query_case']
        file=Path(q['feature_packet_file']).resolve();receipt=Path(q['feature_receipt_file']).resolve()
        assert file==Path(args['feature_packet_file']).resolve() and file.is_relative_to(W/'feature_packets') and receipt==file.with_suffix('.json')
        assert sha(file)==q['feature_packet_sha256'] and sha(receipt)==q['feature_receipt_sha256']
        info=read(receipt);assert info['feature_shape']==[4,768] and info['feature_dtype']=='float32' and info['feature_packet_sha256']==q['feature_packet_sha256']
        assert info['additional_encoder_or_head_or_sampling_calls']==0 and info['packet_write_and_hash_seconds']<=q['feature_packet_io_seconds']
        assert info['query_seconds_before_packet_io']==q['query_seconds']
        with np.load(file) as z:
            assert z['spatial_feature'].shape==(4,768) and z['spatial_feature'].dtype==np.float32 and np.isfinite(z['spatial_feature']).all()
            assert str(z['actor'].item())==info['actor']==q['actor']=='head_uniform_bc'
            assert str(z['language'].item())==info['language']==args['language']
            assert str(z['case_key'].item())==info['case_key']==args['query_case']
            assert int(z['action_index'])==info['action_index']==args['query_step'] and int(z['step'])==args['query_step']+1
            assert int(z['history_length'])==info['history_length']==q['history_length']
            assert str(z['decode_mode'].item())==info['decode_mode']==q['decode_mode']=='learned'
            assert str(z['query_phase'].item())==info['query_phase']==args['query_phase']=='case'
            np.testing.assert_array_equal(z['proposal_action'],np.asarray(q['action'],dtype=np.float32))
            assert float(z['proposal_log_probability'])==info['proposal_log_probability']==q['log_probability']
        packets.append(file);by_case.setdefault(end['case'],[]).append((args,q))
    assert len(set(packets))==len(packets)==len(list((W/'feature_packets').rglob('*.npz')))==len(list((W/'feature_packets').rglob('*.json')))
    rows=[]
    for case in p['cases']:
        e=case['episode'];folder=W/'evaluation';file=folder/'results'/f"{e['key']}.json";r=read(file)
        assert r['registration_sha256']==regsha and r['actor_file_sha256']==r['actor_load_receipt']['sha256']==actor['sha256']
        assert r['actor_load_receipt']['all_parameter_digest']==actor['all_parameter_digest'] and r['original_joint_instruction'] and not r['privileged_controller']
        for path,h in [(file.with_suffix('.npz'),r['trace_sha256']),(folder/r['events'],r['events_sha256']),(folder/r['video'],r['video_sha256']),(folder/r['initial_sensors'],r['initial_sensors_sha256'])]:assert sha(path)==h
        with np.load(file.with_suffix('.npz')) as z:a={k:z[k].copy() for k in z.files}
        with np.load(folder/r['initial_sensors']) as z:initial={k:z[k].copy() for k in z.files}
        guard=read(W/'initial_input_guards'/f"{e['key']}.json");assert all(guard[k] for k in ['before_policy_or_task_action','public_arrays_equal','model_hash_equal','full_state_hash_equal'])
        references={}
        for name,ref in case['references'].items():
            previous=read(ROOT/ref['result']);references[name]=dict(success520=previous['success_by520'],success900=previous['success_by900'],result_sha256=sha(ROOT/ref['result']))
            with np.load(ROOT/ref['initial_sensors']) as z:
                for key in p['public_sensors']+['initial_sim_state']:np.testing.assert_array_equal(initial[key],z[key])
                if 'initial_goal_predicates' in z:np.testing.assert_array_equal(initial['initial_goal_predicates'],z['initial_goal_predicates'])
        events=[json.loads(line) for line in (folder/r['events']).read_text('utf8').splitlines()];queries=by_case[e['key']];n=len(events)
        assert n==len(queries)==len(a['actions'])==r['policy_queries']==r['executed_physics_steps']<=900
        assert a['diagnostic_subgoals'].ndim==2 and len(a['diagnostic_subgoals'])==n
        for i,event in enumerate(events):
            args,q=queries[i];assert args['query_step']==i and event['step']==i+1 and event['history_length']==q['history_length']==min(i+1,10)
            assert args['language']==event['language']==e['context']['setting']['language'] and event['actor']==q['actor']=='head_uniform_bc'
            np.testing.assert_array_equal(np.asarray(q['action'],dtype=np.float32),a['actions'][i]);np.testing.assert_array_equal(np.asarray(event['action'],dtype=np.float32),a['actions'][i])
            assert q['log_probability']==event['log_probability'] and np.isfinite(event['log_probability'])
            assert q['query_seconds']==event['query_seconds']
            assert event['goal_predicates']==a['diagnostic_subgoals'][i].tolist() and event['warning_counts']==a['native_warning_counts'][i].tolist()
            assert event['post_state_sha256']==array_hash({'sim':a['privileged_sim_states_for_replay_only'][i]})
            assert event['official_success']==bool(a['official_success'][i])==bool(a['diagnostic_subgoals'][i].all() and not event['instability'])
        assert r['instability_terminated']==any(event['instability'] for event in events)
        assert r['success_by900']==bool(a['official_success'].any()) and r['success_by520']==bool(a['official_success'][:520].any())
        np.testing.assert_allclose(r['inference_seconds'],sum(q['query_seconds'] for args,q in queries),rtol=1e-12,atol=1e-9)
        np.testing.assert_array_equal(initial['initial_sim_state'],a['initial_sim_state'])
        rows.append(dict(key=e['key'],context=e['context']['context_id'],partition=e['context']['partition'],policy_seed=e['context']['policy_seed'],
                         success520=r['success_by520'],success900=r['success_by900'],instability=r['instability_terminated'],task_controls=n,
                         inference_seconds=r['inference_seconds'],episode_wall_seconds=r['wall_seconds'],packet_io_seconds=sum(q['feature_packet_io_seconds'] for args,q in queries),
                         initial_goals=a['initial_goal_predicates'].tolist(),terminal_goals=a['diagnostic_subgoals'][-1].tolist(),references=references,
                         net900_vs_parent=int(r['success_by900'])-int(references['parent']['success900']),net900_vs_uniform=int(r['success_by900'])-int(references['uniform_bc']['success900']),net900_vs_local=int(r['success_by900'])-int(references['local_bc']['success900']),
                         net520_vs_parent=int(r['success_by520'])-int(references['parent']['success520']),net520_vs_uniform=int(r['success_by520'])-int(references['uniform_bc']['success520']),net520_vs_local=int(r['success_by520'])-int(references['local_bc']['success520']),result_sha256=sha(file)))
    assert len(rows)==12 and sum(x['task_controls'] for x in rows)==len(ends)
    controls=[json.loads(line) for line in (W/'evaluation_policy_control_receipts.jsonl').read_text('utf8').splitlines()]
    assert [x['arguments']['seed'] for x in controls if x['command']=='reset']==[x['episode']['context']['policy_seed'] for x in p['cases']]
    assert [x['arguments']['actor'] for x in controls if x['command']=='load_actor']==['head_uniform_bc']*12
    totals={}
    for partition in ['training_context','test_context','source_regression']:
        subset=[x for x in rows if x['partition']==partition];assert len(subset)==4;totals[partition]={}
        for name in ['parent','uniform_bc','local_bc','head_uniform_bc']:
            get=lambda row,h:row['success'+h] if name=='head_uniform_bc' else row['references'][name]['success'+h]
            totals[partition][name]=dict(successes900=sum(get(x,'900') for x in subset),successes520=sum(get(x,'520') for x in subset),conditions=4)
        for horizon in ['520','900']:
            totals[partition]['head_vs_parent_by'+horizon]=dict(repairs=sum(x['net'+horizon+'_vs_parent']>0 for x in subset),harms=sum(x['net'+horizon+'_vs_parent']<0 for x in subset))
    cross=read(ROOT/p['old_cross_analysis_file']);original=read(ROOT/p['original_student_analysis_file'])
    out=dict(analyzed_utc=now(),registration_sha256=regsha,rows=rows,totals=totals,
             prior_teacher_test=cross['fixed_teacher_test_context'],prior_teacher_test_all_rows=[x for x in cross['rows'] if x['kind']=='teacher_test_context'],
             prior_original24_totals=original['totals'],new_test_repairs_by520_observed=any(x['partition']=='test_context' and x['net520_vs_parent']>0 for x in rows),new_test_repairs_by900_observed=any(x['partition']=='test_context' and x['net900_vs_parent']>0 for x in rows),
             costs=dict(new_training_runs=1,optimizer_steps=20,loss_presentations=640,teacher_presentations=320,retention_presentations=320,training_distribution_forwards=1280,parity_queries_encodings_samples=8,parity_physical_controls=0,parity_query_seconds=parity_seconds,
                        new_physical_episodes=12,new_task_controls=sum(x['task_controls'] for x in rows),new_settling_controls=120,evaluation_policy_queries_encodings_samples=len(ends),
                        evaluation_inference_seconds=sum(q['answer']['query_seconds'] for q in ends),packet_io_seconds_included_in_episode_wall=sum(x['packet_io_seconds'] for x in rows),
                        evaluation_registered_run_elapsed_seconds=complete['within_attempt_elapsed_seconds'],training_loop_wall_seconds=actor['training_loop_wall_seconds'],peak_torch_allocation_not_total_VRAM=actor['peak_torch_allocated_bytes'],
                        new_teacher_acquisition=0,reused_old_trials_new_physical_increment=0,offline_new_head_density_budget=3291,offline_actual_not_yet_run=None,whole_project_end_to_end_seconds=None),
             limitations=p['limitations']+['All12fixedconditionsclosedbeforeanalysis; no p-values or checkpoint selection.','No active ordinaryRL control performed; privileged teacher servo and A/B behavior are not joint on-policy training data.','Train-context success never counts as generalization/work.'],work_demonstrated=False)
    folder=W/'analysis';folder.mkdir();save(folder/'head_readout.json',out)
    with (folder/'episode_rows.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    status('frozen_analysis_complete_requires_head_offline_fit',completed=12,expected=12)
    print(json.dumps(dict(totals=totals,work_demonstrated=False)))
if __name__=='__main__':main()

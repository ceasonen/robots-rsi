"""All48 prospective paired rollouts, public learned policy only; no GT controller."""
from pathlib import Path
import sys,os,time,json,traceback
import numpy as np
from state_coverage_study_v40 import ROOT,W,read,sha,save,status,verify,now
sys.path.insert(0,str(ROOT/'rsi_group_meeting'));E=W/'evaluation'

def preserve_exception_prefix(error,current):
    """Keep traceback locals only; no environment reads after finally/close."""
    from paired_observation_v4 import array_hash
    saved=[];tb=error.__traceback__
    while tb:
        v=tb.tb_frame.f_locals
        if tb.tb_frame.f_code.co_name=='episode' and 'actions' in v:
            arrays={};errors={}
            def keep(k,value):
                try:
                    a=np.asarray(value)
                    if a.dtype==object:raise TypeError('Refuse object arrays')
                    arrays[k]=a.copy()
                except BaseException as exc:errors[k]=repr(exc)
            for k in ['actions','states','robots','goals','warning_rows','successes','query_seconds','initial_state','initial_goals','base_warning','action','state','rs']:
                if k in v:keep(k,v[k])
            for source in ['raw','public']:
                if isinstance(v.get(source),dict):
                    for k in p_sensors()+['robot0_eef_pos','robot0_eef_quat']:
                        if k in v[source]:keep('latest_'+source+'_'+k,v[source][k])
            f=W/'failure_prefixes'/f"{current['key']}.npz";f.parent.mkdir(parents=True,exist_ok=True);assert not f.exists()
            np.savez_compressed(f,**arrays)
            info=dict(file=str(f.resolve()),sha256=sha(f),appended_controls=len(v['actions']),attempted_controls=v.get('attempted_controls'),query_count=v.get('query_count'),loop_index=v.get('i'),current_iteration_completion_unknown=v.get('attempted_controls',0)>len(v['actions']),array_shapes={k:list(a.shape) for k,a in arrays.items()},array_conversion_errors=errors,failure_source_file=tb.tb_frame.f_code.co_filename,failure_source_line=tb.tb_lineno,no_env_getter_or_encoder_or_step_or_reset_after_close=True)
            save(f.with_suffix('.json'),info);saved.append(info)
        tb=tb.tb_next
    return saved

def p_sensors():return ['agentview_image','robot0_eye_in_hand_image','robot0_joint_pos','robot0_gripper_qpos']

def expected_actors(p,metadata):
    actors=read(W/'trained_actors.json')
    assert set(actors)=={'oldA_KL','coverage_KL'}
    parent=dict(p['parent'],all_parameter_digest=metadata['parent_all_parameter_digest'])
    assert all(x['initial_all_parameter_digest']==parent['all_parameter_digest'] for x in actors.values())
    return dict(parent=parent,**actors)

def episode(e,p,client):
    import imageio.v2 as imageio
    from evaluate_dmpel_source_probe_v6 import reset_shared
    from paired_observation_v4 import canonicalize,array_hash
    from xvla_benchmark_v4 import diagnostic_subgoals
    key=e['key'];c=e['context'];actor=e['actor'];cap=int(c['max_task_controls'])
    assert cap==(900 if c['setting']['suite']=='libero_10' else 520)
    env=writer=stream=None;actions=[];states=[];robots=[];goals=[];warning_rows=[];successes=[];query_seconds=[]
    query_count=0;action_samples=0;attempted_controls=0;physics_instability=False;numeric_failure=False;terminal_query=None;done=False;started=time.perf_counter()
    save(E/'attempts'/f'{key}.json',dict(started_utc=now(),registration_sha256=sha(W/'registration.json'),episode=e,maximum_task_controls=cap))
    try:
        env,raw=reset_shared(c['setting'],c['state_index'],read(ROOT/'rsi_group_meeting/v4/xvla/protocol.json'))
        raw,pair=canonicalize(env,raw,c['setting'],c['state_index'],E/'canonical')
        for robot in env.env.robots:robot.controller.use_delta=True
        initial_state=np.asarray(env.get_sim_state()).copy();initial_goals=diagnostic_subgoals(env).copy()
        base_warning=np.asarray(env.env.sim.data.warning.number).copy()
        robot_dim=len(np.r_[raw['robot0_eef_pos'],raw['robot0_eef_quat'],raw['robot0_gripper_qpos']])
        initial=E/'initial_sensors'/f'{key}.npz';initial.parent.mkdir(parents=True,exist_ok=True)
        assert not initial.exists()
        np.savez_compressed(initial,**{n:np.asarray(raw[n]) for n in p['public_sensors']},initial_sim_state=initial_state,initial_goal_predicates=initial_goals)
        parent_key=c['context_id']+'__parent'
        guard=dict(checked_utc=now(),before_policy_or_task_action=True,parent_reference_case=parent_key,public_arrays_equal=True,full_state_equal=True,initial_goals_equal=True,model_hash_equal=True)
        if actor!='parent':
            parent_file=E/'results'/f'{parent_key}.json';parent=read(parent_file)
            with np.load(E/parent['initial_sensors'],allow_pickle=False) as z:
                for k in p['public_sensors']:np.testing.assert_array_equal(raw[k],z[k])
                np.testing.assert_array_equal(initial_state,z['initial_sim_state']);np.testing.assert_array_equal(initial_goals,z['initial_goal_predicates'])
            assert pair['model_sha256']==parent['paired_observation']['model_sha256'] and pair['full_state_sha256']==parent['paired_observation']['full_state_sha256']
            guard['parent_result_sha256']=sha(parent_file)
        save(W/'initial_input_guards'/f'{key}.json',guard)
        expected=expected_actors(p,client.metadata)[actor]
        loaded=client.request('load_actor',actor=actor)
        assert loaded['actor']==actor and loaded['sha256']==expected['sha256'] and loaded['all_parameter_digest']==expected['all_parameter_digest']
        reset=client.request('reset',seed=c['policy_seed']);assert reset['actor']==actor and reset['history_length']==0 and reset['physical_resets']==0
        video=E/'videos'/f'{key}.working.mp4';video.parent.mkdir(parents=True,exist_ok=True);assert not video.exists()
        writer=imageio.get_writer(video,fps=20,quality=7,macro_block_size=1)
        writer.append_data(np.concatenate([raw['agentview_image'][::-1],raw['robot0_eye_in_hand_image'][::-1]],axis=1))
        events=E/'events'/f'{key}.jsonl';events.parent.mkdir(parents=True,exist_ok=True);stream=events.open('x',encoding='utf8');shard_start=0
        def shard():
            nonlocal shard_start
            if len(actions)==shard_start:return
            f=E/'partial_shards'/key/f'{shard_start:04d}_{len(actions):04d}.npz';f.parent.mkdir(parents=True,exist_ok=True)
            assert not f.exists()
            np.savez_compressed(f,actions=np.asarray(actions[shard_start:]),sim_states=np.asarray(states[shard_start:]),robot_states=np.asarray(robots[shard_start:]),goal_predicates=np.asarray(goals[shard_start:]),warning_counts=np.asarray(warning_rows[shard_start:]))
            shard_start=len(actions)
        for i in range(cap):
            public={n:np.asarray(raw[n]).copy() for n in p['public_sensors']};observation=W/'current_observation.npz';np.savez_compressed(observation,**public)
            q=client.request('query',observation_file=str(observation),language=c['setting']['language']);query_count+=1;query_seconds.append(q['query_seconds']);action_samples+=int(q['action_sample_completed'])
            assert q['actor']==actor and q['history_length']==min(i+1,10) and q['public_pre_action_sensor_sha256']==array_hash(public)
            if q['policy_numerical_failure']:
                numeric_failure=True;terminal_query=q
                event=dict(type='query_failure',step=i+1,query_step=i,actor=actor,language=c['setting']['language'],history_length=q['history_length'],query_seconds=q['query_seconds'],pre_action_sensor_sha256=array_hash(public),policy_numerical_failure=True,numerical_failure_reason=q['numerical_failure_reason'],physical_action_executed=False,official_success=False,instability=True,episode_elapsed_seconds=time.perf_counter()-started)
                stream.write(json.dumps(event,allow_nan=False)+'\n');stream.flush();shard();break
            action=np.asarray(q['action'],dtype=np.float32);assert action.shape==(7,) and np.isfinite(action).all() and np.isfinite(q['log_probability'])
            attempted_controls+=1
            raw,_,done,_=env.step(action)
            state=np.asarray(env.get_sim_state()).copy();rs=np.r_[raw['robot0_eef_pos'],raw['robot0_eef_quat'],raw['robot0_gripper_qpos']]
            goal=diagnostic_subgoals(env);warnings=np.asarray(env.env.sim.data.warning.number).copy()
            physics_instability=not(np.isfinite(state).all() and np.isfinite(rs).all() and np.isfinite(raw['robot0_joint_pos']).all()) or bool(np.any(warnings[4:7]>base_warning[4:7]))
            success=bool(env.check_success()) and not physics_instability
            actions.append(action);states.append(state);robots.append(rs);goals.append(goal);warning_rows.append(warnings);successes.append(success)
            event=dict(type='action',step=i+1,query_step=i,actor=actor,language=c['setting']['language'],history_length=q['history_length'],action=action.tolist(),log_probability=q['log_probability'],query_seconds=q['query_seconds'],episode_elapsed_seconds=time.perf_counter()-started,pre_action_sensor_sha256=array_hash(public),post_state_sha256=array_hash({'sim':state}),goal_predicates=goal.tolist(),warning_counts=warnings.tolist(),official_success=success,instability=bool(physics_instability),policy_numerical_failure=False)
            stream.write(json.dumps(event)+'\n');stream.flush()
            writer.append_data(np.concatenate([raw['agentview_image'][::-1],raw['robot0_eye_in_hand_image'][::-1]],axis=1))
            if (i+1)%30==0 or success or done or physics_instability:shard()
            if success or done or physics_instability:break
        shard();writer.close();writer=None;stream.close();stream=None
        final_video=video.with_name(video.name.replace('.working',''));assert not final_video.exists();video.replace(final_video)
        result=E/'results'/f'{key}.json';result.parent.mkdir(parents=True,exist_ok=True);trace=result.with_suffix('.npz');assert not trace.exists()
        np.savez_compressed(trace,actions=np.asarray(actions,dtype=np.float32).reshape((-1,7)),privileged_sim_states_for_replay_only=np.asarray(states,dtype=np.float64).reshape((-1,len(initial_state))),robot_states=np.asarray(robots,dtype=np.float64).reshape((-1,robot_dim)),diagnostic_subgoals=np.asarray(goals,dtype=bool).reshape((-1,len(initial_goals))),native_warning_counts=np.asarray(warning_rows,dtype=np.int64).reshape((-1,len(base_warning))),official_success=np.asarray(successes,dtype=bool),initial_sim_state=initial_state,initial_goal_predicates=initial_goals)
        any_success=bool(np.asarray(successes).any())
        save(result,dict(completed_utc=now(),episode=e,registration_sha256=sha(W/'registration.json'),actor_load_receipt=loaded,actor_file_sha256=expected['sha256'],success_by520=bool(np.asarray(successes[:520]).any()),success_by900=any_success if cap==900 else None,success_by_cap=any_success,executed_physics_steps=len(actions),attempted_physics_steps=attempted_controls,policy_queries=query_count,action_samples_completed=action_samples,maximum_task_controls=cap,settling_controls=10,instability_terminated=bool(physics_instability or numeric_failure),physics_instability_terminated=bool(physics_instability),policy_numerical_failure=numeric_failure,terminal_query_failure=terminal_query,terminal_done=bool(done),inference_seconds=sum(query_seconds),wall_seconds=time.perf_counter()-started,paired_observation=pair,initial_sensors=str(initial.relative_to(E)),initial_sensors_sha256=sha(initial),events=str(events.relative_to(E)),events_sha256=sha(events),video=str(final_video.relative_to(E)),video_sha256=sha(final_video),trace_sha256=sha(trace),physical_episodes=1,privileged_controller=False,original_joint_instruction=True,initial_all_goals_true=bool(initial_goals.all())))
        print('STATE_COVERAGE_EVALUATION_COMPLETE',key,flush=True)
    finally:
        if writer is not None:writer.close()
        if stream is not None:stream.close()
        if env is not None:env.close()

def main():
    assert Path(__file__).resolve()==ROOT/'rsi_group_meeting/v40/code/evaluate_state_coverage_v40.py','Preparation only; root must review/copy.'
    p=verify();train=read(W/'training_complete.json');assert train['training_runs']==2 and train['total_optimizer_steps']==4000
    assert set(read(W/'trained_actors.json'))=={'oldA_KL','coverage_KL'}
    assert not (W/'evaluation_started.json').exists() and not list((E/'results').glob('*')),'No automatic retry.'
    from local_run_lock_v4 import exclusive
    from policy_pipe_client_v7 import PolicyPipe
    from repair_learning_awake_v32 import awake
    for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
    class Pipe(PolicyPipe):
        def __init__(self,*args):self.case_key=None;self.index=0;self.attempts=0;super().__init__(*args)
        def request(self,command,**args):
            if command=='query':
                packet=W/'feature_packets/cases'/self.case_key/f'{self.index:04d}.npz';assert not packet.exists() and not packet.with_suffix('.json').exists()
                args=dict(args,feature_packet_file=str(packet),query_case=self.case_key,query_step=self.index,query_phase='case');self.index+=1;self.attempts+=1
                with (W/'evaluation_query_receipts.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(phase='attempt',ordinal=self.attempts,case=self.case_key,utc=now(),arguments=args))+'\n')
            answer=super().request(command,**args)
            if command=='query':
                with (W/'evaluation_query_receipts.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(phase='completed',ordinal=self.attempts,case=self.case_key,utc=now(),answer=answer))+'\n')
            elif command in ['load_actor','reset']:
                with (W/'evaluation_policy_control_receipts.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(case=self.case_key,command=command,arguments=args,answer=answer,utc=now()))+'\n')
            return answer
        def close(self):
            try:
                if self.process.poll() is None:
                    ack=self.request('close');assert ack['closed'] and ack['frozen_sha256']==p['frozen_pretrained_parameters_sha256']
                    assert ack['actor_parameter_digests']=={k:v['all_parameter_digest'] for k,v in expected_actors(p,self.metadata).items()}
                    save(W/'evaluation_worker_close.json',ack);self.process.wait(timeout=30)
            finally:self.terminate();self.log.close()
    with exclusive('robotics_gpu'),exclusive('state_coverage_v40'),awake():
        start=time.perf_counter();save(W/'evaluation_started.json',dict(started_utc=now(),registration_sha256=sha(W/'registration.json'),pid=os.getpid()))
        status('evaluation_startup',completed=0,expected=48);client=None;current=None;completed=0
        try:
            from audit_state_coverage_exposure_v40 import run_audit
            run_audit(p)
            stamp=time.perf_counter();client=Pipe(ROOT/'envs/dmpel/Scripts/python.exe',ROOT/'rsi_group_meeting/v40/code/worker_state_coverage_v40.py',W)
            save(W/'evaluation_worker_metadata.json',dict(metadata=client.metadata,worker_pid=client.worker_pid,startup_wall_seconds=time.perf_counter()-stamp))
            for e in p['cases']:
                current=e;client.case_key=e['key'];client.index=0
                status('evaluating',completed=completed,expected=48,current=e['key'],worker_pid=client.worker_pid)
                episode(e,p,client);completed+=1
            verify();client.close();client=None
            assert len(list((E/'results').glob('*.json')))==48
            save(W/'evaluation_complete.json',dict(completed_utc=now(),registration_sha256=sha(W/'registration.json'),physical_episodes=48,settling_controls=480,within_attempt_elapsed_seconds=time.perf_counter()-start))
            status('evaluation_complete_requires_frozen_analysis',completed=48,expected=48)
        except BaseException as error:
            prefixes=[];capture_error=None
            if current is not None:
                try:prefixes=preserve_exception_prefix(error,current)
                except BaseException as captured:capture_error=repr(captured)
            save(W/'evaluation_failure_requires_audit.json',dict(failed_utc=now(),error=repr(error),completed_episodes=completed,current_case=current,preserved_prefixes=prefixes,prefix_capture_error=capture_error,traceback=traceback.format_exc(),automatic_retry=False,unclassified_infrastructure_censoring_not_task_failure=True))
            status('evaluation_stopped_requires_audit',completed=completed,expected=48,error=repr(error));raise
        finally:
            if client is not None:client.close()
if __name__=='__main__':main()

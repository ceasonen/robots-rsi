"""Twenty-four original-instruction rollouts; no privileged controller or training."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys,time,os,ctypes
import numpy as np
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'rsi_group_meeting/v32/repair_learning_dev';E=W/'evaluation'
sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def episode(e,p,client):
    import imageio.v2 as imageio
    import benchmark_v4
    from evaluate_dmpel_source_probe_v6 import reset_shared
    from paired_observation_v4 import canonicalize,array_hash
    from xvla_benchmark_v4 import diagnostic_subgoals
    key=e['key'];c=e['context'];actor=e['actor'];env=writer=stream=None
    save(E/'attempts'/f'{key}.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=sha(W/'registration.json'),episode=e))
    actions=[];states=[];robot=[];goals=[];warnings=[];successes=[];queryseconds=[];started=time.perf_counter()
    try:
        env,raw=reset_shared(c['setting'],c['state_index'],read(ROOT/'rsi_group_meeting/v4/xvla/protocol.json'))
        raw,pair=canonicalize(env,raw,c['setting'],c['state_index'],E/'canonical')
        for r in env.env.robots:r.controller.use_delta=True
        initialstate=np.asarray(env.get_sim_state()).copy();initialgoal=diagnostic_subgoals(env).copy()
        initial=E/'initial_sensors'/f'{key}.npz';initial.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(initial,**{n:np.asarray(raw[n]) for n in p['public_sensors']},initial_sim_state=initialstate,initial_goal_predicates=initialgoal)
        loaded=client.request('load_actor',actor=actor)
        expected=p['parent'] if actor=='parent' else read(W/'trained_actors.json')[actor]
        assert loaded['sha256']==expected['sha256']
        expected_digest=read(W/'cache_worker_metadata.json')['initial_all_parameter_digest'] if actor=='parent' else expected['all_parameter_digest']
        assert loaded['all_parameter_digest']==expected_digest
        client.request('reset',seed=p['evaluation_policy_seed'])
        video=E/'videos'/f'{key}.working.mp4';video.parent.mkdir(parents=True,exist_ok=True);writer=imageio.get_writer(video,fps=20,quality=7,macro_block_size=1)
        events=E/'events'/f'{key}.jsonl';events.parent.mkdir(parents=True,exist_ok=True);stream=events.open('x',encoding='utf8')
        basewarning=np.asarray(env.env.sim.data.warning.number).copy();instability=False;success=False;shard_start=0
        for i in range(900):
            public={n:np.asarray(raw[n]).copy() for n in p['public_sensors']};observation=W/'current_observation.npz';np.savez_compressed(observation,**public)
            q=client.request('query',observation_file=str(observation),language=c['setting']['language']);assert q['actor']==actor
            action=np.asarray(q['action'],dtype=np.float32);assert action.shape==(7,) and np.isfinite(action).all()
            raw,_,done,_=env.step(action);state=np.asarray(env.get_sim_state()).copy();rs=np.r_[raw['robot0_eef_pos'],raw['robot0_eef_quat'],raw['robot0_gripper_qpos']];goal=diagnostic_subgoals(env);warn=np.asarray(env.env.sim.data.warning.number).copy()
            instability=not(np.isfinite(state).all() and np.isfinite(rs).all() and np.isfinite(raw['robot0_joint_pos']).all()) or bool(np.any(warn[4:7]>basewarning[4:7]))
            success=bool(env.check_success()) and not instability;actions.append(action);states.append(state);robot.append(rs);goals.append(goal);warnings.append(warn);successes.append(success);queryseconds.append(q['query_seconds'])
            stream.write(json.dumps(dict(step=i+1,actor=actor,language=c['setting']['language'],history_length=q['history_length'],action=action.tolist(),log_probability=q['log_probability'],query_seconds=q['query_seconds'],episode_elapsed_seconds=time.perf_counter()-started,pre_action_sensor_sha256=array_hash(public),post_state_sha256=array_hash({'sim':state}),goal_predicates=goal.tolist(),warning_counts=warn.tolist(),official_success=success,instability=bool(instability)))+'\n');stream.flush()
            writer.append_data(np.concatenate([raw['agentview_image'][::-1],raw['robot0_eye_in_hand_image'][::-1]],axis=1))
            if (i+1)%30==0 or success or done or instability:
                shard=E/'partial_shards'/key/f'{shard_start:04d}_{len(actions):04d}.npz';shard.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(shard,actions=np.asarray(actions[shard_start:]),sim_states=np.asarray(states[shard_start:]),robot_states=np.asarray(robot[shard_start:]),goal_predicates=np.asarray(goals[shard_start:]),warning_counts=np.asarray(warnings[shard_start:]))
                shard_start=len(actions)
            if success or done or instability:break
        writer.close();writer=None;stream.close();stream=None;finalvideo=video.with_name(video.name.replace('.working',''));assert not finalvideo.exists();video.replace(finalvideo)
        file=E/'results'/f'{key}.json';file.parent.mkdir(parents=True,exist_ok=True);trace=file.with_suffix('.npz');assert not trace.exists()
        np.savez_compressed(trace,actions=np.asarray(actions),privileged_sim_states_for_replay_only=np.asarray(states),robot_states=np.asarray(robot),diagnostic_subgoals=np.asarray(goals),native_warning_counts=np.asarray(warnings),official_success=np.asarray(successes),initial_sim_state=initialstate,initial_goal_predicates=initialgoal)
        save(file,dict(completed_utc=datetime.now(timezone.utc).isoformat(),episode=e,registration_sha256=sha(W/'registration.json'),actor_load_receipt=loaded,actor_file_sha256=expected['sha256'],success_by900=success,success_by520=bool(np.asarray(successes[:520]).any()),executed_physics_steps=len(actions),policy_queries=len(actions),instability_terminated=bool(instability),terminal_done=bool(done),inference_seconds=sum(queryseconds),wall_seconds=time.perf_counter()-started,paired_observation=pair,initial_sensors=str(initial.relative_to(E)),initial_sensors_sha256=sha(initial),events=str(events.relative_to(E)),events_sha256=sha(events),video=str(finalvideo.relative_to(E)),video_sha256=sha(finalvideo),trace_sha256=sha(trace),physical_episodes=1,privileged_controller=False,original_joint_instruction=True))
        print('REPAIR_LEARNING_EVALUATION_COMPLETE',key,flush=True)
    finally:
        if writer is not None:writer.close()
        if stream is not None:stream.close()
        if env is not None:env.close()
def main():
    from local_run_lock_v4 import exclusive
    from policy_pipe_client_v7 import PolicyPipe
    p=read(W/'registration.json');assert not (E/'run_started.json').exists() and (W/'trained_actors.json').exists()
    for f,h in p['implementation_sha256'].items():assert sha(ROOT/f)==h
    assert read(W/'analysis_frozen_before_learner_outcomes.json')['analysis_implementation_sha256']==sha(ROOT/'rsi_group_meeting/analyze_repair_learning_v32.py')
    for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
    with exclusive('robotics_gpu'),exclusive('repair_learning_v32'):
        save(E/'run_started.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),registration_sha256=sha(W/'registration.json')))
        client=None
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            client=PolicyPipe(ROOT/'envs/dmpel/Scripts/python.exe',ROOT/'rsi_group_meeting/repair_learning_worker_v32.py',W)
            save(E/'worker_metadata.json',client.metadata)
            for i,e in enumerate(p['evaluation_episodes']):
                status=dict(updated_utc=datetime.now(timezone.utc).isoformat(),stage='collecting',pid=os.getpid(),worker_pid=client.worker_pid,completed=i,expected=24,current=e['key'])
                (E/'status.json').write_text(json.dumps(status),'utf8');episode(e,p,client)
            save(E/'collection_complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),physical_episodes=24,registration_sha256=sha(W/'registration.json'),work_demonstrated=False))
        finally:
            if client is not None:client.close()
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':main()

"""Replay every registered successful teacher, cache only public joint-language features."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys,time,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'rsi_group_meeting/v32/repair_learning_dev'
sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
    for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
    from local_run_lock_v4 import exclusive
    from policy_pipe_client_v7 import PolicyPipe
    from repair_learning_awake_v32 import awake
    # Import the local bootstrap before reset_shared's first libero import.
    import benchmark_v4
    from evaluate_dmpel_source_probe_v6 import reset_shared
    from paired_observation_v4 import canonicalize,array_hash
    from xvla_benchmark_v4 import diagnostic_subgoals
    p=read(W/'registration.json');assert not (W/'cache_started.json').exists()
    for f,h in p['implementation_sha256'].items():assert sha(ROOT/f)==h
    for row in p['teachers']:
        for f,h in row['files_sha256'].items():assert sha(ROOT/f)==h
    for item in p['teacher_canonical_files']:
        assert sha(ROOT/item['source'])==sha(ROOT/item['copy'])==item['sha256']
    with exclusive('robotics_gpu'),exclusive('repair_learning_v32'),awake():
        save(W/'cache_started.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),registration_sha256=sha(W/'registration.json')))
        client=None;env=None;records=[];overall=time.perf_counter()
        try:
            client=PolicyPipe(ROOT/'envs/dmpel/Scripts/python.exe',ROOT/'rsi_group_meeting/repair_learning_worker_v32.py',W)
            save(W/'cache_worker_metadata.json',client.metadata)
            for teacher in p['teachers']:
                key=teacher['key'];started=time.perf_counter();source=ROOT/teacher['result_file'];r=read(source);folder=source.parent.parent
                with np.load(source.with_suffix('.npz')) as z:a={k:z[k].copy() for k in z.files}
                events=[json.loads(s) for s in (folder/r['events']).read_text('utf8').splitlines()]
                assert r['success_by900'] and not r['instability_terminated']
                save(W/'cache_attempts'/f'{key}.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),teacher_sha256=sha(source)))
                c=r['episode']['context'];env,raw=reset_shared(c['setting'],c['state_index'],read(ROOT/'rsi_group_meeting/v4/xvla/protocol.json'))
                canonical=W/'teacher_canonical';before={str(f):sha(f) for f in canonical.iterdir() if f.is_file()}
                raw,pair=canonicalize(env,raw,c['setting'],c['state_index'],canonical)
                assert before=={str(f):sha(f) for f in canonical.iterdir() if f.is_file()}
                for robot in env.env.robots:robot.controller.use_delta=True
                np.testing.assert_array_equal(np.asarray(env.get_sim_state()),a['initial_sim_state'])
                features=[];encodings=0;seconds=0.;logs=W/'cache_control_records'/key;logs.mkdir(parents=True,exist_ok=False)
                for i,action in enumerate(a['actions']):
                    public={name:np.asarray(raw[name]).copy() for name in p['public_sensors']}
                    actualhash=array_hash(public);expectedhash=events[i]['pre_action_sensor_sha256']
                    if actualhash!=expectedhash:
                        np.savez_compressed(logs/f'{i:04d}_mismatched_public_observation.npz',**public)
                        save(logs/f'{i:04d}_input_mismatch.json',dict(action_index=i,expected_sha256=expectedhash,actual_sha256=actualhash,executed_replay_controls=i,completed_spatial_encodings=encodings,encoding_seconds=seconds,settling_controls=10,wall_seconds=time.perf_counter()-started))
                        raise AssertionError('Public input replay mismatch; preserved actual inputs, no silent rerender substitution')
                    observation=W/'current_observation.npz';np.savez_compressed(observation,**public);featurefile=W/'current_feature.npy'
                    q=client.request('encode',observation_file=str(observation),feature_file=str(featurefile),language=c['setting']['language'])
                    assert q['feature_sha256']==sha(featurefile) and q['policy_action_queries']==0
                    features.append(np.load(featurefile));seconds+=q['query_seconds'];encodings+=1
                    save(logs/f'{i:04d}_pre.json',dict(action_index=i,pre_action_public_sha256=array_hash(public),feature_sha256=sha(featurefile),encoding_seconds=q['query_seconds'],elapsed_seconds=time.perf_counter()-started))
                    raw,*_=env.step(action);state=np.asarray(env.get_sim_state());goals=diagnostic_subgoals(env);warnings=np.asarray(env.env.sim.data.warning.number)
                    robotstate=np.r_[raw['robot0_eef_pos'],raw['robot0_eef_quat'],raw['robot0_gripper_qpos']]
                    # Persist actual replay step before any exactness assertion.
                    np.savez_compressed(logs/f'{i:04d}_post.npz',action=action,sim_state=state,robot_state=robotstate,goal_predicates=goals,warning_counts=warnings,spatial_feature=features[-1])
                    np.testing.assert_array_equal(state,a['privileged_sim_states_for_replay_only'][i])
                    np.testing.assert_array_equal(goals,a['diagnostic_subgoals'][i]);np.testing.assert_array_equal(warnings,a['native_warning_counts'][i])
                    np.testing.assert_array_equal(robotstate,a['robot_states'][i])
                    # Retain complete cached prefix and cost on any later failure.
                    if (i+1)%30==0 or i+1==len(a['actions']):
                        out=W/'cache_prefixes'/key/f'{i+1:04d}.npz';out.parent.mkdir(parents=True,exist_ok=True)
                        assert not out.exists();np.savez_compressed(out,spatial_features=np.asarray(features),actions=a['actions'][:i+1])
                assert bool(env.check_success());env.close();env=None
                out=W/'teacher_features'/f'{key}.npz';out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
                np.savez_compressed(out,spatial_features=np.asarray(features),actions=a['actions'])
                switch=next((ev['step']-1 for ev in events if ev['language']==c['subcommands'][1]),None);assert switch is not None
                record=dict(key=key,file=str(out.relative_to(ROOT)),sha256=sha(out),source_result_sha256=sha(source),joint_language=c['setting']['language'],controls=len(features),settling_controls=10,spatial_encodings=encodings,policy_action_queries=0,encoder_seconds=seconds,wall_seconds=time.perf_counter()-started,switch_action_index=switch,first_local_target_index=max(9,switch-10),new_independent_outcomes=0,exact_all_public_inputs_actions_fullstates_robot_predicates_warnings=True)
                save(W/'cache_receipts'/f'{key}.json',record);records.append(record)
                print('Exact successful-teacher replay and joint-language spatial cache complete',key,flush=True)
            save(W/'cache_complete.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=sha(W/'registration.json'),teachers=records,physical_replays=len(records),controls=sum(r['controls'] for r in records),settling_controls=10*len(records),spatial_encodings=sum(r['spatial_encodings'] for r in records),policy_action_queries=0,wall_seconds=time.perf_counter()-overall,new_independent_outcomes=0))
        finally:
            if env is not None:env.close()
            if client is not None:client.close()
if __name__=='__main__':main()

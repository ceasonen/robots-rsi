"""Eight preregistered development episodes to diagnose compositional instruction limits.

Neither a novel method nor an independent confirmation/RSI experiment.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,os,sys,time,contextlib,ctypes
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'rsi_group_meeting/v30/language_schedule_dev'
sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
SENSORS=['agentview_image','robot0_eye_in_hand_image','robot0_joint_pos','robot0_gripper_qpos']
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def status(stage,**kw):
    p=WORK/'status.working.json';p.write_text(json.dumps(dict(updated_utc=now(),pid=os.getpid(),stage=stage,work_demonstrated=False,**kw),ensure_ascii=False,indent=2),'utf8');p.replace(WORK/'status.json')
def bindings():
    names=['rsi_group_meeting/language_schedule_dev_v30.py','rsi_group_meeting/language_schedule_worker_v30.py','rsi_group_meeting/policy_pipe_client_v7.py','scripts/dmpel_native_backend_v6.py','rsi_group_meeting/dmpel_actor_runtime_v7.py','rsi_group_meeting/dmpel_policy_ports_v6.py','rsi_group_meeting/dmpel_motor_core_v6.py','rsi_group_meeting/evaluate_dmpel_source_probe_v6.py','rsi_group_meeting/xvla_benchmark_v4.py','rsi_group_meeting/paired_observation_v4.py','rsi_group_meeting/local_run_lock_v4.py']
    return {n:sha(ROOT/n) for n in names}
def register():
    dest=WORK/'registration.json';assert not dest.exists() and not list(WORK.glob('results/*.json'))
    old=read(ROOT/'rsi_group_meeting/v22/fresh_transfer/registration.json');inventory=read(ROOT/'rsi_group_meeting/v7/dmpel_pair_retests/actor_inventory.json');training=read(ROOT/'rsi_group_meeting/v7/dmpel_onpolicy_source/registration.json')
    contexts=[];commands={0:['put the alphabet soup in the basket','put the tomato sauce in the basket'],7:['put the alphabet soup in the basket','put the cream cheese box in the basket']}
    for task in [0,7]:
        setting=next(c['setting'] for c in old['contexts'] if c['partition']=='Long' and c['task_id']==task and c['setting']['perturbation']=='original')
        assert 'put both' in setting['language'] and 'in the basket' in setting['language']
        for seed in [20265300,20265301]:contexts.append(dict(task_id=task,state_index=10,policy_seed=seed,setting=setting,subcommands=commands[task],context_id=f'Long_task{task}_original_init10_noise{seed}'))
    episodes=[]
    for c in contexts:
        arms=['joint','fixed_split'] if c['policy_seed']==20265300 else ['fixed_split','joint']
        for arm in arms:episodes.append(dict(context=c,arm=arm,key=c['context_id']+'__'+arm))
    plan=dict(registered_utc=now(),scope='Mechanism development on already exposed Long task families. No held-out confirmation, no learning, no novelty or RSI claim.',hypothesis='A source90 parent may have atomic skills but fail to execute compound language; fixed language scheduling tests one actionable explanation without training.',parent=inventory['parent'],frozen_pretrained_parameters_sha256=training['frozen_pretrained_parameters_sha256'],implementation_sha256=bindings(),episodes=episodes,physical_episode_budget=8,maximum_task_control_steps=7200,horizon=900,switch_before_action_index=450,primary='Official full original-task success by900; compare paired fixed_split minus joint over4 contexts; all outcomes reported, no inferential significance claim from this tiny selected development sample.',secondary='Success by520 and predicate traces descriptive; the fixed split gives only70 second-subcommand actions by520, so no fair fixed520 claim.',policy='Exact frozen parent with canonical float32 GMM sampling, same actor and execution seed across arms. No training. Keep latent history across450 switch; neither arm resets history mid-episode.',inputs='Only original native RGB128, joint7/gripper2 and chosen language. Fixed schedule never reads true goal predicates or privileged state.',semantics='task0 two goals In(alphabet_soup,basket) and In(tomato_sauce,basket); task7 In(alphabet_soup,basket) and In(cream_cheese,basket). Both original goals must hold simultaneously for success.',reset='Same shared10 settling controls [0,0,0,0,0,0,-1] and exact canonical physical/reset sensor comparison across arms; this remains a limitation vs author5zero reset.',stopping='Stop only after all8 registered episodes, an infrastructure exception or registered numerical instability within an episode. No outcome-based early termination. No automatic retry or unresolved-attempt restart.',instability_rule='Nonfinite robot/joint/simulator state or increments in MuJoCo warning indices4:7 retained as task failure, consistent with prior bank.',costs='All controlled actions, reset80 settling actions, initial model preparation, policy queries, inference and wall sums recorded. Video/state replay creates no independent evidence. No claimed actual savings.',assets=[dict(file=c['setting'][k],sha256=c['setting'][h]) for c in contexts for k,h in [('bddl','bddl_sha256'),('init_file','init_sha256')]],prior_exposure='All Long0–9 already exposed. This test cannot be independent confirmation even with new noise seed. Subcommands manually designed using original instructions; no target labels are hidden.',failure_update_rule='Any attempt without final result requires audit, preserving all event/video/NPZ shards; no silent resampling.',new_method_training=False,work_demonstrated=False)
    write(dest,plan);print('Registered8 fixed-parent language-schedule development episodes',flush=True)
def verify():
    p=read(WORK/'registration.json');assert p['implementation_sha256']==bindings()
    assert sha(p['parent']['file'])==p['parent']['sha256']
    for a in p['assets']:assert sha(ROOT/a['file'])==a['sha256']
    return p
def episode(e,p,client):
    import imageio.v2 as imageio
    from evaluate_dmpel_source_probe_v6 import reset_shared
    from paired_observation_v4 import canonicalize,array_hash
    from xvla_benchmark_v4 import diagnostic_subgoals
    c=e['context'];key=e['key'];result=WORK/'results'/f'{key}.json';assert not result.exists()
    write(WORK/'attempts'/f'{key}.json',dict(started_utc=now(),episode=e,registration_sha256=sha(WORK/'registration.json')))
    started=time.perf_counter();env=None;writer=None;stream=None
    actions=[];states=[];robot=[];predicates=[];warnings=[];successes=[];timings=[];shard_start=0
    def shard(stop):
        nonlocal shard_start
        if stop<=shard_start:return
        folder=WORK/'partial_shards'/key;folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/f'{shard_start:04d}_{stop:04d}.npz',actions=np.asarray(actions[shard_start:stop]),sim_states=np.asarray(states[shard_start:stop]),robot_states=np.asarray(robot[shard_start:stop]),diagnostic_subgoals=np.asarray(predicates[shard_start:stop]),warnings=np.asarray(warnings[shard_start:stop]),official_success=np.asarray(successes[shard_start:stop]))
        shard_start=stop
    try:
        primary=read(ROOT/'rsi_group_meeting/v4/xvla/protocol.json');env,raw=reset_shared(c['setting'],c['state_index'],primary)
        raw,pair=canonicalize(env,raw,c['setting'],c['state_index'],WORK/'canonical')
        for r in env.env.robots:r.controller.use_delta=True
        initial_state=np.asarray(env.get_sim_state()).copy();initial_pred=diagnostic_subgoals(env).copy()
        initial=WORK/'initial_sensors'/f'{key}.npz';initial.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(initial,**{n:np.asarray(raw[n]) for n in SENSORS},initial_sim_state=initial_state,initial_goal_predicates=initial_pred)
        client.request('reset',seed=c['policy_seed']);video=WORK/'videos'/f'{key}.working.mp4';video.parent.mkdir(parents=True,exist_ok=True)
        writer=imageio.get_writer(video,fps=20,quality=7,macro_block_size=1)
        events=WORK/'events'/f'{key}.jsonl';events.parent.mkdir(parents=True,exist_ok=True);stream=events.open('x',encoding='utf8')
        base_warnings=np.asarray(env.env.sim.data.warning.number).copy();instability=False;success=False
        for step in range(p['horizon']):
            language=c['setting']['language'] if e['arm']=='joint' else c['subcommands'][int(step>=p['switch_before_action_index'])]
            observed={n:np.asarray(raw[n]).copy() for n in SENSORS};observation=WORK/'current_observation.npz';np.savez_compressed(observation,**observed)
            q=client.request('query',observation_file=str(observation),language=language);action=np.asarray(q['action'],dtype=np.float32);assert action.shape==(7,) and np.isfinite(action).all()
            raw,reward,done,_=env.step(action);sim=np.asarray(env.get_sim_state()).copy();rs=np.concatenate([raw['robot0_eef_pos'],raw['robot0_eef_quat'],raw['robot0_gripper_qpos']]);w=np.asarray(env.env.sim.data.warning.number).copy()
            instability=not(np.isfinite(sim).all() and np.isfinite(rs).all() and np.isfinite(raw['robot0_joint_pos']).all()) or bool(np.any(w[4:7]>base_warnings[4:7]));success=bool(env.check_success()) and not instability;goals=diagnostic_subgoals(env)
            actions.append(action);states.append(sim);robot.append(rs);predicates.append(goals);warnings.append(w);successes.append(success);timings.append(q['query_seconds'])
            stream.write(json.dumps(dict(step=step+1,language=language,history_length=q['history_length'],action=action.tolist(),log_probability=q['log_probability'],official_success=success,goal_predicates=goals.tolist(),warning_counts=w.tolist(),instability=bool(instability),pre_action_sensor_sha256=array_hash(observed),post_state_sha256=array_hash({'sim':sim})))+'\n');stream.flush()
            writer.append_data(np.concatenate([raw['agentview_image'][::-1],raw['robot0_eye_in_hand_image'][::-1]],axis=1))
            if len(actions)%30==0:shard(len(actions))
            if success or done or instability:break
        shard(len(actions));writer.close();writer=None;stream.close();stream=None;finalvideo=video.with_name(video.name.replace('.working',''));assert not finalvideo.exists();video.replace(finalvideo)
        trace=WORK/'results'/f'{key}.npz';trace.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(trace,actions=np.asarray(actions),robot_states=np.asarray(robot),privileged_sim_states_for_replay_only=np.asarray(states),diagnostic_subgoals=np.asarray(predicates),official_success=np.asarray(successes),native_warning_counts=np.asarray(warnings),initial_sim_state=initial_state,initial_goal_predicates=initial_pred)
        write(result,dict(completed_utc=now(),episode=e,registration_sha256=sha(WORK/'registration.json'),actor_sha256=p['parent']['sha256'],success_by900=success,success_by520=success and len(actions)<=520,executed_physics_steps=len(actions),instability_terminated=bool(instability),terminal_done=bool(done),policy_queries=len(actions),inference_seconds=sum(timings),wall_seconds=time.perf_counter()-started,paired_observation=pair,initial_sensors=str(initial.relative_to(WORK)),initial_sensors_sha256=sha(initial),trace_sha256=sha(trace),events=str(events.relative_to(WORK)),events_sha256=sha(events),video=str(finalvideo.relative_to(WORK)),video_sha256=sha(finalvideo),physical_episodes=1,claim='Exploratory development diagnosis, not independent confirmation or RSI evidence'))
        print('DEVELOPMENT_COMPLETE',key,'official_success',success,'steps',len(actions),flush=True)
    finally:
        if writer is not None:writer.close()
        if stream is not None:stream.close()
        if env is not None:env.close()
def run():
    from local_run_lock_v4 import exclusive
    from policy_pipe_client_v7 import PolicyPipe
    p=verify();assert not(WORK/'run_started.json').exists(),'Review existing run; never blindly restart'
    for k,v in dict(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2').items():os.environ[k]=v
    with exclusive('robotics_gpu'),exclusive('language_schedule_v30'):
        write(WORK/'run_started.json',dict(started_utc=now(),pid=os.getpid(),registration_sha256=sha(WORK/'registration.json')))
        client=None
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            client=PolicyPipe(ROOT/'envs/dmpel/Scripts/python.exe',ROOT/'rsi_group_meeting/language_schedule_worker_v30.py',WORK)
            write(WORK/'worker_metadata.json',client.metadata)
            for e in p['episodes']:
                status('collecting_development',current=e['key'],worker_pid=client.worker_pid,completed=len(list(WORK.glob('results/*.json'))),expected=8);episode(e,p,client)
            results=[read(WORK/'results'/f"{e['key']}.json") for e in p['episodes']];assert len(results)==8
            write(WORK/'collection_complete.json',dict(completed_utc=now(),physical_episodes=8,task_controls=sum(r['executed_physics_steps'] for r in results),settling_controls=80,new_training=False,work_demonstrated=False));status('development_collection_complete_requires_analysis',completed=8,expected=8)
        except BaseException:
            status('stopped_requires_audit_no_automatic_retry',completed=len(list(WORK.glob('results/*.json'))));raise
        finally:
            if client is not None:client.close()
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['register','run']);args=ap.parse_args()
    register() if args.command=='register' else run()

"""Twelve preregistered factorial diagnostic episodes, with local import correction.

Neither a novel method nor an independent confirmation/RSI experiment.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,os,sys,time,contextlib,ctypes
import numpy as np
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'rsi_group_meeting/v31/handoff_factorial_dev'
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
    names=['rsi_group_meeting/handoff_factorial_dev_v31.py','rsi_group_meeting/handoff_factorial_worker_v31.py','rsi_group_meeting/policy_pipe_client_v7.py','scripts/dmpel_native_backend_v6.py','rsi_group_meeting/dmpel_actor_runtime_v7.py','rsi_group_meeting/dmpel_policy_ports_v6.py','rsi_group_meeting/dmpel_motor_core_v6.py','rsi_group_meeting/evaluate_dmpel_source_probe_v6.py','rsi_group_meeting/xvla_benchmark_v4.py','rsi_group_meeting/paired_observation_v4.py','rsi_group_meeting/local_run_lock_v4.py']
    names+=['rsi_group_meeting/handoff_home_servo_v31.py','rsi_group_meeting/home_servo_preflight_v31.py','envs/smolvla/Lib/site-packages/robosuite/controllers/base_controller.py','envs/smolvla/Lib/site-packages/robosuite/controllers/osc.py','envs/smolvla/Lib/site-packages/robosuite/utils/control_utils.py']
    return {n:sha(ROOT/n) for n in names}
def register():
    import copy,shutil
    oldfolder=ROOT/'rsi_group_meeting/v30/early_handoff_dev';result=read(oldfolder/'analysis/diagnostic_readout.json')
    assert result['physical_episodes']==4 and result['goal_triggered_successes']==0 and result['all_prefixes_exact'] and not any(r['instability'] for r in result['rows'])
    dest=WORK/'registration.json';assert not dest.exists() and not list(WORK.glob('results/*.json'));old=read(oldfolder/'registration.json');episodes=[]
    arms=[('home_learned',True,False),('nohome_low',False,True),('home_low',True,True)]
    for src in old['episodes']:
        order=arms if src['context']['policy_seed']==20265300 else list(reversed(arms))
        for arm,home,low in order:
            e=copy.deepcopy(src);e.update(arm=arm,home=home,low_noise=low,key=e['context']['context_id']+'__'+arm);episodes.append(e)
    assert len(episodes)==12
    p=dict(old);p.update(registered_utc=now(),scope='One bounded privileged diagnosis of a2x2 factorization: public-proprio home operation and author low_eval_noise branch after the first goal. Not a deployable method, training gain, independent confirmation or RSI.',implementation_sha256=bindings(),episodes=episodes,physical_episode_budget=12,maximum_task_control_steps=10800,horizon=900,home_controls=60,home_recipe='20 raise at trigger XY to max(homeZ,triggerZ+.12),20 move XY home at raised Z,20 return home XYZ; initial controller rotation throughout, gripper open. Proportional public pose error gain.5 converted through actual OSC scale. No object state or teleport. B queries keep history/RNG warm while servo proposals are discarded.',noise_recipe='A always uses original learned Normal scales. After first-goal trigger, low arms toggle the author head low_eval_noise=True at inference; Normal min_std1e-4 with native dtype rounding, Categorical logits and sampling unchanged. This changes execution policy despite unchanged weight hash.',primary='Full original task success by900 for all12 plus all4 reused nohome/learned GT controls. Report all four factor cells, paired wins/harms and interaction descriptively, no inferential significance.',stopping='All12 finite episodes after one40-control servo preflight. If no complete task repair across the new arms, reject this simple scheduling/home/low-noise branch and do not sweep further timing/servo/noise knobs. Errors require audit, no automatic retry.',engineering_preflight='One40-control public-pose roundtrip up.06m then home20+20; errors at20 and40 each<.015m, finite states and no relevant warning increment. Counts40+10settling, no task success evaluated. Failure blocks benchmark run.',costs='Up to10800 task+120settling; separate40 servo preflight+10settling and its wall, every homing control still performs oneB query and consumes normal RNG. All earlier20 physical development episodes and10 CPU engineering queries remain charged. Unknown time not0.',inputs='Actor gets native RGB128/joint7/gripper2/language. Stage timing reads goal0 privilege; home servo uses public controller EE pose/orientation. Both are disclosed teacher/control privileges, not learner inputs.',action_likelihoods='Every event stores proposed policy action/logp and actual executed action. Servo behavior logp is null, not on-policy RL; BC may use actual actions only with declared teacher provenance.',work_demonstrated=False,new_method_training=False,reused_control_analysis_sha256=sha(oldfolder/'analysis/diagnostic_readout.json'))
    p['engineering_revision']=read(WORK/'engineering_revision.json')
    p['engineering_revision_receipt_sha256']=sha(WORK/'engineering_revision.json')
    p['policy']='Frozen parent weight checkpoint, explicitly different execution policies after the first-goal trigger. Noise arms toggle author low_eval_noise, home arms override actor proposals with public-proprio servo. No weight training; unchanged weight SHA does not mean unchanged policy.'
    (WORK/'canonical').mkdir(parents=True,exist_ok=True);p['canonical_source_files']=[]
    for src in sorted((oldfolder/'canonical').iterdir()):
        assert src.is_file();dst=WORK/'canonical'/src.name;assert not dst.exists();shutil.copy2(src,dst);p['canonical_source_files'].append(dict(source=str(src.relative_to(ROOT)),sha256=sha(src)))
    write(dest,p);print('Registered12 factorial episodes and40-control engineering preflight',flush=True)
def verify():
    p=read(WORK/'registration.json');assert p['implementation_sha256']==bindings()
    assert sha(p['parent']['file'])==p['parent']['sha256']
    for a in p['assets']:assert sha(ROOT/a['file'])==a['sha256']
    for a in p['canonical_source_files']:
        assert sha(ROOT/a['source'])==a['sha256']
        assert sha(WORK/'canonical'/Path(a['source']).name)==a['sha256']
    return p
def episode(e,p,client):
    import imageio.v2 as imageio
    from evaluate_dmpel_source_probe_v6 import reset_shared
    from paired_observation_v4 import canonicalize,array_hash
    from xvla_benchmark_v4 import diagnostic_subgoals
    c=e['context'];key=e['key'];result=WORK/'results'/f'{key}.json';assert not result.exists()
    write(WORK/'attempts'/f'{key}.json',dict(started_utc=now(),episode=e,registration_sha256=sha(WORK/'registration.json')))
    started=time.perf_counter();env=None;writer=None;stream=None
    actions=[];states=[];robot=[];predicates=[];warnings=[];successes=[];timings=[];shard_start=0;proposals=[];proposal_logps=[];action_sources=[];decoder_modes=[]
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
        from handoff_home_servo_v31 import pose,servo,home_target
        ctrl=env.env.robots[0].controller;home_pos,home_ori=pose(ctrl);trigger_pos=None;home_ticks=0;home_exit_error=None
        initial_state=np.asarray(env.get_sim_state()).copy();initial_pred=diagnostic_subgoals(env).copy()
        initial=WORK/'initial_sensors'/f'{key}.npz';initial.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(initial,**{n:np.asarray(raw[n]) for n in SENSORS},initial_sim_state=initial_state,initial_goal_predicates=initial_pred)
        client.request('reset',seed=c['policy_seed']);video=WORK/'videos'/f'{key}.working.mp4';video.parent.mkdir(parents=True,exist_ok=True)
        writer=imageio.get_writer(video,fps=20,quality=7,macro_block_size=1)
        events=WORK/'events'/f'{key}.jsonl';events.parent.mkdir(parents=True,exist_ok=True);stream=events.open('x',encoding='utf8')
        base_warnings=np.asarray(env.env.sim.data.warning.number).copy();instability=False;success=False
        for step in range(p['horizon']):
            stage=int(bool(predicates) and bool(np.asarray(predicates)[:,0].any()))
            language=c['subcommands'][stage];mode='author_low_eval_noise' if stage and e['low_noise'] else 'learned'
            if stage and trigger_pos is None:trigger_pos,_=pose(ctrl)
            is_home=bool(stage and e['home'] and home_ticks<p['home_controls'])
            observed={n:np.asarray(raw[n]).copy() for n in SENSORS};observation=WORK/'current_observation.npz';np.savez_compressed(observation,**observed)
            q=client.request('query',observation_file=str(observation),language=language,decode_mode=mode);proposal=np.asarray(q['action'],dtype=np.float32)
            if is_home:action,servo_info=servo(ctrl,home_target(home_pos,trigger_pos,home_ticks),home_ori,details=True)
            else:action,servo_info=proposal,None
            if is_home:home_ticks+=1
            proposals.append(proposal);proposal_logps.append(q['log_probability']);action_sources.append('servo' if is_home else 'policy');decoder_modes.append(mode)
            assert action.shape==(7,) and np.isfinite(action).all()
            raw,reward,done,_=env.step(action);sim=np.asarray(env.get_sim_state()).copy();rs=np.concatenate([raw['robot0_eef_pos'],raw['robot0_eef_quat'],raw['robot0_gripper_qpos']]);w=np.asarray(env.env.sim.data.warning.number).copy()
            instability=not(np.isfinite(sim).all() and np.isfinite(rs).all() and np.isfinite(raw['robot0_joint_pos']).all()) or bool(np.any(w[4:7]>base_warnings[4:7]));success=bool(env.check_success()) and not instability;goals=diagnostic_subgoals(env)
            if is_home and home_ticks==60:home_exit_error=float(np.linalg.norm(pose(ctrl)[0]-home_pos))
            actions.append(action);states.append(sim);robot.append(rs);predicates.append(goals);warnings.append(w);successes.append(success);timings.append(q['query_seconds'])
            stream.write(json.dumps(dict(step=step+1,language=language,history_length=q['history_length'],action=action.tolist(),proposal_action=proposal.tolist(),proposal_log_probability=q['log_probability'],behavior_log_probability=None if is_home else q['log_probability'],action_source=action_sources[-1],servo_proprioception=servo_info,decode_mode=mode,normal_scale_min=q['normal_scale_min'],normal_scale_max=q['normal_scale_max'],query_seconds=q['query_seconds'],episode_elapsed_seconds=time.perf_counter()-started,official_success=success,goal_predicates=goals.tolist(),warning_counts=w.tolist(),instability=bool(instability),pre_action_sensor_sha256=array_hash(observed),post_state_sha256=array_hash({'sim':sim})))+'\n');stream.flush()
            writer.append_data(np.concatenate([raw['agentview_image'][::-1],raw['robot0_eye_in_hand_image'][::-1]],axis=1))
            if len(actions)%30==0:shard(len(actions))
            if success or done or instability:break
        shard(len(actions));writer.close();writer=None;stream.close();stream=None;finalvideo=video.with_name(video.name.replace('.working',''));assert not finalvideo.exists();video.replace(finalvideo)
        trace=WORK/'results'/f'{key}.npz';trace.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(trace,proposal_actions=np.asarray(proposals),proposal_log_probabilities=np.asarray(proposal_logps),action_sources=np.asarray(action_sources),decoder_modes=np.asarray(decoder_modes),actions=np.asarray(actions),robot_states=np.asarray(robot),privileged_sim_states_for_replay_only=np.asarray(states),diagnostic_subgoals=np.asarray(predicates),official_success=np.asarray(successes),native_warning_counts=np.asarray(warnings),initial_sim_state=initial_state,initial_goal_predicates=initial_pred)
        write(result,dict(completed_utc=now(),episode=e,registration_sha256=sha(WORK/'registration.json'),actor_sha256=p['parent']['sha256'],success_by900=success,success_by520=success and len(actions)<=520,executed_physics_steps=len(actions),instability_terminated=bool(instability),terminal_done=bool(done),policy_queries=len(actions),inference_seconds=sum(timings),wall_seconds=time.perf_counter()-started,paired_observation=pair,initial_sensors=str(initial.relative_to(WORK)),initial_sensors_sha256=sha(initial),trace_sha256=sha(trace),events=str(events.relative_to(WORK)),events_sha256=sha(events),video=str(finalvideo.relative_to(WORK)),video_sha256=sha(finalvideo),physical_episodes=1,servo_controls=home_ticks,home_position=home_pos.tolist(),home_rotation=home_ori.tolist(),home_exit_position_error_m=home_exit_error,terminal_EE_distance_to_home_m=float(np.linalg.norm(pose(ctrl)[0]-home_pos)),execution_rule='Author learned sigma A; declared B sigma mode and optional public home servo',claim='Exploratory development diagnosis, not independent confirmation or RSI evidence'))
        print('DEVELOPMENT_COMPLETE',key,'official_success',success,'steps',len(actions),flush=True)
    finally:
        if writer is not None:writer.close()
        if stream is not None:stream.close()
        if env is not None:env.close()
def run():
    from local_run_lock_v4 import exclusive
    from policy_pipe_client_v7 import PolicyPipe
    p=verify();preflight=read(WORK/'servo_preflight_receipt.json');assert preflight['passed'] and preflight['registration_sha256']==sha(WORK/'registration.json');assert not(WORK/'run_started.json').exists(),'Review existing run; never blindly restart'
    for k,v in dict(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2').items():os.environ[k]=v
    with exclusive('robotics_gpu'),exclusive('handoff_factorial_v31'):
        write(WORK/'run_started.json',dict(started_utc=now(),pid=os.getpid(),registration_sha256=sha(WORK/'registration.json')))
        client=None
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            client=PolicyPipe(ROOT/'envs/dmpel/Scripts/python.exe',ROOT/'rsi_group_meeting/handoff_factorial_worker_v31.py',WORK)
            write(WORK/'worker_metadata.json',client.metadata)
            for e in p['episodes']:
                status('collecting_development',current=e['key'],worker_pid=client.worker_pid,completed=len(list(WORK.glob('results/*.json'))),expected=12);episode(e,p,client)
            results=[read(WORK/'results'/f"{e['key']}.json") for e in p['episodes']];assert len(results)==12
            write(WORK/'collection_complete.json',dict(completed_utc=now(),physical_episodes=12,task_controls=sum(r['executed_physics_steps'] for r in results),settling_controls=120,new_training=False,work_demonstrated=False));status('development_collection_complete_requires_analysis',completed=12,expected=12)
        except BaseException:
            status('stopped_requires_audit_no_automatic_retry',completed=len(list(WORK.glob('results/*.json'))));raise
        finally:
            if client is not None:client.close()
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['register','preflight','run']);args=ap.parse_args()
    if args.command=='register':register()
    elif args.command=='preflight':
        from home_servo_preflight_v31 import run as check;check()
    else:run()

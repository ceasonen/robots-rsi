"""One registered40-control engineering probe, never a robot task gain."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,time,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'rsi_group_meeting/v31/handoff_factorial_dev';sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
sys.path.insert(0,str(ROOT/'external/LIBERO-PRO'))
os.environ['LIBERO_CONFIG_PATH']=str(ROOT/'.cache/libero')
for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[key]='1'
def run():
    from handoff_factorial_dev_v31 import verify,write,sha
    from local_run_lock_v4 import exclusive
    from evaluate_dmpel_source_probe_v6 import reset_shared
    from handoff_home_servo_v31 import pose,servo
    import importlib
    p=verify();assert not(W/'servo_preflight_attempt.json').exists()
    modules={}
    for name,relative in [('robosuite.controllers.base_controller','envs/smolvla/Lib/site-packages/robosuite/controllers/base_controller.py'),('robosuite.controllers.osc','envs/smolvla/Lib/site-packages/robosuite/controllers/osc.py'),('robosuite.utils.control_utils','envs/smolvla/Lib/site-packages/robosuite/utils/control_utils.py')]:
        module=importlib.import_module(name);actual=Path(module.__file__).resolve();assert actual==(ROOT/relative).resolve();assert sha(actual)==p['implementation_sha256'][relative];modules[name]=dict(file=str(actual),sha256=sha(actual))
    with exclusive('robotics_gpu'):
        write(W/'servo_preflight_attempt.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=sha(W/'registration.json'),task_actions_budget=40,task_success_evaluation=False))
        env=None;started=time.perf_counter();actions=[];states=[];position_errors=[]
        try:
            c=p['episodes'][0]['context'];env,raw=reset_shared(c['setting'],c['state_index'],json.loads((ROOT/'rsi_group_meeting/v4/xvla/protocol.json').read_text('utf8')))
            ctrl=env.env.robots[0].controller;ctrl.use_delta=True;home,ori=pose(ctrl);initial=np.asarray(env.get_sim_state()).copy();warn=np.asarray(env.env.sim.data.warning.number).copy()
            np.save(W/'servo_preflight_initial_sim_state.npy',initial)
            for step in range(40):
                target=home+np.asarray([0,0,.06]) if step<20 else home
                action=servo(ctrl,target,ori);raw,*_=env.step(action);state=np.asarray(env.get_sim_state()).copy();pos,rot=pose(ctrl)
                actions.append(action);states.append(state);position_errors.append(float(np.linalg.norm(pos-target)))
                write(W/f'servo_preflight_control_{step+1:02d}.json',dict(action=action.tolist(),target_position=target.tolist(),measured_public_position=pos.tolist(),position_error_m=position_errors[-1],state_sha256=__import__('paired_observation_v4').array_hash({'sim':state}),episode_elapsed_seconds=time.perf_counter()-started))
                np.savez_compressed(W/'servo_preflight_latest_prefix.npz',actions=np.asarray(actions),states=np.asarray(states),initial_sim_state=initial)
                assert np.isfinite(state).all() and np.isfinite(pos).all()
                assert not np.any(np.asarray(env.env.sim.data.warning.number)[4:7]>warn[4:7])
                if (step+1)%10==0:np.savez_compressed(W/f'servo_preflight_prefix_{step+1:02d}.npz',actions=np.asarray(actions),states=np.asarray(states),initial_sim_state=initial)
            passed=position_errors[19]<.015 and position_errors[-1]<.015
            write(W/'servo_preflight_receipt.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=sha(W/'registration.json'),actual_robosuite_modules=modules,passed=bool(passed),controlled_steps=40,settling_controls=10,policy_queries=0,raise_target_error_m=position_errors[19],return_target_error_m=position_errors[-1],wall_seconds=time.perf_counter()-started,new_independent_task_outcomes=0,claim='Proprioceptive servo engineering only, not task success or learned repair'))
            assert passed,'Servo roundtrip did not meet the registered engineering tolerance; preserve/audit before any study run'
            print('Servo engineering roundtrip passed; no task gain claim',flush=True)
        finally:
            if env is not None:env.close()
if __name__=='__main__':run()
